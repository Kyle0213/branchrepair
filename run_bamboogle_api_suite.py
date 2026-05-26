from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from _local_bootstrap import PROJECT_ROOT
from run_compare_origin_vs_nl_graph_batch import (
    _build_focus_summary,
    _build_mode_round_metrics,
    _build_summary,
    _public_payload_view,
)


DEFAULT_CASES_PATH = (
    "results/full_eval_equal_token_1536_20260517/qwen3/bamboogle/"
    "bare_origin_mad/bamboogle_bare/cases.json"
)
DEFAULT_OUTPUT_ROOT = "results/bamboogle_api_suite"
DEFAULT_PROFILE = "bamboogle_relation_v7_equal_token"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
        handle.flush()


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _complete_json(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = _load_json(path)
    except Exception:
        return False
    return int(payload.get("num_cases") or len(payload.get("results", []))) == expected


def _split_cases(cases: list[dict], shards: int) -> list[list[dict]]:
    buckets = [[] for _ in range(max(1, int(shards)))]
    for idx, case in enumerate(cases):
        buckets[idx % len(buckets)].append(case)
    return buckets


def _run_processes(tasks: list[tuple[list[str], Path]], env: dict[str, str], master_log: Path, max_parallel: int) -> None:
    pending = list(tasks)
    running: list[tuple[subprocess.Popen, Path, object, list[str]]] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            command, log_path = pending.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            _append(master_log, f"[{_now()}] START {' '.join(command)}")
            proc = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((proc, log_path, handle, command))

        still_running = []
        for proc, log_path, handle, command in running:
            code = proc.poll()
            if code is None:
                still_running.append((proc, log_path, handle, command))
                continue
            handle.close()
            _append(master_log, f"[{_now()}] END code={code} log={log_path}")
            if code != 0:
                raise RuntimeError(f"Command failed code={code}: {' '.join(command)}; log={log_path}")
        running = still_running
        if pending or running:
            time.sleep(5)


def _merge_history(shard_histories: list[Path], output_path: Path, cases: list[dict]) -> None:
    rows = []
    for path in shard_histories:
        rows.extend(_load_jsonl(path))
    order = {str(case["unique_id"]): idx for idx, case in enumerate(cases)}
    rows.sort(key=lambda row: order.get(str(row.get("unique_id")), 10**9))
    for idx, row in enumerate(rows):
        row["sample_index"] = idx
    _write_jsonl(output_path, rows)


def _merge_graph(
    *,
    cases_path: Path,
    history_path: Path,
    output_path: Path,
    shard_outputs: list[Path],
    profile: str,
    graph_rounds: int,
    graph_focus_round: int,
    api_model: str,
) -> dict:
    results = []
    for path in shard_outputs:
        results.extend(_load_json(path).get("results", []))
    order = {case["unique_id"]: idx for idx, case in enumerate(_load_json(cases_path))}
    results.sort(key=lambda item: order.get(item.get("unique_id"), 10**9))
    origin_num_correct, origin_accuracy = _build_summary(results, "origin")
    graph_final_num_correct, graph_final_accuracy = _build_summary(results, "graph")
    graph_num_correct, graph_accuracy = _build_focus_summary(results, "graph", graph_rounds, graph_focus_round)
    payload = {
        "dataset": "bamboogle",
        "cases_path": str(cases_path),
        "history_path": str(history_path),
        "prompt_profile": profile,
        "round0_generated_by": api_model,
        "graph_only": True,
        "num_cases": len(results),
        "origin_rounds": 0,
        "graph_rounds": graph_rounds,
        "graph_focus_round": graph_focus_round,
        "origin_num_correct": origin_num_correct,
        "origin_accuracy": origin_accuracy,
        "graph_num_correct": graph_num_correct,
        "graph_accuracy": graph_accuracy,
        "graph_final_num_correct": graph_final_num_correct,
        "graph_final_accuracy": graph_final_accuracy,
        "origin_round_metrics": _build_mode_round_metrics(results, "origin", 0),
        "graph_round_metrics": _build_mode_round_metrics(results, "graph", graph_rounds),
        "shard_outputs": [str(path) for path in shard_outputs],
        "results": results,
    }
    public_payload = _public_payload_view(payload)
    _write_json(output_path, public_payload)
    return public_payload


def _baseline_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return _load_json(path).get("method_summaries", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Bamboogle evaluation through an OpenAI-compatible API.")
    parser.add_argument("--cases_path", type=Path, default=PROJECT_ROOT / DEFAULT_CASES_PATH)
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--max_parallel", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--api_model", default="deepseek-chat")
    parser.add_argument("--model", default="")
    parser.add_argument("--api_base_url", default="https://api.deepseek.com")
    parser.add_argument("--api_beta_base_url", default="https://api.deepseek.com/beta")
    parser.add_argument("--api_timeout", type=float, default=180)
    parser.add_argument("--api_max_retries", type=int, default=4)
    parser.add_argument("--round0_prompt_style", default="claim_atomic")
    parser.add_argument("--prompt_profile", default=DEFAULT_PROFILE)
    parser.add_argument("--graph_rounds", type=int, default=2)
    parser.add_argument("--graph_focus_round", type=int, default=2)
    parser.add_argument("--case_batch_size", type=int, default=1)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    parser.add_argument("--cot_sc_samples", type=int, default=3)
    parser.add_argument("--origin_mad_rounds", type=int, default=3)
    parser.add_argument("--methods", default="cot_sc,som,origin_mad")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_round0", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_graph", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in the environment; do not put it in command logs.")

    model_name = args.model or args.api_model
    args.output_root.mkdir(parents=True, exist_ok=True)
    master_log = args.output_root / "master.log"
    _append(master_log, f"[{_now()}] Bamboogle API suite start output_root={args.output_root}")

    cases = _load_json(args.cases_path)
    full_cases_path = args.output_root / "cases.json"
    _write_json(full_cases_path, cases)
    case_shards = _split_cases(cases, args.shards)

    round0_tasks: list[tuple[list[str], Path]] = []
    graph_tasks: list[tuple[list[str], Path]] = []
    shard_history_paths: list[Path] = []
    shard_graph_outputs: list[Path] = []

    for shard_idx, shard_cases in enumerate(case_shards):
        shard_dir = args.output_root / f"shard_{shard_idx:02d}"
        shard_cases_path = shard_dir / "cases.json"
        shard_history_path = shard_dir / "round0.history.jsonl"
        shard_graph_path = shard_dir / "graph.json"
        _write_json(shard_cases_path, shard_cases)
        shard_history_paths.append(shard_history_path)
        shard_graph_outputs.append(shard_graph_path)
        if not args.skip_round0 and not (args.resume and _jsonl_count(shard_history_path) == len(shard_cases)):
            round0_tasks.append(
                (
                    [
                        sys.executable,
                        "scripts/generate_deepseek_round0_history.py",
                        "--cases_path",
                        str(shard_cases_path),
                        "--history_path",
                        str(shard_history_path),
                        "--dataset",
                        "bamboogle",
                        "--round0_prompt_style",
                        args.round0_prompt_style,
                        "--model",
                        model_name,
                        "--api_model",
                        args.api_model,
                        "--api_base_url",
                        args.api_base_url,
                        "--api_beta_base_url",
                        args.api_beta_base_url,
                        "--max_new_tokens",
                        str(args.max_new_tokens),
                        "--api_timeout",
                        str(args.api_timeout),
                        "--api_max_retries",
                        str(args.api_max_retries),
                        "--prompt_batch_size",
                        str(args.prompt_batch_size),
                    ],
                    shard_dir / "round0.console.log",
                )
            )

    env = dict(os.environ)
    if round0_tasks:
        _run_processes(round0_tasks, env, master_log, args.max_parallel)

    full_history_path = args.output_root / "round0.history.jsonl"
    if all(_jsonl_count(path) == len(shard_cases) for path, shard_cases in zip(shard_history_paths, case_shards)):
        _merge_history(shard_history_paths, full_history_path, cases)
    elif not args.skip_round0:
        raise RuntimeError("Round0 history is incomplete; inspect shard round0.console.log files.")

    baseline_path = args.output_root / "baselines_som_sc_mad.json"
    if not args.skip_baselines and not (args.resume and _complete_json(baseline_path, len(cases))):
        _run_processes(
            [
                (
                    [
                        sys.executable,
                        "scripts/run_baseline_methods_batch.py",
                        "--dataset",
                        "bamboogle",
                        "--cases_path",
                        str(full_cases_path),
                        "--history_path",
                        str(full_history_path),
                        "--output_path",
                        str(baseline_path),
                        "--log_path",
                        str(args.output_root / "baselines_som_sc_mad.log"),
                        "--methods",
                        args.methods,
                        "--runtime_backend",
                        "openai_api",
                        "--api_base_url",
                        args.api_base_url,
                        "--api_beta_base_url",
                        args.api_beta_base_url,
                        "--api_model",
                        args.api_model,
                        "--model",
                        model_name,
                        "--no-use_vllm",
                        "--case_batch_size",
                        "32",
                        "--prompt_batch_size",
                        str(args.prompt_batch_size),
                        "--max_new_tokens",
                        str(args.max_new_tokens),
                        "--round0_prompt_style",
                        args.round0_prompt_style,
                        "--cot_sc_samples",
                        str(args.cot_sc_samples),
                        "--origin_mad_rounds",
                        str(args.origin_mad_rounds),
                        "--api_timeout",
                        str(args.api_timeout),
                        "--api_max_retries",
                        str(args.api_max_retries),
                        "--compare_all_cases",
                    ],
                    args.output_root / "baselines_som_sc_mad.console.log",
                )
            ],
            env,
            master_log,
            1,
        )

    if not args.skip_graph:
        for shard_idx, shard_cases in enumerate(case_shards):
            shard_dir = args.output_root / f"shard_{shard_idx:02d}"
            shard_graph_path = shard_dir / "graph.json"
            if args.resume and _complete_json(shard_graph_path, len(shard_cases)):
                continue
            graph_tasks.append(
                (
                    [
                        sys.executable,
                        "scripts/run_compare_origin_vs_nl_graph_batch.py",
                        "--cases_path",
                        str(shard_dir / "cases.json"),
                        "--history_path",
                        str(shard_dir / "round0.history.jsonl"),
                        "--output_path",
                        str(shard_graph_path),
                        "--log_path",
                        str(shard_dir / "graph.log"),
                        "--runtime_backend",
                        "openai_api",
                        "--api_base_url",
                        args.api_base_url,
                        "--api_beta_base_url",
                        args.api_beta_base_url,
                        "--api_model",
                        args.api_model,
                        "--model",
                        model_name,
                        "--no-use_vllm",
                        "--graph_only",
                        "--origin_rounds",
                        "0",
                        "--graph_rounds",
                        str(args.graph_rounds),
                        "--graph_focus_round",
                        str(args.graph_focus_round),
                        "--case_batch_size",
                        str(args.case_batch_size),
                        "--max_new_tokens",
                        str(args.max_new_tokens),
                        "--prompt_profile",
                        args.prompt_profile,
                        "--disable_target_check_landing",
                        "--answer_selection_policy",
                        "supported_answer_memory",
                        "--canonical_answer_mode",
                        "select_only",
                        "--stall_policy",
                        "continue",
                        "--api_timeout",
                        str(args.api_timeout),
                        "--api_max_retries",
                        str(args.api_max_retries),
                    ],
                    shard_dir / "graph.console.log",
                )
            )
        if graph_tasks:
            _run_processes(graph_tasks, env, master_log, args.max_parallel)

    summary = {
        "dataset": "bamboogle",
        "num_cases": len(cases),
        "cases_path": str(full_cases_path),
        "history_path": str(full_history_path),
        "api_model": args.api_model,
        "round0_prompt_style": args.round0_prompt_style,
        "prompt_profile": args.prompt_profile,
        "baselines": _baseline_summary(baseline_path),
    }
    graph_merged_path = args.output_root / "graph_merged.json"
    if all(path.exists() for path in shard_graph_outputs):
        graph_payload = _merge_graph(
            cases_path=full_cases_path,
            history_path=full_history_path,
            output_path=graph_merged_path,
            shard_outputs=shard_graph_outputs,
            profile=args.prompt_profile,
            graph_rounds=args.graph_rounds,
            graph_focus_round=args.graph_focus_round,
            api_model=args.api_model,
        )
        summary.update(
            {
                "graph_num_cases": graph_payload.get("num_cases"),
                "round0_num_correct": graph_payload.get("origin_num_correct"),
                "round0_accuracy": graph_payload.get("origin_accuracy"),
                "graph_num_correct": graph_payload.get("graph_num_correct"),
                "graph_accuracy": graph_payload.get("graph_accuracy"),
                "graph_final_num_correct": graph_payload.get("graph_final_num_correct"),
                "graph_final_accuracy": graph_payload.get("graph_final_accuracy"),
                "graph_merged_path": str(graph_merged_path),
            }
        )
    _write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    _append(master_log, f"[{_now()}] Bamboogle API suite end summary={args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
