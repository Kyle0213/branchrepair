from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from _local_bootstrap import PROJECT_ROOT
from run_general_label_reasoning_compare_split import _round0_prompt
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.runtime import (
    close_runtime,
    load_runtime,
    run_prompts,
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = row.get("unique_id")
            if case_id:
                completed.add(str(case_id))
    return completed


def _agent_key(dataset: str, model: str, idx: int) -> str:
    model_name = Path(model).name if "/" in model else model
    return f"{dataset}__{model_name}__None__Agent{idx}"


def _runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        model_dir="",
        use_vllm=False,
        runtime_backend="openai_api",
        api_base_url=args.api_base_url,
        api_beta_base_url=args.api_beta_base_url,
        api_key=args.api_key,
        api_model=args.api_model,
        api_timeout=args.api_timeout,
        api_max_retries=args.api_max_retries,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=0.0,
        max_model_len=None,
        temperature=args.temperature,
        top_p=args.top_p,
        token="",
        multi_persona=False,
        data="math",
        memory_for_model_activations_in_gb=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_path", type=Path, required=True)
    parser.add_argument("--history_path", type=Path, required=True)
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--round0_prompt_style", default="bare")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api_model", default="deepseek-chat")
    parser.add_argument("--api_base_url", default="https://api.deepseek.com")
    parser.add_argument("--api_beta_base_url", default="https://api.deepseek.com/beta")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_timeout", type=float, default=120)
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    args = parser.parse_args()

    cases = _load_json(args.cases_path)
    runtime = load_runtime(_runtime_args(args))
    try:
        prompts_by_case = {}
        dataset_name = str(args.dataset).strip().lower()
        for case in cases:
            prompt = _round0_prompt(case["problem"], args.round0_prompt_style)
            prompts_by_case[case["unique_id"]] = prompt
        completed = _load_completed_case_ids(args.history_path)
        mode = "a" if args.history_path.exists() else "w"
        model_name = Path(args.model).name if "/" in args.model else args.model
        args.history_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with args.history_path.open(mode, encoding="utf-8") as handle:
            for sample_index, case in enumerate(cases):
                case_id = str(case["unique_id"])
                if case_id in completed:
                    continue
                prompt = prompts_by_case[case["unique_id"]]
                prompt_records = []
                for idx in range(1, 4):
                    agent_id = _agent_key(dataset_name, args.model, idx)
                    prompt_records.append((f"{case_id}::{agent_id}", prompt))
                outputs = []
                for start in range(0, len(prompt_records), args.prompt_batch_size):
                    outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))
                grouped = {}
                for (alias, _), output in zip(prompt_records, outputs):
                    _, agent_id = alias.split("::", 1)
                    grouped[agent_id] = output
                row = {
                    "record_type": "debate_trace_v2",
                    "unique_id": case_id,
                    "sample_index": sample_index,
                    "mode": {
                        "mode_key": "ORIGINMAD",
                        "solver": "debate",
                        "debate_scheme": "legacy",
                        "num_agents": 3,
                        "debate_rounds": 0,
                        "model": model_name,
                        "data": dataset_name,
                        "data_size": len(cases),
                        "shard_id": 0,
                        "num_shards": 1,
                    },
                    "task": {
                        "unique_id": case_id,
                        "question": prompt,
                        "gold_answer": case["answer"],
                    },
                    "summary": {
                        "num_rounds_recorded": 1,
                        "final_round_index": 0,
                    },
                    "rounds": [
                        {
                            "round_index": 0,
                            "inputs": {
                                "initial_responses": grouped,
                            },
                        }
                    ],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed.add(case_id)
                written += 1
                print(
                    json.dumps(
                        {
                            "event": "round0_case_done",
                            "case_index": sample_index,
                            "completed_cases": len(completed),
                            "num_cases": len(cases),
                            "history_path": str(args.history_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        print(json.dumps({"num_cases": len(cases), "new_cases": written, "history_path": str(args.history_path)}, ensure_ascii=False, indent=2))
    finally:
        close_runtime(runtime)


if __name__ == "__main__":
    main()
