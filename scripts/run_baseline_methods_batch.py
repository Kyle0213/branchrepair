from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from _local_bootstrap import PROJECT_ROOT
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.scripts.run_compare_origin_vs_nl_graph_batch import (
    _agent_label,
    _build_mode_round_metrics,
    _build_origin_question_prompt,
    _extract_answer_from_trace_text,
    _is_answer_correct,
    _majority_answer,
    _match_history_record,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.scripts.run_general_label_reasoning_compare_split import (
    DEFAULT_MODEL,
    _build_cases,
    _compute_round0_case_view,
    _filter_history_rows,
    _generate_round0_history,
    _load_json,
    _load_jsonl,
    _round0_prompt,
    _write_json,
    _write_jsonl,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.local_origin_mad import (
    _build_direct_response_update_messages,
    _get_answer_map,
    _get_peer_map,
    _parse_direct_update_output,
    _parse_revision_output,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.runtime import (
    close_runtime,
    load_runtime,
    run_prompts,
)


DEFAULT_METHODS = "cot,cot_sc,self_refine,som,origin_mad,dmad,sparse_mad,centralized_mad"
DMAD_PERSONAS = {
    "A1": "a careful evidence checker who tracks exactly what the question asks",
    "A2": "a skeptical verifier who looks for hidden assumptions and unsupported jumps",
    "A3": "a concise solver who prefers the simplest valid chain of reasoning",
}
SOM_ROLES = {
    "MindEvidence": "Check whether each response uses the stated evidence and question wording faithfully.",
    "MindLogic": "Check whether each inference follows from the previous step.",
    "MindAnswer": "Check whether the final answer matches the requested object and format.",
}


def _build_runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    token_path = PROJECT_ROOT / "token"
    token = token_path.read_text().strip() if token_path.exists() else ""
    return SimpleNamespace(
        model=args.model,
        model_dir=args.model_dir,
        use_vllm=args.use_vllm,
        runtime_backend=getattr(args, "runtime_backend", "vllm"),
        api_base_url=getattr(args, "api_base_url", ""),
        api_beta_base_url=getattr(args, "api_beta_base_url", ""),
        api_key=getattr(args, "api_key", ""),
        api_model=getattr(args, "api_model", ""),
        api_timeout=getattr(args, "api_timeout", 120),
        api_max_retries=getattr(args, "api_max_retries", 3),
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
        token=token,
        multi_persona=False,
        data="math",
        memory_for_model_activations_in_gb=2,
    )


def _method_list(raw: str) -> list[str]:
    aliases = {
        "cot": "cot",
        "sc": "cot_sc",
        "self-consistency": "cot_sc",
        "self_refine": "self_refine",
        "self-refine": "self_refine",
        "society_of_mind": "som",
        "society-of-mind": "som",
        "society": "som",
        "mad": "origin_mad",
        "origin-mad": "origin_mad",
        "origin_mad": "origin_mad",
        "diversified_mad": "dmad",
        "diversified-mad": "dmad",
        "sparse": "sparse_mad",
        "sparse-mad": "sparse_mad",
        "centralized": "centralized_mad",
        "centralized-mad": "centralized_mad",
        "centralised": "centralized_mad",
        "centralised_mad": "centralized_mad",
    }
    methods = []
    for item in str(raw or "").split(","):
        key = item.strip().lower()
        if not key:
            continue
        methods.append(aliases.get(key, key))
    supported = {"cot", "cot_sc", "self_refine", "som", "origin_mad", "dmad", "sparse_mad", "centralized_mad"}
    unknown = sorted(set(methods) - supported)
    if unknown:
        raise ValueError(f"Unsupported baseline method(s): {unknown}")
    return methods


def _normalize_dataset_defaults(args: argparse.Namespace) -> None:
    if args.split:
        return
    if args.dataset == "strategyqa":
        args.split = "train"
    elif args.dataset == "prontoqa":
        args.split = "test"
    elif args.dataset == "anli":
        args.split = "dev_r3"
    elif args.dataset in {"mmlu_pro", "logiqa", "bamboogle", "medqa"}:
        args.split = "test"
    elif args.dataset == "csqa":
        args.split = "validation"
    elif args.dataset in {"boolq", "hotpotqa", "musique", "proofwriter", "folio"}:
        args.split = "validation"


def _ensure_default_limits(args: argparse.Namespace) -> None:
    if args.limit is None and args.dataset in {"boolq", "medqa", "hotpotqa", "musique", "proofwriter", "folio"}:
        args.limit = 1000
    if args.round0_prompt_style == "claim_atomic" and args.dataset in {"proofwriter", "folio"}:
        args.round0_prompt_style = "logic_atomic"


def _build_case_state(case: dict, history_record: dict) -> dict:
    round0 = history_record["rounds"][0]
    raw_responses = round0["inputs"]["initial_responses"]
    initial_responses = {_agent_label(agent_id): text for agent_id, text in raw_responses.items()}
    initial_responses = {agent_id: initial_responses[agent_id] for agent_id in sorted(initial_responses)}
    initial_answers = {
        agent_id: _extract_answer_from_trace_text(text, case["problem"])
        for agent_id, text in initial_responses.items()
    }
    return {
        "unique_id": case["unique_id"],
        "problem": case["problem"],
        "gold_answer": case["answer"],
        "initial_responses": copy.deepcopy(initial_responses),
        "initial_answers": initial_answers,
    }


def _build_independent_case_state(case: dict) -> dict:
    return {
        "unique_id": case["unique_id"],
        "problem": case["problem"],
        "gold_answer": case["answer"],
        "initial_responses": {"IndependentInit": ""},
        "initial_answers": {"IndependentInit": None},
    }


def _prepare_states(cases: list[dict], history_records: list[dict]) -> list[dict]:
    return [_build_case_state(case, _match_history_record(case, history_records)) for case in cases]


def _answer_map_from_responses(responses: dict[str, str], question: str) -> dict[str, str | None]:
    return {
        agent_id: _extract_answer_from_trace_text(response, question)
        for agent_id, response in responses.items()
    }


def _answers_from_texts(texts: dict[str, str], question: str) -> dict[str, str | None]:
    return {key: _extract_answer_from_trace_text(text, question) for key, text in texts.items()}


def _is_correct(answer: str | None, gold_answer, question: str) -> bool:
    return _is_answer_correct(answer, gold_answer, question)


def _most_common_with_tie_order(answer_map: dict[str, str | None]) -> str | None:
    answers = [answer for answer in answer_map.values() if answer is not None]
    if not answers:
        return None
    counts = Counter(answers)
    best_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == best_count}
    for answer in answer_map.values():
        if answer in tied:
            return answer
    return None


def _final_answer_instruction() -> str:
    return (
        'End with exactly one final-answer line in curly brackets, for example: '
        '"{final answer: \\boxed{...}}".'
    )


def _responses_block(responses: dict[str, str]) -> str:
    blocks = []
    for agent_id, response in sorted(responses.items()):
        blocks.append(f"{agent_id} response:\n{response}")
    return "\n\n".join(blocks)


def _run_cot_sc(states: list[dict], runtime, args: argparse.Namespace) -> None:
    prompt_records = []
    for state in states:
        if args.cot_sc_use_history:
            continue
        prompt = _round0_prompt(state["problem"], "cot")
        for sample_idx in range(1, args.cot_sc_samples + 1):
            prompt_records.append((f"{state['unique_id']}::SC{sample_idx}", prompt))

    generated_by_case: dict[str, dict[str, str]] = {}
    if prompt_records:
        outputs = []
        for start in range(0, len(prompt_records), args.prompt_batch_size):
            outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))
        for (alias, _), output in zip(prompt_records, outputs):
            case_id, sample_id = alias.split("::", 1)
            generated_by_case.setdefault(case_id, {})[sample_id] = output

    for state in states:
        if args.cot_sc_use_history:
            sample_responses = copy.deepcopy(state["initial_responses"])
        else:
            sample_responses = generated_by_case.get(state["unique_id"], {})
        sample_answers = _answers_from_texts(sample_responses, state["problem"])
        chosen = _most_common_with_tie_order(sample_answers)
        round_result = {
            "round_index": 1,
            "sample_responses": sample_responses,
            "revised_answers": sample_answers,
            "chosen_answer": chosen,
        }
        state["cot_sc"] = {
            "num_rounds": 1,
            "round_results": [round_result],
            "sample_responses": sample_responses,
            "revised_answers": sample_answers,
            "chosen_answer": chosen,
            "is_correct": _is_correct(chosen, state["gold_answer"], state["problem"]),
        }


def _run_cot(states: list[dict], runtime, args: argparse.Namespace) -> None:
    prompt_records = []
    for state in states:
        prompt_records.append((state["unique_id"], _round0_prompt(state["problem"], "cot")))

    generated_by_case: dict[str, str] = {}
    if prompt_records:
        outputs = []
        for start in range(0, len(prompt_records), args.prompt_batch_size):
            outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))
        generated_by_case = {
            case_id: output
            for (case_id, _), output in zip(prompt_records, outputs)
        }

    for state in states:
        response = generated_by_case.get(state["unique_id"], "")
        answer = _extract_answer_from_trace_text(response, state["problem"])
        sample_responses = {"CoT": response}
        sample_answers = {"CoT": answer}
        round_result = {
            "round_index": 1,
            "sample_responses": sample_responses,
            "revised_answers": sample_answers,
            "chosen_answer": answer,
        }
        state["cot"] = {
            "num_rounds": 1,
            "round_results": [round_result],
            "sample_responses": sample_responses,
            "revised_answers": sample_answers,
            "chosen_answer": answer,
            "is_correct": _is_correct(answer, state["gold_answer"], state["problem"]),
        }


def _build_self_refine_prompt(question: str, current_response: str) -> str:
    return (
        "Refine your own solution without using any peer responses.\n\n"
        f"Question:\n{question}\n\n"
        f"Your current solution:\n{current_response}\n\n"
        "First identify the earliest concrete mistake, if any. Then either keep the solution or provide a corrected full solution. "
        "Do not change the answer just to be different; revise only after a local check supports the change.\n\n"
        f"{_final_answer_instruction()}\n\n"
        "Return STRICT JSON only:\n"
        '{"decision":"REVISE|KEEP","verification":"specific local check",'
        '"incorrect_claim":"concrete wrong claim, or empty",'
        '"corrected_claim":"correct replacement claim, or empty",'
        '"revised_final_answer":"exact final answer expression, or empty",'
        '"revised_response":"full revised solution text, or empty when KEEP"}'
    )


def _run_self_refine(states: list[dict], runtime, args: argparse.Namespace) -> None:
    for state in states:
        state["self_refine"] = {
            "responses": copy.deepcopy(state["initial_responses"]),
            "round_results": [],
        }

    parser_args = SimpleNamespace()
    for round_idx in range(1, args.self_refine_rounds + 1):
        prompt_records = []
        for state in states:
            for agent_id, response in state["self_refine"]["responses"].items():
                prompt = _build_self_refine_prompt(state["problem"], response)
                prompt_records.append((f"{state['unique_id']}::{agent_id}", prompt))

        outputs = []
        for start in range(0, len(prompt_records), args.prompt_batch_size):
            outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))

        outputs_by_case: dict[str, dict[str, str]] = {}
        for (alias, _), output in zip(prompt_records, outputs):
            case_id, agent_id = alias.split("::", 1)
            outputs_by_case.setdefault(case_id, {})[agent_id] = output

        for state in states:
            responses = state["self_refine"]["responses"]
            metadata = {}
            for agent_id, raw_output in outputs_by_case.get(state["unique_id"], {}).items():
                revised, meta = _parse_revision_output(parser_args, raw_output, responses[agent_id])
                responses[agent_id] = revised
                metadata[agent_id] = meta
            revised_answers = _answer_map_from_responses(responses, state["problem"])
            chosen = _majority_answer(revised_answers)
            state["self_refine"]["round_results"].append(
                {
                    "round_index": round_idx,
                    "revision_metadata": metadata,
                    "responses": copy.deepcopy(responses),
                    "revised_answers": revised_answers,
                    "chosen_answer": chosen,
                }
            )

    for state in states:
        round_results = state["self_refine"]["round_results"]
        final_round = round_results[-1] if round_results else {}
        chosen = final_round.get("chosen_answer") or _majority_answer(state["initial_answers"])
        state["self_refine"].update(
            {
                "num_rounds": args.self_refine_rounds,
                "revised_answers": copy.deepcopy(final_round.get("revised_answers", state["initial_answers"])),
                "chosen_answer": chosen,
                "is_correct": _is_correct(chosen, state["gold_answer"], state["problem"]),
            }
        )


def _origin_mad_args() -> SimpleNamespace:
    return SimpleNamespace(
        centralized=False,
        sparse=False,
        limit_one_peer=False,
        finalfilter=False,
        multi_persona=False,
        data="math",
        system_prompt_text="",
    )


def _run_origin_mad(states: list[dict], runtime, args: argparse.Namespace) -> None:
    for state in states:
        state["origin_mad"] = {
            "responses": copy.deepcopy(state["initial_responses"]),
            "round_results": [],
            "topology": "full",
        }

    prompt_args = _origin_mad_args()
    for round_idx in range(1, args.origin_mad_rounds + 1):
        prompt_records = []
        context_by_case = {}
        for state in states:
            responses = state["origin_mad"]["responses"]
            prev_answers = _get_answer_map(prompt_args, responses, question=state["problem"])
            peer_map = _get_peer_map(prompt_args, list(responses.keys()), prev_answers)
            update_messages = _build_direct_response_update_messages(
                prompt_args,
                _build_origin_question_prompt(state["problem"]),
                responses,
                peer_map,
                personas=None,
            )
            context_by_case[state["unique_id"]] = {
                "prev_answers": prev_answers,
                "peer_map": peer_map,
                "responses": copy.deepcopy(responses),
            }
            for agent_id, message in update_messages.items():
                prompt_records.append((f"{state['unique_id']}::{agent_id}", message))

        raw_outputs = []
        for start in range(0, len(prompt_records), args.prompt_batch_size):
            raw_outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))

        outputs_by_case: dict[str, dict[str, str]] = {}
        for (alias, _), output in zip(prompt_records, raw_outputs):
            case_id, agent_id = alias.split("::", 1)
            outputs_by_case.setdefault(case_id, {})[agent_id] = output

        for state in states:
            case_id = state["unique_id"]
            previous_responses = context_by_case[case_id]["responses"]
            updated_responses = dict(previous_responses)
            raw_updates = outputs_by_case.get(case_id, {})
            for agent_id, raw_output in raw_updates.items():
                updated_responses[agent_id] = _parse_direct_update_output(
                    raw_output,
                    previous_responses.get(agent_id, ""),
                )
            revised_answers = _answer_map_from_responses(updated_responses, state["problem"])
            chosen = _majority_answer(revised_answers)
            state["origin_mad"]["round_results"].append(
                {
                    "round_index": round_idx,
                    "input_answers": context_by_case[case_id]["prev_answers"],
                    "peer_map": copy.deepcopy(context_by_case[case_id]["peer_map"]),
                    "raw_updates": raw_updates,
                    "responses": copy.deepcopy(updated_responses),
                    "revised_answers": revised_answers,
                    "chosen_answer": chosen,
                }
            )
            state["origin_mad"]["responses"] = updated_responses

    for state in states:
        round_results = state["origin_mad"]["round_results"]
        final_round = round_results[-1] if round_results else {}
        chosen = final_round.get("chosen_answer") or _majority_answer(state["initial_answers"])
        state["origin_mad"].update(
            {
                "num_rounds": args.origin_mad_rounds,
                "revised_answers": copy.deepcopy(final_round.get("revised_answers", state["initial_answers"])),
                "chosen_answer": chosen,
                "is_correct": _is_correct(chosen, state["gold_answer"], state["problem"]),
            }
        )


def _build_som_mind_prompt(role_name: str, role_instruction: str, question: str) -> str:
    return (
        f"You are {role_name} in a Society-of-Mind solver.\n"
        f"Role: {role_instruction}\n\n"
        f"Question:\n{question}\n\n"
        "Solve the problem from your role's perspective. Use the question text directly, not another solver's answer. "
        "Keep the reasoning concise and make the final answer explicit.\n\n"
        f"{_final_answer_instruction()}\n\n"
        "Output a short solution followed by the final-answer line."
    )


def _build_som_chair_prompt(question: str, mind_outputs: dict[str, str]) -> str:
    return (
        "You are the chair of a Society-of-Mind solver. Synthesize the specialist minds into one final solution.\n\n"
        f"Question:\n{question}\n\n"
        f"Specialist mind solutions:\n{_responses_block(mind_outputs)}\n\n"
        "Choose the answer with the strongest local support. If the minds disagree, resolve the disagreement from the question text.\n\n"
        f"{_final_answer_instruction()}\n\n"
        "Output only the final solution."
    )


def _run_som(states: list[dict], runtime, args: argparse.Namespace) -> None:
    prompt_records = []
    for state in states:
        for role_name, instruction in SOM_ROLES.items():
            prompt = _build_som_mind_prompt(role_name, instruction, state["problem"])
            prompt_records.append((f"{state['unique_id']}::{role_name}", prompt))

    outputs = []
    for start in range(0, len(prompt_records), args.prompt_batch_size):
        outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))

    mind_outputs_by_case: dict[str, dict[str, str]] = {}
    for (alias, _), output in zip(prompt_records, outputs):
        case_id, role_name = alias.split("::", 1)
        mind_outputs_by_case.setdefault(case_id, {})[role_name] = output

    chair_outputs_by_case: dict[str, str] = {}
    if args.som_use_chair:
        chair_records = []
        for state in states:
            mind_outputs = mind_outputs_by_case.get(state["unique_id"], {})
            prompt = _build_som_chair_prompt(state["problem"], mind_outputs)
            chair_records.append((state["unique_id"], prompt))
        chair_outputs = []
        for start in range(0, len(chair_records), args.prompt_batch_size):
            chair_outputs.extend(run_prompts(runtime, chair_records[start : start + args.prompt_batch_size]))
        for (case_id, _), output in zip(chair_records, chair_outputs):
            chair_outputs_by_case[case_id] = output

    for state in states:
        mind_outputs = mind_outputs_by_case.get(state["unique_id"], {})
        mind_answers = _answers_from_texts(mind_outputs, state["problem"])
        mind_vote = _majority_answer(mind_answers)
        chair_output = chair_outputs_by_case.get(state["unique_id"], "")
        chair_answer = _extract_answer_from_trace_text(chair_output, state["problem"]) if chair_output else None
        chosen = chair_answer or mind_vote
        revised_answers = copy.deepcopy(mind_answers)
        if chair_output:
            revised_answers["Chair"] = chair_answer
        round_result = {
            "round_index": 1,
            "mind_outputs": mind_outputs,
            "chair_output": chair_output,
            "revised_answers": revised_answers,
            "chosen_answer": chosen,
        }
        state["som"] = {
            "num_rounds": 1,
            "round_results": [round_result],
            "mind_outputs": mind_outputs,
            "chair_output": chair_output,
            "revised_answers": revised_answers,
            "chosen_answer": chosen,
            "is_correct": _is_correct(chosen, state["gold_answer"], state["problem"]),
        }


def _build_dmad_update_prompt(
    agent_id: str,
    persona: str,
    question: str,
    own_response: str,
    peer_responses: dict[str, str],
) -> str:
    return (
        "You are participating in a Diversified Multi-Agent Debate.\n"
        f"Your role: {persona}.\n\n"
        f"Question:\n{question}\n\n"
        f"Your current response:\n{own_response}\n\n"
        f"Other agents' current responses:\n{_responses_block(peer_responses)}\n\n"
        "Revise your full solution if another response exposes a concrete local error. "
        "Keep your answer if your role-specific check still supports it. "
        "Do not appeal to popularity or authority; cite the local fact, inference, or requested object that decides the issue.\n\n"
        f"{_final_answer_instruction()}\n\n"
        f"Output only {agent_id}'s updated full solution."
    )


def _run_dmad(states: list[dict], runtime, args: argparse.Namespace) -> None:
    for state in states:
        state["dmad"] = {
            "responses": copy.deepcopy(state["initial_responses"]),
            "round_results": [],
            "personas": copy.deepcopy(DMAD_PERSONAS),
        }

    for round_idx in range(1, args.dmad_rounds + 1):
        prompt_records = []
        for state in states:
            responses = state["dmad"]["responses"]
            for agent_id, response in responses.items():
                peers = {peer_id: text for peer_id, text in responses.items() if peer_id != agent_id}
                persona = DMAD_PERSONAS.get(agent_id, "a careful independent reasoner")
                prompt = _build_dmad_update_prompt(agent_id, persona, state["problem"], response, peers)
                prompt_records.append((f"{state['unique_id']}::{agent_id}", prompt))

        outputs = []
        for start in range(0, len(prompt_records), args.prompt_batch_size):
            outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))

        outputs_by_case: dict[str, dict[str, str]] = {}
        for (alias, _), output in zip(prompt_records, outputs):
            case_id, agent_id = alias.split("::", 1)
            outputs_by_case.setdefault(case_id, {})[agent_id] = str(output or "").strip()

        for state in states:
            responses = state["dmad"]["responses"]
            raw_updates = outputs_by_case.get(state["unique_id"], {})
            for agent_id, output in raw_updates.items():
                if output:
                    responses[agent_id] = output
            revised_answers = _answer_map_from_responses(responses, state["problem"])
            chosen = _majority_answer(revised_answers)
            state["dmad"]["round_results"].append(
                {
                    "round_index": round_idx,
                    "personas": copy.deepcopy(DMAD_PERSONAS),
                    "raw_updates": raw_updates,
                    "responses": copy.deepcopy(responses),
                    "revised_answers": revised_answers,
                    "chosen_answer": chosen,
                }
            )

    for state in states:
        round_results = state["dmad"]["round_results"]
        final_round = round_results[-1] if round_results else {}
        chosen = final_round.get("chosen_answer") or _majority_answer(state["initial_answers"])
        state["dmad"].update(
            {
                "num_rounds": args.dmad_rounds,
                "revised_answers": copy.deepcopy(final_round.get("revised_answers", state["initial_answers"])),
                "chosen_answer": chosen,
                "is_correct": _is_correct(chosen, state["gold_answer"], state["problem"]),
            }
        )


def _origin_like_args(topology: str) -> SimpleNamespace:
    return SimpleNamespace(
        centralized=topology == "centralized",
        sparse=topology == "sparse",
        limit_one_peer=False,
        finalfilter=False,
        multi_persona=False,
        data="math",
        system_prompt_text="",
    )


def _sparse_ring_peer_map(agent_ids: list[str], prev_answers: dict[str, str | None]) -> dict[str, list[str]]:
    def differs(left: str | None, right: str | None) -> bool:
        if left is None and right is None:
            return False
        return left != right

    peer_map = {}
    for idx, agent_id in enumerate(agent_ids):
        if len(agent_ids) <= 1:
            peer_map[agent_id] = []
            continue
        neighbors = [
            agent_ids[(idx - 1) % len(agent_ids)],
            agent_ids[(idx + 1) % len(agent_ids)],
        ]
        deduped_neighbors = []
        for peer_id in neighbors:
            if peer_id != agent_id and peer_id not in deduped_neighbors:
                deduped_neighbors.append(peer_id)
        peer_map[agent_id] = [
            peer_id
            for peer_id in deduped_neighbors
            if differs(prev_answers.get(peer_id), prev_answers.get(agent_id))
        ]
    return peer_map


def _centralized_peer_map(agent_ids: list[str], prev_answers: dict[str, str | None]) -> dict[str, list[str]]:
    def differs(left: str | None, right: str | None) -> bool:
        if left is None and right is None:
            return False
        return left != right

    if not agent_ids:
        return {}
    center = agent_ids[0]
    peer_map = {}
    for agent_id in agent_ids:
        if agent_id == center:
            peers = [peer_id for peer_id in agent_ids if peer_id != center]
        else:
            peers = [center]
        peer_map[agent_id] = [
            peer_id
            for peer_id in peers
            if differs(prev_answers.get(peer_id), prev_answers.get(agent_id))
        ]
    return peer_map


def _build_topology_peer_map(
    topology: str,
    agent_ids: list[str],
    prev_answers: dict[str, str | None],
) -> dict[str, list[str]]:
    if topology == "sparse":
        return _sparse_ring_peer_map(agent_ids, prev_answers)
    if topology == "centralized":
        return _centralized_peer_map(agent_ids, prev_answers)
    raise ValueError(f"Unknown MAD topology: {topology}")


def _run_topology_mad(
    states: list[dict],
    runtime,
    args: argparse.Namespace,
    *,
    method: str,
    topology: str,
    rounds: int,
) -> None:
    for state in states:
        state[method] = {
            "responses": copy.deepcopy(state["initial_responses"]),
            "round_results": [],
            "topology": topology,
        }

    prompt_args = _origin_like_args(topology)
    for round_idx in range(1, rounds + 1):
        prompt_records = []
        context_by_case = {}
        for state in states:
            responses = state[method]["responses"]
            prev_answers = _answer_map_from_responses(responses, state["problem"])
            agent_ids = sorted(responses.keys())
            peer_map = _build_topology_peer_map(topology, agent_ids, prev_answers)
            update_messages = _build_direct_response_update_messages(
                prompt_args,
                state["problem"],
                responses,
                peer_map,
                personas=None,
            )
            context_by_case[state["unique_id"]] = {
                "prev_answers": prev_answers,
                "peer_map": peer_map,
                "responses": copy.deepcopy(responses),
            }
            for agent_id, message in update_messages.items():
                prompt_records.append((f"{state['unique_id']}::{agent_id}", message))

        raw_outputs = []
        for start in range(0, len(prompt_records), args.prompt_batch_size):
            raw_outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))

        outputs_by_case: dict[str, dict[str, str]] = {}
        for (alias, _), output in zip(prompt_records, raw_outputs):
            case_id, agent_id = alias.split("::", 1)
            outputs_by_case.setdefault(case_id, {})[agent_id] = output

        for state in states:
            case_id = state["unique_id"]
            previous_responses = context_by_case[case_id]["responses"]
            updated_responses = dict(previous_responses)
            raw_updates = outputs_by_case.get(case_id, {})
            for agent_id, raw_output in raw_updates.items():
                updated_responses[agent_id] = _parse_direct_update_output(
                    raw_output,
                    previous_responses.get(agent_id, ""),
                )
            revised_answers = _answer_map_from_responses(updated_responses, state["problem"])
            chosen = _majority_answer(revised_answers)
            state[method]["round_results"].append(
                {
                    "round_index": round_idx,
                    "topology": topology,
                    "input_answers": context_by_case[case_id]["prev_answers"],
                    "peer_map": copy.deepcopy(context_by_case[case_id]["peer_map"]),
                    "raw_updates": raw_updates,
                    "responses": copy.deepcopy(updated_responses),
                    "revised_answers": revised_answers,
                    "chosen_answer": chosen,
                }
            )
            state[method]["responses"] = updated_responses

    for state in states:
        round_results = state[method]["round_results"]
        final_round = round_results[-1] if round_results else {}
        chosen = final_round.get("chosen_answer") or _majority_answer(state["initial_answers"])
        state[method].update(
            {
                "num_rounds": rounds,
                "revised_answers": copy.deepcopy(final_round.get("revised_answers", state["initial_answers"])),
                "chosen_answer": chosen,
                "is_correct": _is_correct(chosen, state["gold_answer"], state["problem"]),
            }
        )


def _run_sparse_mad(states: list[dict], runtime, args: argparse.Namespace) -> None:
    _run_topology_mad(
        states,
        runtime,
        args,
        method="sparse_mad",
        topology="sparse",
        rounds=args.sparse_mad_rounds,
    )


def _run_centralized_mad(states: list[dict], runtime, args: argparse.Namespace) -> None:
    _run_topology_mad(
        states,
        runtime,
        args,
        method="centralized_mad",
        topology="centralized",
        rounds=args.centralized_mad_rounds,
    )


def _round0_subset_summary(results: list[dict], methods: list[str]) -> dict:
    num_cases = len(results)
    round0_correct = 0
    round0_available = 0
    per_method = {
        method: {
            "round0_correct_to_method_wrong": 0,
            "round0_wrong_to_method_correct": 0,
        }
        for method in methods
    }
    for item in results:
        initial_answers = item.get("initial_answers", {})
        round0_answer = _majority_answer(initial_answers) if initial_answers else None
        has_round0 = bool(initial_answers)
        round0_ok = has_round0 and _is_correct(round0_answer, item["gold_answer"], item.get("problem", ""))
        if has_round0:
            round0_available += 1
        if round0_ok:
            round0_correct += 1
        for method in methods:
            method_ok = bool(item.get(method, {}).get("is_correct"))
            if has_round0 and round0_ok and not method_ok:
                per_method[method]["round0_correct_to_method_wrong"] += 1
            if has_round0 and (not round0_ok) and method_ok:
                per_method[method]["round0_wrong_to_method_correct"] += 1
    return {
        "num_cases": num_cases,
        "round0_available_cases": round0_available,
        "round0_num_correct": round0_correct,
        "round0_accuracy": (round0_correct / round0_available) if round0_available else None,
        "per_method": per_method,
    }


def _method_num_rounds(method: str, args: argparse.Namespace) -> int:
    if method in {"cot", "cot_sc"}:
        return 1
    if method == "self_refine":
        return args.self_refine_rounds
    if method == "som":
        return 1
    if method == "origin_mad":
        return args.origin_mad_rounds
    if method == "dmad":
        return args.dmad_rounds
    if method == "sparse_mad":
        return args.sparse_mad_rounds
    if method == "centralized_mad":
        return args.centralized_mad_rounds
    raise KeyError(method)


def _method_summary(results: list[dict], method: str) -> dict:
    num_cases = len(results)
    num_correct = sum(1 for item in results if item.get(method, {}).get("is_correct"))
    return {
        "num_correct": num_correct,
        "accuracy": (num_correct / num_cases) if num_cases else 0.0,
    }


def _public_result(state: dict, methods: list[str]) -> dict:
    item = {
        "unique_id": state["unique_id"],
        "problem": state["problem"],
        "gold_answer": state["gold_answer"],
        "initial_answers": copy.deepcopy(state["initial_answers"]),
    }
    for method in methods:
        item[method] = copy.deepcopy(state[method])
    return item


def _render_log(payload: dict, methods: list[str]) -> str:
    lines = [
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {payload.get('dataset')}",
        f"Split: {payload.get('split')}",
        f"Cases: {payload.get('num_cases')}",
        f"Cases path: {payload.get('cases_path')}",
        f"History path: {payload.get('history_path')}",
    ]
    round0_summary = payload.get("round0_subset_summary", {})
    round0_acc = round0_summary.get("round0_accuracy")
    if round0_acc is None:
        lines.append("Round0 accuracy: n/a (independent-initialization methods)")
    else:
        lines.append(
            f"Round0 accuracy: {round0_summary.get('round0_num_correct')}/{round0_summary.get('round0_available_cases')} = {round0_acc:.4f}"
        )
    for method in methods:
        summary = payload["method_summaries"][method]
        lines.append(f"{method} accuracy: {summary['num_correct']}/{payload.get('num_cases')} = {summary['accuracy']:.4f}")
        metrics = payload["method_round_metrics"][method]
        for round_idx, value in enumerate(metrics["vote_acc"]):
            lines.append(f"[{method}] Round {round_idx} Vote Acc.: {value:.4f}")
    return "\n".join(lines) + "\n"


def _load_or_prepare_cases_and_history(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict, dict]:
    if args.cases_path and args.history_path:
        cases = _load_json(Path(args.cases_path))
        history_rows = _load_jsonl(Path(args.history_path))
        return cases, history_rows, {}, {}
    if args.cases_path:
        cases = _load_json(Path(args.cases_path))
        return cases, [], {}, {}

    if not args.dataset:
        raise ValueError("Either pass --cases_path/--history_path or pass --dataset with --work_dir.")
    if not args.work_dir:
        raise ValueError("--work_dir is required when preparing round0 history from --dataset.")

    _normalize_dataset_defaults(args)
    _ensure_default_limits(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cases = _build_cases(args)
    case_lookup = {case["unique_id"]: case for case in cases}
    full_cases_path = work_dir / "full.json"
    full_history_path = work_dir / "full.history.jsonl"
    _write_json(full_cases_path, cases)

    if args.reuse_prepared_round0 and full_history_path.exists():
        full_history_rows = _load_jsonl(full_history_path)
    else:
        _generate_round0_history(args=args, cases=cases, history_path=full_history_path)
        full_history_rows = _load_jsonl(full_history_path)

    round0_cases, round0_full_summary = _compute_round0_case_view(full_history_rows, case_lookup)
    _write_json(work_dir / "round0_full_case_view.json", round0_cases)

    if args.compare_all_cases:
        compare_cases = cases
        compare_history_rows = full_history_rows
        compare_cases_path = work_dir / "compare_all_cases.json"
        compare_history_path = work_dir / "compare_all.history.jsonl"
        round0_compare_summary = round0_full_summary
    else:
        nonunanim_ids = [item["unique_id"] for item in round0_cases if not item["is_unanimous"]]
        compare_cases = [case_lookup[case_id] for case_id in nonunanim_ids]
        compare_history_rows = _filter_history_rows(full_history_rows, set(nonunanim_ids))
        compare_cases_path = work_dir / "nonunanim_cases.json"
        compare_history_path = work_dir / "nonunanim.history.jsonl"
        subset_lookup = {case["unique_id"]: case for case in compare_cases}
        _, round0_compare_summary = _compute_round0_case_view(compare_history_rows, subset_lookup)

    _write_json(compare_cases_path, compare_cases)
    _write_jsonl(compare_history_path, compare_history_rows)
    args.cases_path = str(compare_cases_path)
    args.history_path = str(compare_history_path)
    return compare_cases, compare_history_rows, round0_full_summary, round0_compare_summary


def run_baselines(args: argparse.Namespace) -> dict:
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    methods = _method_list(args.methods)
    shared_round0_methods = {"self_refine", "origin_mad", "dmad", "sparse_mad", "centralized_mad"}
    cases, history_rows, round0_full_summary, round0_prep_summary = _load_or_prepare_cases_and_history(args)
    if args.case_ids:
        wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        cases = [case for case in cases if case.get("unique_id") in wanted]
        history_rows = _filter_history_rows(history_rows, {case["unique_id"] for case in cases})

    if history_rows:
        states = _prepare_states(cases, history_rows)
    elif set(methods) & shared_round0_methods:
        raise ValueError(f"Methods {sorted(set(methods) & shared_round0_methods)} require --history_path or prepared round0 history.")
    else:
        states = [_build_independent_case_state(case) for case in cases]
    runtime = load_runtime(_build_runtime_args(args))
    try:
        for method in methods:
            if method == "cot":
                _run_cot(states, runtime, args)
            elif method == "cot_sc":
                _run_cot_sc(states, runtime, args)
            elif method == "self_refine":
                _run_self_refine(states, runtime, args)
            elif method == "som":
                _run_som(states, runtime, args)
            elif method == "origin_mad":
                _run_origin_mad(states, runtime, args)
            elif method == "dmad":
                _run_dmad(states, runtime, args)
            elif method == "sparse_mad":
                _run_sparse_mad(states, runtime, args)
            elif method == "centralized_mad":
                _run_centralized_mad(states, runtime, args)
            else:  # pragma: no cover
                raise KeyError(method)
    finally:
        close_runtime(runtime)

    results = [_public_result(state, methods) for state in states]
    method_summaries = {method: _method_summary(results, method) for method in methods}
    method_round_metrics = {
        method: _build_mode_round_metrics(results, method, _method_num_rounds(method, args))
        for method in methods
    }
    payload = {
        "dataset": args.dataset or None,
        "split": args.split or None,
        "cases_path": args.cases_path,
        "history_path": args.history_path,
        "methods": methods,
        "num_cases": len(results),
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "round0_prompt_style": getattr(args, "round0_prompt_style", None),
        "round0_full_summary": round0_full_summary,
        "round0_subset_prep_summary": round0_prep_summary,
        "round0_subset_summary": _round0_subset_summary(results, methods),
        "method_summaries": method_summaries,
        "method_round_metrics": method_round_metrics,
        "results": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(_render_log(payload, methods), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--cases_path", default="")
    parser.add_argument("--history_path", default="")
    parser.add_argument("--work_dir", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--log_path", default="")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model_dir", default="/data/yyr/model")
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime_backend", default="vllm", choices=["vllm", "openai_api"])
    parser.add_argument("--api_base_url", default="")
    parser.add_argument("--api_beta_base_url", default="")
    parser.add_argument("--api_model", default="")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_timeout", type=float, default=120)
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--case_batch_size", type=int, default=100)
    parser.add_argument("--prompt_batch_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--sample_seed", type=int, default=7)
    parser.add_argument("--round0_prompt_style", default="claim_atomic")
    parser.add_argument("--origin_rounds", type=int, default=3)
    parser.add_argument("--compare_all_cases", action="store_true")
    parser.add_argument("--reuse_prepared_round0", action="store_true")
    parser.add_argument("--case_ids", default="")
    parser.add_argument("--cot_sc_samples", type=int, default=3)
    parser.add_argument("--cot_sc_use_history", action="store_true")
    parser.add_argument("--self_refine_rounds", type=int, default=2)
    parser.add_argument("--som_use_chair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--origin_mad_rounds", type=int, default=3)
    parser.add_argument("--dmad_rounds", type=int, default=2)
    parser.add_argument("--sparse_mad_rounds", type=int, default=2)
    parser.add_argument("--centralized_mad_rounds", type=int, default=2)
    args = parser.parse_args()

    payload = run_baselines(args)
    print(
        json.dumps(
            {
                "num_cases": payload["num_cases"],
                "methods": payload["methods"],
                "method_summaries": payload["method_summaries"],
                "output_path": args.output_path,
                "log_path": args.log_path,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
