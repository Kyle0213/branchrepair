from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from _local_bootstrap import PROJECT_ROOT
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.pipeline import NaturalLanguageGraphDebatePipeline
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.models import AgentTrace, ResolutionDecision
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.prompts import (
    build_atomic_trace_prompt,
    build_divergence_relation_analysis_prompt,
    build_divergence_resolution_prompt,
    build_incremental_claim_merge_prompt,
    build_pairwise_relation_analysis_prompt,
    build_pairwise_divergence_resolution_prompt,
    build_prefix_conflict_graph_prompt,
    build_shared_graph_audit_prompt,
    resolve_prompt_profile,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.protocol import (
    parse_atomic_trace,
    parse_graph_dossier,
    parse_resolution_note,
    parse_resolution_note_json_only,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.runtime import close_runtime, load_runtime, run_prompts
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.splitter import build_step_objects, compact_step_objects
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.local_evaluator import (
    extract_math_answer,
    extract_numeric_answer,
    is_math_correct,
    normalize_math_answer,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.label_answer import (
    allowed_label_answers,
    extract_label_answer,
    label_claim_is_bare_answer,
    nli_label_supported_by_claim,
    nli_label_supported_by_resolution,
    question_uses_label_answers,
    truth_status_label_supported_by_claim,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.mcq_answer import (
    extract_option_map,
    canonical_mcq_answer,
    mcq_answers_match,
    question_requests_option_text_answer,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.short_answer import (
    extract_short_answer,
    normalize_short_answer,
    question_requests_short_answer,
    short_answers_match,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.local_origin_mad import (
    _build_direct_response_update_messages,
    _get_answer_map,
    _get_peer_map,
    _parse_direct_update_output,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.local_runtime_backend import engine
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.target_answer import (
    target_note_provenance,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.target_contract import (
    analyze_question_target,
    clean_question_text,
)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _load_jsonl(path: Path):
    items = []
    with path.open() as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


_TARGET_NOTE_PIPELINE: NaturalLanguageGraphDebatePipeline | None = None


def _target_focus_note_for_problem(problem: str) -> str:
    global _TARGET_NOTE_PIPELINE
    if _TARGET_NOTE_PIPELINE is None:
        _TARGET_NOTE_PIPELINE = NaturalLanguageGraphDebatePipeline()
    return _TARGET_NOTE_PIPELINE._answer_split_target_note(problem)


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
        save_api_trace=bool(getattr(args, "save_api_trace", False)),
        max_new_tokens=args.max_new_tokens,
        max_model_len=getattr(args, "max_model_len", None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        token=token,
        multi_persona=False,
        data="math",
        memory_for_model_activations_in_gb=2,
    )


def _build_origin_args() -> SimpleNamespace:
    return SimpleNamespace(
        centralized=False,
        sparse=False,
        limit_one_peer=False,
        finalfilter=False,
        use_verifier=False,
        verifier_stage=False,
        verifier_mode="legacy",
        multi_persona=False,
        data="math",
        system_prompt_text="",
    )


def _pop_runtime_api_trace(runtime) -> list[dict]:
    pop_api_trace = getattr(getattr(runtime, "agent", None), "pop_api_trace", None)
    if callable(pop_api_trace):
        return pop_api_trace()
    return []


def _resolution_trace_context_name(value: str | None) -> str:
    normalized = str(value or "window").strip().lower()
    return normalized if normalized in {"window", "full"} else "window"


def _resolution_prompt_style_name(value: str | None) -> str:
    normalized = str(value or "profile").strip().lower()
    if normalized in {"", "default", "profile"}:
        return "profile"
    if normalized in {"minimal", "minimal_strategy", "strategy_only"}:
        return "minimal_strategy"
    if normalized in {"ledger", "ledger_strategy", "evidence_ledger"}:
        return "ledger_strategy"
    return "profile"


def _resolution_acceptance_policy_name(value: str | None) -> str:
    normalized = str(value or "guarded").strip().lower()
    if normalized in {"", "default", "guarded"}:
        return "guarded"
    if normalized in {"trust_model", "trust", "raw_model"}:
        return "trust_model"
    return "guarded"


def _strip_instruction_suffix(question: str) -> str:
    marker = " Make sure to state your final answer in curly brackets"
    if marker in question:
        return question.split(marker, 1)[0].strip()
    return question.strip()


def _build_origin_question_prompt(question: str) -> str:
    base_question = _strip_instruction_suffix(question)
    return (
        f'{base_question} '
        'Make sure to state your final answer in curly brackets at the very end of your response, '
        'just like: "{final answer: \\boxed{1}}".'
    )


def _agent_label(raw_agent_id: str) -> str:
    match = re.search(r"Agent(\d+)$", raw_agent_id)
    if match:
        return f"A{match.group(1)}"
    return raw_agent_id


def _looks_malformed_answer_candidate(candidate: str) -> bool:
    text = str(candidate or "").strip()
    if not text:
        return True
    if text.lower() in {"even", "odd"}:
        return False
    compact = re.sub(r"[^A-Za-z0-9]+", "", text).lower()
    if any(
        marker in compact
        for marker in (
            "letscompute",
            "letswrite",
            "writeasmallprogram",
            "useatable",
            "notfullyenumerated",
            "thereforetheresult",
            "thustheresult",
            "theremainingcondition",
        )
    ):
        return True
    if "\\" not in text and re.search(r"[A-Za-z]{8,}", compact):
        return True
    if re.fullmatch(r"[A-Za-z](?:\^T)?[A-Za-z]", text.replace(" ", "")):
        return True
    if re.match(r"^\d+\.\s*[*_`>#-]*\s*[A-Za-z]", text):
        return True
    if re.match(r"^>\s*[A-Za-z]", text):
        return True
    if re.fullmatch(r"A\d+\.s\d+:.+", text):
        return True
    if re.search(r"\d[A-Za-z]{3,}|[A-Za-z]{3,}\d", text) and "\\" not in text:
        return True
    if re.fullmatch(r"[A-Za-z\s]+", text) and len(text.replace(" ", "")) > 1:
        return True
    if len(re.findall(r"(?<!\\)\$", text)) % 2 == 1:
        return True
    if text.count("{") != text.count("}"):
        return True
    bracket_depth = 0
    for char in text:
        if char in "([":
            bracket_depth += 1
        elif char in ")]":
            bracket_depth -= 1
            if bracket_depth < 0:
                return True
    if bracket_depth != 0:
        return True
    if re.search(r"\\(?:frac|sqrt)\{[^{}]*$", text):
        return True
    return False


def _strip_answer_markup(candidate: str) -> str:
    text = str(candidate or "").strip()
    if not text:
        return ""
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"^\s*[*_`~#>\-]+\s*", "", text).strip()
        text = re.sub(r"\s*[*_`~#>\-]+\s*$", "", text).strip()
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*[:=\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"^\s*[*_`~#>\-]+\s*", "", text).strip()
        text = re.sub(r"\s*[*_`~#>\-]+\s*$", "", text).strip()
    return text


def _extract_explicit_label_answer(text: str) -> str | None:
    clean = str(text or "").replace("\n", " ").strip()
    if not clean:
        return None
    final_braced = re.findall(r"\{[^{}]*final answer[^{}]*\}", clean, flags=re.IGNORECASE)
    for seg in final_braced[::-1]:
        boxed = re.search(r"boxed\{\s*(-?\d{1,2})\s*\}", seg, flags=re.IGNORECASE)
        if boxed:
            return boxed.group(1)
        direct = re.search(r"final answer[^0-9-]*(-?\d{1,2})(?!\d)", seg, flags=re.IGNORECASE)
        if direct:
            return direct.group(1)
    tail = clean[-200:]
    boxed_tail = re.search(r"boxed\{\s*(-?\d{1,2})\s*\}\s*[\}\]\"'. ]*$", tail, flags=re.IGNORECASE)
    if boxed_tail:
        return boxed_tail.group(1)
    final_tail = re.search(r"final answer\s*[:\-]?\s*(?:\\boxed\{\s*)?(-?\d{1,2})(?:\s*\})?\s*[\]})\"'. ]*$", tail, flags=re.IGNORECASE)
    if final_tail:
        return final_tail.group(1)
    answer_tail = re.search(r"(?:therefore\s+)?answer\s*[:=\-]\s*(-?\d{1,2})\s*[\]})\"'. ]*$", tail, flags=re.IGNORECASE)
    if answer_tail:
        return answer_tail.group(1)
    return None


def _extract_answer_from_trace_text(text: str, question: str = "") -> str | None:
    if question_uses_label_answers(question):
        return extract_label_answer(text, question)
    if question_requests_option_text_answer(question):
        return canonical_mcq_answer(text, question)
    if question_requests_short_answer(question):
        return extract_short_answer(text)
    explicit_label = _extract_explicit_label_answer(text)
    if explicit_label is not None:
        return explicit_label
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if lines:
        if "=" in lines[-1]:
            rhs = lines[-1].rsplit("=", 1)[-1].strip().strip(".")
            rhs_is_simple = bool(
                re.fullmatch(
                    r"(?:"
                    r"[-+]?\d+(?:\.\d+)?"
                    r"|\\frac\{[^{}]+\}\{[^{}]+\}"
                    r"|[-+]?\d+\s*/\s*\d+"
                    r"|[-+]?(?:\\sqrt\{[^{}]+\}|\\pi)"
                    r"|[\(\[][^\]\[]*\\infty[^\]\[]*[\)\]]"
                    r"|[\(\[][^\]\[]*infty[^\]\[]*[\)\]]"
                    r")",
                    rhs,
                )
            )
            lhs = lines[-1].rsplit("=", 1)[0].strip()
            if rhs_is_simple and len(rhs) <= 120 and re.search(r"[A-Za-z\\]", lhs):
                normalized_rhs = normalize_math_answer(rhs)
                if normalized_rhs and not _looks_malformed_answer_candidate(normalized_rhs):
                    return normalized_rhs
        tail_value = _simple_answer_from_natural_sentence(lines[-1])
        if tail_value:
            return tail_value
        stripped_tail_line = _strip_answer_markup(lines[-1])
        if stripped_tail_line and stripped_tail_line != lines[-1]:
            normalized_stripped_tail_line = normalize_math_answer(stripped_tail_line)
            if normalized_stripped_tail_line and not _looks_malformed_answer_candidate(normalized_stripped_tail_line):
                return normalized_stripped_tail_line
    normalized = normalize_math_answer(extract_math_answer(text))
    if normalized and _looks_malformed_answer_candidate(normalized):
        stripped_normalized = normalize_math_answer(_strip_answer_markup(normalized))
        if stripped_normalized and not _looks_malformed_answer_candidate(stripped_normalized):
            return stripped_normalized
    if normalized:
        if not _looks_malformed_answer_candidate(normalized):
            return normalized
    if not lines:
        return None
    if "=" in lines[-1]:
        rhs = lines[-1].rsplit("=", 1)[-1].strip().strip(".")
        rhs_is_simple = bool(re.fullmatch(r"(?:[-+]?\d+(?:\.\d+)?|\\frac\{[^{}]+\}\{[^{}]+\}|[-+]?\\?pi|[-+]?\\sqrt\{?[^{}]+\}?)", rhs))
        if rhs_is_simple and len(rhs) <= 80:
            normalized_rhs = normalize_math_answer(rhs)
            if normalized_rhs:
                return normalized_rhs
    if re.search(r"\b(?:but|not|rejected|candidate|fails?|failed|invalid)\b", lines[-1], flags=re.IGNORECASE):
        return None
    if _looks_malformed_answer_candidate(lines[-1]):
        return None
    normalized_tail = normalize_math_answer(lines[-1])
    if normalized_tail and _looks_malformed_answer_candidate(normalized_tail):
        stripped_tail = normalize_math_answer(_strip_answer_markup(lines[-1]))
        if stripped_tail and not _looks_malformed_answer_candidate(stripped_tail):
            return stripped_tail
    if normalized_tail:
        if _looks_malformed_answer_candidate(normalized_tail):
            return None
        return normalized_tail
    return None


def _normalized_question_lower(question: str) -> str:
    return clean_question_text(question).lower()


def _question_requests_set_like_answer(question: str) -> bool:
    lowered = _normalized_question_lower(question)
    contract = analyze_question_target(question)
    requested = str(contract.requested_object or "").lower()
    requested_set_like = any(
        marker in requested
        for marker in ("range", "domain", "interval", "set", "solution set")
    )
    question_set_like = any(marker in lowered for marker in (" range", "domain", "interval")) or lowered.startswith("solve ")
    return requested_set_like or question_set_like


def _question_requests_count_like_answer(question: str) -> bool:
    lowered = _normalized_question_lower(question)
    contract = analyze_question_target(question)
    requested = str(contract.requested_object or "").lower()
    return bool(
        re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", lowered)
        or "degree of the polynomial" in lowered
        or "degree of the polynomial" in requested
        or requested.startswith("the degree of")
    )


def _question_requests_math_like_answer(question: str) -> bool:
    lowered = _normalized_question_lower(question)
    return (
        _question_requests_count_like_answer(question)
        or _question_requests_set_like_answer(question)
        or _question_requests_equation_like_answer(question)
        or _question_requests_vector_like_answer(question)
        or bool(re.search(r"\$[^$]+\$|\\(?:frac|sqrt|angle|pi|pmod)\b|[=<>^]", str(question or "")))
        or any(
            marker in lowered
            for marker in (
                "probability",
                "least possible value",
                "greatest possible value",
                "minimum value",
                "maximum value",
                "sum of",
                "product of",
                "coefficient",
                "root",
                "remainder",
                "modulo",
                "area",
                "volume",
                "perimeter",
                "distance",
                "length",
                "angle between",
            )
        )
    )


def _question_requests_equation_like_answer(question: str) -> bool:
    lowered = _normalized_question_lower(question)
    return any(
        marker in lowered
        for marker in (
            "image of the line",
            "equation of the line",
            "equation of line",
            "image line",
            "resulting line",
            "line of reflection",
            "transformed line",
        )
    )


def _question_requests_vector_like_answer(question: str) -> bool:
    lowered = _normalized_question_lower(question)
    return any(marker in lowered for marker in ("vector", "matrix", "column vector"))


def _question_requests_matrix_valued_answer(question: str) -> bool:
    cleaned = clean_question_text(question)
    lowered = cleaned.lower()
    if re.search(r"\b(?:find|compute|determine|evaluate)\s+the\s+matrix\b", lowered):
        return True
    if "matrix" not in lowered and "\\mathbf" not in cleaned:
        return False
    return bool(
        re.search(
            r"\b(?:compute|evaluate|find|determine)\b[^?.]*(?:\\mathbf\{?[A-Za-z]\}?|\b[A-Z]\b)\s*\^",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def _looks_matrix_valued_answer(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(
        re.search(r"\\begin\{[bp]?matrix\}|\\mathbf\{?I\}?|\\mathrm\{?I\}?|\bI\b", candidate)
    )


def _is_bare_symbolic_object_name(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(re.fullmatch(r"[A-Za-z](?:_\{?[\w]+\}?)?(?:\([^)]{0,20}\))?", candidate))


def _looks_interval_or_set_surface(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return any(
        marker in candidate
        for marker in ("\\cup", "cup", "\\infty", "infty", "∞", "{", "}", "(", ")", "[", "]")
    )


def _looks_inequality_fragment(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(re.search(r"(?:<|>|\\le|\\ge|≤|≥)", candidate))


def _looks_process_or_placeholder_answer(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return True
    normalized = clean_question_text(candidate).lower()
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    if any(
        marker in normalized
        for marker in (
            "correct claim:",
            "rewrite from:",
            "first conflict:",
            "why the other side fails:",
            "no new same-object claim",
            "keep paths",
            "drop paths",
            "in progress",
            "not yet",
            "use correct points",
            "calculate total revenue",
        )
    ):
        return True
    if any(marker in compact for marker in ("thereforetheresult", "thustheresult", "theremainingcondition")):
        return True
    if ":" in candidate and not re.match(r"^\s*[A-Za-z]\s*:", candidate):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", candidate) and len(candidate) > 1:
        return True
    return False


def _looks_short_answer_process_or_placeholder(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return True
    normalized = clean_question_text(candidate).lower()
    if any(
        marker in normalized
        for marker in (
            "correct claim:",
            "rewrite from:",
            "first conflict:",
            "why the other side fails:",
            "no new same-object claim",
            "keep paths",
            "drop paths",
            "in progress",
            "not yet",
            "use correct points",
            "calculate total revenue",
        )
    ):
        return True
    if ":" in candidate and not re.match(r"^\s*[A-Za-z]\s*:", candidate):
        return True
    return False


def _answer_matches_requested_object(question: str, answer: str | None) -> bool:
    text = str(answer or "").strip()
    if not text:
        return False
    if question_uses_label_answers(question):
        return text in set(allowed_label_answers(question))
    if question_requests_option_text_answer(question):
        return canonical_mcq_answer(text, question) is not None
    if question_requests_short_answer(question):
        extracted_short = extract_short_answer(text)
        check_text = extracted_short or text
        normalized_short = normalize_short_answer(check_text)
        if not normalized_short:
            return False
        if len(check_text) > 160:
            return False
        if _looks_short_answer_process_or_placeholder(check_text):
            return False
        return True
    if _looks_malformed_answer_candidate(text):
        return False
    if _is_bare_symbolic_object_name(text):
        return False
    if _looks_process_or_placeholder_answer(text):
        return False
    if _question_requests_matrix_valued_answer(question) and not _looks_matrix_valued_answer(text):
        return False
    return True


def _claim_tracks_requested_object(question: str, claim: str, answer: str | None) -> bool:
    if not answer:
        return False
    if not question:
        return True
    lowered = str(claim or "").lower()
    if any(
        marker in lowered
        for marker in (
            "final answer",
            "requested answer",
            "requested final answer",
            "requested set",
            "requested interval",
            "requested range",
            "image of the line",
            "image line",
        )
    ):
        return _answer_matches_requested_object(question, answer)
    if _question_requests_set_like_answer(question):
        if any(marker in lowered for marker in ("range", "domain", "interval", "set", "solution set")):
            return _answer_matches_requested_object(question, answer)
        return False
    if _question_requests_equation_like_answer(question) or _question_requests_vector_like_answer(question):
        return _answer_matches_requested_object(question, answer)
    return _answer_matches_requested_object(question, answer)


def _majority_answer(answer_map: dict[str, str | None]) -> str | None:
    answers = [answer for answer in answer_map.values() if answer]
    if not answers:
        return None
    groups: list[list[str]] = []
    for answer in answers:
        for group in groups:
            if _answers_equivalent_surface(answer, group[0]):
                group.append(answer)
                break
        else:
            groups.append([answer])
    best_count = max(len(group) for group in groups)
    tied = [group for group in groups if len(group) == best_count]
    for _, answer in answer_map.items():
        if answer and any(_answers_equivalent_surface(answer, group[0]) for group in tied):
            return answer
    return None


def _stable_majority_answer(answer_map: dict[str, str | None], *, min_support: int = 2) -> tuple[str | None, int]:
    answers = [answer for answer in answer_map.values() if answer]
    if not answers:
        return None, 0
    groups: list[list[str]] = []
    for answer in answers:
        for group in groups:
            if _answers_equivalent_surface(answer, group[0]):
                group.append(answer)
                break
        else:
            groups.append([answer])
    best_count = max(len(group) for group in groups)
    if best_count < min_support:
        return None, 0
    tied = [group for group in groups if len(group) == best_count]
    if len(tied) != 1:
        return None, 0
    representative = tied[0][0]
    for answer in answer_map.values():
        if answer and _answers_equivalent_surface(answer, representative):
            return answer, best_count
    return representative, best_count


def _all_answers_unanimous(answer_map: dict[str, str | None]) -> bool:
    answers = [answer for answer in answer_map.values() if answer]
    return (
        bool(answer_map)
        and len(answers) == len(answer_map)
        and _unique_equivalent_answer(answers) is not None
    )


def _response_answers(response_map: dict[str, str], question: str = "") -> dict[str, str | None]:
    return {
        agent_id: _extract_answer_from_trace_text(text, question)
        for agent_id, text in response_map.items()
    }


def _is_pairwise_resolution(decision: ResolutionDecision | dict) -> bool:
    divergence_id = decision.get("divergence_id", "") if isinstance(decision, dict) else decision.divergence_id
    return str(divergence_id or "").startswith("PAIR_")


def _is_synthesize_resolution(decision: ResolutionDecision | dict) -> bool:
    action = decision.get("action", "") if isinstance(decision, dict) else decision.action
    return str(action or "").strip() == "synthesize"


def _is_verified_synthesize_resolution(decision: ResolutionDecision | dict) -> bool:
    if not _is_synthesize_resolution(decision):
        return False
    canonical = decision.get("canonical_answer", "") if isinstance(decision, dict) else getattr(decision, "canonical_answer", "")
    return bool(str(canonical or "").strip())


def _has_actionable_resolution(resolutions: list[ResolutionDecision]) -> bool:
    return any(not _is_synthesize_resolution(decision) for decision in resolutions)


def _drop_synthesize_only_resolutions(resolutions: list[ResolutionDecision]) -> list[ResolutionDecision]:
    return [decision for decision in resolutions if not _is_synthesize_resolution(decision)]


def _public_resolution_log_text(text: str) -> str:
    public_text = str(text or "")
    replacements = (
        (r"\bchoose_left\b", "choose_A"),
        (r"\bchoose_right\b", "choose_B"),
        (r"\bwinning side is left\b", "winning side is Claim A"),
        (r"\bwinning side is right\b", "winning side is Claim B"),
        (r"\bwinning side: left\b", "winning side: Claim A"),
        (r"\bwinning side: right\b", "winning side: Claim B"),
    )
    for pattern, replacement in replacements:
        public_text = re.sub(pattern, replacement, public_text, flags=re.IGNORECASE)
    return public_text


def _public_action_name(action: str | None) -> str:
    action_text = str(action or "").strip()
    if action_text == "choose_left":
        return "choose_A"
    if action_text == "choose_right":
        return "choose_B"
    return action_text


def _public_winning_side(side: str | None) -> str:
    side_text = str(side or "").strip()
    if side_text == "left":
        return "Claim A"
    if side_text == "right":
        return "Claim B"
    return side_text


def _public_payload_view(payload):
    public_payload = copy.deepcopy(payload)

    def convert_item(item):
        if isinstance(item, dict):
            converted = {}
            for key, value in item.items():
                if key == "action":
                    converted[key] = _public_action_name(value)
                elif key == "winning_side":
                    converted[key] = _public_winning_side(value)
                else:
                    converted[key] = convert_item(value)
            return converted
        if isinstance(item, list):
            return [convert_item(value) for value in item]
        if isinstance(item, str):
            return _public_resolution_log_text(item)
        return item

    public_payload = convert_item(public_payload)
    metrics = public_payload.get("graph_round_metrics") if isinstance(public_payload, dict) else None
    if isinstance(metrics, dict):
        public_counts = []
        for round_counts in metrics.get("graph_action_counts", []) or []:
            merged = Counter()
            if isinstance(round_counts, dict):
                for action, count in round_counts.items():
                    merged[_public_action_name(action)] += int(count)
            public_counts.append(_counter_to_dict(merged))
        metrics["graph_action_counts"] = public_counts
        public_sides = []
        for round_counts in metrics.get("graph_winning_side_counts", []) or []:
            merged = Counter()
            if isinstance(round_counts, dict):
                for side, count in round_counts.items():
                    merged[_public_winning_side(side)] += int(count)
            public_sides.append(_counter_to_dict(merged))
        metrics["graph_winning_side_counts"] = public_sides
    return public_payload


def _coerce_synthesize_side_selection(decision: ResolutionDecision, graph) -> ResolutionDecision:
    if not _is_synthesize_resolution(decision):
        return decision
    if not decision.keep_paths or not decision.drop_paths:
        return decision
    divergence = next(
        (item for item in getattr(graph, "divergences", []) if item.divergence_id == decision.divergence_id),
        None,
    )
    if divergence is None:
        return decision
    keep = set(decision.keep_paths or [])
    drop = set(decision.drop_paths or [])
    if divergence.left_path_id in keep and divergence.right_path_id in drop:
        coerced = copy.deepcopy(decision)
        coerced.action = "choose_left"
        coerced.winning_side = "left"
        return coerced
    if divergence.right_path_id in keep and divergence.left_path_id in drop:
        coerced = copy.deepcopy(decision)
        coerced.action = "choose_right"
        coerced.winning_side = "right"
        return coerced
    return decision


def _swap_divergence_for_prompt_order(divergence):
    swapped = copy.deepcopy(divergence)
    swapped.left_path_id, swapped.right_path_id = divergence.right_path_id, divergence.left_path_id
    swapped.left_claim, swapped.right_claim = divergence.right_claim, divergence.left_claim
    return swapped


def _should_swap_resolution_prompt(case_id: str, divergence_id: str) -> bool:
    key = f"{case_id}:{divergence_id}"
    return sum(ord(ch) for ch in key) % 2 == 1


def _map_prompt_order_decision(decision: ResolutionDecision, *, swapped: bool) -> ResolutionDecision:
    if not swapped:
        return decision
    mapped = copy.deepcopy(decision)
    if mapped.action == "choose_left":
        mapped.action = "choose_right"
        mapped.winning_side = "right"
    elif mapped.action == "choose_right":
        mapped.action = "choose_left"
        mapped.winning_side = "left"
    elif mapped.winning_side == "left":
        mapped.winning_side = "right"
    elif mapped.winning_side == "right":
        mapped.winning_side = "left"
    return mapped


def _resolution_from_dict(payload: dict) -> ResolutionDecision:
    return ResolutionDecision(
        divergence_id=str(payload.get("divergence_id", "")),
        action=str(payload.get("action", "")),
        winning_side=str(payload.get("winning_side", "")),
        resolved_claim=str(payload.get("resolved_claim", "")),
        rationale=str(payload.get("rationale", "")),
        rewrite_from_claim_id=str(payload.get("rewrite_from_claim_id", "") or "C1"),
        keep_paths=list(payload.get("keep_paths") or []),
        drop_paths=list(payload.get("drop_paths") or []),
        canonical_answer=str(payload.get("canonical_answer", "") or ""),
    )


def _last_inline_math(text: str) -> str | None:
    matches = re.findall(r"\$([^$]+)\$", str(text or ""))
    for match in reversed(matches):
        candidate = match.strip()
        if candidate:
            return candidate
    return None


def _allowed_natural_answer_tail(tail: str, *, compact: bool = False) -> bool:
    cleaned = str(tail or "").strip()
    if not cleaned:
        return True
    if re.fullmatch(r"(?:\}|\$|[.,;:]|\s)*", cleaned):
        return True
    if compact:
        return bool(re.match(r"^(?:\}|\$|[.,;:])*(?:achieved|attained|with|by|from|when|as)", cleaned, flags=re.IGNORECASE))
    return bool(
        re.match(
            r"^(?:\}|\$|[.,;:]|\s)*(?:achieved|attained|with|by|from|when|as)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def _simple_answer_from_natural_sentence(text: str) -> str | None:
    sentence = str(text or "").strip()
    if not sentence:
        return None
    answer_pattern = r"(-?\d+(?:\.\d+)?|\\frac\{[^{}]+\}\{[^{}]+\}|-?\d+\s*/\s*\d+)"
    match = None
    for candidate_match in re.finditer(
        rf"\b(?:is|equals|computed\s+as|=)\s*(?:therefore|hence|thus)?\s*(?:\$|\\boxed\{{|\{{final answer:\s*\\boxed\{{)*"
        rf"{answer_pattern}",
        sentence,
        flags=re.IGNORECASE,
    ):
        tail = sentence[candidate_match.end() :].strip()
        if _allowed_natural_answer_tail(tail):
            match = candidate_match
    compact_match = None
    if not match:
        compact_sentence = re.sub(r"\s+", "", sentence)
        for candidate_match in re.finditer(
            rf"(?:is|equals|computedas|=)(?:therefore|hence|thus)?(?:\$|\\boxed\{{|\{{finalanswer:\s*\\boxed\{{)*"
            rf"{answer_pattern}",
            compact_sentence,
            flags=re.IGNORECASE,
        ):
            tail = compact_sentence[candidate_match.end() :].strip()
            if _allowed_natural_answer_tail(tail, compact=True):
                compact_match = candidate_match
        if not compact_match:
            return None
        prefix = compact_sentence[: compact_match.start()].lower()
        candidate = compact_match.group(1)
    else:
        prefix = sentence[: match.start()].lower()
        candidate = match.group(1)
    if not any(
        marker in prefix
        for marker in (
            "requested",
            "answer",
            "least",
            "smallest",
            "greatest",
            "largest",
            "minimum",
            "maximum",
            "sum",
            "total",
            "product",
            "count",
            "number",
            "value",
            "probability",
            "rate",
            "range",
            "gcd",
            "greatest common divisor",
            "divisor",
        )
    ):
        return None
    return normalize_math_answer(candidate.replace(" ", ""))


def _sorted_integer_list(values: list[str]) -> str:
    parsed = sorted(int(value) for value in values)
    return ",".join(str(value) for value in parsed)


def _canonical_column_vector(values: list[str]) -> str | None:
    cleaned = []
    for value in values:
        normalized = normalize_math_answer(value.strip())
        if not normalized:
            return None
        letter_remainder = re.sub(r"\\(?:frac|sqrt|pi|mathbf|text)\b", "", normalized)
        letter_remainder = re.sub(r"\bpi\b|\\?pi\b", "", letter_remainder, flags=re.IGNORECASE)
        letter_remainder = re.sub(r"(?<=\d)i\b|\bi(?=$|[+\-*/),}])", "", letter_remainder)
        if re.search(r"[A-Za-z]", letter_remainder):
            return None
        cleaned.append(normalized)
    return "\\begin{pmatrix}" + "\\\\".join(cleaned) + "\\end{pmatrix}"


def _canonical_answer_from_resolved_claim(resolved_claim: str) -> str | None:
    claim = str(resolved_claim or "").strip()
    if not claim:
        return None
    lowered = claim.lower()
    nonfinal_local_claim = any(
        marker in lowered
        for marker in (
            "must subtract",
            "need to subtract",
            "needs to subtract",
            "before subtract",
            "not all lanes",
            "not all cases",
            "excluded from the domain",
            "exclude from the domain",
            "denominator is zero",
        )
    )
    category_count_claim = bool(
        re.search(r"\bcounts?\s+for\b", lowered)
        or re.search(r"\bthe\s+counts?\s+(?:are|is)\b", lowered)
        or "respectively" in lowered and re.search(r"\bcounts?\b", lowered)
    )
    if nonfinal_local_claim or category_count_claim:
        return None
    singular_root_only = bool(re.search(r"\b(?:a|one)\s+root\b|\bas\s+a\s+root\b", lowered)) and not re.search(
        r"\b(?:all|integer|rational)?\s*roots?\s+(?:are|is|include|yielding|given by)\b",
        lowered,
    )
    direction_vector_only = "direction vector" in lowered and not any(
        marker in lowered for marker in ("requested", "final answer", "answer is", "answer:")
    )

    def explicit_equation_answer(segment: str) -> str | None:
        match = re.match(
            r"^(?:[a-zA-Z](?:_\{?[\w]+\}?)?|[a-zA-Z]\s*\([^)]{1,20}\))\s*=\s*(.+)$",
            segment.strip(),
        )
        if not match:
            return None
        rhs = match.group(1).strip().rstrip(".")
        if not rhs or len(rhs) > 120 or _looks_malformed_answer_candidate(rhs):
            return None
        letter_remainder = re.sub(r"\\(?:frac|sqrt|pi|mathbf|text)\b", "", rhs)
        letter_remainder = re.sub(r"\bpi\b|\\?pi\b", "", letter_remainder, flags=re.IGNORECASE)
        letter_remainder = re.sub(r"(?<=\d)i\b|\bi(?=$|[+\-*/),}])", "", letter_remainder)
        if re.search(r"[A-Za-z]", letter_remainder):
            return None
        normalized = normalize_math_answer(rhs)
        return normalized or rhs

    component_matches = re.findall(
        r"\bv_?(\d)\s*=\s*(\$[^$]+\$|-?\\frac\{[^}]+\}\{[^}]+\}|-?\d+(?:/\d+)?)",
        claim,
    )
    if len(component_matches) >= 2:
        ordered = sorted(
            ((int(index), value.strip("$")) for index, value in component_matches),
            key=lambda item: item[0],
        )
        if [index for index, _ in ordered] == list(range(1, len(ordered) + 1)):
            vector = _canonical_column_vector([value for _, value in ordered])
            if vector:
                return vector

    vector_matches = re.findall(r"\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}", claim, flags=re.DOTALL)
    if vector_matches and not direction_vector_only:
        body = vector_matches[-1] if len(vector_matches) > 1 else vector_matches[0]
        values = [part.strip() for part in re.split(r"\\\\", body) if part.strip()]
        if len(values) >= 2:
            vector = _canonical_column_vector(values)
            if vector:
                return vector

    if "range" in lowered and (" is " in lowered or " ranges from " in lowered):
        inline = _last_inline_math(claim)
        if inline and any(token in inline for token in ("\\infty", "infty", "∞", "[", "]", "(", ")")):
            normalized = normalize_math_answer(inline)
            return normalized or inline.replace(" ", "")
        if re.search(r"ranges?\s+from\s+\$?\\?-?infty\$?\s+to\s+\$?0\$?", lowered) and "including 0" in lowered:
            return "(-\\infty,0]"
    if any(marker in lowered for marker in ("requested set is", "requested interval is", "requested range is", "requested set remains", "requested interval remains")):
        inline = _last_inline_math(claim)
        if inline and any(token in inline for token in ("\\infty", "infty", "∞", "[", "]", "(", ")")):
            normalized = normalize_math_answer(inline)
            return normalized or inline.replace(" ", "")
        interval_match = re.search(r"(\([^\n]{0,80}\\infty[^\n]{0,80}\)|\[[^\n]{0,80}\\infty[^\n]{0,80}\]|\([^\n]{0,80}infty[^\n]{0,80}\)|\[[^\n]{0,80}infty[^\n]{0,80}\])", claim)
        if interval_match:
            candidate = interval_match.group(1).replace(" ", "")
            normalized = normalize_math_answer(candidate)
            return normalized or candidate

    final_sum_match = re.search(
        r"\b(?:least possible|minimal|minimum)\s+sum\b[^.:\n]*\bis\s+\$?(-?\d+)\$?",
        claim,
        flags=re.IGNORECASE,
    )
    if final_sum_match:
        return final_sum_match.group(1)

    roots_match = re.search(
        r"\b(?:integer\s+)?roots?\s+(?:(?:are|is|include|yielding|yielded by|given by)\s+)?([^.\n]+)",
        claim,
        flags=re.IGNORECASE,
    )
    if roots_match:
        root_values = re.findall(r"(?<![\w/])-?\d+(?![\w/])", roots_match.group(1))
        if len(root_values) >= 2:
            return _sorted_integer_list(root_values)

    yielding_value_match = re.search(
        r"\b(?:yielding|giving|resulting in|with)\s+(?:an?\s+)?(?:final\s+)?(?:value|answer|result)\s+of\s+"
        r"(?:\$|\\boxed\{|\{final answer:\s*\\boxed\{)*\s*([^$}.\n]{1,80})",
        claim,
        flags=re.IGNORECASE,
    )
    if yielding_value_match:
        candidate = re.split(
            r",\s*(?:calculated|computed|found|obtained|which|where|as)\b",
            yielding_value_match.group(1).strip().rstrip("."),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        normalized = normalize_math_answer(candidate)
        if normalized:
            return normalized

    if re.search(r"=\s*\$?-?\\mathbf\{I\}\$?\s*$", claim):
        return "-\\mathbf{I}"

    final_value_match = re.search(
        r"\b(?:the\s+)?(?:number of [^.\n]{1,80}|value of [a-zA-Z]|shortest distance|requested final answer)\s+"
        r"(?:is|=)\s+(?:\$|\\boxed\{|\{final answer:\s*\\boxed\{)*\s*([^$}.\n]{1,80})",
        claim,
        flags=re.IGNORECASE,
    )
    if final_value_match:
        candidate = re.split(
            r",\s*(?:calculated|computed|found|obtained|which|where|as)\b",
            final_value_match.group(1).strip().rstrip("."),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        normalized = normalize_math_answer(candidate)
        if normalized:
            return normalized

    generic_object_match = re.search(
        r"\b(?:the\s+)?(?:requested\s+)?"
        r"(?:remainder|expression|probability|area|count|sum|total|product|value|interval|set|range)"
        r"[^.\n]{0,80}?\s+(?:is|=)\s+"
        r"(?:\$|\\boxed\{|\{final answer:\s*\\boxed\{)*\s*([^$}\n]{1,120})",
        claim,
        flags=re.IGNORECASE,
    )
    if generic_object_match:
        candidate = generic_object_match.group(1).strip().rstrip(".")
        normalized = normalize_math_answer(candidate)
        if normalized:
            return normalized
        if not _looks_malformed_answer_candidate(candidate):
            return candidate.replace(" ", "")

    target_value_match = re.search(
        r"\b(?:answer|count|sum|total|product|value)\s+of\s+(?:\$|\\boxed\{)*\s*"
        r"(-?\d+(?:\.\d+)?|\\frac\{[^{}]+\}\{[^{}]+\}|-?\d+\s*/\s*\d+)",
        claim,
        flags=re.IGNORECASE,
    )
    if target_value_match:
        candidate = target_value_match.group(1).strip()
        normalized = normalize_math_answer(candidate)
        if normalized:
            return normalized

    if any(marker in lowered for marker in ("image line", "image of the line", "resulting in the image", "final answer", "requested answer")):
        inline_segments = re.findall(r"\$([^$]+)\$", claim)
        for segment in reversed(inline_segments):
            if re.match(r"\s*y\s*=", segment, flags=re.IGNORECASE):
                candidate = segment.strip().rstrip(".")
                return candidate.replace(" ", "")
        line_match = re.search(r"\by\s*=\s*[^.\n,;]+", claim, flags=re.IGNORECASE)
        if line_match:
            candidate = line_match.group(0).strip().rstrip(".")
            return candidate.replace(" ", "")

    simple_sentence_answer = _simple_answer_from_natural_sentence(claim)
    if simple_sentence_answer:
        return simple_sentence_answer

    inline_segments = re.findall(r"\$([^$]+)\$", claim)
    if not singular_root_only:
        for segment in reversed(inline_segments):
            answer = explicit_equation_answer(segment)
            if answer:
                return answer
        answer = explicit_equation_answer(claim)
        if answer:
            return answer

    return None


def _short_answer_from_slot_claim(resolved_claim: str, *, question: str = "") -> str | None:
    if not question_requests_short_answer(question):
        return None
    claim = str(resolved_claim or "").strip()
    if not claim:
        return None
    lowered = claim.lower()
    if not any(marker in lowered for marker in ("requested slot", "answer slot", "exact short answer phrase")):
        return None
    slot_nouns = (
        "answer",
        "answer phrase",
        "answer slot",
        "candidate",
        "city",
        "country",
        "date",
        "entity",
        "event",
        "exact phrase",
        "name",
        "organization",
        "outcome",
        "person",
        "place",
        "requested slot",
        "result",
        "short answer phrase",
        "slot",
        "title",
        "value",
        "year",
    )
    slot_noun_pattern = "|".join(re.escape(item) for item in sorted(slot_nouns, key=len, reverse=True))
    exact_phrase_patterns = [
        r"\b(?:exact\s+short\s+answer\s+phrase|exact\s+phrase|short\s+answer\s+phrase)\s+(?:is|=)\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        r"\b(?:with|as|to)\s+(?:the\s+)?(?:exact\s+short\s+answer\s+phrase|exact\s+phrase|short\s+answer\s+phrase)\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        r"[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]\s*,?\s+so\s+(?:keep|use)\s+that\s+exact\s+phrase",
        r"\bbridges?[^.\n;]{0,160}?\s+to\s+(?:the\s+)?[a-z][a-z\s-]{0,40}\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        r"\bso\s+(?:keep|use)\s+(?:that\s+exact\s+phrase\s+)?[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
    ]
    for pattern in exact_phrase_patterns:
        for match in re.finditer(pattern, claim, flags=re.IGNORECASE):
            candidate = match.group(1).strip().strip("\"'`“”‘’").strip()
            if not candidate or len(candidate) > 120:
                continue
            if any(bad in candidate.lower() for bad in ("cannot be determined", "not enough information", "unknown", "none")):
                continue
            if question and not _answer_matches_requested_object(question, candidate):
                continue
            if normalize_short_answer(candidate):
                return candidate
    patterns = [
        r"\bexact\s+short\s+answer\s+phrase\s+(?:is|=)\s*[\"'“”‘’`]*([^\"'“”‘’`.\n;]+)",
        r"\bexact\s+short\s+answer\s+phrase\s*[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        r"\bto\s+the\s+exact\s+short\s+answer\s+phrase\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        r"\bwith\s+the\s+exact\s+short\s+answer\s+phrase\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        r"\bso\s+(?:keep|use)\s+that\s+exact\s+phrase\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        r"\bso\s+(?:keep|use)\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        r"\bthe\s+exact\s+phrase\s+is\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        rf"\bto\s+(?:the\s+)?(?:{slot_noun_pattern})\s+[\"'“”‘’`]([^\"'“”‘’`\n]+)[\"'“”‘’`]",
        rf"\bto\s+(?:the\s+)?(?:{slot_noun_pattern})\s+([A-Z0-9][^\"'“”‘’`,.;\n]+)",
        r"\bbridges?\s+(?:the\s+)?(?:evidence|context|question|answer\s+slot|requested\s+slot|battle|war|path|trace|person|entity|hop|chain|this|it)[^.\n;]{0,120}?\s+to\s+[\"'“”‘’`]*([^\"'“”‘’`,.;\n]+)",
        r"\bevidence\s+bridges?[^.\n;]{0,120}?\s+to\s+[\"'“”‘’`]*([^\"'“”‘’`,.;\n]+)",
        r"\bcontext\s+(?:states|gives|identifies|bridges?)[^.\n;]{0,120}?\s+(?:as|to)\s+[\"'“”‘’`]*([^\"'“”‘’`,.;\n]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, claim, flags=re.IGNORECASE):
            candidate = extract_short_answer(match.group(1)) or match.group(1).strip()
            candidate = candidate.strip().strip("\"'`“”‘’").strip()
            candidate = re.sub(
                rf"^(?:the\s+)?(?:{slot_noun_pattern})\s+(?=[A-Z0-9])",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip()
            if not candidate or len(candidate) > 120:
                continue
            if re.fullmatch(rf"(?:the\s+)?(?:{slot_noun_pattern})", candidate, flags=re.IGNORECASE):
                continue
            if any(bad in candidate.lower() for bad in ("cannot be determined", "not enough information", "unknown")):
                continue
            if question and not _answer_matches_requested_object(question, candidate):
                continue
            normalized = normalize_short_answer(candidate)
            if normalized:
                return candidate
    return None


def _canonical_answer_from_resolutions(resolutions: list[ResolutionDecision], *, question: str = "") -> str | None:
    for decision in resolutions:
        if _is_synthesize_resolution(decision) and not _is_verified_synthesize_resolution(decision):
            continue
        if not _resolution_has_strong_action(decision):
            continue
        if (
            str(getattr(decision, "action", "") or "").strip() not in {"choose_left", "choose_right"}
            and not _is_verified_synthesize_resolution(decision)
        ):
            continue
        answer = str(getattr(decision, "canonical_answer", "") or "").strip()
        if answer:
            if question_requests_short_answer(question) and not _question_requests_math_like_answer(question):
                return answer
            if question_uses_label_answers(question):
                return extract_label_answer(answer, question) or answer
            if extract_option_map(question):
                landed = _landing_answer(answer, question=question)
                if landed:
                    return landed
            if question_requests_option_text_answer(question):
                return canonical_mcq_answer(answer, question) or answer
            if _is_synthesize_resolution(decision) and _is_verified_synthesize_resolution(decision):
                normalized = normalize_math_answer(answer)
                return normalized or answer
        explicit_answer = _resolution_explicit_final_answer(decision, question=question)
        if explicit_answer:
            return explicit_answer
        if question_requests_short_answer(question):
            continue
        if not question_uses_label_answers(question):
            continue
        claim_answer = _resolution_claim_answer(decision, question=question)
        if claim_answer:
            return claim_answer
    return None


def _strict_resolution_landing_profile(prompt_profile: str | None) -> bool:
    resolved = resolve_prompt_profile(prompt_profile)
    return resolved.name in {"logiqa_relation_v8", "logiqa_relation_v9", "logiqa_relation_v10"}


def _rewrite_supported_landing_profile(prompt_profile: str | None) -> bool:
    resolved = resolve_prompt_profile(prompt_profile)
    name = str(resolved.name or "")
    return name.endswith("_bridge_audit") or name.endswith("_rewrite_supported")


def _resolution_has_strong_action(decision: ResolutionDecision | dict) -> bool:
    action = decision.get("action", "") if isinstance(decision, dict) else getattr(decision, "action", "")
    return str(action or "").strip() in {"choose_left", "choose_right", "synthesize"}


def _weak_resolution_action(decision: ResolutionDecision | dict) -> bool:
    action = decision.get("action", "") if isinstance(decision, dict) else getattr(decision, "action", "")
    return str(action or "").strip() not in {"choose_left", "choose_right", "synthesize"}


def _allowed_resolution_canonical_answer(
    resolutions: list[ResolutionDecision],
    *,
    question: str = "",
    prompt_profile: str | None = None,
    visible_answers: dict[str, str | None] | None = None,
    revised_answers: dict[str, str | None] | None = None,
) -> str | None:
    if _strict_resolution_landing_profile(prompt_profile):
        for decision in resolutions:
            explicit_answer = _resolution_explicit_final_answer(decision, question=question)
            if explicit_answer and _resolution_answer_allowed_by_visible_support(
                explicit_answer,
                resolutions,
                question=question,
                visible_answers=visible_answers,
                revised_answers=revised_answers,
            ):
                return explicit_answer
        return None
    answer = _canonical_answer_from_resolutions(resolutions, question=question)
    if not answer:
        return None
    if (
        _rewrite_supported_landing_profile(prompt_profile)
        and not _answer_has_observed_support(answer, visible_answers, revised_answers)
    ):
        return None
    if not _resolution_answer_allowed_by_visible_support(
        answer,
        resolutions,
        question=question,
        visible_answers=visible_answers,
        revised_answers=revised_answers,
    ):
        return None
    return answer


def _resolution_answer_allowed_by_visible_support(
    answer: str | None,
    resolutions: list[ResolutionDecision],
    *,
    question: str = "",
    visible_answers: dict[str, str | None] | None = None,
    revised_answers: dict[str, str | None] | None = None,
) -> bool:
    if not answer:
        return False
    if not visible_answers and not revised_answers:
        return True
    if _answer_has_observed_support(answer, visible_answers, revised_answers):
        return True
    for decision in resolutions or []:
        if not _resolution_has_strong_action(decision):
            continue
        explicit_answer = _resolution_explicit_final_answer(decision, question=question)
        if explicit_answer and _answers_equivalent_surface(explicit_answer, answer):
            return True
        claim_answer = _resolution_completed_requested_object_answer(decision, question=question)
        if claim_answer and _answers_equivalent_surface(claim_answer, answer):
            return True
    if _question_requests_math_like_answer(question) and not question_uses_label_answers(question):
        return any(
            _resolution_completed_requested_object_answer(decision, question=question)
            and _answers_equivalent_surface(_resolution_completed_requested_object_answer(decision, question=question), answer)
            for decision in resolutions
            if _resolution_has_strong_action(decision)
        )
    return False


def _landing_answer(answer: str | None, *, question: str = "") -> str | None:
    if answer is None:
        return None
    text = str(answer).strip()
    if question_uses_label_answers(question):
        return extract_label_answer(text, question)
    if question_requests_option_text_answer(question):
        return canonical_mcq_answer(text, question)
    if question_requests_short_answer(question):
        extracted_short = extract_short_answer(text)
        if extracted_short:
            return extracted_short
        short_candidate = normalize_short_answer(text)
        if short_candidate:
            return short_candidate
    if _question_requests_equation_like_answer(question):
        normalized_equation = text.replace(" ", "")
        if _answer_matches_requested_object(question, normalized_equation):
            return normalized_equation
    if "=" in text and not _question_requests_equation_like_answer(question):
        rhs = text.rsplit("=", 1)[-1].strip().strip(".")
        lhs = text.rsplit("=", 1)[0].strip()
        if rhs and re.search(r"[A-Za-z\\]", lhs):
            normalized_rhs = normalize_math_answer(rhs)
            if normalized_rhs and not _looks_malformed_answer_candidate(normalized_rhs):
                if not question or _answer_matches_requested_object(question, normalized_rhs):
                    return normalized_rhs
    extracted = _extract_answer_from_trace_text(str(answer), question)
    if extracted:
        if not question or _answer_matches_requested_object(question, extracted):
            return extracted
    normalized = normalize_math_answer(answer)
    candidate = text if _question_requests_equation_like_answer(question) and "=" in text else (normalized or answer)
    if question and not _answer_matches_requested_object(question, candidate):
        return None
    return candidate


def _inferred_answer_from_resolutions(resolutions: list[ResolutionDecision | dict]) -> str | None:
    for decision in resolutions:
        if _is_synthesize_resolution(decision):
            continue
        resolved_claim = decision.get("resolved_claim", "") if isinstance(decision, dict) else getattr(decision, "resolved_claim", "")
        answer = _canonical_answer_from_resolved_claim(resolved_claim)
        if answer:
            normalized = normalize_math_answer(answer)
            return normalized or answer
    return None


def _answers_equivalent_surface(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    left_short = extract_short_answer(str(left)) or str(left)
    right_short = extract_short_answer(str(right)) or str(right)
    left_short_normalized = normalize_short_answer(left_short)
    right_short_normalized = normalize_short_answer(right_short)
    if left_short_normalized and right_short_normalized and left_short_normalized == right_short_normalized:
        return True
    left_normalized = normalize_math_answer(left)
    right_normalized = normalize_math_answer(right)
    if left_normalized and right_normalized and left_normalized == right_normalized:
        return True
    left_ints = re.findall(r"(?<![\w/])-?\d+(?![\w/])", str(left_normalized or left))
    right_ints = re.findall(r"(?<![\w/])-?\d+(?![\w/])", str(right_normalized or right))
    if len(left_ints) >= 2 and len(right_ints) >= 2 and sorted(map(int, left_ints)) == sorted(map(int, right_ints)):
        return True
    return is_math_correct(left, right) or is_math_correct(right, left)


def _answers_semantically_changed(left: str | None, right: str | None) -> bool:
    if left is None and right is None:
        return False
    if left is None or right is None:
        return True
    return normalize_math_answer(left) != normalize_math_answer(right)


def _is_decimal_approximation_answer(answer: str | None) -> bool:
    text = str(answer or "").strip().lower()
    if not text:
        return False
    return bool(re.search(r"(?<![A-Za-z])[-+]?\d+\.\d+", text)) or "approx" in text or "approximately" in text


def _is_exact_symbolic_answer(answer: str | None) -> bool:
    text = str(answer or "").strip()
    if not text:
        return False
    exact_markers = ("\\sqrt", "\\frac", "\\pi", "^\\circ", "\\text", "{", "}")
    return any(marker in text for marker in exact_markers)


def _is_exactness_downgrade(previous_answer: str | None, revised_answer: str | None) -> bool:
    if not previous_answer or not revised_answer:
        return False
    if not _is_exact_symbolic_answer(previous_answer):
        return False
    if not _is_decimal_approximation_answer(revised_answer):
        return False
    if _answers_equivalent_surface(previous_answer, revised_answer):
        return False
    return True


def _observed_graph_answers(initial_answers: dict[str, str | None], round_results: list[dict]) -> list[str]:
    answers = [answer for answer in initial_answers.values() if answer]
    for round_result in round_results:
        answers.extend(answer for answer in (round_result.get("revised_answers") or {}).values() if answer)
        chosen = round_result.get("chosen_answer")
        if chosen:
            answers.append(chosen)
    return answers


def _final_inferred_answer(
    initial_answers: dict[str, str | None],
    graph_round_results: list[dict],
) -> str | None:
    if not graph_round_results:
        return None
    final_round = graph_round_results[-1]
    final_answers = [
        answer
        for answer in (final_round.get("revised_answers") or {}).values()
        if answer
    ]
    candidate = _inferred_answer_from_resolutions(final_round.get("resolutions", []) or [])
    if candidate and any(_answers_equivalent_surface(candidate, answer) for answer in final_answers):
        return candidate
    return None


def _supported_inferred_resolution_answer(
    resolutions: list[ResolutionDecision],
    revised_answers: dict[str, str | None],
    *,
    min_support: int = 2,
) -> str | None:
    candidate = _inferred_answer_from_resolutions(resolutions)
    if not candidate:
        return None
    support = sum(1 for answer in revised_answers.values() if _answers_equivalent_surface(candidate, answer))
    return candidate if support >= min_support else None


def _answer_support_count(answer: str | None, answer_map: dict[str, str | None]) -> int:
    if not answer:
        return 0
    return sum(
        1
        for item in (answer_map or {}).values()
        if item and _answers_equivalent_surface(answer, item)
    )


def _answer_has_observed_support(answer: str | None, *answer_maps: dict[str, str | None] | None) -> bool:
    if not answer:
        return False
    return any(
        _answers_equivalent_surface(answer, observed)
        for answer_map in answer_maps
        for observed in (answer_map or {}).values()
        if observed
    )


def _resolution_names_bridge_error_for_answer(
    resolutions: list[ResolutionDecision],
    answer: str | None,
    *,
    question: str = "",
) -> bool:
    if not answer:
        return False
    answer_text = re.escape(str(answer).strip())
    bridge_terms = (
        "bridge",
        "mapping",
        "maps",
        "option text",
        "requested",
        "wrong object",
        "nearby",
        "polarity",
        "eliminat",
        "not support",
        "unsupported",
        "contradict",
        "does not answer",
        "fails",
        "invalid",
    )
    for decision in resolutions or []:
        if not _resolution_has_strong_action(decision):
            continue
        text = " ".join(
            str(getattr(decision, field, "") or "")
            for field in ("resolved_claim", "rationale", "canonical_answer")
        ).lower()
        if not text:
            continue
        mentions_answer = bool(re.search(rf"\b{answer_text}\b", text, flags=re.IGNORECASE))
        if not mentions_answer and question_uses_label_answers(question):
            option_map = extract_option_map(question)
            option_text = (option_map or {}).get(str(answer).strip())
            if option_text:
                compact_option = re.sub(r"\s+", " ", option_text.lower()).strip()
                mentions_answer = compact_option and compact_option[:40] in text
        negated_bridge_error = bool(
            re.search(
                r"\b(?:no|not|without)\s+(?:explicit\s+|clear\s+|visible\s+)?(?:bridge|mapping|requested|option text|wrong object|unsupported|contradict|invalid|error)",
                text,
            )
            or re.search(
                r"\b(?:does not|do not|did not|cannot|can't)\s+[^.\n]{0,80}\b(?:name|show|identify|establish)\s+[^.\n]{0,80}\b(?:bridge|mapping|unsupported|contradict|invalid|error)",
                text,
            )
        )
        if mentions_answer and any(term in text for term in bridge_terms) and not negated_bridge_error:
            return True
    return False


def _answer_selection_policy_name(policy: str | None) -> str:
    name = str(policy or "standard").strip().lower()
    return name if name in {"standard", "supported_answer_memory", "stable_majority_guard"} else "standard"


def _canonical_answer_mode_name(mode: str | None) -> str:
    name = str(mode or "broadcast_stop").strip().lower()
    return name if name in {"broadcast_stop", "select_only"} else "broadcast_stop"


def _stall_policy_name(policy: str | None) -> str:
    name = str(policy or "stop").strip().lower()
    return name if name in {"stop", "continue"} else "stop"


def _unanimous_policy_name(policy: str | None) -> str:
    name = str(policy or "skip").strip().lower()
    return name if name in {"skip", "audit"} else "skip"


def _memory_answer_record(
    answer: str,
    revised_answers: dict[str, str | None],
    *,
    round_index: int,
    source: str,
) -> dict:
    return {
        "answer": answer,
        "support": _answer_support_count(answer, revised_answers),
        "round_index": int(round_index),
        "source": source,
    }


def _memory_answer_source(
    answer: str | None,
    *,
    canonical_answer: str | None,
    continuation_answer: str | None,
    preferred_answer: str | None,
    majority_answer: str | None,
    accepted_answer_changed: bool,
) -> str:
    if answer and canonical_answer and _answers_equivalent_surface(answer, canonical_answer):
        return "canonical_resolution"
    if answer and continuation_answer and _answers_equivalent_surface(answer, continuation_answer):
        return "accepted_continuation"
    if answer and preferred_answer and _answers_equivalent_surface(answer, preferred_answer):
        return "resolution_preference"
    if (
        answer
        and majority_answer
        and accepted_answer_changed
        and _answers_equivalent_surface(answer, majority_answer)
    ):
        return "accepted_rewrite_majority"
    return ""


def _apply_supported_answer_memory_policy(
    state_graph: dict,
    question: str,
    chosen_answer: str | None,
    revised_answers: dict[str, str | None],
    *,
    round_index: int,
    canonical_answer: str | None = None,
    continuation_answer: str | None = None,
    preferred_answer: str | None = None,
    majority_answer: str | None = None,
    accepted_answer_changed: bool = False,
) -> tuple[str | None, dict]:
    policy_report = {
        "policy": "supported_answer_memory",
        "applied": False,
        "current_answer": chosen_answer,
        "current_support": _answer_support_count(chosen_answer, revised_answers),
        "memory_before": copy.deepcopy(state_graph.get("supported_answer_memory")),
        "memory_after": None,
        "reason": "",
    }
    if not chosen_answer or not _answer_matches_requested_object(question, chosen_answer):
        policy_report["reason"] = "no valid current answer"
        policy_report["memory_after"] = copy.deepcopy(state_graph.get("supported_answer_memory"))
        return chosen_answer, policy_report

    current_source = _memory_answer_source(
        chosen_answer,
        canonical_answer=canonical_answer,
        continuation_answer=continuation_answer,
        preferred_answer=preferred_answer,
        majority_answer=majority_answer,
        accepted_answer_changed=accepted_answer_changed,
    )
    current_support = policy_report["current_support"]
    current_is_supported_process = bool(current_source) and current_support >= 1
    current_is_stable = current_support >= 2
    memory = state_graph.get("supported_answer_memory") or {}
    memory_answer = memory.get("answer")
    memory_support = int(memory.get("support") or 0)
    memory_source = str(memory.get("source") or "")

    if (
        memory_answer
        and not _answers_equivalent_surface(chosen_answer, memory_answer)
        and memory_support >= 2
        and not current_is_supported_process
        and current_support < memory_support
        and _answer_matches_requested_object(question, memory_answer)
    ):
        policy_report["applied"] = True
        policy_report["reason"] = "preserved earlier supported graph answer over weak late drift"
        policy_report["memory_after"] = copy.deepcopy(memory)
        return memory_answer, policy_report

    if (
        memory_answer
        and not _answers_equivalent_surface(chosen_answer, memory_answer)
        and memory_support >= 2
        and current_support >= memory_support
        and current_source in {"accepted_rewrite_majority", "accepted_continuation", "resolution_preference"}
        and not memory_source.startswith("accepted_rewrite_majority")
        and _answer_matches_requested_object(question, memory_answer)
    ):
        policy_report["applied"] = True
        policy_report["reason"] = "preserved earlier supported graph answer over late rewrite consensus"
        policy_report["memory_after"] = copy.deepcopy(memory)
        return memory_answer, policy_report

    if current_is_supported_process and current_is_stable:
        previous_support = int(memory.get("support") or 0)
        previous_round = int(memory.get("round_index") or 0)
        previous_source = str(memory.get("source") or "")
        source_priority = {
            "canonical_resolution": 4,
            "accepted_continuation": 3,
            "resolution_preference": 2,
            "accepted_rewrite_majority": 1,
        }
        previous_priority = 3 if previous_source.startswith("stable_majority_guard") else source_priority.get(previous_source, 0)
        current_priority = source_priority.get(current_source, 0)
        protected_stable_memory = (
            memory_answer
            and previous_source.startswith("stable_majority_guard")
            and not _answers_equivalent_surface(chosen_answer, memory_answer)
            and memory_support >= 2
            and current_source != "canonical_resolution"
            and current_support <= memory_support
        )
        if protected_stable_memory:
            policy_report["applied"] = True
            policy_report["reason"] = "preserved guarded stable-majority memory over equal-or-weaker late rewrite"
            policy_report["memory_after"] = copy.deepcopy(memory)
            return memory_answer, policy_report
        should_update = (
            not memory_answer
            or _answers_equivalent_surface(chosen_answer, memory_answer)
            or current_priority > previous_priority
            or (current_priority == previous_priority and current_support >= previous_support and round_index >= previous_round)
        )
        if should_update:
            state_graph["supported_answer_memory"] = _memory_answer_record(
                chosen_answer,
                revised_answers,
                round_index=round_index,
                source=current_source,
            )
            policy_report["reason"] = "updated supported graph answer memory"
        else:
            policy_report["reason"] = "kept stronger earlier graph answer memory"
    else:
        policy_report["reason"] = "current answer not eligible for graph answer memory"
    policy_report["memory_after"] = copy.deepcopy(state_graph.get("supported_answer_memory"))
    return chosen_answer, policy_report


def _apply_stable_majority_guard_policy(
    state_graph: dict,
    question: str,
    chosen_answer: str | None,
    revised_answers: dict[str, str | None],
    *,
    round_index: int,
    input_answers: dict[str, str | None] | None = None,
    canonical_answer: str | None = None,
    continuation_answer: str | None = None,
    preferred_answer: str | None = None,
    majority_answer: str | None = None,
    accepted_answer_changed: bool = False,
    resolutions: list[ResolutionDecision] | None = None,
) -> tuple[str | None, dict]:
    chosen_answer, policy_report = _apply_supported_answer_memory_policy(
        state_graph,
        question,
        chosen_answer,
        revised_answers,
        round_index=round_index,
        canonical_answer=canonical_answer,
        continuation_answer=continuation_answer,
        preferred_answer=preferred_answer,
        majority_answer=majority_answer,
        accepted_answer_changed=accepted_answer_changed,
    )
    policy_report["policy"] = "stable_majority_guard"
    guard_report = {
        "checked": True,
        "applied": False,
        "stable_answer": None,
        "stable_support": 0,
        "chosen_support": _answer_support_count(chosen_answer, revised_answers),
        "reason": "",
    }
    policy_report["stable_majority_guard"] = guard_report
    memory_after_policy = policy_report.get("memory_after") or {}
    if (
        policy_report.get("applied")
        and memory_after_policy.get("answer")
        and _answers_equivalent_surface(chosen_answer, memory_after_policy.get("answer"))
    ):
        guard_report["checked"] = False
        guard_report["reason"] = "supported answer memory already selected the answer"
        return chosen_answer, policy_report

    stable_answer, stable_support = _stable_majority_answer(revised_answers, min_support=2)
    stable_source = "revised_answers"
    input_stable_answer, input_stable_support = _stable_majority_answer(input_answers or {}, min_support=2)
    memory = state_graph.get("supported_answer_memory") or {}
    memory_answer = memory.get("answer")
    memory_support = int(memory.get("support") or 0)
    memory_source = str(memory.get("source") or "")
    if (
        memory_answer
        and memory_support >= stable_support
        and memory_support >= 2
        and _answer_matches_requested_object(question, memory_answer)
        and not _answers_equivalent_surface(stable_answer, memory_answer)
        and not _resolution_names_bridge_error_for_answer(resolutions or [], memory_answer, question=question)
        and memory_source not in {"accepted_rewrite_majority", ""}
    ):
        stable_answer, stable_support = memory_answer, memory_support
        stable_source = "supported_answer_memory"
    if input_stable_answer and input_stable_support > stable_support:
        stable_answer, stable_support = input_stable_answer, input_stable_support
        stable_source = "input_answers"
    if (
        stable_answer
        and input_stable_answer
        and input_stable_support >= stable_support
        and not _answers_equivalent_surface(stable_answer, input_stable_answer)
    ):
        stable_answer, stable_support = input_stable_answer, input_stable_support
        stable_source = "input_answers"

    guard_report["stable_answer"] = stable_answer
    guard_report["stable_support"] = stable_support
    guard_report["stable_source"] = stable_source
    if not stable_answer or stable_support < 2:
        guard_report["reason"] = "no stable majority to protect"
        return chosen_answer, policy_report
    if not _answer_matches_requested_object(question, stable_answer):
        guard_report["reason"] = "stable majority answer does not match requested object"
        return chosen_answer, policy_report
    if not chosen_answer or _answers_equivalent_surface(chosen_answer, stable_answer):
        guard_report["reason"] = "chosen answer already agrees with stable majority"
        return chosen_answer, policy_report

    chosen_support = _answer_support_count(chosen_answer, revised_answers)
    canonical_support = _answer_support_count(canonical_answer, revised_answers)
    canonical_is_strong = (
        canonical_answer
        and _answers_equivalent_surface(chosen_answer, canonical_answer)
        and canonical_support >= stable_support
    )
    source = _memory_answer_source(
        chosen_answer,
        canonical_answer=canonical_answer,
        continuation_answer=continuation_answer,
        preferred_answer=preferred_answer,
        majority_answer=majority_answer,
        accepted_answer_changed=accepted_answer_changed,
    )
    bridge_error_named = _resolution_names_bridge_error_for_answer(resolutions or [], stable_answer, question=question)
    guard_report["bridge_error_named"] = bridge_error_named
    weak_overturn = chosen_support < stable_support or source in {"accepted_continuation", "resolution_preference"}
    if canonical_is_strong:
        guard_report["reason"] = "strong canonical answer has support comparable to stable majority"
        return chosen_answer, policy_report
    if not bridge_error_named:
        guard_report["applied"] = True
        guard_report["chosen_support"] = chosen_support
        guard_report["chosen_source"] = source
        guard_report["reason"] = "preserved stable visible majority because no explicit bridge error was named"
        policy_report["applied"] = True
        policy_report["reason"] = guard_report["reason"]
        state_graph["supported_answer_memory"] = _memory_answer_record(
            stable_answer,
            revised_answers if stable_source == "revised_answers" else (input_answers or revised_answers),
            round_index=round_index,
            source=f"stable_majority_guard:{stable_source}:no_bridge_error",
        )
        policy_report["memory_after"] = copy.deepcopy(state_graph.get("supported_answer_memory"))
        return stable_answer, policy_report
    if weak_overturn:
        guard_report["applied"] = True
        guard_report["chosen_support"] = chosen_support
        guard_report["chosen_source"] = source
        guard_report["reason"] = "preserved stable visible majority over weak graph overturn"
        policy_report["applied"] = True
        policy_report["reason"] = guard_report["reason"]
        state_graph["supported_answer_memory"] = _memory_answer_record(
            stable_answer,
            revised_answers if stable_source == "revised_answers" else (input_answers or revised_answers),
            round_index=round_index,
            source=f"stable_majority_guard:{stable_source}",
        )
        policy_report["memory_after"] = copy.deepcopy(state_graph.get("supported_answer_memory"))
        return stable_answer, policy_report

    guard_report["reason"] = "chosen answer has enough visible support to pass guard"
    return chosen_answer, policy_report


def _agents_for_paths(graph, path_ids: list[str]) -> list[str]:
    path_map = graph.path_map() if graph is not None else {}
    agents: list[str] = []
    seen = set()
    for path_id in _normalize_path_or_agent_tokens(path_ids):
        path = path_map.get(path_id)
        if path is None:
            path_index = re.fullmatch(r"P(\d+)", path_id)
            if path_index and graph is not None:
                index = int(path_index.group(1)) - 1
                paths = list(getattr(graph, "method_paths", []) or [])
                if 0 <= index < len(paths):
                    path = paths[index]
                else:
                    agent_id = f"A{path_index.group(1)}"
                    if agent_id not in seen:
                        agents.append(agent_id)
                        seen.add(agent_id)
                    continue
            if path is None:
                if re.fullmatch(r"A\d+", path_id):
                    if path_id not in seen:
                        agents.append(path_id)
                        seen.add(path_id)
                continue
        for agent_id in path.agent_ids or []:
            if agent_id not in seen:
                agents.append(agent_id)
                seen.add(agent_id)
    return agents


def _normalize_path_or_agent_tokens(path_ids: list[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in path_ids or []:
        for token in re.findall(r"\b(?:P|A)\d+\b", str(raw or "")):
            if token not in seen:
                tokens.append(token)
                seen.add(token)
    return tokens


def _unique_equivalent_answer(answers: list[str | None]) -> str | None:
    groups: list[list[str]] = []
    for answer in answers:
        if not answer:
            continue
        for group in groups:
            if _answers_equivalent_surface(answer, group[0]):
                group.append(answer)
                break
        else:
            groups.append([answer])
    if len(groups) != 1:
        return None
    return groups[0][0]


def _resolution_claim_answer(decision: ResolutionDecision | dict, *, question: str = "") -> str | None:
    if _weak_resolution_action(decision):
        return None
    if _is_synthesize_resolution(decision):
        return None
    canonical = decision.get("canonical_answer", "") if isinstance(decision, dict) else getattr(decision, "canonical_answer", "")
    canonical = str(canonical or "").strip()
    if canonical:
        if question and not _answer_matches_requested_object(question, canonical):
            return None
        if question_requests_short_answer(question):
            return canonical
        if question_uses_label_answers(question):
            return extract_label_answer(canonical, question) or canonical
        if extract_option_map(question):
            landed = _landing_answer(canonical, question=question)
            if landed:
                return landed
        if question_requests_option_text_answer(question):
            return canonical_mcq_answer(canonical, question) or canonical
        if _is_synthesize_resolution(decision) and _is_verified_synthesize_resolution(decision):
            normalized = normalize_math_answer(canonical)
            return normalized or canonical
    resolved_claim = decision.get("resolved_claim", "") if isinstance(decision, dict) else getattr(decision, "resolved_claim", "")
    short_answer = _short_answer_from_slot_claim(str(resolved_claim or ""), question=question)
    if short_answer:
        return short_answer
    if question_uses_label_answers(question):
        rationale = decision.get("rationale", "") if isinstance(decision, dict) else getattr(decision, "rationale", "")
        label_answer = truth_status_label_supported_by_claim(resolved_claim, question)
        if label_answer is None:
            label_answer = nli_label_supported_by_resolution(resolved_claim, rationale, question)
        if label_answer is None:
            # The corrected claim is the continuation anchor. Rationales often
            # mention the rejected side and can contain stale yes/no tokens, so
            # they must not set the rewrite target for label tasks.
            label_answer = extract_label_answer(str(resolved_claim or ""), question)
        if label_answer and not label_claim_is_bare_answer(str(resolved_claim or ""), question):
            return label_answer
    answer = _canonical_answer_from_resolved_claim(str(resolved_claim or ""))
    if answer:
        if question and not _claim_tracks_requested_object(question, str(resolved_claim or ""), answer):
            return None
        normalized = normalize_math_answer(answer)
        return normalized or answer
    return None


def _resolution_explicit_final_answer(decision: ResolutionDecision | dict, *, question: str = "") -> str | None:
    if _weak_resolution_action(decision):
        return None
    if _is_synthesize_resolution(decision):
        return None
    canonical = decision.get("canonical_answer", "") if isinstance(decision, dict) else getattr(decision, "canonical_answer", "")
    canonical = str(canonical or "").strip()
    if canonical:
        if question and not _answer_matches_requested_object(question, canonical):
            return None
        if question_requests_short_answer(question) and not _question_requests_math_like_answer(question):
            return canonical
        if question_uses_label_answers(question):
            return extract_label_answer(canonical, question) or canonical
        if extract_option_map(question):
            landed = _landing_answer(canonical, question=question)
            if landed:
                return landed
        if question_requests_option_text_answer(question):
            return canonical_mcq_answer(canonical, question) or canonical
        if _is_synthesize_resolution(decision) and _is_verified_synthesize_resolution(decision):
            normalized = normalize_math_answer(canonical)
            return normalized or canonical
    resolved_claim = decision.get("resolved_claim", "") if isinstance(decision, dict) else getattr(decision, "resolved_claim", "")
    resolved_claim = str(resolved_claim or "")
    if question_uses_label_answers(question) and label_claim_is_bare_answer(resolved_claim, question):
        return None
    lowered = resolved_claim.lower()
    explicit_final_markers = (
        "final answer",
        "requested answer",
        "answer is",
        "answer:",
        "image line",
        "image of the line",
        "resulting in the image",
    )
    requested_object_markers = (
        "least possible",
        "greatest possible",
        "smallest possible",
        "largest possible",
        "minimum sum",
        "minimal sum",
        "maximum sum",
        "range of",
        "domain of",
        "range is",
        "domain is",
        "requested set",
        "requested interval",
        "requested range",
    )
    if not any(
        marker in lowered
        for marker in explicit_final_markers + requested_object_markers
    ):
        return None
    answer = _canonical_answer_from_resolved_claim(resolved_claim)
    if answer:
        if not any(marker in lowered for marker in explicit_final_markers):
            if not question:
                return None
            if not _claim_tracks_requested_object(question, resolved_claim, answer):
                return None
        if question and not _answer_matches_requested_object(question, answer):
            return None
        if re.match(r"\s*y\s*=", str(answer), flags=re.IGNORECASE):
            return str(answer).replace(" ", "")
        normalized = normalize_math_answer(answer)
        return normalized or answer
    return None


def _resolution_completed_requested_object_answer(
    decision: ResolutionDecision | dict,
    *,
    question: str = "",
) -> str | None:
    explicit = _resolution_explicit_final_answer(decision, question=question)
    if explicit:
        return explicit
    resolved_claim = decision.get("resolved_claim", "") if isinstance(decision, dict) else getattr(decision, "resolved_claim", "")
    claim = str(resolved_claim or "")
    lowered = claim.lower()
    if not any(
        marker in lowered
        for marker in (
            "final answer",
            "requested answer",
            "requested final",
            "requested object",
            "requested sum",
            "requested count",
            "requested value",
            "requested probability",
            "requested expression",
            "requested range",
            "requested interval",
            "requested set",
            "answer is",
        )
    ):
        return None
    answer = _canonical_answer_from_resolved_claim(claim)
    if not answer:
        return None
    if question and not _claim_tracks_requested_object(question, claim, answer):
        return None
    if question and not _answer_matches_requested_object(question, answer):
        return None
    normalized = normalize_math_answer(answer)
    return normalized or answer


def _path_answer_support(graph, path_ids: list[str], answers: dict[str, str | None], target_answer: str | None) -> int:
    if not target_answer:
        return 0
    return sum(
        1
        for agent_id in _agents_for_paths(graph, path_ids)
        if _answers_equivalent_surface(answers.get(agent_id), target_answer)
    )


def _pairwise_agent_ids(decision: ResolutionDecision) -> tuple[str, str] | None:
    match = re.match(r"PAIR_([^_]+)_([^_]+)$", str(decision.divergence_id or "").strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _decision_side_answer_support(
    decision: ResolutionDecision,
    graph,
    current_answers: dict[str, str | None],
    target_answer: str | None,
    side: str,
) -> int:
    if not target_answer:
        return 0
    pair_agents = _pairwise_agent_ids(decision)
    if pair_agents is not None:
        agent_id = pair_agents[0] if side == "left" else pair_agents[1]
        return int(_answers_equivalent_surface(current_answers.get(agent_id), target_answer))
    path_ids = decision.keep_paths if side == "keep" else decision.drop_paths
    return _path_answer_support(graph, path_ids, current_answers, target_answer)


def _maybe_swap_resolution_paths_by_evidence(
    decision: ResolutionDecision,
    graph,
    current_answers: dict[str, str | None],
    *,
    question: str = "",
) -> ResolutionDecision:
    if question and question_uses_label_answers(question):
        return decision
    if decision.action not in {"choose_left", "choose_right"}:
        return decision
    target_answer = _resolution_claim_answer(decision, question=question)
    if not target_answer:
        return decision
    pair_agents = _pairwise_agent_ids(decision)
    if pair_agents is not None:
        left_support = _decision_side_answer_support(decision, graph, current_answers, target_answer, "left")
        right_support = _decision_side_answer_support(decision, graph, current_answers, target_answer, "right")
        if decision.action == "choose_left":
            winner_support, loser_support = left_support, right_support
        else:
            winner_support, loser_support = right_support, left_support
        if winner_support or not loser_support:
            return decision
        repaired = copy.deepcopy(decision)
        repaired.action = "choose_right" if decision.action == "choose_left" else "choose_left"
        repaired.winning_side = "right" if repaired.action == "choose_right" else "left"
        repaired.rationale = (
            f"Evidence gate corrected the pairwise side assignment because the repaired claim's answer {target_answer} "
            f"is supported by the opposite trace, not by the chosen trace. {decision.rationale}"
        ).strip()
        return repaired

    keep_support = _path_answer_support(graph, decision.keep_paths, current_answers, target_answer)
    drop_support = _path_answer_support(graph, decision.drop_paths, current_answers, target_answer)
    if keep_support or not drop_support:
        return decision
    repaired = copy.deepcopy(decision)
    repaired.keep_paths, repaired.drop_paths = list(decision.drop_paths), list(decision.keep_paths)
    if repaired.action == "choose_left":
        repaired.action = "choose_right"
        repaired.winning_side = "right"
    elif repaired.action == "choose_right":
        repaired.action = "choose_left"
        repaired.winning_side = "left"
    repaired.rationale = (
        f"Evidence gate corrected the path assignment because the repaired claim's answer {target_answer} "
        f"is supported by the dropped path, not by the kept path. {decision.rationale}"
    ).strip()
    return repaired


def _claim_answer_conflicts_with_deterministic_target(
    pipeline: NaturalLanguageGraphDebatePipeline,
    question: str,
    decision: ResolutionDecision,
) -> bool:
    return False


def _parse_integer_component(text: str) -> int | None:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("$", "").replace(" ", "")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    return None


def _extract_pmatrix_integer_vectors(text: str) -> list[tuple[int, int]]:
    vectors: list[tuple[int, int]] = []
    for body in re.findall(r"\\begin\{pmatrix\}(.+?)\\end\{pmatrix\}", str(text or ""), flags=re.DOTALL):
        parts = [part.strip() for part in re.split(r"\\\\+", body) if part.strip()]
        if len(parts) != 2:
            continue
        first = _parse_integer_component(parts[0])
        second = _parse_integer_component(parts[1])
        if first is not None and second is not None:
            vectors.append((first, second))
    return vectors


def _expected_matrix_mapping_equations(question: str) -> list[tuple[dict[str, int], int]]:
    normalized = re.sub(r"\s+", " ", str(question or "").lower())
    if "matrix" not in normalized or "takes" not in normalized:
        return []
    vectors = _extract_pmatrix_integer_vectors(question)
    if len(vectors) < 4:
        return []
    equations: list[tuple[dict[str, int], int]] = []
    for source, target in zip(vectors[0::2], vectors[1::2]):
        x, y = source
        out_x, out_y = target
        equations.append(({"a": x, "b": y}, out_x))
        equations.append(({"c": x, "d": y}, out_y))
    return equations


def _parse_linear_abcd_equation(text: str) -> tuple[dict[str, int], int] | None:
    expr = str(text or "")
    expr = expr.replace("$", "").replace(" ", "")
    expr = expr.replace("\\left", "").replace("\\right", "")
    if "=" not in expr:
        return None
    left, right = expr.split("=", 1)
    right_match = re.match(r"(-?\d+)(?:[^\d].*)?$", right)
    if not right_match:
        return None
    rhs = int(right_match.group(1))
    coeffs: dict[str, int] = {}
    for raw_term in re.findall(r"[+-]?[^+-]+", left):
        term = raw_term.strip()
        match = re.fullmatch(r"([+-]?)(\d*)([abcd])", term)
        if not match:
            return None
        sign = -1 if match.group(1) == "-" else 1
        magnitude = int(match.group(2)) if match.group(2) else 1
        var = match.group(3)
        coeffs[var] = coeffs.get(var, 0) + sign * magnitude
    coeffs = {var: coeff for var, coeff in coeffs.items() if coeff}
    return (coeffs, rhs) if coeffs else None


def _matrix_mapping_claim_conflicts_with_question(question: str, claim: str) -> bool:
    expected = _expected_matrix_mapping_equations(question)
    if not expected:
        return False
    snippets = re.findall(r"\$([^$=]+=[^$]+)\$", str(claim or ""))
    snippets.extend(re.findall(r"(?<![A-Za-z])[-+]?\d*[abcd](?:\s*[+-]\s*\d*[abcd])*\s*=\s*-?\d+", str(claim or "")))
    for snippet in snippets:
        parsed = _parse_linear_abcd_equation(snippet)
        if parsed is None:
            continue
        coeffs, rhs = parsed
        vars_key = set(coeffs)
        same_slot = [
            expected_coeffs
            for expected_coeffs, expected_rhs in expected
            if set(expected_coeffs) == vars_key and expected_rhs == rhs
        ]
        if same_slot and all(coeffs != expected_coeffs for expected_coeffs in same_slot):
            return True
    return False


def _path_best_target_alignment_rank(
    pipeline: NaturalLanguageGraphDebatePipeline,
    question: str,
    graph,
    path_ids: list[str],
    traces: dict[str, AgentTrace] | None,
) -> int | None:
    if not traces:
        return None
    ranks = []
    for agent_id in _agents_for_paths(graph, path_ids):
        trace = traces.get(agent_id)
        if trace is not None:
            ranks.append(pipeline._trace_target_alignment_rank(question, trace))
    return min(ranks) if ranks else None


def _resolution_claim_fails_local_evidence_gate(
    decision: ResolutionDecision,
    graph,
    current_answers: dict[str, str | None],
    *,
    question: str = "",
    traces: dict[str, AgentTrace] | None = None,
    pipeline: NaturalLanguageGraphDebatePipeline | None = None,
) -> bool:
    if decision.action not in {"choose_left", "choose_right"}:
        return False
    pipeline = pipeline or NaturalLanguageGraphDebatePipeline()
    if _claim_answer_conflicts_with_deterministic_target(pipeline, question, decision):
        return True
    if _matrix_mapping_claim_conflicts_with_question(question, decision.resolved_claim):
        return True
    if question and _question_requests_math_like_answer(question):
        stable_majority, stable_support = _stable_majority_answer(current_answers)
        if stable_majority and stable_support >= 2:
            winning_answers = _decision_winning_observed_answers(decision, graph, current_answers)
            winner_has_majority = any(
                _answers_equivalent_surface(answer, stable_majority)
                for answer in winning_answers
                if answer
            )
            completed_answer = _resolution_completed_requested_object_answer(decision, question=question)
            completed_supports_winner = bool(
                completed_answer
                and any(
                    _answers_equivalent_surface(completed_answer, answer)
                    for answer in winning_answers
                    if answer
                )
            )
            if (
                winning_answers
                and not winner_has_majority
                and not completed_supports_winner
            ):
                return True

    pair_agents = _pairwise_agent_ids(decision)
    if pair_agents is not None:
        if not traces:
            return False
        left_trace = traces.get(pair_agents[0])
        right_trace = traces.get(pair_agents[1])
        if left_trace is None or right_trace is None:
            return False
        left_rank = pipeline._trace_target_alignment_rank(question, left_trace)
        right_rank = pipeline._trace_target_alignment_rank(question, right_trace)
        winner_rank = left_rank if decision.action == "choose_left" else right_rank
        loser_rank = right_rank if decision.action == "choose_left" else left_rank
        if loser_rank + 2 <= winner_rank:
            return True
        target_answer = _resolution_claim_answer(decision, question=question)
        if not target_answer:
            return False
        winner_support = (
            _decision_side_answer_support(decision, graph, current_answers, target_answer, "left")
            if decision.action == "choose_left"
            else _decision_side_answer_support(decision, graph, current_answers, target_answer, "right")
        )
        loser_support = (
            _decision_side_answer_support(decision, graph, current_answers, target_answer, "right")
            if decision.action == "choose_left"
            else _decision_side_answer_support(decision, graph, current_answers, target_answer, "left")
        )
        if loser_rank < winner_rank and winner_support and not loser_support:
            return True
        return False

    keep_rank = _path_best_target_alignment_rank(pipeline, question, graph, decision.keep_paths, traces)
    drop_rank = _path_best_target_alignment_rank(pipeline, question, graph, decision.drop_paths, traces)
    if keep_rank is None or drop_rank is None:
        return False
    if drop_rank + 2 <= keep_rank:
        return True

    target_answer = _resolution_claim_answer(decision, question=question)
    if not target_answer:
        return False
    keep_support = _path_answer_support(graph, decision.keep_paths, current_answers, target_answer)
    drop_support = _path_answer_support(graph, decision.drop_paths, current_answers, target_answer)
    if drop_rank < keep_rank and keep_support and not drop_support:
        return True
    return False


def _evidence_gate_resolutions(
    resolutions: list[ResolutionDecision],
    graph,
    current_answers: dict[str, str | None],
    *,
    question: str = "",
    traces: dict[str, AgentTrace] | None = None,
    pipeline: NaturalLanguageGraphDebatePipeline | None = None,
) -> list[ResolutionDecision]:
    pipeline = pipeline or NaturalLanguageGraphDebatePipeline()
    gated: list[ResolutionDecision] = []
    for decision in resolutions:
        if _resolution_claim_fails_local_evidence_gate(
            decision,
            graph,
            current_answers,
            question=question,
            traces=traces,
            pipeline=pipeline,
        ):
            continue
        gated.append(_maybe_swap_resolution_paths_by_evidence(decision, graph, current_answers, question=question))
    return gated


def _resolution_landing_target(
    decision: ResolutionDecision,
    graph,
    current_answers: dict[str, str | None],
    *,
    question: str = "",
) -> str | None:
    return _resolution_explicit_final_answer(decision, question=question)


def _resolution_rewrite_agents(decision: ResolutionDecision, graph) -> set[str]:
    if str(decision.divergence_id or "").startswith("PAIR_"):
        match = re.match(r"PAIR_([^_]+)_([^_]+)$", str(decision.divergence_id or "").strip())
        if not match:
            return set()
        left_agent, right_agent = match.group(1), match.group(2)
        if decision.action == "choose_left":
            return {right_agent}
        if decision.action == "choose_right":
            return {left_agent}
        return set()
    if decision.action in {"choose_left", "choose_right"}:
        return set(_agents_for_paths(graph, decision.drop_paths))
    return set()


def _label_resolution_rewrite_agents(
    decision: ResolutionDecision,
    graph,
    current_answers: dict[str, str | None],
    target: str | None,
) -> set[str]:
    agents = set(_resolution_rewrite_agents(decision, graph))
    if not target:
        return agents
    for agent_id in _agents_for_paths(graph, decision.keep_paths):
        current = current_answers.get(agent_id)
        if not current or not _answers_equivalent_surface(current, target):
            agents.add(agent_id)
    return agents


def _expected_rewrite_targets_by_agent(
    resolutions: list[ResolutionDecision],
    graph,
    current_answers: dict[str, str | None],
    *,
    question: str = "",
) -> dict[str, list[str]]:
    targets_by_agent: dict[str, list[str]] = defaultdict(list)
    for decision in resolutions:
        target = _resolution_landing_target(decision, graph, current_answers, question=question)
        if not target and question_uses_label_answers(question):
            # For label tasks, a resolved predicate-level claim often already
            # states the supported label without using final-answer wording.
            # Treat that as a rewrite consistency target, not as a backend
            # answer override.
            target = _resolution_claim_answer(decision, question=question)
        if not target:
            continue
        rewrite_agents = (
            _label_resolution_rewrite_agents(decision, graph, current_answers, target)
            if question_uses_label_answers(question)
            else _resolution_rewrite_agents(decision, graph)
        )
        for agent_id in rewrite_agents:
            targets_by_agent[agent_id].append(target)
    return targets_by_agent


def _expected_method_rewrite_agents(
    resolutions: list[ResolutionDecision],
    graph,
    *,
    question: str = "",
) -> set[str]:
    agents: set[str] = set()
    for decision in resolutions:
        if _resolution_explicit_final_answer(decision, question=question):
            continue
        if question_uses_label_answers(question) and _resolution_claim_answer(decision, question=question):
            continue
        agents.update(_resolution_rewrite_agents(decision, graph))
    return agents


def _decision_winning_observed_answers(
    decision: ResolutionDecision,
    graph,
    current_answers: dict[str, str | None],
) -> list[str]:
    if decision.action not in {"choose_left", "choose_right"}:
        return []
    if str(decision.divergence_id or "").startswith("PAIR_"):
        pair_agents = _pairwise_agent_ids(decision)
        if pair_agents is None:
            return []
        winner = pair_agents[0] if decision.action == "choose_left" else pair_agents[1]
        answer = current_answers.get(winner)
        return [answer] if answer else []
    agents = _agents_for_paths(graph, decision.keep_paths)
    return [current_answers[agent_id] for agent_id in agents if current_answers.get(agent_id)]


def _expected_method_winning_answers_by_agent(
    resolutions: list[ResolutionDecision],
    graph,
    current_answers: dict[str, str | None],
    *,
    question: str = "",
) -> dict[str, list[str]]:
    answers_by_agent: dict[str, list[str]] = defaultdict(list)
    for decision in resolutions:
        if _resolution_explicit_final_answer(decision, question=question):
            continue
        if question_uses_label_answers(question) and _resolution_claim_answer(decision, question=question):
            continue
        winning_answers = _decision_winning_observed_answers(decision, graph, current_answers)
        if not winning_answers:
            continue
        for agent_id in _resolution_rewrite_agents(decision, graph):
            answers_by_agent[agent_id].extend(winning_answers)
    return answers_by_agent


def _closed_answer_candidate_task(question: str) -> bool:
    return question_uses_label_answers(question) or question_requests_option_text_answer(question)


def _method_rewrite_must_preserve_observed_winner(question: str, method_targets: list[str]) -> bool:
    return _method_rewrite_must_preserve_observed_winner_with_context(question, method_targets)


def _method_rewrite_must_preserve_observed_winner_with_context(
    question: str,
    method_targets: list[str],
    current_answers: dict[str, str | None] | None = None,
) -> bool:
    if not method_targets:
        return False
    if not str(question or "").strip():
        return False
    if _closed_answer_candidate_task(question):
        return True
    # For open-form computation, a method-level claim is not a license to
    # overturn a stable observed majority. Without such a majority, Math must
    # remain open to finishing a verified checkpoint into a new final object.
    if _question_requests_math_like_answer(question):
        stable_majority, stable_support = _stable_majority_answer(current_answers or {})
        return bool(
            stable_majority
            and stable_support >= 2
            and any(_answers_equivalent_surface(target, stable_majority) for target in method_targets)
        )
    return False


def _resolved_claim_surface_supports_answer(resolutions: list[ResolutionDecision], answer: str | None) -> bool:
    if not answer:
        return False
    answer_compact = re.sub(r"[^a-zA-Z0-9]+", "", str(answer or "").lower())
    if not answer_compact:
        return False
    for decision in resolutions:
        claim = str(getattr(decision, "resolved_claim", "") or "")
        if _answers_equivalent_surface(_canonical_answer_from_resolved_claim(claim), answer):
            return True
        claim_compact = re.sub(r"[^a-zA-Z0-9]+", "", claim.lower())
        if answer_compact and answer_compact in claim_compact:
            return True
    return False


def _filter_revisions_by_consistency(
    revised_traces: dict[str, AgentTrace],
    resolutions: list[ResolutionDecision],
    graph,
    current_answers: dict[str, str | None],
    observed_answers: list[str | None] | None = None,
    *,
    question: str = "",
) -> tuple[dict[str, AgentTrace], dict[str, dict], bool]:
    targets_by_agent = _expected_rewrite_targets_by_agent(resolutions, graph, current_answers, question=question)
    method_rewrite_agents = _expected_method_rewrite_agents(resolutions, graph, question=question)
    method_winning_answers_by_agent = _expected_method_winning_answers_by_agent(
        resolutions,
        graph,
        current_answers,
        question=question,
    )
    observed_answers = [answer for answer in current_answers.values() if answer] + list(observed_answers or [])
    if not targets_by_agent and not method_rewrite_agents:
        return revised_traces, {}, True
    accepted: dict[str, AgentTrace] = {}
    report: dict[str, dict] = {}
    allow_resolution_landing = True

    def has_observed_target_support(targets: list[str]) -> bool:
        for target in targets:
            if any(_answers_equivalent_surface(target, answer) for answer in observed_answers if answer):
                return True
            if any(_answers_equivalent_surface(target, answer) for answer in current_answers.values() if answer):
                return True
        return False

    def stable_majority_overturn_without_bridge_error(targets: list[str]) -> str | None:
        if not _closed_answer_candidate_task(question) or not targets:
            return None
        stable_answer, stable_support = _stable_majority_answer(current_answers, min_support=2)
        if not stable_answer or stable_support < 2:
            return None
        if any(_answers_equivalent_surface(target, stable_answer) for target in targets):
            return None
        if _resolution_names_bridge_error_for_answer(resolutions, stable_answer, question=question):
            return None
        return stable_answer

    for agent_id, trace in revised_traces.items():
        targets = targets_by_agent.get(agent_id, [])
        revised_answer = _extract_answer_from_trace_text(trace.normalized_trace_text, question)
        if _is_exactness_downgrade(current_answers.get(agent_id), revised_answer):
            report[agent_id] = {
                "accepted": False,
                "expected_answers": [current_answers.get(agent_id)],
                "revised_answer": revised_answer,
                "reason": "rewrite replaced an exact symbolic answer with a decimal approximation",
            }
            continue
        if revised_answer and not _answer_matches_requested_object(question, revised_answer):
            report[agent_id] = {
                "accepted": False,
                "expected_answers": targets or ["a completed final answer for the requested object"],
                "revised_answer": revised_answer,
                "reason": "rewrite answer does not match the requested object",
            }
            if targets and not has_observed_target_support(targets):
                allow_resolution_landing = False
            continue
        if not revised_answer and (targets or agent_id in method_rewrite_agents or current_answers.get(agent_id)):
            text_looks_wrong_object = _looks_process_or_placeholder_answer(trace.normalized_trace_text) or bool(
                re.search(
                    r"\b(?:therefore\s+the\s+result\s+is|thus\s+the\s+remaining\s+condition\s+is)\b",
                    str(trace.normalized_trace_text or ""),
                    flags=re.IGNORECASE,
                )
            )
            report[agent_id] = {
                "accepted": False,
                "expected_answers": targets or ["preserve a final answer for the original requested object"],
                "revised_answer": revised_answer,
                "reason": (
                    "rewrite answer does not match the requested object"
                    if targets or text_looks_wrong_object
                    else "rewrite dropped the previous final answer instead of completing the requested object"
                ),
            }
            if targets and not has_observed_target_support(targets):
                allow_resolution_landing = False
            continue
        if targets and not any(_answers_equivalent_surface(revised_answer, target) for target in targets):
            allow_resolution_landing = False
            report[agent_id] = {
                "accepted": False,
                "expected_answers": targets,
                "revised_answer": revised_answer,
                "reason": "rewrite answer did not agree with the winning local claim",
            }
            continue
        protected_stable_answer = stable_majority_overturn_without_bridge_error(targets)
        if (
            protected_stable_answer
            and revised_answer
            and any(_answers_equivalent_surface(revised_answer, target) for target in targets)
            and not _answers_equivalent_surface(revised_answer, protected_stable_answer)
        ):
            allow_resolution_landing = False
            report[agent_id] = {
                "accepted": False,
                "expected_answers": [protected_stable_answer],
                "revised_answer": revised_answer,
                "reason": "rewrite would overturn stable visible majority without explicit bridge error",
            }
            continue
        method_targets = method_winning_answers_by_agent.get(agent_id, [])
        protected_stable_answer = stable_majority_overturn_without_bridge_error(method_targets)
        if (
            not targets
            and protected_stable_answer
            and revised_answer
            and any(_answers_equivalent_surface(revised_answer, target) for target in method_targets)
            and not _answers_equivalent_surface(revised_answer, protected_stable_answer)
        ):
            report[agent_id] = {
                "accepted": False,
                "expected_answers": [protected_stable_answer],
                "revised_answer": revised_answer,
                "reason": "claim-continuation rewrite would overturn stable visible majority without explicit bridge error",
            }
            continue
        if (
            not targets
            and agent_id in method_rewrite_agents
            and method_targets
            and revised_answer
            and _answers_equivalent_surface(revised_answer, current_answers.get(agent_id))
            and not any(_answers_equivalent_surface(revised_answer, target) for target in method_targets)
        ):
            report[agent_id] = {
                "accepted": False,
                "expected_answers": method_targets,
                "revised_answer": revised_answer,
                "reason": "claim-continuation rewrite kept the same answer instead of adopting the winning path answer",
            }
            continue
        if (
            not targets
            and agent_id in method_rewrite_agents
            and method_targets
            and _method_rewrite_must_preserve_observed_winner_with_context(
                question,
                method_targets,
                current_answers,
            )
            and revised_answer
            and not any(_answers_equivalent_surface(revised_answer, target) for target in method_targets)
        ):
            report[agent_id] = {
                "accepted": False,
                "expected_answers": method_targets,
                "revised_answer": revised_answer,
                "reason": "claim-continuation rewrite changed to an unobserved final answer",
            }
            continue
        if (
            not targets
            and agent_id in method_rewrite_agents
            and method_targets
            and not str(question or "").strip()
            and revised_answer
            and not any(_answers_equivalent_surface(revised_answer, target) for target in method_targets)
            and not _resolved_claim_surface_supports_answer(resolutions, revised_answer)
        ):
            report[agent_id] = {
                "accepted": False,
                "expected_answers": method_targets,
                "revised_answer": revised_answer,
                "reason": "claim-continuation rewrite changed to an unobserved final answer",
            }
            continue
        accepted[agent_id] = trace
        if targets:
            report[agent_id] = {
                "accepted": True,
                "expected_answers": targets,
                "revised_answer": revised_answer,
            }
        elif agent_id in method_rewrite_agents:
            report[agent_id] = {
                "accepted": True,
                "expected_answers": ["a completed final answer for the requested object"],
                "revised_answer": revised_answer,
            }
    return accepted, report, allow_resolution_landing


def _accepted_revision_changes_answer(
    revised_traces: dict[str, AgentTrace],
    previous_answers: dict[str, str | None],
    *,
    question: str = "",
) -> bool:
    for agent_id, trace in revised_traces.items():
        revised_answer = _extract_answer_from_trace_text(trace.normalized_trace_text, question)
        if revised_answer and not _answers_equivalent_surface(revised_answer, previous_answers.get(agent_id)):
            return True
    return False


def _accepted_continuation_answer(
    revised_traces: dict[str, AgentTrace],
    previous_answers: dict[str, str | None],
    support_answers: dict[str, str | None] | None = None,
    *,
    question: str = "",
) -> str | None:
    accepted_answers: list[str | None] = []
    changed_answers: list[str | None] = []
    for agent_id, trace in revised_traces.items():
        revised_answer = _extract_answer_from_trace_text(trace.normalized_trace_text, question)
        if not revised_answer:
            continue
        accepted_answers.append(revised_answer)
        if not _answers_equivalent_surface(revised_answer, previous_answers.get(agent_id)):
            changed_answers.append(revised_answer)
    changed_answer = _unique_equivalent_answer(changed_answers)
    if changed_answer and support_answers is not None:
        support = sum(
            1
            for answer in support_answers.values()
            if _answers_equivalent_surface(changed_answer, answer)
        )
        if support >= 2:
            return changed_answer
        return None
    return changed_answer or _unique_equivalent_answer(accepted_answers)


def _has_rewrite_rejections(rewrite_consistency: dict[str, dict]) -> bool:
    return any(not item.get("accepted") for item in rewrite_consistency.values())


def _preferred_answer_from_resolutions(
    resolutions: list[ResolutionDecision],
    graph,
    revised_answers: dict[str, str | None],
    input_answers: dict[str, str | None] | None = None,
    *,
    question: str = "",
) -> str | None:
    for decision in resolutions:
        claim_answer = _resolution_claim_answer(decision, question=question)
        if claim_answer and _answer_has_observed_support(claim_answer, revised_answers, input_answers):
            return claim_answer
        if str(decision.divergence_id or "").startswith("PAIR_"):
            match = re.match(r"PAIR_([^_]+)_([^_]+)$", str(decision.divergence_id or ""))
            if not match:
                continue
            if not claim_answer:
                continue
            left_agent, right_agent = match.group(1), match.group(2)
            winner = left_agent if decision.action == "choose_left" else right_agent if decision.action == "choose_right" else ""
            if winner and revised_answers.get(winner):
                return revised_answers[winner]
            continue

        if decision.action not in {"choose_left", "choose_right"}:
            continue
        if not claim_answer:
            continue
        selected_agents = _agents_for_paths(graph, decision.keep_paths)
        if not selected_agents and decision.drop_paths:
            dropped = set(_agents_for_paths(graph, decision.drop_paths))
            selected_agents = [agent_id for agent_id in revised_answers if agent_id not in dropped]
        selected_answers = [revised_answers.get(agent_id) for agent_id in selected_agents if revised_answers.get(agent_id)]
        if not selected_answers:
            continue
        counts = Counter(selected_answers)
        best_count = max(counts.values())
        tied = {answer for answer, count in counts.items() if count == best_count}
        if len(selected_answers) == 1 or best_count > 1 or len(tied) == 1:
            for answer in selected_answers:
                if answer in tied:
                    return answer


def _trusted_model_answer_from_resolutions(
    resolutions: list[ResolutionDecision],
    graph,
    revised_answers: dict[str, str | None],
    input_answers: dict[str, str | None] | None = None,
    *,
    question: str = "",
) -> str | None:
    visible_answers = dict(input_answers or {})
    visible_answers.update({k: v for k, v in (revised_answers or {}).items() if v})
    for decision in resolutions or []:
        answer = _resolution_claim_answer(decision, question=question)
        if not answer:
            answer = _resolution_explicit_final_answer(decision, question=question)
        if not answer:
            answer = _resolution_completed_requested_object_answer(decision, question=question)
        if not answer:
            answer = _canonical_answer_from_resolved_claim(getattr(decision, "resolved_claim", ""))
        if answer:
            landed = _landing_answer(answer, question=question)
            if landed:
                return landed
            if not question or _answer_matches_requested_object(question, answer):
                return answer
        selected_agents = _agents_for_paths(graph, getattr(decision, "keep_paths", []) or [])
        if not selected_agents and getattr(decision, "drop_paths", None):
            dropped = set(_agents_for_paths(graph, getattr(decision, "drop_paths", []) or []))
            selected_agents = [agent_id for agent_id in visible_answers if agent_id not in dropped]
        selected_answers = [visible_answers.get(agent_id) for agent_id in selected_agents if visible_answers.get(agent_id)]
        selected_majority = _majority_answer({str(i): answer for i, answer in enumerate(selected_answers)})
        if selected_majority:
            return selected_majority
    return None


def _choose_landed_answer(question: str, *candidates: str | None) -> str | None:
    landed_candidates: list[str | None] = []
    for candidate in candidates:
        landed = _landing_answer(candidate, question=question)
        landed_candidates.append(landed)
    for index, landed in enumerate(landed_candidates):
        if landed:
            if (
                _question_requests_set_like_answer(question)
                and _looks_inequality_fragment(landed)
                and any(_looks_interval_or_set_surface(item or "") for item in landed_candidates[index + 1 :])
            ):
                continue
            return landed
    return None
    return None


def _canonical_final_line(answer: str) -> str:
    return f"{{final answer: \\boxed{{{answer}}}}}"


def _with_canonical_final_answer(text: str, answer: str) -> str:
    body = str(text or "").rstrip()
    final_line = _canonical_final_line(answer)
    if final_line in body:
        return body
    return f"{body}\n{final_line}".strip()


def _apply_canonical_answer_to_responses(responses: dict[str, str], answer: str) -> dict[str, str]:
    return {
        agent_id: _with_canonical_final_answer(text, answer)
        for agent_id, text in responses.items()
    }


def _canonical_answer_map(agent_ids, answer: str) -> dict[str, str]:
    return {agent_id: answer for agent_id in agent_ids}


def _remember_structured_resolutions(state: dict, resolutions: list[ResolutionDecision]) -> None:
    memory = state["graph"].setdefault("claim_invariants", [])
    seen = {
        (
            item.get("divergence_id", ""),
            item.get("resolved_claim", ""),
            item.get("canonical_answer", ""),
        )
        for item in memory
    }
    for decision in resolutions:
        if _is_pairwise_resolution(decision) or _is_synthesize_resolution(decision):
            continue
        payload = copy.deepcopy(decision.__dict__)
        key = (payload.get("divergence_id", ""), payload.get("resolved_claim", ""), payload.get("canonical_answer", ""))
        if key not in seen:
            memory.append(payload)
            seen.add(key)
    if any(not _is_pairwise_resolution(decision) and not _is_synthesize_resolution(decision) for decision in resolutions):
        state["graph"]["had_structured_resolution"] = True


def _memory_resolutions_for_state(state: dict) -> list[ResolutionDecision]:
    return [_resolution_from_dict(item) for item in state["graph"].get("claim_invariants", [])]


def _match_history_record(case: dict, history_records: list[dict]) -> dict:
    case_id = str(case.get("unique_id", "") or "").strip()
    if case_id:
        id_matches = []
        for record in history_records:
            record_case_id = str(
                record.get("unique_id")
                or record.get("task", {}).get("unique_id")
                or ""
            ).strip()
            if record_case_id == case_id:
                id_matches.append(record)
        if len(id_matches) == 1:
            return id_matches[0]
        if len(id_matches) > 1:
            raise ValueError(f"Multiple history records matched unique_id: {case_id}")

    problem = str(case.get("problem", "") or "").strip()
    matches = []
    for record in history_records:
        question = record.get("task", {}).get("question", "")
        if _strip_instruction_suffix(question) == problem:
            matches.append(record)
    if not matches:
        raise KeyError(f"No history record matched case: {case_id or problem[:120]}")
    if len(matches) > 1:
        raise ValueError(f"Multiple history records matched problem: {problem[:120]}")
    return matches[0]


def _trace_text_map(traces: dict) -> dict[str, str]:
    return {
        agent_id: trace.normalized_trace_text
        for agent_id, trace in traces.items()
        if getattr(trace, "normalized_trace_text", "").strip()
    }


def _build_case_state(case: dict, history_record: dict, unanimous_policy: str = "skip") -> dict:
    round0 = history_record["rounds"][0]
    raw_responses = round0["inputs"]["initial_responses"]
    initial_responses = {_agent_label(agent_id): text for agent_id, text in raw_responses.items()}
    initial_answers = {
        agent_id: _extract_answer_from_trace_text(text, case["problem"])
        for agent_id, text in initial_responses.items()
    }
    unanimous_policy = _unanimous_policy_name(unanimous_policy)
    initial_answers_unanimous = _all_answers_unanimous(initial_answers)
    graph_done = initial_answers_unanimous and unanimous_policy == "skip"
    return {
        "unique_id": case["unique_id"],
        "problem": case["problem"],
        "gold_answer": case["answer"],
        "question": case["problem"],
        "origin_question": _build_origin_question_prompt(history_record["task"]["question"]),
        "initial_responses": copy.deepcopy(initial_responses),
        "initial_answers": initial_answers,
        "origin": {
            "responses": copy.deepcopy(initial_responses),
            "round_results": [],
            "done": False,
        },
        "graph": {
            "responses": copy.deepcopy(initial_responses),
            "round_results": [],
            "done": graph_done,
            "unanimous_policy": unanimous_policy,
            "initial_answers_unanimous": initial_answers_unanimous,
            "claim_invariants": [],
            "canonical_answer": None,
            "had_structured_resolution": False,
            "supported_answer_memory": None,
        },
    }


def _prepare_batch_states(batch_cases: list[dict], history_records: list[dict], unanimous_policy: str = "skip") -> list[dict]:
    return [
        _build_case_state(case, _match_history_record(case, history_records), unanimous_policy=unanimous_policy)
        for case in batch_cases
    ]


def _finalize_case_result(state: dict, origin_rounds: int, graph_rounds: int) -> dict:
    initial_answers = copy.deepcopy(state["initial_answers"])
    origin_round_results = list(state["origin"]["round_results"])
    graph_round_results = list(state["graph"]["round_results"])

    origin_final_round = origin_round_results[-1] if origin_round_results else {
        "revised_answers": copy.deepcopy(initial_answers),
        "chosen_answer": _majority_answer(initial_answers),
        "responses": copy.deepcopy(state["initial_responses"]),
    }
    graph_final_round = graph_round_results[-1] if graph_round_results else {
        "revised_answers": copy.deepcopy(initial_answers),
        "chosen_answer": _majority_answer(initial_answers),
        "graph_dossier": "",
        "resolutions": [],
        "raw_resolution_notes": [],
        "revised_traces": {},
    }

    origin_chosen = origin_final_round.get("chosen_answer") or _majority_answer(origin_final_round.get("revised_answers", {}) or initial_answers)
    graph_chosen = graph_final_round.get("chosen_answer") or _majority_answer(graph_final_round.get("revised_answers", {}) or initial_answers)
    graph_final_revised_answers = graph_final_round.get("revised_answers", {}) or initial_answers
    inferred_graph_answer = None if _all_answers_unanimous(graph_final_revised_answers) else _final_inferred_answer(initial_answers, graph_round_results)
    if inferred_graph_answer:
        graph_chosen = inferred_graph_answer
    graph_chosen = _choose_landed_answer(
        state["problem"],
        graph_chosen,
        inferred_graph_answer,
        _majority_answer(graph_final_revised_answers),
    )
    target_focus_note = _target_focus_note_for_problem(state["problem"])
    provenance = target_note_provenance(target_focus_note)
    graph_selection = {
        "source": "base_graph",
        "target_focus_note": target_focus_note,
        "target_focus_provenance": provenance.kind,
        "target_focus_confidence": provenance.confidence,
        "target_focus_provenance_reason": provenance.reason,
        "answer_override": "disabled",
        "legacy_target_check_answer": None,
        "legacy_target_check_changed_answer": False,
        "legacy_target_check_semantic_changed_answer": False,
    }

    return {
        "unique_id": state["unique_id"],
        "problem": state["problem"],
        "gold_answer": state["gold_answer"],
        "initial_answers": initial_answers,
        "graph_selection": graph_selection,
        "origin": {
            "num_rounds": origin_rounds,
            "round_results": origin_round_results,
            "revised_answers": copy.deepcopy(origin_final_round.get("revised_answers", initial_answers)),
            "chosen_answer": origin_chosen,
            "is_correct": _is_answer_correct(origin_chosen, state["gold_answer"], state["problem"]),
        },
        "graph": {
            "num_rounds": graph_rounds,
            "round_results": graph_round_results,
            "revised_answers": copy.deepcopy(graph_final_round.get("revised_answers", initial_answers)),
            "chosen_answer": graph_chosen,
            "is_correct": _is_answer_correct(graph_chosen, state["gold_answer"], state["problem"]),
            "unanimous_policy": _unanimous_policy_name(state["graph"].get("unanimous_policy", "skip")),
            "initial_answers_unanimous": bool(state["graph"].get("initial_answers_unanimous", False)),
            "graph_dossier": graph_final_round.get("graph_dossier", ""),
            "resolutions": copy.deepcopy(graph_final_round.get("resolutions", [])),
            "raw_resolution_notes": copy.deepcopy(graph_final_round.get("raw_resolution_notes", [])),
            "api_trace": copy.deepcopy(graph_final_round.get("api_trace", [])),
            "revised_traces": copy.deepcopy(graph_final_round.get("revised_traces", {})),
            "claim_invariants": copy.deepcopy(state["graph"].get("claim_invariants", [])),
            "canonical_answer": state["graph"].get("canonical_answer"),
            "supported_answer_memory": copy.deepcopy(state["graph"].get("supported_answer_memory")),
        },
    }


def _run_origin_rounds_batched(batch_states: list[dict], runtime, rounds: int) -> None:
    args = _build_origin_args()
    for round_idx in range(1, rounds + 1):
        active_states = [state for state in batch_states if not state["origin"]["done"]]
        if not active_states:
            break

        update_prompt_records = []
        state_context = {}
        for state in active_states:
            responses = state["origin"]["responses"]
            prev_answers = _get_answer_map(args, responses, question=state["problem"])
            peer_map = _get_peer_map(args, list(responses.keys()), prev_answers)
            update_messages = _build_direct_response_update_messages(
                args,
                state["origin_question"],
                responses,
                peer_map,
                personas=None,
            )
            state_context[state["unique_id"]] = {
                "prev_answers": prev_answers,
                "peer_map": peer_map,
                "responses": copy.deepcopy(responses),
                "update_messages": update_messages,
            }
            for agent_id, message in update_messages.items():
                update_prompt_records.append((state["unique_id"], agent_id, message))

        updated_outputs = {}
        if update_prompt_records:
            raw_outputs = engine([message for _, _, message in update_prompt_records], runtime.agent, len(update_prompt_records))
            for (case_id, agent_id, _), output in zip(update_prompt_records, raw_outputs):
                updated_outputs.setdefault(case_id, {})[agent_id] = output

        for state in active_states:
            case_id = state["unique_id"]
            previous_responses = state_context[case_id]["responses"]
            updated_responses = dict(previous_responses)
            for agent_id in previous_responses.keys():
                raw_output = updated_outputs.get(case_id, {}).get(agent_id)
                if raw_output is None:
                    continue
                clean_response = _parse_direct_update_output(raw_output, previous_responses.get(agent_id, ""))
                updated_responses[agent_id] = clean_response

            revised_answers = {
                agent_id: _extract_answer_from_trace_text(text, state["problem"])
                for agent_id, text in updated_responses.items()
            }
            chosen_answer = _majority_answer(revised_answers)
            state["origin"]["round_results"].append(
                {
                    "round_index": round_idx,
                    "input_answers": {
                        agent_id: _extract_answer_from_trace_text(text, state["problem"])
                        for agent_id, text in previous_responses.items()
                    },
                    "revised_answers": revised_answers,
                    "chosen_answer": chosen_answer,
                    "peer_map": copy.deepcopy(state_context[case_id]["peer_map"]),
                    "critique_responses": {},
                    "revision_metadata": {},
                    "responses": copy.deepcopy(updated_responses),
                }
            )
            state["origin"]["responses"] = updated_responses
            if len({answer for answer in revised_answers.values() if answer}) <= 1:
                state["origin"]["done"] = True


def _normalize_traces_batched(
    batch_entries: list[tuple[dict, dict]],
    runtime,
    pipeline: NaturalLanguageGraphDebatePipeline,
) -> dict[str, dict[str, AgentTrace]]:
    prompt_records = []
    for state, mode_state in batch_entries:
        for agent_id, response in mode_state["responses"].items():
            prompt_records.append(
                (
                    state["unique_id"],
                    agent_id,
                    response,
                    build_atomic_trace_prompt(state["question"], agent_id, response),
                )
            )

    traces_by_case: dict[str, dict[str, AgentTrace]] = {}
    if not prompt_records:
        return traces_by_case

    outputs = run_prompts(
        runtime,
        [(f"{case_id}:{agent_id}:normalize", prompt) for case_id, agent_id, _, prompt in prompt_records],
        pipeline.system_prompt,
    )
    for (case_id, agent_id, original_response, _), output in zip(prompt_records, outputs):
        normalized = parse_atomic_trace(output)
        normalized = pipeline._restore_source_final_answer(normalized, original_response)
        raw_steps = build_step_objects(agent_id, normalized)
        compacted_steps = compact_step_objects(agent_id, raw_steps)
        traces_by_case.setdefault(case_id, {})[agent_id] = AgentTrace(
            agent_id=agent_id,
            original_response=original_response,
            normalized_trace_text=pipeline._serialize_trace_steps(compacted_steps),
            steps=compacted_steps,
        )
    return traces_by_case


def _merge_graphs_batched(batch_states: list[dict], traces_by_case: dict[str, dict[str, AgentTrace]], pipeline: NaturalLanguageGraphDebatePipeline, runtime) -> dict[str, object]:
    merge_states = {}
    for state in batch_states:
        case_id = state["unique_id"]
        traces = traces_by_case.get(case_id, {})
        max_steps = max((len(trace.steps) for trace in traces.values()), default=0)
        if max_steps == 0:
            merge_states[case_id] = {
                "graph": parse_graph_dossier(""),
                "dossier": "",
                "next_start": 0,
                "chunk_size": 1,
                "max_steps": 0,
                "traces": traces,
                "question": state["question"],
            }
            continue
        merge_states[case_id] = {
            "graph": parse_graph_dossier(""),
            "dossier": "",
            "next_start": 0,
            "chunk_size": pipeline._merge_chunk_size_for_traces(traces),
            "max_steps": max_steps,
            "traces": traces,
            "question": state["question"],
        }

    while True:
        prompt_records = []
        for state in batch_states:
            case_id = state["unique_id"]
            merge_state = merge_states[case_id]
            if merge_state["next_start"] >= merge_state["max_steps"]:
                continue
            start_index = merge_state["next_start"]
            stop_index = start_index + merge_state["chunk_size"]
            prompt_records.append(
                (
                    case_id,
                    build_incremental_claim_merge_prompt(
                        question=state["question"],
                        traces=merge_state["traces"],
                        start_index=start_index,
                        stop_index=stop_index,
                        existing_dossier=merge_state["dossier"],
                        profile=pipeline.prompt_profile,
                    ),
                )
            )
        if not prompt_records:
            break
        outputs = run_prompts(
            runtime,
            [(f"{case_id}:merge", prompt) for case_id, prompt in prompt_records],
            pipeline.system_prompt,
        )
        for (case_id, _), output in zip(prompt_records, outputs):
            merge_state = merge_states[case_id]
            parsed = pipeline._sanitize_graph(
                parse_graph_dossier(output),
                traces=merge_state["traces"],
                question=merge_state["question"],
            )
            if pipeline._should_accept_graph_update(merge_state["graph"], parsed, merge_state["traces"]):
                merge_state["dossier"] = parsed.raw_dossier
                merge_state["graph"] = parsed
            if pipeline.should_stop_prefix_expansion(merge_state["graph"], merge_state["traces"]):
                merge_state["next_start"] = merge_state["max_steps"]
            else:
                merge_state["next_start"] += merge_state["chunk_size"]

    return {case_id: merge_state["graph"] for case_id, merge_state in merge_states.items()}


def _audit_graphs_batched(batch_states: list[dict], traces_by_case: dict[str, dict[str, AgentTrace]], graphs_by_case: dict[str, object], pipeline: NaturalLanguageGraphDebatePipeline, runtime) -> dict[str, object]:
    prompt_records = []
    for state in batch_states:
        case_id = state["unique_id"]
        graph = graphs_by_case[case_id]
        if not graph.raw_dossier.strip():
            continue
        prompt_records.append(
            (
                case_id,
                build_shared_graph_audit_prompt(
                    state["question"],
                    graph,
                    traces_by_case.get(case_id, {}),
                    profile=pipeline.prompt_profile,
                ),
            )
        )
    if not prompt_records:
        return graphs_by_case

    outputs = run_prompts(
        runtime,
        [(f"{case_id}:audit", prompt) for case_id, prompt in prompt_records],
        pipeline.system_prompt,
    )
    for (case_id, _), output in zip(prompt_records, outputs):
        state = next(item for item in batch_states if item["unique_id"] == case_id)
        parsed = pipeline._sanitize_graph(
            parse_graph_dossier(output),
            traces=traces_by_case.get(case_id, {}),
            question=state["question"],
        )
        if pipeline._should_accept_graph_update(graphs_by_case[case_id], parsed, traces_by_case.get(case_id, {})):
            graphs_by_case[case_id] = parsed
    return graphs_by_case


def _repair_graphs_batched(batch_states: list[dict], traces_by_case: dict[str, dict[str, AgentTrace]], graphs_by_case: dict[str, object], pipeline: NaturalLanguageGraphDebatePipeline, runtime) -> dict[str, object]:
    prompt_records = []
    for state in batch_states:
        case_id = state["unique_id"]
        traces = traces_by_case.get(case_id, {})
        graph = graphs_by_case[case_id]
        if not pipeline._needs_prefix_conflict_repair(graph, traces):
            continue
        prompt_records.append(
            (
                case_id,
                build_prefix_conflict_graph_prompt(
                    state["question"],
                    graph,
                    traces,
                    profile=pipeline.prompt_profile,
                ),
            )
        )
    if not prompt_records:
        return graphs_by_case

    outputs = run_prompts(
        runtime,
        [(f"{case_id}:prefix_repair", prompt) for case_id, prompt in prompt_records],
        pipeline.system_prompt,
    )
    for (case_id, _), output in zip(prompt_records, outputs):
        traces = traces_by_case.get(case_id, {})
        state = next(item for item in batch_states if item["unique_id"] == case_id)
        parsed = pipeline._sanitize_graph(
            parse_graph_dossier(output),
            traces=traces,
            question=state["question"],
        )
        if pipeline._should_accept_graph_update(graphs_by_case[case_id], parsed, traces):
            graphs_by_case[case_id] = parsed
    return graphs_by_case


def _resolve_graphs_batched(batch_states: list[dict], traces_by_case: dict[str, dict[str, AgentTrace]], graphs_by_case: dict[str, object], pipeline: NaturalLanguageGraphDebatePipeline, runtime) -> tuple[dict[str, list], dict[str, list]]:
    resolutions_by_case = {state["unique_id"]: [] for state in batch_states}
    raw_notes_by_case = {state["unique_id"]: [] for state in batch_states}
    strict_json_io = bool(getattr(pipeline, "strict_json_io", False))
    parse_resolution = parse_resolution_note_json_only if strict_json_io else parse_resolution_note

    analysis_prompt_records = []
    for state in batch_states:
        case_id = state["unique_id"]
        graph = graphs_by_case[case_id]
        divergences = pipeline._select_real_claim_divergences(graph)
        selected_divergences = divergences if pipeline.divergence_selection_variant == "all" else [
            pipeline._select_primary_real_claim_divergence(graph)
        ]
        for divergence in [item for item in selected_divergences if item is not None]:
            swapped = _should_swap_resolution_prompt(case_id, divergence.divergence_id)
            prompt_divergence = _swap_divergence_for_prompt_order(divergence) if swapped else divergence
            analysis_prompt_records.append(
                (
                    case_id,
                    divergence.divergence_id,
                    divergence,
                    prompt_divergence,
                    swapped,
                    build_divergence_relation_analysis_prompt(
                        state["question"],
                        graph,
                        prompt_divergence,
                        traces_by_case.get(case_id, {}),
                        profile=pipeline.prompt_profile,
                        graph_format=pipeline.graph_format,
                        resolution_trace_context=getattr(pipeline, "resolution_trace_context", "window"),
                        resolution_prompt_style=getattr(pipeline, "resolution_prompt_style", "profile"),
                    ),
                )
            )
    if analysis_prompt_records:
        analysis_outputs = run_prompts(
            runtime,
            [(f"{case_id}:{divergence_id}:analysis", prompt) for case_id, divergence_id, _, _, _, prompt in analysis_prompt_records],
            pipeline.system_prompt,
        )
        resolve_prompt_records = []
        for (case_id, divergence_id, divergence, prompt_divergence, swapped, _), analysis_output in zip(analysis_prompt_records, analysis_outputs):
            state = next(item for item in batch_states if item["unique_id"] == case_id)
            backend_decision = pipeline._backend_graph_divergence_decision(
                state["question"],
                graphs_by_case[case_id],
                prompt_divergence,
                traces_by_case.get(case_id, {}),
            )
            if backend_decision is not None:
                backend_decision = _map_prompt_order_decision(backend_decision, swapped=swapped)
                backend_decision = pipeline._align_resolution_paths_with_resolved_claim(
                    backend_decision,
                    divergence,
                    question=state["question"],
                )
                if _is_synthesize_resolution(backend_decision):
                    continue
                raw_notes_by_case[case_id].append(
                    {
                        "divergence_id": divergence_id,
                        "analysis_text": analysis_output,
                        "raw_text": _public_resolution_log_text(pipeline.build_resolution_text([backend_decision])),
                        "decision_source": "backend_after_analysis",
                        "backend_adjustment": backend_decision.rationale,
                    }
                )
                resolutions_by_case[case_id].append(backend_decision)
                continue
            resolve_prompt_records.append(
                (
                    case_id,
                    divergence_id,
                    swapped,
                    build_divergence_resolution_prompt(
                        state["question"],
                        graphs_by_case[case_id],
                        prompt_divergence,
                        relation_analysis=analysis_output,
                        traces=traces_by_case.get(case_id, {}),
                        profile=pipeline.prompt_profile,
                        graph_format=pipeline.graph_format,
                        resolution_trace_context=getattr(pipeline, "resolution_trace_context", "window"),
                        resolution_prompt_style=getattr(pipeline, "resolution_prompt_style", "profile"),
                    ),
                    analysis_output,
                )
            )
        if resolve_prompt_records:
            resolve_outputs = run_prompts(
                runtime,
                [(f"{case_id}:{divergence_id}:resolve", prompt) for case_id, divergence_id, _, prompt, _ in resolve_prompt_records],
                pipeline.system_prompt,
            )
            for (case_id, divergence_id, swapped, _, analysis_output), output in zip(resolve_prompt_records, resolve_outputs):
                decision = parse_resolution(output, default_divergence_id=divergence_id)
                if _is_synthesize_resolution(decision):
                    continue
                decision = _map_prompt_order_decision(decision, swapped=swapped)
                divergence = next(
                    (
                        item
                        for item in getattr(graphs_by_case[case_id], "divergences", [])
                        if item.divergence_id == divergence_id
                    ),
                    None,
                )
                if divergence is not None:
                    decision = pipeline._align_resolution_paths_with_resolved_claim(
                        decision,
                        divergence,
                        question=state["question"],
                    )
                decision = _coerce_synthesize_side_selection(decision, graphs_by_case[case_id])
                raw_notes_by_case[case_id].append(
                    {
                        "divergence_id": divergence_id,
                        "analysis_text": analysis_output,
                        "raw_text": _public_resolution_log_text(output),
                        "decision_source": "model_after_analysis",
                    }
                )
                if decision.action or decision.resolved_claim or decision.keep_paths or decision.drop_paths:
                    resolutions_by_case[case_id].append(decision)

    for state in batch_states:
        case_id = state["unique_id"]
        if resolutions_by_case[case_id] and not _has_actionable_resolution(resolutions_by_case[case_id]):
            raw_notes_by_case[case_id].append(
                {
                    "divergence_id": "D_NON_ACTIONABLE_BLOCKED",
                    "analysis_text": "Structured resolution produced only non-actionable merge notes, so answer-disagreement fallback may run.",
                    "raw_text": "",
                    "decision_source": "backend_non_actionable_block",
                }
            )
            resolutions_by_case[case_id] = _drop_synthesize_only_resolutions(resolutions_by_case[case_id])
        elif resolutions_by_case[case_id]:
            resolutions_by_case[case_id] = _drop_synthesize_only_resolutions(resolutions_by_case[case_id])

    if getattr(pipeline, "resolution_acceptance_policy", "guarded") != "trust_model":
        for state in batch_states:
            case_id = state["unique_id"]
            if not resolutions_by_case[case_id]:
                continue
            before_gate = list(resolutions_by_case[case_id])
            current_answers = _response_answers(state["graph"]["responses"], state["question"])
            resolutions_by_case[case_id] = _evidence_gate_resolutions(
                resolutions_by_case[case_id],
                graphs_by_case[case_id],
                current_answers,
                question=state["question"],
                traces=traces_by_case.get(case_id, {}),
                pipeline=pipeline,
            )
            if before_gate and not resolutions_by_case[case_id]:
                raw_notes_by_case[case_id].append(
                    {
                        "divergence_id": "D_EVIDENCE_GATE_BLOCKED",
                        "analysis_text": "Structured resolution was rejected by local evidence gates, so answer-disagreement fallback may run.",
                        "raw_text": _public_resolution_log_text(pipeline.build_resolution_text(before_gate)),
                        "decision_source": "backend_evidence_gate_block",
                    }
                )
                state["graph"]["gate_blocked_this_round"] = True

    for state in batch_states:
        case_id = state["unique_id"]
        if resolutions_by_case[case_id]:
            continue
        if state["graph"].pop("gate_blocked_this_round", False):
            continue
        if not state["graph"].get("had_structured_resolution"):
            continue
        memory_resolutions = _memory_resolutions_for_state(state)
        if not memory_resolutions:
            continue
        resolutions_by_case[case_id].extend(memory_resolutions)
        raw_notes_by_case[case_id].append(
            {
                "divergence_id": "D_MEMORY",
                "analysis_text": "Reusing prior structured claim resolution; pairwise fallback is skipped for this case.",
                "raw_text": _public_resolution_log_text(pipeline.build_resolution_text(memory_resolutions)),
                "decision_source": "structured_memory",
            }
        )

    if bool(getattr(pipeline, "disable_pairwise_fallback", False)):
        return resolutions_by_case, raw_notes_by_case

    fallback_analysis_records = []
    for state in batch_states:
        case_id = state["unique_id"]
        if resolutions_by_case[case_id]:
            continue
        traces = traces_by_case.get(case_id, {})
        if not pipeline._has_answer_disagreement(traces):
            continue
        disagreement_pairs = pipeline._select_disagreement_pairs(state["question"], traces)
        for disagreement_pair in disagreement_pairs:
            left_trace, right_trace = disagreement_pair
            pair_key = f"PAIR_{left_trace.agent_id}_{right_trace.agent_id}"
            swapped = _should_swap_resolution_prompt(case_id, pair_key)
            prompt_left_trace, prompt_right_trace = (right_trace, left_trace) if swapped else (left_trace, right_trace)
            fallback_analysis_records.append(
                (
                    case_id,
                    pair_key,
                    left_trace,
                    right_trace,
                    prompt_left_trace,
                    prompt_right_trace,
                    swapped,
                    build_pairwise_relation_analysis_prompt(
                        state["question"],
                        prompt_left_trace,
                        prompt_right_trace,
                        profile=pipeline.prompt_profile,
                        resolution_trace_context=getattr(pipeline, "resolution_trace_context", "window"),
                        resolution_prompt_style=getattr(pipeline, "resolution_prompt_style", "profile"),
                        graph_format=pipeline.graph_format,
                    ),
                )
            )
    if fallback_analysis_records:
        analysis_outputs = run_prompts(
            runtime,
            [(f"{case_id}:{pair_key}:pair_analysis", prompt) for case_id, pair_key, _, _, _, _, _, prompt in fallback_analysis_records],
            pipeline.system_prompt,
        )
        resolve_prompt_records = []
        for (case_id, pair_key, left_trace, right_trace, prompt_left_trace, prompt_right_trace, swapped, _), analysis_output in zip(fallback_analysis_records, analysis_outputs):
            state = next(item for item in batch_states if item["unique_id"] == case_id)
            guard_note = pipeline._build_pairwise_guard_note(state["question"], left_trace, right_trace)
            if guard_note:
                analysis_output = f"{guard_note}\n\n{analysis_output}".strip()
            backend_decision = pipeline._backend_pairwise_decision(state["question"], pair_key, left_trace, right_trace)
            if backend_decision is not None:
                if _is_synthesize_resolution(backend_decision):
                    continue
                raw_notes_by_case[case_id].append(
                    {
                        "divergence_id": pair_key,
                        "analysis_text": analysis_output,
                        "raw_text": _public_resolution_log_text(pipeline.build_resolution_text([backend_decision])),
                        "decision_source": "backend_after_analysis",
                        "backend_adjustment": backend_decision.rationale,
                    }
                )
                resolutions_by_case[case_id].append(backend_decision)
                continue
            resolve_prompt_records.append(
                (
                    case_id,
                    pair_key,
                    swapped,
                    build_pairwise_divergence_resolution_prompt(
                        state["question"],
                        prompt_left_trace,
                        prompt_right_trace,
                        relation_analysis=analysis_output,
                        profile=pipeline.prompt_profile,
                        resolution_trace_context=getattr(pipeline, "resolution_trace_context", "window"),
                        resolution_prompt_style=getattr(pipeline, "resolution_prompt_style", "profile"),
                        graph_format=pipeline.graph_format,
                    ),
                    analysis_output,
                )
            )
        if resolve_prompt_records:
            resolve_outputs = run_prompts(
                runtime,
                [(f"{case_id}:{pair_key}:pair_resolve", prompt) for case_id, pair_key, _, prompt, _ in resolve_prompt_records],
                pipeline.system_prompt,
            )
            for (case_id, pair_key, swapped, _, analysis_output), output in zip(resolve_prompt_records, resolve_outputs):
                decision = parse_resolution(output, default_divergence_id=pair_key)
                if _is_synthesize_resolution(decision):
                    continue
                decision = _map_prompt_order_decision(decision, swapped=swapped)
                decision = _coerce_synthesize_side_selection(decision, graphs_by_case[case_id])
                raw_notes_by_case[case_id].append(
                    {
                        "divergence_id": pair_key,
                        "analysis_text": analysis_output,
                        "raw_text": _public_resolution_log_text(output),
                        "decision_source": "model_after_analysis",
                    }
                )
                if decision.action or decision.resolved_claim or decision.keep_paths or decision.drop_paths:
                    resolutions_by_case[case_id].append(decision)

    for state in batch_states:
        case_id = state["unique_id"]
        if resolutions_by_case[case_id] and not _has_actionable_resolution(resolutions_by_case[case_id]):
            raw_notes_by_case[case_id].append(
                {
                    "divergence_id": "D_NON_ACTIONABLE_BLOCKED",
                    "analysis_text": "Only non-actionable merge notes remained after fallback.",
                    "raw_text": "",
                    "decision_source": "backend_non_actionable_block",
                }
            )
            resolutions_by_case[case_id] = _drop_synthesize_only_resolutions(resolutions_by_case[case_id])
        elif resolutions_by_case[case_id]:
            resolutions_by_case[case_id] = _drop_synthesize_only_resolutions(resolutions_by_case[case_id])

    return resolutions_by_case, raw_notes_by_case


def _run_graph_rounds_batched(
    batch_states: list[dict],
    runtime,
    rounds: int,
    prompt_profile: str | None = None,
    divergence_selection_variant: str = "first_real",
    rewrite_context_variant: str = "current_suffix",
    graph_format: str = "natural",
    divergence_random_seed: int = 7,
    disable_resolution_landing: bool = False,
    disable_pairwise_fallback: bool = False,
    strict_json_io: bool = False,
    answer_selection_policy: str = "standard",
    canonical_answer_mode: str = "broadcast_stop",
    stall_policy: str = "stop",
    unanimous_policy: str = "skip",
    resolution_trace_context: str = "window",
    resolution_prompt_style: str = "profile",
    resolution_acceptance_policy: str = "guarded",
) -> None:
    pipeline = NaturalLanguageGraphDebatePipeline(
        prompt_profile=prompt_profile,
        divergence_selection_variant=divergence_selection_variant,
        rewrite_context_variant=rewrite_context_variant,
        graph_format=graph_format,
        divergence_random_seed=divergence_random_seed,
    )
    pipeline.disable_pairwise_fallback = bool(disable_pairwise_fallback)
    pipeline.strict_json_io = bool(strict_json_io)
    pipeline.resolution_trace_context = _resolution_trace_context_name(resolution_trace_context)
    pipeline.resolution_prompt_style = _resolution_prompt_style_name(resolution_prompt_style)
    answer_selection_policy = _answer_selection_policy_name(answer_selection_policy)
    canonical_answer_mode = _canonical_answer_mode_name(canonical_answer_mode)
    stall_policy = _stall_policy_name(stall_policy)
    unanimous_policy = _unanimous_policy_name(unanimous_policy)
    resolution_acceptance_policy = _resolution_acceptance_policy_name(resolution_acceptance_policy)
    pipeline.resolution_acceptance_policy = resolution_acceptance_policy
    for round_idx in range(1, rounds + 1):
        _pop_runtime_api_trace(runtime)
        for state in batch_states:
            if state["graph"]["done"]:
                continue
            current_answers = _response_answers(state["graph"]["responses"], state["question"])
            if unanimous_policy == "skip" and _all_answers_unanimous(current_answers):
                state["graph"]["done"] = True
        active_states = [state for state in batch_states if not state["graph"]["done"]]
        if not active_states:
            break

        batch_entries = [(state, state["graph"]) for state in active_states]
        traces_by_case = _normalize_traces_batched(batch_entries, runtime, pipeline)
        graphs_by_case = _merge_graphs_batched(active_states, traces_by_case, pipeline, runtime)
        graphs_by_case = _audit_graphs_batched(active_states, traces_by_case, graphs_by_case, pipeline, runtime)
        graphs_by_case = _repair_graphs_batched(active_states, traces_by_case, graphs_by_case, pipeline, runtime)
        graphs_by_case = {
            state["unique_id"]: pipeline.ensure_real_claim_divergence(
                state["question"],
                traces_by_case.get(state["unique_id"], {}),
                graphs_by_case[state["unique_id"]],
            )
            for state in active_states
        }
        resolutions_by_case, raw_notes_by_case = _resolve_graphs_batched(active_states, traces_by_case, graphs_by_case, pipeline, runtime)

        revision_payload_records = []
        for state in active_states:
            case_id = state["unique_id"]
            input_answers = {
                agent_id: _extract_answer_from_trace_text(text, state["problem"])
                for agent_id, text in state["graph"]["responses"].items()
            }
            resolutions_by_case[case_id] = _evidence_gate_resolutions(
                resolutions_by_case[case_id],
                graphs_by_case[case_id],
                input_answers,
                question=state["question"],
                traces=traces_by_case.get(case_id, {}),
                pipeline=pipeline,
            )
            current_canonical = None
            if not disable_resolution_landing:
                current_canonical = _allowed_resolution_canonical_answer(
                    resolutions_by_case[case_id],
                    question=state["problem"],
                    prompt_profile=prompt_profile,
                    visible_answers=input_answers,
                )
            if current_canonical and canonical_answer_mode == "broadcast_stop":
                state["graph"]["canonical_answer"] = current_canonical
            payloads = pipeline.build_revision_message_payloads(
                state["question"],
                traces_by_case.get(case_id, {}),
                graphs_by_case[case_id],
                resolutions_by_case[case_id],
            )
            for agent_id, payload in payloads.items():
                messages = payload.get("messages") or []
                continuation_prefix = ""
                if len(messages) > 1 and isinstance(messages[1], dict):
                    continuation_prefix = str(messages[1].get("content", "") or "")
                revision_payload_records.append((case_id, agent_id, payload, continuation_prefix))

        revised_traces_by_case: dict[str, dict[str, AgentTrace]] = {state["unique_id"]: {} for state in active_states}
        if revision_payload_records:
            outputs = run_prompts(
                runtime,
                [(f"{case_id}:{agent_id}:revise", payload) for case_id, agent_id, payload, _ in revision_payload_records],
                pipeline.system_prompt,
            )
            for (case_id, agent_id, _, continuation_prefix), output in zip(revision_payload_records, outputs):
                source_trace = traces_by_case.get(case_id, {}).get(agent_id)
                if source_trace is None:
                    normalized = parse_atomic_trace(output)
                    revised_trace = AgentTrace(
                        agent_id=agent_id,
                        original_response=output,
                        normalized_trace_text=normalized,
                        steps=build_step_objects(agent_id, normalized),
                    )
                else:
                    revised_trace = pipeline.build_revised_trace_from_output(
                        agent_id,
                        source_trace,
                        output,
                        graphs_by_case[case_id],
                        resolutions_by_case[case_id],
                        continuation_prefix=continuation_prefix,
                    )
                revised_traces_by_case.setdefault(case_id, {})[agent_id] = revised_trace

        api_trace_by_case: dict[str, list[dict]] = defaultdict(list)
        for trace_item in _pop_runtime_api_trace(runtime):
            alias = str(trace_item.get("alias", "") or "")
            case_id = alias.split(":", 1)[0] if ":" in alias else ""
            if case_id:
                api_trace_by_case[case_id].append(trace_item)

        for state in active_states:
            case_id = state["unique_id"]
            previous_responses = copy.deepcopy(state["graph"]["responses"])
            revised_traces = revised_traces_by_case.get(case_id, {})
            input_answers = {
                agent_id: _extract_answer_from_trace_text(text, state["problem"])
                for agent_id, text in previous_responses.items()
            }
            if resolution_acceptance_policy == "trust_model":
                rewrite_consistency = {
                    agent_id: {
                        "accepted": True,
                        "reason": "trust_model resolution acceptance policy",
                    }
                    for agent_id in revised_traces
                }
                allow_resolution_landing = True
            else:
                revised_traces, rewrite_consistency, allow_resolution_landing = _filter_revisions_by_consistency(
                    revised_traces,
                    resolutions_by_case[case_id],
                    graphs_by_case[case_id],
                    input_answers,
                    _observed_graph_answers(state["initial_answers"], state["graph"]["round_results"]),
                    question=state["problem"],
                )
            if disable_resolution_landing:
                allow_resolution_landing = False
            accepted_answer_changed = _accepted_revision_changes_answer(
                revised_traces,
                input_answers,
                question=state["problem"],
            )
            next_responses = _trace_text_map(revised_traces)
            updated_responses = copy.deepcopy(previous_responses)
            updated_responses.update(next_responses)
            canonical_answer = None
            if allow_resolution_landing:
                canonical_answer = _allowed_resolution_canonical_answer(
                    resolutions_by_case[case_id],
                    question=state["problem"],
                    prompt_profile=prompt_profile,
                    visible_answers=input_answers,
                    revised_answers={
                        agent_id: _extract_answer_from_trace_text(text, state["problem"])
                        for agent_id, text in updated_responses.items()
                    },
                )
            if not disable_resolution_landing and canonical_answer_mode == "broadcast_stop":
                canonical_answer = canonical_answer or state["graph"].get("canonical_answer")
            if canonical_answer:
                if canonical_answer_mode == "broadcast_stop":
                    updated_responses = _apply_canonical_answer_to_responses(updated_responses, canonical_answer)
                    revised_answers = _canonical_answer_map(updated_responses.keys(), canonical_answer)
                else:
                    revised_answers = {
                        agent_id: _extract_answer_from_trace_text(text, state["problem"])
                        for agent_id, text in updated_responses.items()
                    }
                chosen_answer = canonical_answer
            else:
                revised_answers = {
                    agent_id: _extract_answer_from_trace_text(text, state["problem"])
                    for agent_id, text in updated_responses.items()
                }
                continuation_answer = _accepted_continuation_answer(
                    revised_traces,
                    input_answers,
                    revised_answers,
                    question=state["problem"],
                )
                if resolution_acceptance_policy == "trust_model":
                    preferred_answer = _trusted_model_answer_from_resolutions(
                        resolutions_by_case[case_id],
                        graphs_by_case[case_id],
                        revised_answers,
                        input_answers,
                        question=state["problem"],
                    )
                else:
                    preferred_answer = _preferred_answer_from_resolutions(
                        resolutions_by_case[case_id],
                        graphs_by_case[case_id],
                        revised_answers,
                        input_answers,
                        question=state["problem"],
                    ) if allow_resolution_landing else None
                majority_answer = _majority_answer(revised_answers)
                chosen_answer = continuation_answer or preferred_answer or majority_answer
            chosen_answer = _choose_landed_answer(
                state["problem"],
                chosen_answer,
                continuation_answer if not canonical_answer else None,
                preferred_answer if not canonical_answer else None,
                majority_answer if not canonical_answer else None,
            )
            answer_selection_report = {
                "policy": answer_selection_policy,
                "applied": False,
                "reason": "standard answer selection",
            }
            if answer_selection_policy in {"supported_answer_memory", "stable_majority_guard"}:
                if answer_selection_policy == "stable_majority_guard":
                    chosen_answer, answer_selection_report = _apply_stable_majority_guard_policy(
                        state["graph"],
                        state["problem"],
                        chosen_answer,
                        revised_answers,
                        round_index=round_idx,
                        input_answers=input_answers,
                        canonical_answer=canonical_answer,
                        continuation_answer=continuation_answer if not canonical_answer else None,
                        preferred_answer=preferred_answer if not canonical_answer else None,
                        majority_answer=majority_answer if not canonical_answer else None,
                        accepted_answer_changed=accepted_answer_changed,
                        resolutions=resolutions_by_case[case_id],
                    )
                else:
                    chosen_answer, answer_selection_report = _apply_supported_answer_memory_policy(
                        state["graph"],
                        state["problem"],
                        chosen_answer,
                        revised_answers,
                        round_index=round_idx,
                        canonical_answer=canonical_answer,
                        continuation_answer=continuation_answer if not canonical_answer else None,
                        preferred_answer=preferred_answer if not canonical_answer else None,
                        majority_answer=majority_answer if not canonical_answer else None,
                        accepted_answer_changed=accepted_answer_changed,
                    )
            stalled_after_failed_rewrite = (
                not canonical_answer
                and bool(rewrite_consistency)
                and not allow_resolution_landing
                and not accepted_answer_changed
            )
            if stalled_after_failed_rewrite:
                prior_chosen = (
                    state["graph"]["round_results"][-1].get("chosen_answer")
                    if state["graph"]["round_results"]
                    else _majority_answer(input_answers)
                )
                chosen_answer = _choose_landed_answer(state["problem"], prior_chosen, majority_answer if not canonical_answer else None)
            if (
                allow_resolution_landing
                and not _has_rewrite_rejections(rewrite_consistency)
                and (
                    canonical_answer
                    or _accepted_revision_changes_answer(
                        revised_traces,
                        input_answers,
                        question=state["problem"],
                    )
                )
            ):
                _remember_structured_resolutions(state, resolutions_by_case[case_id])
            state["graph"]["round_results"].append(
                {
                    "round_index": round_idx,
                    "input_answers": {
                        agent_id: _extract_answer_from_trace_text(text, state["problem"])
                        for agent_id, text in previous_responses.items()
                    },
                    "revised_answers": revised_answers,
                    "chosen_answer": chosen_answer,
                    "graph_dossier": graphs_by_case[case_id].raw_dossier,
                    "resolutions": [item.__dict__ for item in resolutions_by_case[case_id]],
                    "raw_resolution_notes": copy.deepcopy(raw_notes_by_case[case_id]),
                    "rewrite_consistency": rewrite_consistency,
                    "resolution_landing_allowed": allow_resolution_landing,
                    "resolution_acceptance_policy": resolution_acceptance_policy,
                    "resolution_landing_disabled": bool(disable_resolution_landing),
                    "canonical_answer_mode": canonical_answer_mode,
                    "stall_policy": stall_policy,
                    "unanimous_policy": unanimous_policy,
                    "answer_selection_policy": answer_selection_policy,
                    "answer_selection_report": answer_selection_report,
                    "api_trace": copy.deepcopy(api_trace_by_case.get(case_id, [])),
                    "revised_traces": {
                        agent_id: {
                            "normalized_trace_text": trace.normalized_trace_text,
                            "steps": [step.text for step in trace.steps],
                        }
                        for agent_id, trace in revised_traces.items()
                    },
                }
            )
            state["graph"]["responses"] = updated_responses
            if (
                (canonical_answer and canonical_answer_mode == "broadcast_stop")
                or (state["graph"].get("canonical_answer") if (not disable_resolution_landing and canonical_answer_mode == "broadcast_stop") else None)
                or (unanimous_policy == "skip" and _all_answers_unanimous(revised_answers))
                or (stalled_after_failed_rewrite and stall_policy == "stop")
            ):
                state["graph"]["done"] = True


def _run_origin_rounds(
    question: str,
    initial_responses: dict[str, str],
    runtime,
    rounds: int,
) -> dict:
    args = _build_origin_args()
    origin_question = _build_origin_question_prompt(question)
    responses = dict(initial_responses)
    round_results = []
    for round_idx in range(1, rounds + 1):
        prev_answers = _get_answer_map(args, responses, question=question)
        peer_map = _get_peer_map(args, list(responses.keys()), prev_answers)
        update_messages = _build_direct_response_update_messages(
            args,
            origin_question,
            responses,
            peer_map,
            personas=None,
        )
        update_items = list(update_messages.items())
        update_outputs = engine([msg for _, msg in update_items], runtime.agent, len(update_items)) if update_items else []

        updated_responses = dict(responses)
        for (agent_id, _), output in zip(update_items, update_outputs):
            previous_response = responses.get(agent_id, "")
            clean_response = _parse_direct_update_output(output, previous_response)
            updated_responses[agent_id] = clean_response

        revised_answers = {
            agent_id: _extract_answer_from_trace_text(text, question)
            for agent_id, text in updated_responses.items()
        }
        chosen_answer = _majority_answer(revised_answers)
        round_results.append(
            {
                "round_index": round_idx,
                "input_answers": {
                    agent_id: _extract_answer_from_trace_text(text, question)
                    for agent_id, text in responses.items()
                },
                "revised_answers": revised_answers,
                "chosen_answer": chosen_answer,
                "peer_map": copy.deepcopy(peer_map),
                "critique_responses": {},
                "revision_metadata": {},
                "responses": copy.deepcopy(updated_responses),
            }
        )
        responses = updated_responses
        if len({answer for answer in revised_answers.values() if answer}) <= 1:
            break

    final_round = round_results[-1] if round_results else {
        "revised_answers": {
            agent_id: _extract_answer_from_trace_text(text, question)
            for agent_id, text in initial_responses.items()
        },
        "chosen_answer": _majority_answer(
            {
                agent_id: _extract_answer_from_trace_text(text, question)
                for agent_id, text in initial_responses.items()
            }
        ),
        "responses": copy.deepcopy(initial_responses),
    }
    return {
        "round_results": round_results,
        "revised_answers": final_round["revised_answers"],
        "chosen_answer": final_round["chosen_answer"],
        "final_responses": final_round["responses"],
    }


def _run_graph_rounds(
    question: str,
    initial_responses: dict[str, str],
    runtime,
    rounds: int,
    prompt_profile: str | None = None,
    divergence_selection_variant: str = "first_real",
    rewrite_context_variant: str = "current_suffix",
    graph_format: str = "natural",
    divergence_random_seed: int = 7,
    disable_resolution_landing: bool = False,
    disable_pairwise_fallback: bool = False,
    strict_json_io: bool = False,
    answer_selection_policy: str = "standard",
    canonical_answer_mode: str = "broadcast_stop",
    stall_policy: str = "stop",
    unanimous_policy: str = "skip",
    resolution_trace_context: str = "window",
    resolution_prompt_style: str = "profile",
    resolution_acceptance_policy: str = "guarded",
) -> dict:
    unanimous_policy = _unanimous_policy_name(unanimous_policy)
    initial_answers = {
        agent_id: _extract_answer_from_trace_text(text, question)
        for agent_id, text in initial_responses.items()
    }
    if unanimous_policy == "skip" and _all_answers_unanimous(initial_answers):
        return {
            "round_results": [],
            "revised_answers": initial_answers,
            "chosen_answer": _majority_answer(initial_answers),
            "final_responses": copy.deepcopy(initial_responses),
            "graph_dossier": "",
            "resolutions": [],
            "raw_resolution_notes": [],
            "revised_traces": {},
        }

    pipeline = NaturalLanguageGraphDebatePipeline(
        prompt_profile=prompt_profile,
        divergence_selection_variant=divergence_selection_variant,
        rewrite_context_variant=rewrite_context_variant,
        graph_format=graph_format,
        divergence_random_seed=divergence_random_seed,
    )
    pipeline.disable_pairwise_fallback = bool(disable_pairwise_fallback)
    pipeline.strict_json_io = bool(strict_json_io)
    pipeline.resolution_trace_context = _resolution_trace_context_name(resolution_trace_context)
    pipeline.resolution_prompt_style = _resolution_prompt_style_name(resolution_prompt_style)
    resolution_acceptance_policy = _resolution_acceptance_policy_name(resolution_acceptance_policy)
    pipeline.resolution_acceptance_policy = resolution_acceptance_policy
    answer_selection_policy = _answer_selection_policy_name(answer_selection_policy)
    canonical_answer_mode = _canonical_answer_mode_name(canonical_answer_mode)
    stall_policy = _stall_policy_name(stall_policy)
    responses = dict(initial_responses)
    round_results = []
    artifacts = None
    claim_invariants: list[dict] = []
    canonical_answer: str | None = None
    supported_answer_memory: dict | None = None
    for round_idx in range(1, rounds + 1):
        current_answers = _response_answers(responses, question)
        if unanimous_policy == "skip" and _all_answers_unanimous(current_answers):
            break
        artifacts = pipeline.run(question, responses, runtime)
        input_answers = {
            agent_id: _extract_answer_from_trace_text(text, question)
            for agent_id, text in responses.items()
        }
        if resolution_acceptance_policy == "trust_model":
            resolutions = _drop_synthesize_only_resolutions(artifacts.resolutions)
            revised_traces = dict(artifacts.revised_traces)
            rewrite_consistency = {
                agent_id: {
                    "accepted": True,
                    "reason": "trust_model resolution acceptance policy",
                }
                for agent_id in revised_traces
            }
            allow_resolution_landing = True
        else:
            resolutions = _drop_synthesize_only_resolutions(
                _evidence_gate_resolutions(
                    artifacts.resolutions,
                    artifacts.graph,
                    input_answers,
                    question=question,
                    traces={
                        agent_id: AgentTrace(
                            agent_id=agent_id,
                            original_response=text,
                            normalized_trace_text=(normalized := parse_atomic_trace(text)),
                            steps=build_step_objects(agent_id, normalized),
                        )
                        for agent_id, text in responses.items()
                    },
                    pipeline=pipeline,
                )
            )
            revised_traces, rewrite_consistency, allow_resolution_landing = _filter_revisions_by_consistency(
                artifacts.revised_traces,
                resolutions,
                artifacts.graph,
                input_answers,
                question=question,
            )
        if disable_resolution_landing:
            allow_resolution_landing = False
        accepted_answer_changed = _accepted_revision_changes_answer(revised_traces, input_answers, question=question)
        next_responses = _trace_text_map(revised_traces)
        current_canonical = (
            _allowed_resolution_canonical_answer(
                resolutions,
                question=question,
                prompt_profile=prompt_profile,
                visible_answers=input_answers,
                revised_answers={
                    agent_id: _extract_answer_from_trace_text(text, question)
                    for agent_id, text in (next_responses or responses).items()
                },
            )
            if allow_resolution_landing
            else None
        )
        if current_canonical and not disable_resolution_landing and canonical_answer_mode == "broadcast_stop":
            canonical_answer = current_canonical
        round_canonical_answer = canonical_answer if canonical_answer_mode == "broadcast_stop" else current_canonical
        if round_canonical_answer:
            if canonical_answer_mode == "broadcast_stop":
                base_responses = next_responses or responses
                next_responses = _apply_canonical_answer_to_responses(base_responses, round_canonical_answer)
                revised_answers = _canonical_answer_map(next_responses.keys(), round_canonical_answer)
            else:
                revised_answers = {
                    agent_id: _extract_answer_from_trace_text(text, question)
                    for agent_id, text in (next_responses or responses).items()
                }
            chosen_answer = round_canonical_answer
        else:
            revised_answers = {
                agent_id: _extract_answer_from_trace_text(text, question)
                for agent_id, text in (next_responses or responses).items()
            }
            continuation_answer = _accepted_continuation_answer(
                revised_traces,
                input_answers,
                revised_answers,
                question=question,
            )
            if resolution_acceptance_policy == "trust_model":
                preferred_answer = _trusted_model_answer_from_resolutions(
                    resolutions,
                    artifacts.graph,
                    revised_answers,
                    input_answers,
                    question=question,
                )
            else:
                preferred_answer = _preferred_answer_from_resolutions(
                    resolutions,
                    artifacts.graph,
                    revised_answers,
                    input_answers,
                    question=question,
                ) if allow_resolution_landing else None
            majority_answer = _majority_answer(revised_answers)
            chosen_answer = continuation_answer or preferred_answer or majority_answer
        chosen_answer = _choose_landed_answer(
            question,
            chosen_answer,
            continuation_answer if not round_canonical_answer else None,
            preferred_answer if not round_canonical_answer else None,
            majority_answer if not round_canonical_answer else None,
        )
        answer_selection_report = {
            "policy": answer_selection_policy,
            "applied": False,
            "reason": "standard answer selection",
        }
        if answer_selection_policy in {"supported_answer_memory", "stable_majority_guard"}:
            memory_state = {"supported_answer_memory": copy.deepcopy(supported_answer_memory)}
            if answer_selection_policy == "stable_majority_guard":
                chosen_answer, answer_selection_report = _apply_stable_majority_guard_policy(
                    memory_state,
                    question,
                    chosen_answer,
                    revised_answers,
                    round_index=round_idx,
                    input_answers=input_answers,
                    canonical_answer=round_canonical_answer,
                    continuation_answer=continuation_answer if not round_canonical_answer else None,
                    preferred_answer=preferred_answer if not round_canonical_answer else None,
                    majority_answer=majority_answer if not round_canonical_answer else None,
                    accepted_answer_changed=accepted_answer_changed,
                    resolutions=resolutions,
                )
            else:
                chosen_answer, answer_selection_report = _apply_supported_answer_memory_policy(
                    memory_state,
                    question,
                    chosen_answer,
                    revised_answers,
                    round_index=round_idx,
                    canonical_answer=round_canonical_answer,
                    continuation_answer=continuation_answer if not round_canonical_answer else None,
                    preferred_answer=preferred_answer if not round_canonical_answer else None,
                    majority_answer=majority_answer if not round_canonical_answer else None,
                    accepted_answer_changed=accepted_answer_changed,
                )
            supported_answer_memory = copy.deepcopy(memory_state.get("supported_answer_memory"))
        stalled_after_failed_rewrite = (
            not round_canonical_answer
            and bool(rewrite_consistency)
            and not allow_resolution_landing
            and not accepted_answer_changed
        )
        if stalled_after_failed_rewrite:
            prior_chosen = round_results[-1].get("chosen_answer") if round_results else _majority_answer(input_answers)
            chosen_answer = _choose_landed_answer(question, prior_chosen, majority_answer if not round_canonical_answer else None)
        if (
            allow_resolution_landing
            and not _has_rewrite_rejections(rewrite_consistency)
            and (round_canonical_answer or _accepted_revision_changes_answer(revised_traces, input_answers, question=question))
        ):
            for decision in resolutions:
                if not _is_pairwise_resolution(decision) and not _is_synthesize_resolution(decision):
                    claim_invariants.append(copy.deepcopy(decision.__dict__))
        round_results.append(
            {
                "round_index": round_idx,
                "input_answers": {
                    agent_id: _extract_answer_from_trace_text(text, question)
                    for agent_id, text in responses.items()
                },
                "revised_answers": revised_answers,
                "chosen_answer": chosen_answer,
                "graph_dossier": artifacts.graph.raw_dossier,
                "resolutions": [item.__dict__ for item in resolutions],
                "raw_resolution_notes": artifacts.raw_resolution_notes,
                "rewrite_consistency": rewrite_consistency,
                "resolution_landing_allowed": allow_resolution_landing,
                "resolution_acceptance_policy": resolution_acceptance_policy,
                "resolution_landing_disabled": bool(disable_resolution_landing),
                "canonical_answer_mode": canonical_answer_mode,
                "stall_policy": stall_policy,
                "unanimous_policy": unanimous_policy,
                "answer_selection_policy": answer_selection_policy,
                "answer_selection_report": answer_selection_report,
                "revised_traces": {
                    agent_id: {
                        "normalized_trace_text": trace.normalized_trace_text,
                        "steps": [step.text for step in trace.steps],
                    }
                    for agent_id, trace in revised_traces.items()
                },
            }
        )
        if next_responses:
            responses = next_responses
        if (
            (round_canonical_answer and canonical_answer_mode == "broadcast_stop")
            or (unanimous_policy == "skip" and _all_answers_unanimous(revised_answers))
            or (stalled_after_failed_rewrite and stall_policy == "stop")
        ):
            break

    final_round = round_results[-1] if round_results else {
        "revised_answers": {
            agent_id: _extract_answer_from_trace_text(text, question)
            for agent_id, text in initial_responses.items()
        },
        "chosen_answer": _majority_answer(
            {
                agent_id: _extract_answer_from_trace_text(text, question)
                for agent_id, text in initial_responses.items()
            }
        ),
        "graph_dossier": "",
        "resolutions": [],
        "raw_resolution_notes": [],
        "revised_traces": {},
    }
    return {
        "round_results": round_results,
        "revised_answers": final_round["revised_answers"],
        "chosen_answer": final_round["chosen_answer"],
        "graph_dossier": final_round["graph_dossier"],
        "resolutions": final_round["resolutions"],
        "raw_resolution_notes": final_round["raw_resolution_notes"],
        "revised_traces": final_round["revised_traces"],
        "claim_invariants": claim_invariants,
        "canonical_answer": canonical_answer,
        "supported_answer_memory": supported_answer_memory,
    }


def _build_summary(results: list[dict], key: str) -> tuple[int, float]:
    num_correct = sum(1 for item in results if item[key]["is_correct"])
    return num_correct, (num_correct / len(results) if results else 0.0)


def _default_graph_focus_round(graph_rounds: int) -> int:
    return min(2, max(int(graph_rounds), 0))


def _resolve_graph_focus_round(graph_rounds: int, requested_focus_round: int | None = None) -> int:
    if requested_focus_round is None:
        return _default_graph_focus_round(graph_rounds)
    return min(max(int(requested_focus_round), 0), max(int(graph_rounds), 0))


def _answers_match(left: str | None, right: str | None, question: str = "") -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if extract_option_map(question) and mcq_answers_match(left, right, question):
        return True
    left_math = normalize_math_answer(left)
    right_math = normalize_math_answer(right)
    if left_math is not None or right_math is not None:
        return left_math == right_math
    return normalize_short_answer(left) == normalize_short_answer(right)


def _is_answer_correct(answer: str | None, gold_answer: str, question: str = "") -> bool:
    if answer is None:
        return False
    if extract_option_map(question) and mcq_answers_match(answer, gold_answer, question):
        return True
    if isinstance(gold_answer, list):
        return short_answers_match(answer, gold_answer)
    if short_answers_match(answer, gold_answer):
        return True
    return is_math_correct(answer, gold_answer)


def _all_wrong_same(answer_map: dict[str, str | None], gold_answer: str, question: str = "") -> bool:
    answers = [answer for answer in answer_map.values() if answer is not None]
    if not answers:
        return False
    if any(_is_answer_correct(answer, gold_answer, question) for answer in answers):
        return False
    normalized = [normalize_math_answer(answer) for answer in answers]
    return len(set(normalized)) == 1


def _round_answer_maps(result_item: dict, key: str, num_rounds: int) -> tuple[list[dict[str, str | None]], list[str | None], list[dict | None]]:
    round_results = list(result_item[key].get("round_results", []))
    answer_maps = [copy.deepcopy(result_item["initial_answers"])]
    chosen_answers = [_majority_answer(answer_maps[0])]
    round_payloads: list[dict | None] = [None]
    current_answers = copy.deepcopy(answer_maps[0])
    current_choice = chosen_answers[0]
    for round_idx in range(1, num_rounds + 1):
        if round_idx <= len(round_results):
            round_payload = round_results[round_idx - 1]
            next_answers = round_payload.get("revised_answers") or {}
            current_answers = {
                **copy.deepcopy(current_answers),
                **copy.deepcopy(next_answers),
            }
            current_choice = round_payload.get("chosen_answer") or _majority_answer(current_answers)
            round_payloads.append(round_payload)
        else:
            round_payloads.append(None)
        answer_maps.append(copy.deepcopy(current_answers))
        chosen_answers.append(current_choice)
    return answer_maps, chosen_answers, round_payloads


def _build_focus_summary(results: list[dict], key: str, num_rounds: int, focus_round: int) -> tuple[int, float]:
    bounded_focus_round = min(max(int(focus_round), 0), max(int(num_rounds), 0))
    num_correct = 0
    for item in results:
        _, chosen_answers, _ = _round_answer_maps(item, key, num_rounds)
        if _is_answer_correct(
            chosen_answers[bounded_focus_round],
            item["gold_answer"],
            item.get("problem", ""),
        ):
            num_correct += 1
    return num_correct, (num_correct / len(results) if results else 0.0)


def _counter_to_dict(counter: Counter) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _build_mode_round_metrics(results: list[dict], key: str, num_rounds: int) -> dict:
    num_cases = len(results)
    num_round_slots = num_rounds + 1
    vote_acc = [0.0] * num_round_slots
    agent_acc = [0.0] * num_round_slots
    all_correct = [0.0] * num_round_slots
    all_wrong_same = [0.0] * num_round_slots
    flip_vote = [0.0] * num_round_slots
    flip_agent = [0.0] * num_round_slots
    wrong_to_right = [0.0] * num_round_slots
    right_to_wrong = [0.0] * num_round_slots
    wrong_majority_overturned = [0.0] * num_round_slots
    minority_correct_adopted = [0.0] * num_round_slots
    mutual_update_agent_rate = [0.0] * num_round_slots
    reason_driven_correction = [0.0] * num_round_slots
    verifier_signal_accuracy = [0.0] * num_round_slots
    verifier_revision_outcome_accuracy = [0.0] * num_round_slots
    verifier_signal_denominator = [0] * num_round_slots
    verifier_outcome_denominator = [0] * num_round_slots

    graph_debate_cases = [0] * num_round_slots
    graph_total_debates = [0] * num_round_slots
    graph_pairwise_debates = [0] * num_round_slots
    graph_structured_debates = [0] * num_round_slots
    graph_action_counts = [Counter() for _ in range(num_round_slots)]
    graph_winning_side_counts = [Counter() for _ in range(num_round_slots)]

    for result_item in results:
        gold_answer = result_item["gold_answer"]
        question = result_item.get("problem", "")
        answer_maps, chosen_answers, round_payloads = _round_answer_maps(result_item, key, num_rounds)
        agent_ids = sorted(answer_maps[0].keys())

        for round_idx in range(num_round_slots):
            answer_map = answer_maps[round_idx]
            chosen_answer = chosen_answers[round_idx]
            correctness = [_is_answer_correct(answer_map.get(agent_id), gold_answer, question) for agent_id in agent_ids]
            vote_acc[round_idx] += 1.0 if _is_answer_correct(chosen_answer, gold_answer, question) else 0.0
            agent_acc[round_idx] += sum(1.0 for item in correctness if item) / float(len(agent_ids))
            all_correct[round_idx] += 1.0 if all(correctness) else 0.0
            all_wrong_same[round_idx] += 1.0 if _all_wrong_same(answer_map, gold_answer, question) else 0.0
            if round_idx == 0:
                continue

            prev_answer_map = answer_maps[round_idx - 1]
            prev_chosen_answer = chosen_answers[round_idx - 1]
            prev_correctness = [_is_answer_correct(prev_answer_map.get(agent_id), gold_answer, question) for agent_id in agent_ids]
            changed_agents = 0
            wrong_to_right_count = 0
            right_to_wrong_count = 0
            for agent_id, prev_ok, curr_ok in zip(agent_ids, prev_correctness, correctness):
                if not _answers_match(prev_answer_map.get(agent_id), answer_map.get(agent_id), question):
                    changed_agents += 1
                if (not prev_ok) and curr_ok:
                    wrong_to_right_count += 1
                if prev_ok and (not curr_ok):
                    right_to_wrong_count += 1

            flip_vote[round_idx] += 1.0 if _is_answer_correct(prev_chosen_answer, gold_answer, question) and (not _is_answer_correct(chosen_answer, gold_answer, question)) else 0.0
            flip_agent[round_idx] += right_to_wrong_count / float(len(agent_ids))
            wrong_to_right[round_idx] += wrong_to_right_count / float(len(agent_ids))
            right_to_wrong[round_idx] += right_to_wrong_count / float(len(agent_ids))
            wrong_majority_overturned[round_idx] += 1.0 if (not _is_answer_correct(prev_chosen_answer, gold_answer, question)) and _is_answer_correct(chosen_answer, gold_answer, question) else 0.0
            had_correct_minority = any(prev_correctness) and (not _is_answer_correct(prev_chosen_answer, gold_answer, question))
            minority_correct_adopted[round_idx] += 1.0 if had_correct_minority and _is_answer_correct(chosen_answer, gold_answer, question) else 0.0
            mutual_update_agent_rate[round_idx] += (changed_agents / float(len(agent_ids))) if changed_agents >= 2 else 0.0

            round_payload = round_payloads[round_idx] or {}
            if key == "origin":
                revision_meta = round_payload.get("revision_metadata") or {}
                had_reason_signal = any(
                    str((meta or {}).get("verification", "")).strip() or str((meta or {}).get("incorrect_claim", "")).strip()
                    for meta in revision_meta.values()
                )
            else:
                had_reason_signal = bool(round_payload.get("resolutions") or round_payload.get("raw_resolution_notes"))
            if had_reason_signal and wrong_to_right_count > 0:
                reason_driven_correction[round_idx] += wrong_to_right_count / float(len(agent_ids))

            if key == "graph":
                raw_notes = round_payload.get("raw_resolution_notes") or []
                resolutions = round_payload.get("resolutions") or []
                if raw_notes or resolutions:
                    graph_debate_cases[round_idx] += 1
                total_debates = len(raw_notes) if raw_notes else len(resolutions)
                pairwise_debates = sum(
                    1
                    for item in (raw_notes or resolutions)
                    if str(item.get("divergence_id", "")).startswith("PAIR_")
                )
                graph_total_debates[round_idx] += total_debates
                graph_pairwise_debates[round_idx] += pairwise_debates
                graph_structured_debates[round_idx] += max(total_debates - pairwise_debates, 0)
                graph_action_counts[round_idx].update(
                    str(item.get("action", "")).strip()
                    for item in resolutions
                    if str(item.get("action", "")).strip()
                )
                graph_winning_side_counts[round_idx].update(
                    str(item.get("winning_side", "")).strip()
                    for item in resolutions
                    if str(item.get("winning_side", "")).strip()
                )
                chosen_correct = _is_answer_correct(chosen_answer, gold_answer, question)
                for item in resolutions:
                    winning_side = str(item.get("winning_side", "")).strip()
                    if winning_side:
                        verifier_signal_denominator[round_idx] += 1
                        verifier_signal_accuracy[round_idx] += 1.0 if chosen_correct else 0.0
                if resolutions:
                    verifier_outcome_denominator[round_idx] += 1
                    verifier_revision_outcome_accuracy[round_idx] += 1.0 if chosen_correct else 0.0

    if num_cases:
        vote_acc = [item / num_cases for item in vote_acc]
        agent_acc = [item / num_cases for item in agent_acc]
        all_correct = [item / num_cases for item in all_correct]
        all_wrong_same = [item / num_cases for item in all_wrong_same]
        flip_vote = [item / num_cases for item in flip_vote]
        flip_agent = [item / num_cases for item in flip_agent]
        wrong_to_right = [item / num_cases for item in wrong_to_right]
        right_to_wrong = [item / num_cases for item in right_to_wrong]
        wrong_majority_overturned = [item / num_cases for item in wrong_majority_overturned]
        minority_correct_adopted = [item / num_cases for item in minority_correct_adopted]
        mutual_update_agent_rate = [item / num_cases for item in mutual_update_agent_rate]
        reason_driven_correction = [item / num_cases for item in reason_driven_correction]

    verifier_suggestion_accuracy = []
    verifier_revise_outcome_accuracy = []
    for round_idx in range(num_round_slots):
        if verifier_signal_denominator[round_idx]:
            verifier_suggestion_accuracy.append(verifier_signal_accuracy[round_idx] / verifier_signal_denominator[round_idx])
        else:
            verifier_suggestion_accuracy.append(0.0)
        if verifier_outcome_denominator[round_idx]:
            verifier_revise_outcome_accuracy.append(verifier_revision_outcome_accuracy[round_idx] / verifier_outcome_denominator[round_idx])
        else:
            verifier_revise_outcome_accuracy.append(0.0)

    return {
        "vote_acc": vote_acc,
        "agent_acc": agent_acc,
        "all_correct_acc": all_correct,
        "all_wrong_same_acc": all_wrong_same,
        "flip_vote_acc": flip_vote,
        "flip_agent_acc": flip_agent,
        "wrong_to_right_acc": wrong_to_right,
        "right_to_wrong_acc": right_to_wrong,
        "wrong_majority_overturned_acc": wrong_majority_overturned,
        "minority_correct_adopted_acc": minority_correct_adopted,
        "reason_driven_correction_acc": reason_driven_correction,
        "mutual_update_agent_rate_acc": mutual_update_agent_rate,
        "verifier_suggestion_accuracy_acc": verifier_suggestion_accuracy,
        "verifier_revise_outcome_accuracy_acc": verifier_revise_outcome_accuracy,
        "graph_debate_cases": graph_debate_cases,
        "graph_total_debates": graph_total_debates,
        "graph_pairwise_debates": graph_pairwise_debates,
        "graph_structured_debates": graph_structured_debates,
        "graph_action_counts": [_counter_to_dict(counter) for counter in graph_action_counts],
        "graph_winning_side_counts": [_counter_to_dict(counter) for counter in graph_winning_side_counts],
    }


def _render_mode_metrics(label: str, metrics: dict) -> list[str]:
    lines: list[str] = []
    for round_idx, value in enumerate(metrics["vote_acc"]):
        lines.append(f"[{label}] Round {round_idx} Vote Acc.: {value:.4f}")
    for round_idx, value in enumerate(metrics["agent_acc"]):
        lines.append(f"[{label}] Round {round_idx} Agent Acc.: {value:.4f}")
    for round_idx, value in enumerate(metrics["all_correct_acc"]):
        lines.append(f"[{label}] Round {round_idx} All-Correct Acc.: {value:.4f}")
    for round_idx, value in enumerate(metrics["all_wrong_same_acc"]):
        lines.append(f"[{label}] Round {round_idx} All-Wrong-Same Acc.: {value:.4f}")
    for round_idx, value in enumerate(metrics["flip_vote_acc"]):
        lines.append(f"[{label}] Round {round_idx} Flip Correct->Wrong (Vote): {value:.4f}")
    for round_idx, value in enumerate(metrics["flip_agent_acc"]):
        lines.append(f"[{label}] Round {round_idx} Flip Correct->Wrong (Agent): {value:.4f}")
    for round_idx, value in enumerate(metrics["wrong_to_right_acc"]):
        lines.append(f"[{label}] Round {round_idx} Wrong->Right (Agent): {value:.4f}")
    for round_idx, value in enumerate(metrics["right_to_wrong_acc"]):
        lines.append(f"[{label}] Round {round_idx} Right->Wrong (Agent): {value:.4f}")
    for round_idx, value in enumerate(metrics["wrong_majority_overturned_acc"]):
        lines.append(f"[{label}] Round {round_idx} Wrong-Majority Overturned: {value:.4f}")
    for round_idx, value in enumerate(metrics["minority_correct_adopted_acc"]):
        lines.append(f"[{label}] Round {round_idx} Minority-Correct Adopted: {value:.4f}")
    for round_idx, value in enumerate(metrics["reason_driven_correction_acc"]):
        lines.append(f"[{label}] Round {round_idx} Reason-Driven Correction: {value:.4f}")
    for round_idx, value in enumerate(metrics["mutual_update_agent_rate_acc"]):
        lines.append(f"[{label}] Round {round_idx} Mutual-Update Agent Rate: {value:.4f}")
    for round_idx, value in enumerate(metrics["verifier_suggestion_accuracy_acc"]):
        lines.append(f"[{label}] Round {round_idx} Verifier Suggestion Acc.: {value:.4f}")
    for round_idx, value in enumerate(metrics["verifier_revise_outcome_accuracy_acc"]):
        lines.append(f"[{label}] Round {round_idx} Verifier-Recommended Revision Outcome Acc.: {value:.4f}")

    if any(metrics["graph_total_debates"]):
        for round_idx, value in enumerate(metrics["graph_debate_cases"]):
            lines.append(f"[{label}] Round {round_idx} Debate Cases: {value}")
        for round_idx, value in enumerate(metrics["graph_total_debates"]):
            lines.append(f"[{label}] Round {round_idx} Total Debates: {value}")
        for round_idx, value in enumerate(metrics["graph_pairwise_debates"]):
            lines.append(f"[{label}] Round {round_idx} Pairwise Fallback Debates: {value}")
        for round_idx, value in enumerate(metrics["graph_structured_debates"]):
            lines.append(f"[{label}] Round {round_idx} Structured Divergence Debates: {value}")
        for round_idx, value in enumerate(metrics["graph_action_counts"]):
            public_value = {
                _public_action_name(action): count
                for action, count in (value or {}).items()
            }
            lines.append(f"[{label}] Round {round_idx} Resolution Action counts: {public_value}")
        for round_idx, value in enumerate(metrics["graph_winning_side_counts"]):
            public_value = {
                _public_winning_side(side): count
                for side, count in (value or {}).items()
            }
            lines.append(f"[{label}] Round {round_idx} Winning side counts: {public_value}")
    return lines


def _build_log_text(args: argparse.Namespace, results: list[dict], origin_metrics: dict, graph_metrics: dict) -> str:
    lines = [
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Cases processed: {len(results)}",
        f"Cases path: {args.cases_path}",
        f"History path: {args.history_path}",
        f"Output path: {args.output_path}",
        f"Origin rounds: {args.origin_rounds}",
        f"Graph rounds: {args.graph_rounds}",
        f"Graph only: {bool(getattr(args, 'graph_only', False))}",
        f"Prompt profile: {getattr(args, 'prompt_profile', None)}",
        f"Divergence selection variant: {getattr(args, 'divergence_selection_variant', 'first_real')}",
        f"Rewrite context variant: {getattr(args, 'rewrite_context_variant', 'current_suffix')}",
        f"Graph format: {getattr(args, 'graph_format', 'natural')}",
        f"Divergence random seed: {getattr(args, 'divergence_random_seed', 7)}",
        f"Disable target-check landing: {bool(getattr(args, 'disable_target_check_landing', False))}",
        f"Disable resolution landing: {bool(getattr(args, 'disable_resolution_landing', False))}",
        f"Disable pairwise fallback: {bool(getattr(args, 'disable_pairwise_fallback', False))}",
        f"Strict JSON IO: {bool(getattr(args, 'strict_json_io', False))}",
        f"Resolution trace context: {_resolution_trace_context_name(getattr(args, 'resolution_trace_context', 'window'))}",
        f"Resolution prompt style: {_resolution_prompt_style_name(getattr(args, 'resolution_prompt_style', 'profile'))}",
        f"Resolution acceptance policy: {_resolution_acceptance_policy_name(getattr(args, 'resolution_acceptance_policy', 'guarded'))}",
        f"Answer selection policy: {_answer_selection_policy_name(getattr(args, 'answer_selection_policy', 'standard'))}",
        f"Canonical answer mode: {_canonical_answer_mode_name(getattr(args, 'canonical_answer_mode', 'broadcast_stop'))}",
        f"Stall policy: {_stall_policy_name(getattr(args, 'stall_policy', 'stop'))}",
        f"Unanimous policy: {_unanimous_policy_name(getattr(args, 'unanimous_policy', 'skip'))}",
        f"Case batch size: {args.case_batch_size}",
        f"GPU memory utilization: {args.gpu_memory_utilization}",
        f"Max model len: {getattr(args, 'max_model_len', None)}",
        "",
    ]
    lines.extend(_render_mode_metrics("ORIGINMAD", origin_metrics))
    lines.append("")
    lines.extend(_render_mode_metrics("NL_GRAPH", graph_metrics))
    return "\n".join(lines).strip() + "\n"


def _build_log_header(args: argparse.Namespace, total_cases: int) -> str:
    lines = [
        f"Run started: {datetime.now().isoformat(timespec='seconds')}",
        f"Cases path: {args.cases_path}",
        f"History path: {args.history_path}",
        f"Output path: {args.output_path}",
        f"Log path: {args.log_path}",
        f"Origin rounds: {args.origin_rounds}",
        f"Graph rounds: {args.graph_rounds}",
        f"Graph only: {bool(getattr(args, 'graph_only', False))}",
        f"Prompt profile: {getattr(args, 'prompt_profile', None)}",
        f"Divergence selection variant: {getattr(args, 'divergence_selection_variant', 'first_real')}",
        f"Rewrite context variant: {getattr(args, 'rewrite_context_variant', 'current_suffix')}",
        f"Graph format: {getattr(args, 'graph_format', 'natural')}",
        f"Divergence random seed: {getattr(args, 'divergence_random_seed', 7)}",
        f"Disable target-check landing: {bool(getattr(args, 'disable_target_check_landing', False))}",
        f"Disable resolution landing: {bool(getattr(args, 'disable_resolution_landing', False))}",
        f"Disable pairwise fallback: {bool(getattr(args, 'disable_pairwise_fallback', False))}",
        f"Strict JSON IO: {bool(getattr(args, 'strict_json_io', False))}",
        f"Resolution trace context: {_resolution_trace_context_name(getattr(args, 'resolution_trace_context', 'window'))}",
        f"Resolution prompt style: {_resolution_prompt_style_name(getattr(args, 'resolution_prompt_style', 'profile'))}",
        f"Resolution acceptance policy: {_resolution_acceptance_policy_name(getattr(args, 'resolution_acceptance_policy', 'guarded'))}",
        f"Answer selection policy: {_answer_selection_policy_name(getattr(args, 'answer_selection_policy', 'standard'))}",
        f"Canonical answer mode: {_canonical_answer_mode_name(getattr(args, 'canonical_answer_mode', 'broadcast_stop'))}",
        f"Stall policy: {_stall_policy_name(getattr(args, 'stall_policy', 'stop'))}",
        f"Unanimous policy: {_unanimous_policy_name(getattr(args, 'unanimous_policy', 'skip'))}",
        f"Case batch size: {args.case_batch_size}",
        f"GPU memory utilization: {args.gpu_memory_utilization}",
        f"Max model len: {getattr(args, 'max_model_len', None)}",
        f"Total planned cases: {total_cases}",
        "",
    ]
    return "\n".join(lines)


def _build_log_checkpoint(
    processed_cases: int,
    total_cases: int,
    origin_metrics: dict,
    graph_metrics: dict,
) -> str:
    lines = [
        "=" * 80,
        f"Checkpoint: {datetime.now().isoformat(timespec='seconds')}",
        f"Processed cases: {processed_cases}/{total_cases}",
        "",
    ]
    lines.extend(_render_mode_metrics("ORIGINMAD", origin_metrics))
    lines.append("")
    lines.extend(_render_mode_metrics("NL_GRAPH", graph_metrics))
    lines.append("")
    return "\n".join(lines)


def _write_log_header(log_path: Path, args: argparse.Namespace, total_cases: int) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        f.write(_build_log_header(args, total_cases))
        f.flush()


def _append_log_checkpoint(
    log_path: Path,
    processed_cases: int,
    total_cases: int,
    origin_metrics: dict,
    graph_metrics: dict,
) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(_build_log_checkpoint(processed_cases, total_cases, origin_metrics, graph_metrics))
        f.flush()


def run_compare_batch(args: argparse.Namespace) -> dict:
    cases = _load_json(Path(args.cases_path))
    case_ids = {
        item.strip()
        for item in str(getattr(args, "case_ids", "") or "").split(",")
        if item.strip()
    }
    if case_ids:
        cases = [case for case in cases if case.get("unique_id") in case_ids]
    history_records = _load_jsonl(Path(args.history_path))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    graph_focus_round = _resolve_graph_focus_round(
        args.graph_rounds,
        getattr(args, "graph_focus_round", None),
    )
    _write_log_header(log_path, args, len(cases))
    if not cases:
        origin_round_metrics = _build_mode_round_metrics([], "origin", args.origin_rounds)
        graph_round_metrics = _build_mode_round_metrics([], "graph", args.graph_rounds)
        payload = {
            "cases_path": args.cases_path,
            "history_path": args.history_path,
            "prompt_profile": getattr(args, "prompt_profile", None),
            "graph_only": bool(getattr(args, "graph_only", False)),
            "divergence_selection_variant": getattr(args, "divergence_selection_variant", "first_real"),
            "rewrite_context_variant": getattr(args, "rewrite_context_variant", "current_suffix"),
            "graph_format": getattr(args, "graph_format", "natural"),
            "graph_json_protocol": (
                "json_graph_and_json_io"
                if str(getattr(args, "graph_format", "natural") or "").strip().lower() == "json"
                else "natural_graph_and_natural_io"
            ),
            "resolution_landing_disabled": bool(getattr(args, "disable_resolution_landing", False)),
            "pairwise_fallback_disabled": bool(getattr(args, "disable_pairwise_fallback", False)),
            "strict_json_io": bool(getattr(args, "strict_json_io", False)),
            "resolution_trace_context": _resolution_trace_context_name(getattr(args, "resolution_trace_context", "window")),
            "resolution_prompt_style": _resolution_prompt_style_name(getattr(args, "resolution_prompt_style", "profile")),
            "resolution_acceptance_policy": _resolution_acceptance_policy_name(getattr(args, "resolution_acceptance_policy", "guarded")),
            "answer_selection_policy": _answer_selection_policy_name(getattr(args, "answer_selection_policy", "standard")),
            "canonical_answer_mode": _canonical_answer_mode_name(getattr(args, "canonical_answer_mode", "broadcast_stop")),
            "stall_policy": _stall_policy_name(getattr(args, "stall_policy", "stop")),
            "unanimous_policy": _unanimous_policy_name(getattr(args, "unanimous_policy", "skip")),
            "divergence_random_seed": getattr(args, "divergence_random_seed", 7),
            "num_cases": 0,
            "origin_rounds": args.origin_rounds,
            "graph_rounds": args.graph_rounds,
            "graph_focus_round": graph_focus_round,
            "case_batch_size": args.case_batch_size,
            "origin_num_correct": 0,
            "origin_accuracy": 0.0,
            "graph_final_num_correct": 0,
            "graph_final_accuracy": 0.0,
            "graph_num_correct": 0,
            "graph_accuracy": 0.0,
            "origin_round_metrics": origin_round_metrics,
            "graph_round_metrics": graph_round_metrics,
            "results": [],
        }
        public_payload = _public_payload_view(payload)
        output_path.write_text(json.dumps(public_payload, ensure_ascii=False, indent=2))
        _append_log_checkpoint(
            log_path=log_path,
            processed_cases=0,
            total_cases=0,
            origin_metrics=payload["origin_round_metrics"],
            graph_metrics=payload["graph_round_metrics"],
        )
        print(
            json.dumps(
                {
                    "processed_cases": 0,
                    "origin_num_correct": 0,
                    "origin_accuracy": 0.0,
                    "graph_num_correct": 0,
                    "graph_accuracy": 0.0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return public_payload
    runtime = load_runtime(_build_runtime_args(args))
    try:
        results = []

        for batch_start in range(0, len(cases), args.case_batch_size):
            batch_cases = cases[batch_start : batch_start + args.case_batch_size]
            batch_states = _prepare_batch_states(
                batch_cases,
                history_records,
                unanimous_policy=getattr(args, "unanimous_policy", "skip"),
            )
            for state in batch_states:
                state["disable_target_check_landing"] = bool(getattr(args, "disable_target_check_landing", False))
            if not bool(getattr(args, "graph_only", False)):
                _run_origin_rounds_batched(batch_states, runtime, args.origin_rounds)
            _run_graph_rounds_batched(
                batch_states,
                runtime,
                args.graph_rounds,
                getattr(args, "prompt_profile", None),
                divergence_selection_variant=getattr(args, "divergence_selection_variant", "first_real"),
                rewrite_context_variant=getattr(args, "rewrite_context_variant", "current_suffix"),
                graph_format=getattr(args, "graph_format", "natural"),
                divergence_random_seed=getattr(args, "divergence_random_seed", 7),
                disable_resolution_landing=bool(getattr(args, "disable_resolution_landing", False)),
                disable_pairwise_fallback=bool(getattr(args, "disable_pairwise_fallback", False)),
                strict_json_io=bool(getattr(args, "strict_json_io", False)),
                answer_selection_policy=getattr(args, "answer_selection_policy", "standard"),
                canonical_answer_mode=getattr(args, "canonical_answer_mode", "broadcast_stop"),
                stall_policy=getattr(args, "stall_policy", "stop"),
                unanimous_policy=getattr(args, "unanimous_policy", "skip"),
                resolution_trace_context=getattr(args, "resolution_trace_context", "window"),
                resolution_prompt_style=getattr(args, "resolution_prompt_style", "profile"),
                resolution_acceptance_policy=getattr(args, "resolution_acceptance_policy", "guarded"),
            )
            results.extend(
                _finalize_case_result(state, args.origin_rounds, args.graph_rounds)
                for state in batch_states
            )

            origin_num_correct, origin_accuracy = _build_summary(results, "origin")
            graph_final_num_correct, graph_final_accuracy = _build_summary(results, "graph")
            graph_num_correct, graph_accuracy = _build_focus_summary(
                results,
                "graph",
                args.graph_rounds,
                graph_focus_round,
            )
            origin_round_metrics = _build_mode_round_metrics(results, "origin", args.origin_rounds)
            graph_round_metrics = _build_mode_round_metrics(results, "graph", args.graph_rounds)
            payload = {
                "cases_path": args.cases_path,
                "history_path": args.history_path,
                "prompt_profile": getattr(args, "prompt_profile", None),
                "graph_only": bool(getattr(args, "graph_only", False)),
                "divergence_selection_variant": getattr(args, "divergence_selection_variant", "first_real"),
                "rewrite_context_variant": getattr(args, "rewrite_context_variant", "current_suffix"),
                "graph_format": getattr(args, "graph_format", "natural"),
                "graph_json_protocol": (
                    "json_graph_and_json_io"
                    if str(getattr(args, "graph_format", "natural") or "").strip().lower() == "json"
                    else "natural_graph_and_natural_io"
                ),
                "resolution_landing_disabled": bool(getattr(args, "disable_resolution_landing", False)),
                "pairwise_fallback_disabled": bool(getattr(args, "disable_pairwise_fallback", False)),
                "strict_json_io": bool(getattr(args, "strict_json_io", False)),
                "resolution_trace_context": _resolution_trace_context_name(getattr(args, "resolution_trace_context", "window")),
                "resolution_prompt_style": _resolution_prompt_style_name(getattr(args, "resolution_prompt_style", "profile")),
                "resolution_acceptance_policy": _resolution_acceptance_policy_name(getattr(args, "resolution_acceptance_policy", "guarded")),
                "answer_selection_policy": _answer_selection_policy_name(getattr(args, "answer_selection_policy", "standard")),
                "canonical_answer_mode": _canonical_answer_mode_name(getattr(args, "canonical_answer_mode", "broadcast_stop")),
                "stall_policy": _stall_policy_name(getattr(args, "stall_policy", "stop")),
                "unanimous_policy": _unanimous_policy_name(getattr(args, "unanimous_policy", "skip")),
                "divergence_random_seed": getattr(args, "divergence_random_seed", 7),
                "num_cases": len(results),
                "origin_rounds": args.origin_rounds,
                "graph_rounds": args.graph_rounds,
                "graph_focus_round": graph_focus_round,
                "case_batch_size": args.case_batch_size,
                "origin_num_correct": origin_num_correct,
                "origin_accuracy": origin_accuracy,
                "graph_final_num_correct": graph_final_num_correct,
                "graph_final_accuracy": graph_final_accuracy,
                "graph_num_correct": graph_num_correct,
                "graph_accuracy": graph_accuracy,
                "origin_round_metrics": origin_round_metrics,
                "graph_round_metrics": graph_round_metrics,
                "results": results,
            }
            public_payload = _public_payload_view(payload)
            output_path.write_text(json.dumps(public_payload, ensure_ascii=False, indent=2))
            _append_log_checkpoint(
                log_path=log_path,
                processed_cases=len(results),
                total_cases=len(cases),
                origin_metrics=origin_round_metrics,
                graph_metrics=graph_round_metrics,
            )
            print(
                json.dumps(
                    {
                        "processed_cases": len(results),
                        "origin_num_correct": origin_num_correct,
                        "origin_accuracy": origin_accuracy,
                        "graph_focus_round": graph_focus_round,
                        "graph_num_correct": graph_num_correct,
                        "graph_accuracy": graph_accuracy,
                        "graph_final_num_correct": graph_final_num_correct,
                        "graph_final_accuracy": graph_final_accuracy,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )

        return _public_payload_view(payload)
    finally:
        close_runtime(runtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases_path",
        type=str,
        default="/data/yyr/Graph_Debate/results/20260421_origin3_vs_onlygraphnext5/data/math.json",
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default="/data/yyr/Graph_Debate/results/20260421_origin3_vs_onlygraphnext5/history/math_500__qwen3_N=3_R=5_OR3_MM5_ORIGINMAD.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/yyr/Graph_Debate/Natural_Language_Graph_Debate/results/math500_compare_origin_vs_current_graph.json",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="/data/yyr/Graph_Debate/Natural_Language_Graph_Debate/results/math500_compare_origin_vs_current_graph.log",
    )
    parser.add_argument("--model", type=str, default="qwen3")
    parser.add_argument("--model_dir", type=str, default="/data/yyr/model")
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime_backend", type=str, default="vllm", choices=["vllm", "openai_api"])
    parser.add_argument("--api_base_url", type=str, default="")
    parser.add_argument("--api_beta_base_url", type=str, default="")
    parser.add_argument("--api_model", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_timeout", type=float, default=120)
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument(
        "--save_api_trace",
        action="store_true",
        help="Save per-stage OpenAI-compatible API messages, outputs, latency, finish reason, and usage in graph round artifacts. API keys are never logged.",
    )
    parser.add_argument("--origin_rounds", type=int, default=1)
    parser.add_argument("--graph_rounds", type=int, default=1)
    parser.add_argument(
        "--graph_focus_round",
        type=int,
        default=None,
        help="Round used for top-level graph_num_correct/graph_accuracy; defaults to round2 when available.",
    )
    parser.add_argument("--case_batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--disable_target_check_landing", action="store_true")
    parser.add_argument(
        "--disable_resolution_landing",
        action="store_true",
        help="Do not let resolution/canonical-answer shortcuts directly land the final response; useful for strict rewrite ablations.",
    )
    parser.add_argument(
        "--disable_pairwise_fallback",
        action="store_true",
        help="Do not run answer-disagreement pairwise fallback when structured divergence resolution has no actionable decision.",
    )
    parser.add_argument(
        "--strict_json_io",
        action="store_true",
        help="For graph_format=json, parse resolution decisions only from valid JSON fields instead of natural-language fallback.",
    )
    parser.add_argument(
        "--answer_selection_policy",
        type=str,
        default="standard",
        choices=["standard", "supported_answer_memory", "stable_majority_guard"],
        help="Final answer selection policy for NLGraph rounds.",
    )
    parser.add_argument(
        "--canonical_answer_mode",
        type=str,
        default="broadcast_stop",
        choices=["broadcast_stop", "select_only"],
        help=(
            "How resolution canonical answers affect later rounds. "
            "broadcast_stop preserves the legacy behavior; select_only uses the answer as this round's chosen answer "
            "without forcing every agent response to that answer or stopping solely because it appeared."
        ),
    )
    parser.add_argument(
        "--stall_policy",
        type=str,
        default="stop",
        choices=["stop", "continue"],
        help="Whether a failed/non-changing rewrite stops graph rounds or allows the configured rounds to continue.",
    )
    parser.add_argument(
        "--unanimous_policy",
        type=str,
        default="skip",
        choices=["skip", "audit"],
        help="Whether unanimous round0/current answers skip graph rounds or still undergo audit-style graph iterations.",
    )
    parser.add_argument(
        "--resolution_trace_context",
        type=str,
        default="window",
        choices=["window", "full"],
        help="How much normalized agent trace text to include in resolution analysis prompts.",
    )
    parser.add_argument(
        "--resolution_prompt_style",
        type=str,
        default="profile",
        choices=["profile", "minimal_strategy", "ledger_strategy"],
        help="Resolution prompt style. ledger_strategy adds a compact visible-evidence ledger before choosing.",
    )
    parser.add_argument(
        "--resolution_acceptance_policy",
        type=str,
        default="guarded",
        choices=["guarded", "trust_model"],
        help="Diagnostic policy for accepting model graph resolutions. trust_model skips local evidence/rewrite guards and prioritizes structured model decisions.",
    )
    parser.add_argument("--prompt_profile", type=str, default=None)
    parser.add_argument("--graph_only", action="store_true")
    parser.add_argument(
        "--divergence_selection_variant",
        type=str,
        default="first_real",
        choices=sorted(NaturalLanguageGraphDebatePipeline.DIVERGENCE_SELECTION_VARIANTS),
    )
    parser.add_argument(
        "--rewrite_context_variant",
        type=str,
        default="current_suffix",
        choices=sorted(NaturalLanguageGraphDebatePipeline.REWRITE_CONTEXT_VARIANTS),
    )
    parser.add_argument(
        "--graph_format",
        type=str,
        default="natural",
        choices=sorted(NaturalLanguageGraphDebatePipeline.GRAPH_FORMATS),
    )
    parser.add_argument("--divergence_random_seed", type=int, default=7)
    parser.add_argument("--case_ids", type=str, default="")
    args = parser.parse_args()

    payload = run_compare_batch(args)
    print(
        json.dumps(
            {
                "num_cases": payload["num_cases"],
                "origin_num_correct": payload["origin_num_correct"],
                "origin_accuracy": payload["origin_accuracy"],
                "graph_focus_round": payload.get("graph_focus_round"),
                "graph_num_correct": payload["graph_num_correct"],
                "graph_accuracy": payload["graph_accuracy"],
                "graph_final_num_correct": payload.get("graph_final_num_correct"),
                "graph_final_accuracy": payload.get("graph_final_accuracy"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(str(args.output_path), flush=True)


if __name__ == "__main__":
    main()
