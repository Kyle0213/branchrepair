from __future__ import annotations

import re
from typing import List

from .models import NaturalLanguageStep


_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)")
_INLINE_STEP_BREAK_RE = re.compile(r"(?<=[\.\?!;])\s+(?=(?:[-*]|\d+[.)]|[A-Za-z0-9$\(\[\{]))")


def clean_step_line(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    text = _LIST_PREFIX_RE.sub("", text)
    return text.strip()


def coarse_split_lines(text: str) -> List[str]:
    return [clean_step_line(line) for line in str(text or "").splitlines() if clean_step_line(line)]


def split_single_line_into_steps(text: str) -> List[str]:
    raw = clean_step_line(text)
    if not raw:
        return []
    pieces = re.split(_INLINE_STEP_BREAK_RE, raw)
    cleaned = [clean_step_line(piece) for piece in pieces if clean_step_line(piece)]
    return cleaned or [raw]


def split_natural_language_steps(text: str) -> List[str]:
    lines = coarse_split_lines(text)
    if len(lines) > 1:
        return lines
    if not lines:
        return []
    return split_single_line_into_steps(lines[0])


def build_step_objects(agent_id: str, text: str) -> List[NaturalLanguageStep]:
    steps = []
    for idx, line in enumerate(split_natural_language_steps(text), start=1):
        steps.append(
            NaturalLanguageStep(
                step_id=f"{agent_id}.s{idx}",
                agent_id=agent_id,
                index=idx,
                text=line,
                source_lines=[idx],
            )
        )
    return steps


def _copy_step(step: NaturalLanguageStep, *, text: str | None = None, source_lines: List[int] | None = None) -> NaturalLanguageStep:
    return NaturalLanguageStep(
        step_id=step.step_id,
        agent_id=step.agent_id,
        index=step.index,
        text=step.text if text is None else text,
        source_lines=list(step.source_lines if source_lines is None else source_lines),
    )


def _build_compacted_steps(agent_id: str, items: List[tuple[str, List[int]]]) -> List[NaturalLanguageStep]:
    compacted: List[NaturalLanguageStep] = []
    for idx, (text, source_lines) in enumerate(items, start=1):
        compacted.append(
            NaturalLanguageStep(
                step_id=f"{agent_id}.s{idx}",
                agent_id=agent_id,
                index=idx,
                text=text,
                source_lines=source_lines,
            )
        )
    return compacted


def _step_starts_candidate_check(text: str) -> bool:
    lowered = clean_step_line(text).lower()
    return bool(
        re.match(r"^\d+\s*[x×]\s*\d+\s*=", lowered)
        or re.match(r"^(?:check|try|consider)\s+.+\bmultiple of 30\b", lowered)
        or re.match(r"^(?:check|try|consider)\s+\d+\b", lowered)
        or re.match(r"^(?:first|next)\s+multiple\b.*\b\d+\b", lowered)
        or re.match(r"^\d+\s+is\s+a\s+multiple of 30\b", lowered)
        or "candidate" in lowered
    )


def _step_is_bare_candidate_outcome(text: str) -> bool:
    lowered = clean_step_line(text).lower()
    if not re.match(r"^\d+\b", lowered):
        return False
    markers = (
        "contains digit",
        "contains the digit",
        "is not allowed",
        "not allowed",
        "fails",
        "forbidden digit",
        "works",
        "digits are valid",
        "contains only",
        "is a multiple of 30",
        "is not a multiple of 30",
    )
    return any(marker in lowered for marker in markers)


def _step_is_validation_outcome(text: str) -> bool:
    lowered = clean_step_line(text).lower()
    markers = (
        "contains digit",
        "contains the digit",
        "contains only",
        "contains only digits",
        "contains only the digits",
        "digits are valid",
        "digits are ",
        "is not allowed",
        "not allowed",
        "is valid",
        "works",
        "fails",
        "is a multiple of",
        "is not a multiple of",
        "divisible",
        "satisfies",
    )
    return any(marker in lowered for marker in markers)


def _step_has_conflicting_validation(text: str) -> bool:
    lowered = clean_step_line(text).lower()
    positive_markers = (
        "contains only",
        "digits are valid",
        "is valid",
        "works",
        "satisfies",
    )
    negative_markers = (
        "contains digit",
        "contains the digit",
        "is not allowed",
        "not allowed",
        "fails",
        "forbidden digit",
        "is not a multiple of",
    )
    return any(marker in lowered for marker in positive_markers) and any(marker in lowered for marker in negative_markers)


def _step_is_search_failure(text: str) -> bool:
    lowered = clean_step_line(text).lower()
    failure_markers = (
        "contains digit",
        "contains the digit",
        "is not allowed",
        "not allowed",
        "is not a multiple of",
        "fails",
        "forbidden digit",
    )
    starts_candidate = _step_starts_candidate_check(lowered) or _step_is_bare_candidate_outcome(lowered)
    return starts_candidate and not _step_has_conflicting_validation(lowered) and any(marker in lowered for marker in failure_markers)


def _step_candidate_range_token(text: str) -> str:
    cleaned = clean_step_line(text)
    check_match = re.match(r"^(?:check|try|consider)\s+(-?\d+)\b", cleaned, flags=re.IGNORECASE)
    if check_match:
        return check_match.group(1)
    leading_match = re.match(r"^(-?\d+)\b", cleaned)
    if leading_match:
        return leading_match.group(1)
    mult_match = re.search(r"=\s*(-?\d+)\b", cleaned)
    if mult_match:
        return mult_match.group(1)
    trailing_match = re.search(r":\s*(-?\d+)\b", cleaned)
    if trailing_match:
        return trailing_match.group(1)
    int_matches = re.findall(r"-?\d+\b", cleaned)
    return int_matches[-1] if int_matches else "unknown"


def _summarize_search_failure_steps(steps: List[NaturalLanguageStep]) -> str:
    start = _step_candidate_range_token(steps[0].text)
    end = _step_candidate_range_token(steps[-1].text)
    lowered = " ".join(step.text.lower() for step in steps)
    if "is not a multiple of" in lowered and ("contains digit" in lowered or "contains the digit" in lowered):
        reason = "was not a multiple of 30 or contained a forbidden digit"
    elif "contains digit" in lowered or "contains the digit" in lowered:
        reason = "contained a forbidden digit"
    elif "is not a multiple of" in lowered:
        reason = "was not a multiple of 30"
    else:
        reason = "failed the local candidate check"
    return f"Checked candidates from {start} through {end} in increasing order; each {reason}."


def _step_grouping_terminal_value(text: str) -> str:
    cleaned = clean_step_line(text)
    lowered = cleaned.lower()
    if "grouping" not in lowered and "parenthes" not in lowered:
        return ""
    match = re.search(r"=\s*(-?\d+(?:\.\d+)?)\s*\$?\.?\s*$", cleaned)
    return match.group(1) if match else ""


def _step_is_unresolved_candidate_tail(text: str) -> bool:
    cleaned = clean_step_line(text)
    if cleaned.lower().startswith("checked candidates from "):
        return False
    if not _step_starts_candidate_check(cleaned):
        return False
    if _step_is_validation_outcome(cleaned) or _step_has_conflicting_validation(cleaned):
        return False
    return True


def _step_looks_truncated(text: str) -> bool:
    cleaned = clean_step_line(text)
    if not cleaned:
        return True
    if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
        return True
    if cleaned.endswith(("=", "+", "-", "*", "/", "×", "(", "[", "{", "$", ":")):
        return True
    if cleaned.count("$") % 2 == 1:
        return True
    for left, right in (("(", ")"), ("[", "]"), ("{", "}")):
        if cleaned.count(left) > cleaned.count(right):
            return True
    return False


def _prune_incomplete_tail_steps(steps: List[NaturalLanguageStep]) -> List[NaturalLanguageStep]:
    pruned = list(steps)
    while pruned:
        tail_text = pruned[-1].text
        if _step_looks_truncated(tail_text) or _step_is_unresolved_candidate_tail(tail_text):
            pruned.pop()
            continue
        break
    return _build_compacted_steps(
        pruned[0].agent_id if pruned else steps[0].agent_id,
        [(step.text, list(step.source_lines)) for step in pruned],
    )


def compact_step_objects(
    agent_id: str,
    steps: List[NaturalLanguageStep],
    *,
    search_failure_min_run: int = 8,
    grouping_min_run: int = 4,
) -> List[NaturalLanguageStep]:
    if not steps:
        return []

    merged_headers: List[NaturalLanguageStep] = []
    index = 0
    while index < len(steps):
        current = steps[index]
        current_text = clean_step_line(current.text)
        if (
            current_text.endswith(":")
            and index + 1 < len(steps)
            and "final answer" not in current_text.lower()
        ):
            next_step = steps[index + 1]
            merged_headers.append(
                _copy_step(
                    current,
                    text=f"{current_text} {clean_step_line(next_step.text)}",
                    source_lines=current.source_lines + next_step.source_lines,
                )
            )
            index += 2
            continue
        merged_headers.append(_copy_step(current, text=current_text))
        index += 1

    merged_candidates: List[NaturalLanguageStep] = []
    index = 0
    while index < len(merged_headers):
        current = merged_headers[index]
        if _step_starts_candidate_check(current.text) and index + 1 < len(merged_headers):
            merged_text = current.text
            source_lines = list(current.source_lines)
            next_index = index + 1
            consumed = False
            while (
                next_index < len(merged_headers)
                and not _step_starts_candidate_check(merged_headers[next_index].text)
                and _step_is_validation_outcome(merged_headers[next_index].text)
            ):
                merged_text = f"{merged_text} {clean_step_line(merged_headers[next_index].text)}"
                source_lines.extend(merged_headers[next_index].source_lines)
                next_index += 1
                consumed = True
            if consumed:
                merged_candidates.append(_copy_step(current, text=merged_text, source_lines=source_lines))
                index = next_index
                continue
        merged_candidates.append(current)
        index += 1

    compressed_search: List[NaturalLanguageStep] = []
    index = 0
    while index < len(merged_candidates):
        if not _step_is_search_failure(merged_candidates[index].text):
            compressed_search.append(merged_candidates[index])
            index += 1
            continue
        run_end = index
        while run_end < len(merged_candidates) and _step_is_search_failure(merged_candidates[run_end].text):
            run_end += 1
        run = merged_candidates[index:run_end]
        if len(run) >= search_failure_min_run:
            compressed_search.append(
                _copy_step(
                    run[0],
                    text=_summarize_search_failure_steps(run),
                    source_lines=[line for step in run for line in step.source_lines],
                )
            )
        else:
            compressed_search.extend(run)
        index = run_end

    compacted_items: List[tuple[str, List[int]]] = []
    index = 0
    while index < len(compressed_search):
        current = compressed_search[index]
        value = _step_grouping_terminal_value(current.text)
        if not value:
            compacted_items.append((current.text, list(current.source_lines)))
            index += 1
            continue
        run_end = index
        while run_end < len(compressed_search) and _step_grouping_terminal_value(compressed_search[run_end].text) == value:
            run_end += 1
        run = compressed_search[index:run_end]
        if len(run) >= grouping_min_run:
            compacted_items.append((run[0].text, list(run[0].source_lines)))
            compacted_items.append((run[1].text, list(run[1].source_lines)))
            compacted_items.append(
                (
                    f"Several other parenthesizations or groupings also evaluate to {value}.",
                    [line for step in run[2:] for line in step.source_lines],
                )
            )
        else:
            compacted_items.extend((step.text, list(step.source_lines)) for step in run)
        index = run_end

    compacted_steps = _build_compacted_steps(agent_id, compacted_items)
    return _prune_incomplete_tail_steps(compacted_steps)
