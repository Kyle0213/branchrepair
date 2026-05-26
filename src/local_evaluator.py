from __future__ import annotations

import re
from typing import Optional

from .local_rm import extract_boxed_answer, grade_answer_mathd, grade_answer_sympy, mathd_normalize_answer

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_FINAL_ANSWER_RE = re.compile(r"\{[^{}]*final answer[^{}]*\}", flags=re.IGNORECASE)
_PLACEHOLDER_MATH_ANSWER_RE = re.compile(
    r"^(?:[\{\[\(]?\s*)?(?:x|y|z|n|m|k|a|b|c|\?+|placeholder|null|none|todo)(?:\s*[\}\]\)]?)$",
    flags=re.IGNORECASE,
)
_MATRIX_ENTRY_RE = re.compile(
    r"[-+]?\s*(?:\\frac\{[^{}]+\}\{[^{}]+\}|\d+(?:/\d+)?|\d+(?:\.\d+)?)"
)


def _last_number_in(text: str):
    matches = _NUM_RE.findall(text)
    return matches[-1] if matches else None


def _to_float(num_str: str) -> float:
    return float(num_str.replace(",", ""))


def extract_numeric_answer(text: str):
    clean = str(text or "").replace("\\", "").replace("\n", " ")
    final_braced = re.findall(r"\{[^{}]*final answer[^{}]*\}", clean, flags=re.IGNORECASE)
    for seg in final_braced[::-1]:
        num_str = _last_number_in(seg)
        if num_str is not None:
            return _to_float(num_str)
    boxed_matches = re.findall(r"boxed\{([^}]*)\}", clean, flags=re.IGNORECASE)
    for boxed in boxed_matches[::-1]:
        num_str = _last_number_in(boxed)
        if num_str is not None:
            return _to_float(num_str)
    final_matches = re.findall(
        r"final answer[^0-9\\-]*(-?\d+(?:,\d{3})*(?:\.\d+)?)(?=[^0-9.]|$)",
        clean,
        flags=re.IGNORECASE,
    )
    if final_matches:
        return _to_float(final_matches[-1])
    brace_matches = re.findall(r"\{([^{}]*)\}", clean)
    for brace in brace_matches[::-1]:
        num_str = _last_number_in(brace)
        if num_str is not None:
            return _to_float(num_str)
    num_str = _last_number_in(clean)
    return None if num_str is None else _to_float(num_str)


def extract_mcq_answer(text: str):
    clean = str(text or "").replace("\\", "").replace("\n", " ")
    final_braced = re.findall(r"\{[^{}]*final answer[^{}]*\}", clean, flags=re.IGNORECASE)
    for seg in final_braced[::-1]:
        seg_matches = re.findall(r"\(([A-J])\)", seg, flags=re.IGNORECASE)
        if seg_matches:
            return f"({seg_matches[-1].upper()})"
        seg_matches = re.findall(r"\b([A-J])\b", seg, flags=re.IGNORECASE)
        if seg_matches:
            return f"({seg_matches[-1].upper()})"
    final_matches = re.findall(r"final answer\s*[:\-]?\s*\(?\s*([A-J])\s*\)?", clean, flags=re.IGNORECASE)
    if final_matches:
        return f"({final_matches[-1].upper()})"
    answer_matches = re.findall(r"answer\s*[:\-]?\s*\(?\s*([A-J])\s*\)?", clean, flags=re.IGNORECASE)
    if answer_matches:
        return f"({answer_matches[-1].upper()})"
    brace_matches = re.findall(r"\{[^{}]*\}", clean)
    for seg in brace_matches[::-1]:
        seg_matches = re.findall(r"\(?([A-J])\)?", seg, flags=re.IGNORECASE)
        if seg_matches:
            return f"({seg_matches[-1].upper()})"
    tail = clean[-200:]
    tail_matches = re.findall(r"\(([A-J])\)", tail, flags=re.IGNORECASE)
    if tail_matches:
        return f"({tail_matches[-1].upper()})"
    return None


def _is_invalid_math_answer_text(text) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    compact = raw.rstrip("\\").strip()
    lowered = compact.lower()
    if lowered in {"[no_answer]", "no answer", "none", "null", "placeholder", "todo"}:
        return True
    if _PLACEHOLDER_MATH_ANSWER_RE.fullmatch(compact):
        return True
    if compact.startswith("{") and compact.endswith("}") and _PLACEHOLDER_MATH_ANSWER_RE.fullmatch(compact[1:-1].strip()):
        return True
    if any(token in lowered for token in ('"decision"', '"verification"', '"revised_response"', '"revised_final_answer"')):
        return True
    if "boxed" in lowered or "final answer" in lowered or "```" in compact:
        return True
    unescaped_dollars = re.findall(r"(?<!\\)\$", compact)
    if len(unescaped_dollars) % 2 == 1:
        return True
    for left, right in (("{", "}"), ("[", "]"), ("(", ")")):
        if compact.count(left) != compact.count(right):
            return True
    if re.search(r"\\(?:frac|sqrt)\{[^{}]*$", compact):
        return True
    return False


def extract_math_answer(text):
    clean = str(text or "").replace("\n", " ")
    if "\\boxed" in clean:
        boxed = extract_boxed_answer(clean)
        if boxed is not None:
            candidate = boxed.strip().rstrip("\\").strip()
            if not _is_invalid_math_answer_text(candidate):
                return candidate
    matches = _FINAL_ANSWER_RE.findall(clean)
    if matches:
        seg = matches[-1]
        inner = seg.strip("{}")
        parts = re.split(r"final answer\s*[:\-]?\s*", inner, flags=re.IGNORECASE)
        candidate = parts[-1].strip() if parts else inner.strip()
        if candidate and not _is_invalid_math_answer_text(candidate):
            return candidate.rstrip("\\").strip()
    tail = clean[-200:]
    tail_match = re.search(r"final answer\s*[:\-]?\s*(.+?)\s*$", tail, flags=re.IGNORECASE)
    if tail_match:
        candidate = tail_match.group(1).strip()
        if candidate and not _is_invalid_math_answer_text(candidate):
            return candidate.rstrip("\\").strip()
    return None


def normalize_math_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    text = str(answer).replace("π", r"\pi").strip().strip("$").strip().rstrip("\\").strip()
    text = re.sub(r"[.。]+$", "", text).strip()
    text = re.sub(
        r"^(?:the\s+)?(?:requested\s+)?(?:final\s+)?(?:answer|object|value)\s+(?:is|=)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(r"^is\s+", "", text, flags=re.IGNORECASE).strip()
    math_segments = re.findall(r"\$([^$]+)\$", text)
    if math_segments and re.search(r"[A-Za-z]", re.sub(r"\$[^$]+\$", "", text)):
        text = math_segments[-1].strip()
    prose_tail = re.search(
        r"(?:is|equals|=)\s*(\\frac\{[^{}]+\}\{[^{}]+\}|-?\d+(?:/\d+)?(?:\s*\\pi)?|-?\d*\\pi)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if prose_tail and re.search(r"[A-Za-z]", text[: prose_tail.start()]):
        text = prose_tail.group(1).strip()
    text = text.strip().strip("$").strip()
    if _is_invalid_math_answer_text(text):
        return None
    assign_match = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", text)
    if assign_match:
        text = assign_match.group(1).strip()
    if _is_invalid_math_answer_text(text):
        return None
    normalized = mathd_normalize_answer(text)
    if normalized is not None:
        normalized = normalized.rstrip("\\").strip()
        return None if _is_invalid_math_answer_text(normalized) else normalized
    return None if _is_invalid_math_answer_text(text) else text


def _simple_matrix_entries(text: str) -> list[str] | None:
    raw = str(text or "")
    if not re.search(r"\\begin\{[pbv]?matrix\}", raw):
        return None
    entries = []
    for match in _MATRIX_ENTRY_RE.finditer(raw):
        entry = re.sub(r"\s+", "", match.group(0))
        normalized = normalize_math_answer(entry)
        entries.append(normalized or entry)
    return entries or None


def _entries_are_equal(given_entries: list[str], gold_entries: list[str]) -> bool:
    if len(given_entries) != len(gold_entries):
        return False
    for given, gold in zip(given_entries, gold_entries):
        if not (grade_answer_mathd(given, gold) or grade_answer_sympy(given, gold)):
            return False
    return True


def _is_negative_identity_entries(entries: list[str] | None) -> bool:
    if not entries:
        return False
    size = int(len(entries) ** 0.5)
    if size * size != len(entries):
        return False
    for idx, entry in enumerate(entries):
        row, col = divmod(idx, size)
        expected = "-1" if row == col else "0"
        if not (grade_answer_mathd(entry, expected) or grade_answer_sympy(entry, expected)):
            return False
    return True


def _is_negative_identity_symbol(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or "")).lower()
    compact = compact.replace("\\mathbf{i}", "i").replace("\\mathrm{i}", "i")
    return compact in {"-i", "-\\i"}


def _bare_comma_number_set(text: str) -> list[str] | None:
    raw = str(text or "").strip().strip("$")
    if any(ch in raw for ch in "()[]{}"):
        return None
    if "," not in raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) < 2 or any(not part for part in parts):
        return None
    normalized = [normalize_math_answer(part) for part in parts]
    if any(part is None for part in normalized):
        return None
    if not all(re.fullmatch(r"[-+]?\d+(?:/\d+)?|[-+]?\\frac\{[^{}]+\}\{[^{}]+\}", part or "") for part in normalized):
        return None
    return sorted(normalized or [])


def is_math_correct(given_answer, ground_truth) -> bool:
    if given_answer in (None, ""):
        return False
    given_text = str(given_answer).strip()
    gold_text = str(ground_truth).strip()
    given_matrix = _simple_matrix_entries(given_text)
    gold_matrix = _simple_matrix_entries(gold_text)
    if given_matrix and gold_matrix and _entries_are_equal(given_matrix, gold_matrix):
        return True
    if _is_negative_identity_symbol(given_text) and _is_negative_identity_entries(gold_matrix):
        return True
    given_set = _bare_comma_number_set(given_text)
    gold_set = _bare_comma_number_set(gold_text)
    if given_set and gold_set and given_set == gold_set:
        return True
    try:
        return bool(grade_answer_mathd(given_text, gold_text) or grade_answer_sympy(given_text, gold_text))
    except Exception:
        return False
