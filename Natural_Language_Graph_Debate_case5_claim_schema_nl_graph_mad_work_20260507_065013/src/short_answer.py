from __future__ import annotations

import re
from collections.abc import Iterable


def question_requests_short_answer(question: str) -> bool:
    lowered = str(question or "").lower()
    return any(
        marker in lowered
        for marker in (
            "answer with the short answer",
            "use a short answer",
            "short answer",
            "answer with the answer phrase",
            "answer with the entity",
        )
    )


def normalize_short_answer(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\\&", "&")
    text = re.sub(r"\\(?:mathrm|text)\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"^\{?\s*final answer\s*[:=\-]?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("{}[]()").strip()
    text = re.sub(
        r"^(?:the\s+)?(?:answer|short answer|answer phrase|short answer phrase|final answer)\s*(?:is|=|:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().strip("\"'`“”‘’").strip()
    text = re.sub(r"^[\s:=-]+", "", text).strip()
    text = re.sub(r"[.。!?,;:]+$", "", text).strip()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _clean_short_candidate(text: str) -> str | None:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    candidate = candidate.strip().strip("{}[]()").strip()
    candidate = candidate.strip("\"'`“”‘’").strip()
    candidate = re.sub(r"[.。!?,;:]+$", "", candidate).strip()
    candidate = re.sub(
        r"^(?:the\s+)?(?:answer|short answer|answer phrase|short answer phrase|final answer)\s*(?:is|=|:|was|were)\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"^(?:is|are|was|were)\s+", "", candidate, flags=re.IGNORECASE)
    numeric_with_unit = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s+[A-Za-z][A-Za-z\s/-]*", candidate)
    if numeric_with_unit:
        candidate = numeric_with_unit.group(1)
    candidate = re.sub(r"^[\s:=-]+", "", candidate).strip()
    return candidate if normalize_short_answer(candidate) else None


def short_answers_match(given, gold) -> bool:
    given_norm = normalize_short_answer(given)
    if not given_norm:
        return False
    if isinstance(gold, str):
        gold_values = [gold]
    elif isinstance(gold, Iterable):
        gold_values = list(gold)
    else:
        gold_values = [gold]
    gold_norms = [normalize_short_answer(item) for item in gold_values]
    gold_norms = [item for item in gold_norms if item]
    return any(given_norm == gold_norm for gold_norm in gold_norms)


def extract_short_answer(text: str) -> str | None:
    clean = str(text or "").replace("\n", " ").strip()
    if not clean:
        return None

    final_braced = re.findall(r"\{[^{}]*final answer[^{}]*\}", clean, flags=re.IGNORECASE)
    for seg in final_braced[::-1]:
        boxed = re.search(r"\\?boxed\{\s*([^{}]+?)\s*\}", seg, flags=re.IGNORECASE)
        if boxed:
            candidate = _clean_short_candidate(boxed.group(1))
            if candidate:
                return candidate
        direct = re.search(r"final answer\s*[:=\-]?\s*(?:\\?boxed\{\s*)?(.+?)(?:\s*\})?\s*$", seg.strip("{} "), flags=re.IGNORECASE)
        if direct:
            candidate = _clean_short_candidate(direct.group(1))
            if candidate:
                return candidate

    tail = clean[-400:]
    patterns = [
        r"final answer\s*[:=\-]?\s*(?:\\?boxed\{\s*)?(.+?)(?:\s*\})?\s*$",
        r"(?:therefore\s+)?(?:the\s+)?(?:short\s+)?answer\s+(?:is|=)\s*(?:\\?boxed\{\s*)?(.+?)(?:\s*\})?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, tail, flags=re.IGNORECASE)
        if match:
            candidate = _clean_short_candidate(match.group(1))
            if candidate:
                return candidate

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        candidate = _clean_short_candidate(last)
        if candidate and len(candidate.split()) <= 8:
            return candidate
    return None
