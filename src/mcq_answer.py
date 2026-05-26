from __future__ import annotations

import re

from .short_answer import extract_short_answer, normalize_short_answer, short_answers_match


def question_requests_option_text_answer(question: str) -> bool:
    lowered = str(question or "").lower()
    if "option number" in lowered or "use only a number" in lowered:
        return False
    return "options:" in lowered and (
        "option text" in lowered
        or "option letter" in lowered
        or "answer with the option" in lowered
    )


def extract_option_map(question: str) -> dict[str, str]:
    option_map: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*([A-J])[\.)]\s+(.+?)\s*$", str(question or "")):
        option_map[match.group(1).upper()] = match.group(2).strip()
    return option_map


def _unique_option_text_mention(raw: str, option_map: dict[str, str]) -> str | None:
    if not option_map:
        return None
    search_spans = []
    text = str(raw or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        search_spans.append(lines[-1])
    search_spans.append(text[-500:])
    search_spans.append(text)
    for span in search_spans:
        span_norm = normalize_short_answer(span)
        if not span_norm:
            continue
        matches = []
        for option_text in option_map.values():
            option_norm = normalize_short_answer(option_text)
            if option_norm and re.search(rf"(?<!\w){re.escape(option_norm)}(?!\w)", span_norm):
                matches.append(option_text)
        unique = list(dict.fromkeys(matches))
        if len(unique) == 1:
            return unique[0]
    return None


def canonical_mcq_answer(text: str, question: str = "") -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    option_map = extract_option_map(question)

    stripped = raw.strip().strip("()[]{}").strip().upper()
    if len(stripped) == 1 and stripped in option_map:
        return option_map[stripped]

    boxed_letter = re.search(r"\\?boxed\{\s*([A-J])\s*\}?", raw, flags=re.IGNORECASE)
    if boxed_letter:
        key = boxed_letter.group(1).upper()
        if key in option_map:
            return option_map[key]
        return key

    leading_letter = re.search(r"^\s*[:\-–—]?\s*([A-J])[\.)]\s*(.+?)\s*$", raw, flags=re.IGNORECASE | re.DOTALL)
    if leading_letter:
        key = leading_letter.group(1).upper()
        tail = leading_letter.group(2).strip()
        if key in option_map:
            option_text = option_map[key]
            if not tail or short_answers_match(tail, option_text) or option_text.lower() in tail.lower():
                return option_text
        if tail:
            return tail

    candidate = extract_short_answer(raw) or raw
    candidate_norm = normalize_short_answer(candidate)
    if candidate_norm:
        for letter, option_text in option_map.items():
            if short_answers_match(candidate_norm, option_text):
                return option_text
        mentioned = _unique_option_text_mention(raw, option_map)
        if mentioned:
            return mentioned

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        line_match = re.search(
            r"(?:^|[:;\-–—]\s*|\s+)([A-J])[\.)]\s*(.+?)\s*$",
            last,
            flags=re.IGNORECASE,
        )
        if line_match:
            key = line_match.group(1).upper()
            tail = line_match.group(2).strip()
            if key in option_map:
                option_text = option_map[key]
                if not tail or short_answers_match(tail, option_text) or option_text.lower() in tail.lower():
                    return option_text
            if tail:
                return tail

    if re.fullmatch(r"\(?[A-J]\)?", raw.strip(), flags=re.IGNORECASE):
        key = raw.strip().strip("()").upper()
        if key in option_map:
            return option_map[key]
        return key
    tail_letter = re.search(
        r"(?:final answer|answer|option)\s*[:=\-]?\s*(?:\\?boxed\{\s*)?\(?([A-J])\)?\s*\}?\s*$",
        raw,
        flags=re.IGNORECASE,
    )
    if tail_letter:
        key = tail_letter.group(1).upper()
        if key in option_map:
            return option_map[key]
        return key
    return candidate if candidate_norm else None


def mcq_answers_match(given, gold, question: str = "") -> bool:
    given_canonical = canonical_mcq_answer(str(given or ""), question)
    if given_canonical is None:
        return False
    gold_canonical = canonical_mcq_answer(str(gold or ""), question) or gold
    return short_answers_match(given_canonical, gold_canonical)
