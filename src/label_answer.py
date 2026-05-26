from __future__ import annotations

import re


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean(text).lower())


def _extract_option_map(question: str) -> dict[str, str]:
    text = str(question or "")
    option_map: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*(\d{1,2})\.\s*(.+?)\s*$", text):
        option_map[match.group(1)] = match.group(2).strip()
    return option_map


def _extract_label_descriptions(question: str) -> dict[str, str]:
    text = _clean(question)
    tail = text.split("Answer", 1)[1] if "Answer" in text else text
    descriptions: dict[str, str] = {}
    lowered = text.lower()
    if re.search(r"\banswer\s+(?:with\s+)?yes\s+or\s+no\b|\buse\s+only\s+yes\s+or\s+no\b", lowered):
        descriptions["yes"] = "yes"
        descriptions["no"] = "no"
    if re.search(r"\banswer\s+(?:with\s+)?true\s+or\s+false\b|\buse\s+only\s+true\s+or\s+false\b", lowered):
        descriptions["true"] = "true"
        descriptions["false"] = "false"
    chained_pattern = (
        r"(?:^|,\s*|\band\s+)"
        r"(\d{1,2})\s+(?:for|if)\s+"
        r"(.+?)(?=(?:,\s*\d{1,2}\s+(?:for|if)\s+)|(?:\s+and\s+\d{1,2}\s+(?:for|if)\s+)|(?:\.\s*Use\b)|[.;]|$)"
    )
    for match in re.finditer(chained_pattern, tail, flags=re.IGNORECASE):
        label = match.group(1).strip()
        desc = match.group(2).strip().strip(" .")
        if desc:
            descriptions[label] = desc
    patterns = [
        r"Answer\s+(\d{1,2})\s+(?:for|if)\s+(.+?)(?=(?:,\s*\d{1,2}\s+(?:for|if)\s+)|(?:\s+and\s+\d{1,2}\s+(?:for|if)\s+)|(?:\.\s*Use\b)|[.;])",
        r"Answer\s+(\d{1,2})\s+(?:for|if)\s+(.+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = match.group(1).strip()
            desc = match.group(2).strip().strip(" .")
            if desc and label not in descriptions:
                descriptions[label] = desc
    return descriptions


def truth_value_target_statement(question: str) -> str:
    labels = set(allowed_label_answers(question))
    if labels != {"true", "false"}:
        return ""
    text = _clean(question)
    text = re.split(
        r"\bAnswer\s+(?:with\s+)?true\s+or\s+false\b|\bUse\s+only\s+true\s+or\s+false\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if "Question:" in text:
        text = text.rsplit("Question:", 1)[-1].strip()
    match = re.search(
        r"(?:is\s+the\s+following\s+statement\s+true\s+or\s+false\??|statement\s*[:\-])\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip()
    text = re.sub(r"^(?:is|whether)\s+", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip(" ?.")
    return text if 0 < len(text) <= 180 else ""


def truth_value_label_supported_by_claim(claim: str, question: str = "") -> str | None:
    target = truth_value_target_statement(question)
    if not target:
        return None
    text = _clean(claim)
    lowered = text.lower()
    if re.search(r"\b(?:the\s+)?(?:statement|target statement)\b[^.\n]{0,120}?\bis\s+not\s+true\b", lowered):
        return "false"
    if re.search(r"\b(?:the\s+)?(?:statement|target statement)\b[^.\n]{0,120}?\bis\s+not\s+false\b", lowered):
        return "true"
    if re.search(r"\b(?:the\s+)?(?:statement|target statement)\b[^.\n]{0,120}?\bis\s+false\b", lowered):
        return "false"
    if re.search(r"\b(?:the\s+)?(?:statement|target statement)\b[^.\n]{0,120}?\bis\s+true\b", lowered):
        return "true"
    compact_target = _compact(target)
    compact_claim = _compact(text)
    if compact_target and compact_target in compact_claim:
        if any(marker in lowered for marker in ("not prove", "not proven", "unsupported", "does not support", "fails to prove")):
            return None
        return "true"
    return None


def _nli_label_map(question: str) -> dict[str, str]:
    desc_map = _extract_label_descriptions(question)
    if not desc_map:
        return {}
    out: dict[str, str] = {}
    for label, desc in desc_map.items():
        lowered = _clean(desc).lower()
        if "entail" in lowered:
            out["entailment"] = label
        elif "neutral" in lowered:
            out["neutral"] = label
        elif "contrad" in lowered:
            out["contradiction"] = label
    return out if {"entailment", "neutral", "contradiction"}.issubset(out) else {}


def _truth_status_label_map(question: str) -> dict[str, str]:
    desc_map = _extract_label_descriptions(question)
    if not desc_map:
        return {}
    out: dict[str, str] = {}
    for label, desc in desc_map.items():
        lowered = _clean(desc).lower().strip(" ,.;")
        if "unknown" in lowered or "uncertain" in lowered or "undetermined" in lowered:
            out["unknown"] = label
        elif "false" in lowered:
            out["false"] = label
        elif "true" in lowered:
            out["true"] = label
    return out if {"true", "false", "unknown"}.issubset(out) else {}


def truth_status_label_supported_by_claim(claim: str, question: str = "") -> str | None:
    """Return true/false/unknown label for formal proof-status tasks.

    This is deliberately separate from NLI extraction. In ProofWriter/FOLIO
    prompts, phrases like "supported label is 3" contain the word "supported",
    which must not be interpreted as evidence for the true label.
    """
    label_map = _truth_status_label_map(question)
    if not label_map:
        return None
    text = _clean(claim)
    lowered = text.lower()
    if not lowered:
        return None

    label_match = re.search(
        r"\b(?:supported|final|correct|answer|truth[- ]value)\s+(?:truth[- ]value\s+|label\s+)?is\s+(\d{1,2}|[A-Za-z]+)\b",
        lowered,
    )
    if label_match:
        raw = label_match.group(1)
        if raw in label_map.values():
            return raw
        if raw in label_map:
            return label_map[raw]

    unknown_patterns = [
        r"\b(?:unknown|uncertain|undetermined|indeterminate)\b",
        r"\bcannot be determined\b",
        r"\bcan't be determined\b",
        r"\bnot enough information\b",
        r"\binsufficient information\b",
        r"\b(?:do|does|did)\s+not\s+prove\s+(?:the\s+)?(?:conclusion|query|queried statement)\b",
        r"\b(?:do|does|did)\s+not\s+prove\s+(?:the\s+)?(?:conclusion|query|queried statement)\s+or\s+(?:its\s+)?negation\b",
        r"\b(?:leave|leaves|left)\s+(?:the\s+)?(?:conclusion|query|queried statement)\s+(?:open|unknown|uncertain)\b",
        r"\bno information\b",
        r"\bthere is no information\b",
        r"\bnot stated\b",
        r"\bnot given\b",
    ]
    false_patterns = [
        r"\bprove(?:s|d)?\s+(?:the\s+)?negation\b",
        r"\b(?:conclusion|query|queried statement|target statement)\s+is\s+(?:false|disproved)\b",
        r"\b(?:disprove|disproves|disproved)\s+(?:the\s+)?(?:conclusion|query|queried statement|target statement)\b",
        r"\bcontradict(?:s|ed)?\s+(?:the\s+)?(?:conclusion|query|queried statement|target statement)\b",
    ]
    true_patterns = [
        r"\bprove(?:s|d)?\s+(?:the\s+)?(?:conclusion|query|queried statement|target statement)\b",
        r"\b(?:conclusion|query|queried statement|target statement)\s+is\s+(?:true|proved)\b",
        r"\b(?:the\s+)?premises?\s+(?:entail|entails|prove|proves)\s+(?:the\s+)?(?:conclusion|query|queried statement|target statement)\b",
    ]

    def has(patterns: list[str]) -> bool:
        return any(re.search(pattern, lowered) for pattern in patterns)

    if has(unknown_patterns):
        return label_map["unknown"]
    if has(false_patterns):
        return label_map["false"]
    if has(true_patterns) and not re.search(r"\b(?:do|does|did|not|cannot|can't)\s+[^.\n]{0,30}\bprove", lowered):
        return label_map["true"]
    return None


def _extract_nli_hypothesis(question: str) -> str:
    text = _clean(question)
    match = re.search(
        r"\bHypothesis:\s*(.+?)(?=\s+Answer\s+\d{1,2}\s+(?:for|if)\b|\s+Use\s+only\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    hypothesis = match.group(1).strip().strip(" .")
    return hypothesis if 3 <= len(hypothesis) <= 240 else ""


def nli_label_supported_by_claim(claim: str, question: str = "") -> str | None:
    """Return the allowed NLI label supported by a relation claim, if explicit."""
    label_map = _nli_label_map(question)
    if not label_map:
        return None
    text = _clean(claim)
    lowered = text.lower()
    if not lowered:
        return None

    explicit_label_match = re.search(
        r"\b(?:supported|correct|final|answer|relation)\s+(?:nli\s+|relation\s+)?label\s+(?:is|=)\s+(\d{1,2}|[A-Za-z]+)\b",
        lowered,
    )
    if explicit_label_match:
        raw = explicit_label_match.group(1)
        if raw in label_map.values():
            return raw
        if raw in label_map:
            return label_map[raw]

    neutral_patterns = [
        r"\bneutral\b",
        r"\bnot enough information\b",
        r"\binsufficient information\b",
        r"\bcannot be determined\b",
        r"\bcan't be determined\b",
        r"\bunknown\b",
        r"\bnot (?:directly |fully )?(?:supported|entailed)(?: by the premise)?\s+(?:or|nor)\s+contradicted\b",
        r"\bdoes not (?:directly |fully )?(?:support|entail)[^.\n]{0,60}\b(?:or|nor)\s+contradict\b",
        r"\bneither (?:supported|entailed) nor contradicted\b",
        r"\bunsupported (?:and|but) not contradicted\b",
        r"\bnot (?:clearly |directly |explicitly |fully )?contradicted\b",
        r"\bno contradiction\b",
        r"\bneither entailed nor contradicted\b",
        r"\bunsupported\b",
        r"\bnot (?:directly |fully )?supported\b",
        r"\bnot (?:clearly |definitively |strongly |directly |fully )?supported\b",
        r"\bnot (?:directly |fully )?entailed\b",
        r"\breasonable inference but not\b",
        r"\bnot (?:explicitly |directly )?(?:stated|confirmed|mentioned|addressed|evaluated)\b",
        r"\bdoes not (?:explicitly |directly |definitively )?(?:state|confirm|mention|address|evaluate|support|provide evidence for)\b",
        r"\bdoes not definitively (?:say|show|establish|prove)\b",
        r"\bnot definitively (?:stated|confirmed|established|shown|proved)\b",
        r"\bdoes not provide (?:direct )?(?:evidence|information)\b",
        r"\bnot (?:confirmed|specified|stated|mentioned|addressed|evaluated) by the premise\b",
        r"\bdoes not (?:directly |fully )?entail\b",
        r"\bpremise does not (?:say|state|mention|show|prove)\b",
        r"\bintroduces? (?:new|additional|extra) information\b",
        r"\bonly provides? (?:a )?conditional\b",
        r"\bwithout (?:confirming|stating|showing|establishing|proving)\b",
        r"\bdoes not (?:provide|give|contain|state|show|establish|prove|confirm|indicate)[^.\n]{0,90}\b(?:contradict|contradiction)\b",
        r"\bdoes not (?:provide|give)[^.\n]{0,60}\benough evidence\b",
        r"\bdoes not[^.\n]{0,60}\b(?:support|imply|establish|prove|confirm|conclude)\b",
        r"\bnot enough information[^.\n]{0,60}\b(?:contradict|contradiction)\b",
        r"\bno (?:clear |direct |explicit )?evidence[^.\n]{0,60}\b(?:contradict|contradiction)\b",
        r"\bdoes not contradict\b",
    ]
    contradiction_patterns = [
        r"\bcontradiction\b",
        r"\bcontradicts?\b",
        r"\bcontradicted\b",
        r"\bconflicts? with\b",
        r"\binconsistent with\b",
        r"\bopposite of\b",
        r"\bmutually exclusive\b",
        r"\bcannot both be true\b",
    ]
    entailment_patterns = [
        r"\bentailment\b",
        r"\bpremise entails? (?:the )?hypothesis\b",
        r"\bthe hypothesis is entailed by the premise\b",
        r"\bentailed by the premise\b",
        r"\bfollows from the premise\b",
        r"\bpremise supports? (?:the )?hypothesis\b",
        r"\bdirectly supports? (?:the )?hypothesis\b",
        r"\baligns? with (?:the )?hypothesis\b",
    ]

    def has(patterns: list[str]) -> bool:
        return any(re.search(pattern, lowered) for pattern in patterns)

    def negates_contradiction() -> bool:
        return bool(
            re.search(
                r"\b(?:does not|do not|did not|cannot|can't|not enough information to|no (?:clear |direct |explicit )?evidence to)\s+"
                r"(?:\w+\s+){0,8}?contradict\b",
                lowered,
            )
            or re.search(r"\b(?:not|nor)(?:\s+\w+){0,3}\s+contradicted\b", lowered)
            or re.search(r"\bno\s+(?:clear\s+|direct\s+|explicit\s+)?contradiction\b", lowered)
            or re.search(r"\bdoes not [^.\n]{0,90}\bcontradiction\b", lowered)
        )

    neutral = has(neutral_patterns)
    contradiction = has(contradiction_patterns)
    entailment = has(entailment_patterns)

    if neutral and (
        re.search(r"\bnot (?:directly |fully )?(?:supported|entailed)(?: by the premise)?\s+(?:or|nor)\s+contradicted\b", lowered)
        or re.search(r"\bdoes not (?:directly |fully )?(?:support|entail)[^.\n]{0,60}\b(?:or|nor)\s+contradict\b", lowered)
        or re.search(r"\bneither (?:supported|entailed) nor contradicted\b", lowered)
        or re.search(r"\bunsupported (?:and|but) not contradicted\b", lowered)
        or re.search(r"\bnot contradicted\b", lowered)
        or re.search(r"\bno contradiction\b", lowered)
        or negates_contradiction()
    ):
        return label_map["neutral"]
    if contradiction and not negates_contradiction():
        return label_map["contradiction"]
    if neutral:
        return label_map["neutral"]
    if entailment and not re.search(r"\bnot\s+(?:directly\s+|fully\s+)?entail", lowered):
        return label_map["entailment"]
    hypothesis = _extract_nli_hypothesis(question)
    if hypothesis:
        compact_hypothesis = _compact(hypothesis)
        compact_claim = _compact(text)
        if len(compact_hypothesis) >= 10 and compact_hypothesis in compact_claim:
            if not any(
                marker in lowered
                for marker in (
                    "unsupported",
                    "not supported",
                    "not definitively supported",
                    "not directly supported",
                    "not directly supported",
                    "not entailed",
                    "not directly entailed",
                    "does not support",
                    "does not entail",
                    "does not provide enough evidence",
                    "does not provide sufficient evidence",
                    "does not strongly imply",
                    "does not imply",
                    "does not establish",
                    "does not prove",
                    "does not confirm",
                    "reasonable inference but not",
                    "cannot be determined",
                    "not enough information",
                    "contradicts",
                    "contradiction",
                    "conflicts with",
                    "inconsistent with",
                )
            ):
                return label_map["entailment"]
    return None


def nli_label_supported_by_resolution(
    resolved_claim: str,
    rationale: str = "",
    question: str = "",
) -> str | None:
    """Infer an NLI label from a resolution, preferring the corrected claim.

    Resolution rationales sometimes mention the rejected side or contain stale
    label words. The corrected claim is the continuation anchor, so it should
    not be overridden by a contradictory rationale.
    """
    claim_label = nli_label_supported_by_claim(resolved_claim, question)
    if claim_label:
        return claim_label
    rationale_lowered = _clean(rationale).lower()
    if re.search(r"\b(?:other side|fails|assumes|incorrectly|misinterprets?)\b", rationale_lowered):
        return None
    return nli_label_supported_by_claim(rationale, question)


def _semantic_aliases(description: str) -> list[str]:
    lowered = _clean(description).lower()
    lowered = lowered.replace("the statement is ", "")
    lowered = lowered.replace("the hypothesis is ", "")
    aliases = [re.escape(lowered)] if lowered else []
    if "entail" in lowered:
        aliases += [
            r"\bentailed?\b",
            r"\bentailment\b",
            r"\bentails?\b",
            r"\bsupported by the premise\b",
            r"\bis entailed by the premise\b",
            r"\baligns? with the hypothesis\b",
            r"\bsupports? the hypothesis\b",
            r"\bdirectly supports? the hypothesis\b",
        ]
    if "contrad" in lowered or "false" in lowered:
        aliases += [r"\bcontradiction\b", r"\bcontradicts?\b", r"\bcontradicted\b", r"\bfalse\b", r"\bconflicts? with\b"]
    if lowered == "neutral" or "neutral" in lowered:
        aliases += [
            r"\bneutral\b",
            r"\bnot enough information\b",
            r"\bcannot be determined\b",
            r"\bneither entailed nor contradicted\b",
            r"\bnot (?:directly |fully )?entailed\b",
            r"\bdoes not (?:directly |fully )?entail\b",
            r"\bunsupported\b",
            r"\bnot directly supported\b",
            r"\bintroduces new information\b",
        ]
    if "uncertain" in lowered or "unknown" in lowered:
        aliases += [
            r"\buncertain\b",
            r"\bunknown\b",
            r"\bindeterminate\b",
            r"\binsufficient information\b",
            r"\bnot enough information\b",
            r"\bcannot be determined\b",
        ]
    if "yes" in lowered:
        aliases += [r"\byes\b", r"\btrue\b"]
    if "no" in lowered:
        aliases += [r"\bno\b", r"\bfalse\b"]
    if "true" in lowered:
        aliases += [r"\btrue\b", r"\bcorrect\b", r"\bsupported\b"]
    if "false" in lowered:
        aliases += [r"\bfalse\b", r"\bincorrect\b", r"\bnot true\b"]
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _question_requests_negative_option_relation(question: str) -> bool:
    lowered = _clean(question).lower()
    return bool(
        re.search(
            r"\b(?:contradict|contradicts|contradicted|weaken|weakens|except|false|must\s+be\s+false|cannot\s+be\s+true|could\s+not|not\s+follow|does\s+not\s+follow|flaw)\b",
            lowered,
        )
    )


def _explicit_option_label_is_negated(
    text: str,
    label: str,
    question: str,
    *,
    match_start: int | None = None,
) -> bool:
    option_map = _extract_option_map(question)
    if not option_map or label not in option_map:
        return False
    if _question_requests_negative_option_relation(question):
        return False

    clean = _clean(text)
    if not clean:
        return False
    window_start = max(0, (match_start if match_start is not None else len(clean)) - 220)
    window = clean[window_start : (match_start if match_start is not None else len(clean))]
    lowered = window.lower()
    if not re.search(
        r"\b(?:contradict(?:s|ed)?|eliminat(?:e|es|ed|ing)|ruled?\s+out|invalid|impossible|cannot\s+support|can't\s+support|does\s+not\s+support|do\s+not\s+support|fails?\s+to\s+support|rejected?)\b",
        lowered,
    ):
        return False

    label_pattern = rf"\b(?:option|choice|label|answer)\s*{re.escape(label)}\b|\b{re.escape(label)}\s*(?:is|was)\s+(?:eliminated|invalid|impossible|wrong|rejected)\b"
    if re.search(label_pattern, lowered, flags=re.IGNORECASE):
        return True

    option_text = option_map.get(label, "")
    option_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", option_text.lower())
        if len(token) >= 4
        and token
        not in {
            "that",
            "this",
            "with",
            "from",
            "which",
            "following",
            "statement",
            "area",
            "district",
            "option",
            "answer",
        }
    ]
    directional_tokens = {
        "north",
        "south",
        "east",
        "west",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
        "true",
        "false",
    }
    overlap = [token for token in option_tokens if token in lowered]
    if any(token in directional_tokens for token in overlap):
        return True
    return len(overlap) >= 2


def question_uses_label_answers(question: str) -> bool:
    return bool(_extract_label_descriptions(question) or _extract_option_map(question))


def allowed_label_answers(question: str) -> list[str]:
    labels = list(_extract_label_descriptions(question).keys())
    if labels:
        return labels
    return list(_extract_option_map(question).keys())


def label_answer_hint(question: str) -> str:
    labels = allowed_label_answers(question)
    if not labels:
        return ""
    joined = ", ".join(labels[:6])
    if len(labels) > 6:
        joined += ", ..."
    hint = f"The final non-empty line must be exactly one allowed label token, such as {joined}."
    if set(labels) == {"true", "false"}:
        hint += " If the target statement is proven, the label is true even when the statement contains 'not'; if its negation is proven, the label is false."
    return hint


def label_claim_is_bare_answer(claim: str, question: str = "") -> bool:
    allowed = {label.lower() for label in allowed_label_answers(question)}
    if not allowed:
        return False
    text = _clean(claim).lower()
    if not text:
        return False
    text = re.sub(r"\\boxed\{\s*([^{}]+?)\s*\}", r"\1", text)
    text = re.sub(r"\{[^{}]*final answer\s*:?\s*([^{}]+?)\}", r"\1", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return False
    boilerplate = {
        "the",
        "a",
        "an",
        "requested",
        "final",
        "answer",
        "label",
        "token",
        "claim",
        "is",
        "be",
        "should",
        "must",
        "as",
        "for",
        "this",
        "question",
        "direct",
        "output",
    }
    content_tokens = [token for token in tokens if token not in boilerplate and token not in allowed]
    label_tokens = [token for token in tokens if token in allowed]
    return bool(label_tokens) and not content_tokens


def extract_label_answer(text: str, question: str = "") -> str | None:
    allowed = allowed_label_answers(question)
    if not allowed:
        return None

    clean = _clean(text)
    if not clean:
        return None
    tail = clean[-320:]
    compact_tail = _compact(tail)
    compact_all = _compact(clean)
    option_map = _extract_option_map(question)
    desc_map = _extract_label_descriptions(question)
    truth_value_labels = set(allowed) == {"true", "false"}
    nli_labels = _nli_label_map(question)
    truth_status_labels = _truth_status_label_map(question)

    def canonical(candidate: str | None) -> str | None:
        if candidate is None:
            return None
        candidate = str(candidate).strip()
        return candidate if candidate in allowed else None

    explicit_supported_label_matches = list(
        re.finditer(
            r"\b(?:supported|correct|final|answer|chosen|selected)\s+"
            r"(?:option\s+|choice\s+|label\s+)?(?:is|=)\s+"
            r"(\d{1,2}|[A-J]|yes|no|true|false)\b",
            tail,
            flags=re.IGNORECASE,
        )
    )
    if explicit_supported_label_matches:
        match = explicit_supported_label_matches[-1]
        raw = match.group(1)
        out = canonical(raw.upper() if raw.isalpha() and len(raw) == 1 else raw.lower() if raw.isalpha() else raw)
        if out:
            if _explicit_option_label_is_negated(clean, out, question, match_start=match.start()):
                return None
            return out

    def explicit_truth_value_label() -> str | None:
        if not truth_value_labels:
            return None
        patterns = [
            (r"\b(?:the\s+)?(?:statement|target statement)\b[^.\n]{0,120}?\bis\s+true\b", "true"),
            (r"\b(?:the\s+)?(?:statement|target statement)\b[^.\n]{0,120}?\bis\s+false\b", "false"),
            (r"\b(?:the\s+)?(?:statement|target statement)\b[^.\n]{0,120}?\bis\s+not\s+true\b", "false"),
            (r"\b(?:the\s+)?(?:statement|target statement)\b[^.\n]{0,120}?\bis\s+not\s+false\b", "true"),
        ]
        found: list[tuple[int, str]] = []
        for pattern, label in patterns:
            for match in re.finditer(pattern, tail, flags=re.IGNORECASE):
                found.append((match.end(), label))
        if not found:
            return None
        found.sort()
        return found[-1][1]

    def explicit_nli_label() -> str | None:
        if not nli_labels:
            return None
        return nli_label_supported_by_claim(tail, question)

    def explicit_truth_status_label() -> str | None:
        if not truth_status_labels:
            return None
        return truth_status_label_supported_by_claim(tail, question)

    boxed_matches = list(re.finditer(r"\\?boxed\s*\{\s*(\d{1,2}|[A-J])\s*\}?", clean, flags=re.IGNORECASE))
    if boxed_matches:
        raw = boxed_matches[-1].group(1)
        out = canonical(raw.upper() if raw.isalpha() else raw)
        if out:
            return out

    for seg in re.findall(r"\{[^{}]*final answer[^{}]*\}", clean, flags=re.IGNORECASE):
        m = re.search(r"\b(\d{1,2}|[A-J])\b", seg, flags=re.IGNORECASE)
        if m:
            out = canonical(m.group(1).upper() if m.group(1).isalpha() else m.group(1))
            if out:
                return out

    patterns = [
        r"final answer[^0-9A-Ja-j]{0,80}\\?boxed\s*\{\s*(\d{1,2}|[A-J])\s*\}?",
        r"(?:final answer|answer|option|choice)\s*[:=\-]?\s*(\d{1,2}|[A-J])\b",
        r"\b(?:therefore\s+)?(?:final answer|answer)\s+is\s+(\d{1,2}|[A-J])\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, tail, flags=re.IGNORECASE))
        if matches:
            raw = matches[-1].group(1)
            out = canonical(raw.upper() if raw.isalpha() else raw)
            if out:
                if truth_value_labels:
                    explicit = explicit_truth_value_label()
                    if explicit and explicit != out:
                        return explicit
                explicit_nli = explicit_nli_label()
                if explicit_nli and explicit_nli != out:
                    return explicit_nli
                explicit_status = explicit_truth_status_label()
                if explicit_status and explicit_status != out:
                    return explicit_status
                return out

    if clean in allowed:
        return clean
    if tail in allowed:
        return tail

    if len(allowed) == 2:
        binary_pair = {label: desc.lower() for label, desc in desc_map.items()}
        yes_label = next((label for label, desc in binary_pair.items() if "yes" in desc or "true" in desc or "entail" in desc), None)
        no_label = next((label for label, desc in binary_pair.items() if "no" in desc or "false" in desc or "contrad" in desc or "neutral" in desc), None)
        if yes_label and no_label:
            binary_hits: list[tuple[int, int, str]] = []
            binary_patterns = [
                (yes_label, r"\b(?:yes|true|entail(?:ed|ment)?|supported)\b"),
                (no_label, r"\b(?:no|false|contradiction|unsupported|not enough information|neutral|not true)\b"),
            ]
            for label, pattern in binary_patterns:
                for match in re.finditer(pattern, tail, flags=re.IGNORECASE):
                    binary_hits.append((match.end(), match.end() - match.start(), label))
            if binary_hits:
                binary_hits.sort()
                candidate = binary_hits[-1][2]
                explicit = explicit_truth_value_label()
                if explicit and explicit != candidate:
                    return explicit
                explicit_nli = explicit_nli_label()
                if explicit_nli and explicit_nli != candidate:
                    return explicit_nli
                explicit_status = explicit_truth_status_label()
                if explicit_status and explicit_status != candidate:
                    return explicit_status
                return candidate

    explicit_status = explicit_truth_status_label()
    if explicit_status:
        return explicit_status

    best_semantic: tuple[int, str] | None = None
    for label, desc in desc_map.items():
        for alias in _semantic_aliases(desc):
            for match in re.finditer(alias, tail, flags=re.IGNORECASE):
                candidate = (match.end(), label)
                if best_semantic is None or candidate[0] > best_semantic[0]:
                    best_semantic = candidate
    if best_semantic is not None:
        return best_semantic[1]

    if option_map:
        matched: list[tuple[str, int]] = []
        for label, option_text in option_map.items():
            opt = _compact(option_text)
            if not opt:
                continue
            if opt in compact_tail or opt in compact_all:
                matched.append((label, len(opt)))
                continue
            prefix = opt[: max(12, min(len(opt), 28))]
            if prefix and prefix in compact_tail:
                matched.append((label, len(prefix)))
        if len(matched) == 1:
            return matched[0][0]
        if len(matched) > 1:
            matched.sort(key=lambda item: item[1], reverse=True)
            return matched[0][0]

    for label in allowed:
        if re.search(rf"(?:^|[^0-9A-Za-z]){re.escape(label)}(?:$|[^0-9A-Za-z])", tail, flags=re.IGNORECASE):
            return label

    return None
