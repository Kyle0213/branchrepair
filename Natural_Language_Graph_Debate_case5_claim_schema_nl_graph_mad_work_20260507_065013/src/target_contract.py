from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class TargetContract:
    requested_object: str = ""
    answer_format: str = ""
    guardrails: tuple[str, ...] = field(default_factory=tuple)

    def note(self) -> str:
        parts: list[str] = []
        if self.requested_object:
            parts.append(f"The requested object is {self.requested_object}.")
        if self.answer_format:
            parts.append(f"The answer format constraint is {self.answer_format}.")
        parts.extend(self.guardrails)
        return " ".join(parts).strip()


def _normalize_question(question: str) -> str:
    text = str(question or "").replace("\n", " ")
    for marker in (
        " Make sure to state your final answer",
        " Use short atomic reasoning lines.",
    ):
        if marker in text:
            text = text.split(marker, 1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_question_text(question: str) -> str:
    return _normalize_question(question)


def _normalized_lower(question: str) -> str:
    return _normalize_question(question).lower()


def _first_match(patterns: tuple[str, ...], text: str, flags: int = re.IGNORECASE) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=flags)
        if match:
            return match.group(1).strip(" ,.;:")
    return ""


def _infer_requested_object(question: str) -> str:
    text = _normalize_question(question)
    lowered = text.lower()
    math_find = _first_match((r"\bfind\s+(\$[^$]+\$)",), text)
    if math_find:
        math_find = re.sub(r"\.\$$", "$", math_find)
        return math_find
    value_request = _first_match(
        (
            r"\b(?:least|smallest|min(?:imum)?|greatest|largest|max(?:imum)?)\s+(?:possible\s+)?value\s+of\s+(.+?)(?:\?|\.|$)",
        ),
        text,
    )
    if value_request:
        return f"the requested value of {value_request}"
    if re.search(r"\bexpress\b", lowered) and "base" in lowered:
        return "the given value expressed in the requested base"
    if "how many" in lowered:
        object_text = _first_match((r"\bhow many\s+(.+?)(?:\?|\.|$)",), text)
        return object_text or "the requested count"
    if re.search(r"\bwhat is the probability\b", lowered):
        object_text = _first_match((r"\bprobability\s+(?:that|of)\s+(.+?)(?:\?|\.|$)",), text)
        return f"the probability of {object_text}" if object_text else "the requested probability"
    if re.search(r"\bfind\s+the\s+greatest\s+common\s+divisor\b", lowered) or re.search(r"\bwhat\s+is\s+the\s+(?:gcd|greatest\s+common\s+divisor)\b", lowered):
        return "the greatest common divisor of the stated integers"
    if "range of the function" in lowered or re.search(r"\bwhat is the range\b", lowered):
        return "the range as a set or interval, not just a sample value"
    if "domain of" in lowered or re.search(r"\bwhat is the domain\b", lowered):
        return "the domain as a set or interval, not just a sample value"
    if "angle between" in lowered:
        return "the angle between the requested objects"
    if re.search(r"\barea\b", lowered):
        return "the requested area"
    if re.search(r"\bvolume\b", lowered):
        return "the requested volume"
    if re.search(r"\bperimeter\b", lowered):
        return "the requested perimeter"
    if re.search(r"\bdistance\b", lowered):
        return "the requested distance"
    if re.search(r"\blength\b", lowered):
        return "the requested length"
    if "sum of" in lowered:
        sum_object = _first_match((r"\b(sum of\s+.+?)(?:\?|\.|$)",), text)
        return sum_object or "the requested sum"
    if "product of" in lowered:
        product_object = _first_match((r"\b(product of\s+.+?)(?:\?|\.|$)",), text)
        return product_object or "the requested product"
    if "roots" in lowered:
        return "the requested root object: value, set, count, sum, or product as stated"
    extremal_object = _first_match(
        (
            r"\bwhat is\s+(the\s+(?:least|smallest|min(?:imum)?|greatest|largest|max(?:imum)?)\s+.+?)(?:\?|\.|$)",
            r"\bfind\s+(the\s+(?:least|smallest|min(?:imum)?|greatest|largest|max(?:imum)?)\s+.+?)(?:\?|\.|$)",
        ),
        text,
    )
    if extremal_object:
        return extremal_object
    if re.search(r"\b(?:least|smallest|min(?:imum)?)\b", lowered):
        return "the requested minimum or least object, with attainability checked"
    if re.search(r"\b(?:greatest|largest|max(?:imum)?)\b", lowered):
        return "the requested maximum or greatest object, with attainability checked"

    object_text = _first_match(
        (
            r"\bfind\s+(.+?)(?:\s+if\b|\s+given\b|\?|\.|$)",
            r"\bdetermine\s+(.+?)(?:\s+if\b|\s+given\b|\?|\.|$)",
            r"\bcompute\s+(.+?)(?:\s+if\b|\s+given\b|\?|\.|$)",
            r"\bevaluate\s+(.+?)(?:\s+if\b|\s+given\b|\?|\.|$)",
            r"\bexpress\s+(.+?)(?:\s+in\b|\s+as\b|\?|\.|$)",
            r"\bwhat is\s+(.+?)(?:\?|\.|$)",
        ),
        text,
    )
    return object_text or "the final object explicitly asked for by the question"


def _infer_answer_format(question: str) -> str:
    text = _normalize_question(question)
    lowered = text.lower()
    constraints: list[str] = []
    form = _first_match((r"\bin the form\s+(.+?)(?:,|\?|\.|$)",), text)
    if form:
        constraints.append(f"use the stated form {form}")
    if re.search(r"\bbase\s*\$?\d+|base-\w+|_\{?\d+\}?", lowered):
        constraints.append("preserve the requested base notation")
    if re.search(r"\bdegrees\b", lowered):
        constraints.append("return an angle in degrees when the question asks for degrees")
    if re.search(r"\bradians\b", lowered):
        constraints.append("return an angle in radians when the question asks for radians")
    if re.search(r"\b0\s*\\?le\s*[a-zA-Z]\s*<\s*\d+", text):
        constraints.append("normalize the residue into the stated interval")
    if "ordered pair" in lowered or "ordered triple" in lowered:
        constraints.append("preserve the ordered tuple")
    if "as a fraction" in lowered or "fraction" in lowered:
        constraints.append("preserve the requested fractional form")
    if "units" in lowered or re.search(r"\bpercent(?:age)?\b", lowered):
        constraints.append("preserve the requested units or percent interpretation")
    return "; ".join(dict.fromkeys(constraints))


def _infer_guardrails(question: str) -> tuple[str, ...]:
    text = _normalize_question(question)
    lowered = text.lower()
    guardrails: list[str] = []
    asks_gcd = "greatest common divisor" in lowered or re.search(r"\bgcd\b", lowered)
    if "by what number" in lowered and "multiply" in lowered and "divide" in lowered:
        guardrails.append("Do not confuse the multiplier or reciprocal with the quotient obtained after applying it.")
    if re.search(r"\bmod\b|\\pmod|\\equiv", lowered):
        guardrails.append("Separate the raw congruent value from the normalized representative requested by the problem.")
    if "probability" in lowered:
        guardrails.append("Keep the event definition, favorable cases, total cases, and final probability separate.")
    if "all" in lowered and re.search(r"\b(values|solutions|roots|integers|numbers)\b", lowered):
        guardrails.append("Separate exhibited examples from proof that the list is complete.")
    if re.search(r"\bhow many|number of|count\b", lowered):
        guardrails.append("Keep candidate generation, validity filtering, and the final count separate.")
        guardrails.append("A generic reminder that a restriction matters is not yet a decisive count claim; prefer a concrete count, formula, or validity split.")
        if any(marker in lowered for marker in ("no two", "not adjacent", "adjacent", "at least", "at most", "exactly")):
            guardrails.append("For restricted counting, do not split on unrestricted setup totals or bookkeeping descriptions before the restriction-specific count or validity check appears.")
    if not asks_gcd and re.search(r"\bleast|smallest|minimum|greatest|largest|maximum\b", lowered):
        guardrails.append("Distinguish a bound or candidate from an actually attained optimum.")
    if "angle between" in lowered:
        guardrails.append("Keep the underlying objects, vectors or slopes, dot product or angle formula, and final angle separate.")
    if "range" in lowered or "domain" in lowered:
        guardrails.append("Distinguish sample outputs from the full set requested by the problem.")
        guardrails.append("Prefer a set-level or interval-level split over an isolated boundary-point note when comparing paths.")
        guardrails.append("A limit or approach statement does not by itself make a boundary point part of the requested set; attainability needs an allowed assignment.")
    if "root" in lowered:
        guardrails.append("Track whether the question asks for a root, a root family, a count, a sum, or a product.")
    if "coefficient" in lowered or re.search(r"find\s+\$?[a-z]\$?", lowered):
        guardrails.append("Separate the expanded expression from the specific coefficient or parameter being asked for.")
    if "matrix" in lowered or "vector" in lowered:
        guardrails.append("Preserve the requested matrix or vector object instead of collapsing it to a nearby scalar.")
    if any(marker in lowered for marker in ("image of", "rotation", "rotated", "reflection", "translated", "transformation")):
        guardrails.append("Separate intermediate transformed quantities from the final requested image or output object.")
    if "percent" in lowered or "percentage" in lowered:
        guardrails.append("Separate the decimal rate from the percentage requested by the problem.")
    return tuple(dict.fromkeys(guardrails))


def analyze_question_target(question: str) -> TargetContract:
    return TargetContract(
        requested_object=_infer_requested_object(question),
        answer_format=_infer_answer_format(question),
        guardrails=_infer_guardrails(question),
    )


def target_focus_note(question: str) -> str:
    return analyze_question_target(question).note()
