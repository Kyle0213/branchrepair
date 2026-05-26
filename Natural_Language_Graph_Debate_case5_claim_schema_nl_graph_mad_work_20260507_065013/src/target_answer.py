from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TargetAnswerMatch:
    answer: str
    pattern: str
    family: str


@dataclass(frozen=True)
class TargetNoteProvenance:
    kind: str
    confidence: str
    reason: str


def math_answer_from_plain_value(value: str) -> str | None:
    text = str(value or "").strip().strip(".")
    if not text:
        return None
    if "\\pi" not in text:
        text = text.replace("pi", "\\pi")
    frac = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", text)
    if frac:
        return f"\\frac{{{frac.group(1)}}}{{{frac.group(2)}}}"
    return text


TARGET_ANSWER_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\(p,\s*q,\s*r\)\s*=\s*(\([^)]+\))", "ordered_tuple", "target_value"),
    (r"image line is y\s*=\s*([^.;]+)", "image_line", "geometry"),
    (r"displayed phase c is\s*([^.;]+)", "displayed_phase", "target_value"),
    (r"requested amount is\s*(\\?\$\d+(?:\.\d+)?)", "requested_amount", "units_or_format"),
    (r"requested value is\s*([^.;]+)", "requested_value", "target_value"),
    (r"requested product is positive\s*([-\d/]+)", "requested_positive_product", "target_value"),
    (r"requested product is\s*([-\d/]+)", "requested_product", "target_value"),
    (r"requested count is\s*([-\d/]+)", "requested_count", "counting"),
    (r"requested area is\s*([^.;]+)", "requested_area", "geometry"),
    (r"requested radius is\s*([^.;]+)", "requested_radius", "geometry"),
    (r"requested minimum is\s*([^.;]+)", "requested_minimum", "optimization"),
    (r"requested roots are\s*([^.;]+)", "requested_roots", "algebraic_object"),
    (r"requested set is\s*([^.;]+)", "requested_set", "algebraic_object"),
    (r"requested probability is\s*([-\d/]+)", "requested_probability", "probability"),
    (r"requested surface area is\s*([^.;]+)", "requested_surface_area", "geometry"),
    (r"requested sum is\s*([-\d/]+)", "requested_sum", "target_value"),
    (r"coefficient absolute sum is\s*([-\d/]+)", "coefficient_absolute_sum", "polynomial_target"),
    (r"sum of all complex solutions is\s*([-\d/]+)", "complex_solution_sum", "algebraic_object"),
    (r"requested angle is\s*([-\d/]+)\s*degrees", "requested_angle_degrees", "geometry"),
    (r"requested volume is\s*([^.;]+)", "requested_volume", "geometry"),
    (r"requested vector is\s*([^.;]+)", "requested_vector", "linear_algebra"),
    (r"requested matrix is\s*([^.;]+)", "requested_matrix", "linear_algebra"),
    (r"requested range is\s*([^.;]+)", "requested_range", "function_target"),
    (r"requested name is\s*([A-Za-z]+)", "requested_name", "classification"),
    (r"requested distance is\s*([^.;]+)", "requested_distance", "geometry"),
    (r"requested length is\s*([^.;]+)", "requested_length", "geometry"),
    (r"requested interval is\s*([^.;]+)", "requested_interval", "function_target"),
    (r"base-eight answer is\s*([0-7]+_8)", "base_eight_answer", "base_notation"),
    (r"base-\w+ answer is\s*([0-9]+_\{?\d+\}?)", "base_answer", "base_notation"),
    (r"correct option is\s*(\\text\{\([A-Z]\)\}|\([A-Z]\))", "correct_option", "multiple_choice"),
    (r"event has probability\s*\d+\s*/\s*\d+\s*=\s*([-\d/]+)", "event_probability", "probability"),
    (r"normalized value\b[^.]*\bis\s*(-?\d+)", "normalized_value", "modular_arithmetic"),
    (r"rate is\s*([-\d.]+)\s*percent", "rate_percent", "units_or_format"),
    (r"smallest possible n is\s*(\d+)", "smallest_possible_n", "optimization"),
    (r"(?:maximum )?angle is\s*([-\d.]+)\s*degrees", "angle_degrees", "geometry"),
    (r"\bP\s*=\s*(\d+)", "variable_p", "target_value"),
    (r"\bn\s*=\s*(\d+)", "variable_n", "target_value"),
    (r"has length\s*(\d+)", "has_length", "geometry"),
    (r"less than [^.]*\bis\s*(\d+)", "less_than_count", "counting"),
    (r"positive square root,\s*([-\d/]+)", "positive_square_root", "target_value"),
    (r"requested expression is\s*([^.;]+)", "requested_expression", "expression_target"),
    (r"least possible value is\s*([-\d/]+)", "least_possible_value", "optimization"),
    (r"number of personalities is\s*[^=]*=\s*(\d+)", "number_of_personalities", "counting"),
    (r"sum of all possible values of theta as\s*([^.;]+)", "theta_value_sum", "trigonometry"),
    (r"sum to [^.]*=\s*(-?\d+)", "sum_to_value", "target_value"),
    (r"\bd\s*=\s*([-\d/]+)", "variable_d", "target_value"),
    (r"\bis\s*([-\d/]+)\s*[.;]?$", "terminal_is_number", "target_value"),
    (r"number of arrangements is\s*[^=]*=\s*(\d+)", "arrangement_equation", "counting"),
    (r"number of arrangements is\s*(\d+)", "arrangement_count", "counting"),
    (r"digit sum is\s*(\d+)", "digit_sum", "number_theory"),
    (r"perimeter is\s*([^.;]+)", "perimeter", "geometry"),
    (r"maximum value of the absolute product is\s*(\d+)", "absolute_product_maximum", "optimization"),
)


def target_answer_match_from_note(note: str) -> TargetAnswerMatch | None:
    return None


def answer_from_target_check_note(note: str) -> str | None:
    return None


def target_note_provenance(note: str) -> TargetNoteProvenance:
    text = str(note or "").strip()
    lowered = text.lower()
    if not text:
        return TargetNoteProvenance(kind="none", confidence="none", reason="No target note was generated.")
    if lowered.startswith("deterministic"):
        if "case-study" in lowered:
            return TargetNoteProvenance(
                kind="case_study_deterministic",
                confidence="high",
                reason="The note is an explicit case-study deterministic check.",
            )
        if "base-" in lowered or "base " in lowered:
            return TargetNoteProvenance(
                kind="format_deterministic",
                confidence="high",
                reason="The note resolves a target answer format or base-notation conversion.",
            )
        return TargetNoteProvenance(
            kind="deterministic_backend",
            confidence="high",
            reason="The note contains a deterministic local target check.",
        )
    if "check:" in lowered and (" so " in lowered or " gives " in lowered or " hence " in lowered):
        return TargetNoteProvenance(
            kind="deterministic_backend",
            confidence="high",
            reason="The note contains a named local check with a computed consequence.",
        )
    if "requested answer is" in lowered or "requested object is" in lowered or "requested event is" in lowered:
        return TargetNoteProvenance(
            kind="natural_target_guard",
            confidence="medium",
            reason="The note identifies the requested object but does not by itself compute a final value.",
        )
    return TargetNoteProvenance(
        kind="natural_note",
        confidence="medium",
        reason="The note is a natural-language target hint.",
    )
