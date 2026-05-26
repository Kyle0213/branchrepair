from __future__ import annotations

import json
import re
from typing import Dict, List

from .models import ClaimNode, DivergenceCase, MethodPath, NaturalLanguageGraph, ResolutionDecision
from .splitter import clean_step_line


_CLAIM_RE = re.compile(r"^- \[(C\d+)\]\s+(.+)$")
_CLAIM_REF_RE = re.compile(r"^- \[(C\d+)\]\s*$")
_PATH_RE = re.compile(r"^- \[(P\d+)\]\s+(.+)$")
_DIVERGENCE_RE = re.compile(r"^- \[(D\d+)\]\s*$")
_AGENT_ID_RE = re.compile(r"\bA\d+\b")
_AGENT_GROUP = r"(?:A\d+|P\d+)(?:\s*(?:,|/|\band\b)\s*(?:A\d+|P\d+))*"
_NATURAL_DIV_VERB = (
    r"says|say|claims|claim|gets|get|uses|use|concludes|conclude|computes|compute|"
    r"finds|find|identifies|identify|supports|support|gives|give|answers|answer|"
    r"selects|select|keeps|keep"
)
_NATURAL_DIV_RE = re.compile(
    rf"(?:first\s+(?:real\s+)?(?:split|conflict|divergence)\s*:\s*)?"
    rf"(?P<left>{_AGENT_GROUP})\s+"
    rf"(?:{_NATURAL_DIV_VERB})\s+"
    rf"(?P<left_claim>.+?)"
    rf"(?:;|\.|\bwhile\b|\bbut\b|\bwhereas\b)\s*"
    rf"(?P<right>{_AGENT_GROUP})\s+"
    rf"(?:{_NATURAL_DIV_VERB})\s+"
    rf"(?P<right_claim>.+?)(?:\.\s*Why this matters:|\.|$)",
    flags=re.IGNORECASE,
)
_CLAIM_INLINE_META_RE = re.compile(
    r"^(?P<claim>.+?)\.\s*"
    r"(?:It is about|This is about)\s+(?P<object>.+?)\.\s*"
    r"(?:It focuses on|It concerns)\s+(?P<aspect>.+?)\.\s*"
    r"(?:It currently has status|Its status is)\s+(?P<status>.+?)\.\s*"
    r"(?:It is aligned as|Its alignment is)\s+(?P<alignment>.+?)\.\s*"
    r"(?:The supporting steps are|Supporting steps:)\s+(?P<members>.+?)\.?\s*$",
    flags=re.IGNORECASE,
)
_PATH_INLINE_META_RE = re.compile(
    r"^(?P<agents>.+?)\s+follow(?:s)?\s+this\s+path\.\s*"
    r"(?:Its method summary is:?|Summary:)\s+(?P<summary>.+?)\.\s*"
    r"(?:It uses|Claims on this path:)\s+(?P<claims>.+?)\.?\s*$",
    flags=re.IGNORECASE,
)
_DIV_INLINE_META_RE = re.compile(
    r"^(?:After\s+\[(?P<frontier>C\d+)\],\s*)?"
    r"\[(?P<left_path>P\d+)\]\s+says\s+(?P<left_claim>.+?)\.\s*"
    r"\[(?P<right_path>P\d+)\]\s+says\s+(?P<right_claim>.+?)\.\s*"
    r"(?:This is about|It is about)\s+(?P<object>.+?),\s*"
    r"(?:specifically\s+the\s+aspect|and concerns)\s+(?P<aspect>.+?)\.\s*"
    r"(?:The current alignment tag is|Its alignment is)\s+(?P<alignment>.+?),\s*"
    r"(?:and the relation label is|and its relation is)\s+(?P<relation>.+?)\.\s*"
    r"(?:Why this is currently considered minimal|Why minimal):\s+(?P<why>.+?)\.?\s*$",
    flags=re.IGNORECASE,
)
_RESOLUTION_SENTENCE_PATTERNS = {
    "Action": re.compile(r"\b(?:the chosen action is|action is)\s+([a-z_ -]+?)(?:[.,]| and )", flags=re.IGNORECASE),
    "Winning side": re.compile(r"\b(?:winning side is|chosen side is)\s+([a-z_ -]+?)(?:[.,]| and )", flags=re.IGNORECASE),
    "Resolved claim": re.compile(r"\b(?:correct claim|repaired claim to continue from|resolved claim|repaired claim)(?:\s+is)?\s*:\s*(.+?)(?:\.\s+Revision should restart|\.\s+Reason:|\.\s+Why the other side fails:|\.\s+Keep these paths|\.\s+Drop these paths|$)", flags=re.IGNORECASE),
    "Rationale": re.compile(r"\b(?:Reason|Why the other side fails):\s+(.+?)(?:\.\s*$|$)", flags=re.IGNORECASE),
    "Rewrite from": re.compile(r"\b(?:restart from|rewrite from)\s+\[?((?:C\d+)|(?:A\d+\.s\d+))\]?", flags=re.IGNORECASE),
    "Keep paths": re.compile(r"\bKeep these paths active:\s+(.+?)(?:\.\s+Drop these paths|\.\s+Reason:|$)", flags=re.IGNORECASE),
    "Drop paths": re.compile(r"\bDrop these paths:\s+(.+?)(?:\.\s+Reason:|$)", flags=re.IGNORECASE),
    "Canonical answer": re.compile(r"\bcanonical final answer is:?\s*(?:\{final answer:\s*)?\\boxed\{(.+?)\}", flags=re.IGNORECASE),
}
_FIELD_RE = re.compile(
    r"^(Members|Object|Aspect|Status|Alignment|Summary|Claims|Frontier|Relation|Left path|Right path|Left claim|Right claim|Why minimal|Action|Winning side|First conflict|Resolved claim|Correct claim|Rationale|Why the other side fails|Rewrite from|Keep paths|Drop paths|Canonical answer):\s*(.*)$"
)
_FIELD_LINE_FLEX_RE = re.compile(
    r"^\s*(?:(?:[-*]|\d+[.)])\s+)?(?:\*\*|__)?"
    r"(Members|Summary|Claims|Frontier|Relation|Left path|Right path|Left claim|Right claim|Why minimal|Action|Winning side|First conflict|Resolved claim|Rationale|Rewrite from|Keep paths|Drop paths|Decision|Winning path|Winning branch|Corrected claim|Correct claim|Reason|Why|Why the other side fails|Rewrite frontier|Rewrite claim|Rewrite point|Keep path|Drop path|Paths to keep|Paths to drop|Canonical answer|Canonical final answer)"
    r"(?:\*\*|__)?\s*[:\-]\s*(.*)$",
    flags=re.IGNORECASE,
)
_ACTION_TOKEN_RE = re.compile(
    r"\b(choose[_ -]?(?:left|right|a|b|claim[_ -]?a|claim[_ -]?b)|synthesize|keep[_ -]?parallel|keep|resolve|repair)\b",
    flags=re.IGNORECASE,
)
_WINNING_TOKEN_RE = re.compile(
    r"\b(left(?:\s+path)?|right(?:\s+path)?|claim\s+a|claim\s+b|a|b|A\d+|both|both\s+paths|synthesized|synthesis)\b",
    flags=re.IGNORECASE,
)
_PATH_ID_RE = re.compile(r"\[?(P\d+)\]?")
_PATH_OR_AGENT_ID_RE = re.compile(r"\[?((?:P|A)\d+)\]?")
_CLAIM_ID_RE = re.compile(r"\[?(C\d+)\]?")
_CLAIM_OR_STEP_ID_RE = re.compile(r"\[?((?:C\d+)|(?:A\d+\.s\d+))\]?")
_DOSSIER_STOP_MARKERS = (
    "Agent traces for audit:",
    "Current dossier:",
    "Current dossier to update:",
    "Newly revealed trace steps:",
)


def _strip_json_fence(text: str) -> str:
    raw = str(text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    return fence_match.group(1).strip() if fence_match else raw


def _first_json_object(text: str) -> dict:
    raw = _strip_json_fence(text)
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            if isinstance(payload.get("resolution"), dict):
                return payload["resolution"]
            resolutions = payload.get("resolutions")
            if isinstance(resolutions, list) and resolutions and isinstance(resolutions[0], dict):
                return resolutions[0]
            return payload
    return {}


def _json_string_value(payload: dict, *keys: str) -> str:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value or "").strip()
    return ""


def _json_list_value(payload: dict, *keys: str) -> str:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value or "").strip()
    return ""


def _normalize_alignment_label(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", str(value or "").strip().lower()).strip("._")
    mapping = {
        "sync": "synchronous",
        "synchronous": "synchronous",
        "async": "async_same_object",
        "asynchronous": "async_same_object",
        "async_same_object": "async_same_object",
        "asynchronous_same_object": "async_same_object",
        "same_object": "same_object_conflict",
        "same_object_conflict": "same_object_conflict",
        "same_object_mismatch": "same_object_conflict",
        "parallel_method": "parallel_method",
        "method_only": "method_only",
    }
    return mapping.get(normalized, str(value or "").strip())


def _canonical_field_name(key: str) -> str:
    normalized = re.sub(r"\s+", " ", str(key or "").strip().lower())
    mapping = {
        "members": "Members",
        "summary": "Summary",
        "claims": "Claims",
        "frontier": "Frontier",
        "relation": "Relation",
        "left path": "Left path",
        "right path": "Right path",
        "left claim": "Left claim",
        "right claim": "Right claim",
        "why minimal": "Why minimal",
        "action": "Action",
        "decision": "Action",
        "winning side": "Winning side",
        "winning path": "Winning side",
        "winning branch": "Winning side",
        "first conflict": "First conflict",
        "resolved claim": "Resolved claim",
        "corrected claim": "Resolved claim",
        "correct claim": "Resolved claim",
        "rationale": "Rationale",
        "reason": "Rationale",
        "why": "Rationale",
        "why the other side fails": "Rationale",
        "rewrite from": "Rewrite from",
        "rewrite frontier": "Rewrite from",
        "rewrite claim": "Rewrite from",
        "rewrite point": "Rewrite from",
        "keep paths": "Keep paths",
        "keep path": "Keep paths",
        "paths to keep": "Keep paths",
        "drop paths": "Drop paths",
        "drop path": "Drop paths",
        "paths to drop": "Drop paths",
        "canonical answer": "Canonical answer",
        "canonical final answer": "Canonical answer",
    }
    return mapping.get(normalized, key)


def _extract_first(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(str(text or ""))
    return match.group(1) if match else ""


def _normalize_action(value: str) -> str:
    raw_value = str(value or "").strip()
    token_match = _ACTION_TOKEN_RE.search(raw_value)
    if token_match:
        raw_value = token_match.group(1)
    normalized = re.sub(r"[\s-]+", "_", raw_value.lower())
    mapping = {
        "choose_left": "choose_left",
        "left": "choose_left",
        "choose_a": "choose_left",
        "choose_claim_a": "choose_left",
        "a": "choose_left",
        "claim_a": "choose_left",
        "choose_right": "choose_right",
        "right": "choose_right",
        "choose_b": "choose_right",
        "choose_claim_b": "choose_right",
        "b": "choose_right",
        "claim_b": "choose_right",
        "synthesize": "synthesize",
        "synthesis": "synthesize",
        "keep_parallel": "keep_parallel",
        "keep_both": "keep_parallel",
        "parallel": "keep_parallel",
        "both": "keep_parallel",
    }
    return mapping.get(normalized, normalized)


def _action_from_winning_side(value: str) -> str:
    side = _normalize_winning_side(value)
    if side == "left":
        return "choose_left"
    if side == "right":
        return "choose_right"
    return ""


def _normalize_winning_side(value: str) -> str:
    raw_value = str(value or "").strip()
    token_match = _WINNING_TOKEN_RE.search(raw_value)
    if token_match:
        raw_value = token_match.group(1)
    normalized = re.sub(r"\s+", " ", raw_value.lower())
    agent_match = re.fullmatch(r"a(\d+)", normalized)
    if agent_match:
        return "left" if agent_match.group(1) == "1" else "right"
    mapping = {
        "left": "left",
        "left path": "left",
        "a": "left",
        "claim a": "left",
        "right": "right",
        "right path": "right",
        "b": "right",
        "claim b": "right",
        "both": "both",
        "both paths": "both",
        "synthesized": "synthesized",
        "synthesis": "synthesized",
    }
    return mapping.get(normalized, normalized)


def _search_field_value(raw_text: str, aliases: List[str]) -> str:
    if not raw_text.strip():
        return ""
    escaped_aliases = [re.escape(alias) for alias in aliases if alias]
    if not escaped_aliases:
        return ""
    alias_group = "|".join(escaped_aliases)
    pattern = re.compile(
        rf"(?ims)(?:^|[\n\r])\s*(?:(?:[-*]|\d+[.)])\s+)?(?:\*\*|__)?"
        rf"(?P<key>{alias_group})(?:\*\*|__)?\s*[:\-]\s*"
        rf"(?P<value>.*?)(?=(?:[\n\r]+\s*(?:(?:[-*]|\d+[.)])\s+)?(?:\*\*|__)?(?:{alias_group})(?:\*\*|__)?\s*[:\-])|\Z)"
    )
    match = pattern.search(raw_text)
    return match.group("value").strip() if match else ""


def _sanitize_graph_dossier_text(text: str) -> str:
    raw_lines = str(text or "").splitlines()
    if not raw_lines:
        return ""
    start_index = None
    for index, raw_line in enumerate(raw_lines):
        if raw_line.strip() in {"Shared Claims", "Common Ground", "Shared Prefix", "Common Prefix"}:
            start_index = index
            break
    if start_index is None:
        return str(text or "")

    kept_lines: List[str] = []
    for raw_line in raw_lines[start_index:]:
        stripped = raw_line.strip()
        if any(stripped.startswith(marker) for marker in _DOSSIER_STOP_MARKERS):
            break
        kept_lines.append(raw_line.rstrip())
    return "\n".join(kept_lines).strip()


def parse_atomic_trace(text: str) -> str:
    lines = [clean_step_line(line) for line in str(text or "").splitlines() if clean_step_line(line)]
    return "\n".join(lines)


def _split_pipe_list(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split("|") if item.strip()]


def _split_claim_list(text: str) -> List[str]:
    return [item.strip("[] ,.") for item in re.split(r"[|,]", str(text or "")) if _CLAIM_ID_RE.search(item)]


def _split_path_list(text: str) -> List[str]:
    return [item.strip("[] ,.") for item in re.split(r"[|,]", str(text or "")) if _PATH_ID_RE.search(item)]


def _split_resolution_path_list(text: str) -> List[str]:
    tokens: List[str] = []
    seen = set()
    for match in _PATH_OR_AGENT_ID_RE.finditer(str(text or "")):
        token = match.group(1).strip()
        if token and token not in seen:
            tokens.append(token)
            seen.add(token)
    if tokens:
        return tokens
    for item in re.split(r"[|,;/]|\band\b", str(text or ""), flags=re.IGNORECASE):
        cleaned = item.strip("[] ,.")
        if _PATH_OR_AGENT_ID_RE.fullmatch(cleaned) and cleaned not in seen:
            tokens.append(cleaned)
            seen.add(cleaned)
    return tokens


def parse_graph_dossier(text: str) -> NaturalLanguageGraph:
    text = _sanitize_graph_dossier_text(text)
    section = ""
    shared_claims: List[ClaimNode] = []
    method_paths: List[MethodPath] = []
    divergences: List[DivergenceCase] = []
    current_claim: ClaimNode | None = None
    current_path: MethodPath | None = None
    current_div: Dict[str, str] | None = None
    in_path_claim_list = False

    def _agent_tokens(token: str) -> List[str]:
        return _AGENT_ID_RE.findall(str(token or ""))

    def path_for_agent_or_path(token: str) -> str:
        token = str(token or "").strip("[] ")
        if token.startswith("P"):
            return token
        tokens = _agent_tokens(token)
        if len(tokens) > 1:
            token_set = set(tokens)
            for path in method_paths:
                if token_set.issubset(set(path.agent_ids or [])):
                    return path.path_id
            token = tokens[0]
        for path in method_paths:
            if token in (path.agent_ids or []):
                return path.path_id
        path_id = f"P{len(method_paths) + 1}"
        method_paths.append(
            MethodPath(
                path_id=path_id,
                agent_ids=[token] if token else [],
                summary=f"{token} path" if token else "path",
                claim_ids=[claim.claim_id for claim in shared_claims],
            )
        )
        return path_id

    def flush_divergence() -> None:
        nonlocal current_div
        if not current_div:
            return
        divergences.append(
            DivergenceCase(
                divergence_id=current_div.get("divergence_id", ""),
                frontier_claim_id=current_div.get("frontier", ""),
                relation=current_div.get("relation", ""),
                left_path_id=current_div.get("left_path", ""),
                right_path_id=current_div.get("right_path", ""),
                left_claim=current_div.get("left_claim", ""),
                right_claim=current_div.get("right_claim", ""),
                why_minimal=current_div.get("why_minimal", ""),
                claim_object=current_div.get("claim_object", ""),
                aspect=current_div.get("aspect", ""),
                alignment=current_div.get("alignment", ""),
            )
        )
        current_div = None

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {"Shared Claims", "Common Ground", "Shared Prefix", "Common Prefix"}:
            flush_divergence()
            section = "shared"
            current_claim = None
            current_path = None
            in_path_claim_list = False
            continue
        if stripped in {"Method Paths", "Paths", "Path Notes", "Current Paths"}:
            flush_divergence()
            section = "paths"
            current_path = None
            in_path_claim_list = False
            continue
        if stripped in {"Path Summaries", "Path Summary", "Agent Paths"}:
            flush_divergence()
            section = "paths"
            current_path = None
            in_path_claim_list = False
            continue
        if stripped in {"Minimal Divergences", "First Split", "First Conflict", "First Real Split"}:
            flush_divergence()
            section = "divergences"
            current_div = None
            in_path_claim_list = False
            continue
        if stripped in {"First Divergence", "First Split", "First Real Split", "First Conflict"}:
            flush_divergence()
            section = "divergences"
            current_div = None
            in_path_claim_list = False
            continue

        claim_match = _CLAIM_RE.match(stripped)
        if section == "shared" and claim_match:
            claim_body = claim_match.group(2).strip()
            current_claim = ClaimNode(claim_id=claim_match.group(1), text=claim_body)
            inline_meta = _CLAIM_INLINE_META_RE.match(claim_body)
            if inline_meta:
                current_claim.text = inline_meta.group("claim").strip()
                current_claim.claim_object = inline_meta.group("object").strip()
                current_claim.aspect = inline_meta.group("aspect").strip()
                current_claim.status = inline_meta.group("status").strip()
                current_claim.alignment = _normalize_alignment_label(inline_meta.group("alignment"))
                current_claim.members = _split_pipe_list(inline_meta.group("members"))
            shared_claims.append(current_claim)
            continue
        if section == "shared" and stripped.startswith("- "):
            claim_id = f"C{len(shared_claims) + 1}"
            claim_body = stripped[2:].strip()
            current_claim = ClaimNode(
                claim_id=claim_id,
                text=claim_body,
                claim_object="local claim",
                aspect="claim",
                status="stated",
                alignment="synchronous",
            )
            inline_meta = _CLAIM_INLINE_META_RE.match(claim_body)
            if inline_meta:
                current_claim.text = inline_meta.group("claim").strip()
                current_claim.claim_object = inline_meta.group("object").strip()
                current_claim.aspect = inline_meta.group("aspect").strip()
                current_claim.status = inline_meta.group("status").strip()
                current_claim.alignment = _normalize_alignment_label(inline_meta.group("alignment"))
                current_claim.members = _split_pipe_list(inline_meta.group("members"))
            shared_claims.append(current_claim)
            continue
        if section == "paths":
            path_match = _PATH_RE.match(stripped)
            if path_match:
                path_body = path_match.group(2).strip()
                inline_meta = _PATH_INLINE_META_RE.match(path_body)
                if inline_meta:
                    agents_text = inline_meta.group("agents").strip()
                    agent_ids = [item.strip() for item in re.split(r",| and ", agents_text) if item.strip()]
                    claim_ids = _split_claim_list(inline_meta.group("claims"))
                    current_path = MethodPath(
                        path_id=path_match.group(1),
                        agent_ids=agent_ids,
                        summary=inline_meta.group("summary").strip(),
                        claim_ids=claim_ids,
                    )
                    method_paths.append(current_path)
                    in_path_claim_list = False
                    continue
                current_path = MethodPath(
                    path_id=path_match.group(1),
                    agent_ids=[item.strip() for item in re.sub(r"^Agents:\s*", "", path_match.group(2)).split(",") if item.strip()],
                    summary="",
                    claim_ids=[],
                )
                method_paths.append(current_path)
                in_path_claim_list = False
                continue
            claim_ref_match = _CLAIM_REF_RE.match(stripped)
            if claim_ref_match and current_path is not None and in_path_claim_list:
                current_path.claim_ids.append(claim_ref_match.group(1))
                continue
            if stripped.startswith("- "):
                path_body = stripped[2:].strip()
                agent_ids = _AGENT_ID_RE.findall(path_body)
                if not agent_ids:
                    agent_ids = [f"A{len(method_paths) + 1}"]
                path_id = f"P{len(method_paths) + 1}"
                current_path = MethodPath(
                    path_id=path_id,
                    agent_ids=agent_ids,
                    summary=path_body,
                    claim_ids=[claim.claim_id for claim in shared_claims],
                )
                method_paths.append(current_path)
                in_path_claim_list = False
                continue
        if section == "divergences":
            div_match = _DIVERGENCE_RE.match(stripped)
            if div_match:
                flush_divergence()
                current_div = {"divergence_id": div_match.group(1)}
                continue
            natural_div = _NATURAL_DIV_RE.search(stripped)
            if natural_div:
                flush_divergence()
                left_path = path_for_agent_or_path(natural_div.group("left"))
                right_path = path_for_agent_or_path(natural_div.group("right"))
                left_claim = re.sub(r"\s*Why this matters:.*$", "", natural_div.group("left_claim").strip(), flags=re.IGNORECASE)
                right_claim = re.sub(r"\s*Why this matters:.*$", "", natural_div.group("right_claim").strip(), flags=re.IGNORECASE)
                current_div = {
                    "divergence_id": f"D{len(divergences) + 1}",
                    "frontier": shared_claims[-1].claim_id if shared_claims else "",
                    "left_path": left_path,
                    "right_path": right_path,
                    "left_claim": left_claim,
                    "right_claim": right_claim,
                    "claim_object": "requested object",
                    "aspect": "local conclusion",
                    "alignment": "same_object_conflict",
                    "relation": "contradiction",
                    "why_minimal": stripped,
                }
                continue
            if current_div is not None:
                inline_div = _DIV_INLINE_META_RE.match(stripped)
                if inline_div:
                    current_div["frontier"] = (inline_div.group("frontier") or current_div.get("frontier", "")).strip()
                    current_div["left_path"] = inline_div.group("left_path").strip()
                    current_div["right_path"] = inline_div.group("right_path").strip()
                    current_div["left_claim"] = inline_div.group("left_claim").strip()
                    current_div["right_claim"] = inline_div.group("right_claim").strip()
                    current_div["claim_object"] = inline_div.group("object").strip()
                    current_div["aspect"] = inline_div.group("aspect").strip()
                    current_div["alignment"] = _normalize_alignment_label(inline_div.group("alignment"))
                    current_div["relation"] = inline_div.group("relation").strip()
                    current_div["why_minimal"] = inline_div.group("why").strip()
                    continue

        field_match = _FIELD_RE.match(stripped)
        if not field_match:
            continue
        key = field_match.group(1)
        value = field_match.group(2).strip()

        if section == "shared" and current_claim is not None:
            if key == "Members":
                current_claim.members = _split_pipe_list(value)
            elif key == "Object":
                current_claim.claim_object = value
            elif key == "Aspect":
                current_claim.aspect = value
            elif key == "Status":
                current_claim.status = value
            elif key == "Alignment":
                current_claim.alignment = _normalize_alignment_label(value)
        elif section == "paths" and current_path is not None:
            if key == "Summary":
                current_path.summary = value
            elif key == "Claims":
                in_path_claim_list = True
        elif section == "divergences" and current_div is not None:
            mapping = {
                "Frontier": "frontier",
                "Relation": "relation",
                "Left path": "left_path",
                "Right path": "right_path",
                "Left claim": "left_claim",
                "Right claim": "right_claim",
                "Why minimal": "why_minimal",
                "Object": "claim_object",
                "Aspect": "aspect",
                "Alignment": "alignment",
            }
            if key in mapping:
                if key in {"Frontier", "Left path", "Right path"}:
                    current_div[mapping[key]] = value.strip("[]")
                elif key == "Alignment":
                    current_div[mapping[key]] = _normalize_alignment_label(value)
                else:
                    current_div[mapping[key]] = value

    flush_divergence()
    return NaturalLanguageGraph(
        shared_claims=shared_claims,
        method_paths=method_paths,
        divergences=divergences,
        raw_dossier=str(text or ""),
    )


def parse_resolution_note(text: str, default_divergence_id: str = "") -> ResolutionDecision:
    fields: Dict[str, str] = {}
    json_payload = _first_json_object(text)
    if json_payload:
        fields["Action"] = _json_string_value(json_payload, "action", "decision")
        fields["Winning side"] = _json_string_value(json_payload, "winning_side", "winningSide", "winning_path", "winning_branch")
        fields["First conflict"] = _json_string_value(json_payload, "first_conflict", "firstConflict")
        fields["Resolved claim"] = _json_string_value(
            json_payload,
            "correct_claim",
            "corrected_claim",
            "resolved_claim",
            "repaired_claim",
            "claim",
        )
        fields["Rationale"] = _json_string_value(json_payload, "reason", "rationale", "why")
        fields["Rewrite from"] = _json_string_value(json_payload, "rewrite_from", "rewriteFrom", "rewrite_frontier", "rewrite_point")
        fields["Keep paths"] = _json_list_value(json_payload, "keep_paths", "keepPaths", "paths_to_keep")
        fields["Drop paths"] = _json_list_value(json_payload, "drop_paths", "dropPaths", "paths_to_drop")
        fields["Canonical answer"] = _json_string_value(json_payload, "canonical_answer", "canonicalAnswer", "final_answer")
    alias_groups = {
        "Action": ["Action", "Decision"],
        "Winning side": ["Winning side", "Winning path", "Winning branch"],
        "First conflict": ["First conflict"],
        "Resolved claim": ["Resolved claim", "Corrected claim", "Correct claim"],
        "Rationale": ["Rationale", "Reason", "Why", "Why the other side fails"],
        "Rewrite from": ["Rewrite from", "Rewrite frontier", "Rewrite claim", "Rewrite point"],
        "Keep paths": ["Keep paths", "Keep path", "Paths to keep"],
        "Drop paths": ["Drop paths", "Drop path", "Paths to drop"],
        "Canonical answer": ["Canonical answer", "Canonical final answer", "Final answer"],
    }
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        match = _FIELD_RE.match(stripped)
        if not match:
            match = _FIELD_LINE_FLEX_RE.match(stripped)
        if not match:
            continue
        key = _canonical_field_name(match.group(1))
        value = match.group(2).strip()
        if key not in fields or value:
            fields[key] = value

    raw_text = str(text or "")
    for canonical_key, aliases in alias_groups.items():
        if not fields.get(canonical_key):
            fallback_value = _search_field_value(raw_text, aliases)
            if fallback_value:
                fields[canonical_key] = fallback_value
    for canonical_key, pattern in _RESOLUTION_SENTENCE_PATTERNS.items():
        if not fields.get(canonical_key):
            match = pattern.search(raw_text)
            if match:
                fields[canonical_key] = match.group(1).strip()

    raw_action = fields.get("Action", "").strip()
    if not raw_action:
        raw_action = _extract_first(_ACTION_TOKEN_RE, raw_text)
    action = _normalize_action(raw_action)
    winning_side = _normalize_winning_side(fields.get("Winning side", ""))
    if action not in {"choose_left", "choose_right", "keep_parallel", "synthesize"}:
        action_from_side = _action_from_winning_side(winning_side)
        if action_from_side:
            action = action_from_side
        elif action in {"keep", "resolve", "repair"}:
            action = ""
    if not winning_side:
        if action == "choose_left":
            winning_side = "left"
        elif action == "choose_right":
            winning_side = "right"
        elif action == "keep_parallel":
            winning_side = "both"
        elif action == "synthesize":
            winning_side = "synthesized"
        else:
            winning_side = _normalize_winning_side(_extract_first(_WINNING_TOKEN_RE, raw_text))

    resolved_claim = fields.get("Resolved claim", "").strip()
    rationale = fields.get("Rationale", "").strip()
    rewrite_from = fields.get("Rewrite from", "").strip()
    if not rewrite_from:
        rewrite_from = _extract_first(_CLAIM_OR_STEP_ID_RE, raw_text)
    keep_paths_raw = fields.get("Keep paths", "").strip()
    drop_paths_raw = fields.get("Drop paths", "").strip()
    canonical_answer = fields.get("Canonical answer", "").strip()
    keep_paths = _split_resolution_path_list(keep_paths_raw)
    drop_paths = _split_resolution_path_list(drop_paths_raw)
    if not keep_paths:
        keep_paths = [item.strip("[] ") for item in _split_pipe_list(keep_paths_raw)]
    if not drop_paths:
        drop_paths = [item.strip("[] ") for item in _split_pipe_list(drop_paths_raw)]
    if not keep_paths:
        keep_paths = _PATH_ID_RE.findall(raw_text if action == "keep_parallel" else keep_paths_raw)

    if not resolved_claim:
        if action == "choose_left":
            left_match = re.search(r"Left claim:\s*(.+)", raw_text, flags=re.IGNORECASE)
            resolved_claim = left_match.group(1).strip() if left_match else ""
        elif action == "choose_right":
            right_match = re.search(r"Right claim:\s*(.+)", raw_text, flags=re.IGNORECASE)
            resolved_claim = right_match.group(1).strip() if right_match else ""

    return ResolutionDecision(
        divergence_id=default_divergence_id,
        action=action,
        winning_side=winning_side,
        resolved_claim=resolved_claim,
        rationale=rationale,
        rewrite_from_claim_id=rewrite_from.strip("[]"),
        keep_paths=[item for item in keep_paths if item],
        drop_paths=[item for item in drop_paths if item],
        canonical_answer=canonical_answer,
        raw_action=raw_action.strip(),
    )


def parse_resolution_note_json_only(text: str, default_divergence_id: str = "") -> ResolutionDecision:
    json_payload = _first_json_object(text)
    if not json_payload:
        return ResolutionDecision(
            divergence_id=default_divergence_id,
            action="",
            winning_side="",
            resolved_claim="",
            rationale="",
            rewrite_from_claim_id="",
        )
    raw_action = _json_string_value(json_payload, "action", "decision")
    action = _normalize_action(raw_action)
    winning_side = _normalize_winning_side(
        _json_string_value(json_payload, "winning_side", "winningSide", "winning_path", "winning_branch")
    )
    if action not in {"choose_left", "choose_right", "keep_parallel", "synthesize"}:
        action_from_side = _action_from_winning_side(winning_side)
        if action_from_side:
            action = action_from_side
        elif action in {"keep", "resolve", "repair"}:
            action = ""
    if not winning_side:
        if action == "choose_left":
            winning_side = "left"
        elif action == "choose_right":
            winning_side = "right"
        elif action == "keep_parallel":
            winning_side = "both"
        elif action == "synthesize":
            winning_side = "synthesized"
    keep_paths = _split_resolution_path_list(_json_list_value(json_payload, "keep_paths", "keepPaths", "paths_to_keep"))
    drop_paths = _split_resolution_path_list(_json_list_value(json_payload, "drop_paths", "dropPaths", "paths_to_drop"))
    return ResolutionDecision(
        divergence_id=default_divergence_id,
        action=action,
        winning_side=winning_side,
        resolved_claim=_json_string_value(
            json_payload,
            "correct_claim",
            "corrected_claim",
            "resolved_claim",
            "repaired_claim",
            "claim",
        ),
        rationale=_json_string_value(json_payload, "reason", "rationale", "why"),
        rewrite_from_claim_id=_json_string_value(
            json_payload,
            "rewrite_from",
            "rewriteFrom",
            "rewrite_frontier",
            "rewrite_point",
        ),
        keep_paths=[item for item in keep_paths if item],
        drop_paths=[item for item in drop_paths if item],
        canonical_answer=_json_string_value(json_payload, "canonical_answer", "canonicalAnswer", "final_answer"),
        raw_action=raw_action,
    )
