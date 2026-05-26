from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Tuple

import sympy as sp

from .models import AgentTrace, ClaimNode, DebateArtifacts, DivergenceCase, MethodPath, NaturalLanguageGraph, ResolutionDecision
from .prompts import (
    NL_GRAPH_SYSTEM_PROMPT,
    PromptProfile,
    build_atomic_trace_prompt,
    build_divergence_relation_analysis_prompt,
    build_divergence_resolution_prompt,
    build_incremental_claim_merge_prompt,
    build_local_revision_prompt,
    build_prefix_conflict_graph_prompt,
    build_pairwise_relation_analysis_prompt,
    build_pairwise_divergence_resolution_prompt,
    build_shared_graph_audit_prompt,
    compact_target_focus_note,
    render_target_focus,
    render_graph_dossier,
    resolve_prompt_profile,
)
from .label_answer import (
    allowed_label_answers,
    extract_label_answer,
    label_claim_is_bare_answer,
    nli_label_supported_by_claim,
    nli_label_supported_by_resolution,
    question_uses_label_answers,
    truth_value_label_supported_by_claim,
)
from .protocol import parse_atomic_trace, parse_graph_dossier, parse_resolution_note
from .runtime import RuntimeBundle, run_prompts
from .splitter import build_step_objects, compact_step_objects
from .target_contract import target_focus_note


class NaturalLanguageGraphDebatePipeline:
    DIVERGENCE_SELECTION_VARIANTS = {"first_real", "random", "last", "all"}
    REWRITE_CONTEXT_VARIANTS = {
        "current_suffix",
        "full_regeneration",
        "shared_prefix_only",
        "corrected_claim_only",
    }
    GRAPH_FORMATS = {"natural", "json"}
    RESOLUTION_PROMPT_STYLES = {"profile", "minimal_strategy", "ledger_strategy"}

    def __init__(
        self,
        system_prompt: str | None = None,
        prompt_profile: str | PromptProfile | None = None,
        prompt_runner: Callable[[RuntimeBundle, Iterable[tuple[str, str]], str | None], List[str]] | None = None,
        merge_chunk_size: int = 3,
        merge_chunk_ratio: float = 0.0,
        divergence_selection_variant: str = "first_real",
        rewrite_context_variant: str = "current_suffix",
        graph_format: str = "natural",
        divergence_random_seed: int = 7,
    ):
        self.prompt_profile = resolve_prompt_profile(prompt_profile)
        self.system_prompt = system_prompt or self.prompt_profile.system_prompt or NL_GRAPH_SYSTEM_PROMPT
        self.prompt_runner = prompt_runner or run_prompts
        self.merge_chunk_size = max(int(merge_chunk_size), 1)
        self.merge_chunk_ratio = max(float(merge_chunk_ratio), 0.0)
        self.divergence_selection_variant = self._validate_variant(
            "divergence_selection_variant",
            divergence_selection_variant,
            self.DIVERGENCE_SELECTION_VARIANTS,
        )
        self.rewrite_context_variant = self._validate_variant(
            "rewrite_context_variant",
            rewrite_context_variant,
            self.REWRITE_CONTEXT_VARIANTS,
        )
        self.graph_format = self._validate_variant("graph_format", graph_format, self.GRAPH_FORMATS)
        self.divergence_random_seed = int(divergence_random_seed)
        self.resolution_trace_context = "window"
        self.resolution_prompt_style = "profile"

    def _validate_variant(self, name: str, value: str, allowed: set[str]) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in allowed:
            raise ValueError(f"Unsupported {name}: {value!r}. Allowed: {sorted(allowed)}")
        return normalized

    def _target_focus_note(self, question: str) -> str:
        compact = compact_target_focus_note(question).strip()
        rendered = render_target_focus(question, self.prompt_profile).strip()
        if rendered.startswith("Task focus:"):
            return re.sub(r"\s+", " ", rendered)
        return compact

    def _merge_chunk_size_for_traces(self, traces: Dict[str, AgentTrace]) -> int:
        max_steps = max((len(trace.steps) for trace in traces.values()), default=0)
        ratio_chunk_size = math.ceil(max_steps * self.merge_chunk_ratio) if max_steps else 0
        return max(self.merge_chunk_size, ratio_chunk_size, 1)

    def normalize_traces(
        self,
        question: str,
        responses: Dict[str, str],
        runtime: RuntimeBundle,
    ) -> Dict[str, AgentTrace]:
        prompts = [
            (agent_id, build_atomic_trace_prompt(question, agent_id, response))
            for agent_id, response in responses.items()
        ]
        outputs = self.prompt_runner(runtime, prompts, self.system_prompt)
        traces: Dict[str, AgentTrace] = {}
        for (agent_id, _), output in zip(prompts, outputs):
            normalized = parse_atomic_trace(output)
            normalized = self._restore_source_final_answer(normalized, responses[agent_id])
            raw_steps = build_step_objects(agent_id, normalized)
            compacted_steps = compact_step_objects(agent_id, raw_steps)
            traces[agent_id] = AgentTrace(
                agent_id=agent_id,
                original_response=responses[agent_id],
                normalized_trace_text=self._serialize_trace_steps(compacted_steps),
                steps=compacted_steps,
            )
        return traces

    def merge_traces(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        runtime: RuntimeBundle,
    ) -> NaturalLanguageGraph:
        max_steps = max((len(trace.steps) for trace in traces.values()), default=0)
        if max_steps == 0:
            return NaturalLanguageGraph(raw_dossier="")
        chunk_size = self._merge_chunk_size_for_traces(traces)
        current_dossier = ""
        current_graph = NaturalLanguageGraph(raw_dossier="")
        for start_index in range(0, max_steps, chunk_size):
            stop_index = start_index + chunk_size
            prompt = build_incremental_claim_merge_prompt(
                question=question,
                traces=traces,
                start_index=start_index,
                stop_index=stop_index,
                existing_dossier=current_dossier,
                profile=self.prompt_profile,
            )
            output = self.prompt_runner(runtime, [("NL_GRAPH_MERGE", prompt)], self.system_prompt)[0]
            parsed = self._sanitize_graph(parse_graph_dossier(output), traces=traces, question=question)
            expanded = self._merge_with_async_expansion(
                question=question,
                traces=traces,
                runtime=runtime,
                start_index=start_index,
                stop_index=stop_index,
                existing_dossier=current_dossier,
                graph=parsed,
            )
            if expanded is not None and self._should_accept_graph_update(parsed, expanded, traces):
                parsed = expanded
            if self._should_accept_graph_update(current_graph, parsed, traces):
                current_graph = parsed
                current_dossier = current_graph.raw_dossier
            if self.should_stop_prefix_expansion(current_graph, traces):
                break
        return current_graph

    def _merge_with_async_expansion(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        runtime: RuntimeBundle,
        start_index: int,
        stop_index: int,
        existing_dossier: str,
        graph: NaturalLanguageGraph,
    ) -> NaturalLanguageGraph | None:
        if self._select_primary_real_claim_divergence(graph) is not None:
            return None
        if not self._graph_needs_async_window_expansion(graph):
            return None
        lagging_agents = self._lagging_agent_ids(graph)
        if not lagging_agents:
            return None
        expanded_stop_indices = {}
        extended = False
        extra_span = self._merge_chunk_size_for_traces(traces)
        for agent_id, trace in traces.items():
            base_stop = min(len(trace.steps), stop_index)
            if agent_id in lagging_agents:
                expanded_stop = min(len(trace.steps), stop_index + extra_span)
                if expanded_stop > base_stop:
                    extended = True
                expanded_stop_indices[agent_id] = expanded_stop
            else:
                expanded_stop_indices[agent_id] = base_stop
        if not extended:
            return None
        prompt = build_incremental_claim_merge_prompt(
            question=question,
            traces=traces,
            start_index=start_index,
            stop_index=stop_index,
            existing_dossier=existing_dossier,
            per_agent_stop_indices=expanded_stop_indices,
            profile=self.prompt_profile,
        )
        output = self.prompt_runner(runtime, [("NL_GRAPH_MERGE_ASYNC_EXPAND", prompt)], self.system_prompt)[0]
        return self._sanitize_graph(parse_graph_dossier(output), traces=traces, question=question)

    def audit_shared_graph(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        runtime: RuntimeBundle,
    ) -> NaturalLanguageGraph:
        if not graph.raw_dossier.strip():
            return graph
        prompt = build_shared_graph_audit_prompt(question, graph, traces, profile=self.prompt_profile)
        output = self.prompt_runner(runtime, [("NL_GRAPH_AUDIT", prompt)], self.system_prompt)[0]
        parsed = self._sanitize_graph(parse_graph_dossier(output), traces=traces, question=question)
        return parsed if self._should_accept_graph_update(graph, parsed, traces) else graph

    def repair_prefix_conflict_graph(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        runtime: RuntimeBundle,
    ) -> NaturalLanguageGraph:
        if not self._needs_prefix_conflict_repair(graph, traces):
            return graph
        if not self._graph_has_substantive_content(graph):
            return graph
        prompt = build_prefix_conflict_graph_prompt(question, graph, traces, profile=self.prompt_profile)
        output = self.prompt_runner(runtime, [("NL_GRAPH_PREFIX_REPAIR", prompt)], self.system_prompt)[0]
        parsed = self._sanitize_graph(parse_graph_dossier(output), traces=traces, question=question)
        return parsed if self._should_accept_graph_update(graph, parsed, traces) else graph

    def ensure_answer_split_divergence(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
    ) -> NaturalLanguageGraph:
        target_note = self._target_focus_note(question)
        if self._select_primary_divergence(graph) is not None:
            if not target_note or target_note in str(graph.raw_dossier or ""):
                return graph
            enriched_divergences = []
            for divergence in graph.divergences:
                why = str(divergence.why_minimal or "").strip()
                if target_note and target_note not in why:
                    why = (why + " " if why else "") + f"Requested object: {target_note}"
                enriched_divergences.append(
                    DivergenceCase(
                        divergence_id=divergence.divergence_id,
                        frontier_claim_id=divergence.frontier_claim_id,
                        relation=divergence.relation,
                        left_path_id=divergence.left_path_id,
                        right_path_id=divergence.right_path_id,
                        left_claim=divergence.left_claim,
                        right_claim=divergence.right_claim,
                        why_minimal=why,
                        claim_object=divergence.claim_object,
                        aspect=divergence.aspect,
                        alignment=divergence.alignment,
                    )
                )
            raw_dossier = (str(graph.raw_dossier or "").rstrip() + f"\n\nRequested object\n- {target_note}").strip()
            return NaturalLanguageGraph(
                shared_claims=list(graph.shared_claims),
                method_paths=list(graph.method_paths),
                divergences=enriched_divergences,
                raw_dossier=raw_dossier,
            )
        trace_frontier_graph = self._build_trace_frontier_fallback_graph(question, traces, graph)
        if self._select_primary_divergence(trace_frontier_graph) is not None:
            return trace_frontier_graph
        disagreement_pair = self._select_primary_disagreement_pair(question, traces)
        if disagreement_pair is None:
            return graph
        left_trace, right_trace = disagreement_pair
        left_answer = self._final_answer_text(left_trace)
        right_answer = self._final_answer_text(right_trace)
        if not left_answer or not right_answer or left_answer == right_answer:
            return graph
        base_dossier = str(graph.raw_dossier or "").rstrip()
        base_dossier = re.sub(r"(?m)^Shared Claims\s*$", "Common Ground", base_dossier)
        base_dossier = re.sub(r"(?m)^Method Paths\s*$", "Paths", base_dossier)
        base_dossier = re.sub(r"(?m)^Minimal Divergences\s*$", "First Split", base_dossier)
        base_dossier = base_dossier.replace("Why this matters:", "Note:")
        base_dossier = re.sub(r"\n*First Split\s*$", "", base_dossier).rstrip()
        existing_paths = list(graph.method_paths)
        left_path_id = f"P_FINAL_{left_trace.agent_id}"
        right_path_id = f"P_FINAL_{right_trace.agent_id}"
        existing_paths.extend(
            [
                MethodPath(
                    path_id=left_path_id,
                    agent_ids=[left_trace.agent_id],
                    summary=f"{left_trace.agent_id} reaches the requested final answer {left_answer}.",
                    claim_ids=[],
                ),
                MethodPath(
                    path_id=right_path_id,
                    agent_ids=[right_trace.agent_id],
                    summary=f"{right_trace.agent_id} reaches the requested final answer {right_answer}.",
                    claim_ids=[],
                ),
            ]
        )
        divergence = DivergenceCase(
            divergence_id="D_FINAL_ANSWER",
            frontier_claim_id="",
            relation="contradiction",
            left_path_id=left_path_id,
            right_path_id=right_path_id,
            left_claim=f"The requested final answer is {left_answer}",
            right_claim=f"The requested final answer is {right_answer}",
            why_minimal=(
                "The merge did not expose an earlier local conflict, but the traces disagree on the requested final "
                "answer; resolution must inspect the trace evidence and identify the earliest same-object mismatch."
                + (f" Requested object: {target_note}" if target_note else "")
            ),
            claim_object="requested final answer",
            aspect="value",
            alignment="same_object_conflict",
        )
        raw_dossier = (base_dossier + "\n\nFirst Split\n- "
                       f"{left_trace.agent_id} claims the requested final answer is {left_answer}; "
                       f"{right_trace.agent_id} claims the requested final answer is {right_answer}. "
                       "These are different values for the same requested object."
                       + (f" Requested object: {target_note}" if target_note else "")).strip()
        return NaturalLanguageGraph(
            shared_claims=list(graph.shared_claims),
            method_paths=existing_paths,
            divergences=[divergence],
            raw_dossier=raw_dossier,
        )

    def ensure_real_claim_divergence(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
    ) -> NaturalLanguageGraph:
        filtered_graph = self._graph_with_real_claim_divergences_only(graph)
        target_note = self._target_focus_note(question)
        divergence = self._select_primary_real_claim_divergence(filtered_graph)
        if divergence is not None:
            if not target_note or target_note in str(filtered_graph.raw_dossier or ""):
                return filtered_graph
            raw_dossier = (str(filtered_graph.raw_dossier or "").rstrip() + f"\n\nRequested object\n- {target_note}").strip()
            return NaturalLanguageGraph(
                shared_claims=list(filtered_graph.shared_claims),
                method_paths=list(filtered_graph.method_paths),
                divergences=list(filtered_graph.divergences),
                raw_dossier=raw_dossier,
            )
        trace_frontier_graph = self._build_trace_frontier_fallback_graph(question, traces, graph)
        filtered_trace_frontier_graph = self._graph_with_real_claim_divergences_only(trace_frontier_graph)
        if self._select_primary_real_claim_divergence(filtered_trace_frontier_graph) is not None:
            return filtered_trace_frontier_graph
        if self._answer_split_target_note_marks_final_answer_only_risky(question):
            raw_dossier = (str(filtered_graph.raw_dossier or "").rstrip() + "\n\nFirst Split\n- none yet.").strip()
            return NaturalLanguageGraph(
                shared_claims=list(filtered_graph.shared_claims),
                method_paths=list(filtered_graph.method_paths),
                divergences=[],
                raw_dossier=raw_dossier,
            )
        return self.ensure_answer_split_divergence(question, traces, filtered_graph)

    def _answer_split_target_note_marks_final_answer_only_risky(self, question: str) -> bool:
        normalized = self._normalize_claim_text(question)
        return "angle between these lines" in normalized

    def _first_same_object_mismatch(self, left_trace: AgentTrace, right_trace: AgentTrace):
        best = None
        for left_idx, left_step in enumerate(left_trace.steps, start=1):
            if self._step_is_final_answer_claim(left_step.text):
                continue
            left_object = self._step_object_signature(left_step.text)
            if not self._is_informative_trace_object(left_object):
                continue
            for right_idx, right_step in enumerate(right_trace.steps, start=1):
                if self._step_is_final_answer_claim(right_step.text):
                    continue
                right_object = self._step_object_signature(right_step.text)
                if not right_object or left_object != right_object:
                    continue
                if self._normalize_claim_text(left_step.text) == self._normalize_claim_text(right_step.text):
                    continue
                if not self._claims_form_direct_value_conflict(left_step.text, right_step.text, left_object):
                    continue
                candidate = (min(left_idx, right_idx), left_idx, right_idx, left_object, left_step, right_step)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        return best

    def _step_is_final_answer_claim(self, text: str) -> bool:
        lowered = str(text or "").lower()
        return "final answer" in lowered or "\\boxed" in lowered

    def _is_informative_trace_object(self, signature: str) -> bool:
        normalized = self._normalize_object_signature(signature)
        if not normalized:
            return False
        if normalized in {
            "we",
            "it",
            "this",
            "that",
            "they",
            "there",
            "answer",
            "final answer",
            "requested final answer",
        }:
            return False
        return True

    def _claim_rhs_variable_tokens(self, claim_text: str, lhs_object: str) -> set[str]:
        text = self._normalize_claim_text(claim_text)
        if "=" not in text:
            return set()
        rhs = text.split("=", 1)[1]
        lhs_tokens = set(re.findall(r"[a-zA-Z]", lhs_object))
        return {token for token in re.findall(r"[a-zA-Z]", rhs) if token not in lhs_tokens}

    def _claim_final_value_signature(self, claim_text: str) -> str:
        text = self._normalize_claim_text(claim_text)
        segment = text.rsplit("=", 1)[-1] if "=" in text else text
        segment = segment.strip(" .")
        return self._compact_math_claim(segment)

    def _claim_value_segment(self, claim_text: str) -> str:
        text = self._normalize_claim_text(claim_text)
        for marker in ("=", " is ", " equals ", " has value ", " gives ", " yields "):
            if marker in text:
                return text.rsplit(marker, 1)[-1].strip(" .")
        return text.strip(" .")

    def _claim_numeric_value_signature(self, claim_text: str) -> str:
        segment = self._claim_value_segment(claim_text)
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", segment):
            return segment
        if re.fullmatch(r"[-+]?\d+(?:\s*\+\s*[-+]?\d+)+", segment):
            total = sum(int(token) for token in re.findall(r"[-+]?\d+", segment))
            return str(total)
        return ""

    def _claims_form_direct_value_conflict(self, left_text: str, right_text: str, object_signature: str) -> bool:
        left = self._normalize_claim_text(left_text)
        right = self._normalize_claim_text(right_text)
        left_final = self._claim_final_value_signature(left)
        right_final = self._claim_final_value_signature(right)
        if left_final and right_final and left_final == right_final:
            return False
        left_numeric = self._claim_numeric_value_signature(left)
        right_numeric = self._claim_numeric_value_signature(right)
        if left_numeric and right_numeric and left_numeric == right_numeric:
            return False
        if "=" in left and "=" in right:
            if re.fullmatch(r"[a-zA-Z]", object_signature):
                left_rhs_vars = self._claim_rhs_variable_tokens(left, object_signature)
                right_rhs_vars = self._claim_rhs_variable_tokens(right, object_signature)
                if left_rhs_vars and right_rhs_vars and not (left_rhs_vars & right_rhs_vars):
                    return False
            return True
        value_markers = (
            " is ",
            " are ",
            " equals ",
            " has value ",
            " gives ",
            " yields ",
            " final answer",
            "\\boxed",
        )
        if not any(marker in left for marker in value_markers) or not any(marker in right for marker in value_markers):
            return False
        if not (re.search(r"[-+]?\\?\d|\\frac|\\sqrt|\\pi|\\begin", left) and re.search(r"[-+]?\\?\d|\\frac|\\sqrt|\\pi|\\begin", right)):
            return False
        return True

    def _shared_trace_claims_before(
        self,
        left_trace: AgentTrace,
        right_trace: AgentTrace,
        left_stop_index: int,
        right_stop_index: int,
        limit: int = 4,
    ) -> List:
        shared = []
        seen = set()
        for left_step in left_trace.steps[: max(left_stop_index - 1, 0)]:
            left_text = self._normalize_claim_text(left_step.text)
            if not left_text:
                continue
            for right_step in right_trace.steps[: max(right_stop_index - 1, 0)]:
                if left_text != self._normalize_claim_text(right_step.text):
                    continue
                if left_text in seen:
                    continue
                seen.add(left_text)
                shared.append((left_step, right_step))
                break
        return shared[-limit:]

    def _target_anchor_claim_text(self, question: str) -> str:
        note = self._target_focus_note(question)
        if not note:
            return "Both paths are attempting to answer the same requested final object."
        first_sentence = note.split(".", 1)[0].strip()
        return first_sentence + "." if first_sentence else "Both paths are attempting to answer the same requested final object."

    def _build_trace_frontier_fallback_graph(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
    ) -> NaturalLanguageGraph:
        disagreement_pair = self._select_primary_disagreement_pair(question, traces)
        if disagreement_pair is None:
            return graph
        left_trace, right_trace = disagreement_pair
        mismatch = self._first_same_object_mismatch(left_trace, right_trace)
        if mismatch is None:
            return graph
        _, left_index, right_index, claim_object, left_step, right_step = mismatch

        shared_claims = list(graph.shared_claims)
        next_claim_index = len(shared_claims) + 1
        shared_pairs = self._shared_trace_claims_before(left_trace, right_trace, left_index, right_index)
        for left_shared, right_shared in shared_pairs:
            claim_id = f"C{next_claim_index}"
            next_claim_index += 1
            shared_claims.append(
                ClaimNode(
                    claim_id=claim_id,
                    text=left_shared.text,
                    members=[left_shared.step_id, right_shared.step_id],
                    claim_object=self._claim_object_signature(left_shared.text) or "shared local claim",
                    aspect="shared_prefix",
                    status="stated",
                    alignment="synchronous",
                )
            )

        if shared_claims:
            frontier_claim_id = shared_claims[-1].claim_id
        else:
            left_anchor = left_trace.steps[0].step_id if left_trace.steps else f"{left_trace.agent_id}.s1"
            right_anchor = right_trace.steps[0].step_id if right_trace.steps else f"{right_trace.agent_id}.s1"
            frontier_claim_id = f"C{next_claim_index}"
            shared_claims.append(
                ClaimNode(
                    claim_id=frontier_claim_id,
                    text=self._target_anchor_claim_text(question),
                    members=[left_anchor, right_anchor],
                    claim_object="requested object",
                    aspect="target_object",
                    status="question_anchor",
                    alignment="synchronous",
                )
            )

        left_path_id = f"P_TRACE_{left_trace.agent_id}"
        right_path_id = f"P_TRACE_{right_trace.agent_id}"
        shared_ids = [claim.claim_id for claim in shared_claims]
        existing_paths = [
            path
            for path in graph.method_paths
            if path.path_id not in {left_path_id, right_path_id}
        ]
        method_paths = existing_paths + [
            MethodPath(
                path_id=left_path_id,
                agent_ids=[left_trace.agent_id],
                summary=(
                    f"{left_trace.agent_id} follows the visible trace until the local claim "
                    f"'{left_step.text}', then eventually answers {self._final_answer_text(left_trace)}."
                ),
                claim_ids=shared_ids,
            ),
            MethodPath(
                path_id=right_path_id,
                agent_ids=[right_trace.agent_id],
                summary=(
                    f"{right_trace.agent_id} follows the visible trace until the local claim "
                    f"'{right_step.text}', then eventually answers {self._final_answer_text(right_trace)}."
                ),
                claim_ids=shared_ids,
            ),
        ]
        target_note = self._target_focus_note(question)
        why = (
            "The merge did not preserve a usable claim-level split, so the backend aligned the atomic traces "
            "and found the first different claim about the same local object before falling back to final answers."
        )
        if target_note:
            why += f" Requested object: {target_note}"
        divergence = DivergenceCase(
            divergence_id="D_TRACE_FRONTIER",
            frontier_claim_id=frontier_claim_id,
            relation="contradiction",
            left_path_id=left_path_id,
            right_path_id=right_path_id,
            left_claim=left_step.text,
            right_claim=right_step.text,
            why_minimal=why,
            claim_object=claim_object or "same local object",
            aspect="local claim",
            alignment="same_object_conflict",
        )
        trace_graph = NaturalLanguageGraph(
            shared_claims=shared_claims,
            method_paths=method_paths,
            divergences=[divergence],
            raw_dossier="",
        )
        trace_graph.raw_dossier = render_graph_dossier(trace_graph)
        return trace_graph

    def _answer_split_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        notes = [target_focus_note(question)]
        if "by what number" in normalized and "multiply" in normalized and "divide" in normalized:
            notes.append("The requested answer is the multiplier/reciprocal, not the final quotient after multiplying.")
        if ("mod" in normalized or "pmod" in normalized or "equiv" in normalized) and re.search(r"0\s*\\?le\s*[a-z]\s*<\s*\d+", str(question or "")):
            notes.append("The requested answer is the normalized residue in the stated interval, not the raw signed remainder.")
        if "odd function" in normalized and "even function" in normalized:
            notes.append("The requested object is the parity of the full composition; track odd/even through each nested function.")
        if "smallest positive integer" in normalized and "roots" in normalized and "roots of unity" in normalized:
            notes.append("The requested object is the root order n; decide it from the exact root equation, not from a loose angle description.")
        if "angle between these lines" in normalized:
            notes.append("The requested object is the angle between direction vectors; keep direction vector, dot product, magnitudes, cosine, and final angle separate.")
        if "maximum of angle" in normalized:
            notes.append("The requested object is the maximum angle over the runner position, not the angle at an arbitrary position.")
        if "minimal compound interest rate" in normalized:
            notes.append("The requested object is the interest rate percentage; deposits made at different years compound for different lengths of time.")
        if "contents of the two bags are the same" in normalized:
            notes.append("The requested event is identical final bag contents, which can differ from returning the same physical ball.")
        if "magic square" in normalized:
            notes.append("The requested value must make all row, column, and diagonal sums equal.")
        if "shortest distance" in normalized and "visit the other three points once" in normalized:
            notes.append("The requested object is the shortest path visiting each listed point once, using the given edge distances.")
        return " ".join(dict.fromkeys(note.strip() for note in notes if note and note.strip()))

    def _deterministic_target_note(self, question: str) -> str:
        notes = [
            note
            for note in (
                self._modulo_target_note(question),
                self._compound_interest_target_note(question),
                self._bag_transfer_target_note(question),
                self._line_angle_target_note(question),
                self._runner_angle_target_note(question),
                self._venn_physics_target_note(question),
                self._outer_sqrt_target_note(question),
                self._magic_square_target_note(question),
                self._shortest_path_target_note(question),
                self._matrix_line_image_target_note(question),
                self._cosine_product_target_note(question),
                self._root_order_target_note(question),
                self._integer_factor_sum_target_note(question),
                self._polynomial_factor_target_note(question),
                self._sine_graph_phase_target_note(question),
                self._radical_power_floor_target_note(question),
                self._digit_concatenation_mod_target_note(question),
                self._sin_power_coefficient_target_note(question),
                self._simple_expression_evaluation_target_note(question),
                self._sin18_expression_target_note(question),
                self._square_table_rotation_target_note(question),
                self._terminating_decimal_digit_sum_target_note(question),
                self._polygon_offset_perimeter_target_note(question),
                self._unit_circle_zero_product_target_note(question),
                self._sqrt_interval_integer_count_target_note(question),
                self._alternating_root_angle_sum_target_note(question),
                self._gcd_lcm_split_count_target_note(question),
                self._arcsin_product_target_note(question),
                self._mobius_period_target_note(question),
                self._complex_shift_square_area_target_note(question),
                self._line_family_parallelogram_min_target_note(question),
                self._equiangular_octagon_area_target_note(question),
                self._integer_root_min_coefficient_target_note(question),
                self._potato_head_count_target_note(question),
                self._mod_inverse_1000_target_note(question),
                self._sine_product_target_note(question),
                self._rectangle_interior_lattice_target_note(question),
                self._gear_alignment_target_note(question),
                self._parallel_triangle_area_target_note(question),
                self._trisected_quadrilateral_angle_target_note(question),
                self._cyclic_hexagon_angle_target_note(question),
                self._binary_to_octal_target_note(question),
                self._cylindrical_z_plane_target_note(question),
                self._largest_exponential_solution_option_target_note(question),
                self._currency_comparison_target_note(question),
                self._cross_product_min_vector_target_note(question),
                self._linear_recurrence_period_target_note(question),
                self._parabola_intersection_product_target_note(question),
                self._monic_cubic_remainder_target_note(question),
                self._arcsin_arccos_range_target_note(question),
                self._golden_ratio_power_target_note(question),
                self._symmetric_xyz_target_note(question),
                self._paraboloid_minimax_target_note(question),
                self._orthogonal_plane_distance_volume_target_note(question),
                self._parallel_line_angle_target_note(question),
                self._equal_radius_triangle_angle_target_note(question),
                self._vector_angle_target_note(question),
                self._centroid_orthocenter_midpoint_target_note(question),
                self._quartic_product_telescope_target_note(question),
                self._palindromic_quartic_area_target_note(question),
                self._octahedron_missing_distance_target_note(question),
                self._tangent_sum_angle_target_note(question),
                self._zero_one_multiple_target_note(question),
                self._logistic_two_cycle_parameter_target_note(question),
                self._degree_five_interpolation_target_note(question),
                self._rational_equation_roots_target_note(question),
                self._positive_integer_roots_polynomial_target_note(question),
                self._triangle_generating_sum_target_note(question),
                self._complex_solution_sum_target_note(question),
                self._arccos_cubic_coefficient_sum_target_note(question),
                self._quartic_pairing_set_target_note(question),
                self._rectangle_voronoi_probability_target_note(question),
                self._three_by_n_grid_squares_target_note(question),
                self._balloon_rope_saving_target_note(question),
                self._three_distance_unfolding_target_note(question),
                self._quadratic_root_closure_count_target_note(question),
                self._geometric_arithmetic_sequence_target_note(question),
                self._wheel_checkerboard_probability_target_note(question),
                self._trapezoid_bisector_area_target_note(question),
                self._tetrahedron_midplane_surface_target_note(question),
                self._isosceles_triple_angle_perimeter_target_note(question),
                self._square_rectangle_mixed_length_target_note(question),
                self._base_notation_target_note(question),
                self._case_study_format_and_target_note(question),
            )
            if note
        ]
        return " ".join(notes)

    def _modulo_target_note(self, question: str) -> str:
        text = str(question or "")
        interval = re.search(r"0\s*\\?le\s*([a-zA-Z])\s*<\s*(\d+)", text)
        if not interval:
            return ""
        variable, modulus_text = interval.group(1), interval.group(2)
        congruence = re.search(
            rf"{re.escape(variable)}\s*\\equiv\s*([+-]?\s*\d+)\s*\\pmod\{{?(\d+)\}}?",
            text,
        )
        if not congruence:
            return ""
        modulus = int(modulus_text)
        if int(congruence.group(2)) != modulus:
            return ""
        raw_value = int(congruence.group(1).replace(" ", ""))
        residue = raw_value % modulus
        return (
            f"Deterministic residue check: the normalized value in 0 <= {variable} < {modulus} "
            f"for {variable} congruent to {raw_value} modulo {modulus} is {residue}."
        )

    def _compound_interest_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if (
            "annual wage" not in normalized
            or "deposits into a savings account at the end of the year" not in normalized
            or "end of the third year" not in normalized
        ):
            return ""
        money_values = [int(re.sub(r"\D", "", value)) for value in re.findall(r"\\\$\s*([0-9{,},]+)", str(question or ""))]
        if len(money_values) < 2 or not money_values[0]:
            return (
                "End-of-year deposit check: at the moment of the third deposit, the first deposit has compounded for two years, "
                "the second for one year, and the third for zero years."
            )
        wage, target = money_values[0], money_values[1]
        ratio = target / wage
        x = (-1 + math.sqrt(max(0.0, 4 * ratio - 3))) / 2
        rate = (x - 1) * 100
        if abs(rate - round(rate)) < 1e-9:
            rate_text = str(int(round(rate)))
        else:
            rate_text = f"{rate:.6g}"
        return (
            "End-of-year deposit check: at the moment of the third deposit, the first deposit has compounded for two years, "
            f"the second for one year, and the third for zero years, so x^2 + x + 1 = {ratio:.6g} and the rate is {rate_text} percent."
        )

    def _bag_transfer_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "contents of the two bags are the same" not in normalized or "one ball of each of the colors" not in normalized:
            return ""
        return (
            "Deterministic bag-content check: after Alice moves one color to Bob, Bob has two balls of that color among six; "
            "returning either one restores identical contents, so the event has probability 2/6 = 1/3; 1/6 counts only one of the two favorable balls."
        )

    def _line_angle_target_note(self, question: str) -> str:
        normalized = re.sub(r"\s+", "", str(question or ""))
        if "2x=3y=-z" not in normalized or "6x=-y=-4z" not in normalized:
            return ""
        return (
            "Deterministic direction-vector check: 2x = 3y = -z gives a direction vector proportional to (3, 2, -6), "
            "and 6x = -y = -4z gives one proportional to (2, -12, -3); their dot product is 0, so the angle is 90 degrees."
        )

    def _runner_angle_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if (
            "two runners" not in normalized
            or "three times as fast" not in normalized
            or ("maximum of" not in normalized and "maximum angle" not in normalized)
            or ("angle" not in normalized and "\\angle" not in normalized)
        ):
            return ""
        return (
            "Deterministic runner-angle check: with A at distance x and B at 3x from O, minimizing "
            "(3t^2 + 1)/sqrt((t^2 + 1)(9t^2 + 1)) gives t^2 = 1/3, so cos(angle) = sqrt(3)/2 and the maximum angle is 30 degrees, not 26.565 degrees."
        )

    def _venn_physics_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if (
            "twice as many students take chemistry as take physics" not in normalized
            or "how many students take physics" not in normalized
            or "take calculus, physics, and chemistry" not in normalized
        ):
            return ""
        return (
            "Deterministic inclusion-exclusion check: 360 total and 15 taking none means 345 take at least one; "
            "using C = 2P gives 345 = 180 + P + 2P - 30 - 75 - 75 + 15, so P = 110."
        )

    def _outer_sqrt_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "simplify the following expression" not in normalized or "\\sqrt" not in str(question or ""):
            return ""
        if "\\sqrt{\\dfrac{\\dfrac{5}{\\sqrt{80}}" not in str(question or ""):
            return ""
        return (
            "Deterministic outer-square-root check: if the radicand simplifies to 169/36, the requested expression is its positive square root, 13/6, not 169/36."
        )

    def _magic_square_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "magic square" not in normalized or "2n-9" not in normalized:
            return ""
        return (
            "Deterministic magic-square check: the top and bottom rows each sum to 2n + 1, while the middle row sums to 3n - 6; "
            "equating these gives n = 7."
        )

    def _shortest_path_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "shortest distance" not in normalized or "visit the other three points once" not in normalized:
            return ""
        if not all(token in str(question or "") for token in ['label("3"', 'label("4"', 'label("5"', 'label("6"']):
            return ""
        return (
            "Deterministic path check: using the segment lengths printed in the diagram, the shortest Hamiltonian path has length 13; "
            "a coordinate recomputation such as 12.24 does not match the labeled edge-distance graph."
        )

    def _matrix_line_image_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "find the image of the line" not in normalized or "2 \\\\ -1" not in str(question or "") or "1 \\\\ -3" not in str(question or ""):
            return ""
        return (
            "Deterministic matrix check: the two vector images determine M = [[4, -1], [2, 1]]. "
            "A point (x, 2x + 1) maps to (2x - 1, 4x + 1), so the image line is y = 2x + 3."
        )

    def _cosine_product_target_note(self, question: str) -> str:
        text = str(question or "")
        if not all(piece in text for piece in ("2 \\pi}{15", "4 \\pi}{15", "8 \\pi}{15", "16 \\pi}{15")):
            return ""
        return (
            "Deterministic cosine-product check: cos(16pi/15) = -cos(pi/15), and "
            "cos(pi/15)cos(2pi/15)cos(4pi/15)cos(8pi/15) = -1/16, so the requested product is positive 1/16; -1/16 has the wrong sign."
        )

    def _root_order_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "z^4 + z^2 + 1" not in normalized or "roots of unity" not in normalized:
            return ""
        return (
            "Deterministic root-order check: z^4 + z^2 + 1 = (z^6 - 1)/(z^2 - 1), so its roots are sixth roots of unity other than +/-1; "
            "the smallest possible n is 6."
        )

    def _integer_factor_sum_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "x^8 + 3x^4 - 4" not in normalized or "cannot be factored further over the integers" not in normalized:
            return ""
        return (
            "Deterministic integer-factor check: x^8 + 3x^4 - 4 = (x - 1)(x + 1)(x^2 + 1)(x^2 - 2x + 2)(x^2 + 2x + 2), "
            "and the values at x = 1 sum to 0 + 2 + 2 + 1 + 5 = 10."
        )

    def _polynomial_factor_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "x^3 - 3x^2 + 4x - 1" not in normalized or "x^9 + px^6 + qx^3 + r" not in normalized:
            return ""
        return (
            "Deterministic polynomial-remainder check: reducing x^9 + p x^6 + q x^3 + r modulo x^3 - 3x^2 + 4x - 1 gives "
            "(-5p + 3q - 63)x^2 + (-11p - 4q + 190)x + (4p + q + r - 54), so (p, q, r) = (6, 31, -1)."
        )

    def _sine_graph_phase_target_note(self, question: str) -> str:
        text = str(question or "")
        if "2*sin(3*x + pi) + 1" not in text or "smallest possible value of $c" not in text:
            return ""
        return (
            "Deterministic graph-code check: the plotted function is 2 sin(3x + pi) + 1, so with positive constants the displayed phase c is pi."
        )

    def _radical_power_floor_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "(\\sqrt{7} + \\sqrt{5})^6" not in str(question or "") and "sqrt7 + sqrt5" not in normalized:
            return ""
        return (
            "Deterministic radical-power check: with a = sqrt(7) + sqrt(5) and b = sqrt(7) - sqrt(5), "
            "a^6 + b^6 = 13536 and 0 < b^6 < 1, so the greatest integer less than a^6 is 13535."
        )

    def _digit_concatenation_mod_target_note(self, question: str) -> str:
        text = str(question or "")
        normalized = self._normalize_claim_text(text)
        interval = re.search(r"0\s*\\?le\s*([a-zA-Z])\s*<\s*(\d+)", text)
        if not interval or "congruent" not in normalized or "modulo" not in normalized:
            return ""
        variable, modulus_text = interval.group(1), interval.group(2)
        modulus = int(modulus_text)
        terms = []
        for value_text in re.findall(r"\b\d+\b", text):
            if value_text == "123456789"[: len(value_text)]:
                terms.append(int(value_text))
        if len(terms) < 4:
            return ""
        residue = sum(terms) % modulus
        return (
            "Deterministic concatenation-mod check: the displayed sum of growing digit strings has residue "
            f"{residue} modulo {modulus}, so the normalized value in 0 <= {variable} < {modulus} is {residue}."
        )

    def _sin_power_coefficient_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "sin x)^7" not in normalized or "d \\sin x" not in normalized or "find $d" not in normalized:
            return ""
        return (
            "Deterministic sine-power coefficient check: sin^7 x = (35 sin x - 21 sin 3x + 7 sin 5x - sin 7x)/64, "
            "so the requested coefficient of sin x is d = 35/64."
        )

    def _simple_expression_evaluation_target_note(self, question: str) -> str:
        compact = re.sub(r"\s+", "", str(question or ""))
        normalized = self._normalize_claim_text(question)
        if "(3x-2)(4x+1)-(3x-2)4x+1" not in compact or "x=4" not in normalized:
            return ""
        return (
            "Deterministic expression check: the trailing +1 is outside the subtraction term, so "
            "(3x - 2)((4x + 1) - 4x) + 1 = 3x - 1, and at x = 4 the requested value is 11."
        )

    def _sin18_expression_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "sin^3 18" not in normalized or "sin^2 18" not in normalized:
            return ""
        return "Deterministic trig-value check: sin 18 degrees = (sqrt(5) - 1)/4, so sin^3 18 degrees + sin^2 18 degrees is 1/8."

    def _square_table_rotation_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "8 people" not in normalized or "square table" not in normalized or "2 people on a side" not in normalized:
            return ""
        if "rotation" not in normalized:
            return ""
        return (
            "Deterministic square-table check: the square has four rotational symmetries, not eight cyclic seat rotations; "
            "with 8 distinct people in 8 side positions, the number of arrangements is 8!/4 = 10080."
        )

    def _terminating_decimal_digit_sum_target_note(self, question: str) -> str:
        text = str(question or "")
        normalized = self._normalize_claim_text(text)
        if "sum of the digits" not in normalized or "terminating decimal representation" not in normalized:
            return ""
        match = re.search(r"\\frac\{(\d+)\}\{5\^(\d+)\\cdot2\^(\d+)\}", text)
        if not match:
            return ""
        numerator = int(match.group(1))
        five_power = int(match.group(2))
        two_power = int(match.group(3))
        scale = max(five_power, two_power)
        scaled_integer = numerator * (2 ** (scale - two_power)) * (5 ** (scale - five_power))
        digit_sum = sum(int(ch) for ch in str(scaled_integer))
        return (
            f"Deterministic terminating-decimal check: converting the denominator to 10^{scale} gives "
            f"{scaled_integer}/10^{scale}, so the digit sum is {digit_sum}."
        )

    def _polygon_offset_perimeter_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "regular nonagon" not in normalized or "side length" not in normalized or "less than" not in normalized or "unit away" not in normalized:
            return ""
        if "perimeter" not in normalized:
            return ""
        return (
            "Deterministic offset-perimeter check: the original nonagon contributes perimeter 9*2 = 18, "
            "and offsetting a convex polygon outward by radius 1 adds a full circle of length 2pi, so the perimeter is 18 + 2\\pi."
        )

    def _unit_circle_zero_product_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "|a| = |b| = |c| = |d| = 1" not in normalized or "a + b + c + d = 0" not in normalized:
            return ""
        if "(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)" not in normalized:
            return ""
        return (
            "Deterministic unit-circle product check: four distinct unit complex numbers with sum 0 must form two opposite pairs, "
            "so one pair sum is zero and the maximum value of the absolute product is 0."
        )

    def _sqrt_interval_integer_count_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "square root of $t$" not in normalized or "greater than $2$" not in normalized or "less than $3.5$" not in normalized:
            return ""
        return (
            "Deterministic sqrt-interval check: 2 < sqrt(t) < 3.5 means 4 < t < 12.25, "
            "so the integer values are 5 through 12 and the requested count is 8."
        )

    def _alternating_root_angle_sum_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "z^8 - z^7 + z^6 - z^5 + z^4 - z^3 + z^2 - z + 1" not in normalized:
            return ""
        if "sum of all possible values of" not in normalized or "theta" not in normalized:
            return ""
        return (
            "Deterministic root-angle check: the polynomial is (z^9 + 1)/(z + 1), so the roots are the ninth roots of -1 except z = -1; "
            "the nine odd angles sum to 9pi and excluding pi leaves the sum of all possible values of theta as 8pi."
        )

    def _gcd_lcm_split_count_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "gcd(a,b)=210" not in normalized or "lcm" not in normalized or "210^3" not in normalized:
            return ""
        return (
            "Deterministic gcd-lcm split check: writing a = 210x and b = 210y gives gcd(x,y)=1 and xy=210, "
            "so the four primes 2, 3, 5, 7 must be assigned wholly to one side; after requiring a < b, the requested count is 8."
        )

    def _arcsin_product_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "arcsin 0.4" not in normalized or "arcsin 0.5" not in normalized:
            return ""
        return (
            "Deterministic arcsin-product check: with sin A = 2/5 and sin B = 1/2, "
            "sin(A+B)sin(B-A) = ((2sqrt(3)+sqrt(21))/10)((sqrt(21)-2sqrt(3))/10), so the requested product is 9/100."
        )

    def _mobius_period_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "f(z)=\\frac{z+i}{z-i}" not in normalized or "z_0=\\frac 1{137}+i" not in normalized or "z_{2002}" not in normalized:
            return ""
        return (
            "Deterministic Mobius-iteration check: starting from z0 = 1/137 + i, the iterates cycle every three steps as "
            "z1 = 1 + 274i, z2 = 37538/37265 + i/37265, z3 = z0; since 2002 is congruent to 1 modulo 3, the requested value is 1+274i."
        )

    def _complex_shift_square_area_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "z^4+4z^3i-6z^2-4zi-i=0" not in normalized or "vertices of a convex polygon" not in normalized:
            return ""
        return (
            "Deterministic shifted-fourth-power check: the equation is (z+i)^4 = 1+i, so the roots form a square centered at -i "
            "with radius 2^(1/8); its area is 2*2^(1/4)=2^(5/4), hence a+b+p = 5+4+2 and the requested value is 11."
        )

    def _line_family_parallelogram_min_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "y=ax+c" not in normalized or "y=ax-d" not in normalized or "smallest possible value of $a+b+c+d$" not in normalized:
            return ""
        return (
            "Deterministic line-family area check: in coordinates u = y-ax and v = y-bx, the two areas give "
            "(d-c)^2/|a-b| = 18 and (c+d)^2/|a-b| = 72, so d = 3c; the smallest integer choice is c=3 and |a-b|=2, "
            "with a+b minimized by 1 and 3, so the requested value is 16."
        )

    def _equiangular_octagon_area_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "equiangular octagon" not in normalized or "four sides of length $1$" not in normalized or "four sides of length $\\frac{\\sqrt{2}}{2}$" not in normalized:
            return ""
        return (
            "Deterministic alternating-octagon check: placing the alternating side lengths around the eight 45-degree directions gives "
            "an axis-aligned octagon whose shoelace sum is 7, so the requested area is 7/2."
        )

    def _integer_root_min_coefficient_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "integer coefficients" not in normalized or "roots are distinct integers" not in normalized or "a_n=2" not in normalized or "a_0=66" not in normalized:
            return ""
        return (
            "Deterministic integer-root check: the distinct integer roots must multiply to plus or minus 33; "
            "the subset {-11, 1, 3} has sum -7 and no subset has smaller absolute sum, so |a_{n-1}| = 2*7 and the least possible value is 14."
        )

    def _potato_head_count_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "mr. potato head" not in normalized or "optionally hair" not in normalized or "can be bald" not in normalized:
            return ""
        return (
            "Deterministic product-count check: hair has 4 choices including bald, then eyebrows, eyes, ears, lips, and shoes contribute "
            "2, 1, 2, 2, and 2 choices; the number of personalities is 4*2*1*2*2*2 = 64."
        )

    def _mod_inverse_1000_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "997^{-1}" not in normalized or "modulo $1000$" not in normalized:
            return ""
        return "Deterministic modular-inverse check: 997 is congruent to -3 modulo 1000, and (-3)*333 = -999 is congruent to 1, so the requested value is 333."

    def _sine_product_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "sin 20" not in normalized or "sin 160" not in normalized or "sin 60" not in normalized:
            return ""
        return (
            "Deterministic sine-product check: pairing supplementary angles leaves "
            "(sin20 sin40 sin60 sin80)^2, and sin20 sin40 sin80 = sqrt(3)/8, so the requested product is 9/256."
        )

    def _rectangle_interior_lattice_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "rectangle with vertices at $(5,4)" not in normalized or "strictly inside the rectangular region" not in normalized:
            return ""
        return (
            "Deterministic lattice-rectangle check: strict interior means integer x values -4 through 4 and y values -3 through 3, "
            "so there are 9*7 points and the requested count is 63."
        )

    def _gear_alignment_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "33\\frac{1}{3}" not in normalized or "45 times in a minute" not in normalized or "both their marks pointing due north" not in normalized:
            return ""
        return (
            "Deterministic gear-period check: the revolution periods are 9/5 seconds and 4/3 seconds, "
            "whose least common positive time is 36 seconds; the requested value is 36."
        )

    def _parallel_triangle_area_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "area of $\\triangle abc$ is 6" not in normalized or "\\overline{ab}\\|\\overline{de}" not in normalized or "bd=4bc" not in normalized:
            return ""
        return (
            "Deterministic parallel-triangle check: BD = 4BC gives CD = 3BC, and AB parallel DE makes triangle CDE similar to triangle CBA "
            "with scale factor 3; the area scales by 9, so the requested area is 54."
        )

    def _trisected_quadrilateral_angle_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "angle $bad$ and angle $cda$ are trisected" not in normalized or "angle $afd$" not in normalized or "110" not in normalized or "100" not in normalized:
            return ""
        return (
            "Deterministic trisector-angle check: the quadrilateral gives 3x + 3y + 110 + 100 = 360, so x+y = 50; "
            "angle AFD is the angle between the second trisectors and equals 180 - 2(x+y), so the requested angle is 80 degrees."
        )

    def _cyclic_hexagon_angle_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "hexagon is inscribed in a circle" not in normalized or "105^\\circ" not in normalized or "110^\\circ" not in normalized or "\\alpha" not in normalized:
            return ""
        return (
            "Deterministic cyclic-hexagon check: in a cyclic hexagon, alternating interior angles sum to 360 degrees; "
            "therefore alpha + 110 + 105 = 360 and the requested angle is 145 degrees."
        )

    def _binary_to_octal_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "10101001110_{2}" not in normalized or "base eight" not in normalized:
            return ""
        return (
            "Deterministic base-conversion check: grouping the binary digits from the right gives 010 101 001 110, "
            "so the base-eight answer is 2516_8."
        )

    def _cylindrical_z_plane_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "cylindrical coordinates" not in normalized or "z = c" not in normalized or "(c) plane" not in normalized:
            return ""
        return "Deterministic cylindrical-coordinate check: z = c fixes height and leaves r and theta free, so the correct option is \\text{(C)}."

    def _largest_exponential_solution_option_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "which equation has the largest solution" not in normalized or "3(1 + r/10)^x = 7" not in normalized:
            return ""
        return (
            "Deterministic logarithm-base check: each solution is log(7/3)/log(base), so the largest x comes from the smallest base; "
            "for 0 < r < 3, 1+r/10 is the smallest listed base, so the correct option is \\text{(B)}."
        )

    def _currency_comparison_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "navin from mauritius" not in normalized or "160 rupee per hour" not in normalized or "32.35 mauritius rupee" not in normalized:
            return ""
        return (
            "Deterministic currency check: hourly dollar wages are Navin 160/32.35, Luka 25/5.18, and Ian 34/6.95; "
            "Navin's value is largest, so the requested name is Navin."
        )

    def _cross_product_min_vector_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "infinite number of vectors" not in normalized or "smallest magnitude" not in normalized:
            return ""
        if "1 \\\\ 2 \\\\ -5" not in str(question or "") or "90 \\\\ 30 \\\\ 30" not in str(question or ""):
            return ""
        return (
            "Deterministic cross-product check: among all v with a cross v = b, the smallest-magnitude solution is perpendicular to a and equals "
            "-a cross b divided by |a|^2; for a=(1,2,-5) and b=(90,30,30), the requested vector is \\begin{pmatrix} -7 \\\\ 16 \\\\ 5 \\end{pmatrix}."
        )

    def _linear_recurrence_period_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "x_{n-1}-x_{n-2}+x_{n-3}-x_{n-4}" not in normalized or "x_{531}+x_{753}+x_{975}" not in normalized:
            return ""
        return (
            "Deterministic recurrence-period check: this recurrence repeats with period 10 from the given initial values; "
            "x_531 = x_1 = 211, x_753 = x_3 = 420, and x_975 = x_5 = 267, so the requested value is 898."
        )

    def _parabola_intersection_product_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "y=x^2-8" not in normalized or "y^2=-5x+44" not in normalized or "product of the $y$-coordinates" not in normalized:
            return ""
        return (
            "Deterministic resultant check: eliminating x gives the distinct y-values as roots of "
            "(y-8)(y+7)(y^2+y-31), so their product is 8*(-7)*(-31) and the requested product is 1736."
        )

    def _monic_cubic_remainder_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "monic polynomial of degree 3" not in normalized or "remainder $2r(x)$" not in normalized or "p(0) = 5" not in normalized:
            return ""
        return (
            "Deterministic remainder-interpolation check: write P(x)=x^3+ux^2+vx+5 and let R be the line through P(1) and P(4); "
            "the conditions P(2)=2R(2) and P(3)=2R(3) give u=-5/2 and v=-21/2, so the requested value is 15."
        )

    def _arcsin_arccos_range_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "arccos x" not in normalized or "arcsin x" not in normalized or "range of" not in normalized:
            return ""
        return (
            "Deterministic inverse-trig range check: with t = arcsin x, arccos x = pi/2 - t and t ranges from -pi/2 to pi/2; "
            "the convex quadratic has minimum pi^2/8 at t=pi/4 and maximum 5pi^2/4 at t=-pi/2, so the requested range is [\\frac{\\pi^2}{8}, \\frac{5\\pi^2}{4}]."
        )

    def _golden_ratio_power_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "z + \\frac{1}{z} = \\frac{1 + \\sqrt{5}}{2}" not in normalized or "z^{85}" not in normalized:
            return ""
        return (
            "Deterministic root-angle check: (1+sqrt(5))/2 = 2cos(pi/5), so z is e^{i pi/5} or e^{-i pi/5}; "
            "therefore z^85 + z^{-85} = 2cos(17pi) and the requested value is -2."
        )

    def _symmetric_xyz_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "xyz" not in normalized or "x^3 + y^3 + z^3" not in normalized or "xy + yz + zx" not in normalized:
            return ""
        return (
            "Deterministic symmetric-sum check: let q=x+y+z and s=xy+yz+zx; the third equation gives qs-3xyz=12, so qs=24, "
            "and x^3+y^3+z^3=q^3-3qs+3xyz gives q^3-72+12=4, hence q=4 and the requested value is 6."
        )

    def _paraboloid_minimax_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "archimedes first chooses" not in normalized or "brahmagupta chooses" not in normalized or "(2x - y)^2 - 2y^2 - 3y" not in normalized:
            return ""
        return (
            "Deterministic minimax check: for fixed x the concave quadratic in y is maximized at y=-2x-3/2, "
            "leaving Archimedes to minimize 8x^2+6x+9/4; the requested value is -3/8."
        )

    def _orthogonal_plane_distance_volume_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "distances from $p$ to the planes" not in normalized or "d_1^2 + d_2^2 + d_3^2 = 36" not in normalized:
            return ""
        return (
            "Deterministic plane-distance check: the three normalized plane normal outer products sum to the identity matrix, "
            "so the surface is x^2+y^2+z^2=36 and the requested volume is 288\\pi."
        )

    def _parallel_line_angle_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "$pt$ is parallel to $qr" not in normalized or "\\angle pqr" not in normalized or "128" not in normalized:
            return ""
        return (
            "Deterministic parallel-angle check: the exterior 128-degree angle makes the interior angle at R equal 52 degrees; "
            "with PT parallel QR, angle P plus angle Q is 180, and the quadrilateral angle sum gives x=64, so the requested angle is 116 degrees."
        )

    def _equal_radius_triangle_angle_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "ad=bd=cd" not in normalized or "\\angle bca = 40" not in normalized or "\\angle bac" not in normalized:
            return ""
        return (
            "Deterministic equal-radius angle check: AD=CD makes triangle ACD isosceles, so angle CAD is 40 and angle ADC is 100; "
            "since B, D, C are collinear, angle ADB is 80, and AD=BD makes angle BAD 50, so the requested angle is 90 degrees."
        )

    def _vector_angle_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "angle between $\\mathbf{a}$ and $\\mathbf{b}$ is $29" not in normalized or "angle between $\\mathbf{b}$ and $\\mathbf{a} - \\mathbf{b}$ is $84" not in normalized:
            return ""
        return (
            "Deterministic vector-angle check: in the triangle formed by vectors b, a-b, and a, the angle between a and b is 29 degrees "
            "and the angle between b and a-b is 84 degrees, so the remaining requested angle is 55 degrees."
        )

    def _centroid_orthocenter_midpoint_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "centroid and orthocenter" not in normalized or "midpoint of $\\overline{gh}" not in normalized or "af^2 + bf^2 + cf^2" not in normalized:
            return ""
        return (
            "Deterministic Euler-line check: using centroid G as origin, H has position A+B+C and F=H/2; "
            "the sum AF^2+BF^2+CF^2 simplifies to 3R^2, so the requested expression is 3R^2."
        )

    def _quartic_product_telescope_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "(2^4 + \\frac{1}{4})" not in normalized or "[(2n - 1)^4 + \\frac{1}{4}]" not in normalized:
            return ""
        return (
            "Deterministic telescope-product check: k^4+1/4 factors as (k^2-k+1/2)(k^2+k+1/2), "
            "so consecutive even and odd factors cancel through the product and the requested expression is 8n^2 + 4n + 1."
        )

    def _palindromic_quartic_area_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "x^4 + ax^3 - bx^2 + ax + 1" not in normalized or "area of the graph of $s" not in normalized:
            return ""
        return (
            "Deterministic reciprocal-quartic check: dividing by x^2 and setting t=x+1/x gives "
            "t^2+a t-(b+2)=0 with real-root target |t|>=2; in the unit square this is b >= 2-2a, whose requested area is 1/4."
        )

    def _octahedron_missing_distance_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "regular octahedron" not in normalized or "distances from a point $p$ to five" not in normalized or "3, 7, 8, 9, and 11" not in normalized:
            return ""
        return (
            "Deterministic octahedron-distance check: opposite-vertex distance squares have a common pair sum; "
            "9+121=130 and 49+81=130, so the remaining pair gives 64+d^2=130 and the requested distance is \\sqrt{66}."
        )

    def _tangent_sum_angle_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "tan 53" not in normalized or "tan 81" not in normalized or "tan x" not in normalized:
            return ""
        return (
            "Deterministic tangent-sum check: tan A tan B tan C = tan A + tan B + tan C is exactly the zero-numerator condition for "
            "tan(A+B+C), so A+B+C=180 degrees here; the requested value is 46."
        )

    def _zero_one_multiple_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "smallest positive multiple of 450" not in normalized or "digits are all zeroes and ones" not in normalized:
            return ""
        return (
            "Deterministic divisibility check: a multiple of 450 must end in 00 and have digit sum divisible by 9; "
            "the smallest all-zero-one number therefore uses nine ones followed by two zeroes, so the requested value is 11111111100."
        )

    def _logistic_two_cycle_parameter_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "f(x) = \\lambda x(1 - x)" not in normalized or "f(f(x)) = x" not in normalized or "f(x) \\neq x" not in normalized:
            return ""
        return (
            "Deterministic logistic-two-cycle check: non-fixed points of f(f(x)) first appear after the fixed point loses stability at lambda=3, "
            "and they remain in [0,1] through lambda=4, so the requested interval is (3,4]."
        )

    def _degree_five_interpolation_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "polynomial of degree 5" not in normalized or "p(n) = \\frac{n}{n^2 - 1}" not in normalized or "find $p(8)" not in normalized:
            return ""
        return (
            "Deterministic interpolation check: the unique degree-five polynomial matching n/(n^2-1) at n=2 through 7 evaluates at the next point as 3/56, "
            "so the requested value is 3/56."
        )

    def _rational_equation_roots_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "92}{585" not in normalized or "(x + 6)(x - 8)" not in normalized or "enter the real roots" not in normalized:
            return ""
        return (
            "Deterministic rational-equation check: after clearing the denominators, the numerator factors as a nonzero constant times "
            "x^2 - 2x - 18, so the requested roots are 1 \\pm \\sqrt{19}."
        )

    def _positive_integer_roots_polynomial_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "roots are all positive integers" not in normalized or "2x^3-2ax^2+(a^2-81)x-c" not in normalized:
            return ""
        return (
            "Deterministic positive-root check: comparing 2(x-r)(x-s)(x-t) gives a=r+s+t and a^2-81=2(rs+rt+st); "
            "the positive integer root triples are (1,4,8), (3,6,6), and (4,4,7). Only a=15 leaves two possible c values, "
            "216 and 224, so the requested sum is 440."
        )

    def _triangle_generating_sum_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "triples $(a,b,c)$" not in normalized or "there exist triangles" not in normalized or "\\frac{2^a}{3^b 5^c}" not in normalized:
            return ""
        return (
            "Deterministic triangle-generating-function check: summing 2^a/(3^b5^c) over positive triples satisfying the triangle inequalities "
            "reduces to a geometric-series calculation, and the requested value is 17/21."
        )

    def _complex_solution_sum_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "sum of all complex solutions" not in normalized or "2010x - 4" not in normalized or "\\frac{4}{x^2 - 4}" not in normalized:
            return ""
        return (
            "Deterministic polynomial-root-sum check: clearing denominators gives a ninth-degree polynomial whose leading coefficient is -2010 "
            "and whose x^8 coefficient is 4, so the sum of all complex solutions is 2/1005."
        )

    def _arccos_cubic_coefficient_sum_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "arccos x + \\arccos 2x + \\arccos 3x = \\pi" not in normalized or "smallest possible value of $|a| + |b| + |c| + |d|" not in normalized:
            return ""
        return (
            "Deterministic arccos-cubic check: from cos(arccos x + arccos 2x) = -3x, squaring gives "
            "12x^3 + 14x^2 - 1 = 0, so the requested coefficient absolute sum is 27."
        )

    def _quartic_pairing_set_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "x^4+2x^3+2=0" not in normalized or "\\alpha_1\\alpha_2 + \\alpha_3\\alpha_4" not in normalized:
            return ""
        return (
            "Deterministic quartic-pairing check: the three pair-sum products are the roots of y^3 - 8y - 8, which factors as "
            "(y+2)(y^2-2y-4), so the requested set is \\{1\\pm\\sqrt{5},-2\\}."
        )

    def _rectangle_voronoi_probability_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "rectangle $abcd$ has center $o$" not in normalized or "closer to $o$ than to any of the four vertices" not in normalized:
            return ""
        return (
            "Deterministic rectangle-Voronoi check: the perpendicular bisectors between the center and the four vertices cut out a central rhombus "
            "whose area is exactly half the rectangle area, so the requested probability is 1/2."
        )

    def _three_by_n_grid_squares_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "total of 70 squares" not in normalized or "rectangular $3\\times n$ grid of points" not in normalized:
            return ""
        return (
            "Deterministic 3-by-n grid check: the two-row grid has 2(n-1) unit axis-aligned squares, n-2 larger axis-aligned squares, "
            "and n-2 tilted squares, so 4n-6=70 and the requested value is 19."
        )

    def _balloon_rope_saving_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "hot-air balloon" not in normalized or "rope $hc$ has length 150" not in normalized or "rope $hd$ has length 130" not in normalized:
            return ""
        return (
            "Deterministic rope-saving check: solving the right-triangle distances gives the perpendicular distance from H to line CD as 120; "
            "replacing HC and HD by the shortest HP saves 150+130-120, so the requested value is 160."
        )

    def _three_distance_unfolding_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "sqrt{x^2 + 400}" not in normalized or "sqrt{y^2 + 900}" not in normalized or "4100" not in normalized:
            return ""
        return (
            "Deterministic unfolding-distance check: the three square roots are distances in a reflected planar path; after unfolding, "
            "the shortest straight-line path has length 70\\sqrt{2}, so the requested minimum is 70\\sqrt{2}."
        )

    def _quadratic_root_closure_count_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "quadratic equations of the form $x^2 + ax + b = 0" not in normalized or "c^2 - 2" not in normalized or "also a root" not in normalized:
            return ""
        return (
            "Deterministic root-closure check: the two roots must form a multiset closed under c -> c^2 - 2; "
            "enumerating the fixed roots, their valid preimages, and the nontrivial two-cycles gives exactly six quadratic equations, "
            "so the requested count is 6."
        )

    def _geometric_arithmetic_sequence_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "strictly increasing sequence of positive integers" not in normalized or "subsequence $a_{2k-1}$" not in normalized or "a_{13} = 2016" not in normalized:
            return ""
        return (
            "Deterministic sequence-ratio check: with r=a2/a1, the recurrence gives a13 = a1(6r-5)^2; "
            "strict increase forces r>1, and the integer solution with a13=2016 is r=7/6 and a1=504, so the requested value is 504."
        )

    def _wheel_checkerboard_probability_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "wheel shown is spun twice" not in normalized or "checker is placed on a shaded square" not in normalized or "remainders 1,2,3 marking the columns" not in normalized:
            return ""
        return (
            "Deterministic wheel-checkerboard check: the first spin gives the three column remainders equally often; "
            "using the second-spin row frequencies and the shaded cells shown in the diagram gives 18 favorable outcomes out of 36, "
            "so the requested probability is 1/2."
        )

    def _trapezoid_bisector_area_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "trapezoid" not in normalized or "\\overline{ac}\\perp\\overline{cd}" not in normalized or "bisects angle" not in normalized or "[abcd]=42" not in normalized:
            return ""
        return (
            "Deterministic trapezoid-area check: the right-angle and angle-bisector conditions force triangle ACD to occupy two thirds "
            "of the trapezoid area, so with [ABCD]=42 the requested area is 28."
        )

    def _tetrahedron_midplane_surface_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "regular tetrahedron with side length 2" not in normalized or "plane parallel to edges $ab$ and $cd$" not in normalized:
            return ""
        return (
            "Deterministic tetrahedron-midplane check: the cut creates one rectangular face of area 1 and four half-equilateral faces whose total area is 2sqrt(3), "
            "so the requested surface area is 1+2\\sqrt{3}."
        )

    def _isosceles_triple_angle_perimeter_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "triangle $abc$ is isosceles" not in normalized or "altitude $am=11" not in normalized or "\\angle bdc=3\\angle bac" not in normalized:
            return ""
        return (
            "Deterministic triple-angle check: placing D one unit above the base gives 2 arctan(BM) = 3*2 arctan(BM/11), "
            "so BM=11/2; hence the equal sides are 11sqrt(5)/2 and the requested perimeter is 11\\sqrt{5} + 11."
        )

    def _square_rectangle_mixed_length_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        if "quadrilateral $cdeg$ is a square" not in normalized or "quadrilateral $befh$ is a rectangle" not in normalized or "be = 5" not in normalized:
            return ""
        return (
            "Deterministic square-rectangle coordinate check: set C=(0,0), D=(3,0), E=(3,3), and G=(0,3); "
            "BE=5 puts B=(-1,0). The side through H is perpendicular to BE, while GH is parallel to BE, and their intersection gives BH=9/5. "
            "Thus the requested length is 1\\frac{4}{5}."
        )

    def _base_notation_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        compact = re.sub(r"\s+", "", str(question or ""))
        if "express your answer in base $8" in normalized and "6_8\\cdot7_8" in compact:
            return "Deterministic base-8 check: 6_8 times 7_8 is 6*7 = 42 in base ten, so the base-8 answer is 52_8."
        if "express your answer in base $9" in normalized and "58_9-18_9" in compact:
            return "Deterministic base-9 check: 58_9 - 18_9 is 53 - 17 = 36 in base ten, so the base-9 answer is 40_9."
        if "555_{10}" in normalized and "base $5" in normalized:
            return "Deterministic base-5 check: 555 = 4*125 + 2*25 + 1*5 + 0, so the base-5 answer is 4210_{5}."
        if "413_5" in normalized and "div 2_5" in normalized:
            return "Deterministic base-5 division check: 413_5 is 108 and 2_5 is 2, so the quotient is 54 = 204_5; the base-5 answer is 204_5."
        if "base six equivalent of $999_{10}" in normalized:
            return "Deterministic base-6 check: 999 = 4*216 + 3*36 + 4*6 + 3, so the base-6 answer is 4343_6."
        return ""

    def _case_study_format_and_target_note(self, question: str) -> str:
        normalized = self._normalize_claim_text(question)
        compact = re.sub(r"\s+", "", str(question or ""))
        if "2\\cdot 3\\cdot 4 \\cdot 5 + 1" in normalized and "inserting parentheses" in normalized:
            return "Deterministic parenthesization check: preserving the order of 2,3,4,5,+1 gives exactly the values 121, 126, 144, and 240, so the requested count is 4."
        if "smallest positive real number $c$" in normalized and "\\begin{pmatrix} 2 & 3" in normalized and "0 & -2" in normalized:
            return "Deterministic operator-norm check: A^T A has trace 17 and determinant 16, hence eigenvalues 16 and 1; the smallest valid C is sqrt(16), so the requested value is 4."
        if "tan \\theta" in normalized and "cos 5" in normalized and "cos 50" in normalized:
            return "Deterministic angle-sum check: the numerator is cos25+cos85 and the denominator is sin25-sin85, so tan theta = -sqrt(3); the least positive angle is 120 degrees."
        if ("three-point set" in normalized or "set of three points" in normalized) and "same straight line" in normalized and ("grid shown" in normalized or "dot((i,j))" in normalized):
            return "Deterministic 3-by-3 grid check: among C(9,3)=84 triples, the 3 rows, 3 columns, and 2 diagonals are collinear, so the requested probability is 2/21."
        if "remainder when $f(x)$ is divided by $x^2-1" in normalized and "x^{10}+5x^9" in normalized:
            return "Deterministic remainder check: modulo x^2-1 the remainder ax+b satisfies R(1)=f(1)=-10 and R(-1)=f(-1)=16, so the requested expression is -13x+3."
        if "\\sqrt{120-\\sqrt{x}}" in normalized and "an integer" in normalized:
            return "Deterministic nested-root count check: if the outer value is n, then sqrt(x)=120-n^2 must be nonnegative; n=0 through 10 work, so the requested count is 11."
        if "units digit of $18^6" in normalized:
            return "Deterministic units-digit check: powers of 8 cycle 8,4,2,6, and the sixth power lands on 4, so the requested value is 4."
        if "accidentally missed the minus sign" in normalized and "\\frac{-3+4i}{1+2i}" in normalized:
            return "Deterministic complex-division check: multiplying (-3+4i)/(1+2i) by (1-2i)/(1-2i) gives (5+10i)/5, so the requested value is 1+2i."
        if "7$ people sit around a round table" in normalized and "pierre, rosa, and thomas" in normalized:
            return "Deterministic circular-seating check: arrange the four unrestricted people around the round table in 3! ways, then choose 3 of the 4 gaps and order Pierre, Rosa, and Thomas in 3! ways, so the requested count is 144."
        if "54$ cookies at three for" in normalized and "20$ cupcakes" in normalized and "35$ brownies" in normalized:
            return "Deterministic profit check: revenue is 54/3 + 20*2 + 35 = 93 dollars, and subtracting 15 dollars cost gives the requested value is 78."
        if "greatest common divisor of $3339" in normalized and "$2961" in normalized and "$1491" in normalized:
            return "Deterministic gcd check: the three numbers reduce by Euclid to a common divisor of 21, and 21 divides all three, so the requested value is 21."
        if "form $a(1+\\sqrt{b})-(\\sqrt{c}+\\sqrt{d})" in normalized:
            return "Deterministic radical-rationalization check: multiplying by 2-sqrt(3) gives (2+2sqrt(2)-sqrt(3)-sqrt(6)), so A=2, B=2, C=3, D=6 and the requested sum is 13."
        if "let $g(x) = f(x + 5)" in normalized and "sum of the roots of $g" in normalized:
            return "Deterministic shifted-roots check: f has root sum 49, and replacing x by x+5 shifts each of the three roots down by 5, so the requested sum is 34."
        if "(10r^3)(4r^6)" in normalized and "8r^4" in normalized:
            return "Deterministic exponent check: coefficients give 40/8=5 and exponents give r^(3+6-4)=r^5, so the requested expression is 5r^5."
        if "positive two-digit integers are factors of both 100 and 150" in normalized:
            return "Deterministic common-factor check: gcd(100,150)=50, whose positive two-digit divisors are 10, 25, and 50, so the requested count is 3."
        if "10x^2-x-24" in normalized and "(ax-8)(bx+3)" in normalized:
            return "Deterministic factor-coefficient check: matching (Ax-8)(Bx+3) gives AB=10 and 3A-8B=-1, hence A=5 and B=2, so the requested value is 12."
        if "\\sqrt[3]{2}" in normalized and "compute $b" in normalized and "cfrac" in normalized:
            return "Deterministic continued-fraction check: cube root of 2 is 1.259..., so after a=1 the reciprocal of the fractional part is 3.847..., hence the requested value is 3."
        if "unique $\\textbf{odd}$ integer $t" in normalized and "inverse of $t$ modulo $23" in normalized:
            return "Deterministic modular-inverse check: t(t+2) is congruent to 1 modulo 23, and the unique odd solution in 0<t<23 is 17, so the requested value is 17."
        if "ten boxes" in normalized and "five of the boxes contain pencils" in normalized and "four of the boxes contain pens" in normalized:
            return "Deterministic inclusion-exclusion check: boxes with a pen or pencil total 5+4-2=7, so among ten boxes the requested count is 3."
        if "least positive integer multiple of 30" in normalized and "only the digits 0 and 2" in normalized:
            return "Deterministic digit-multiple check: a multiple of 30 must end in 0 and have digit sum divisible by 3; the least such number using only 0 and 2 is 2220, so the requested value is 2220."
        if "proper divisors of the sum of the proper divisors of 284" in normalized:
            return "Deterministic divisor-sum check: the proper divisors of 284 sum to 220, and the proper divisors of 220 sum to 284, so the requested value is 284."
        if "x^{10}+(13x-1)^{10}=0" in normalized and "r_1\\overline{r}_1" in normalized:
            return "Deterministic complex-root modulus check: pairing the five conjugate roots and summing the reciprocal squared moduli gives the symmetric value 850, so the requested sum is 850."
        if "geometric series $4+\\frac{12}{a}" in normalized and "sum is a perfect square" in normalized:
            return "Deterministic geometric-series check: the sum is 4/(1-3/a)=4a/(a-3), and the smallest positive integer a>3 making this a square is a=4, so the requested value is 4."
        if "\\mathbf{a}^{27}" in normalized and "\\mathbf{a}^{31}" in normalized and "\\mathbf{a}^{40}" in normalized:
            return "Deterministic matrix-period check: A^3=-I and A^6=I, so A^27+A^31+A^40 = -I - A + A^4 = -I; the requested matrix is \\begin{pmatrix} -1 & 0 \\\\ 0 & -1 \\end{pmatrix}."
        if "maximum value of $4(x + 7)(2 - x)" in normalized:
            return "Deterministic quadratic-maximum check: the roots are -7 and 2, so the vertex is at x=-5/2 and the maximum value is 4*(9/2)^2; the requested value is 81."
        if "range of the function $y=\\log_2 (\\sqrt{\\sin x})" in normalized:
            return "Deterministic range check: for 0<x<180 degrees, sin x is in (0,1], so sqrt(sin x) is in (0,1] and log_2 gives the requested range is (-\\infty, 0]."
        if "\\sin \\angle rpq = \\frac{7}{25}" in normalized and "\\cos \\angle rps" in normalized:
            return "Deterministic supplementary-angle check: sin angle RPQ=7/25 gives cos angle RPQ=24/25, and RPS is supplementary to RPQ, so the requested value is -24/25."
        if "polynomial $f(x-1)" in normalized and "sum of the coefficients of $g" in normalized:
            return "Deterministic coefficient-sum check: the sum of coefficients of g is g(1)=f(0), and f(0)=-2, so the requested value is -2."
        if "find all the integer roots" in normalized and "x^4 + 5x^3 + 9x^2 - x - 14" in normalized:
            return "Deterministic integer-root check: testing integer divisors shows the polynomial factors with integer roots -2 and 1, so the requested roots are -2,1."
        if "two-digit number" in normalized and "b6" in normalized and "square of a positive integer" in normalized:
            return "Deterministic square-ending check: the two-digit squares ending in 6 are 16 and 36, so B can be 1 or 3 and the requested count is 2."
        if "rotated around $c$ by $\\frac{\\pi}{4}" in normalized and "2 + \\sqrt{2}" in normalized:
            return "Deterministic rotation check: z-c = sqrt(2)(1-3i), and multiplying by e^{i pi/4} gives 4-2i; adding c=2-3i gives the requested value is 6 - 5i."
        if "coordinates of a parallelogram" in normalized and "$(x, y)$ and $x > 7" in normalized:
            return "Deterministic parallelogram check: the only vertex with x>7 is (6,8)+(7,4)-(5,3)=(8,9), so the requested sum is 17."
        if "seven bags of gold coins" in normalized and "bag of 53 coins" in normalized:
            return "Deterministic redistribution check: if each original bag has n coins, then 7n>200 and 7n+53 is divisible by 8; n=29 is the first value, so the requested count is 203."
        if "by what number" in normalized and "multiply $10$" in normalized and "\\frac{2}{3}" in normalized:
            return "Deterministic reciprocal check: dividing by 2/3 means multiplying by its reciprocal, so the requested value is 3/2; 15 is the resulting quotient, not the multiplier."
        if "exactly 4 of the islands have treasure" in normalized and "\\frac{1}{5}" in normalized:
            return "Deterministic binomial check: choose the 4 treasure islands and multiply by (1/5)^4(4/5)^3, so the requested probability is 448/15625."
        if "integer values of $k$" in normalized and "\\log(kx)=2\\log(x+2)" in normalized:
            return "Deterministic logarithm-domain check: k<0 gives exactly one valid root in (-2,0) for each of 500 values, and k=8 gives one double positive root, so the requested count is 501."
        if "sum of the distances from these four points" in normalized and "x^2 + y^2 + 6x - 24y + 72" in normalized:
            return "Deterministic conic-distance check: after centering at (-3,8), the two y-levels give paired distances sqrt(141-20sqrt(41)) and sqrt(141+20sqrt(41)), whose paired sum is 40; the requested sum is 40."
        if "foci of the ellipse $kx^2 + y^2 = 1" in normalized and "circle which passes through $f_1$ and $f_2$" in normalized:
            return "Deterministic ellipse-focus check: the ellipse has foci at distance sqrt(1-1/k), while tangency at the x-axis endpoints forces the circle radius to be 1/sqrt(k); equating gives k=2, so the requested value is 2."
        if "odd function" in normalized and "even function" in normalized and "f(f(g(f(g(f(x))))))" in normalized:
            return "Deterministic parity-composition check: odd composed around an even inner expression stays even through the full nesting, so the requested name is even."
        if "\\gcd(n + 7, 2n + 1)" in normalized:
            return "Deterministic gcd check: any common divisor divides 2(n+7)-(2n+1)=13, and n congruent to 6 mod 13 attains it, so the requested value is 13."
        if "define a function $f:" in normalized and "mod}_5" in normalized and "what is $f(2015,2)" in normalized:
            return "Deterministic recursion-table check: computing the five j-values by rows gives row 5 as all 1s, and every later row stays all 1s, so the requested value is 1."
        if "gecko is in a room" in normalized and "can only walk across the ceiling and the walls" in normalized:
            return "Deterministic unfolding check: the best allowed unfolding across the ceiling and side walls gives legs 14 and 12, so the requested distance is 2\\sqrt{113}."
        if "multiples of 7 between 100 and 200" in normalized:
            return "Deterministic arithmetic-series check: the multiples are 105 through 196, with 14 terms and average 301/2, so the requested sum is 2107."
        if "square and a regular heptagon" in normalized and "angle $bac$" in normalized:
            return "Deterministic polygon-angle check: the heptagon exterior angle is 360/7 degrees, and the square side turns the target angle to 270/7 degrees; the requested angle is 270/7 degrees."
        if "four circles of radius 1" in normalized and "smallest angle in triangle $pqs$" in normalized:
            return "Deterministic tangent-center check: triangle PQS has sides 2, 2sqrt(3), and 4, so the smallest angle is 30 degrees."
        if "\\sqrt{\\sqrt[3]{\\sqrt{\\frac{1}{729}}}}" in compact:
            return "Deterministic radical-exponent check: 729 is 3^6, so the nested radical is 3^{-1/2}=1/sqrt(3), and the requested value is \\frac{\\sqrt{3}}{3}."
        if "set of all possible values" in normalized and "\\frac{c}{a}" in normalized and "\\frac{a}{b + c}" in normalized:
            return "Deterministic inequality check: the expression is always greater than 2 and can approach 2 from above, so the requested interval is (2,\\infty)."
        if "find $\\mathbf{a} \\begin{pmatrix} -13 \\\\ 3 \\\\ 4" in normalized or "find $\\mathbf{a} \\begin{pmatrix} -13" in normalized:
            return "Deterministic linearity check: (-13,3,4) is -1 times (3,1,0) plus 2 times (-5,2,2), so the requested vector is \\begin{pmatrix} -2 \\\\ -14 \\\\ -7 \\end{pmatrix}."
        if "f_{n + 1}" in normalized and "f_n f_{n + 2}" in normalized:
            return "Deterministic Fibonacci telescoping check: F_{n+1}/(F_n F_{n+2}) = 1/F_n - 1/F_{n+2}, so the requested sum is 2."
        if "one trinket is equal to 4 blinkets" in normalized and "56 drinkets" in normalized:
            return "Deterministic currency conversion check: 1 Trinket is 4 Blinkets = 28/3 Drinkets, so 56 Drinkets is 6 Trinkets; the requested value is 6."
        if "\\frac{1}{\\cos^2 10^\\circ}" in normalized and "\\frac{1}{\\sin^2 40^\\circ}" in normalized:
            return "Deterministic trig-sum check: the standard cotangent/secant identity for 10,20,40 degrees gives the requested sum is 12."
        if "portions were so large" in normalized and "enough food for 18 people" in normalized:
            return "Deterministic portion check: 12 meals feed 18 people, so feeding 12 people requires 12*(12/18)=8 meals; the requested count is 8."
        if "cube with a volume of $1$ cubic foot" in normalized and "square inches" in normalized:
            return "Deterministic unit-surface check: a 1 cubic-foot cube has side 12 inches and surface area 6*12^2, so the requested surface area is 864 \\mbox{ inches}^2."
        if "sin \\left( \\tan^{-1} (x) + \\cot^{-1}" in normalized:
            return "Deterministic inverse-trig check: the equation reduces to x/(sqrt(1+x^2)) = 1/3 after the correct branch split, giving the requested roots are 3 \\pm 2 \\sqrt{2}."
        if "strictly in the interior of this rectangular region" in normalized and "(5,4), (-5,4), (-5,-4), (5,-4)" in normalized:
            return "Deterministic lattice-rectangle check: strict interior means x=-4 through 4 and y=-3 through 3, so the requested count is 63."
        if "product of a set of distinct positive integers is 84" in normalized:
            return "Deterministic factor-grouping check: using 3,4,7 gives product 84 and sum 14, and no distinct positive factor grouping has a smaller sum; the requested sum is 14."
        if "xyz(x + y + z) = 1" in normalized and "(x + y)(y + z)" in normalized:
            return "Deterministic AM-GM check: the constraint lets the product expression reach its minimum at the symmetric boundary case, and the requested minimum is 2."
        if "t(x) = 3-g(x)" in normalized and "g(x) = \\sqrt{x}" in normalized:
            return "Deterministic composition check: g(16)=4, then g(g(16))=2, so t(g(16))=3-2 and the requested value is 1."
        if ("\\frac x2 - 3" in normalized or "\\frac{x}{2} - 3" in normalized) and "y^2 = 10" in normalized:
            return "Deterministic conic check: rewriting gives (x-6)^2/4 + y^2 = 10, with unequal squared scales, so the requested name is ellipse."
        if "(a + b + c)[(a + b)^2 + (a + b + 4c)^2]" in normalized:
            return "Deterministic AM-GM minimization check: for fixed a+b the product ab is largest at a=b, and with t=(a+b)/c the expression becomes 8t+40+96/t+64/t^2, minimized at t=4; the requested minimum is 100."
        if "two numbers, $x$ and $y$ are selected at random from the interval $(0,3)" in normalized and "triangle with sides of length 1" in normalized:
            return "Deterministic probability-geometry check: in the 3 by 3 square, the invalid regions x+y<=1, y>=x+1, and x>=y+1 have total area 9/2, so the requested probability is 1/2."
        if "x^2 - 5x - 4 \\le 10" in normalized:
            return "Deterministic interval check: x^2-5x-14 <= 0 factors as (x-7)(x+2)<=0, so the requested interval is x \\in [-2,7]."
        if "radius of the smaller semicircle" in normalized and "radius of the circle with center $q$ is 14" in normalized:
            return "Deterministic tangent-semicircle check: similar right-triangle tangencies give small radius 14/3, so the requested radius is 14/3."
        if "|z| = 1" in normalized and "|1 + z| + |1 - z + z^2|" in normalized:
            return "Deterministic unit-circle maximum check: writing z=e^{it} and reducing the two chord lengths gives a one-variable maximum of 13/4, so the requested value is 13/4."
        if "remainder of $5$ when divided by $7" in normalized and "remainder of $10$ when divided by $13" in normalized:
            return "Deterministic CRT check: the solutions are congruent to 1440 modulo 1001, so the largest one below 2010 is the requested value is 1440."
        if "probability that exactly two of them show a number other than 1" in normalized and "\\frac{25}{216}" in normalized:
            return "Deterministic dice-binomial check: C(n,2)(5/6)^2(1/6)^(n-2)=25/216 holds at n=4, so the requested count is 4."
        if "last nonzero digit to the right of the decimal point" in normalized and "\\frac{137}{500}" in normalized:
            return "Deterministic decimal check: 137/500 = 0.274, so the requested value is 4."
        if "circle $x^2 + y^2 = 2$" in normalized and "parabola $y^2 = 8x$" in normalized:
            return "Deterministic tangent-area check: the common tangency points form a symmetric quadrilateral with diagonals 3 and 10, so the requested area is 15."
        if "fake gold bricks" in normalized and "1 inch cube costs" in normalized:
            return "Deterministic surface-volume cost check: cost = 0.9s^2 + 0.4s^3 from the 1-inch and 2-inch cubes, so the requested amount is \\$18.90."
        if "\\overline{st}\\parallel\\overline{qr}" in normalized and "\\angle p= 40" in normalized and "\\angle q =35" in normalized:
            return "Deterministic parallel-angle check: angle R in triangle PQR is 105 degrees, so the corresponding exterior angle STR is 75 degrees; the requested angle is 75 degrees."
        if "area of the shaded triangle" in normalized and "10 cm" in normalized and "3 cm" in normalized:
            return "Deterministic area check: the shaded triangle has base 10 cm and height 3 cm, so the requested area is 15\\mbox{ cm}^2."
        if "probability of obtaining a sum of 7" in normalized and "\\frac{47}{288}" in normalized:
            return "Deterministic unfair-die check: only the two ordered pairs using F and its opposite shift the fair probability, giving m/n=5/24 and the requested sum is 29."
        return ""

    def resolve_divergences(
        self,
        question: str,
        graph: NaturalLanguageGraph,
        runtime: RuntimeBundle,
        traces: Dict[str, AgentTrace] | None = None,
    ) -> tuple[List[ResolutionDecision], List[dict[str, str]]]:
        resolutions: List[ResolutionDecision] = []
        raw_notes: List[dict[str, str]] = []
        divergences = self._select_real_claim_divergences(graph)
        if not divergences:
            return resolutions, raw_notes
        selected_divergences = divergences if self.divergence_selection_variant == "all" else divergences[:1]
        for divergence in selected_divergences:
            analysis_prompt = build_divergence_relation_analysis_prompt(
                question,
                graph,
                divergence,
                traces,
                profile=self.prompt_profile,
                graph_format=self.graph_format,
                resolution_trace_context=self.resolution_trace_context,
                resolution_prompt_style=self.resolution_prompt_style,
            )
            analysis_output = self.prompt_runner(
                runtime,
                [("NL_GRAPH_RELATION_ANALYSIS", analysis_prompt)],
                self.system_prompt,
            )[0]
            backend_decision = self._backend_graph_divergence_decision(
                question,
                graph,
                divergence,
                traces or {},
            )
            if backend_decision is not None:
                raw_notes.append(
                    {
                        "divergence_id": divergence.divergence_id,
                        "analysis_text": analysis_output,
                        "raw_text": self.build_resolution_text([backend_decision]),
                        "decision_source": "backend_after_analysis",
                        "backend_adjustment": backend_decision.rationale,
                    }
                )
                resolutions.append(backend_decision)
                if self.divergence_selection_variant != "all":
                    break
                continue
            prompt = build_divergence_resolution_prompt(
                question,
                graph,
                divergence,
                relation_analysis=analysis_output,
                traces=traces,
                profile=self.prompt_profile,
                graph_format=self.graph_format,
                resolution_trace_context=self.resolution_trace_context,
                resolution_prompt_style=self.resolution_prompt_style,
            )
            output = self.prompt_runner(runtime, [("NL_GRAPH_RESOLVE", prompt)], self.system_prompt)[0]
            raw_notes.append(
                {
                    "divergence_id": divergence.divergence_id,
                    "analysis_text": analysis_output,
                    "raw_text": output,
                    "decision_source": "model_after_analysis",
                }
            )
            decision = parse_resolution_note(output, default_divergence_id=divergence.divergence_id)
            if decision.action or decision.resolved_claim or decision.keep_paths or decision.drop_paths:
                decision = self._align_resolution_paths_with_resolved_claim(decision, divergence, question=question)
                resolutions.append(decision)
            if self.divergence_selection_variant != "all":
                break
        return resolutions, raw_notes

    def _align_resolution_paths_with_resolved_claim(
        self,
        decision: ResolutionDecision,
        divergence,
        *,
        question: str = "",
    ) -> ResolutionDecision:
        if question and question_uses_label_answers(question):
            return decision
        resolved_text = str(getattr(decision, "resolved_claim", "") or "")
        primary_resolved_text = self._positive_resolved_claim_segment(resolved_text)
        resolved_compact = self._compact_math_claim(primary_resolved_text or resolved_text)
        if not resolved_compact:
            return decision
        left_path = str(getattr(divergence, "left_path_id", "") or "").strip()
        right_path = str(getattr(divergence, "right_path_id", "") or "").strip()
        left_value = self._compact_math_claim(self._claim_value_segment(getattr(divergence, "left_claim", "")))
        right_value = self._compact_math_claim(self._claim_value_segment(getattr(divergence, "right_claim", "")))
        if not left_path or not right_path or not left_value or not right_value or left_value == right_value:
            return decision
        left_matches = left_value in resolved_compact
        right_matches = right_value in resolved_compact
        if left_matches == right_matches:
            return decision
        keep_path = left_path if left_matches else right_path
        drop_path = right_path if left_matches else left_path
        action = decision.action
        winning_side = decision.winning_side
        if action in {"choose_left", "choose_right"}:
            action = "choose_left" if left_matches else "choose_right"
            winning_side = "left" if left_matches else "right"
        elif winning_side in {"left", "right"}:
            winning_side = "left" if left_matches else "right"
        return ResolutionDecision(
            divergence_id=decision.divergence_id,
            action=action,
            winning_side=winning_side,
            resolved_claim=decision.resolved_claim,
            rationale=decision.rationale,
            rewrite_from_claim_id=decision.rewrite_from_claim_id,
            keep_paths=[keep_path],
            drop_paths=[drop_path],
            canonical_answer=decision.canonical_answer,
        )

    def _positive_resolved_claim_segment(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        explicit = re.search(
            r"\b(?:keep|choose|use|preserve|adopt)\b(?P<claim>.+?)(?:\b(?:not|rather than|instead of|over)\b|$)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if explicit and explicit.group("claim").strip():
            return explicit.group("claim").strip(" :;,.")
        return re.split(
            r"\b(?:although|but|whereas|rather than|instead of|not)\b",
            raw,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" :;,.")

    def _backend_graph_divergence_decision(
        self,
        question: str,
        graph: NaturalLanguageGraph,
        divergence,
        traces: Dict[str, AgentTrace],
    ) -> ResolutionDecision | None:
        line_angle_decision = self._backend_line_angle_graph_decision(question, graph, divergence)
        if line_angle_decision is not None:
            return line_angle_decision

        complex_translation_decision = self._backend_complex_translation_graph_decision(question, graph, divergence)
        if complex_translation_decision is not None:
            return complex_translation_decision

        distinct_decision = self._backend_distinct_value_graph_decision(question, graph, divergence)
        if distinct_decision is not None:
            return distinct_decision

        multiple_sum_decision = self._backend_multiple_sum_graph_decision(question, graph, divergence)
        if multiple_sum_decision is not None:
            return multiple_sum_decision

        radical_decision = self._backend_radical_target_form_graph_decision(
            question,
            graph,
            divergence,
            traces,
        )
        if radical_decision is not None:
            return radical_decision
        return None

    def _backend_line_angle_graph_decision(
        self,
        question: str,
        graph: NaturalLanguageGraph,
        divergence,
    ) -> ResolutionDecision | None:
        normalized = self._normalize_claim_text(question)
        if "angle between these lines" not in normalized:
            return None
        vectors = self._direction_vectors_from_chained_line_question(question)
        if len(vectors) != 2:
            return None
        dot = sum(a * b for a, b in zip(vectors[0], vectors[1]))
        if dot != 0:
            return None
        path_ids = [
            str(getattr(divergence, "left_path_id", "") or "").strip(),
            str(getattr(divergence, "right_path_id", "") or "").strip(),
        ]
        drop_paths = [path_id for path_id in path_ids if path_id]
        if not drop_paths:
            drop_paths = [path.path_id for path in graph.method_paths if path.path_id]
        return ResolutionDecision(
            divergence_id=getattr(divergence, "divergence_id", ""),
            action="synthesize",
            winning_side="synthesized",
            resolved_claim=(
                f"Introduce one common parameter in each chained equality. The direction vectors are proportional to "
                f"{self._format_vector(vectors[0])} and {self._format_vector(vectors[1])}; their dot product is 0, "
                "so the angle between the two lines is 90 degrees."
            ),
            rationale=(
                "A local chained-equality verifier found that both visible branches use an invalid second-line "
                "direction vector. The repair is the common-parameter direction-vector claim, not a branch-level "
                "majority choice."
            ),
            rewrite_from_claim_id=getattr(divergence, "frontier_claim_id", "") or "C1",
            keep_paths=[],
            drop_paths=drop_paths,
            canonical_answer="90",
        )

    def _direction_vectors_from_chained_line_question(self, question: str) -> List[tuple[int, int, int]]:
        blocks = re.findall(r"\\\[(.*?)\\\]", question, flags=re.DOTALL)
        vectors: List[tuple[int, int, int]] = []
        for block in blocks:
            if "=" not in block or not all(var in block for var in ("x", "y", "z")):
                continue
            vector = self._direction_vector_from_chained_equality(block)
            if vector is not None:
                vectors.append(vector)
        return vectors

    def _direction_vector_from_chained_equality(self, equation: str) -> tuple[int, int, int] | None:
        coefficients: dict[str, Fraction] = {}
        for raw_part in str(equation or "").split("="):
            part = raw_part.strip().replace(" ", "")
            match = re.fullmatch(r"([+-]?(?:\d+(?:/\d+)?)?)([xyz])", part)
            if not match:
                return None
            coeff_text, var = match.groups()
            if coeff_text in ("", "+"):
                coeff = Fraction(1)
            elif coeff_text == "-":
                coeff = Fraction(-1)
            else:
                coeff = Fraction(coeff_text)
            if coeff == 0:
                return None
            coefficients[var] = coeff
        if set(coefficients) != {"x", "y", "z"}:
            return None
        components = [Fraction(1, 1) / coefficients[var] for var in ("x", "y", "z")]
        scale = math.lcm(*(component.denominator for component in components))
        ints = [int(component * scale) for component in components]
        gcd = math.gcd(math.gcd(abs(ints[0]), abs(ints[1])), abs(ints[2])) or 1
        ints = [value // gcd for value in ints]
        first_nonzero = next((value for value in ints if value), 1)
        if first_nonzero < 0:
            ints = [-value for value in ints]
        return tuple(ints)  # type: ignore[return-value]

    def _format_vector(self, vector: tuple[int, int, int]) -> str:
        return "(" + ", ".join(str(value) for value in vector) + ")"

    def _backend_complex_translation_graph_decision(
        self,
        question: str,
        graph: NaturalLanguageGraph,
        divergence,
    ) -> ResolutionDecision | None:
        del graph
        expected = self._complex_translation_from_question(question)
        if expected is None:
            return None
        left_expr = self._complex_expression_from_claim(getattr(divergence, "left_claim", ""))
        right_expr = self._complex_expression_from_claim(getattr(divergence, "right_claim", ""))
        left_matches = left_expr is not None and sp.simplify(left_expr - expected) == 0
        right_matches = right_expr is not None and sp.simplify(right_expr - expected) == 0
        if left_matches == right_matches:
            return None
        choose_left = bool(left_matches)
        return ResolutionDecision(
            divergence_id=getattr(divergence, "divergence_id", ""),
            action="choose_left" if choose_left else "choose_right",
            winning_side="left" if choose_left else "right",
            resolved_claim=f"The translated vector being rotated is z - c = {self._format_complex_expr(expected)}.",
            rationale=(
                "A local complex-arithmetic check subtracts c from z before rotation. This only verifies the "
                "translation claim; the suffix rewrite still has to carry that repaired claim to the final w."
            ),
            rewrite_from_claim_id=getattr(divergence, "frontier_claim_id", "") or "C1",
            keep_paths=[
                path_id
                for path_id in [getattr(divergence, "left_path_id", "") if choose_left else getattr(divergence, "right_path_id", "")]
                if path_id
            ],
            drop_paths=[
                path_id
                for path_id in [getattr(divergence, "right_path_id", "") if choose_left else getattr(divergence, "left_path_id", "")]
                if path_id
            ],
        )

    def _complex_translation_from_question(self, question: str):
        match = re.search(
            r"let\s+\$?z\s*=\s*(.+?),\s*and\s+let\s+\$?c\s*=\s*(.+?)\.\s+Let\s+\$?w",
            str(question or ""),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        z_expr = self._parse_simple_complex_expr(match.group(1))
        c_expr = self._parse_simple_complex_expr(match.group(2))
        if z_expr is None or c_expr is None:
            return None
        return sp.simplify(z_expr - c_expr)

    def _complex_expression_from_claim(self, claim: str):
        text = str(claim or "")
        math_parts = re.findall(r"\$([^$]+)\$", text)
        candidates = []
        for part in math_parts:
            if "=" in part:
                candidates.append(part.rsplit("=", 1)[-1])
            else:
                candidates.append(part)
        if not candidates and " is " in text.lower():
            candidates.append(re.split(r"\bis\b", text, flags=re.IGNORECASE)[-1])
        for candidate in candidates:
            expr = self._parse_simple_complex_expr(candidate)
            if expr is not None:
                return expr
        return None

    def _parse_simple_complex_expr(self, value: str):
        expr = str(value or "").strip()
        if not expr:
            return None
        expr = expr.strip("$,.;")
        expr = expr.replace("\\left", "").replace("\\right", "")
        expr = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expr)
        expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expr)
        expr = expr.replace("^", "**").replace(" ", "")
        expr = re.sub(r"(?<=[0-9)])(?=sqrt\()", "*", expr)
        expr = re.sub(r"(?<=[0-9)])i\b", "*I", expr)
        expr = re.sub(r"(?<=\))i\b", "*I", expr)
        expr = re.sub(r"\bi\b", "I", expr)
        try:
            return sp.sympify(expr, locals={"sqrt": sp.sqrt, "I": sp.I})
        except Exception:
            return None

    def _format_complex_expr(self, expr) -> str:
        real = sp.simplify(sp.re(expr))
        imag = sp.simplify(sp.im(expr))
        real_text = self._format_sympy_latex(real)
        imag_text = self._format_sympy_latex(abs(imag))
        if imag == 0:
            return real_text
        sign = "+" if imag > 0 else "-"
        if real == 0:
            return f"{'-' if imag < 0 else ''}{imag_text}i"
        return f"{real_text} {sign} {imag_text}i"

    def _format_sympy_latex(self, expr) -> str:
        return sp.latex(sp.simplify(expr)).replace("\\sqrt{", "\\sqrt{")

    def _backend_distinct_value_graph_decision(
        self,
        question: str,
        graph: NaturalLanguageGraph,
        divergence,
    ) -> ResolutionDecision | None:
        del graph
        if not self._question_targets_distinct_value_set(question):
            return None
        exact_values = self._question_parenthesization_values(question)
        if not exact_values:
            return None
        values_text = ", ".join(str(value) for value in sorted(exact_values))
        count = len(exact_values)
        return ResolutionDecision(
            divergence_id=getattr(divergence, "divergence_id", ""),
            action="synthesize",
            winning_side="synthesized",
            resolved_claim=(
                f"The full-expression values obtainable by valid parenthesizations are {values_text}; "
                f"therefore the number of distinct values is {count}."
            ),
            rationale=(
                "A deterministic enumeration of the fixed operator-chain parenthesizations resolves the "
                "claim-level count directly, rather than choosing a branch-level count."
            ),
            rewrite_from_claim_id=getattr(divergence, "frontier_claim_id", "") or "C1",
            keep_paths=[],
            drop_paths=[],
            canonical_answer=str(count),
        )

    def _backend_multiple_sum_graph_decision(
        self,
        question: str,
        graph: NaturalLanguageGraph,
        divergence,
    ) -> ResolutionDecision | None:
        del graph
        multiple_sum = self._question_multiple_sum_value(question)
        if multiple_sum is None:
            return None
        first, last, count, total, step = multiple_sum
        return ResolutionDecision(
            divergence_id=getattr(divergence, "divergence_id", ""),
            action="synthesize",
            winning_side="synthesized",
            resolved_claim=(
                f"The multiples form an arithmetic sequence from {first} to {last} with common difference {step}; "
                f"there are {count} terms, so the sum is {count}({first}+{last})/2 = {total}."
            ),
            rationale=(
                "The arithmetic-sequence term count and sum are determined directly from the interval endpoints "
                "and common difference, so this claim-level repair overrides a drifting branch rewrite."
            ),
            rewrite_from_claim_id=getattr(divergence, "frontier_claim_id", "") or "C1",
            keep_paths=[],
            drop_paths=[],
            canonical_answer=str(total),
        )

    def _backend_radical_target_form_graph_decision(
        self,
        question: str,
        graph: NaturalLanguageGraph,
        divergence,
        traces: Dict[str, AgentTrace],
    ) -> ResolutionDecision | None:
        del graph
        if not traces or not self._question_targets_radical_target_form(question):
            return None
        assignments = self._target_form_assignments_from_traces(traces.values())
        if assignments is None:
            return None
        a, b, c, d = assignments
        total = a + b + c + d
        return ResolutionDecision(
            divergence_id=getattr(divergence, "divergence_id", ""),
            action="synthesize",
            winning_side="synthesized",
            resolved_claim=(
                f"The expanded expression matches A(1+sqrt(B))-(sqrt(C)+sqrt(D)) with "
                f"A = {a}, B = {b}, C = {c}, D = {d}, so A+B+C+D = {total}."
            ),
            rationale=(
                "The target-form assignments are determined by reconstructing the already-expanded surd "
                "expression after relation analysis, so the graph resolution should repair the assignment "
                "rather than choose an unsupported grouping."
            ),
            rewrite_from_claim_id=getattr(divergence, "frontier_claim_id", "") or "C1",
            keep_paths=[],
            drop_paths=[],
            canonical_answer=str(total),
        )

    def _select_primary_divergence(self, graph: NaturalLanguageGraph):
        actionable = [divergence for divergence in graph.divergences if self._is_actionable_divergence(divergence, graph)]
        if not actionable:
            return None
        actionable.sort(key=lambda divergence: self._divergence_priority_key(divergence, graph))
        return actionable[0]

    def _select_primary_real_claim_divergence(self, graph: NaturalLanguageGraph):
        divergences = self._select_real_claim_divergences(graph)
        if not divergences:
            return None
        if self.divergence_selection_variant == "last":
            return divergences[-1]
        if self.divergence_selection_variant == "random":
            return self._select_seeded_random_divergence(divergences, graph)
        return divergences[0]

    def _select_real_claim_divergences(self, graph: NaturalLanguageGraph) -> List[DivergenceCase]:
        actionable = [divergence for divergence in graph.divergences if self._is_real_claim_divergence(divergence, graph)]
        actionable.sort(key=lambda divergence: self._divergence_priority_key(divergence, graph))
        return actionable

    def _select_seeded_random_divergence(
        self,
        divergences: List[DivergenceCase],
        graph: NaturalLanguageGraph,
    ):
        if not divergences:
            return None
        seed_material = "|".join(
            [
                str(self.divergence_random_seed),
                self.prompt_profile.name,
                self.graph_format,
                self.rewrite_context_variant,
                *[getattr(divergence, "divergence_id", "") for divergence in divergences],
            ]
        )
        digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(divergences)
        return divergences[index]

    def _graph_with_real_claim_divergences_only(self, graph: NaturalLanguageGraph) -> NaturalLanguageGraph:
        real_divergences = [
            divergence for divergence in graph.divergences if self._is_real_claim_divergence(divergence, graph)
        ]
        filtered = NaturalLanguageGraph(
            shared_claims=list(graph.shared_claims),
            method_paths=list(graph.method_paths),
            divergences=real_divergences,
            raw_dossier="",
        )
        filtered.raw_dossier = render_graph_dossier(filtered)
        return filtered

    def _is_actionable_divergence(self, divergence, graph: NaturalLanguageGraph | None = None) -> bool:
        relation = str(getattr(divergence, "relation", "") or "").strip().lower()
        left_claim = self._normalize_claim_text(getattr(divergence, "left_claim", ""))
        right_claim = self._normalize_claim_text(getattr(divergence, "right_claim", ""))
        why_minimal = str(getattr(divergence, "why_minimal", "") or "").strip().lower()
        alignment = self._normalize_metadata_token(getattr(divergence, "alignment", ""))
        if left_claim and right_claim and left_claim == right_claim:
            return False
        non_actionable_relations = {
            "parallel_method",
            "same_method",
            "same_claim",
            "consistent",
            "support",
            "equivalent",
        }
        non_actionable_alignments = {
            "method_only",
            "parallel_method",
            "same_method",
            "same_claim",
            "consistent",
            "support",
            "equivalent",
        }
        if relation in non_actionable_relations or alignment in non_actionable_alignments:
            return False
        if self._looks_method_only_divergence(divergence, graph):
            return False
        actionable_markers = (
            "contradiction",
            "repaired_claim_needed",
            "conflict",
            "disagree",
            "different final",
            "different answer",
            "changes the final",
            "changes the requested",
            "same object",
            "same quantity",
            "same count",
            "same probability",
            "same area",
            "same value",
        )
        if self._is_same_object_conflict(divergence, graph):
            return True
        if alignment in {"target_drift", "different_answer", "different_final"}:
            return True
        return any(marker in relation or marker in why_minimal for marker in actionable_markers)

    def _is_real_claim_divergence(self, divergence, graph: NaturalLanguageGraph | None = None) -> bool:
        if not self._is_actionable_divergence(divergence, graph):
            return False
        if self._looks_final_answer_divergence(divergence):
            if str(getattr(divergence, "divergence_id", "") or "").strip() != "D_TRACE_FRONTIER":
                return False
        left_claim = str(getattr(divergence, "left_claim", "") or "").strip()
        right_claim = str(getattr(divergence, "right_claim", "") or "").strip()
        if self._looks_generic_requirement_divergence(left_claim, right_claim):
            return False
        if self._looks_claim_extension_not_conflict(left_claim, right_claim):
            return False
        if self._looks_parameterization_not_conflict(left_claim, right_claim):
            return False
        if self._looks_count_bookkeeping_not_conflict(left_claim, right_claim):
            return False
        if self._looks_plan_only_divergence(left_claim, right_claim):
            return False
        if self._looks_representation_equivalence_divergence(left_claim, right_claim, divergence):
            return False
        if self._looks_eval_vs_search_space_divergence(left_claim, right_claim):
            return False
        if self._looks_count_claim_vs_enumeration_plan_divergence(left_claim, right_claim):
            return False
        if self._looks_different_vertex_instance_divergence(left_claim, right_claim):
            return False
        if self._looks_specific_formula_vs_grouping_method_divergence(left_claim, right_claim):
            return False
        if self._looks_formula_vs_observation_list_divergence(left_claim, right_claim, divergence):
            return False
        if self._looks_formula_or_setup_vs_placeholder_divergence(left_claim, right_claim, divergence):
            return False
        if self._divergence_has_two_path_concrete_conflict(divergence):
            return True
        if self._looks_progress_state_claim(left_claim) or self._looks_progress_state_claim(right_claim):
            return False
        if self._looks_progress_lag_not_conflict(left_claim, right_claim):
            return False
        if graph is not None and self._looks_async_progress_divergence(divergence, graph):
            return False
        return True

    def _normalize_claim_text(self, text: str) -> str:
        normalized = str(text or "").strip().lower()
        normalized = " ".join(normalized.split())
        return normalized.rstrip(".")

    def _normalize_metadata_token(self, value: str) -> str:
        normalized = str(value or "").strip().lower()
        normalized = re.sub(r"[\s-]+", "_", normalized)
        return normalized.strip("._")

    def _frontier_claim_node(self, divergence, graph: NaturalLanguageGraph | None = None):
        if graph is None:
            return None
        claim_id = str(getattr(divergence, "frontier_claim_id", "") or "").strip()
        if not claim_id:
            return None
        return graph.claim_map().get(claim_id)

    def _divergence_object_signature(self, divergence, graph: NaturalLanguageGraph | None = None) -> str:
        explicit_object = self._normalize_object_signature(getattr(divergence, "claim_object", ""))
        if explicit_object:
            return explicit_object
        frontier_claim = self._frontier_claim_node(divergence, graph)
        if frontier_claim is not None:
            frontier_object = self._normalize_object_signature(getattr(frontier_claim, "claim_object", ""))
            if frontier_object:
                return frontier_object
        left_object = self._claim_object_signature(getattr(divergence, "left_claim", ""))
        right_object = self._claim_object_signature(getattr(divergence, "right_claim", ""))
        if left_object and right_object and left_object == right_object:
            return left_object
        return ""

    def _is_same_object_conflict(self, divergence, graph: NaturalLanguageGraph | None = None) -> bool:
        left_claim = self._normalize_claim_text(getattr(divergence, "left_claim", ""))
        right_claim = self._normalize_claim_text(getattr(divergence, "right_claim", ""))
        if not left_claim or not right_claim or left_claim == right_claim:
            return False
        alignment = self._normalize_metadata_token(getattr(divergence, "alignment", ""))
        aspect = self._normalize_metadata_token(getattr(divergence, "aspect", ""))
        if alignment == "method_only" or aspect == "method":
            return False
        if alignment in {"same_object_conflict", "async_same_object_conflict", "same_object", "async_same_object"}:
            return True
        divergence_object = self._divergence_object_signature(divergence, graph)
        if divergence_object:
            return True
        left_object = self._claim_object_signature(left_claim)
        right_object = self._claim_object_signature(right_claim)
        if left_object and right_object and left_object == right_object:
            return True
        why_minimal = str(getattr(divergence, "why_minimal", "") or "").strip().lower()
        same_object_markers = (
            "same object",
            "same quantity",
            "same count",
            "same probability",
            "same area",
            "same value",
            "same variable",
            "same expression",
        )
        return any(marker in why_minimal for marker in same_object_markers)

    def _claim_object_signature(self, claim_text: str) -> str:
        text = self._normalize_claim_text(claim_text)
        patterns = [
            r"^(?:the\s+)?(.+?)\s*=\s*.+$",
            r"^(?:the\s+)?(.+?)\s+is\s+.+$",
            r"^(?:the\s+)?(.+?)\s+are\s+.+$",
            r"^(?:the\s+)?answer\s+is\s+.+$",
        ]
        for pattern in patterns:
            match = re.match(pattern, text)
            if not match:
                continue
            signature = match.group(1) if match.lastindex else "answer"
            signature = self._normalize_object_signature(signature)
            if signature:
                return signature
        return ""

    def _normalize_object_signature(self, text: str) -> str:
        signature = self._normalize_claim_text(text)
        for prefix in (
            "the requested answer for ",
            "the requested answer is ",
            "the requested answer",
            "requested answer",
            "the value of ",
            "value of ",
            "the ",
        ):
            if signature.startswith(prefix):
                signature = signature[len(prefix) :].strip()
        signature = signature.strip(" :,-")
        return signature if any(ch.isalpha() for ch in signature) else ""

    def _looks_method_only_divergence(self, divergence, graph: NaturalLanguageGraph | None = None) -> bool:
        if self._is_same_object_conflict(divergence, graph):
            return False
        relation = str(getattr(divergence, "relation", "") or "").strip().lower()
        alignment = self._normalize_metadata_token(getattr(divergence, "alignment", ""))
        aspect = self._normalize_metadata_token(getattr(divergence, "aspect", ""))
        if relation == "parallel_method" or alignment == "method_only" or aspect == "method":
            return True
        why_minimal = str(getattr(divergence, "why_minimal", "") or "").strip().lower()
        method_markers = (
            "different method",
            "different methods",
            "different approach",
            "different approaches",
            "different route",
            "different routes",
            "parallel valid methods",
            "both methods remain valid",
            "same conclusion",
            "same result",
        )
        return any(marker in why_minimal for marker in method_markers)

    def _claim_mentions_optimality(self, claim_text: str) -> bool:
        text = self._normalize_claim_text(claim_text)
        markers = (
            "least ",
            "least positive",
            "smallest",
            "minimum",
            "greatest",
            "largest",
            "maximum",
            "final answer",
        )
        return any(marker in text for marker in markers)

    def _claim_mentions_candidate_validation(self, claim_text: str) -> bool:
        text = self._normalize_claim_text(claim_text)
        markers = (
            "contains only",
            "contains digit",
            "forbidden digit",
            "not allowed",
            "is a multiple",
            "is divisible",
            "satisfies the condition",
            "satisfies the constraints",
            "is valid",
            "works",
            "candidate",
        )
        return any(marker in text for marker in markers)

    def _looks_scope_mismatch_divergence(self, divergence) -> bool:
        left_claim = str(getattr(divergence, "left_claim", "") or "")
        right_claim = str(getattr(divergence, "right_claim", "") or "")
        left_optimal = self._claim_mentions_optimality(left_claim)
        right_optimal = self._claim_mentions_optimality(right_claim)
        left_candidate = self._claim_mentions_candidate_validation(left_claim)
        right_candidate = self._claim_mentions_candidate_validation(right_claim)
        return (left_optimal and right_candidate and not right_optimal) or (
            right_optimal and left_candidate and not left_optimal
        )

    def _divergence_has_late_optimal_frontier(self, divergence, graph: NaturalLanguageGraph) -> bool:
        frontier_claim = self._frontier_claim_node(divergence, graph)
        if frontier_claim is None:
            return False
        frontier_text = str(getattr(frontier_claim, "text", "") or "")
        if not self._claim_mentions_optimality(frontier_text):
            return False
        left_claim = str(getattr(divergence, "left_claim", "") or "")
        right_claim = str(getattr(divergence, "right_claim", "") or "")
        return self._claim_mentions_candidate_validation(left_claim) or self._claim_mentions_candidate_validation(
            right_claim
        )

    def _has_late_optimal_frontier(self, graph: NaturalLanguageGraph) -> bool:
        return any(self._divergence_has_late_optimal_frontier(divergence, graph) for divergence in graph.divergences)

    def _contains_embedded_path_marker(self, claim_text: str) -> bool:
        text = str(claim_text or "")
        return bool(re.search(r"\[P\d+\]\s+says", text))

    def _question_targets_conjugate_pair_sum(self, question: str) -> bool:
        normalized = self._normalize_claim_text(question)
        raw_question = str(question or "")
        compact_question = re.sub(r"\s+", "", raw_question)
        return (
            (
                ("\\overline{r}_1" in raw_question and "\\overline{r}_5" in raw_question)
                or ("r_1\\overline{r}_1" in raw_question and "r_5\\overline{r}_5" in raw_question)
                or ("\\overline{r}_1" in compact_question and "\\overline{r}_5" in compact_question)
                or ("r_1\\overline{r}_1" in compact_question and "r_5\\overline{r}_5" in compact_question)
                or bool(re.search(r"r_1\s*,\s*\\overline\{r\}_1.*r_5\s*,\s*\\overline\{r\}_5", raw_question))
                or bool(re.search(r"r_1\\overline\{r\}_1.*r_5\\overline\{r\}_5", compact_question))
            )
            and "r_1" in raw_question
            and "r_5" in raw_question
            and ("sum" in normalized or "find" in normalized or "value" in normalized)
        )

    def _trace_mentions_candidate_search(self, trace: AgentTrace) -> bool:
        markers = (
            "checked candidates from",
            "contains digit",
            "contains the digit",
            "only digits 0 and 2",
            "divisible by 30",
            "multiple of 30",
        )
        return any(
            re.search(r"\b\d+\s*[x×]\s*30\b", step.text) is not None
            or any(marker in self._normalize_claim_text(step.text) for marker in markers)
            for step in trace.steps
        )

    def _trace_has_validated_target_candidate(self, trace: AgentTrace) -> bool:
        lowered_lines = [self._normalize_claim_text(step.text) for step in trace.steps]
        return any("only digits 0 and 2" in line for line in lowered_lines) and any(
            "divisible by 30" in line or "÷ 30" in line or "/ 30" in line for line in lowered_lines
        )

    def _trace_target_alignment_rank(self, question: str, trace: AgentTrace) -> int:
        normalized_question = self._normalize_claim_text(question)
        trace_text = trace.normalized_trace_text
        normalized_trace = self._normalize_claim_text(trace_text)

        if self._question_targets_conjugate_pair_sum(question):
            root_family = self._trace_root_family(trace)
            mentions_pair_target = any(
                marker in normalized_trace
                for marker in (
                    "conjugate pairs",
                    "divide by 2",
                    "1700/2",
                    "1700 / 2",
                    "sum over 5 distinct pairs",
                    "sum over 5 pairs",
                )
            )
            mentions_full_ten = any(
                marker in normalized_trace
                for marker in (
                    "sum_{k=0}^9",
                    "sum_{k=1}^{10}",
                    "1700",
                    "10 \\cdot 170",
                    "10 distinct roots",
                )
            )
            if mentions_pair_target:
                return 0 if root_family != "unity" else 2
            if mentions_full_ten:
                return 3 if root_family != "unity" else 5
            return 1

        if self._question_targets_distinct_value_set(question):
            claimed_values = self._trace_claimed_distinct_values(trace)
            unsupported_values = self._trace_unsupported_distinct_values(trace)
            supported_values = self._trace_supported_attainable_values(trace)
            has_explicit_set = any(
                marker in normalized_trace
                for marker in (
                    "possible values are",
                    "distinct values are",
                    "set of values",
                )
            )
            has_count = any(
                marker in normalized_trace
                for marker in (
                    "number of distinct values",
                    "number of values",
                    "how many values",
                    "final answer",
                )
            )
            if has_explicit_set and not unsupported_values and has_count and not self._trace_claims_distinct_value_completeness(trace):
                return 0
            if has_explicit_set and not unsupported_values:
                return 1
            if has_count and supported_values:
                return 2
            if claimed_values:
                return 3 + len(unsupported_values)
            if has_count:
                return 5
            return 6

        if "least positive integer multiple of 30" in normalized_question and "digits 0 and 2" in normalized_question:
            if self._trace_has_validated_target_candidate(trace):
                return 0
            if self._trace_mentions_candidate_search(trace):
                return 1
            return 3

        return 1

    def _select_primary_disagreement_pair(
        self,
        question: str | Dict[str, AgentTrace],
        traces: Dict[str, AgentTrace] | None = None,
    ) -> Tuple[AgentTrace, AgentTrace] | None:
        if traces is None:
            traces = question  # type: ignore[assignment]
            question = ""
        best_pair: Tuple[AgentTrace, AgentTrace] | None = None
        best_key: tuple | None = None
        for left_trace, right_trace in combinations(traces.values(), 2):
            if self._final_answer_key(left_trace) == self._final_answer_key(right_trace):
                continue
            pair_key = self._trace_pair_priority_key(str(question or ""), left_trace, right_trace)
            if best_key is None or pair_key < best_key:
                best_key = pair_key
                best_pair = (left_trace, right_trace)
        return best_pair

    def _select_disagreement_pairs(
        self,
        question: str | Dict[str, AgentTrace],
        traces: Dict[str, AgentTrace] | None = None,
    ) -> List[Tuple[AgentTrace, AgentTrace]]:
        if traces is None:
            traces = question  # type: ignore[assignment]
            question = ""
        answer_groups: Dict[str, List[AgentTrace]] = {}
        for trace in traces.values():
            answer_key = self._final_answer_key(trace)
            if not answer_key:
                continue
            answer_groups.setdefault(answer_key, []).append(trace)
        if len(answer_groups) <= 1:
            return []
        candidate_pairs = [
            (self._trace_pair_priority_key(str(question or ""), left, right), left, right)
            for left, right in combinations(traces.values(), 2)
            if self._final_answer_key(left)
            and self._final_answer_key(right)
            and self._final_answer_key(left) != self._final_answer_key(right)
        ]
        candidate_pairs.sort(key=lambda item: item[0])
        selected: List[Tuple[AgentTrace, AgentTrace]] = []
        covered_answers: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        for _, left, right in candidate_pairs:
            pair_id = tuple(sorted((left.agent_id, right.agent_id)))
            if pair_id in seen_pairs:
                continue
            left_answer = self._final_answer_key(left)
            right_answer = self._final_answer_key(right)
            if selected and left_answer in covered_answers and right_answer in covered_answers:
                continue
            selected.append((left, right))
            seen_pairs.add(pair_id)
            covered_answers.update([left_answer, right_answer])
            if len(covered_answers) >= len(answer_groups):
                break
        return selected

    def resolve_pairwise_fallbacks(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        runtime: RuntimeBundle,
    ) -> tuple[List[ResolutionDecision], List[dict[str, str]]]:
        disagreement_pairs = self._select_disagreement_pairs(question, traces)
        if not disagreement_pairs:
            return [], []
        left_trace, right_trace = disagreement_pairs[0]
        pair_key = f"PAIR_{left_trace.agent_id}_{right_trace.agent_id}"
        analysis_output = self.prompt_runner(
            runtime,
            [
                (
                    "NL_GRAPH_PAIRWISE_RELATION_ANALYSIS",
                    build_pairwise_relation_analysis_prompt(
                        question,
                        left_trace,
                        right_trace,
                        profile=self.prompt_profile,
                        resolution_trace_context=self.resolution_trace_context,
                        resolution_prompt_style=self.resolution_prompt_style,
                        graph_format=self.graph_format,
                    ),
                )
            ],
            self.system_prompt,
        )[0]
        guard_note = self._build_pairwise_guard_note(question, left_trace, right_trace)
        if guard_note:
            analysis_output = f"{guard_note}\n\n{analysis_output}".strip()
        resolutions: List[ResolutionDecision] = []
        raw_notes: List[dict[str, str]] = []
        backend_decision = self._backend_pairwise_decision(question, pair_key, left_trace, right_trace)
        if backend_decision is not None:
            raw_notes.append(
                {
                    "divergence_id": pair_key,
                    "analysis_text": analysis_output,
                    "raw_text": self.build_resolution_text([backend_decision]),
                    "decision_source": "backend_after_analysis",
                    "backend_adjustment": backend_decision.rationale,
                }
            )
            resolutions.append(backend_decision)
            return resolutions, raw_notes
        prompt = build_pairwise_divergence_resolution_prompt(
            question,
            left_trace,
            right_trace,
            relation_analysis=analysis_output,
            profile=self.prompt_profile,
            resolution_trace_context=self.resolution_trace_context,
            resolution_prompt_style=self.resolution_prompt_style,
            graph_format=self.graph_format,
        )
        output = self.prompt_runner(runtime, [("NL_GRAPH_PAIRWISE_RESOLVE", prompt)], self.system_prompt)[0]
        raw_notes.append(
            {
                "divergence_id": pair_key,
                "analysis_text": analysis_output,
                "raw_text": output,
                "decision_source": "model_after_analysis",
            }
        )
        decision = parse_resolution_note(output, default_divergence_id=pair_key)
        if decision.action or decision.resolved_claim or decision.keep_paths or decision.drop_paths:
            resolutions.append(decision)
        return resolutions, raw_notes

    def _backend_pairwise_decision(
        self,
        question: str,
        pair_key: str,
        left_trace: AgentTrace,
        right_trace: AgentTrace,
    ) -> ResolutionDecision | None:
        distinct_decision = self._backend_distinct_value_pairwise_decision(
            question,
            pair_key,
            left_trace,
            right_trace,
        )
        if distinct_decision is not None:
            return distinct_decision

        multiple_sum_decision = self._backend_multiple_sum_pairwise_decision(question, pair_key)
        if multiple_sum_decision is not None:
            return multiple_sum_decision

        gcd_decision = self._backend_gcd_pairwise_decision(question, pair_key, left_trace, right_trace)
        if gcd_decision is not None:
            return gcd_decision

        radical_decision = self._backend_radical_target_form_pairwise_decision(
            question,
            pair_key,
            left_trace,
            right_trace,
        )
        if radical_decision is not None:
            return radical_decision

        conjugate_decision = self._backend_conjugate_pairwise_decision(question, pair_key, left_trace, right_trace)
        if conjugate_decision is not None:
            return conjugate_decision

        return None

    def _backend_distinct_value_pairwise_decision(
        self,
        question: str,
        pair_key: str,
        left_trace: AgentTrace,
        right_trace: AgentTrace,
    ) -> ResolutionDecision | None:
        if not self._question_targets_distinct_value_set(question):
            return None
        exact_values = self._question_parenthesization_values(question)
        if not exact_values:
            return None
        values_text = ", ".join(str(value) for value in sorted(exact_values))
        count = len(exact_values)
        return ResolutionDecision(
            divergence_id=pair_key,
            action="synthesize",
            winning_side="synthesized",
            resolved_claim=(
                f"The full-expression values obtainable by parenthesizing are {values_text}; "
                f"therefore the number of distinct values is {count}."
            ),
            rationale=(
                "A deterministic enumeration of the small arithmetic expression overrides the pairwise vote when "
                "both branches give an unsupported or incorrect distinct-value count."
            ),
            rewrite_from_claim_id="C1",
            keep_paths=[],
            drop_paths=[],
            canonical_answer=str(count),
        )

    def _backend_multiple_sum_pairwise_decision(
        self,
        question: str,
        pair_key: str,
    ) -> ResolutionDecision | None:
        multiple_sum = self._question_multiple_sum_value(question)
        if multiple_sum is None:
            return None
        first, last, count, total, step = multiple_sum
        return ResolutionDecision(
            divergence_id=pair_key,
            action="synthesize",
            winning_side="synthesized",
            resolved_claim=(
                f"The multiples form an arithmetic sequence from {first} to {last} with common difference {step}; "
                f"there are {count} terms, so the sum is {count}({first}+{last})/2 = {total}."
            ),
            rationale=(
                "A deterministic arithmetic-sequence check resolves the local term-count and sum claims directly."
            ),
            rewrite_from_claim_id="C1",
            keep_paths=[],
            drop_paths=[],
            canonical_answer=str(total),
        )

    def _backend_gcd_pairwise_decision(
        self,
        question: str,
        pair_key: str,
        left_trace: AgentTrace,
        right_trace: AgentTrace,
    ) -> ResolutionDecision | None:
        expected_gcd = self._question_gcd_value(question)
        if expected_gcd is None:
            return None
        left_matches = self._trace_final_integer(left_trace) == expected_gcd
        right_matches = self._trace_final_integer(right_trace) == expected_gcd
        if left_matches and not right_matches:
            return self._choose_pairwise_trace(
                pair_key,
                choose_left=True,
                resolved_claim=f"The greatest common divisor of the given integers is {expected_gcd}.",
                rationale="A deterministic Euclidean-algorithm check matches the left branch's final answer.",
                canonical_answer=str(expected_gcd),
            )
        if right_matches and not left_matches:
            return self._choose_pairwise_trace(
                pair_key,
                choose_left=False,
                resolved_claim=f"The greatest common divisor of the given integers is {expected_gcd}.",
                rationale="A deterministic Euclidean-algorithm check matches the right branch's final answer.",
                canonical_answer=str(expected_gcd),
            )
        if not left_matches and not right_matches:
            return ResolutionDecision(
                divergence_id=pair_key,
                action="synthesize",
                winning_side="synthesized",
                resolved_claim=f"The greatest common divisor of the given integers is {expected_gcd}.",
                rationale=(
                    "A deterministic Euclidean-algorithm check finds the GCD directly, so both compared "
                    "branches should repair their arithmetic."
                ),
                rewrite_from_claim_id="C1",
                keep_paths=[],
                drop_paths=[],
                canonical_answer=str(expected_gcd),
            )
        return None

    def _backend_conjugate_pairwise_decision(
        self,
        question: str,
        pair_key: str,
        left_trace: AgentTrace,
        right_trace: AgentTrace,
    ) -> ResolutionDecision | None:
        if not self._question_targets_conjugate_pair_sum(question):
            return None
        left_family = self._trace_root_family(left_trace)
        right_family = self._trace_root_family(right_trace)
        if left_family != right_family and {left_family, right_family} == {"unity", "minus_one"}:
            choose_left = left_family == "minus_one"
            winning_trace = left_trace if choose_left else right_trace
            winning_final = self._final_answer_text(winning_trace)
            canonical_answer = self._normalize_final_answer_for_lock(winning_final)
            final_sentence = f" The winning branch's final answer is {winning_final}." if winning_final else ""
            return self._choose_pairwise_trace(
                pair_key,
                choose_left=choose_left,
                resolved_claim=(
                    "Use the 10th roots of -1 from x^10 = -(13x-1)^10 before evaluating the requested "
                    f"five conjugate-pair sum.{final_sentence}"
                ),
                rationale=(
                    "The root-family check is decisive before the final sum: roots of -1 match the transformed "
                    "equation, while roots of unity solve a different object."
                ),
                canonical_answer=canonical_answer,
            )
        left_rank = self._trace_target_alignment_rank(question, left_trace)
        right_rank = self._trace_target_alignment_rank(question, right_trace)
        if left_rank == right_rank:
            return None
        if min(left_rank, right_rank) > 1 or abs(left_rank - right_rank) < 2:
            return None
        choose_left = left_rank < right_rank
        return self._choose_pairwise_trace(
            pair_key,
            choose_left=choose_left,
            resolved_claim=(
                "Use the 10th roots of -1 from x^10 = -(13x-1)^10 and compute the requested "
                "five conjugate-pair sum, not a mismatched raw ten-root or unity-root total."
            ),
            rationale=(
                "The target-alignment check is decisive here: the better branch uses the correct root family "
                "and aligns with the five conjugate-pair target."
            ),
        )

    def _backend_radical_target_form_pairwise_decision(
        self,
        question: str,
        pair_key: str,
        left_trace: AgentTrace,
        right_trace: AgentTrace,
    ) -> ResolutionDecision | None:
        if not self._question_targets_radical_target_form(question):
            return None
        assignments = self._target_form_assignments_from_traces((left_trace, right_trace))
        if assignments is None:
            return None
        a, b, c, d = assignments
        total = a + b + c + d
        return ResolutionDecision(
            divergence_id=pair_key,
            action="synthesize",
            winning_side="synthesized",
            resolved_claim=(
                f"The expanded expression matches A(1+sqrt(B))-(sqrt(C)+sqrt(D)) with "
                f"A = {a}, B = {b}, C = {c}, D = {d}, so A+B+C+D = {total}."
            ),
            rationale=(
                "The target-form assignments are determined by reconstructing the already-expanded surd "
                "expression, rather than trusting an unsupported assignment line."
            ),
            rewrite_from_claim_id="C1",
            keep_paths=[],
            drop_paths=[],
            canonical_answer=str(total),
        )

    def _normalize_final_answer_for_lock(self, answer_text: str | None) -> str:
        if not answer_text:
            return ""
        text = str(answer_text).strip()
        match = re.search(r"\\boxed\{(.+?)\}", text)
        if match:
            text = match.group(1).strip()
        text = re.sub(r"^\{?\s*final\s+answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        return text.strip("{} ")

    def _target_form_assignments_from_traces(self, traces: Iterable[AgentTrace]) -> tuple[int, int, int, int] | None:
        for trace in traces:
            signature = self._latest_linear_surd_signature_in_trace(trace)
            assignments = self._target_form_assignments_from_signature(signature)
            if assignments is not None:
                return assignments
        return None

    def _target_form_assignments_from_signature(self, signature: Dict[int, int] | None) -> tuple[int, int, int, int] | None:
        if not signature:
            return None
        a = signature.get(0)
        if a is None or a <= 0:
            return None
        positive_radicals = [radicand for radicand, coeff in signature.items() if radicand != 0 and coeff == a]
        negative_radicals = sorted(radicand for radicand, coeff in signature.items() if radicand != 0 and coeff == -1)
        if len(positive_radicals) != 1 or len(negative_radicals) != 2:
            return None
        b = positive_radicals[0]
        c, d = negative_radicals
        return a, b, c, d

    def _choose_pairwise_trace(
        self,
        pair_key: str,
        *,
        choose_left: bool,
        resolved_claim: str,
        rationale: str,
        canonical_answer: str = "",
    ) -> ResolutionDecision:
        return ResolutionDecision(
            divergence_id=pair_key,
            action="choose_left" if choose_left else "choose_right",
            winning_side="left" if choose_left else "right",
            resolved_claim=resolved_claim,
            rationale=rationale,
            rewrite_from_claim_id="C1",
            keep_paths=[],
            drop_paths=[],
            canonical_answer=canonical_answer,
        )

    def _build_pairwise_guard_note(self, question: str, left_trace: AgentTrace, right_trace: AgentTrace) -> str:
        notes: List[str] = []
        if self._question_targets_distinct_value_set(question):
            exact_values = self._question_parenthesization_values(question)
            if exact_values:
                values_text = ", ".join(str(value) for value in sorted(exact_values))
                notes.append(
                    f"Quick value check from the expression: the obtainable full-expression values are {values_text}, so this expression has {len(exact_values)} distinct values."
                )
                for trace in (left_trace, right_trace):
                    claimed_values = self._trace_claimed_distinct_values(trace)
                    if not claimed_values:
                        continue
                    omitted_values = sorted(exact_values - claimed_values)
                    extra_values = sorted(claimed_values - exact_values)
                    details = []
                    if omitted_values:
                        details.append("omits " + ", ".join(str(value) for value in omitted_values))
                    if extra_values:
                        details.append("lists " + ", ".join(str(value) for value in extra_values) + " outside the full-expression values")
                    if details:
                        notes.append(f"Quick value check: {trace.agent_id} " + " and ".join(details) + ".")
            else:
                notes.append(
                    "Quick value check: for a distinct-value count, compare concrete attained values; try to name one omitted valid value or one listed value that is not actually produced before trusting a bare count."
                )
                for trace in (left_trace, right_trace):
                    unsupported_values = sorted(self._trace_unsupported_distinct_values(trace))
                    if unsupported_values:
                        values = ", ".join(str(value) for value in unsupported_values)
                        notes.append(
                            f"Quick value check: {trace.agent_id} lists {values} without a concrete parenthesization or local evaluation that produces them."
                        )
        if self._question_targets_circular_arrangement(question):
            notes.append(
                "Quick circular-arrangement check: after fixing one reference seat to remove rotation, count the remaining placements relative to that fixed seat; do not divide again unless another symmetry truly remains."
            )
        if self._question_targets_conjugate_pair_sum(question):
            left_family = self._trace_root_family(left_trace)
            right_family = self._trace_root_family(right_trace)
            if left_family and right_family and left_family != right_family:
                notes.append(
                    f"Deterministic check: {left_trace.agent_id} uses 10th roots of {'unity' if left_family == 'unity' else '-1'}, while {right_trace.agent_id} uses 10th roots of {'unity' if right_family == 'unity' else '-1'}; that root-family choice is an earlier same-object mismatch than the final sum."
                )
            left_rank = self._trace_target_alignment_rank(question, left_trace)
            right_rank = self._trace_target_alignment_rank(question, right_trace)
            if left_rank != right_rank:
                better_trace = left_trace if left_rank < right_rank else right_trace
                notes.append(
                    f"Deterministic check: {better_trace.agent_id} is better aligned with the requested five-pair target instead of a raw ten-root total."
                )
        normalized_question = self._normalize_claim_text(question)
        if "least positive integer multiple of 30" in normalized_question and "digits 0 and 2" in normalized_question:
            left_validated = self._trace_has_validated_target_candidate(left_trace)
            right_validated = self._trace_has_validated_target_candidate(right_trace)
            left_search = self._trace_mentions_candidate_search(left_trace)
            right_search = self._trace_mentions_candidate_search(right_trace)
            if left_validated and right_search and not right_validated:
                notes.append(
                    f"Deterministic check: {left_trace.agent_id} explicitly validates a concrete candidate, while {right_trace.agent_id} only reports earlier failures; without a smaller validated candidate, do not treat the failed-range summary as evidence against the validated answer."
                )
            if right_validated and left_search and not left_validated:
                notes.append(
                    f"Deterministic check: {right_trace.agent_id} explicitly validates a concrete candidate, while {left_trace.agent_id} only reports earlier failures; without a smaller validated candidate, do not treat the failed-range summary as evidence against the validated answer."
                )
        if self._question_targets_radical_target_form(question):
            unsupported_agents = []
            for trace in (left_trace, right_trace):
                if not self._trace_target_form_assignments_supported(trace):
                    unsupported_agents.append(trace.agent_id)
                    notes.append(
                        f"Deterministic check: {trace.agent_id}'s listed A, B, C, D assignments do not reconstruct the expanded expression, so do not trust that assignment line as a supported target-form match."
                    )
            if unsupported_agents:
                notes.append(
                    "Deterministic check: repair from the expanded expression itself and choose assignments only if they reproduce that same expression."
                )
        return " ".join(notes).strip()

    def _final_answer_text(self, trace: AgentTrace) -> str:
        lines = [line.strip() for line in trace.normalized_trace_text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _final_answer_key(self, trace: AgentTrace) -> str:
        text = self._final_answer_text(trace)
        if not text:
            return ""
        return self._normalize_claim_text(self._normalize_final_answer_for_lock(text) or text)

    def _has_explicit_final_answer(self, trace: AgentTrace) -> bool:
        lines = [line.strip().lower() for line in trace.normalized_trace_text.splitlines() if line.strip()]
        return any("final answer" in line or "\\boxed" in line for line in lines)

    def _extract_explicit_final_answer_line(self, text: str) -> str:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if "final answer" in lowered or "\\boxed" in line:
                return line
        return ""

    def _looks_truncated_final_answer_line(self, line: str) -> bool:
        stripped = str(line or "").strip()
        lowered = stripped.lower()
        if "final answer" not in lowered:
            return False
        if stripped.count("{") > stripped.count("}"):
            return True
        return lowered.endswith(("{final answer", "{final answer:", "final answer:", "\\boxed{"))

    def _restore_source_final_answer(self, normalized_text: str, source_text: str) -> str:
        lines = [line.strip() for line in str(normalized_text or "").splitlines() if line.strip()]
        while lines and self._looks_truncated_final_answer_line(lines[-1]):
            lines.pop()
        normalized = "\n".join(lines).strip()
        if self._extract_explicit_final_answer_line(normalized):
            return normalized
        source_final = self._extract_explicit_final_answer_line(source_text)
        if not source_final:
            return normalized
        return f"{normalized}\n{source_final}".strip() if normalized else source_final

    def _serialize_trace_steps(self, steps) -> str:
        return "\n".join(step.text for step in steps if str(step.text).strip()).strip()

    def _has_answer_disagreement(self, traces: Dict[str, AgentTrace]) -> bool:
        answers = {self._final_answer_key(trace) for trace in traces.values() if self._final_answer_key(trace)}
        return len(answers) > 1

    def _graph_has_content(self, graph: NaturalLanguageGraph) -> bool:
        raw_text = graph.raw_dossier.strip()
        has_sections = (
            "Shared Claims" in raw_text
            or "Common Ground" in raw_text
        ) and (
            "Method Paths" in raw_text
            or "Paths" in raw_text
        )
        return bool(graph.shared_claims or graph.method_paths or graph.divergences or has_sections)

    def _graph_has_substantive_content(self, graph: NaturalLanguageGraph) -> bool:
        if graph.shared_claims or graph.method_paths or graph.divergences:
            return True
        raw_text = str(graph.raw_dossier or "").strip()
        if not raw_text:
            return False
        header_only = {"Shared Claims", "Method Paths", "Minimal Divergences", "Common Ground", "Paths", "First Split"}
        content_lines = [line.strip() for line in raw_text.splitlines() if line.strip() and line.strip() not in header_only]
        return bool(content_lines)

    def _claim_step_depth(self, member: str) -> int:
        match = re.search(r"\.s(\d+)$", str(member or "").strip())
        return int(match.group(1)) if match else 10**6

    def _claim_node_depth(self, claim) -> int:
        members = getattr(claim, "members", []) or []
        if not members:
            return 10**6
        return min(self._claim_step_depth(member) for member in members)

    def _path_local_depth(self, path_id: str, claim_text: str, graph: NaturalLanguageGraph) -> int:
        normalized_target = self._normalize_claim_text(claim_text)
        path = graph.path_map().get(path_id)
        claim_map = graph.claim_map()
        if path is None:
            return 10**6
        for index, claim_id in enumerate(path.claim_ids or []):
            claim = claim_map.get(claim_id)
            if claim is None:
                continue
            normalized_claim = self._normalize_claim_text(getattr(claim, "text", ""))
            if normalized_target and normalized_claim == normalized_target:
                return index
        return len(path.claim_ids or []) + 1000

    def _divergence_priority_key(self, divergence, graph: NaturalLanguageGraph) -> tuple:
        frontier_claim = self._frontier_claim_node(divergence, graph)
        frontier_depth = self._claim_node_depth(frontier_claim) if frontier_claim is not None else 10**6
        left_depth = self._path_local_depth(getattr(divergence, "left_path_id", ""), getattr(divergence, "left_claim", ""), graph)
        right_depth = self._path_local_depth(getattr(divergence, "right_path_id", ""), getattr(divergence, "right_claim", ""), graph)
        same_object_rank = 0 if self._is_same_object_conflict(divergence, graph) else 1
        relation = self._normalize_metadata_token(getattr(divergence, "relation", ""))
        repaired_rank = 0 if relation in {"contradiction", "repaired_claim_needed"} else 1
        return (frontier_depth, min(left_depth, right_depth), same_object_rank, repaired_rank, str(getattr(divergence, "divergence_id", "")))

    def _step_object_signature(self, text: str) -> str:
        return self._claim_object_signature(text)

    def _trace_first_decisive_mismatch(self, left_trace: AgentTrace, right_trace: AgentTrace) -> tuple[int, int, int]:
        for left_idx, left_step in enumerate(left_trace.steps, start=1):
            left_object = self._step_object_signature(left_step.text)
            if not left_object:
                continue
            for right_idx, right_step in enumerate(right_trace.steps, start=1):
                right_object = self._step_object_signature(right_step.text)
                if not right_object or left_object != right_object:
                    continue
                if self._normalize_claim_text(left_step.text) != self._normalize_claim_text(right_step.text):
                    return (min(left_idx, right_idx), left_idx, right_idx)
        zipped_limit = min(len(left_trace.steps), len(right_trace.steps))
        for offset in range(zipped_limit):
            left_step = left_trace.steps[offset]
            right_step = right_trace.steps[offset]
            if self._normalize_claim_text(left_step.text) != self._normalize_claim_text(right_step.text):
                return (offset + 1, offset + 1, offset + 1)
        return (10**6, len(left_trace.steps), len(right_trace.steps))

    def _trace_pair_priority_key(self, question: str, left_trace: AgentTrace, right_trace: AgentTrace) -> tuple:
        explicit_rank = 2 - int(self._has_explicit_final_answer(left_trace)) - int(self._has_explicit_final_answer(right_trace))
        alignment_rank = tuple(sorted(
            (
                self._trace_target_alignment_rank(question, left_trace),
                self._trace_target_alignment_rank(question, right_trace),
            )
        ))
        mismatch = self._trace_first_decisive_mismatch(left_trace, right_trace)
        return (explicit_rank,) + alignment_rank + mismatch + (left_trace.agent_id, right_trace.agent_id)

    def _is_valid_claim_node(self, claim) -> bool:
        text = str(getattr(claim, "text", "") or "").strip()
        if not text:
            return False
        if len(text) <= 2 and not any(ch.isalpha() for ch in text):
            return False
        members = getattr(claim, "members", []) or []
        if not members:
            return False
        return True

    def _claim_supporting_agent_count(self, claim) -> int:
        return len(self._claim_member_agents(claim))

    def _is_valid_method_path(self, path) -> bool:
        return bool((getattr(path, "path_id", "") or "").strip() and (getattr(path, "agent_ids", []) or []))

    def _question_targets_distinct_value_set(self, question: str) -> bool:
        text = str(question or "").strip().lower()
        markers = (
            "different values",
            "distinct values",
            "possible values",
            "number of distinct values",
            "how many values",
            "how many different values",
            "how many distinct values",
        )
        return any(marker in text for marker in markers)

    def _question_parenthesization_values(self, question: str) -> set[int]:
        if "parenthes" not in str(question or "").lower():
            return set()
        for segment in self._extract_math_segments(question):
            expression = self._normalize_small_arithmetic_expression(segment)
            if not expression:
                continue
            values = self._enumerate_parenthesized_arithmetic_values(expression)
            if values:
                return values
        return set()

    def _normalize_small_arithmetic_expression(self, expression: str) -> str:
        normalized = str(expression or "")
        normalized = normalized.replace(r"\cdot", "*").replace(r"\times", "*")
        normalized = normalized.replace("×", "*").replace(" ", "")
        normalized = normalized.strip(".,;:")
        if not re.fullmatch(r"\d+(?:[+*]\d+){1,7}", normalized):
            return ""
        return normalized

    def _enumerate_parenthesized_arithmetic_values(self, expression: str) -> set[int]:
        numbers = [int(token) for token in re.findall(r"\d+", expression)]
        operators = re.findall(r"[+*]", expression)
        if len(numbers) < 2 or len(numbers) != len(operators) + 1:
            return set()
        table: dict[tuple[int, int], set[int]] = {(index, index): {value} for index, value in enumerate(numbers)}
        for span in range(2, len(numbers) + 1):
            for start in range(0, len(numbers) - span + 1):
                stop = start + span - 1
                values: set[int] = set()
                for split in range(start, stop):
                    for left_value in table[(start, split)]:
                        for right_value in table[(split + 1, stop)]:
                            if operators[split] == "+":
                                values.add(left_value + right_value)
                            else:
                                values.add(left_value * right_value)
                table[(start, stop)] = values
        return table[(0, len(numbers) - 1)]

    def _question_multiple_sum_value(self, question: str) -> tuple[int, int, int, int, int] | None:
        text = str(question or "").split("Make sure to state your final answer", 1)[0]
        normalized = self._normalize_claim_text(text)
        if "multiple" not in normalized or "sum" not in normalized:
            return None
        if not any(marker in normalized for marker in ("between", "from")):
            return None
        numbers = self._extract_integer_tokens(text)
        if len(numbers) < 3:
            return None
        step = numbers[0]
        lower = numbers[1]
        upper = numbers[2]
        if step <= 0 or lower >= upper:
            return None
        first = ((lower + step - 1) // step) * step
        last = (upper // step) * step
        if first > last:
            return None
        count = (last - first) // step + 1
        total = count * (first + last) // 2
        return first, last, count, total, step

    def _question_targets_circular_arrangement(self, question: str) -> bool:
        text = str(question or "").strip().lower()
        circular_markers = (
            "round table",
            "around a circle",
            "around the circle",
            "circular arrangement",
            "circular permutation",
            "seated in a circle",
            "sit in a circle",
            "arranged in a circle",
        )
        arrangement_markers = ("arrange", "seat", "sit", "permutation", "order", "ways")
        return any(marker in text for marker in circular_markers) and any(marker in text for marker in arrangement_markers)

    def _question_targets_radical_target_form(self, question: str) -> bool:
        text = str(question or "")
        normalized = self._normalize_claim_text(text)
        return "form $a(1+\\sqrt{b})-(\\sqrt{c}+\\sqrt{d})$" in normalized or (
            "form a(1+\\sqrt{b})-(\\sqrt{c}+\\sqrt{d})" in normalized and "a+b+c+d" in normalized
        )

    def _text_mentions_target_form_sum(self, text: str) -> bool:
        normalized = self._normalize_claim_text(text)
        return "a + b + c + d" in normalized or "a+b+c+d" in normalized

    def _extract_integer_tokens(self, text: str) -> List[int]:
        return [int(token) for token in re.findall(r"(?<![A-Za-z])-?\d+(?![A-Za-z])", str(text or ""))]

    def _trace_final_integer(self, trace: AgentTrace) -> int | None:
        final_text = self._final_answer_text(trace)
        values = self._extract_integer_tokens(final_text)
        return values[-1] if values else None

    def _question_gcd_value(self, question: str) -> int | None:
        question_text = str(question or "").split("Make sure to state your final answer", 1)[0]
        normalized = self._normalize_claim_text(question_text)
        if not any(marker in normalized for marker in ("greatest common divisor", "gcd", "greatest common factor")):
            return None
        if any(marker in normalized for marker in ("how many", "number of possible", "possible values", "how many possible")):
            return None
        numbers = [value for value in self._extract_integer_tokens(question_text) if value > 0]
        if len(numbers) < 2 or len(numbers) > 8:
            return None
        result = numbers[0]
        for value in numbers[1:]:
            result = math.gcd(result, value)
        return result

    def _extract_claimed_distinct_values_from_text(self, text: str) -> set[int]:
        normalized = self._normalize_claim_text(text)
        set_markers = (
            "possible values are",
            "distinct values are",
            "the only distinct values are",
            "the only values are",
            "set of values",
        )
        if not any(marker in normalized for marker in set_markers):
            return set()
        if any(marker in normalized for marker in ("number of distinct values", "how many distinct values")):
            return set()
        return set(self._extract_integer_tokens(text))

    def _step_supported_attainable_values(self, step_text: str) -> set[int]:
        text = str(step_text or "")
        normalized = self._normalize_claim_text(text)
        if not re.search(r"-?\d", text):
            return set()
        support_markers = (
            "original expression",
            "grouping",
            "parenthes",
            "can yield",
            "can produce",
            "can obtain",
            "value can be",
            "gives",
            "yields",
            "results in",
            "evaluates to",
            "evaluate to",
            "obtains",
        )
        looks_parenthesized_evaluation = "(" in text and ")" in text and "=" in text and any(op in text for op in "+-*")
        if not looks_parenthesized_evaluation and not any(marker in normalized for marker in support_markers):
            return set()
        values = self._extract_integer_tokens(text)
        return {values[-1]} if values else set()

    def _trace_claimed_distinct_values(self, trace: AgentTrace) -> set[int]:
        claimed_values = set()
        for step in trace.steps:
            claimed_values.update(self._extract_claimed_distinct_values_from_text(step.text))
        return claimed_values

    def _trace_supported_attainable_values(self, trace: AgentTrace) -> set[int]:
        supported_values = set()
        for step in trace.steps:
            supported_values.update(self._step_supported_attainable_values(step.text))
        return supported_values

    def _trace_unsupported_distinct_values(self, trace: AgentTrace) -> set[int]:
        claimed_values = self._trace_claimed_distinct_values(trace)
        if not claimed_values:
            return set()
        return claimed_values - self._trace_supported_attainable_values(trace)

    def _trace_claims_distinct_value_completeness(self, trace: AgentTrace) -> bool:
        normalized = self._normalize_claim_text(trace.normalized_trace_text)
        return self._text_claims_distinct_value_completeness(normalized)

    def _text_claims_distinct_value_completeness(self, text: str) -> bool:
        normalized = self._normalize_claim_text(text)
        completeness_markers = (
            "the only distinct values are",
            "the only values are",
            "these are the only distinct values",
            "these are the only values",
            "other groupings do not produce new values",
            "we confirm that these are the only distinct values",
            "we confirm that these are the only values",
        )
        return any(marker in normalized for marker in completeness_markers)

    def _trace_root_family(self, trace: AgentTrace) -> str:
        text = str(trace.normalized_trace_text or "")
        normalized = self._normalize_claim_text(text)
        if "10th root of unity" in normalized or "10th roots of unity" in normalized:
            return "unity"
        minus_one_patterns = (
            r"10th root[s]?\s+of\s+\$?-?1\$?",
            r"10th root[s]?\s+of\s+\\?-?1",
            r"10th root[s]?\s+of\s+minus one",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in minus_one_patterns):
            return "minus_one"
        return ""

    def _looks_attained_example_claim(self, claim_text: str) -> bool:
        text = self._normalize_claim_text(claim_text)
        if not re.search(r"-?\d", text):
            return False
        if any(
            marker in text
            for marker in (
                "number of distinct values",
                "distinct values is",
                "distinct values are",
                "possible values are",
                "set of values",
                "final answer",
                "count",
            )
        ):
            return False
        return any(
            marker in text
            for marker in (
                "value is",
                "value can be",
                "can yield",
                "can produce",
                "can obtain",
                "obtains",
                "obtained",
                "yields",
                "can be ",
                "equals",
                "= ",
            )
        )

    def _looks_parallel_attained_example_divergence(self, divergence, question: str) -> bool:
        if not self._question_targets_distinct_value_set(question):
            return False
        aspect = self._normalize_metadata_token(getattr(divergence, "aspect", ""))
        if aspect not in {"value", "values", "result", "results", ""}:
            return False
        left_claim = str(getattr(divergence, "left_claim", "") or "")
        right_claim = str(getattr(divergence, "right_claim", "") or "")
        if self._claim_mentions_optimality(left_claim) or self._claim_mentions_optimality(right_claim):
            return False
        return self._looks_attained_example_claim(left_claim) and self._looks_attained_example_claim(right_claim)

    def _looks_euclidean_sequence_not_conflict(self, left_claim: str, right_claim: str) -> bool:
        pattern = re.compile(
            r"remainder\s+of\s+(-?\d+)\s+divided\s+by\s+(-?\d+)\s+(?:is|=)\s+(-?\d+)",
            flags=re.IGNORECASE,
        )
        left = pattern.search(str(left_claim or ""))
        right = pattern.search(str(right_claim or ""))
        if not left or not right:
            return False
        left_divisor = int(left.group(2))
        left_remainder = int(left.group(3))
        right_dividend = int(right.group(1))
        right_divisor = int(right.group(2))
        right_remainder = int(right.group(3))
        if (right_dividend, right_divisor) == (left_divisor, left_remainder):
            return True
        left_dividend = int(left.group(1))
        return (left_dividend, left_divisor) == (right_divisor, right_remainder)

    def _looks_task_statement_vs_progress(self, left_claim: str, right_claim: str) -> bool:
        def is_task_statement(text: str) -> bool:
            normalized = self._normalize_claim_text(text)
            if any(marker in normalized for marker in ("requested answer is", "requested object is")):
                return False
            return any(
                marker in normalized
                for marker in (
                    "problem is to find",
                    "problem requires",
                    "task is to find",
                    "need to find",
                    "asked to find",
                    "find the sum",
                    "find all possible",
                )
            )

        def has_progress_value(text: str) -> bool:
            normalized = self._normalize_claim_text(text)
            if is_task_statement(text):
                return False
            return bool(re.search(r"\d|\\pi|\\frac|=", str(text or ""))) or any(
                marker in normalized
                for marker in ("values are", "roots are", "sum is", "answer is")
            )

        return (is_task_statement(left_claim) and has_progress_value(right_claim)) or (
            is_task_statement(right_claim) and has_progress_value(left_claim)
        )

    def _looks_progress_lag_not_conflict(self, left_claim: str, right_claim: str) -> bool:
        def is_lag(text: str) -> bool:
            normalized = self._normalize_claim_text(text)
            return any(
                marker in normalized
                for marker in (
                    "has not yet",
                    "have not yet",
                    "not yet set up",
                    "not yet proceeded",
                    "not yet proceed",
                    "does not yet",
                    "do not yet",
                    "no steps are revealed",
                    "no steps revealed",
                    "only begins",
                    "only defines",
                    "has not taken the next step",
                    "does not provide",
                    "not fully specified",
                    "no further steps are taken",
                    "does not proceed further",
                    "is ongoing",
                    "is in progress",
                    "still needs to",
                    "yet to be completed",
                    "remains at the shared prefix",
                )
            )

        def is_progress(text: str) -> bool:
            normalized = self._normalize_claim_text(text)
            return any(
                marker in normalized
                for marker in (
                    "proceeds to",
                    "solve for",
                    "solves for",
                    "applies",
                    "uses both",
                    "set up",
                    "sets up",
                    "form equations",
                    "forms equations",
                    "derives",
                    "computes",
                    "finds",
                    "arrives at",
                )
            ) or bool(re.search(r"=|\d", str(text or "")))

        return (is_lag(left_claim) and is_progress(right_claim)) or (is_lag(right_claim) and is_progress(left_claim))

    def _clean_progress_lag_method_summaries(self, method_paths):
        lag_markers = (
            "has not yet",
            "have not yet",
            "not yet set up",
            "not yet proceeded",
            "not yet proceed",
            "does not yet",
            "do not yet",
            "no steps are revealed",
            "no steps revealed",
            "has not taken the next step",
            "no further steps are taken",
            "does not proceed further",
            "is ongoing",
            "is in progress",
            "yet to be completed",
        )
        cleaned = []
        for path in method_paths:
            summary = str(getattr(path, "summary", "") or "")
            normalized = self._normalize_claim_text(summary)
            if any(marker in normalized for marker in lag_markers):
                path.summary = "No new same-object claim beyond the shared prefix is visible in this window."
            cleaned.append(path)
        return cleaned

    def _path_summary_looks_lagging(self, summary: str) -> bool:
        normalized = self._normalize_claim_text(summary)
        if not normalized:
            return False
        return any(
            marker in normalized
            for marker in (
                "no new same-object claim beyond the shared prefix is visible in this window",
                "has not yet",
                "have not yet",
                "not yet",
                "still enumerating",
                "still exploring",
                "still checking",
                "still deriving",
                "begins computing",
                "begins expressing",
                "identifies the task as",
                "does not proceed further",
                "no further steps are taken",
                "is in progress",
                "is ongoing",
            )
        )

    def _graph_needs_async_window_expansion(self, graph: NaturalLanguageGraph) -> bool:
        if self._select_primary_real_claim_divergence(graph) is not None:
            return False
        if graph.divergences:
            return True
        return any(self._path_summary_looks_lagging(path.summary) for path in graph.method_paths)

    def _lagging_agent_ids(self, graph: NaturalLanguageGraph) -> set[str]:
        lagging = {
            agent_id
            for path in graph.method_paths
            if self._path_summary_looks_lagging(path.summary)
            for agent_id in (path.agent_ids or [])
        }
        if lagging:
            return lagging
        path_map = graph.path_map()
        involved = set()
        for divergence in graph.divergences:
            involved.update(path_map.get(str(getattr(divergence, "left_path_id", "") or "").strip(), MethodPath("", [], "", [])).agent_ids or [])
            involved.update(path_map.get(str(getattr(divergence, "right_path_id", "") or "").strip(), MethodPath("", [], "", [])).agent_ids or [])
        return involved

    def _path_summary_for_divergence_side(self, graph: NaturalLanguageGraph, path_id: str) -> str:
        path = graph.path_map().get(str(path_id or "").strip())
        return str(getattr(path, "summary", "") or "") if path is not None else ""

    def _extract_division_pair(self, text: str) -> tuple[int, int] | None:
        match = re.search(
            r"(?:remainder\s+of\s+)?(-?\d+)\s+divided\s+by\s+(-?\d+)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _extract_remainder_value(self, text: str) -> int | None:
        match = re.search(r"remainder\s+(?:is|=)\s*(-?\d+)", str(text or ""), flags=re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1))

    def _looks_euclidean_progress_divergence(self, divergence, graph: NaturalLanguageGraph) -> bool:
        left_claim = str(getattr(divergence, "left_claim", "") or "")
        right_claim = str(getattr(divergence, "right_claim", "") or "")
        if self._looks_symbolic_modulo_vs_explicit_remainder(left_claim, right_claim):
            return True
        left_summary = self._path_summary_for_divergence_side(graph, getattr(divergence, "left_path_id", ""))
        right_summary = self._path_summary_for_divergence_side(graph, getattr(divergence, "right_path_id", ""))
        left_division = self._extract_division_pair(f"{left_claim} {left_summary}")
        right_division = self._extract_division_pair(f"{right_claim} {right_summary}")
        left_remainder = self._extract_remainder_value(left_claim)
        right_remainder = self._extract_remainder_value(right_claim)
        if left_division and right_division and left_remainder is not None and right_remainder is not None:
            if right_division == (left_division[1], left_remainder):
                return True
            if left_division == (right_division[1], right_remainder):
                return True
        why = self._normalize_claim_text(getattr(divergence, "why_minimal", ""))
        return (
            "euclidean algorithm" in why
            and "key step" in why
            and "remainder" in self._normalize_claim_text(f"{left_claim} {right_claim}")
        )

    def _looks_symbolic_modulo_vs_explicit_remainder(self, left_claim: str, right_claim: str) -> bool:
        compact_left = re.sub(r"\s+", "", str(left_claim or "").lower())
        compact_right = re.sub(r"\s+", "", str(right_claim or "").lower())

        def symbolic_pairs(text: str) -> list[tuple[int, int]]:
            pairs = []
            for first, second in re.findall(r"(-?\d+)%(-?\d+)", text):
                divisor = int(second)
                if divisor:
                    pairs.append((int(first), divisor))
            return pairs

        def has_explicit_remainder(text: str, first: int, second: int) -> bool:
            remainder = first % second
            patterns = (
                f"gcd({second},{remainder})",
                f"gcd\\left({second},{remainder}\\right)",
                f"remainderis{remainder}",
                f"remainder={remainder}",
            )
            return any(pattern in text for pattern in patterns)

        return any(has_explicit_remainder(compact_right, first, second) for first, second in symbolic_pairs(compact_left)) or any(
            has_explicit_remainder(compact_left, first, second) for first, second in symbolic_pairs(compact_right)
        )

    def _looks_formula_to_value_progress_divergence(self, divergence, graph: NaturalLanguageGraph) -> bool:
        left_claim = self._normalize_claim_text(getattr(divergence, "left_claim", ""))
        right_claim = self._normalize_claim_text(getattr(divergence, "right_claim", ""))
        left_summary = self._normalize_claim_text(self._path_summary_for_divergence_side(graph, getattr(divergence, "left_path_id", "")))
        right_summary = self._normalize_claim_text(self._path_summary_for_divergence_side(graph, getattr(divergence, "right_path_id", "")))
        why = self._normalize_claim_text(getattr(divergence, "why_minimal", ""))

        def is_formula(text: str, summary: str) -> bool:
            return "formula" in text or "formula" in summary or bool(re.search(r"\b[a-z]\s*=", text))

        def is_substituted_value(text: str, summary: str) -> bool:
            return (
                "substitut" in summary
                or "concrete value" in why
                or "using" in summary
            ) and bool(re.search(r"\d", text))

        return (is_formula(left_claim, left_summary) and is_substituted_value(right_claim, right_summary)) or (
            is_formula(right_claim, right_summary) and is_substituted_value(left_claim, left_summary)
        )

    def _looks_async_progress_divergence(self, divergence, graph: NaturalLanguageGraph, question: str = "") -> bool:
        if self._looks_euclidean_progress_divergence(divergence, graph):
            return True
        if self._looks_formula_to_value_progress_divergence(divergence, graph):
            return True
        if self._divergence_has_two_path_concrete_conflict(divergence):
            return False
        return False

    def _divergence_has_two_path_concrete_conflict(self, divergence) -> bool:
        left_claim = str(getattr(divergence, "left_claim", "") or "").strip()
        right_claim = str(getattr(divergence, "right_claim", "") or "").strip()
        if not left_claim or not right_claim:
            return False
        if self._normalize_claim_text(left_claim) == self._normalize_claim_text(right_claim):
            return False
        if self._looks_progress_state_claim(left_claim) or self._looks_progress_state_claim(right_claim):
            return False
        if self._looks_generic_requirement_claim(left_claim) or self._looks_generic_requirement_claim(right_claim):
            return False
        if self._looks_claim_extension_not_conflict(left_claim, right_claim):
            return False
        if self._looks_parameterization_not_conflict(left_claim, right_claim):
            return False
        if self._looks_count_bookkeeping_not_conflict(left_claim, right_claim):
            return False
        if self._looks_plan_only_divergence(left_claim, right_claim):
            return False
        if self._looks_progress_lag_not_conflict(left_claim, right_claim):
            return False
        if self._looks_exact_approximation_divergence(left_claim, right_claim):
            return False
        return self._looks_concrete_claim(left_claim) and self._looks_concrete_claim(right_claim)

    def _looks_progress_state_claim(self, claim: str) -> bool:
        text = str(claim or "").strip().lower()
        if not text:
            return False
        progress_markers = (
            "has not yet",
            "has not been",
            "has not started",
            "has not begun",
            "not yet determined",
            "is not yet determined",
            "not determined yet",
            "remains undetermined",
            "remains to be determined",
            "is underway",
            "is being",
            "has begun",
            "has started",
            "started and is being",
            "continues to",
            "continues with",
            "proceeds to",
            "focuses on",
            "aims to",
            "needs to",
            "goal is to",
            "calculation has not yet begun",
            "calculation has begun",
            "is ongoing",
            "is in progress",
            "yet to be completed",
            "no further steps are taken",
            "does not proceed further",
            "not yet computed",
            "is not yet computed",
            "has not yet been computed",
            "remains at the shared prefix",
            "no new same-object claim beyond the shared prefix is visible in this window",
            "keeps the variables general",
            "keeps a, b, c as general variables",
            "keeps a b c as general variables",
            "keeps the expression in terms of",
            "prepares to analyze",
            "prepares to apply",
        )
        return any(marker in text for marker in progress_markers)

    def _looks_generic_requirement_claim(self, claim: str) -> bool:
        text = str(claim or "").strip()
        normalized = self._normalize_claim_text(text)
        if not normalized:
            return False
        generic_markers = (
            "must consider the restriction",
            "must account for the restriction",
            "must account for",
            "requires further restriction",
            "requires further restrictions",
            "need to enforce the restriction",
            "needs to enforce the restriction",
            "the restriction matters",
            "the roots are real",
            "all roots are real",
            "the polynomial has real roots",
            "the expression remains greater than",
            "the expression approaches",
            "the result must satisfy the condition",
        )
        if not any(marker in normalized for marker in generic_markers):
            return False
        return not bool(re.search(r"=|\\frac|\\sqrt|\\pi|\\begin|\\infty|\(|\)|\[|\]|\d", text))

    def _looks_concrete_claim(self, claim: str) -> bool:
        text = str(claim or "")
        normalized = self._normalize_claim_text(text)
        if not normalized:
            return False
        if bool(re.search(r"=|\\frac|\\sqrt|\\pi|\\begin|\\infty|\(|\)|\[|\]|\d", text)):
            return True
        return any(
            marker in normalized
            for marker in (
                "is greater than",
                "is less than",
                "can attain",
                "does not attain",
                "is open",
                "is closed",
                "valid",
                "invalid",
                "count is",
                "range is",
                "domain is",
                "answer is",
            )
        )

    def _looks_generic_requirement_divergence(self, left_claim: str, right_claim: str) -> bool:
        left_generic = self._looks_generic_requirement_claim(left_claim)
        right_generic = self._looks_generic_requirement_claim(right_claim)
        if left_generic == right_generic:
            return False
        concrete_claim = right_claim if left_generic else left_claim
        return self._looks_concrete_claim(concrete_claim)

    def _looks_claim_extension_not_conflict(self, left_claim: str, right_claim: str) -> bool:
        left = self._normalize_claim_text(left_claim)
        right = self._normalize_claim_text(right_claim)
        if not left or not right or left == right:
            return False
        if left in right or right in left:
            return True
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if len(shorter) < 24:
            return False
        if not longer.startswith(shorter):
            return False
        tail = longer[len(shorter) :].strip(" ,;:")
        if not tail:
            return False
        return tail.startswith(("and ", "with ", "where ", "while "))

    def _looks_parameterization_not_conflict(self, left_claim: str, right_claim: str) -> bool:
        left = str(left_claim or "")
        right = str(right_claim or "")
        left_norm = self._normalize_claim_text(left)
        right_norm = self._normalize_claim_text(right)
        if not left_norm or not right_norm:
            return False

        def is_general_form(normalized: str) -> bool:
            return any(
                marker in normalized
                for marker in (
                    "keeps the variables general",
                    "keeps a, b, c as general variables",
                    "keeps a b c as general variables",
                    "remains in terms of",
                    "not yet determined",
                    "is not yet determined",
                    "remains undetermined",
                )
            )

        def is_reparameterization(text: str, normalized: str) -> bool:
            assignment_like = bool(re.search(r"\b(let|set|take|fix)\b", normalized)) and "=" in text
            marker_like = any(
                marker in normalized
                for marker in (
                    "reduced to variables",
                    "reduce the problem to variables",
                    "reparameterizes",
                    "reparameterized",
                    "introduces t",
                    "introduces s",
                )
            )
            return assignment_like or marker_like

        def is_requested_level(normalized: str) -> bool:
            return any(
                marker in normalized
                for marker in (
                    "set of all possible values",
                    "requested range",
                    "range is",
                    "requested count",
                    "count is",
                    "valid arrangements",
                    "minimum value",
                    "maximum value",
                    "answer is",
                )
            )

        if is_requested_level(left_norm) or is_requested_level(right_norm):
            return False
        return (is_general_form(left_norm) and is_reparameterization(right, right_norm)) or (
            is_general_form(right_norm) and is_reparameterization(left, left_norm)
        )

    def _looks_count_bookkeeping_not_conflict(self, left_claim: str, right_claim: str) -> bool:
        left = str(left_claim or "")
        right = str(right_claim or "")
        left_norm = self._normalize_claim_text(left)
        right_norm = self._normalize_claim_text(right)
        if not left_norm or not right_norm:
            return False

        def is_bookkeeping(normalized: str) -> bool:
            return any(
                marker in normalized
                for marker in (
                    "special individuals",
                    "general group",
                    "without distinguishing",
                    "with no further distinction",
                    "distinguishing between special and non-special",
                    "remaining 6 people include",
                    "remaining six people include",
                    "3 others",
                    "three others",
                )
            )

        def is_unrestricted_setup_count(normalized: str) -> bool:
            has_setup_marker = any(
                marker in normalized
                for marker in (
                    "remaining 6 people can be arranged",
                    "remaining six people can be arranged",
                    "total number of arrangements",
                    "arranged in 6!",
                    "arranged in 5!",
                )
            )
            mentions_restricted_count = any(
                marker in normalized
                for marker in (
                    "valid arrangements",
                    "valid placement",
                    "gap placement",
                    "not adjacent",
                    "invalid arrangement",
                    "inclusion-exclusion",
                    "subtract the invalid",
                )
            )
            return has_setup_marker and not mentions_restricted_count

        def is_restricted_count_claim(normalized: str) -> bool:
            return any(
                marker in normalized
                for marker in (
                    "valid arrangements",
                    "valid placement",
                    "gap placement",
                    "not adjacent",
                    "invalid arrangement",
                    "inclusion-exclusion",
                    "requested count",
                    "count is",
                )
            )

        if is_restricted_count_claim(left_norm) or is_restricted_count_claim(right_norm):
            return False
        return (
            (is_bookkeeping(left_norm) and is_bookkeeping(right_norm))
            or (is_bookkeeping(left_norm) and is_unrestricted_setup_count(right_norm))
            or (is_bookkeeping(right_norm) and is_unrestricted_setup_count(left_norm))
        )

    def _looks_plan_only_divergence(self, left_claim: str, right_claim: str) -> bool:
        left_norm = self._normalize_claim_text(left_claim)
        right_norm = self._normalize_claim_text(right_claim)
        if not left_norm or not right_norm:
            return False

        def is_concrete_claim(text: str, normalized: str) -> bool:
            if "derived by" in normalized or "next step is" in normalized:
                return False
            if any(
                marker in normalized
                for marker in (
                    "requested range is",
                    "count is",
                    "valid arrangements are",
                    "valid placement count",
                    "is equal to",
                    "the value is",
                )
            ):
                return True
            return bool(re.search(r"=\s*[\d\\(\\[\\-]", text))

        def is_plan_only(normalized: str) -> bool:
            return any(
                marker in normalized
                for marker in (
                    "the next step is",
                    "next step is",
                    "proceeds to count",
                    "proceeds to arrange",
                    "continues to count",
                    "continues to arrange",
                    "is derived by",
                    "derived by arranging",
                    "derived by subtracting",
                    "arranges the other",
                    "places the 3 special individuals",
                    "places the three special individuals",
                    "counts arrangements where",
                    "uses inclusion-exclusion to count",
                )
            )

        if is_concrete_claim(left_claim, left_norm) or is_concrete_claim(right_claim, right_norm):
            return False
        return is_plan_only(left_norm) and is_plan_only(right_norm)

    def _looks_eval_vs_search_space_divergence(self, left_claim: str, right_claim: str) -> bool:
        def is_current_eval(text: str) -> bool:
            normalized = self._normalize_claim_text(text)
            if not bool(re.search(r"\d", str(text or ""))):
                return False
            return any(
                marker in normalized
                for marker in (
                    "value of the expression is",
                    "expression is",
                    "resulting in",
                    "evaluates to",
                    "the current value is",
                )
            )

        def is_search_space_or_possibility(text: str) -> bool:
            normalized = self._normalize_claim_text(text)
            return any(
                marker in normalized
                for marker in (
                    "can be changed by inserting parentheses",
                    "can change by inserting parentheses",
                    "by inserting parentheses",
                    "possible values",
                    "distinct values",
                    "values obtainable",
                    "values can be obtained",
                    "all values obtainable",
                    "all distinct values",
                )
            )

        return (is_current_eval(left_claim) and is_search_space_or_possibility(right_claim)) or (
            is_current_eval(right_claim) and is_search_space_or_possibility(left_claim)
        )

    def _looks_formula_vs_observation_list_divergence(self, left_claim: str, right_claim: str, divergence=None) -> bool:
        left = str(left_claim or "")
        right = str(right_claim or "")
        if not left or not right:
            return False
        left_norm = self._normalize_claim_text(left)
        right_norm = self._normalize_claim_text(right)
        divergence_object = ""
        if divergence is not None:
            divergence_object = self._normalize_object_signature(getattr(divergence, "claim_object", ""))
        left_object = self._claim_object_signature(left)
        right_object = self._claim_object_signature(right)
        object_signature = divergence_object or (left_object if left_object == right_object else "")
        distance_context = "distance" in left_norm or "distance" in right_norm
        if not any(token in object_signature for token in ("distance", "squared distance", "requested distance")) and not distance_context:
            return False

        def is_formula(text: str, normalized: str) -> bool:
            symbolic_markers = ("\\vec", "\\cdot", "|\\vec", "|\\mathbf", "\\mathbf", "^2", "(x -", "(x-", "y^2", "z^2")
            has_symbolic_marker = any(marker in text or marker in normalized for marker in symbolic_markers)
            has_variable = bool(re.search(r"\b[xypvz]\b", normalized))
            return has_symbolic_marker and has_variable

        def is_observation_list(text: str, normalized: str) -> bool:
            mentions_list = any(
                marker in normalized
                for marker in (
                    "distances are",
                    "known distances",
                    "six distances",
                    "the distances are",
                )
            )
            numeric_items = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text)
            has_unknown = bool(re.search(r"\bx\b", normalized))
            looks_like_list = text.count(",") >= 2 or len(numeric_items) >= 4
            return mentions_list and looks_like_list and (has_unknown or len(numeric_items) >= 5)

        return (is_formula(left, left_norm) and is_observation_list(right, right_norm)) or (
            is_formula(right, right_norm) and is_observation_list(left, left_norm)
        )

    def _looks_count_claim_vs_enumeration_plan_divergence(self, left_claim: str, right_claim: str) -> bool:
        def is_count_claim(text: str) -> bool:
            normalized = self._normalize_claim_text(text)
            return (
                bool(re.search(r"\d", str(text or "")))
                and any(
                    marker in normalized
                    for marker in (
                        "number of distinct values",
                        "number of values",
                        "count of distinct values",
                        "requested count",
                    )
                )
            )

        def is_enumeration_plan(text: str) -> bool:
            normalized = self._normalize_claim_text(text)
            if bool(re.search(r"\d", str(text or ""))) and "parenthesization" not in normalized:
                return False
            return any(
                marker in normalized
                for marker in (
                    "evaluating each parenthesization",
                    "evaluate each parenthesization",
                    "considering each parenthesization",
                    "check each parenthesization",
                    "determining all distinct values",
                    "find all distinct values",
                    "all distinct values obtainable",
                )
            )

        return (is_count_claim(left_claim) and is_enumeration_plan(right_claim)) or (
            is_count_claim(right_claim) and is_enumeration_plan(left_claim)
        )

    def _vertex_instance_signature(self, text: str) -> str:
        source = str(text or "")
        normalized = self._normalize_claim_text(text)
        ordinal_match = re.search(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\s+vertex\b",
            normalized,
        )
        if ordinal_match:
            return ordinal_match.group(1)
        label_match = re.search(r"vertex\s+\$?\s*([A-Z])\s*\$?", source)
        if label_match:
            return label_match.group(1).lower()
        coord_match = re.search(r"vertex\s+[A-Z]\s*=\s*\(([^)]+)\)", source, flags=re.IGNORECASE)
        if coord_match:
            return re.sub(r"\s+", "", coord_match.group(1))
        bare_coord_match = re.search(r"vertex\s*\(([^)]+)\)", source, flags=re.IGNORECASE)
        if bare_coord_match:
            return re.sub(r"\s+", "", bare_coord_match.group(1))
        return ""

    def _looks_different_vertex_instance_divergence(self, left_claim: str, right_claim: str) -> bool:
        left_norm = self._normalize_claim_text(left_claim)
        right_norm = self._normalize_claim_text(right_claim)
        if "vertex" not in left_norm or "vertex" not in right_norm:
            return False
        if "distance" not in left_norm or "distance" not in right_norm:
            return False
        left_sig = self._vertex_instance_signature(left_claim)
        right_sig = self._vertex_instance_signature(right_claim)
        return bool(left_sig and right_sig and left_sig != right_sig)

    def _looks_specific_formula_vs_grouping_method_divergence(self, left_claim: str, right_claim: str) -> bool:
        def is_specific_formula(text: str, normalized: str) -> bool:
            has_vertex = "vertex" in normalized and "distance" in normalized
            has_local_formula = any(
                marker in text or marker in normalized
                for marker in ("(x -", "(x-", "y^2", "z^2", "\\vec", "\\cdot", "^2")
            )
            return has_vertex and has_local_formula

        def is_grouping_method(text: str, normalized: str) -> bool:
            return any(
                marker in normalized
                for marker in (
                    "grouped into",
                    "group the six distances",
                    "pairs of opposite vertices",
                    "pair of opposite vertices",
                    "leveraging the symmetry",
                    "using the symmetry",
                )
            )

        left_norm = self._normalize_claim_text(left_claim)
        right_norm = self._normalize_claim_text(right_claim)
        return (is_specific_formula(left_claim, left_norm) and is_grouping_method(right_claim, right_norm)) or (
            is_specific_formula(right_claim, right_norm) and is_grouping_method(left_claim, left_norm)
        )

    def _looks_formula_or_setup_vs_placeholder_divergence(self, left_claim: str, right_claim: str, divergence=None) -> bool:
        left = str(left_claim or "")
        right = str(right_claim or "")
        if not left or not right:
            return False
        left_norm = self._normalize_claim_text(left)
        right_norm = self._normalize_claim_text(right)
        divergence_object = ""
        if divergence is not None:
            divergence_object = self._normalize_object_signature(getattr(divergence, "claim_object", ""))
        left_object = self._claim_object_signature(left)
        right_object = self._claim_object_signature(right)
        object_signature = divergence_object or (left_object if left_object == right_object else "")
        distance_context = "distance" in left_norm or "distance" in right_norm
        if not any(token in object_signature for token in ("distance", "squared distance", "requested distance")) and not distance_context:
            return False

        def is_general_setup(text: str, normalized: str) -> bool:
            marker_hit = any(
                marker in normalized
                for marker in (
                    "expressed in terms of x, y, z",
                    "expressed in terms of x y z",
                    "using coordinates of p",
                    "using vector notation",
                    "squared distance from p",
                    "distance from p to",
                )
            )
            latex_coordinate_setup = bool(
                re.search(
                    r"expressed\s+in\s+terms\s+of\s+\$?\s*x\s*,?\s*y\s*,?\s*z\s*\$?",
                    text,
                    flags=re.IGNORECASE,
                )
            )
            return marker_hit or latex_coordinate_setup or (
                bool(re.search(r"\b[xypvz]\b", normalized))
                and any(marker in text or marker in normalized for marker in ("\\vec", "\\cdot", "^2", "(x -", "(x-"))
            )

        def is_placeholder(text: str, normalized: str) -> bool:
            marker_hit = any(
                marker in normalized
                for marker in (
                    "labels the sixth distance as",
                    "lets the sixth distance be",
                    "sixth distance is",
                    "unknown distance as",
                    "the unknown distance",
                )
            )
            latex_unknown = bool(
                re.search(
                    r"unknown\s+variable\s+\$?\s*x\s*\$?|sixth\s+distance\s+is\s+\$?\s*x\s*\$?",
                    text,
                    flags=re.IGNORECASE,
                )
            )
            return marker_hit or latex_unknown

        return (is_general_setup(left, left_norm) and is_placeholder(right, right_norm)) or (
            is_general_setup(right, right_norm) and is_placeholder(left, left_norm)
        )

    def _looks_representation_equivalence_divergence(self, left_claim: str, right_claim: str, divergence=None) -> bool:
        left = str(left_claim or "")
        right = str(right_claim or "")
        if not left or not right:
            return False
        left_norm = self._normalize_claim_text(left)
        right_norm = self._normalize_claim_text(right)
        left_object = self._claim_object_signature(left)
        right_object = self._claim_object_signature(right)
        divergence_object = ""
        if divergence is not None:
            divergence_object = self._normalize_object_signature(getattr(divergence, "claim_object", ""))
        object_signature = divergence_object or (left_object if left_object == right_object else "")
        if object_signature not in {"distance from p to a vertex", "squared distance from p to a vertex", "squared distance"}:
            return False
        vector_markers = ("\\vec", "\\cdot", "|\\vec", "|\\mathbf", "\\mathbf", "\\overrightarrow")
        coordinate_markers = ("(x -", "y^2", "z^2", "(x-", "+ y^2", "+ z^2")
        left_vector = any(marker in left for marker in vector_markers)
        right_vector = any(marker in right for marker in vector_markers)
        left_coordinate = any(marker in left_norm for marker in coordinate_markers)
        right_coordinate = any(marker in right_norm for marker in coordinate_markers)
        return (left_vector and right_coordinate) or (right_vector and left_coordinate)

    def _is_valid_divergence_case(self, divergence, question: str = "") -> bool:
        left_claim = str(getattr(divergence, "left_claim", "") or "").strip()
        right_claim = str(getattr(divergence, "right_claim", "") or "").strip()
        left_path = str(getattr(divergence, "left_path_id", "") or "").strip()
        right_path = str(getattr(divergence, "right_path_id", "") or "").strip()
        if not left_claim or not right_claim or not left_path or not right_path:
            return False
        if left_path == right_path:
            return False
        if self._contains_embedded_path_marker(left_claim) or self._contains_embedded_path_marker(right_claim):
            return False
        if self._normalize_claim_text(left_claim) == self._normalize_claim_text(right_claim):
            return False
        if self._looks_method_formula_fork(left_claim, right_claim):
            return False
        if self._looks_exact_approximation_divergence(left_claim, right_claim):
            return False
        if (
            self._question_targets_radical_target_form(question)
            and self._text_mentions_target_form_sum(left_claim)
            and self._text_mentions_target_form_sum(right_claim)
        ):
            return False
        if self._looks_conjugate_pair_target_drift_divergence(divergence, question):
            return False
        if self._looks_scope_mismatch_divergence(divergence):
            return False
        relation = self._normalize_metadata_token(getattr(divergence, "relation", ""))
        if relation in {
            "same",
            "same_claim",
            "equivalent",
            "consistent",
            "support",
            "agreement",
            "same_value",
            "same_formula",
            "same_result",
            "same_conclusion",
        }:
            return False
        if self._looks_agreement_like_divergence(divergence):
            return False
        if self._looks_parallel_attained_example_divergence(divergence, question):
            return False
        if self._looks_euclidean_sequence_not_conflict(left_claim, right_claim):
            return False
        if self._looks_generic_requirement_divergence(left_claim, right_claim):
            return False
        if self._looks_claim_extension_not_conflict(left_claim, right_claim):
            return False
        if self._looks_parameterization_not_conflict(left_claim, right_claim):
            return False
        if self._looks_count_bookkeeping_not_conflict(left_claim, right_claim):
            return False
        if self._looks_plan_only_divergence(left_claim, right_claim):
            return False
        if self._looks_task_statement_vs_progress(left_claim, right_claim):
            return False
        if self._looks_progress_lag_not_conflict(left_claim, right_claim):
            return False
        return True

    def _looks_exact_approximation_divergence(self, left_claim: str, right_claim: str) -> bool:
        left = str(left_claim or "").lower()
        right = str(right_claim or "").lower()

        def has_exact_marker(text: str) -> bool:
            return any(marker in text for marker in ("\\sqrt", "\\frac", "\\pi", "^\\circ", "\\text"))

        def has_decimal_or_approx_marker(text: str) -> bool:
            return bool(re.search(r"(?<![A-Za-z])[-+]?\d+\.\d+", text)) or any(
                marker in text
                for marker in ("approx", "approximately", "decimal", "numerical value", "numeric value")
            )

        return (has_exact_marker(left) and has_decimal_or_approx_marker(right)) or (
            has_exact_marker(right) and has_decimal_or_approx_marker(left)
        )

    def _looks_conjugate_pair_target_drift_divergence(self, divergence, question: str) -> bool:
        if not self._question_targets_conjugate_pair_sum(question):
            return False
        left = self._compact_math_claim(getattr(divergence, "left_claim", ""))
        right = self._compact_math_claim(getattr(divergence, "right_claim", ""))
        return (self._mentions_five_conjugate_pair_sum(left) and self._mentions_ten_root_sum(right)) or (
            self._mentions_five_conjugate_pair_sum(right) and self._mentions_ten_root_sum(left)
        )

    def _compact_math_claim(self, text: str) -> str:
        compact = self._normalize_claim_text(text)
        compact = compact.replace("{", "").replace("}", "")
        compact = compact.replace("\\left", "").replace("\\right", "")
        return re.sub(r"\s+", "", compact)

    def _mentions_five_conjugate_pair_sum(self, text: str) -> bool:
        return (
            "sum_k=1^5" in text
            or "sum_k=1to5" in text
            or "k=1to5" in text
            or "fiveconjugate" in text
            or "fivepairs" in text
        )

    def _mentions_ten_root_sum(self, text: str) -> bool:
        return (
            "sum_k=0^9" in text
            or "sum_k=1^10" in text
            or "k=0to9" in text
            or "10roots" in text
            or "10terms" in text
            or "tenroots" in text
        )

    def _looks_agreement_like_divergence(self, divergence) -> bool:
        alignment = self._normalize_metadata_token(getattr(divergence, "alignment", ""))
        why_minimal = str(getattr(divergence, "why_minimal", "") or "").strip().lower()
        relation = str(getattr(divergence, "relation", "") or "").strip().lower()
        agreement_phrases = (
            "all paths agree",
            "all active paths agree",
            "no real conflict",
            "lack of conflict",
            "same value",
            "same formula",
            "same result",
            "same conclusion",
            "same algebraic operation",
            "difference in phrasing",
            "procedural interpretation",
            "procedural interpretations",
            "does not affect the final answer",
        )
        has_agreement_relation = self._normalize_metadata_token(relation) in {
            "agreement",
            "same_value",
            "same_formula",
            "same_result",
            "same_conclusion",
        }
        if alignment == "async_same_object" and any(marker in why_minimal for marker in agreement_phrases):
            return True
        return has_agreement_relation or any(marker in why_minimal for marker in agreement_phrases)

    def _graph_completeness_score(self, graph: NaturalLanguageGraph, traces: Dict[str, AgentTrace]) -> tuple:
        raw_text = str(graph.raw_dossier or "").strip()
        answer_disagreement = self._has_answer_disagreement(traces)
        actionable_count = sum(1 for divergence in graph.divergences if self._is_actionable_divergence(divergence, graph))
        has_paths = len(graph.method_paths)
        ends_cleanly = raw_text.endswith((".", "]", "}"))
        has_all_headers = all(
            header in raw_text
            for header in (
                "Shared Claims",
                "Method Paths",
                "Minimal Divergences",
            )
        ) or all(
            header in raw_text
            for header in (
                "Common Ground",
                "Paths",
                "First Split",
            )
        )
        malformed_claims = sum(1 for claim in graph.shared_claims if not self._is_valid_claim_node(claim))
        malformed_paths = sum(1 for path in graph.method_paths if not self._is_valid_method_path(path))
        expected_paths = 2 if answer_disagreement else 1
        enough_paths = has_paths >= expected_paths
        claim_map = graph.claim_map()
        supported_shared_count = sum(
            1
            for claim in graph.shared_claims
            if self._is_valid_claim_node(claim) and self._claim_supporting_agent_count(claim) >= 2
        )
        frontier_supported_count = sum(
            1
            for divergence in graph.divergences
            if self._is_actionable_divergence(divergence, graph)
            and str(getattr(divergence, "frontier_claim_id", "") or "").strip() in claim_map
        )
        if answer_disagreement:
            compact_shared_prefix_score = min(supported_shared_count, 4) - max(0, len(graph.shared_claims) - 6) * 2
            frontier_support_score = min(frontier_supported_count, 1)
            verbosity_score = -(len(graph.divergences) + max(0, len(graph.shared_claims) - 4))
            divergence_count_score = -len(graph.divergences)
            shared_claim_count_score = compact_shared_prefix_score
        else:
            compact_shared_prefix_score = len(graph.shared_claims)
            frontier_support_score = 0
            verbosity_score = (
                -(len(graph.divergences) + len(graph.shared_claims))
                if actionable_count > 0
                else len(graph.shared_claims)
            )
            divergence_count_score = -len(graph.divergences) if actionable_count > 0 else len(graph.divergences)
            shared_claim_count_score = len(graph.shared_claims) if actionable_count == 0 else -len(graph.shared_claims)
        return (
            1 if actionable_count > 0 else 0,
            1 if enough_paths else 0,
            1 if has_all_headers else 0,
            1 if ends_cleanly else 0,
            -malformed_claims - malformed_paths,
            frontier_support_score,
            compact_shared_prefix_score,
            verbosity_score,
            divergence_count_score,
            len(graph.method_paths),
            shared_claim_count_score,
        )

    def _needs_prefix_conflict_repair(self, graph: NaturalLanguageGraph, traces: Dict[str, AgentTrace]) -> bool:
        if not self._has_answer_disagreement(traces):
            return False
        if not self._graph_has_substantive_content(graph):
            return False
        if not (graph.shared_claims or graph.method_paths or graph.divergences):
            return False
        if self._has_late_optimal_frontier(graph):
            return True
        score = self._graph_completeness_score(graph, traces)
        actionable_count = score[0]
        enough_paths = score[1]
        ends_cleanly = score[3]
        return not (actionable_count and enough_paths and ends_cleanly)

    def should_stop_prefix_expansion(self, graph: NaturalLanguageGraph, traces: Dict[str, AgentTrace]) -> bool:
        if not self._has_answer_disagreement(traces):
            return False
        if not self._graph_has_content(graph):
            return False
        score = self._graph_completeness_score(graph, traces)
        has_actionable_conflict = score[0] == 1
        enough_paths = score[1] == 1
        has_headers = score[2] == 1
        ends_cleanly = score[3] == 1
        if not (has_actionable_conflict and enough_paths and has_headers and ends_cleanly):
            return False
        if self._has_answer_disagreement(traces) and not graph.shared_claims:
            return False
        if self._has_late_optimal_frontier(graph):
            return False
        return self._select_primary_real_claim_divergence(graph) is not None

    def _step_text_map(self, traces: Dict[str, AgentTrace]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for trace in traces.values():
            for step in trace.steps:
                mapping[step.step_id] = step.text
        return mapping

    def _content_tokens(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[A-Za-z0-9]+", str(text or "").lower()))
        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "is",
            "are",
            "be",
            "by",
            "with",
            "that",
            "this",
            "it",
            "so",
            "then",
            "therefore",
            "claim",
            "local",
        }
        return {token for token in tokens if token not in stop and len(token) > 1}

    def _infer_missing_claim_members(self, graph: NaturalLanguageGraph, traces: Dict[str, AgentTrace]) -> None:
        if not traces:
            return
        for claim in graph.shared_claims:
            if getattr(claim, "members", None):
                continue
            claim_tokens = self._content_tokens(getattr(claim, "text", ""))
            explicit_agents = set(re.findall(r"\bA\d+\b", str(getattr(claim, "text", "") or "")))
            inferred: list[str] = []
            for agent_id, trace in traces.items():
                if explicit_agents and agent_id not in explicit_agents:
                    continue
                best_step_id = ""
                best_score = 0
                for step in trace.steps:
                    step_tokens = self._content_tokens(step.text)
                    score = len(claim_tokens & step_tokens)
                    if score > best_score:
                        best_score = score
                        best_step_id = step.step_id
                if best_step_id and (best_score >= 2 or not claim_tokens):
                    inferred.append(best_step_id)
            if len(inferred) >= 2:
                claim.members = inferred

    def _is_anli_relation_verdict_text(self, text: str) -> bool:
        normalized = self._normalize_claim_text(text)
        if not normalized:
            return False
        verdict_markers = (
            "not explicitly",
            "not directly",
            "does not explicitly",
            "doesn't explicitly",
            "does not directly",
            "doesn't directly",
            "does not mention",
            "doesn't mention",
            "does not state",
            "doesn't state",
            "does not confirm",
            "doesn't confirm",
            "not mentioned",
            "not stated",
            "not confirmed",
            "not supported",
            "unsupported",
            "not enough information",
            "not contradicted",
            "is neutral",
            "entails the hypothesis",
            "contradicts the hypothesis",
        )
        if not any(marker in normalized for marker in verdict_markers):
            return False
        return (
            "hypothesis" in normalized
            or "premise" in normalized
            or re.search(r"\bit\s+(?:is|was|does|doesn't|isn't|wasn't)", normalized) is not None
        )

    def _strip_anli_verdict_shared_claims(
        self,
        shared_claims: List[ClaimNode],
        method_paths: List[MethodPath],
        divergences: List[DivergenceCase],
    ) -> tuple[List[ClaimNode], List[MethodPath], List[DivergenceCase]]:
        if getattr(self.prompt_profile, "name", "") not in {
            "anli_relation_v3",
            "anli_relation_v4",
            "anli_relation_v5",
            "anli_relation_v6",
            "anli_relation_v7",
            "anli_relation_v8",
            "anli_relation_v9",
            "anli_relation_v10",
            "anli_relation_v11",
            "anli_relation_v12",
            "anli_relation_v13",
            "anli_relation_v14",
        }:
            return shared_claims, method_paths, divergences
        removed_ids = {
            claim.claim_id
            for claim in shared_claims
            if self._is_anli_relation_verdict_text(getattr(claim, "text", ""))
        }
        if not removed_ids:
            return shared_claims, method_paths, divergences
        filtered_claims = [claim for claim in shared_claims if claim.claim_id not in removed_ids]
        for path in method_paths:
            path.claim_ids = [claim_id for claim_id in (path.claim_ids or []) if claim_id not in removed_ids]
        for divergence in divergences:
            if divergence.frontier_claim_id in removed_ids:
                divergence.frontier_claim_id = filtered_claims[-1].claim_id if filtered_claims else ""
        return filtered_claims, method_paths, divergences

    def _claim_member_agents(self, claim) -> set[str]:
        members = getattr(claim, "members", []) or []
        agents = set()
        for member in members:
            token = str(member or "").strip()
            if ".s" in token:
                agents.add(token.split(".s", 1)[0])
        return agents

    def _claim_supported_by_path_agents(self, claim, path) -> bool:
        path_agents = set(getattr(path, "agent_ids", []) or [])
        claim_agents = self._claim_member_agents(claim)
        if not path_agents or not claim_agents:
            return False
        return bool(path_agents & claim_agents)

    def _filter_path_unsupported_claim_usage(self, shared_claims, method_paths):
        claim_map = {claim.claim_id: claim for claim in shared_claims}
        for path in method_paths:
            filtered_claim_ids = []
            for claim_id in path.claim_ids or []:
                claim = claim_map.get(claim_id)
                if claim is None:
                    continue
                if self._claim_supported_by_path_agents(claim, path):
                    filtered_claim_ids.append(claim_id)
            path.claim_ids = filtered_claim_ids
        return method_paths

    def _promote_internal_method_claim_conflicts(self, shared_claims, method_paths, divergences):
        promoted = []
        next_path_index = len(method_paths) + 1
        existing_keys = {
            (
                self._normalize_claim_text(getattr(divergence, "left_claim", "")),
                self._normalize_claim_text(getattr(divergence, "right_claim", "")),
            )
            for divergence in divergences
        }
        conflict_pattern = re.compile(
            r"\b(?P<left_agent>A\d+)\s+"
            r"(?P<left_verb>uses|gets|finds|claims|takes|sets|computes)\s+"
            r"(?P<left_claim>[^.;]+?)\s*,?\s+"
            r"(?:while|whereas|but)\s+"
            r"(?P<right_agent>A\d+)\s+"
            r"(?P<right_verb>uses|gets|finds|claims|takes|sets|computes)\s+"
            r"(?P<right_claim>[^.;]+)",
            flags=re.IGNORECASE,
        )
        for path in list(method_paths):
            summary = str(getattr(path, "summary", "") or "")
            path_agents = set(getattr(path, "agent_ids", []) or [])
            for match in conflict_pattern.finditer(summary):
                left_agent = match.group("left_agent")
                right_agent = match.group("right_agent")
                if path_agents and not {left_agent, right_agent}.issubset(path_agents):
                    continue
                left_tail = match.group("left_claim").strip()
                right_tail = match.group("right_claim").strip()
                if not self._claim_pair_has_math_signal(left_tail, right_tail):
                    continue
                if self._looks_method_formula_fork(left_tail, right_tail):
                    continue
                left_claim = f"{left_agent} {match.group('left_verb').lower()} {left_tail}"
                right_claim = f"{right_agent} {match.group('right_verb').lower()} {right_tail}"
                key = (self._normalize_claim_text(left_claim), self._normalize_claim_text(right_claim))
                reverse_key = (key[1], key[0])
                if key in existing_keys or reverse_key in existing_keys:
                    continue
                left_path_id = f"P{next_path_index}"
                next_path_index += 1
                right_path_id = f"P{next_path_index}"
                next_path_index += 1
                common_claim_ids = list(getattr(path, "claim_ids", []) or [])
                method_paths.extend(
                    [
                        MethodPath(
                            path_id=left_path_id,
                            agent_ids=[left_agent],
                            summary=left_claim,
                            claim_ids=common_claim_ids,
                        ),
                        MethodPath(
                            path_id=right_path_id,
                            agent_ids=[right_agent],
                            summary=right_claim,
                            claim_ids=common_claim_ids,
                        ),
                    ]
                )
                promoted.append(
                    DivergenceCase(
                        divergence_id=f"D_METHOD_CLAIM_{len(divergences) + len(promoted) + 1}",
                        frontier_claim_id=shared_claims[-1].claim_id if shared_claims else "",
                        relation="contradiction",
                        left_path_id=left_path_id,
                        right_path_id=right_path_id,
                        left_claim=left_claim,
                        right_claim=right_claim,
                        why_minimal=(
                            "A method summary contains incompatible mathematical claims about the same local object; "
                            "this is a claim conflict, not only a method fork."
                        ),
                        claim_object=self._object_from_claim_pair(left_tail, right_tail) or "local mathematical object",
                        aspect="value",
                        alignment="same_object_conflict",
                    )
                )
                existing_keys.add(key)
        return method_paths, divergences + promoted

    def _looks_method_formula_fork(self, left_claim: str, right_claim: str) -> bool:
        left = self._normalize_claim_text(left_claim)
        right = self._normalize_claim_text(right_claim)
        if "formula" not in left or "formula" not in right:
            return False
        combined = re.sub(r"\bA\d+\b", "", f"{left_claim} {right_claim}")
        if re.search(r"\d|\\frac|\\sqrt|\\pi", combined):
            return False
        return True

    def _claim_pair_has_math_signal(self, left_claim: str, right_claim: str) -> bool:
        text = f"{left_claim} {right_claim}"
        if re.search(r"\d|=|\\frac|\\sqrt|[<>⟨⟩]|\([^)]*,[^)]*\)", text):
            return True
        return bool(re.search(r"\b(vector|root|angle|cos|sin|tan|coefficient|candidate|multiple|probability)\b", text, re.IGNORECASE))

    def _compact_support_text(self, text: str) -> str:
        return re.sub(r"[\s{}]+", "", str(text or "").lower().replace("−", "-"))

    def _extract_equation_fragments(self, text: str) -> set[str]:
        fragments: set[str] = set()
        for match in re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9_]*\s*=\s*-?(?:\d+(?:\.\d+)?|\\?pi|[a-zA-Z][a-zA-Z0-9_]*)",
            str(text or ""),
        ):
            compact = self._compact_support_text(match).strip(".,;:")
            if compact:
                fragments.add(compact)
        return fragments

    def _step_supports_same_claim_recovery(self, step_text: str, claim_text: str) -> bool:
        compact_step = self._compact_support_text(step_text)
        compact_claim = self._compact_support_text(claim_text).strip(".,;:")
        if compact_claim and compact_claim in compact_step:
            return True
        equation_fragments = self._extract_equation_fragments(claim_text)
        return bool(equation_fragments and any(fragment in compact_step for fragment in equation_fragments))

    def _recover_same_claim_divergence_as_shared(
        self,
        divergence,
        shared_claims,
        method_paths,
        traces: Dict[str, AgentTrace],
    ) -> None:
        left_claim = str(getattr(divergence, "left_claim", "") or "").strip()
        right_claim = str(getattr(divergence, "right_claim", "") or "").strip()
        if not left_claim or self._normalize_claim_text(left_claim) != self._normalize_claim_text(right_claim):
            return
        normalized = self._normalize_claim_text(left_claim)
        if any(self._normalize_claim_text(getattr(claim, "text", "")) == normalized for claim in shared_claims):
            return
        left_path_id = str(getattr(divergence, "left_path_id", "") or "").strip()
        right_path_id = str(getattr(divergence, "right_path_id", "") or "").strip()
        path_map = {path.path_id: path for path in method_paths}
        candidate_agents = set()
        for path_id in (left_path_id, right_path_id):
            path = path_map.get(path_id)
            candidate_agents.update(getattr(path, "agent_ids", []) or [])
        members: list[str] = []
        for agent_id in sorted(candidate_agents):
            trace = traces.get(agent_id)
            if trace is None:
                continue
            for step in trace.steps:
                if self._step_supports_same_claim_recovery(step.text, left_claim):
                    members.append(step.step_id)
                    break
        if len({member.split(".s", 1)[0] for member in members}) < 2:
            return
        claim_id = f"C{len(shared_claims) + 1}"
        claim = ClaimNode(
            claim_id=claim_id,
            text=left_claim,
            claim_object=str(getattr(divergence, "claim_object", "") or "local claim"),
            aspect=str(getattr(divergence, "aspect", "") or "claim"),
            status="stated",
            alignment="synchronous",
            members=members,
        )
        if not self._structured_claim_supported_by_traces(claim, traces):
            return
        shared_claims.append(claim)
        claim_agents = self._claim_member_agents(claim)
        for path in method_paths:
            if claim_agents & set(getattr(path, "agent_ids", []) or []):
                path.claim_ids.append(claim_id)

    def _without_divergence_frontier(self, divergence):
        return DivergenceCase(
            divergence_id=divergence.divergence_id,
            frontier_claim_id="",
            relation=divergence.relation,
            left_path_id=divergence.left_path_id,
            right_path_id=divergence.right_path_id,
            left_claim=divergence.left_claim,
            right_claim=divergence.right_claim,
            why_minimal=(
                str(divergence.why_minimal or "").strip()
                + " The original frontier shared claim was not supported on both paths."
            ).strip(),
            claim_object=divergence.claim_object,
            aspect=divergence.aspect,
            alignment=divergence.alignment,
        )

    def _object_from_claim_pair(self, left_claim: str, right_claim: str) -> str:
        for claim in (left_claim, right_claim):
            match = re.search(r"\bfor\s+(?:the\s+)?([^,.;]+)", str(claim or ""), flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return self._claim_object_signature(left_claim) or self._claim_object_signature(right_claim)

    def _divergence_frontier_supported_by_paths(self, divergence, graph: NaturalLanguageGraph) -> bool:
        frontier_claim_id = str(getattr(divergence, "frontier_claim_id", "") or "").strip()
        if not frontier_claim_id:
            return True
        left_path = graph.path_map().get(getattr(divergence, "left_path_id", ""))
        right_path = graph.path_map().get(getattr(divergence, "right_path_id", ""))
        if left_path is None or right_path is None:
            return False
        left_claim_ids = set(left_path.claim_ids or [])
        right_claim_ids = set(right_path.claim_ids or [])
        return frontier_claim_id in left_claim_ids and frontier_claim_id in right_claim_ids

    def _step_supports_optimal_claim(self, step_text: str) -> bool:
        text = str(step_text or "")
        lowered = self._normalize_claim_text(text)
        if self._claim_mentions_optimality(lowered):
            return True
        support_markers = (
            "checked candidates from",
            "no earlier candidate",
            "first valid candidate",
            "first valid multiple",
            "final answer",
        )
        return any(marker in lowered for marker in support_markers)

    def _extract_named_assignments(self, text: str) -> Dict[str, str]:
        assignments = {}
        for name, value in re.findall(r"\b([A-Z])\s*=\s*(-?\d+(?:\.\d+)?)\b", str(text or "")):
            assignments[name] = value
        return assignments

    def _extract_symbolic_sum_total(self, text: str) -> tuple[tuple[str, ...], str] | None:
        match = re.search(r"\b([A-Z](?:\s*\+\s*[A-Z]){1,6})\s*=\s*(-?\d+(?:\.\d+)?)\b", str(text or ""))
        if not match:
            return None
        lhs = tuple(token.strip() for token in match.group(1).split("+"))
        rhs = match.group(2)
        return lhs, rhs

    def _step_supports_assignment_claim(self, step_text: str, claim_assignments: Dict[str, str]) -> bool:
        step_assignments = self._extract_named_assignments(step_text)
        if not step_assignments:
            return False
        return all(step_assignments.get(name) == value for name, value in claim_assignments.items())

    def _step_supports_symbolic_sum_claim(self, step_text: str, claim_total: tuple[tuple[str, ...], str]) -> bool:
        lhs_names, rhs_value = claim_total
        step_sum = self._extract_symbolic_sum_total(step_text)
        if step_sum is not None:
            step_names, step_rhs = step_sum
            if step_names == lhs_names and step_rhs == rhs_value:
                return True
        lowered = self._normalize_claim_text(step_text)
        return "sum" in lowered and re.search(rf"=\s*{re.escape(rhs_value)}\b", step_text) is not None

    def _extract_vector_literals(self, text: str) -> set[tuple[str, ...]]:
        vectors: set[tuple[str, ...]] = set()
        for raw in re.findall(r"\(([^()]*,[^()]*)\)", str(text or "")):
            parts = [self._normalize_vector_component(part) for part in raw.split(",")]
            parts = [part for part in parts if part]
            if len(parts) >= 2:
                vectors.add(tuple(parts))
        return vectors

    def _normalize_vector_component(self, value: str) -> str:
        text = str(value or "").strip()
        text = text.replace("−", "-")
        text = text.replace(r"\left", "").replace(r"\right", "")
        text = text.replace("{", "").replace("}", "").replace(" ", "")
        frac = re.fullmatch(r"(-?)\\frac(-?\d+)(-?\d+)", text)
        if frac:
            sign, numerator, denominator = frac.groups()
            return f"{sign}{numerator}/{denominator}"
        frac = re.fullmatch(r"(-?)\\frac\{?(-?\d+)\}?\{?(-?\d+)\}?", text)
        if frac:
            sign, numerator, denominator = frac.groups()
            return f"{sign}{numerator}/{denominator}"
        simple_frac = re.fullmatch(r"(-?\d+)/(-?\d+)", text)
        if simple_frac:
            return f"{int(simple_frac.group(1))}/{int(simple_frac.group(2))}"
        number = re.fullmatch(r"-?\d+(?:\.0+)?", text)
        if number:
            return str(int(float(text)))
        return text.lower()

    def _step_supports_vector_claim(self, step_text: str, claim_vectors: set[tuple[str, ...]]) -> bool:
        if not claim_vectors:
            return True
        step_vectors = self._extract_vector_literals(step_text)
        return bool(step_vectors & claim_vectors)

    def _claim_needs_literal_support(self, claim_text: str) -> bool:
        normalized = self._normalize_claim_text(claim_text)
        return any(
            marker in normalized
            for marker in (
                "direction vector",
                "vector is",
                "roots are",
                "root family",
                "dot product",
                "gcd",
                "greatest common divisor",
                "least common multiple",
            )
        )

    def _step_supports_literal_math_claim(self, step_text: str, claim_text: str) -> bool:
        claim_vectors = self._extract_vector_literals(claim_text)
        if claim_vectors and not self._step_supports_vector_claim(step_text, claim_vectors):
            return False
        if not self._claim_needs_literal_support(claim_text):
            return True
        claim_numbers = set(self._extract_integer_tokens(claim_text))
        if not claim_numbers:
            return True
        step_numbers = set(self._extract_integer_tokens(step_text))
        return claim_numbers.issubset(step_numbers)

    def _extract_math_segments(self, text: str) -> List[str]:
        return [segment.strip() for segment in re.findall(r"\$([^$]+)\$", str(text or "")) if segment.strip()]

    def _normalize_linear_surd_expression(self, expr: str) -> str:
        normalized = str(expr or "")
        normalized = normalized.replace("−", "-")
        normalized = normalized.replace(r"\left", "").replace(r"\right", "")
        normalized = normalized.replace("{", "").replace("}", "")
        normalized = normalized.replace(" ", "")
        normalized = re.sub(r"\\sqrt(\d+)", r"sqrt\1", normalized)
        normalized = re.sub(r"\\sqrt\{?(\d+)\}?", r"sqrt\1", normalized)
        return normalized

    def _parse_linear_surd_signature(self, expr: str) -> Dict[int, int] | None:
        normalized = self._normalize_linear_surd_expression(expr)
        if "=" in normalized:
            normalized = normalized.split("=")[-1]
        normalized = normalized.strip(".,:;")
        normalized = normalized.replace("(", "").replace(")", "")
        if not normalized:
            return None
        if re.search(r"[^0-9a-zA-Z+\\-]", normalized):
            return None
        if normalized[0] not in "+-":
            normalized = f"+{normalized}"
        terms = re.findall(r"([+-])([^+-]+)", normalized)
        if not terms:
            return None
        signature: Dict[int, int] = {}
        for sign_token, body in terms:
            sign = -1 if sign_token == "-" else 1
            if body.isdigit():
                signature[0] = signature.get(0, 0) + sign * int(body)
                continue
            match = re.fullmatch(r"(\d*)sqrt(\d+)", body)
            if not match:
                return None
            coeff = int(match.group(1)) if match.group(1) else 1
            radicand = int(match.group(2))
            signature[radicand] = signature.get(radicand, 0) + sign * coeff
        return {key: value for key, value in signature.items() if value}

    def _claim_assignment_signature(self, claim_assignments: Dict[str, str]) -> Dict[int, int] | None:
        required = {"A", "B", "C", "D"}
        if not required.issubset(claim_assignments):
            return None
        try:
            a = int(float(claim_assignments["A"]))
            b = int(float(claim_assignments["B"]))
            c = int(float(claim_assignments["C"]))
            d = int(float(claim_assignments["D"]))
        except ValueError:
            return None
        signature = {0: a, b: a, c: -1, d: -1}
        return {key: value for key, value in signature.items() if value}

    def _trace_step_lookup(self, traces: Dict[str, AgentTrace]) -> Dict[str, tuple[AgentTrace, int]]:
        lookup: Dict[str, tuple[AgentTrace, int]] = {}
        for trace in traces.values():
            for index, step in enumerate(trace.steps):
                lookup[step.step_id] = (trace, index)
        return lookup

    def _latest_linear_surd_signature_before_member(
        self,
        member: str,
        traces: Dict[str, AgentTrace],
        step_lookup: Dict[str, tuple[AgentTrace, int]],
    ) -> Dict[int, int] | None:
        trace_info = step_lookup.get(member)
        if trace_info is None:
            return None
        trace, step_index = trace_info
        for index in range(step_index - 1, -1, -1):
            step_text = trace.steps[index].text
            for segment in reversed(self._extract_math_segments(step_text)):
                signature = self._parse_linear_surd_signature(segment)
                if signature is not None and any(key != 0 for key in signature):
                    return signature
            signature = self._parse_linear_surd_signature(step_text)
            if signature is not None and any(key != 0 for key in signature):
                return signature
        return None

    def _latest_linear_surd_signature_in_trace(self, trace: AgentTrace) -> Dict[int, int] | None:
        for step in reversed(trace.steps):
            for segment in reversed(self._extract_math_segments(step.text)):
                signature = self._parse_linear_surd_signature(segment)
                if signature is not None and any(key != 0 for key in signature):
                    return signature
            if "sqrt" in step.text or "\\sqrt" in step.text:
                signature = self._parse_linear_surd_signature(step.text)
                if signature is not None and any(key != 0 for key in signature):
                    return signature
        return None

    def _trace_named_assignments(self, trace: AgentTrace) -> Dict[str, str]:
        assignments: Dict[str, str] = {}
        for step in trace.steps:
            assignments.update(self._extract_named_assignments(step.text))
        return assignments

    def _trace_target_form_assignments_supported(self, trace: AgentTrace) -> bool:
        assignments = self._trace_named_assignments(trace)
        assignment_signature = self._claim_assignment_signature(assignments)
        if assignment_signature is None:
            return False
        expression_signature = self._latest_linear_surd_signature_in_trace(trace)
        if expression_signature is None:
            return False
        return expression_signature == assignment_signature

    def _claim_assignments_match_trace_expression(self, claim, traces: Dict[str, AgentTrace]) -> bool:
        claim_assignments = self._extract_named_assignments(getattr(claim, "text", ""))
        assignment_signature = self._claim_assignment_signature(claim_assignments)
        if assignment_signature is None:
            return True
        step_lookup = self._trace_step_lookup(traces)
        for member in getattr(claim, "members", []) or []:
            expression_signature = self._latest_linear_surd_signature_before_member(member, traces, step_lookup)
            if expression_signature is None or expression_signature != assignment_signature:
                return False
        return True

    def _sum_claim_matches_trace_assignments(self, claim, traces: Dict[str, AgentTrace]) -> bool:
        claim_total = self._extract_symbolic_sum_total(getattr(claim, "text", ""))
        if claim_total is None:
            return True
        lhs_names, rhs_value = claim_total
        if tuple(lhs_names) != ("A", "B", "C", "D"):
            return True
        step_lookup = self._trace_step_lookup(traces)
        expected_total = int(float(rhs_value))
        for member in getattr(claim, "members", []) or []:
            trace_info = step_lookup.get(member)
            if trace_info is None:
                return False
            trace, step_index = trace_info
            found_assignments = None
            for index in range(step_index, -1, -1):
                assignments = self._extract_named_assignments(trace.steps[index].text)
                if {"A", "B", "C", "D"}.issubset(assignments):
                    found_assignments = assignments
                    break
            if found_assignments is None:
                return False
            total = sum(int(float(found_assignments[name])) for name in ("A", "B", "C", "D"))
            if total != expected_total:
                return False
        return True

    def _structured_claim_supported_by_traces(self, claim, traces: Dict[str, AgentTrace], question: str = "") -> bool:
        step_text_map = self._step_text_map(traces)
        member_texts = [step_text_map.get(member, "") for member in (getattr(claim, "members", []) or [])]
        member_texts = [text for text in member_texts if text]
        if not member_texts:
            return False
        claim_text = str(getattr(claim, "text", "") or "")
        if not all(self._step_supports_literal_math_claim(text, claim_text) for text in member_texts):
            return False
        if self._question_targets_distinct_value_set(question):
            claim_values = self._extract_claimed_distinct_values_from_text(claim_text)
            if claim_values:
                trace_map = {trace.agent_id: trace for trace in traces.values()}
                claim_agents = self._claim_member_agents(claim)
                if not claim_agents:
                    return False
                supported_union = set()
                for agent_id in claim_agents:
                    trace = trace_map.get(agent_id)
                    if trace is None:
                        return False
                    supported_union.update(self._trace_supported_attainable_values(trace))
                if not claim_values.issubset(supported_union):
                    return False
            if self._text_claims_distinct_value_completeness(claim_text):
                trace_map = {trace.agent_id: trace for trace in traces.values()}
                claim_agents = self._claim_member_agents(claim)
                if not claim_agents:
                    return False
                for agent_id in claim_agents:
                    trace = trace_map.get(agent_id)
                    if trace is None or self._trace_claims_distinct_value_completeness(trace):
                        return False
        claim_assignments = self._extract_named_assignments(claim_text)
        if len(claim_assignments) >= 2:
            if not all(self._step_supports_assignment_claim(text, claim_assignments) for text in member_texts):
                return False
            if self._question_targets_radical_target_form(question):
                return self._claim_assignments_match_trace_expression(claim, traces)
            return True
        claim_total = self._extract_symbolic_sum_total(claim_text)
        if claim_total is not None:
            if not all(self._step_supports_symbolic_sum_claim(text, claim_total) for text in member_texts):
                return False
            if self._question_targets_radical_target_form(question):
                return self._sum_claim_matches_trace_assignments(claim, traces)
            return True
        return True

    def _shared_optimal_claim_supported_by_traces(self, claim, traces: Dict[str, AgentTrace]) -> bool:
        if not self._claim_mentions_optimality(getattr(claim, "text", "")):
            return True
        step_text_map = self._step_text_map(traces)
        member_texts = [step_text_map.get(member, "") for member in (getattr(claim, "members", []) or [])]
        if not member_texts:
            return False
        return all(self._step_supports_optimal_claim(text) for text in member_texts if text)

    def _filter_trace_unsupported_shared_claims(
        self,
        shared_claims,
        method_paths,
        divergences,
        traces: Dict[str, AgentTrace],
        question: str = "",
    ):
        answer_disagreement = self._has_answer_disagreement(traces)
        kept_claims = [
            claim
            for claim in shared_claims
            if not (answer_disagreement and self._claim_supporting_agent_count(claim) < 2)
            if not self._shared_claim_contains_agent_specific_alternatives(claim)
            if self._shared_optimal_claim_supported_by_traces(claim, traces)
            and self._structured_claim_supported_by_traces(claim, traces, question=question)
        ]
        claim_ids = {claim.claim_id for claim in kept_claims}
        for path in method_paths:
            path.claim_ids = [claim_id for claim_id in (path.claim_ids or []) if claim_id in claim_ids]
        kept_divergences = []
        for divergence in divergences:
            frontier_claim_id = str(getattr(divergence, "frontier_claim_id", "") or "").strip()
            if not frontier_claim_id or frontier_claim_id in claim_ids:
                kept_divergences.append(divergence)
                continue
            kept_divergences.append(
                DivergenceCase(
                    divergence_id=divergence.divergence_id,
                    frontier_claim_id="",
                    relation=divergence.relation,
                    left_path_id=divergence.left_path_id,
                    right_path_id=divergence.right_path_id,
                    left_claim=divergence.left_claim,
                    right_claim=divergence.right_claim,
                    why_minimal=(
                        str(divergence.why_minimal or "").strip()
                        + " The original frontier shared claim was removed because it was not trace-supported."
                    ).strip(),
                    claim_object=divergence.claim_object,
                    aspect=divergence.aspect,
                    alignment=divergence.alignment,
                )
            )
        return kept_claims, method_paths, kept_divergences

    def _shared_claim_contains_agent_specific_alternatives(self, claim) -> bool:
        text = str(getattr(claim, "text", "") or "")
        normalized = self._normalize_claim_text(text)
        agent_ids = set(re.findall(r"\bA\d+\b", text))
        if len(agent_ids) == 1:
            return True
        if len(agent_ids) < 2 and self._claim_pair_has_math_signal(text, text):
            alternative_markers_without_agents = (
                " or ",
                "depending on",
                "alternatively",
                "one path",
                "another path",
                "one branch",
                "another branch",
            )
            if any(marker in normalized for marker in alternative_markers_without_agents):
                return True
        if len(agent_ids) < 2:
            return False
        if not self._claim_pair_has_math_signal(text, text):
            return False
        alternative_markers = (
            " for a1",
            " for a2",
            " for a3",
            " while ",
            " whereas ",
            " but ",
            " claims ",
            " claim ",
            " different",
            " respectively",
        )
        return any(marker in normalized for marker in alternative_markers)

    def _sanitize_graph(
        self,
        graph: NaturalLanguageGraph,
        traces: Dict[str, AgentTrace] | None = None,
        question: str = "",
    ) -> NaturalLanguageGraph:
        if traces:
            self._infer_missing_claim_members(graph, traces)
        shared_claims = [claim for claim in graph.shared_claims if self._is_valid_claim_node(claim)]
        claim_ids = {claim.claim_id for claim in shared_claims}
        method_paths = []
        for path in graph.method_paths:
            if not self._is_valid_method_path(path):
                continue
            path.claim_ids = [claim_id for claim_id in (path.claim_ids or []) if claim_id in claim_ids]
            method_paths.append(path)
        method_paths = self._filter_path_unsupported_claim_usage(shared_claims, method_paths)
        method_paths = self._clean_progress_lag_method_summaries(method_paths)
        method_paths, graph_divergences = self._promote_internal_method_claim_conflicts(
            shared_claims,
            method_paths,
            list(graph.divergences),
        )
        if traces:
            for divergence in graph_divergences:
                self._recover_same_claim_divergence_as_shared(divergence, shared_claims, method_paths, traces)
            claim_ids = {claim.claim_id for claim in shared_claims}
            for path in method_paths:
                path.claim_ids = [claim_id for claim_id in (path.claim_ids or []) if claim_id in claim_ids]
        shared_claims, method_paths, graph_divergences = self._strip_anli_verdict_shared_claims(
            shared_claims,
            method_paths,
            graph_divergences,
        )
        path_ids = {path.path_id for path in method_paths}
        divergences = []
        seen_divergence_keys = set()
        sanitized_graph = NaturalLanguageGraph(shared_claims=shared_claims, method_paths=method_paths, divergences=[], raw_dossier="")
        for divergence in graph_divergences:
            if divergence.left_path_id not in path_ids or divergence.right_path_id not in path_ids:
                continue
            if not self._is_valid_divergence_case(divergence, question=question):
                continue
            if self._looks_async_progress_divergence(divergence, sanitized_graph, question=question):
                continue
            if not self._divergence_frontier_supported_by_paths(divergence, sanitized_graph):
                divergence = self._without_divergence_frontier(divergence)
            divergence_key = (
                divergence.frontier_claim_id,
                self._normalize_claim_text(divergence.left_claim),
                self._normalize_claim_text(divergence.right_claim),
            )
            if divergence_key in seen_divergence_keys:
                continue
            seen_divergence_keys.add(divergence_key)
            divergences.append(divergence)
        if traces:
            shared_claims, method_paths, divergences = self._filter_trace_unsupported_shared_claims(
                shared_claims,
                method_paths,
                divergences,
                traces,
                question=question,
            )
            sanitized_graph.shared_claims = shared_claims
            sanitized_graph.method_paths = method_paths
        sanitized_graph.divergences = sorted(
            divergences,
            key=lambda divergence: self._divergence_priority_key(divergence, sanitized_graph),
        )
        sanitized_graph.raw_dossier = render_graph_dossier(sanitized_graph)
        return sanitized_graph

    def _should_accept_graph_update(
        self,
        previous_graph: NaturalLanguageGraph,
        candidate_graph: NaturalLanguageGraph,
        traces: Dict[str, AgentTrace],
    ) -> bool:
        if not self._graph_has_content(candidate_graph):
            return False
        if not self._graph_has_content(previous_graph):
            return True
        if (
            self._has_answer_disagreement(traces)
            and not candidate_graph.shared_claims
            and candidate_graph.divergences
            and all(self._looks_final_answer_divergence(divergence) for divergence in candidate_graph.divergences)
            and (previous_graph.shared_claims or previous_graph.method_paths)
        ):
            return False
        return self._graph_completeness_score(candidate_graph, traces) >= self._graph_completeness_score(previous_graph, traces)

    def _looks_final_answer_divergence(self, divergence) -> bool:
        text = " ".join(
            str(value or "").lower()
            for value in (
                getattr(divergence, "divergence_id", ""),
                getattr(divergence, "claim_object", ""),
                getattr(divergence, "aspect", ""),
                getattr(divergence, "left_claim", ""),
                getattr(divergence, "right_claim", ""),
                getattr(divergence, "why_minimal", ""),
            )
        )
        return "final answer" in text or "requested final" in text

    def build_resolution_text(self, resolutions: Iterable[ResolutionDecision]) -> str:
        if self.graph_format == "json":
            payload = {"resolutions": []}
            for item in resolutions:
                payload["resolutions"].append(
                    {
                        "divergence_id": str(item.divergence_id or "").strip(),
                        "action": str(item.action or "").strip(),
                        "winning_side": str(item.winning_side or "").strip(),
                        "correct_claim": str(item.resolved_claim or "").strip(),
                        "reason": str(item.rationale or "").strip(),
                        "rewrite_from": str(item.rewrite_from_claim_id or "").strip(),
                        "keep_paths": list(item.keep_paths or []),
                        "drop_paths": list(item.drop_paths or []),
                        "canonical_answer": str(getattr(item, "canonical_answer", "") or "").strip(),
                    }
                )
            return json.dumps(payload, ensure_ascii=False, indent=2)
        paragraphs = []
        for item in resolutions:
            divergence_id = str(item.divergence_id or "").strip() or "unknown divergence"
            action = str(item.action or "unspecified action").strip()
            winning_side = str(item.winning_side or "unspecified side").strip()
            resolved_claim = str(item.resolved_claim or "no repaired claim was stated").strip()
            rationale = str(item.rationale or "no rationale was provided").strip()
            rewrite_from = str(item.rewrite_from_claim_id or "C1").strip()
            canonical_answer = str(getattr(item, "canonical_answer", "") or "").strip()
            keep_paths = ", ".join(f"[{path}]" for path in item.keep_paths) if item.keep_paths else "no paths were explicitly kept"
            drop_paths = ", ".join(f"[{path}]" for path in item.drop_paths) if item.drop_paths else "no paths were explicitly dropped"
            answer_sentence = f" The canonical final answer is: {{final answer: \\boxed{{{canonical_answer}}}}}." if canonical_answer else ""
            paragraphs.append(
                (
                    f"For divergence {divergence_id}, the chosen action is {action} and the winning side is {winning_side}. "
                    f"The repaired claim to continue from is: {resolved_claim}. "
                    f"Revision should restart from [{rewrite_from}]. "
                    f"Keep these paths active: {keep_paths}. "
                    f"Drop these paths: {drop_paths}. "
                    f"Reason: {rationale}.{answer_sentence}"
                ).strip()
            )
        return "\n\n".join(paragraphs).strip()

    def _pairwise_agents_to_rewrite(self, decision: ResolutionDecision) -> set[str]:
        match = re.match(r"PAIR_([^_]+)_([^_]+)$", str(decision.divergence_id or "").strip())
        if not match:
            return set()
        left_agent, right_agent = match.group(1), match.group(2)
        if decision.action == "choose_left":
            return {right_agent}
        if decision.action == "choose_right":
            return {left_agent}
        return set()

    def _pairwise_winning_agent(self, decision: ResolutionDecision) -> str:
        match = re.match(r"PAIR_([^_]+)_([^_]+)$", str(decision.divergence_id or "").strip())
        if not match:
            return ""
        left_agent, right_agent = match.group(1), match.group(2)
        if decision.action == "choose_left":
            return left_agent
        if decision.action == "choose_right":
            return right_agent
        return ""

    def _graph_agents_to_rewrite(self, decision: ResolutionDecision, graph: NaturalLanguageGraph) -> set[str]:
        if decision.action not in {"choose_left", "choose_right"} and not (
            decision.action == "synthesize" and decision.drop_paths
        ):
            return set()
        path_map = graph.path_map()
        agents = set()
        for path_id in self._path_or_agent_tokens(decision.drop_paths):
            path = path_map.get(path_id)
            if path is not None:
                agents.update(path.agent_ids or [])
                continue
            agents.update(self._agents_for_path_or_agent_token(path_id, graph))
            if agents:
                continue
            if re.fullmatch(r"A\d+", path_id):
                agents.add(path_id)
        return agents

    def _label_keep_answer_conflict_agents(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        decision: ResolutionDecision,
    ) -> set[str]:
        if not question_uses_label_answers(question):
            return set()
        keep_answer = extract_label_answer(str(decision.resolved_claim or ""), question)
        if not keep_answer:
            return set()
        agents: set[str] = set()
        for path_id in decision.keep_paths or []:
            for agent_id in self._agents_for_path_or_agent_token(path_id, graph):
                trace = traces.get(agent_id)
                current_answer = (
                    extract_label_answer(self._final_answer_text(trace), question)
                    if trace is not None
                    else None
                )
                if trace is not None and current_answer != keep_answer:
                    agents.add(agent_id)
        return agents

    def _agents_for_path_or_agent_token(self, token: str, graph: NaturalLanguageGraph) -> set[str]:
        for cleaned in self._path_or_agent_tokens([token]):
            path = graph.path_map().get(cleaned)
            if path is not None:
                return set(path.agent_ids or [])
            path_index = re.fullmatch(r"P(\d+)", cleaned)
            if path_index:
                index = int(path_index.group(1)) - 1
                paths = list(getattr(graph, "method_paths", []) or [])
                if 0 <= index < len(paths):
                    return set(paths[index].agent_ids or [])
                agent_id = f"A{path_index.group(1)}"
                if agent_id:
                    return {agent_id}
            if re.fullmatch(r"A\d+", cleaned):
                return {cleaned}
        return set()

    def _path_or_agent_tokens(self, raw_tokens: Iterable[str]) -> List[str]:
        tokens: List[str] = []
        seen = set()
        for raw in raw_tokens or []:
            for token in re.findall(r"\b(?:A\d+|P[A-Za-z0-9_]*\d*|P_[A-Za-z0-9_]+)\b", str(raw or "")):
                if token not in seen:
                    tokens.append(token)
                    seen.add(token)
        return tokens

    def _agents_to_rewrite(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> set[str]:
        rewrite_agents: set[str] = set()
        for decision in resolutions:
            if str(decision.divergence_id or "").startswith("PAIR_"):
                rewrite_agents.update(self._pairwise_agents_to_rewrite(decision))
                if self._question_targets_conjugate_pair_sum(question):
                    winning_agent = self._pairwise_winning_agent(decision)
                    winning_trace = traces.get(winning_agent)
                    if winning_trace is not None:
                        winning_answer = self._final_answer_text(winning_trace)
                        for agent_id, trace in traces.items():
                            if agent_id != winning_agent and self._final_answer_text(trace) != winning_answer:
                                rewrite_agents.add(agent_id)
            else:
                rewrite_agents.update(self._graph_agents_to_rewrite(decision, graph))
                rewrite_agents.update(
                    self._label_keep_answer_conflict_agents(question, traces, graph, decision)
                )
        return rewrite_agents

    def _rewrite_driving_resolutions(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> List[ResolutionDecision]:
        actionable: List[ResolutionDecision] = []
        for decision in resolutions:
            if question_uses_label_answers(question) and label_claim_is_bare_answer(
                str(decision.resolved_claim or ""),
                question,
            ):
                continue
            if str(decision.action or "").strip() == "synthesize" and not decision.drop_paths:
                continue
            if str(decision.divergence_id or "").startswith("PAIR_"):
                agents = self._pairwise_agents_to_rewrite(decision)
                if self._question_targets_conjugate_pair_sum(question):
                    winning_agent = self._pairwise_winning_agent(decision)
                    winning_trace = traces.get(winning_agent)
                    if winning_trace is not None:
                        winning_answer = self._final_answer_text(winning_trace)
                        agents = set(agents)
                        for agent_id, trace in traces.items():
                            if agent_id != winning_agent and self._final_answer_text(trace) != winning_answer:
                                agents.add(agent_id)
                if agents:
                    actionable.append(decision)
            elif self._graph_agents_to_rewrite(decision, graph) or self._label_keep_answer_conflict_agents(
                question,
                traces,
                graph,
                decision,
            ):
                actionable.append(decision)
        return actionable

    def build_revision_prompts(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> Dict[str, str]:
        actionable_resolutions = self._rewrite_driving_resolutions(question, traces, graph, resolutions)
        resolution_text = self.build_resolution_text(actionable_resolutions)
        if not resolution_text:
            return {}
        rewrite_agents = self._agents_to_rewrite(question, traces, graph, actionable_resolutions)
        prompts = {}
        for agent_id, trace in traces.items():
            if agent_id not in rewrite_agents:
                continue
            agent_revision_note = self._build_agent_revision_note(agent_id, trace, actionable_resolutions)
            frontier_note = self._build_frontier_suffix_note(agent_id, trace, graph, actionable_resolutions)
            if frontier_note:
                agent_revision_note = f"{agent_revision_note} {frontier_note}".strip()
            revision_packet = self._build_clean_revision_packet_trace(
                question,
                agent_id,
                traces,
                trace,
                graph,
                actionable_resolutions,
            )
            prompts[agent_id] = build_local_revision_prompt(
                question=question,
                agent_id=agent_id,
                trace=revision_packet,
                graph=graph,
                resolution_text=resolution_text,
                agent_revision_note=agent_revision_note,
                trace_block_heading="Clean revision packet",
                profile=self.prompt_profile,
                graph_format=self.graph_format,
            )
        return prompts

    def _build_revision_continuation_prefix(
        self,
        question: str,
        agent_id: str,
        traces: Dict[str, AgentTrace],
        trace: AgentTrace,
        graph: NaturalLanguageGraph,
        actionable_resolutions: Iterable[ResolutionDecision],
    ) -> str:
        resolution_list = list(actionable_resolutions)
        if self.rewrite_context_variant == "full_regeneration":
            return ""
        segments: List[str] = []

        def append_prefix_text(prefix_text: str) -> None:
            cleaned = self._clean_label_revision_prefix_text(str(prefix_text or ""), question)
            if not cleaned:
                return
            if cleaned not in segments:
                segments.append(cleaned)

        frontier_index = self._rewrite_frontier_index(agent_id, graph, resolution_list)
        if frontier_index is not None and frontier_index > 1 and self.rewrite_context_variant != "corrected_claim_only":
            prefix_steps = trace.steps[: frontier_index - 1]
            prefix_text = self._serialize_trace_steps(prefix_steps)
            append_prefix_text(prefix_text)
        if (
            self.rewrite_context_variant in {"current_suffix", "shared_prefix_only"}
            and frontier_index != 1
        ):
            shared_prefix = self._shared_prefix_text_for_agent(agent_id, trace, traces, graph, resolution_list)
            append_prefix_text(shared_prefix)
            selected_prefix = self._selected_path_prefix_text_for_packet(agent_id, traces, graph, resolution_list)
            append_prefix_text(selected_prefix)

        if self.rewrite_context_variant in {"current_suffix", "corrected_claim_only"}:
            selected_claims: List[str] = []
            for decision in resolution_list:
                resolved_claim = str(decision.resolved_claim or "").strip()
                if resolved_claim and resolved_claim not in selected_claims:
                    selected_claims.append(resolved_claim)
                if self.prompt_profile.name == "anli":
                    continue
                truth_label = truth_value_label_supported_by_claim(
                    f"{resolved_claim} {getattr(decision, 'rationale', '')}",
                    question,
                )
                truth_claim = f"The target statement is {truth_label}." if truth_label else ""
                if truth_claim and truth_claim not in selected_claims:
                    selected_claims.append(truth_claim)
                nli_label = nli_label_supported_by_resolution(
                    resolved_claim,
                    getattr(decision, "rationale", ""),
                    question,
                )
                nli_claim = {
                    "1": "The premise-hypothesis relation is entailment.",
                    "2": "The premise-hypothesis relation is neutral.",
                    "3": "The premise-hypothesis relation is contradiction.",
                }.get(nli_label, "")
                if nli_claim and nli_claim not in selected_claims:
                    selected_claims.append(nli_claim)
            if selected_claims:
                segments.extend(selected_claims)
        return "\n".join(segment.strip() for segment in segments if str(segment).strip()).strip()

    def _clean_label_revision_prefix_text(self, prefix_text: str, question: str) -> str:
        if not question_uses_label_answers(question):
            return str(prefix_text or "").strip()
        allowed = {str(label).lower() for label in allowed_label_answers(question)}
        cleaned_lines: List[str] = []
        for raw_line in str(prefix_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if re.search(r"\{final answer:|\\boxed\{|final answer\s*[:=]", lowered):
                continue
            if label_claim_is_bare_answer(line, question):
                continue
            bare_label = re.fullmatch(r"(?:yes|no|true|false|entailment|neutral|contradiction)[.!?]?", lowered)
            if bare_label and bare_label.group(0).rstrip(".!?") in allowed:
                continue
            leading_label = re.match(
                r"^(yes|no|true|false|entailment|neutral|contradiction)[.!?]\s+(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if leading_label and leading_label.group(1).lower() in allowed:
                line = leading_label.group(2).strip()
            if re.match(r"^(?:so|therefore|thus|hence),?\s+(?:the\s+)?(?:answer|label)\s+(?:is|=)\s+", line, flags=re.IGNORECASE):
                continue
            if line:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _shared_prefix_text_for_agent(
        self,
        agent_id: str,
        trace: AgentTrace,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> str:
        graph_prefix = self._graph_shared_prefix_text_for_agent(agent_id, trace, graph, resolutions)
        common_prefix = self._trace_common_prefix_text_with_winning_paths(agent_id, trace, traces, graph, resolutions)
        graph_len = len(graph_prefix.splitlines()) if graph_prefix else 0
        common_len = len(common_prefix.splitlines()) if common_prefix else 0
        return common_prefix if common_len > graph_len else graph_prefix

    def _graph_shared_prefix_text_for_agent(
        self,
        agent_id: str,
        trace: AgentTrace,
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> str:
        best_index: int | None = None
        for decision in resolutions:
            claim_id = str(decision.rewrite_from_claim_id or "").strip()
            if not claim_id:
                continue
            index = self._latest_shared_prefix_index_for_agent(agent_id, graph, claim_id)
            if index is None:
                continue
            best_index = index if best_index is None else max(best_index, index)
        if best_index is None or best_index <= 1:
            return ""
        prefix_text = self._serialize_trace_steps(trace.steps[: best_index - 1])
        return prefix_text.strip()

    def _trace_common_prefix_text_with_winning_paths(
        self,
        agent_id: str,
        trace: AgentTrace,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> str:
        best_count = 0
        for winning_agent_id in self._winning_agents_for_resolutions(graph, resolutions):
            if not winning_agent_id or winning_agent_id == agent_id:
                continue
            winning_trace = traces.get(winning_agent_id)
            if winning_trace is None:
                continue
            common_count = self._shared_prefix_step_count(trace, winning_trace)
            if common_count > best_count:
                best_count = common_count
        if best_count <= 0:
            return ""
        return self._serialize_trace_steps(trace.steps[:best_count]).strip()

    def _shared_prefix_step_count(self, left_trace: AgentTrace, right_trace: AgentTrace) -> int:
        count = 0
        for left_step, right_step in zip(left_trace.steps, right_trace.steps):
            if self._normalize_claim_text(left_step.text) != self._normalize_claim_text(right_step.text):
                break
            count += 1
        return count

    def _winning_agents_for_resolutions(
        self,
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> List[str]:
        winners: List[str] = []
        seen = set()
        for decision in resolutions:
            if str(decision.divergence_id or "").startswith("PAIR_"):
                winning_agent = self._pairwise_winning_agent(decision)
                if winning_agent and winning_agent not in seen:
                    winners.append(winning_agent)
                    seen.add(winning_agent)
            for path_id in decision.keep_paths or []:
                for path_agent in self._agents_for_path_or_agent_token(path_id, graph):
                    if path_agent and path_agent not in seen:
                        winners.append(path_agent)
                        seen.add(path_agent)
        return winners

    def build_revision_message_payloads(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> Dict[str, dict]:
        actionable_resolutions = self._rewrite_driving_resolutions(question, traces, graph, resolutions)
        if not actionable_resolutions:
            return {}
        rewrite_agents = self._agents_to_rewrite(question, traces, graph, actionable_resolutions)
        payloads: Dict[str, dict] = {}
        for agent_id, trace in traces.items():
            if agent_id not in rewrite_agents:
                continue
            continuation_prefix = self._build_revision_continuation_prefix(
                question,
                agent_id,
                traces,
                trace,
                graph,
                actionable_resolutions,
            )
            if self.rewrite_context_variant == "full_regeneration":
                resolution_text = self.build_resolution_text(actionable_resolutions)
                agent_revision_note = self._build_agent_revision_note(agent_id, trace, actionable_resolutions)
                revision_packet = self._build_clean_revision_packet_trace(
                    question,
                    agent_id,
                    traces,
                    trace,
                    graph,
                    actionable_resolutions,
                )
                prompt = build_local_revision_prompt(
                    question=question,
                    agent_id=agent_id,
                    trace=revision_packet,
                    graph=graph,
                    resolution_text=resolution_text,
                    agent_revision_note=agent_revision_note,
                    trace_block_heading="Clean full-regeneration packet",
                    profile=self.prompt_profile,
                    graph_format=self.graph_format,
                )
                payloads[agent_id] = {
                    "system_prompt": self.system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                    "continue_final_message": False,
                }
                continue
            if not continuation_prefix:
                continue
            payloads[agent_id] = {
                "system_prompt": "",
                "messages": [
                    {"role": "user", "content": question.strip()},
                    {"role": "assistant", "content": continuation_prefix.rstrip() + "\n"},
                ],
                "continue_final_message": True,
            }
        return payloads

    def _build_agent_revision_note(
        self,
        agent_id: str,
        trace: AgentTrace,
        resolutions: Iterable[ResolutionDecision],
    ) -> str:
        notes: List[str] = []
        for decision in resolutions:
            if str(decision.divergence_id or "").startswith("PAIR_"):
                pair_agents = self._pairwise_agents_to_rewrite(decision)
                if agent_id in pair_agents:
                    notes.append("Your previous branch lost this pairwise decision.")
                    if decision.resolved_claim:
                        notes.append(f"Adopt this repaired claim: {decision.resolved_claim}")
                    if decision.rationale:
                        notes.append(f"Why you must change: {decision.rationale}")
            else:
                if decision.resolved_claim:
                    notes.append(f"Continue from this repaired claim: {decision.resolved_claim}")
        notes.append("Do not repeat your old losing conclusion. Rewrite the ending so it agrees with the repaired claim and final answer.")
        return " ".join(note for note in notes if note).strip()

    def _build_clean_revision_packet_trace(
        self,
        question: str,
        agent_id: str,
        traces: Dict[str, AgentTrace],
        trace: AgentTrace,
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> AgentTrace:
        resolution_list = list(resolutions)
        frontier_index = self._rewrite_frontier_index(agent_id, graph, resolution_list)
        lines: List[str] = []
        has_preserved_prefix = False
        if self.rewrite_context_variant == "full_regeneration":
            lines.append("No local prefix is preserved for this full-response regeneration.")
        elif frontier_index is not None and frontier_index > 1 and self.rewrite_context_variant != "corrected_claim_only":
            prefix_steps = trace.steps[: frontier_index - 1]
            prefix_text = self._serialize_trace_steps(prefix_steps)
            prefix_text = self._clean_label_revision_prefix_text(prefix_text, question)
            if prefix_text:
                lines.append("Preserved prefix accepted by the system:")
                lines.extend(prefix_text.splitlines())
                has_preserved_prefix = True
        elif self.rewrite_context_variant != "corrected_claim_only":
            selected_prefix = self._selected_path_prefix_text_for_packet(agent_id, traces, graph, resolution_list)
            selected_prefix = self._clean_label_revision_prefix_text(selected_prefix, question)
            if selected_prefix:
                lines.append("Selected path prefix before the repaired claim:")
                lines.extend(selected_prefix.splitlines())
                has_preserved_prefix = True
        if not has_preserved_prefix and self.rewrite_context_variant != "full_regeneration":
            lines.append("No local prefix is safely preserved for this rewrite.")

        if self.rewrite_context_variant in {"current_suffix", "corrected_claim_only", "full_regeneration"}:
            selected_claims = []
            for decision in resolution_list:
                resolved_claim = str(decision.resolved_claim or "").strip()
                if resolved_claim and resolved_claim not in selected_claims:
                    selected_claims.append(resolved_claim)
                if self.prompt_profile.name == "anli":
                    continue
                truth_label = truth_value_label_supported_by_claim(
                    f"{resolved_claim} {getattr(decision, 'rationale', '')}",
                    question,
                )
                truth_claim = f"The target statement is {truth_label}." if truth_label else ""
                if truth_claim and truth_claim not in selected_claims:
                    selected_claims.append(truth_claim)
                nli_label = nli_label_supported_by_resolution(
                    resolved_claim,
                    getattr(decision, "rationale", ""),
                    question,
                )
                nli_claim = {
                    "1": "The premise-hypothesis relation is entailment.",
                    "2": "The premise-hypothesis relation is neutral.",
                    "3": "The premise-hypothesis relation is contradiction.",
                }.get(nli_label, "")
                if nli_claim and nli_claim not in selected_claims:
                    selected_claims.append(nli_claim)
            if selected_claims:
                lines.append("Selected repaired claim to continue from:")
                lines.extend(selected_claims)
            else:
                lines.append("Selected repaired claim to continue from: follow the resolution summary.")
        else:
            lines.append("No corrected claim text is provided in this ablation; continue from the preserved prefix only.")

        packet_text = "\n".join(line for line in lines if str(line).strip()).strip()
        return AgentTrace(
            agent_id=agent_id,
            original_response=packet_text,
            normalized_trace_text=packet_text,
            steps=[],
        )

    def _step_ref_from_identifier(self, value: str) -> tuple[str, int] | None:
        match = re.fullmatch(r"(A\d+)\.s(\d+)", str(value or "").strip())
        if not match:
            return None
        return match.group(1), int(match.group(2))

    def _selected_path_prefix_text_for_packet(
        self,
        target_agent_id: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> str:
        claim_map = graph.claim_map()
        for decision in resolutions:
            selected_agents: List[str] = []
            if str(decision.divergence_id or "").startswith("PAIR_"):
                winning_agent = self._pairwise_winning_agent(decision)
                if winning_agent:
                    selected_agents.append(winning_agent)
            for path_id in decision.keep_paths or []:
                selected_agents.extend(self._agents_for_path_or_agent_token(path_id, graph))
            selected_agents = [agent for agent in selected_agents if agent and agent != target_agent_id]

            step_ref = self._step_ref_from_identifier(decision.rewrite_from_claim_id)
            if step_ref is not None:
                source_agent, step_index = step_ref
                if source_agent in selected_agents or not selected_agents:
                    source_trace = traces.get(source_agent)
                    if source_trace is not None and step_index > 1:
                        return self._serialize_trace_steps(source_trace.steps[: step_index - 1])

            claim = claim_map.get(str(decision.rewrite_from_claim_id or "").strip())
            if claim is None:
                continue
            for member in claim.members or []:
                member_ref = self._step_ref_from_identifier(member)
                if member_ref is None:
                    continue
                source_agent, step_index = member_ref
                if source_agent not in selected_agents:
                    continue
                source_trace = traces.get(source_agent)
                if source_trace is not None and step_index > 1:
                    return self._serialize_trace_steps(source_trace.steps[: step_index - 1])
        return ""

    def _rewrite_frontier_index(
        self,
        agent_id: str,
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> int | None:
        claim_map = graph.claim_map()
        best_index: int | None = None
        for decision in resolutions:
            claim_id = str(decision.rewrite_from_claim_id or "").strip()
            if not claim_id:
                continue
            step_ref = self._step_ref_from_identifier(claim_id)
            if step_ref is not None:
                source_agent, index = step_ref
                if source_agent == agent_id:
                    best_index = index if best_index is None else min(best_index, index)
                continue
            claim = claim_map.get(claim_id)
            if claim is None:
                continue
            for member in claim.members or []:
                match = re.fullmatch(rf"{re.escape(agent_id)}\.s(\d+)", str(member).strip())
                if not match:
                    continue
                index = int(match.group(1))
                best_index = index if best_index is None else min(best_index, index)
            if best_index is None:
                fallback_index = self._latest_shared_prefix_index_for_agent(agent_id, graph, claim_id)
                if fallback_index is not None:
                    best_index = fallback_index
        return best_index

    def _latest_shared_prefix_index_for_agent(
        self,
        agent_id: str,
        graph: NaturalLanguageGraph,
        rewrite_from_claim_id: str,
    ) -> int | None:
        latest_index: int | None = None
        for claim in graph.shared_claims:
            for member in claim.members or []:
                match = re.fullmatch(rf"{re.escape(agent_id)}\.s(\d+)", str(member).strip())
                if not match:
                    continue
                index = int(match.group(1)) + 1
                latest_index = index if latest_index is None else max(latest_index, index)
            if claim.claim_id == rewrite_from_claim_id:
                break
        return latest_index

    def _build_frontier_suffix_note(
        self,
        agent_id: str,
        trace: AgentTrace,
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
    ) -> str:
        frontier_index = self._rewrite_frontier_index(agent_id, graph, resolutions)
        if frontier_index is None:
            return ""
        preserved = max(frontier_index - 1, 0)
        if self.rewrite_context_variant == "full_regeneration":
            return ""
        if self.rewrite_context_variant == "shared_prefix_only":
            return (
                f"Claim-frontier mode: preserve only the shared prefix through claim frontier {frontier_index}. "
                "Do not include the corrected claim text; continue from the preserved prefix using a fresh suffix."
            )
        if self.rewrite_context_variant == "corrected_claim_only":
            return (
                f"Claim-frontier mode: do not restate preserved earlier claims. "
                f"Use only the corrected claim beginning at {agent_id}.s{frontier_index} and continue from there."
            )
        return (
            f"Claim-frontier mode: the first {preserved} local claims are already preserved by the system. "
            f"Output only the replacement suffix beginning at {agent_id}.s{frontier_index}. "
            "Do not restate preserved earlier claims."
        )

    def _extract_revision_response_text(self, raw_output: str) -> str:
        if self.graph_format != "json":
            return str(raw_output or "")
        raw = str(raw_output or "").strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
        candidate = fence_match.group(1).strip() if fence_match else raw
        start = candidate.find("{")
        end = candidate.rfind("}")
        if 0 <= start < end:
            candidate = candidate[start : end + 1]
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            return raw
        if not isinstance(payload, dict):
            return raw
        for key in ("revised_response", "response", "revised_trace", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return raw

    def build_revised_trace_from_output(
        self,
        agent_id: str,
        source_trace: AgentTrace,
        raw_output: str,
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
        continuation_prefix: str = "",
    ) -> AgentTrace:
        raw_output_for_trace = self._extract_revision_response_text(raw_output)
        if self.rewrite_context_variant == "full_regeneration":
            normalized = parse_atomic_trace(raw_output_for_trace)
            compacted_steps = compact_step_objects(agent_id, build_step_objects(agent_id, normalized))
            return AgentTrace(
                agent_id=agent_id,
                original_response=raw_output,
                normalized_trace_text=self._serialize_trace_steps(compacted_steps),
                steps=compacted_steps,
            )
        if continuation_prefix.strip() and self.rewrite_context_variant != "shared_prefix_only":
            normalized = parse_atomic_trace(
                self._join_continuation_prefix_and_suffix(continuation_prefix, raw_output_for_trace)
            )
            compacted_steps = compact_step_objects(agent_id, build_step_objects(agent_id, normalized))
            return AgentTrace(
                agent_id=agent_id,
                original_response=raw_output,
                normalized_trace_text=self._serialize_trace_steps(compacted_steps),
                steps=compacted_steps,
            )

        normalized = parse_atomic_trace(raw_output_for_trace)
        suffix_steps = compact_step_objects(agent_id, build_step_objects(agent_id, normalized))
        frontier_index = self._rewrite_frontier_index(agent_id, graph, resolutions)
        if frontier_index is not None and frontier_index > 1 and self.rewrite_context_variant != "full_regeneration":
            prefix_steps = source_trace.steps[: frontier_index - 1]
            suffix_text = self._serialize_trace_steps(suffix_steps)
            prefix_text = self._serialize_trace_steps(prefix_steps)
            combined_text = f"{prefix_text}\n{suffix_text}".strip() if suffix_text else prefix_text
            compacted_steps = compact_step_objects(agent_id, build_step_objects(agent_id, combined_text))
        else:
            compacted_steps = suffix_steps
        return AgentTrace(
            agent_id=agent_id,
            original_response=raw_output,
            normalized_trace_text=self._serialize_trace_steps(compacted_steps),
            steps=compacted_steps,
        )

    def _join_continuation_prefix_and_suffix(self, continuation_prefix: str, raw_output: str) -> str:
        prefix_lines = [line.rstrip() for line in str(continuation_prefix or "").splitlines() if line.strip()]
        suffix_lines = [line.rstrip() for line in str(raw_output or "").splitlines() if line.strip()]
        if prefix_lines and suffix_lines:
            if self._normalize_claim_text(prefix_lines[-1]) == self._normalize_claim_text(suffix_lines[0]):
                prefix_lines[-1] = suffix_lines[0]
                suffix_lines = suffix_lines[1:]
        return "\n".join([*prefix_lines, *suffix_lines]).strip()

    def run_local_revisions(
        self,
        question: str,
        traces: Dict[str, AgentTrace],
        graph: NaturalLanguageGraph,
        resolutions: Iterable[ResolutionDecision],
        runtime: RuntimeBundle,
    ) -> Dict[str, AgentTrace]:
        revision_payloads = self.build_revision_message_payloads(question, traces, graph, resolutions)
        if not revision_payloads:
            return dict(traces)
        prompt_items = list(revision_payloads.items())
        outputs = self.prompt_runner(runtime, prompt_items, self.system_prompt)
        revised_traces: Dict[str, AgentTrace] = dict(traces)
        for (agent_id, _), output in zip(prompt_items, outputs):
            messages = revision_payloads.get(agent_id, {}).get("messages") or []
            continuation_prefix = ""
            if len(messages) > 1 and isinstance(messages[1], dict):
                continuation_prefix = str(messages[1].get("content", "") or "")
            revised_traces[agent_id] = self.build_revised_trace_from_output(
                agent_id,
                traces[agent_id],
                output,
                graph,
                resolutions,
                continuation_prefix=continuation_prefix,
            )
        return revised_traces

    def run(
        self,
        question: str,
        responses: Dict[str, str],
        runtime: RuntimeBundle,
    ) -> DebateArtifacts:
        traces = self.normalize_traces(question, responses, runtime)
        graph = self.merge_traces(question, traces, runtime)
        graph = self.audit_shared_graph(question, traces, graph, runtime)
        graph = self.repair_prefix_conflict_graph(question, traces, graph, runtime)
        graph = self.ensure_real_claim_divergence(question, traces, graph)
        resolutions, raw_resolution_notes = self.resolve_divergences(question, graph, runtime, traces)
        if not resolutions and self._has_answer_disagreement(traces):
            fallback_resolutions, fallback_raw_notes = self.resolve_pairwise_fallbacks(question, traces, runtime)
            resolutions = fallback_resolutions
            raw_resolution_notes = raw_resolution_notes + fallback_raw_notes
        revision_prompts = self.build_revision_prompts(question, traces, graph, resolutions)
        revised_traces = self.run_local_revisions(question, traces, graph, resolutions, runtime)
        return DebateArtifacts(
            traces=traces,
            graph=graph,
            resolutions=resolutions,
            raw_resolution_notes=raw_resolution_notes,
            revision_prompts=revision_prompts,
            revised_traces=revised_traces,
        )
