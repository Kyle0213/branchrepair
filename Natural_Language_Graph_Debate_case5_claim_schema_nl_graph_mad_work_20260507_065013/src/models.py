from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class NaturalLanguageStep:
    step_id: str
    agent_id: str
    index: int
    text: str
    source_lines: List[int] = field(default_factory=list)


@dataclass
class AgentTrace:
    agent_id: str
    original_response: str
    normalized_trace_text: str
    steps: List[NaturalLanguageStep]


@dataclass
class ClaimNode:
    claim_id: str
    text: str
    members: List[str] = field(default_factory=list)
    claim_object: str = ""
    aspect: str = ""
    status: str = ""
    alignment: str = ""


@dataclass
class MethodPath:
    path_id: str
    agent_ids: List[str]
    summary: str
    claim_ids: List[str]


@dataclass
class DivergenceCase:
    divergence_id: str
    frontier_claim_id: str
    relation: str
    left_path_id: str
    right_path_id: str
    left_claim: str
    right_claim: str
    why_minimal: str
    claim_object: str = ""
    aspect: str = ""
    alignment: str = ""


@dataclass
class ResolutionDecision:
    divergence_id: str
    action: str
    winning_side: str
    resolved_claim: str
    rationale: str
    rewrite_from_claim_id: str
    keep_paths: List[str] = field(default_factory=list)
    drop_paths: List[str] = field(default_factory=list)
    canonical_answer: str = ""
    raw_action: str = ""


@dataclass
class NaturalLanguageGraph:
    shared_claims: List[ClaimNode] = field(default_factory=list)
    method_paths: List[MethodPath] = field(default_factory=list)
    divergences: List[DivergenceCase] = field(default_factory=list)
    raw_dossier: str = ""

    def claim_map(self) -> Dict[str, ClaimNode]:
        return {claim.claim_id: claim for claim in self.shared_claims}

    def path_map(self) -> Dict[str, MethodPath]:
        return {path.path_id: path for path in self.method_paths}


@dataclass
class DebateArtifacts:
    traces: Dict[str, AgentTrace]
    graph: NaturalLanguageGraph
    resolutions: List[ResolutionDecision]
    revision_prompts: Dict[str, str]
    raw_resolution_notes: List[Dict[str, str]] = field(default_factory=list)
    revised_traces: Dict[str, AgentTrace] = field(default_factory=dict)
