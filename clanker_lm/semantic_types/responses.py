"""Responses data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from .affect import AffectReading, AffectVector
from .answers import AnswerContract
from .parses import ParseResult
from .serialization import _json_safe


@dataclass
class GateDecision:
    register: str = "neutral"
    severity: str = "low"
    masking: bool = False
    locked_pools: List[str] = field(default_factory=list)
    allowed_pools: List[str] = field(default_factory=list)
    response_act: str = "answer"
    max_sentences: int = 2
    requires_probe: bool = False
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateResponse:
    text: str
    construction_id: str
    semantic_valid: bool = True
    semantic_reason: str = ""
    affect: Optional[AffectVector] = None
    predicted_state: Optional[AffectVector] = None
    affect_distance: float = 0.0
    priority: int = 0
    score: float = 0.0
    atom_ids: List[str] = field(default_factory=list)
    semantic_plan: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "construction_id": self.construction_id,
            "semantic_valid": self.semantic_valid,
            "semantic_reason": self.semantic_reason,
            "affect": self.affect.to_dict() if self.affect else None,
            "predicted_state": self.predicted_state.to_dict() if self.predicted_state else None,
            "affect_distance": self.affect_distance,
            "priority": self.priority,
            "score": self.score,
            "atom_ids": list(self.atom_ids),
            "semantic_plan": list(self.semantic_plan),
        }


@dataclass
class TurnResult:
    input_text: str
    response: str
    parse: ParseResult
    contract: AnswerContract
    gates: GateDecision
    input_affect: AffectReading
    observed_state: AffectVector
    target_state: AffectVector
    predicted_state: AffectVector
    candidates: List[CandidateResponse] = field(default_factory=list)
    memory_revision: int = 0
    learning: Optional[Dict[str, Any]] = None
    resolver: Optional[Dict[str, Any]] = None
    trajectory: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "response": self.response,
            "parse": self.parse.to_dict(),
            "contract": self.contract.to_dict(),
            "gates": self.gates.to_dict(),
            "input_affect": self.input_affect.to_dict(),
            "observed_state": self.observed_state.to_dict(),
            "target_state": self.target_state.to_dict(),
            "predicted_state": self.predicted_state.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "memory_revision": self.memory_revision,
            "learning": _json_safe(self.learning),
            "resolver": _json_safe(self.resolver),
            "trajectory": _json_safe(self.trajectory),
        }
