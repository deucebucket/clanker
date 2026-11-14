"""Answers data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .enums import AnswerStatus, SourceKind, TruthValue
from .events import EventFrame
from .questions import QuestionFrame
from .references import SemanticRef


@dataclass
class Evidence:
    event: EventFrame
    matched_roles: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "matched_roles": list(self.matched_roles),
            "score": self.score,
        }


@dataclass
class AnswerContract:
    status: AnswerStatus
    question: Optional[QuestionFrame] = None
    proposition: Optional[EventFrame] = None
    values: List[SemanticRef] = field(default_factory=list)
    truth: TruthValue = TruthValue.UNKNOWN
    evidence: List[Evidence] = field(default_factory=list)
    certainty: int = 0
    source: SourceKind = SourceKind.UNKNOWN
    reason: str = ""
    response_goal: str = "help"
    required_slots: Dict[str, str] = field(default_factory=dict)
    forbidden_claims: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "question": self.question.to_dict() if self.question else None,
            "proposition": self.proposition.to_dict() if self.proposition else None,
            "values": [value.to_dict() for value in self.values],
            "truth": self.truth.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "certainty": self.certainty,
            "source": self.source.value,
            "reason": self.reason,
            "response_goal": self.response_goal,
            "required_slots": dict(self.required_slots),
            "forbidden_claims": list(self.forbidden_claims),
            "diagnostics": list(self.diagnostics),
        }
