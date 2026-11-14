"""Questions data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .enums import EmbeddedInterrogativeType, EntityKind, HowKind, QuestionKind, WhyKind
from .events import EventFrame
from .references import UnresolvedReference


@dataclass
class QuestionFrame:
    kind: QuestionKind
    event: EventFrame
    requested_role: Optional[str] = None
    answer_type: EntityKind = EntityKind.UNKNOWN
    why_kind: WhyKind = WhyKind.UNKNOWN
    how_kind: HowKind = HowKind.UNKNOWN
    raw_text: str = ""
    unresolved: List[UnresolvedReference] = field(default_factory=list)
    focus_surface: str = ""
    social_convention: Optional[str] = None
    matrix_polarity: Optional[bool] = None
    embedded_polarity: Optional[bool] = None
    embedded_question: Optional["QuestionFrame"] = None
    embedded_interrogative_type: Optional[EmbeddedInterrogativeType] = None
    embedded_marker: str = ""
    embedded_matrix_predicate: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "event": self.event.to_dict(),
            "requested_role": self.requested_role,
            "answer_type": self.answer_type.value,
            "why_kind": self.why_kind.value,
            "how_kind": self.how_kind.value,
            "raw_text": self.raw_text,
            "unresolved": [item.to_dict() for item in self.unresolved],
            "focus_surface": self.focus_surface,
            "social_convention": self.social_convention,
            "matrix_polarity": self.matrix_polarity,
            "embedded_polarity": self.embedded_polarity,
            "embedded_question": (
                self.embedded_question.to_dict()
                if self.embedded_question is not None
                else None
            ),
            "embedded_interrogative_type": (
                self.embedded_interrogative_type.value
                if self.embedded_interrogative_type is not None
                else None
            ),
            "embedded_marker": self.embedded_marker,
            "embedded_matrix_predicate": self.embedded_matrix_predicate,
        }
