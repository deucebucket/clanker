"""Embedded data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
from .enums import EmbeddedInterrogativeStatus, EmbeddedInterrogativeType, EntityKind, HowKind, QuestionKind, WhyKind
from .events import EventFrame
from .questions import QuestionFrame


@dataclass
class EmbeddedInterrogativeRelation:
    """Typed matrix-to-question link for one embedded interrogative.

    The question proposition is stored as a nonassertive event containing its
    typed variable, while this relation retains the interrogative operator,
    attribution source, and matrix discourse status.  Neither a WH variable nor
    a polar proposition is promoted into an unqualified fact by this relation.
    """

    relation_type: EmbeddedInterrogativeType
    content_status: EmbeddedInterrogativeStatus
    matrix_event_index: int
    question_event_index: int
    marker: str
    matrix_predicate: str
    source_entity_id: str
    predicate_family: str
    question_kind: QuestionKind
    requested_role: Optional[str] = None
    answer_type: EntityKind = EntityKind.UNKNOWN
    certainty: int = 200
    relation_id: str = ""
    matrix_event_id: str = ""
    question_event_id: str = ""
    licensed: bool = True
    direct_answer_request: bool = False
    focus_surface: str = ""
    why_kind: WhyKind = WhyKind.UNKNOWN
    how_kind: HowKind = HowKind.UNKNOWN
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "EmbeddedInterrogativeRelation":
        values: Dict[str, Any] = {
            "relation_type": self.relation_type,
            "content_status": self.content_status,
            "matrix_event_index": self.matrix_event_index,
            "question_event_index": self.question_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "predicate_family": self.predicate_family,
            "question_kind": self.question_kind,
            "requested_role": self.requested_role,
            "answer_type": self.answer_type,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "question_event_id": self.question_event_id,
            "licensed": self.licensed,
            "direct_answer_request": self.direct_answer_request,
            "focus_surface": self.focus_surface,
            "why_kind": self.why_kind,
            "how_kind": self.how_kind,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return EmbeddedInterrogativeRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.relation_type.value,
            self.content_status.value,
            self.matrix_event_id,
            self.question_event_id,
            self.marker,
            self.matrix_predicate,
            self.source_entity_id,
            self.predicate_family,
            self.question_kind.value,
            self.requested_role,
            self.answer_type.value,
            self.licensed,
            self.direct_answer_request,
            self.focus_surface,
            self.why_kind.value,
            self.how_kind.value,
        )

    def to_question_frame(self, event: EventFrame) -> "QuestionFrame":
        return QuestionFrame(
            kind=self.question_kind,
            event=event,
            requested_role=self.requested_role,
            answer_type=self.answer_type,
            why_kind=self.why_kind,
            how_kind=self.how_kind,
            raw_text=event.raw_text,
            focus_surface=self.focus_surface,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "content_status": self.content_status.value,
            "matrix_event_index": self.matrix_event_index,
            "question_event_index": self.question_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "predicate_family": self.predicate_family,
            "question_kind": self.question_kind.value,
            "requested_role": self.requested_role,
            "answer_type": self.answer_type.value,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "question_event_id": self.question_event_id,
            "licensed": self.licensed,
            "direct_answer_request": self.direct_answer_request,
            "focus_surface": self.focus_surface,
            "why_kind": self.why_kind.value,
            "how_kind": self.how_kind.value,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddedInterrogativeRelation":
        return cls(
            relation_type=EmbeddedInterrogativeType(data["relation_type"]),
            content_status=EmbeddedInterrogativeStatus(data["content_status"]),
            matrix_event_index=int(data.get("matrix_event_index", -1)),
            question_event_index=int(data.get("question_event_index", -1)),
            marker=str(data.get("marker", "")),
            matrix_predicate=str(data.get("matrix_predicate", "")),
            source_entity_id=str(data.get("source_entity_id", "")),
            predicate_family=str(data.get("predicate_family", "")),
            question_kind=QuestionKind(data.get("question_kind", QuestionKind.UNKNOWN.value)),
            requested_role=data.get("requested_role"),
            answer_type=EntityKind(data.get("answer_type", EntityKind.UNKNOWN.value)),
            certainty=max(0, min(255, int(data.get("certainty", 200)))),
            relation_id=str(data.get("relation_id", "")),
            matrix_event_id=str(data.get("matrix_event_id", "")),
            question_event_id=str(data.get("question_event_id", "")),
            licensed=bool(data.get("licensed", True)),
            direct_answer_request=bool(data.get("direct_answer_request", False)),
            focus_surface=str(data.get("focus_surface", "")),
            why_kind=WhyKind(data.get("why_kind", WhyKind.UNKNOWN.value)),
            how_kind=HowKind(data.get("how_kind", HowKind.UNKNOWN.value)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class EmbeddedInterrogativeAttachmentAmbiguity:
    """Explicit unresolved matrix/question boundary or interrogative scope."""

    matrix_surface: str
    question_surface: str
    clause_surface: str
    reason: str
    candidate_boundaries: List[int] = field(default_factory=list)
    candidate_markers: List[str] = field(default_factory=list)
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix_surface": self.matrix_surface,
            "question_surface": self.question_surface,
            "clause_surface": self.clause_surface,
            "reason": self.reason,
            "candidate_boundaries": list(self.candidate_boundaries),
            "candidate_markers": list(self.candidate_markers),
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "EmbeddedInterrogativeAttachmentAmbiguity":
        return cls(
            matrix_surface=str(data.get("matrix_surface", "")),
            question_surface=str(data.get("question_surface", "")),
            clause_surface=str(data.get("clause_surface", "")),
            reason=str(data.get("reason", "")),
            candidate_boundaries=[int(item) for item in data.get("candidate_boundaries", [])],
            candidate_markers=[str(item) for item in data.get("candidate_markers", [])],
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )
