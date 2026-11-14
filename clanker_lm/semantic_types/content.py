"""Content data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple
from .enums import ContentRelationType


@dataclass
class ContentRelation:
    """Typed attribution link from a matrix predicate to finite content.

    Parser instances use event-list indices. Conversation memory binds them to
    stable event IDs. The content event is evidence about what a source said,
    believed, knew, or perceived; it is not automatically an unqualified fact.
    """

    relation_type: ContentRelationType
    matrix_event_index: int
    content_event_index: int
    marker: str
    matrix_predicate: str
    source_entity_id: str
    predicate_family: str
    certainty: int = 210
    relation_id: str = ""
    matrix_event_id: str = ""
    content_event_id: str = ""
    attributed: bool = True
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "ContentRelation":
        values: Dict[str, Any] = {
            "relation_type": self.relation_type,
            "matrix_event_index": self.matrix_event_index,
            "content_event_index": self.content_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "predicate_family": self.predicate_family,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "content_event_id": self.content_event_id,
            "attributed": self.attributed,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return ContentRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.relation_type.value,
            self.matrix_event_id,
            self.content_event_id,
            self.marker,
            self.matrix_predicate,
            self.source_entity_id,
            self.predicate_family,
            self.attributed,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "matrix_event_index": self.matrix_event_index,
            "content_event_index": self.content_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "predicate_family": self.predicate_family,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "content_event_id": self.content_event_id,
            "attributed": self.attributed,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContentRelation":
        return cls(
            relation_type=ContentRelationType(data["relation_type"]),
            matrix_event_index=int(data.get("matrix_event_index", -1)),
            content_event_index=int(data.get("content_event_index", -1)),
            marker=str(data.get("marker", "")),
            matrix_predicate=str(data.get("matrix_predicate", "")),
            source_entity_id=str(data.get("source_entity_id", "")),
            predicate_family=str(data.get("predicate_family", "")),
            certainty=max(0, min(255, int(data.get("certainty", 210)))),
            relation_id=str(data.get("relation_id", "")),
            matrix_event_id=str(data.get("matrix_event_id", "")),
            content_event_id=str(data.get("content_event_id", "")),
            attributed=bool(data.get("attributed", True)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class ContentAttachmentAmbiguity:
    """Explicit unresolved boundary or attribution for a content clause."""

    matrix_surface: str
    content_surface: str
    clause_surface: str
    reason: str
    candidate_boundaries: List[int] = field(default_factory=list)
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix_surface": self.matrix_surface,
            "content_surface": self.content_surface,
            "clause_surface": self.clause_surface,
            "reason": self.reason,
            "candidate_boundaries": list(self.candidate_boundaries),
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContentAttachmentAmbiguity":
        return cls(
            matrix_surface=str(data.get("matrix_surface", "")),
            content_surface=str(data.get("content_surface", "")),
            clause_surface=str(data.get("clause_surface", "")),
            reason=str(data.get("reason", "")),
            candidate_boundaries=[int(item) for item in data.get("candidate_boundaries", [])],
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )
