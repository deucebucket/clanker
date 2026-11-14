"""Gerunds data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple
from .enums import GerundContentStatus, GerundRelationType


@dataclass
class GerundRelation:
    """Typed link between a matrix event and one selected ``-ing`` event.

    The embedded event is always stored as attributed/nonassertive content.
    ``entailed`` records only a catalog-reviewed phase inference (for example,
    prior activity under ``stop``); it never promotes the embedded frame to an
    unqualified completed event.
    """

    relation_type: GerundRelationType
    content_status: GerundContentStatus
    matrix_event_index: int
    complement_event_index: int
    marker: str
    matrix_predicate: str
    source_entity_id: str
    controller_entity_id: str
    embedded_subject_entity_id: str
    predicate_family: str
    certainty: int = 200
    relation_id: str = ""
    matrix_event_id: str = ""
    complement_event_id: str = ""
    licensed: bool = True
    entailed: bool = False
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "GerundRelation":
        values: Dict[str, Any] = {
            "relation_type": self.relation_type,
            "content_status": self.content_status,
            "matrix_event_index": self.matrix_event_index,
            "complement_event_index": self.complement_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "controller_entity_id": self.controller_entity_id,
            "embedded_subject_entity_id": self.embedded_subject_entity_id,
            "predicate_family": self.predicate_family,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "complement_event_id": self.complement_event_id,
            "licensed": self.licensed,
            "entailed": self.entailed,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return GerundRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.relation_type.value,
            self.content_status.value,
            self.matrix_event_id,
            self.complement_event_id,
            self.marker,
            self.matrix_predicate,
            self.source_entity_id,
            self.controller_entity_id,
            self.embedded_subject_entity_id,
            self.predicate_family,
            self.licensed,
            self.entailed,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "content_status": self.content_status.value,
            "matrix_event_index": self.matrix_event_index,
            "complement_event_index": self.complement_event_index,
            "marker": self.marker,
            "matrix_predicate": self.matrix_predicate,
            "source_entity_id": self.source_entity_id,
            "controller_entity_id": self.controller_entity_id,
            "embedded_subject_entity_id": self.embedded_subject_entity_id,
            "predicate_family": self.predicate_family,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "matrix_event_id": self.matrix_event_id,
            "complement_event_id": self.complement_event_id,
            "licensed": self.licensed,
            "entailed": self.entailed,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GerundRelation":
        return cls(
            relation_type=GerundRelationType(data["relation_type"]),
            content_status=GerundContentStatus(data["content_status"]),
            matrix_event_index=int(data.get("matrix_event_index", -1)),
            complement_event_index=int(data.get("complement_event_index", -1)),
            marker=str(data.get("marker", "-ing")),
            matrix_predicate=str(data.get("matrix_predicate", "")),
            source_entity_id=str(data.get("source_entity_id", "")),
            controller_entity_id=str(data.get("controller_entity_id", "")),
            embedded_subject_entity_id=str(
                data.get("embedded_subject_entity_id", "")
            ),
            predicate_family=str(data.get("predicate_family", "")),
            certainty=max(0, min(255, int(data.get("certainty", 200)))),
            relation_id=str(data.get("relation_id", "")),
            matrix_event_id=str(data.get("matrix_event_id", "")),
            complement_event_id=str(data.get("complement_event_id", "")),
            licensed=bool(data.get("licensed", True)),
            entailed=bool(data.get("entailed", False)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class GerundAttachmentAmbiguity:
    """Explicit unresolved boundary/controller for selected ``-ing`` content."""

    matrix_surface: str
    complement_surface: str
    clause_surface: str
    reason: str
    candidate_boundaries: List[int] = field(default_factory=list)
    candidate_relation_types: List[GerundRelationType] = field(default_factory=list)
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matrix_surface": self.matrix_surface,
            "complement_surface": self.complement_surface,
            "clause_surface": self.clause_surface,
            "reason": self.reason,
            "candidate_boundaries": list(self.candidate_boundaries),
            "candidate_relation_types": [
                item.value for item in self.candidate_relation_types
            ],
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GerundAttachmentAmbiguity":
        return cls(
            matrix_surface=str(data.get("matrix_surface", "")),
            complement_surface=str(data.get("complement_surface", "")),
            clause_surface=str(data.get("clause_surface", "")),
            reason=str(data.get("reason", "")),
            candidate_boundaries=[
                int(item) for item in data.get("candidate_boundaries", [])
            ],
            candidate_relation_types=[
                GerundRelationType(item)
                for item in data.get("candidate_relation_types", [])
            ],
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )
