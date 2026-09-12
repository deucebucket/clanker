"""Events data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from .enums import ClauseRelationDirection, ClauseRelationType, SourceKind
from .references import SemanticRef


@dataclass
class EventFrame:
    predicate: str
    arguments: Dict[str, SemanticRef] = field(default_factory=dict)
    tense: str = "present"
    aspect: str = "simple"
    polarity: bool = True
    modality: Optional[str] = None
    raw_text: str = ""
    event_id: str = ""
    source: SourceKind = SourceKind.USER
    certainty: int = 230
    turn_index: int = 0
    inferred: bool = False
    discourse_role: str = "main"

    def copy(self, **changes: Any) -> "EventFrame":
        values: Dict[str, Any] = {
            "predicate": self.predicate,
            "arguments": dict(self.arguments),
            "tense": self.tense,
            "aspect": self.aspect,
            "polarity": self.polarity,
            "modality": self.modality,
            "raw_text": self.raw_text,
            "event_id": self.event_id,
            "source": self.source,
            "certainty": self.certainty,
            "turn_index": self.turn_index,
            "inferred": self.inferred,
            "discourse_role": self.discourse_role,
        }
        values.update(changes)
        return EventFrame(**values)

    def fixed_arguments(self) -> Dict[str, SemanticRef]:
        return {key: value for key, value in self.arguments.items() if not value.is_variable}

    def variable_roles(self) -> List[str]:
        return [key for key, value in self.arguments.items() if value.is_variable]

    def signature(self, exclude_roles: Iterable[str] = ()) -> Tuple[Any, ...]:
        excluded = set(exclude_roles)
        args = tuple(
            sorted(
                (role, ref.kind.value, ref.key)
                for role, ref in self.arguments.items()
                if role not in excluded and not ref.is_variable
            )
        )
        return self.predicate, self.tense, self.polarity, self.modality, args

    def proposition_signature(self) -> Tuple[Any, ...]:
        args = tuple(sorted((role, ref.kind.value, ref.key) for role, ref in self.arguments.items() if not ref.is_variable))
        return self.predicate, self.polarity, args

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "arguments": {key: ref.to_dict() for key, ref in self.arguments.items()},
            "tense": self.tense,
            "aspect": self.aspect,
            "polarity": self.polarity,
            "modality": self.modality,
            "raw_text": self.raw_text,
            "event_id": self.event_id,
            "source": self.source.value,
            "certainty": self.certainty,
            "turn_index": self.turn_index,
            "inferred": self.inferred,
            "discourse_role": self.discourse_role,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventFrame":
        return cls(
            predicate=str(data["predicate"]),
            arguments={key: SemanticRef.from_dict(value) for key, value in dict(data.get("arguments", {})).items()},
            tense=str(data.get("tense", "present")),
            aspect=str(data.get("aspect", "simple")),
            polarity=bool(data.get("polarity", True)),
            modality=data.get("modality"),
            raw_text=str(data.get("raw_text", "")),
            event_id=str(data.get("event_id", "")),
            source=SourceKind(data.get("source", SourceKind.USER.value)),
            certainty=int(data.get("certainty", 230)),
            turn_index=int(data.get("turn_index", 0)),
            inferred=bool(data.get("inferred", False)),
            discourse_role=str(data.get("discourse_role", "main")),
        )


@dataclass
class ClauseRelation:
    """A typed relation between one main and one subordinate event.

    Parser instances use event-list indices.  Conversation memory binds those
    indices to stable event IDs without changing the relation semantics.
    """

    relation_type: ClauseRelationType
    main_event_index: int
    subordinate_event_index: int
    marker: str
    direction: ClauseRelationDirection
    certainty: int = 230
    candidate_types: List[ClauseRelationType] = field(default_factory=list)
    relation_id: str = ""
    main_event_id: str = ""
    subordinate_event_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return self.relation_type == ClauseRelationType.AMBIGUOUS

    def copy(self, **changes: Any) -> "ClauseRelation":
        values: Dict[str, Any] = {
            "relation_type": self.relation_type,
            "main_event_index": self.main_event_index,
            "subordinate_event_index": self.subordinate_event_index,
            "marker": self.marker,
            "direction": self.direction,
            "certainty": self.certainty,
            "candidate_types": list(self.candidate_types),
            "relation_id": self.relation_id,
            "main_event_id": self.main_event_id,
            "subordinate_event_id": self.subordinate_event_id,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return ClauseRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.relation_type.value,
            self.main_event_id,
            self.subordinate_event_id,
            self.marker,
            self.direction.value,
            tuple(item.value for item in self.candidate_types),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "main_event_index": self.main_event_index,
            "subordinate_event_index": self.subordinate_event_index,
            "marker": self.marker,
            "direction": self.direction.value,
            "certainty": self.certainty,
            "candidate_types": [item.value for item in self.candidate_types],
            "relation_id": self.relation_id,
            "main_event_id": self.main_event_id,
            "subordinate_event_id": self.subordinate_event_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClauseRelation":
        return cls(
            relation_type=ClauseRelationType(data["relation_type"]),
            main_event_index=int(data.get("main_event_index", -1)),
            subordinate_event_index=int(data.get("subordinate_event_index", -1)),
            marker=str(data.get("marker", "")),
            direction=ClauseRelationDirection(
                data.get("direction", ClauseRelationDirection.UNRESOLVED.value)
            ),
            certainty=max(0, min(255, int(data.get("certainty", 230)))),
            candidate_types=[
                ClauseRelationType(item)
                for item in data.get("candidate_types", [])
            ],
            relation_id=str(data.get("relation_id", "")),
            main_event_id=str(data.get("main_event_id", "")),
            subordinate_event_id=str(data.get("subordinate_event_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )
