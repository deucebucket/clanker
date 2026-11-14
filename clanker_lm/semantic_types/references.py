"""References data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional
from .enums import EntityKind, Gender, GrammaticalNumber, RefKind


@dataclass(frozen=True)
class SemanticRef:
    """A typed value used in a semantic frame.

    ``key`` is canonical and comparison-safe.  ``surface`` preserves a useful
    wording for realization.  Entity references use ``entity_id`` as their key;
    literals use a normalized literal string; variables use the requested slot
    name (for example ``patient`` or ``cause``).
    """

    kind: RefKind
    key: str
    surface: str = ""
    value_type: EntityKind = EntityKind.UNKNOWN

    @classmethod
    def entity(cls, entity_id: str, surface: str = "", value_type: EntityKind = EntityKind.UNKNOWN) -> "SemanticRef":
        return cls(RefKind.ENTITY, entity_id, surface, value_type)

    @classmethod
    def literal(cls, value: str, surface: str = "", value_type: EntityKind = EntityKind.ABSTRACT) -> "SemanticRef":
        return cls(RefKind.LITERAL, value, surface or value, value_type)

    @classmethod
    def variable(cls, role: str, value_type: EntityKind = EntityKind.UNKNOWN) -> "SemanticRef":
        return cls(RefKind.VARIABLE, role, f"?{role}", value_type)

    @classmethod
    def event(cls, event_id: str, surface: str = "") -> "SemanticRef":
        return cls(RefKind.EVENT, event_id, surface, EntityKind.EVENT)

    @property
    def is_variable(self) -> bool:
        return self.kind == RefKind.VARIABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "key": self.key,
            "surface": self.surface,
            "value_type": self.value_type.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticRef":
        return cls(
            kind=RefKind(data["kind"]),
            key=str(data["key"]),
            surface=str(data.get("surface", "")),
            value_type=EntityKind(data.get("value_type", EntityKind.UNKNOWN.value)),
        )


@dataclass
class Entity:
    entity_id: str
    canonical_name: str
    kind: EntityKind = EntityKind.UNKNOWN
    gender: Gender = Gender.UNKNOWN
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR
    owner_id: Optional[str] = None
    relation: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    created_turn: int = 0
    last_mentioned_turn: int = 0
    salience: float = 0.0

    def add_alias(self, alias: str) -> None:
        normalized = " ".join(alias.lower().split()).strip()
        if normalized and normalized not in self.aliases:
            self.aliases.append(normalized)

    def to_ref(self, surface: str = "") -> SemanticRef:
        return SemanticRef.entity(self.entity_id, surface or self.canonical_name, self.kind)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "kind": self.kind.value,
            "gender": self.gender.value,
            "number": self.number.value,
            "owner_id": self.owner_id,
            "relation": self.relation,
            "aliases": list(self.aliases),
            "attributes": dict(self.attributes),
            "created_turn": self.created_turn,
            "last_mentioned_turn": self.last_mentioned_turn,
            "salience": self.salience,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Entity":
        return cls(
            entity_id=str(data["entity_id"]),
            canonical_name=str(data["canonical_name"]),
            kind=EntityKind(data.get("kind", EntityKind.UNKNOWN.value)),
            gender=Gender(data.get("gender", Gender.UNKNOWN.value)),
            number=GrammaticalNumber(data.get("number", GrammaticalNumber.SINGULAR.value)),
            owner_id=data.get("owner_id"),
            relation=data.get("relation"),
            aliases=list(data.get("aliases", [])),
            attributes=dict(data.get("attributes", {})),
            created_turn=int(data.get("created_turn", 0)),
            last_mentioned_turn=int(data.get("last_mentioned_turn", 0)),
            salience=float(data.get("salience", 0.0)),
        )


@dataclass
class UnresolvedReference:
    surface: str
    reason: str
    candidates: List[str] = field(default_factory=list)
    expected_kind: EntityKind = EntityKind.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface": self.surface,
            "reason": self.reason,
            "candidates": list(self.candidates),
            "expected_kind": self.expected_kind.value,
        }
