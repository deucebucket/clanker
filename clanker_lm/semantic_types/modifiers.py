"""Modifiers data contracts; no runtime or persistence dependencies."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple
from .enums import AppositiveRelationType, ModifierGapRole, ModifierRestriction, SourceKind


@dataclass
class EntityModifierRelation:
    """Typed link from one entity to a finite relative-clause event."""

    head_entity_id: str
    modifier_event_index: int
    marker: str
    gap_role: ModifierGapRole
    restriction: ModifierRestriction
    certainty: int = 230
    relation_id: str = ""
    modifier_event_id: str = ""
    modifier_event_signature: str = ""
    possessed_entity_id: str = ""
    inferred: bool = False
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "EntityModifierRelation":
        values: Dict[str, Any] = {
            "head_entity_id": self.head_entity_id,
            "modifier_event_index": self.modifier_event_index,
            "marker": self.marker,
            "gap_role": self.gap_role,
            "restriction": self.restriction,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "modifier_event_id": self.modifier_event_id,
            "modifier_event_signature": self.modifier_event_signature,
            "possessed_entity_id": self.possessed_entity_id,
            "inferred": self.inferred,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return EntityModifierRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.head_entity_id,
            self.modifier_event_id,
            self.marker,
            self.gap_role.value,
            self.restriction.value,
            self.possessed_entity_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_entity_id": self.head_entity_id,
            "modifier_event_index": self.modifier_event_index,
            "marker": self.marker,
            "gap_role": self.gap_role.value,
            "restriction": self.restriction.value,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "modifier_event_id": self.modifier_event_id,
            "modifier_event_signature": self.modifier_event_signature,
            "possessed_entity_id": self.possessed_entity_id,
            "inferred": self.inferred,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityModifierRelation":
        return cls(
            head_entity_id=str(data["head_entity_id"]),
            modifier_event_index=int(data.get("modifier_event_index", -1)),
            marker=str(data.get("marker", "")),
            gap_role=ModifierGapRole(
                data.get("gap_role", ModifierGapRole.AGENT.value)
            ),
            restriction=ModifierRestriction(
                data.get("restriction", ModifierRestriction.RESTRICTIVE.value)
            ),
            certainty=max(0, min(255, int(data.get("certainty", 230)))),
            relation_id=str(data.get("relation_id", "")),
            modifier_event_id=str(data.get("modifier_event_id", "")),
            modifier_event_signature=str(data.get("modifier_event_signature", "")),
            possessed_entity_id=str(data.get("possessed_entity_id", "")),
            inferred=bool(data.get("inferred", False)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class AppositiveRelation:
    """Typed identity/description link licensed by explicit apposition."""

    head_entity_id: str
    primary_surface: str
    appositive_surface: str
    relation_type: AppositiveRelationType
    restriction: ModifierRestriction
    appositive_key: str = ""
    role_owner_id: str = ""
    role_name: str = ""
    certainty: int = 230
    relation_id: str = ""
    source: SourceKind = SourceKind.USER
    diagnostics: List[str] = field(default_factory=list)

    def copy(self, **changes: Any) -> "AppositiveRelation":
        values: Dict[str, Any] = {
            "head_entity_id": self.head_entity_id,
            "primary_surface": self.primary_surface,
            "appositive_surface": self.appositive_surface,
            "relation_type": self.relation_type,
            "restriction": self.restriction,
            "appositive_key": self.appositive_key,
            "role_owner_id": self.role_owner_id,
            "role_name": self.role_name,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "source": self.source,
            "diagnostics": list(self.diagnostics),
        }
        values.update(changes)
        return AppositiveRelation(**values)

    def signature(self) -> Tuple[Any, ...]:
        return (
            self.head_entity_id,
            self.appositive_key,
            self.relation_type.value,
            self.restriction.value,
            self.role_owner_id,
            self.role_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_entity_id": self.head_entity_id,
            "primary_surface": self.primary_surface,
            "appositive_surface": self.appositive_surface,
            "relation_type": self.relation_type.value,
            "restriction": self.restriction.value,
            "appositive_key": self.appositive_key,
            "role_owner_id": self.role_owner_id,
            "role_name": self.role_name,
            "certainty": self.certainty,
            "relation_id": self.relation_id,
            "source": self.source.value,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppositiveRelation":
        return cls(
            head_entity_id=str(data["head_entity_id"]),
            primary_surface=str(data.get("primary_surface", "")),
            appositive_surface=str(data.get("appositive_surface", "")),
            relation_type=AppositiveRelationType(
                data.get("relation_type", AppositiveRelationType.IDENTITY.value)
            ),
            restriction=ModifierRestriction(
                data.get("restriction", ModifierRestriction.NONRESTRICTIVE.value)
            ),
            appositive_key=str(data.get("appositive_key", "")),
            role_owner_id=str(data.get("role_owner_id", "")),
            role_name=str(data.get("role_name", "")),
            certainty=max(0, min(255, int(data.get("certainty", 230)))),
            relation_id=str(data.get("relation_id", "")),
            source=SourceKind(data.get("source", SourceKind.USER.value)),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class AppositiveAttachmentAmbiguity:
    """Explicit unresolved appositive identity or attachment choice."""

    primary_surface: str
    appositive_surface: str
    clause_surface: str
    reason: str
    candidate_entity_ids: List[str] = field(default_factory=list)
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_surface": self.primary_surface,
            "appositive_surface": self.appositive_surface,
            "clause_surface": self.clause_surface,
            "reason": self.reason,
            "candidate_entity_ids": list(self.candidate_entity_ids),
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppositiveAttachmentAmbiguity":
        return cls(
            primary_surface=str(data.get("primary_surface", "")),
            appositive_surface=str(data.get("appositive_surface", "")),
            clause_surface=str(data.get("clause_surface", "")),
            reason=str(data.get("reason", "")),
            candidate_entity_ids=list(data.get("candidate_entity_ids", [])),
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )


@dataclass
class ModifierAttachmentAmbiguity:
    """Explicit unresolved choice between multiple relative attachments."""

    marker: str
    clause_surface: str
    candidate_head_surfaces: List[str]
    reason: str
    ambiguity_id: str = ""
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker": self.marker,
            "clause_surface": self.clause_surface,
            "candidate_head_surfaces": list(self.candidate_head_surfaces),
            "reason": self.reason,
            "ambiguity_id": self.ambiguity_id,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModifierAttachmentAmbiguity":
        return cls(
            marker=str(data.get("marker", "")),
            clause_surface=str(data.get("clause_surface", "")),
            candidate_head_surfaces=list(data.get("candidate_head_surfaces", [])),
            reason=str(data.get("reason", "")),
            ambiguity_id=str(data.get("ambiguity_id", "")),
            diagnostics=list(data.get("diagnostics", [])),
        )
