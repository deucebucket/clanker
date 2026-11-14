"""References responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import Any, Optional, Tuple
from .. import lexicon
from ..database import Atom
from ..model import Entity, EntityKind, RefKind, SemanticRef

class ReferencesComponent:
    """References behavior composed by the stable public SurfaceRealizer API."""

    def _role_surface(self, role: str) -> Tuple[str, Optional[Atom]]:
        normalized = role.lower().replace("_", " ").strip()
        candidates = self.store.atom_candidates(
            "semantic_role_noun",
            register="neutral",
            features={"role": normalized.replace(" ", "_")},
        )
        if candidates:
            return candidates[0].surface, candidates[0]
        if normalized in {"answer", "timezone", "calculation", "meaning", "example", "information"}:
            id_map = {
                "answer": "meta.answer",
                "timezone": "meta.timezone",
                "calculation": "meta.calculation",
                "meaning": "meta.meaning",
                "example": "meta.example",
                "information": "meta.information",
            }
            atom = self.store.atom_by_id(id_map[normalized])
            if atom:
                return atom.surface, atom
        return normalized or "answer", None

    @staticmethod
    def _reference_is_person(reference: str) -> bool:
        lowered = reference.lower().strip(" .?!\"'“”")
        return lowered in {"he", "she", "him", "her", "they", "them", "who"}

    @staticmethod
    def _reason_roles(question: Any) -> Tuple[str, ...]:
        if not question:
            return ("cause", "motive", "purpose", "justification", "evidence", "method", "manner", "process", "mechanism")
        return tuple({
            question.requested_role or "",
            "cause", "motive", "purpose", "justification", "evidence",
            "method", "manner", "process", "mechanism",
        } - {""})

    def render_ref(
        self,
        ref: Optional[SemanticRef],
        *,
        case: str = "object",
        definite: bool = False,
        capitalize: bool = False,
    ) -> str:
        if ref is None:
            return ""
        if ref.kind == RefKind.VARIABLE:
            return ref.surface or f"the {ref.key}"
        if ref.kind == RefKind.EVENT:
            event = next((item for item in self.memory.events if item.event_id == ref.key), None)
            text = self.render_event(event, capitalize=False) if event else (ref.surface or "the event")
            return self._capitalize(text) if capitalize else text
        if ref.kind == RefKind.LITERAL:
            text = ref.surface or ref.key
            return self._capitalize(text) if capitalize else text

        entity = self.memory.get_entity(ref.key)
        if not entity:
            text = ref.surface or ref.key
            return self._capitalize(text) if capitalize else text
        if entity.entity_id == "user":
            text = {"subject": "you", "object": "you", "possessive": "your"}.get(case, "you")
        elif entity.entity_id == "assistant":
            text = {"subject": "I", "object": "me", "possessive": "my"}.get(case, "me")
        elif entity.relation and entity.owner_id:
            owner = "your" if entity.owner_id == "user" else "my" if entity.owner_id == "assistant" else self.memory.describe_entity(entity.owner_id) + "'s"
            text = f"{owner} {entity.relation}"
        else:
            text = entity.canonical_name
            if definite and entity.kind in {EntityKind.THING, EntityKind.PLACE} and not self._has_determiner(text) and not self._looks_proper(entity):
                text = "the " + text
            elif not definite and entity.kind == EntityKind.THING and ref.surface:
                text = ref.surface
        return self._capitalize(text) if capitalize else text

    @staticmethod
    def _looks_proper(entity: Entity) -> bool:
        return bool(entity.canonical_name[:1].isupper())

    @staticmethod
    def _has_determiner(text: str) -> bool:
        first = text.lower().split()[0] if text.split() else ""
        return first in lexicon.DETERMINERS | lexicon.POSSESSIVES

    def _should_be_definite(self, ref: SemanticRef) -> bool:
        entity = self.memory.get_entity(ref.key) if ref.kind == RefKind.ENTITY else None
        return bool(entity and entity.last_mentioned_turn < self.memory.turn_index)
