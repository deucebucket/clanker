"""Persistent symbolic conversation memory for Clanker-LM.

The memory stores compact entities and event frames rather than replaying a text
context window.  Pronoun resolution, evidence lookup, conflict detection, and
JSON snapshots all operate on this state.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import lexicon
from .model import (
    AppositiveAttachmentAmbiguity,
    AppositiveRelation,
    AppositiveRelationType,
    ClauseRelation,
    ContentRelation,
    EmbeddedInterrogativeRelation,
    Entity,
    EntityModifierRelation,
    EntityKind,
    EventFrame,
    GerundContentStatus,
    GerundRelation,
    GerundRelationType,
    Gender,
    GrammaticalNumber,
    InfinitivalRelation,
    InfinitivalRelationType,
    RefKind,
    SemanticRef,
    SourceKind,
    UnresolvedReference,
)


@dataclass
class Resolution:
    status: str
    entity: Optional[Entity] = None
    candidates: List[Entity] = field(default_factory=list)
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and self.entity is not None


@dataclass
class EventMatch:
    event: EventFrame
    matched_roles: List[str]
    score: float
    mismatched_roles: List[str] = field(default_factory=list)


class ConversationMemory:
    """Entity/event store with deterministic salience and provenance."""

    SNAPSHOT_VERSION = 6
    COMPATIBLE_SNAPSHOT_VERSIONS = {1, 2, 3, 4, 5, 6}
    NONASSERTIVE_DISCOURSE_ROLES = {
        "content", "infinitive", "interrogative", "gerund", "participle"
    }

    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.events: List[EventFrame] = []
        self.relations: List[ClauseRelation] = []
        self.modifiers: List[EntityModifierRelation] = []
        self.appositives: List[AppositiveRelation] = []
        self.contents: List[ContentRelation] = []
        self.embedded_interrogatives: List[EmbeddedInterrogativeRelation] = []
        self.infinitivals: List[InfinitivalRelation] = []
        self.gerunds: List[GerundRelation] = []
        self.turn_index: int = 0
        self.revision: int = 0
        self._entity_counter: int = 0
        self._event_counter: int = 0
        self._relation_counter: int = 0
        self._modifier_counter: int = 0
        self._appositive_counter: int = 0
        self._content_counter: int = 0
        self._embedded_interrogative_counter: int = 0
        self._infinitival_counter: int = 0
        self._gerund_counter: int = 0
        self._initialize_participants()

    def _initialize_participants(self) -> None:
        user = Entity(
            entity_id="user",
            canonical_name="user",
            kind=EntityKind.PERSON,
            gender=Gender.UNKNOWN,
            number=GrammaticalNumber.SINGULAR,
            aliases=["i", "me", "myself", "user", "you"],
            salience=1.0,
        )
        assistant = Entity(
            entity_id="assistant",
            canonical_name="assistant",
            kind=EntityKind.PERSON,
            gender=Gender.UNKNOWN,
            number=GrammaticalNumber.SINGULAR,
            aliases=["you", "yourself", "assistant", "i"],
            salience=0.5,
        )
        self.entities[user.entity_id] = user
        self.entities[assistant.entity_id] = assistant

    def begin_turn(self) -> int:
        self.turn_index += 1
        for entity in self.entities.values():
            # Salience decays but never becomes negative.
            entity.salience = max(0.0, entity.salience * 0.82)
        return self.turn_index

    # ------------------------------------------------------------------
    # Entity creation and lookup
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_alias(text: str) -> str:
        text = text.lower().replace("’", "'")
        text = re.sub(r"[^a-z0-9$€£'\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = [word for word in text.split() if word not in lexicon.ARTICLES]
        return " ".join(words)

    def _next_entity_id(self, prefix: str) -> str:
        self._entity_counter += 1
        safe = re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_") or "entity"
        return f"{safe}_{self._entity_counter}"

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_or_create_relation(
        self,
        owner_id: str,
        relation: str,
        *,
        surface: str = "",
        role_salience: float = 1.0,
    ) -> Entity:
        canonical_relation, gender, number, kind = lexicon.relation_features(relation)
        for entity in self.entities.values():
            if entity.owner_id == owner_id and entity.relation == canonical_relation:
                entity.last_mentioned_turn = self.turn_index
                entity.salience += role_salience
                if surface:
                    entity.add_alias(surface)
                entity.add_alias(canonical_relation)
                return entity

        entity_id = self._next_entity_id(canonical_relation)
        entity = Entity(
            entity_id=entity_id,
            canonical_name=canonical_relation,
            kind=kind,
            gender=gender,
            number=number,
            owner_id=owner_id,
            relation=canonical_relation,
            aliases=[canonical_relation],
            created_turn=self.turn_index,
            last_mentioned_turn=self.turn_index,
            salience=role_salience,
        )
        if surface:
            entity.add_alias(surface)
        self.entities[entity_id] = entity
        self.revision += 1
        return entity

    def get_or_create_named_entity(
        self,
        name: str,
        *,
        kind: EntityKind = EntityKind.PERSON,
        gender: Gender = Gender.UNKNOWN,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
        aliases: Iterable[str] = (),
        role_salience: float = 1.0,
    ) -> Entity:
        normalized = self.normalize_alias(name)
        existing = self.find_by_alias(normalized, expected_kind=kind)
        if existing.resolved:
            entity = existing.entity
            assert entity is not None
            entity.last_mentioned_turn = self.turn_index
            entity.salience += role_salience
            for alias in aliases:
                entity.add_alias(alias)
            return entity

        inferred_gender = gender
        if inferred_gender == Gender.UNKNOWN and kind == EntityKind.PERSON:
            inferred_gender = lexicon.infer_name_gender(normalized.split()[0] if normalized else name)
        entity_id = self._next_entity_id(normalized or kind.value)
        entity = Entity(
            entity_id=entity_id,
            canonical_name=name.strip(),
            kind=kind,
            gender=inferred_gender,
            number=number,
            aliases=[],
            created_turn=self.turn_index,
            last_mentioned_turn=self.turn_index,
            salience=role_salience,
        )
        entity.add_alias(name)
        for alias in aliases:
            entity.add_alias(alias)
        # Objects with modifiers should also match by a distinctive head noun.
        parts = normalized.split()
        if len(parts) > 1:
            entity.add_alias(parts[-1])
        self.entities[entity_id] = entity
        self.revision += 1
        return entity

    def ensure_internal_alias(self, entity_id: str) -> str:
        entity = self.entities.get(entity_id)
        if entity is None:
            raise KeyError(f"Unknown entity: {entity_id}")
        alias = "entityref-" + re.sub(r"[^a-z0-9-]+", "-", entity_id.lower()).strip("-")
        entity.add_alias(alias)
        return alias

    def get_or_create_modified_entity(
        self,
        name: str,
        modifier_signature: str,
        *,
        kind: EntityKind = EntityKind.UNKNOWN,
        gender: Gender = Gender.UNKNOWN,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
        role_salience: float = 1.0,
    ) -> Tuple[Entity, str]:
        """Create a deterministic entity identity scoped by its modifier.

        Generic restrictive descriptions must not collapse merely because they
        share a head noun.  The public alias remains available for later
        ambiguity detection; a private deterministic alias lets the current
        parser bind exactly the entity it just introduced.
        """

        normalized = self.normalize_alias(name)
        digest = hashlib.sha256(
            f"{normalized}|{modifier_signature}".encode("utf-8")
        ).hexdigest()[:16]
        identity_alias = f"modified-{digest}"
        existing = self.find_by_alias(identity_alias, expected_kind=kind)
        if existing.resolved:
            entity = existing.entity
            assert entity is not None
            entity.last_mentioned_turn = self.turn_index
            entity.salience += role_salience
            return entity, identity_alias

        entity_id = self._next_entity_id(normalized or kind.value)
        entity = Entity(
            entity_id=entity_id,
            canonical_name=name.strip(),
            kind=kind,
            gender=gender,
            number=number,
            aliases=[],
            attributes={"modifier_signature": modifier_signature},
            created_turn=self.turn_index,
            last_mentioned_turn=self.turn_index,
            salience=role_salience,
        )
        entity.add_alias(name)
        entity.add_alias(normalized)
        entity.add_alias(identity_alias)
        self.entities[entity_id] = entity
        self.revision += 1
        return entity, identity_alias

    def get_or_create_possessed_entity(
        self,
        owner_id: str,
        name: str,
        modifier_signature: str,
        *,
        kind: EntityKind = EntityKind.THING,
    ) -> Tuple[Entity, str]:
        owner = self.entities.get(owner_id)
        if owner is None:
            raise KeyError(f"Unknown owner entity: {owner_id}")
        normalized = self.normalize_alias(name)
        digest = hashlib.sha256(
            f"{owner_id}|{normalized}|{modifier_signature}".encode("utf-8")
        ).hexdigest()[:16]
        identity_alias = f"possessed-{digest}"
        existing = self.find_by_alias(identity_alias, expected_kind=kind)
        if existing.resolved:
            entity = existing.entity
            assert entity is not None
            entity.last_mentioned_turn = self.turn_index
            entity.salience += 0.8
            return entity, identity_alias

        entity_id = self._next_entity_id(normalized or "possessed")
        entity = Entity(
            entity_id=entity_id,
            canonical_name=name.strip(),
            kind=kind,
            owner_id=owner_id,
            aliases=[],
            attributes={"modifier_signature": modifier_signature},
            created_turn=self.turn_index,
            last_mentioned_turn=self.turn_index,
            salience=0.8,
        )
        entity.add_alias(name)
        entity.add_alias(identity_alias)
        self.entities[entity_id] = entity
        self.revision += 1
        return entity, identity_alias


    def bind_appositive_alias(
        self,
        head_entity_id: str,
        appositive_surface: str,
        *,
        relation_type: AppositiveRelationType,
        expected_kind: EntityKind = EntityKind.UNKNOWN,
        role_owner_id: str = "",
        role_name: str = "",
    ) -> Resolution:
        """Bind explicit apposition without collapsing an incompatible entity."""

        head = self.entities.get(head_entity_id)
        if head is None:
            return Resolution("missing", reason="appositive head entity is unknown")
        if (
            expected_kind != EntityKind.UNKNOWN
            and head.kind not in {expected_kind, EntityKind.UNKNOWN}
        ):
            return Resolution(
                "ambiguous",
                candidates=[head],
                reason="appositive type conflicts with the head entity",
            )

        normalized = self.normalize_alias(appositive_surface)
        existing = self.find_by_alias(normalized, expected_kind)
        if existing.resolved and existing.entity and existing.entity.entity_id != head_entity_id:
            return Resolution(
                "ambiguous",
                candidates=[head, existing.entity],
                reason="appositive surface already identifies a different entity",
            )
        if existing.status == "ambiguous":
            candidates = [head] + [
                entity
                for entity in existing.candidates
                if entity.entity_id != head_entity_id
            ]
            return Resolution(
                "ambiguous",
                candidates=candidates,
                reason="appositive surface has multiple existing identities",
            )

        if role_name and role_owner_id:
            role_conflicts = [
                entity
                for entity in self.entities.values()
                if entity.owner_id == role_owner_id
                and entity.relation == role_name
                and entity.entity_id != head_entity_id
            ]
            if role_conflicts:
                return Resolution(
                    "ambiguous",
                    candidates=[head] + role_conflicts,
                    reason="appositive role already belongs to another entity",
                )
            if head.owner_id not in {None, role_owner_id} or head.relation not in {None, role_name}:
                return Resolution(
                    "ambiguous",
                    candidates=[head],
                    reason="appositive role conflicts with existing relationship metadata",
                )
            head.owner_id = role_owner_id
            head.relation = role_name
            head.add_alias(role_name)
            head.add_alias(f"{role_owner_id}:{role_name}")

        head.add_alias(appositive_surface)
        head.add_alias(normalized)
        head.attributes.setdefault("appositive_type", relation_type.value)
        head.last_mentioned_turn = self.turn_index
        head.salience += 0.6
        self.revision += 1
        return Resolution("resolved", entity=head)

    def find_by_alias(self, phrase: str, expected_kind: EntityKind = EntityKind.UNKNOWN) -> Resolution:
        normalized = self.normalize_alias(phrase)
        # First/second-person pronouns are fixed deictic participant references,
        # not nominal aliases. They intentionally take precedence over generic
        # modifier-alias ambiguity (for example, two entities named "woman").
        if normalized in {"i", "me", "myself"}:
            return Resolution("resolved", self.entities["user"])
        if normalized in {"you", "yourself"}:
            return Resolution("resolved", self.entities["assistant"])
        if not normalized:
            return Resolution("missing", reason="empty reference")

        direct: List[Entity] = []
        fuzzy: List[Tuple[float, Entity]] = []
        query_tokens = set(normalized.split())
        for entity in self.entities.values():
            if expected_kind != EntityKind.UNKNOWN and entity.kind not in {expected_kind, EntityKind.UNKNOWN}:
                continue
            aliases = set(entity.aliases) | {self.normalize_alias(entity.canonical_name)}
            if normalized in aliases:
                direct.append(entity)
                continue
            for alias in aliases:
                alias_tokens = set(alias.split())
                if not alias_tokens or not query_tokens:
                    continue
                overlap = len(alias_tokens & query_tokens)
                union = len(alias_tokens | query_tokens)
                score = overlap / union
                # A head-noun match is useful for "the Honda" -> "used Honda".
                if normalized.split()[-1:] == alias.split()[-1:]:
                    score += 0.25
                if score >= 0.72:
                    fuzzy.append((score, entity))

        if len(direct) == 1:
            return Resolution("resolved", direct[0])
        if len(direct) > 1:
            ranked = sorted(direct, key=lambda entity: (entity.last_mentioned_turn, entity.salience), reverse=True)
            if all("modifier_signature" in entity.attributes for entity in ranked):
                return Resolution(
                    "ambiguous",
                    candidates=ranked,
                    reason="multiple modified entities share that generic alias",
                )
            if (
                len(ranked) == 1
                or ranked[0].last_mentioned_turn > ranked[1].last_mentioned_turn
                or ranked[0].salience - ranked[1].salience >= 1.5
            ):
                return Resolution("resolved", ranked[0])
            return Resolution("ambiguous", candidates=ranked, reason="multiple entities share that alias")
        if fuzzy:
            fuzzy.sort(key=lambda item: (item[0], item[1].last_mentioned_turn, item[1].salience), reverse=True)
            top_score, top = fuzzy[0]
            if len(fuzzy) == 1 or top_score - fuzzy[1][0] >= 0.20:
                return Resolution("resolved", top)
            return Resolution("ambiguous", candidates=[item[1] for item in fuzzy[:4]], reason="fuzzy alias collision")
        return Resolution("missing", reason=f"no entity matches {phrase!r}")

    def resolve_pronoun(self, pronoun: str, expected_kind: EntityKind = EntityKind.UNKNOWN) -> Resolution:
        p = pronoun.lower()
        features = lexicon.PRONOUN_FEATURES.get(p)
        if not features:
            return Resolution("missing", reason=f"{pronoun!r} is not a known pronoun")
        fixed_id, gender, number, kind = features
        if fixed_id:
            return Resolution("resolved", self.entities[fixed_id])

        candidates: List[Entity] = []
        for entity in self.entities.values():
            if entity.entity_id in {"user", "assistant"}:
                continue
            if expected_kind != EntityKind.UNKNOWN and entity.kind not in {expected_kind, EntityKind.UNKNOWN}:
                continue
            if kind != EntityKind.UNKNOWN and entity.kind not in {kind, EntityKind.UNKNOWN}:
                continue
            if gender in {Gender.FEMALE, Gender.MALE} and entity.gender not in {gender, Gender.UNKNOWN}:
                continue
            if number == GrammaticalNumber.PLURAL and entity.number != GrammaticalNumber.PLURAL:
                continue
            if number == GrammaticalNumber.SINGULAR and entity.number == GrammaticalNumber.PLURAL:
                continue
            candidates.append(entity)

        if not candidates:
            return Resolution("missing", reason=f"no antecedent for {pronoun!r}")

        # Coreference is primarily a discourse-recency decision; accumulated
        # salience only breaks ties within the same turn.  This prevents an old
        # high-salience subject from stealing a pronoun immediately after a new
        # compatible entity was introduced.
        candidates.sort(key=lambda entity: (entity.last_mentioned_turn, entity.salience), reverse=True)
        top = candidates[0]
        if len(candidates) == 1:
            return Resolution("resolved", top)

        second = candidates[1]
        turn_gap = top.last_mentioned_turn - second.last_mentioned_turn
        salience_gap = top.salience - second.salience
        if turn_gap >= 1 or salience_gap >= 1.5:
            return Resolution("resolved", top)
        return Resolution(
            "ambiguous",
            candidates=candidates[:4],
            reason=f"{pronoun!r} has multiple compatible antecedents",
        )

    def mention(self, entity_id: str, role: str = "other", weight: float = 1.0) -> None:
        entity = self.entities.get(entity_id)
        if not entity:
            return
        boosts = {
            "agent": 3.0,
            "subject": 3.0,
            "experiencer": 3.0,
            "patient": 2.0,
            "theme": 2.0,
            "recipient": 1.5,
            "location": 0.8,
            "other": 1.0,
        }
        entity.salience += boosts.get(role, 1.0) * max(0.0, float(weight))
        entity.last_mentioned_turn = self.turn_index

    # ------------------------------------------------------------------
    # Event storage and query
    # ------------------------------------------------------------------

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"event_{self._event_counter}"

    def add_event(self, event: EventFrame) -> EventFrame:
        stored = event.copy(
            event_id=event.event_id or self._next_event_id(),
            turn_index=event.turn_index or self.turn_index,
        )
        discourse_weight = {
            "main": 1.0,
            "coordinate": 0.85,
            "subordinate": 0.15,
        }.get(stored.discourse_role, 1.0)
        # Exact repeated assertions update certainty/provenance rather than
        # multiplying identical facts indefinitely.
        for existing in self.events:
            if (
                existing.signature() == stored.signature()
                and existing.aspect == stored.aspect
                and existing.discourse_role == stored.discourse_role
                and existing.source == stored.source
            ):
                existing.certainty = max(existing.certainty, stored.certainty)
                existing.turn_index = max(existing.turn_index, stored.turn_index)
                existing.raw_text = stored.raw_text or existing.raw_text
                for role, ref in existing.arguments.items():
                    if ref.kind == RefKind.ENTITY:
                        self.mention(ref.key, role, discourse_weight)
                self.revision += 1
                return existing

        self.events.append(stored)
        for role, ref in stored.arguments.items():
            if ref.kind == RefKind.ENTITY:
                self.mention(ref.key, role, discourse_weight)
        self.revision += 1
        return stored

    def _next_relation_id(self) -> str:
        self._relation_counter += 1
        return f"relation_{self._relation_counter}"

    def add_clause_relations(
        self,
        relations: Sequence[ClauseRelation],
        stored_events: Sequence[EventFrame],
    ) -> List[ClauseRelation]:
        """Bind parser-local relation indices to stable stored event IDs."""

        stored: List[ClauseRelation] = []
        for relation in relations:
            if not (
                0 <= relation.main_event_index < len(stored_events)
                and 0 <= relation.subordinate_event_index < len(stored_events)
            ):
                raise ValueError("Clause relation references an invalid event index")
            main_event = stored_events[relation.main_event_index]
            subordinate_event = stored_events[relation.subordinate_event_index]
            bound = relation.copy(
                relation_id=relation.relation_id or self._next_relation_id(),
                main_event_id=main_event.event_id,
                subordinate_event_id=subordinate_event.event_id,
                certainty=min(
                    relation.certainty,
                    main_event.certainty,
                    subordinate_event.certainty,
                ),
            )
            existing = next(
                (item for item in self.relations if item.signature() == bound.signature()),
                None,
            )
            if existing is not None:
                existing.certainty = max(existing.certainty, bound.certainty)
                existing.diagnostics = list(
                    dict.fromkeys(existing.diagnostics + bound.diagnostics)
                )
                stored.append(existing)
            else:
                self.relations.append(bound)
                stored.append(bound)
            self.revision += 1
        return stored

    @staticmethod
    def _event_signature_key(event: EventFrame) -> str:
        payload = repr(event.signature()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    def _next_modifier_id(self) -> str:
        self._modifier_counter += 1
        return f"modifier_{self._modifier_counter}"

    def add_entity_modifier_relations(
        self,
        modifiers: Sequence[EntityModifierRelation],
        stored_events: Sequence[EventFrame],
    ) -> List[EntityModifierRelation]:
        """Bind parser-local modifier indices to stable event identities."""

        stored: List[EntityModifierRelation] = []
        for modifier in modifiers:
            if not 0 <= modifier.modifier_event_index < len(stored_events):
                raise ValueError("Entity modifier references an invalid event index")
            if modifier.head_entity_id not in self.entities:
                raise ValueError("Entity modifier references an unknown head entity")
            event = stored_events[modifier.modifier_event_index]
            expected_signature = modifier.modifier_event_signature
            if expected_signature:
                actual_signature = self._event_signature_key(event)
                if actual_signature != expected_signature:
                    candidates = [
                        item
                        for item in stored_events
                        if self._event_signature_key(item) == expected_signature
                    ]
                    if len(candidates) != 1:
                        raise ValueError(
                            "Entity modifier event index/signature mismatch"
                        )
                    event = candidates[0]
            if event.discourse_role != "modifier":
                raise ValueError(
                    "Entity modifier must bind to a modifier-role event"
                )
            bound = modifier.copy(
                relation_id=modifier.relation_id or self._next_modifier_id(),
                modifier_event_id=event.event_id,
                certainty=min(modifier.certainty, event.certainty),
            )
            existing = next(
                (item for item in self.modifiers if item.signature() == bound.signature()),
                None,
            )
            if existing is not None:
                existing.certainty = max(existing.certainty, bound.certainty)
                existing.diagnostics = list(
                    dict.fromkeys(existing.diagnostics + bound.diagnostics)
                )
                stored.append(existing)
            else:
                self.modifiers.append(bound)
                stored.append(bound)
            self.mention(bound.head_entity_id, "modifier", 0.5)
            self.revision += 1
        return stored


    def _next_appositive_id(self) -> str:
        self._appositive_counter += 1
        return f"appositive_{self._appositive_counter}"

    def add_appositive_relations(
        self,
        appositives: Sequence[AppositiveRelation],
    ) -> List[AppositiveRelation]:
        """Store validated appositive links with stable relation identities."""

        stored: List[AppositiveRelation] = []
        for relation in appositives:
            if relation.head_entity_id not in self.entities:
                raise ValueError("Appositive relation references an unknown entity")
            bound = relation.copy(
                relation_id=relation.relation_id or self._next_appositive_id()
            )
            existing = next(
                (item for item in self.appositives if item.signature() == bound.signature()),
                None,
            )
            if existing is not None:
                existing.certainty = max(existing.certainty, bound.certainty)
                existing.diagnostics = list(
                    dict.fromkeys(existing.diagnostics + bound.diagnostics)
                )
                stored.append(existing)
            else:
                self.appositives.append(bound)
                stored.append(bound)
            self.mention(bound.head_entity_id, "modifier", 0.5)
            self.revision += 1
        return stored

    def _next_content_id(self) -> str:
        self._content_counter += 1
        return f"content_{self._content_counter}"

    def add_content_relations(
        self,
        contents: Sequence[ContentRelation],
        stored_events: Sequence[EventFrame],
    ) -> List[ContentRelation]:
        """Bind parser-local attributed-content links to stable event IDs."""

        stored: List[ContentRelation] = []
        for relation in contents:
            if not (
                0 <= relation.matrix_event_index < len(stored_events)
                and 0 <= relation.content_event_index < len(stored_events)
            ):
                raise ValueError("Content relation references an invalid event index")
            matrix_event = stored_events[relation.matrix_event_index]
            content_event = stored_events[relation.content_event_index]
            if content_event.discourse_role != "content":
                raise ValueError("Content relation must bind to a content-role event")
            if content_event.source != SourceKind.ATTRIBUTED:
                raise ValueError("Content event must retain attributed provenance")
            if matrix_event.predicate != relation.matrix_predicate:
                raise ValueError("Content relation matrix predicate does not match its event")
            if relation.source_entity_id not in self.entities:
                raise ValueError("Content relation references an unknown source entity")
            source_refs = {
                ref.key
                for role, ref in matrix_event.arguments.items()
                if role in {"agent", "experiencer", "subject", "source"}
                and ref.kind == RefKind.ENTITY
            }
            if relation.source_entity_id not in source_refs:
                raise ValueError("Content relation source is not licensed by the matrix event")
            if relation.attributed != matrix_event.polarity:
                raise ValueError("Content evidence license must match matrix polarity")
            bound = relation.copy(
                relation_id=relation.relation_id or self._next_content_id(),
                matrix_event_id=matrix_event.event_id,
                content_event_id=content_event.event_id,
                certainty=min(
                    relation.certainty,
                    matrix_event.certainty,
                    content_event.certainty,
                ),
            )
            existing = next(
                (item for item in self.contents if item.signature() == bound.signature()),
                None,
            )
            if existing is not None:
                existing.certainty = max(existing.certainty, bound.certainty)
                existing.diagnostics = list(
                    dict.fromkeys(existing.diagnostics + bound.diagnostics)
                )
                stored.append(existing)
            else:
                self.contents.append(bound)
                stored.append(bound)
            self.mention(bound.source_entity_id, "source", 0.6)
            self.revision += 1
        return stored

    def content_relations_for_event(self, event_id: str) -> List[ContentRelation]:
        return [
            relation
            for relation in self.contents
            if event_id in {relation.matrix_event_id, relation.content_event_id}
        ]

    def content_relations_for_source(
        self,
        source_entity_id: str,
        *,
        matrix_predicate: Optional[str] = None,
        include_negated: bool = False,
    ) -> List[ContentRelation]:
        relations = [
            relation
            for relation in self.contents
            if relation.source_entity_id == source_entity_id
            and (include_negated or relation.attributed)
        ]
        if matrix_predicate is not None:
            relations = [
                relation
                for relation in relations
                if relation.matrix_predicate == matrix_predicate
            ]
        return sorted(
            relations,
            key=lambda item: (
                self.get_event(item.matrix_event_id).turn_index
                if self.get_event(item.matrix_event_id) is not None
                else -1,
                item.certainty,
            ),
            reverse=True,
        )

    def _next_embedded_interrogative_id(self) -> str:
        self._embedded_interrogative_counter += 1
        return f"embedded_question_{self._embedded_interrogative_counter}"

    def add_embedded_interrogative_relations(
        self,
        relations: Sequence[EmbeddedInterrogativeRelation],
        stored_events: Sequence[EventFrame],
    ) -> List[EmbeddedInterrogativeRelation]:
        """Bind parser-local matrix/question links to stable event IDs."""

        stored: List[EmbeddedInterrogativeRelation] = []
        for relation in relations:
            if not (
                0 <= relation.matrix_event_index < len(stored_events)
                and 0 <= relation.question_event_index < len(stored_events)
            ):
                raise ValueError(
                    "Embedded interrogative relation references an invalid event index"
                )
            matrix_event = stored_events[relation.matrix_event_index]
            question_event = stored_events[relation.question_event_index]
            if question_event.discourse_role != "interrogative":
                raise ValueError(
                    "Embedded interrogative must bind to an interrogative-role event"
                )
            if question_event.source != SourceKind.ATTRIBUTED:
                raise ValueError(
                    "Embedded interrogative event must retain attributed provenance"
                )
            if matrix_event.predicate != relation.matrix_predicate:
                raise ValueError(
                    "Embedded interrogative matrix predicate does not match its event"
                )
            if relation.source_entity_id not in self.entities:
                raise ValueError(
                    "Embedded interrogative references an unknown source entity"
                )
            source_refs = {
                ref.key
                for role, ref in matrix_event.arguments.items()
                if role in {"agent", "experiencer", "subject", "source"}
                and ref.kind == RefKind.ENTITY
            }
            if relation.source_entity_id not in source_refs:
                raise ValueError(
                    "Embedded interrogative source is not licensed by the matrix event"
                )
            if relation.licensed != matrix_event.polarity:
                raise ValueError(
                    "Embedded interrogative license must match matrix polarity"
                )
            if relation.requested_role:
                variable = question_event.arguments.get(relation.requested_role)
                if variable is None or not variable.is_variable:
                    raise ValueError(
                        "Embedded WH relation must preserve its typed open slot"
                    )
            elif any(ref.is_variable for ref in question_event.arguments.values()):
                raise ValueError(
                    "Embedded polar relation cannot contain an untyped WH slot"
                )

            bound = relation.copy(
                relation_id=(
                    relation.relation_id
                    or self._next_embedded_interrogative_id()
                ),
                matrix_event_id=matrix_event.event_id,
                question_event_id=question_event.event_id,
                certainty=min(
                    relation.certainty,
                    matrix_event.certainty,
                    question_event.certainty,
                ),
            )
            existing = next(
                (
                    item
                    for item in self.embedded_interrogatives
                    if item.signature() == bound.signature()
                ),
                None,
            )
            if existing is not None:
                existing.certainty = max(existing.certainty, bound.certainty)
                existing.diagnostics = list(
                    dict.fromkeys(existing.diagnostics + bound.diagnostics)
                )
                stored.append(existing)
            else:
                self.embedded_interrogatives.append(bound)
                stored.append(bound)
            self.mention(bound.source_entity_id, "source", 0.6)
            self.revision += 1
        return stored

    def embedded_interrogatives_for_source(
        self,
        source_entity_id: str,
        *,
        matrix_predicate: Optional[str] = None,
        include_negated: bool = False,
    ) -> List[EmbeddedInterrogativeRelation]:
        relations = [
            relation
            for relation in self.embedded_interrogatives
            if relation.source_entity_id == source_entity_id
            and (include_negated or relation.licensed)
        ]
        if matrix_predicate is not None:
            normalized = lexicon.lemma(matrix_predicate)
            relations = [
                relation
                for relation in relations
                if lexicon.lemma(relation.matrix_predicate) == normalized
            ]
        return sorted(
            relations,
            key=lambda item: (
                self.get_event(item.matrix_event_id).turn_index
                if self.get_event(item.matrix_event_id) is not None
                else -1,
                item.certainty,
            ),
            reverse=True,
        )

    def embedded_interrogatives_for_event(
        self,
        event_id: str,
    ) -> List[EmbeddedInterrogativeRelation]:
        return [
            relation
            for relation in self.embedded_interrogatives
            if event_id in {
                relation.matrix_event_id,
                relation.question_event_id,
            }
        ]

    def _next_infinitival_id(self) -> str:
        self._infinitival_counter += 1
        return f"infinitival_{self._infinitival_counter}"

    def add_infinitival_relations(
        self,
        relations: Sequence[InfinitivalRelation],
        stored_events: Sequence[EventFrame],
    ) -> List[InfinitivalRelation]:
        """Bind parser-local control/raising links to stable event IDs."""

        stored: List[InfinitivalRelation] = []
        for relation in relations:
            if not (
                0 <= relation.matrix_event_index < len(stored_events)
                and 0 <= relation.complement_event_index < len(stored_events)
            ):
                raise ValueError(
                    "Infinitival relation references an invalid event index"
                )
            matrix_event = stored_events[relation.matrix_event_index]
            complement_event = stored_events[relation.complement_event_index]
            if complement_event.discourse_role != "infinitive":
                raise ValueError(
                    "Infinitival relation must bind to an infinitive-role event"
                )
            if complement_event.source != SourceKind.ATTRIBUTED:
                raise ValueError(
                    "Infinitival complement must retain attributed provenance"
                )
            if complement_event.tense != "infinitive":
                raise ValueError(
                    "Infinitival complement must retain infinitive tense"
                )
            if matrix_event.predicate != relation.matrix_predicate:
                raise ValueError(
                    "Infinitival matrix predicate does not match its event"
                )
            for entity_id, label in (
                (relation.source_entity_id, "source"),
                (relation.controller_entity_id, "controller"),
                (relation.embedded_subject_entity_id, "embedded subject"),
            ):
                if entity_id not in self.entities:
                    raise ValueError(
                        f"Infinitival relation references an unknown {label} entity"
                    )

            matrix_subjects = {
                ref.key
                for role, ref in matrix_event.arguments.items()
                if role in {"agent", "experiencer", "subject", "possessor", "patient"}
                and ref.kind == RefKind.ENTITY
            }
            if relation.source_entity_id not in matrix_subjects:
                raise ValueError(
                    "Infinitival source is not licensed by the matrix event"
                )
            if relation.relation_type == InfinitivalRelationType.OBJECT_CONTROL:
                controller_refs = {
                    ref.key
                    for role, ref in matrix_event.arguments.items()
                    if role in {"patient", "recipient"}
                    and ref.kind == RefKind.ENTITY
                }
            else:
                controller_refs = {
                    ref.key
                    for role, ref in matrix_event.arguments.items()
                    if role in {"agent", "experiencer", "subject", "possessor", "patient"}
                    and ref.kind == RefKind.ENTITY
                }
            if relation.controller_entity_id not in controller_refs:
                raise ValueError(
                    "Infinitival controller is not licensed by the matrix event"
                )
            embedded_subjects = {
                ref.key
                for role, ref in complement_event.arguments.items()
                if role in {"agent", "experiencer", "subject", "possessor", "patient"}
                and ref.kind == RefKind.ENTITY
            }
            if relation.embedded_subject_entity_id not in embedded_subjects:
                raise ValueError(
                    "Infinitival embedded subject does not match its complement"
                )
            if relation.controller_entity_id != relation.embedded_subject_entity_id:
                raise ValueError(
                    "Infinitival controller must bind the embedded subject"
                )
            if relation.licensed != matrix_event.polarity:
                raise ValueError(
                    "Infinitival content license must match matrix polarity"
                )
            if relation.entailed:
                raise ValueError(
                    "This infinitival slice does not license accomplished-event entailment"
                )

            bound = relation.copy(
                relation_id=relation.relation_id or self._next_infinitival_id(),
                matrix_event_id=matrix_event.event_id,
                complement_event_id=complement_event.event_id,
                certainty=min(
                    relation.certainty,
                    matrix_event.certainty,
                    complement_event.certainty,
                ),
            )
            existing = next(
                (
                    item
                    for item in self.infinitivals
                    if item.signature() == bound.signature()
                ),
                None,
            )
            if existing is not None:
                existing.certainty = max(existing.certainty, bound.certainty)
                existing.diagnostics = list(
                    dict.fromkeys(existing.diagnostics + bound.diagnostics)
                )
                stored.append(existing)
            else:
                self.infinitivals.append(bound)
                stored.append(bound)
            self.mention(bound.source_entity_id, "source", 0.4)
            self.mention(bound.controller_entity_id, "controller", 0.5)
            self.revision += 1
        return stored

    def infinitival_relations_for_event(
        self,
        event_id: str,
    ) -> List[InfinitivalRelation]:
        return [
            relation
            for relation in self.infinitivals
            if event_id
            in {relation.matrix_event_id, relation.complement_event_id}
        ]

    def infinitival_relations_for_source(
        self,
        source_entity_id: str,
        *,
        matrix_predicate: Optional[str] = None,
        include_negated: bool = False,
    ) -> List[InfinitivalRelation]:
        relations = [
            relation
            for relation in self.infinitivals
            if relation.source_entity_id == source_entity_id
            and (include_negated or relation.licensed)
        ]
        if matrix_predicate is not None:
            relations = [
                relation
                for relation in relations
                if relation.matrix_predicate == matrix_predicate
            ]
        return sorted(
            relations,
            key=lambda item: (
                self.get_event(item.matrix_event_id).turn_index
                if self.get_event(item.matrix_event_id) is not None
                else -1,
                item.certainty,
            ),
            reverse=True,
        )

    def _next_gerund_id(self) -> str:
        self._gerund_counter += 1
        return f"gerund_{self._gerund_counter}"

    _GERUND_PREDICATE_CATALOG = {
        "enjoy": (
            GerundRelationType.GERUND_CONTENT,
            GerundContentStatus.ENJOYED,
            False,
        ),
        "avoid": (
            GerundRelationType.GERUND_CONTENT,
            GerundContentStatus.AVOIDED,
            False,
        ),
        "start": (
            GerundRelationType.ASPECTUAL_START,
            GerundContentStatus.BEGUN,
            True,
        ),
        "begin": (
            GerundRelationType.ASPECTUAL_START,
            GerundContentStatus.BEGUN,
            True,
        ),
        "stop": (
            GerundRelationType.ASPECTUAL_STOP,
            GerundContentStatus.STOPPED,
            True,
        ),
        "keep": (
            GerundRelationType.ASPECTUAL_CONTINUATION,
            GerundContentStatus.CONTINUED,
            True,
        ),
        "continue": (
            GerundRelationType.ASPECTUAL_CONTINUATION,
            GerundContentStatus.CONTINUED,
            True,
        ),
        "see": (
            GerundRelationType.PERCEPTION_PARTICIPIAL,
            GerundContentStatus.PERCEIVED,
            False,
        ),
        "watch": (
            GerundRelationType.PERCEPTION_PARTICIPIAL,
            GerundContentStatus.PERCEIVED,
            False,
        ),
        "hear": (
            GerundRelationType.PERCEPTION_PARTICIPIAL,
            GerundContentStatus.PERCEIVED,
            False,
        ),
        "notice": (
            GerundRelationType.PERCEPTION_PARTICIPIAL,
            GerundContentStatus.PERCEIVED,
            False,
        ),
    }
    _GERUND_PREDICATE_FAMILIES = {
        "enjoy": "enjoyment",
        "avoid": "avoidance",
        "start": "aspectual_onset",
        "begin": "aspectual_onset",
        "stop": "aspectual_cessation",
        "keep": "aspectual_continuation",
        "continue": "aspectual_continuation",
        "see": "visual_perception",
        "watch": "visual_perception",
        "hear": "auditory_perception",
        "notice": "perception",
    }

    @staticmethod
    def _has_forward_deictic_time(event: EventFrame) -> bool:
        time_ref = event.arguments.get("time")
        if time_ref is None:
            return False
        tokens = set(re.findall(r"[a-z0-9]+", time_ref.key.lower()))
        return bool(tokens & {"tomorrow", "later", "next"})

    def _validate_gerund_binding(
        self,
        relation: GerundRelation,
        matrix_event: EventFrame,
        complement_event: EventFrame,
    ) -> None:
        expected = self._GERUND_PREDICATE_CATALOG.get(matrix_event.predicate)
        if expected is None:
            raise ValueError("Gerund matrix predicate is outside the reviewed catalog")
        relation_type, content_status, phase_entailing = expected
        if (
            relation.relation_type != relation_type
            or relation.content_status != content_status
        ):
            raise ValueError("Gerund relation type/status does not match its matrix predicate")
        if relation.marker != "-ing":
            raise ValueError("Gerund relation must retain the -ing marker")
        expected_role = (
            "participle"
            if relation_type == GerundRelationType.PERCEPTION_PARTICIPIAL
            else "gerund"
        )
        if complement_event.discourse_role != expected_role:
            raise ValueError(
                "Gerund relation does not match its complement discourse role"
            )
        if complement_event.source != SourceKind.ATTRIBUTED:
            raise ValueError("Gerund complement must retain attributed provenance")
        if complement_event.tense != "nonfinite":
            raise ValueError("Gerund complement must retain nonfinite tense")
        if complement_event.aspect != expected_role:
            raise ValueError("Gerund relation does not match its nonfinite aspect")
        if matrix_event.predicate != relation.matrix_predicate:
            raise ValueError("Gerund matrix predicate does not match its event")
        if relation.predicate_family != self._GERUND_PREDICATE_FAMILIES.get(
            matrix_event.predicate
        ):
            raise ValueError("Gerund predicate family does not match its matrix predicate")
        if relation.matrix_event_index < 0 or relation.complement_event_index < 0:
            raise ValueError("Gerund relation has invalid local event indices")
        if relation.matrix_event_index == relation.complement_event_index:
            raise ValueError("Gerund relation cannot bind an event to itself")
        if relation.certainty > min(matrix_event.certainty, complement_event.certainty):
            raise ValueError("Gerund relation certainty exceeds its supporting events")
        for entity_id, label in (
            (relation.source_entity_id, "source"),
            (relation.controller_entity_id, "controller"),
            (relation.embedded_subject_entity_id, "embedded subject"),
        ):
            if entity_id not in self.entities:
                raise ValueError(
                    f"Gerund relation references an unknown {label} entity"
                )
        source_refs = {
            ref.key
            for role, ref in matrix_event.arguments.items()
            if role in {"agent", "experiencer", "subject", "possessor"}
            and ref.kind == RefKind.ENTITY
        }
        if relation.source_entity_id not in source_refs:
            raise ValueError("Gerund source is not licensed by the matrix event")
        if relation_type == GerundRelationType.PERCEPTION_PARTICIPIAL:
            controller_refs = {
                ref.key
                for role, ref in matrix_event.arguments.items()
                if role in {"patient", "recipient"}
                and ref.kind == RefKind.ENTITY
            }
        else:
            controller_refs = source_refs
        if relation.controller_entity_id not in controller_refs:
            raise ValueError("Gerund controller is not licensed by the matrix event")
        embedded_subjects = {
            ref.key
            for role, ref in complement_event.arguments.items()
            if role in {"agent", "experiencer", "subject", "possessor", "patient"}
            and ref.kind == RefKind.ENTITY
        }
        if relation.embedded_subject_entity_id not in embedded_subjects:
            raise ValueError("Gerund embedded subject does not match its complement")
        if relation.controller_entity_id != relation.embedded_subject_entity_id:
            raise ValueError("Gerund controller must bind the embedded subject")
        if relation.licensed != matrix_event.polarity:
            raise ValueError("Gerund content license must match matrix polarity")
        factual_matrix = (
            matrix_event.modality is None
            and matrix_event.tense != "future"
            and matrix_event.aspect in {"simple", "perfect"}
            and not self._has_forward_deictic_time(matrix_event)
            and not self._has_forward_deictic_time(complement_event)
        )
        expected_entailment = bool(
            relation.licensed and phase_entailing and factual_matrix
        )
        if relation.entailed != expected_entailment:
            raise ValueError(
                "Gerund phase entailment does not match matrix factuality"
            )

    def add_gerund_relations(
        self,
        relations: Sequence[GerundRelation],
        stored_events: Sequence[EventFrame],
    ) -> List[GerundRelation]:
        """Bind selected ``-ing`` links while preserving truth boundaries."""

        stored: List[GerundRelation] = []
        for relation in relations:
            if not (
                0 <= relation.matrix_event_index < len(stored_events)
                and 0 <= relation.complement_event_index < len(stored_events)
            ):
                raise ValueError("Gerund relation references an invalid event index")
            matrix_event = stored_events[relation.matrix_event_index]
            complement_event = stored_events[relation.complement_event_index]
            candidate = relation.copy(
                matrix_event_id=matrix_event.event_id,
                complement_event_id=complement_event.event_id,
                certainty=min(
                    relation.certainty,
                    matrix_event.certainty,
                    complement_event.certainty,
                ),
            )
            self._validate_gerund_binding(
                candidate,
                matrix_event,
                complement_event,
            )
            existing = next(
                (
                    item
                    for item in self.gerunds
                    if item.signature() == candidate.signature()
                ),
                None,
            )
            if existing is not None:
                existing.certainty = max(existing.certainty, candidate.certainty)
                existing.diagnostics = list(
                    dict.fromkeys(existing.diagnostics + candidate.diagnostics)
                )
                stored.append(existing)
            else:
                relation_id = candidate.relation_id
                if relation_id:
                    match = re.fullmatch(r"gerund_([1-9][0-9]*)", relation_id)
                    if match is None:
                        raise ValueError("Gerund relation has an invalid relation ID")
                    if any(item.relation_id == relation_id for item in self.gerunds):
                        raise ValueError("Gerund relation ID is already in use")
                    self._gerund_counter = max(
                        self._gerund_counter,
                        int(match.group(1)),
                    )
                else:
                    relation_id = self._next_gerund_id()
                bound = candidate.copy(relation_id=relation_id)
                self.gerunds.append(bound)
                stored.append(bound)
            stored_relation = stored[-1]
            self.mention(stored_relation.source_entity_id, "source", 0.4)
            self.mention(stored_relation.controller_entity_id, "controller", 0.5)
            self.revision += 1
        return stored

    def gerund_relations_for_event(self, event_id: str) -> List[GerundRelation]:
        return [
            relation
            for relation in self.gerunds
            if event_id in {relation.matrix_event_id, relation.complement_event_id}
        ]

    def gerund_relations_for_source(
        self,
        source_entity_id: str,
        *,
        matrix_predicate: Optional[str] = None,
        include_negated: bool = False,
    ) -> List[GerundRelation]:
        relations = [
            relation
            for relation in self.gerunds
            if relation.source_entity_id == source_entity_id
            and (include_negated or relation.licensed)
        ]
        if matrix_predicate is not None:
            relations = [
                relation
                for relation in relations
                if relation.matrix_predicate == matrix_predicate
            ]
        return sorted(
            relations,
            key=lambda item: (
                self.get_event(item.matrix_event_id).turn_index
                if self.get_event(item.matrix_event_id) is not None
                else -1,
                item.certainty,
            ),
            reverse=True,
        )

    def get_event(self, event_id: str) -> Optional[EventFrame]:
        return next((event for event in self.events if event.event_id == event_id), None)

    def appositives_for_entity(self, entity_id: str) -> List[AppositiveRelation]:
        return [
            relation
            for relation in self.appositives
            if relation.head_entity_id == entity_id
        ]

    def modifiers_for_entity(self, entity_id: str) -> List[EntityModifierRelation]:
        return [
            modifier
            for modifier in self.modifiers
            if modifier.head_entity_id == entity_id
        ]

    def relations_for_event(self, event_id: str) -> List[ClauseRelation]:
        return [
            relation
            for relation in self.relations
            if event_id in {relation.main_event_id, relation.subordinate_event_id}
        ]

    def relation_between(
        self,
        main_event_id: str,
        subordinate_event_id: str,
    ) -> Optional[ClauseRelation]:
        return next(
            (
                relation
                for relation in self.relations
                if relation.main_event_id == main_event_id
                and relation.subordinate_event_id == subordinate_event_id
            ),
            None,
        )

    @staticmethod
    def refs_equal(left: SemanticRef, right: SemanticRef) -> bool:
        if left.is_variable or right.is_variable:
            return True
        if left.kind == RefKind.ENTITY and right.kind == RefKind.ENTITY:
            return left.key == right.key
        if left.kind == right.kind and left.key == right.key:
            return True
        # Literal comparison is normalized and allows exact numeric strings.
        return left.key.lower().strip() == right.key.lower().strip()

    def match_events(
        self,
        query: EventFrame,
        *,
        ignore_roles: Iterable[str] = (),
        allow_tense_variation: bool = True,
        include_opposite_polarity: bool = True,
        include_attributed_content: bool = False,
        include_infinitival_content: bool = False,
        include_interrogative_content: bool = False,
        include_gerund_content: bool = False,
    ) -> List[EventMatch]:
        ignored = set(ignore_roles)
        matches: List[EventMatch] = []
        fixed = {role: ref for role, ref in query.arguments.items() if role not in ignored and not ref.is_variable}
        for event in self.events:
            if (
                event.discourse_role == "content"
                and not include_attributed_content
            ):
                continue
            if (
                event.discourse_role == "infinitive"
                and not include_infinitival_content
            ):
                continue
            if (
                event.discourse_role == "interrogative"
                and not include_interrogative_content
            ):
                continue
            if (
                event.discourse_role in {"gerund", "participle"}
                and not include_gerund_content
            ):
                continue
            if event.predicate != query.predicate:
                continue
            # Aspect is proposition-bearing: a habitual/simple assertion is
            # not evidence for an event currently in progress (or vice
            # versa), and neither licenses perfect/perfect-progressive forms.
            if (
                event.aspect != query.aspect
                and event.discourse_role not in self.NONASSERTIVE_DISCOURSE_ROLES
            ):
                continue
            if not allow_tense_variation and event.tense != query.tense:
                continue
            if not include_opposite_polarity and event.polarity != query.polarity:
                continue
            matched_roles: List[str] = []
            mismatched_roles: List[str] = []
            for role, expected in fixed.items():
                actual = event.arguments.get(role)
                if actual is None or not self.refs_equal(expected, actual):
                    mismatched_roles.append(role)
                else:
                    matched_roles.append(role)
            if mismatched_roles:
                continue
            score = len(matched_roles) * 10.0
            if event.tense == query.tense:
                score += 2.0
            if event.polarity == query.polarity:
                score += 3.0
            score += min(2.0, event.certainty / 128.0)
            score += min(1.0, event.turn_index / max(1, self.turn_index))
            matches.append(EventMatch(event, matched_roles, score, mismatched_roles))
        matches.sort(key=lambda item: (item.score, item.event.turn_index, item.event.certainty), reverse=True)
        return matches

    def related_events(self, query: EventFrame, requested_role: Optional[str]) -> List[EventMatch]:
        """Return events matching all fixed anchors except the open slot."""

        ignored: Set[str] = set()
        if requested_role:
            ignored.add(requested_role)
        return self.match_events(
            query,
            ignore_roles=ignored,
            include_opposite_polarity=True,
            include_attributed_content=False,
        )

    def latest_event(self) -> Optional[EventFrame]:
        ordinary = [
            event
            for event in self.events
            if event.discourse_role not in self.NONASSERTIVE_DISCOURSE_ROLES
        ]
        return max(ordinary, key=lambda event: event.turn_index, default=None)

    # ------------------------------------------------------------------
    # Snapshot persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_version": self.SNAPSHOT_VERSION,
            "turn_index": self.turn_index,
            "revision": self.revision,
            "entity_counter": self._entity_counter,
            "event_counter": self._event_counter,
            "relation_counter": self._relation_counter,
            "modifier_counter": self._modifier_counter,
            "appositive_counter": self._appositive_counter,
            "content_counter": self._content_counter,
            "embedded_interrogative_counter": self._embedded_interrogative_counter,
            "infinitival_counter": self._infinitival_counter,
            "gerund_counter": self._gerund_counter,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "events": [event.to_dict() for event in self.events],
            "relations": [relation.to_dict() for relation in self.relations],
            "modifiers": [modifier.to_dict() for modifier in self.modifiers],
            "appositives": [relation.to_dict() for relation in self.appositives],
            "contents": [relation.to_dict() for relation in self.contents],
            "embedded_interrogatives": [
                relation.to_dict()
                for relation in self.embedded_interrogatives
            ],
            "infinitivals": [
                relation.to_dict() for relation in self.infinitivals
            ],
            "gerunds": [relation.to_dict() for relation in self.gerunds],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversationMemory":
        version = int(data.get("snapshot_version", 0))
        if version not in cls.COMPATIBLE_SNAPSHOT_VERSIONS:
            raise ValueError(f"Unsupported memory snapshot version: {version}")
        memory = cls()
        memory.entities = {
            entity.entity_id: entity
            for entity in (Entity.from_dict(item) for item in data.get("entities", []))
        }
        if "user" not in memory.entities or "assistant" not in memory.entities:
            memory._initialize_participants()
        memory.events = [EventFrame.from_dict(item) for item in data.get("events", [])]
        memory.relations = [
            ClauseRelation.from_dict(item)
            for item in data.get("relations", [])
        ]
        memory.modifiers = [
            EntityModifierRelation.from_dict(item)
            for item in data.get("modifiers", [])
        ]
        memory.appositives = [
            AppositiveRelation.from_dict(item)
            for item in data.get("appositives", [])
        ]
        memory.contents = [
            ContentRelation.from_dict(item)
            for item in data.get("contents", [])
        ]
        memory.embedded_interrogatives = [
            EmbeddedInterrogativeRelation.from_dict(item)
            for item in data.get("embedded_interrogatives", [])
        ]
        memory.infinitivals = [
            InfinitivalRelation.from_dict(item)
            for item in data.get("infinitivals", [])
        ]
        memory.gerunds = [
            GerundRelation.from_dict(item)
            for item in data.get("gerunds", [])
        ]
        gerund_ids: set[str] = set()
        max_gerund_index = 0
        for relation in memory.gerunds:
            match = re.fullmatch(r"gerund_([1-9][0-9]*)", relation.relation_id)
            if match is None:
                raise ValueError("Gerund snapshot relation has an invalid relation ID")
            if relation.relation_id in gerund_ids:
                raise ValueError("Gerund snapshot contains duplicate relation IDs")
            gerund_ids.add(relation.relation_id)
            max_gerund_index = max(max_gerund_index, int(match.group(1)))
        for relation in memory.gerunds:
            matrix_event = memory.get_event(relation.matrix_event_id)
            complement_event = memory.get_event(relation.complement_event_id)
            if matrix_event is None or complement_event is None:
                raise ValueError("Gerund snapshot relation references an unknown event")
            memory._validate_gerund_binding(
                relation,
                matrix_event,
                complement_event,
            )
        memory.turn_index = int(data.get("turn_index", 0))
        memory.revision = int(data.get("revision", 0))
        memory._entity_counter = int(data.get("entity_counter", len(memory.entities)))
        memory._event_counter = int(data.get("event_counter", len(memory.events)))
        memory._relation_counter = int(
            data.get("relation_counter", len(memory.relations))
        )
        memory._modifier_counter = int(
            data.get("modifier_counter", len(memory.modifiers))
        )
        memory._appositive_counter = int(
            data.get("appositive_counter", len(memory.appositives))
        )
        memory._content_counter = int(
            data.get("content_counter", len(memory.contents))
        )
        memory._embedded_interrogative_counter = int(
            data.get(
                "embedded_interrogative_counter",
                len(memory.embedded_interrogatives),
            )
        )
        memory._infinitival_counter = int(
            data.get("infinitival_counter", len(memory.infinitivals))
        )
        raw_gerund_counter = data.get("gerund_counter", len(memory.gerunds))
        if isinstance(raw_gerund_counter, bool):
            raise ValueError("Gerund snapshot counter must be an integer")
        memory._gerund_counter = int(raw_gerund_counter)
        if memory._gerund_counter < max_gerund_index:
            raise ValueError("Gerund snapshot counter is below its relation IDs")
        return memory

    def dumps(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def loads(cls, text: str) -> "ConversationMemory":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConversationMemory":
        return cls.loads(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Human-readable labels used in probes and traces
    # ------------------------------------------------------------------

    def describe_entity(self, entity_id: str, *, perspective: str = "assistant") -> str:
        entity = self.entities.get(entity_id)
        if not entity:
            return entity_id
        if entity.entity_id == "user":
            return "you" if perspective == "assistant" else "I"
        if entity.entity_id == "assistant":
            return "I" if perspective == "assistant" else "you"
        if entity.relation and entity.owner_id:
            owner = "your" if entity.owner_id == "user" and perspective == "assistant" else "my"
            if entity.owner_id == "assistant" and perspective == "assistant":
                owner = "my"
            elif entity.owner_id == "assistant" and perspective != "assistant":
                owner = "your"
            return f"{owner} {entity.relation}"
        return entity.canonical_name

    def unresolved_from_resolution(self, surface: str, resolution: Resolution, expected_kind: EntityKind) -> UnresolvedReference:
        return UnresolvedReference(
            surface=surface,
            reason=resolution.reason or resolution.status,
            candidates=[entity.entity_id for entity in resolution.candidates],
            expected_kind=expected_kind,
        )
