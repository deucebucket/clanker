"""Persistent symbolic conversation memory for Clanker-LM.

The memory stores compact entities and event frames rather than replaying a text
context window.  Pronoun resolution, evidence lookup, conflict detection, and
JSON snapshots all operate on this state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import lexicon
from .model import (
    ClauseRelation,
    Entity,
    EntityKind,
    EventFrame,
    Gender,
    GrammaticalNumber,
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

    SNAPSHOT_VERSION = 2
    COMPATIBLE_SNAPSHOT_VERSIONS = {1, 2}

    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.events: List[EventFrame] = []
        self.relations: List[ClauseRelation] = []
        self.turn_index: int = 0
        self.revision: int = 0
        self._entity_counter: int = 0
        self._event_counter: int = 0
        self._relation_counter: int = 0
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

    def find_by_alias(self, phrase: str, expected_kind: EntityKind = EntityKind.UNKNOWN) -> Resolution:
        normalized = self.normalize_alias(phrase)
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
            if existing.proposition_signature() == stored.proposition_signature():
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
    ) -> List[EventMatch]:
        ignored = set(ignore_roles)
        matches: List[EventMatch] = []
        fixed = {role: ref for role, ref in query.arguments.items() if role not in ignored and not ref.is_variable}
        for event in self.events:
            if event.predicate != query.predicate:
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
        return self.match_events(query, ignore_roles=ignored, include_opposite_polarity=True)

    def latest_event(self) -> Optional[EventFrame]:
        return max(self.events, key=lambda event: event.turn_index, default=None)

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
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "events": [event.to_dict() for event in self.events],
            "relations": [relation.to_dict() for relation in self.relations],
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
        memory.turn_index = int(data.get("turn_index", 0))
        memory.revision = int(data.get("revision", 0))
        memory._entity_counter = int(data.get("entity_counter", len(memory.entities)))
        memory._event_counter = int(data.get("event_counter", len(memory.events)))
        memory._relation_counter = int(
            data.get("relation_counter", len(memory.relations))
        )
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
