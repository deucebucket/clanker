"""Persistent symbolic entity, event, and emotional session state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import (
    AffectVector,
    Entity,
    EntityKind,
    Fact,
    Gender,
    GrammaticalNumber,
    QuestionFrame,
    RoleValue,
    SemanticRole,
    ValueKind,
)
from .normalize import (
    DETERMINERS,
    FEMALE_PRONOUNS,
    FIRST_PERSON,
    MALE_PRONOUNS,
    OBJECT_PRONOUNS,
    PLURAL_PRONOUNS,
    SECOND_PERSON,
    choose_article,
    normalize_alias,
)


@dataclass(frozen=True)
class Resolution:
    entity_id: Optional[str]
    candidates: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.entity_id is not None


class ConversationMemory:
    """Compact session memory; no previous raw prompt replay is required."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.turn = 0
        self.entities: Dict[str, Entity] = {}
        self.facts: List[Fact] = []
        self.running_state = AffectVector()
        self._entity_counter = 0
        self._fact_counter = 0
        self._install_special_entities()

    def _install_special_entities(self) -> None:
        self.entities["user"] = Entity(
            entity_id="user",
            canonical_name="user",
            display_name="you",
            kind=EntityKind.PERSON,
            gender=Gender.UNKNOWN,
            number=GrammaticalNumber.SINGULAR,
            aliases={"i", "me", "myself", "user", "speaker"},
            salience=2.0,
            metadata={"special": "user"},
        )
        self.entities["assistant"] = Entity(
            entity_id="assistant",
            canonical_name="assistant",
            display_name="I",
            kind=EntityKind.PERSON,
            gender=Gender.UNKNOWN,
            number=GrammaticalNumber.SINGULAR,
            aliases={"you", "assistant", "listener"},
            salience=1.0,
            metadata={"special": "assistant"},
        )

    def begin_turn(self) -> int:
        self.turn += 1
        for entity in self.entities.values():
            if entity.entity_id in {"user", "assistant"}:
                continue
            entity.salience *= 0.72
        return self.turn

    def _next_entity_id(self) -> str:
        self._entity_counter += 1
        return f"e{self._entity_counter:05d}"

    def _next_fact_id(self) -> str:
        self._fact_counter += 1
        return f"f{self._fact_counter:05d}"

    def upsert_entity(
        self,
        canonical_name: str,
        display_name: Optional[str] = None,
        *,
        kind: EntityKind = EntityKind.UNKNOWN,
        gender: Gender = Gender.UNKNOWN,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
        relation_to_user: Optional[str] = None,
        determiner: Optional[str] = None,
        aliases: Iterable[str] = (),
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Entity:
        canonical = normalize_alias(canonical_name) or canonical_name.lower().strip()
        alias_set = {normalize_alias(alias) for alias in aliases if normalize_alias(alias)}
        alias_set.add(canonical)
        if display_name:
            alias_set.add(normalize_alias(display_name))
        head = canonical.split()[-1] if canonical.split() else canonical
        if head:
            alias_set.add(head)

        existing: Optional[Entity] = None
        if relation_to_user:
            for entity in self.entities.values():
                if entity.relation_to_user == relation_to_user:
                    existing = entity
                    break
        if existing is None:
            candidates = self._entities_matching_aliases(alias_set, include_special=False)
            if len(candidates) == 1:
                existing = candidates[0]

        if existing is not None:
            existing.aliases.update(alias_set)
            if existing.kind == EntityKind.UNKNOWN and kind != EntityKind.UNKNOWN:
                existing.kind = kind
            if existing.gender == Gender.UNKNOWN and gender != Gender.UNKNOWN:
                existing.gender = gender
            if existing.number == GrammaticalNumber.UNKNOWN:
                existing.number = number
            if relation_to_user and not existing.relation_to_user:
                existing.relation_to_user = relation_to_user
            if determiner and not existing.determiner:
                existing.determiner = determiner
            if display_name and (
                existing.display_name == existing.canonical_name
                or existing.display_name.lower() == existing.canonical_name.lower()
            ):
                existing.display_name = display_name
            if metadata:
                existing.metadata.update({str(k): str(v) for k, v in metadata.items()})
            self.mention(existing.entity_id)
            return existing

        entity = Entity(
            entity_id=self._next_entity_id(),
            canonical_name=canonical,
            display_name=display_name or canonical_name.strip(),
            kind=kind,
            gender=gender,
            number=number,
            relation_to_user=relation_to_user,
            determiner=determiner,
            aliases=alias_set,
            salience=1.0,
            last_turn=self.turn,
            metadata={str(k): str(v) for k, v in (metadata or {}).items()},
        )
        self.entities[entity.entity_id] = entity
        return entity

    def mention(self, entity_id: str, amount: float = 1.0) -> None:
        entity = self.entities.get(entity_id)
        if entity is None:
            return
        entity.salience = min(10.0, entity.salience + amount)
        entity.last_turn = self.turn

    def _entities_matching_aliases(
        self, aliases: Iterable[str], *, include_special: bool = True
    ) -> List[Entity]:
        normalized = {normalize_alias(alias) for alias in aliases if normalize_alias(alias)}
        matches: List[Entity] = []
        for entity in self.entities.values():
            if not include_special and entity.entity_id in {"user", "assistant"}:
                continue
            entity_aliases = {normalize_alias(item) for item in entity.aliases}
            entity_aliases.add(normalize_alias(entity.canonical_name))
            entity_aliases.add(normalize_alias(entity.display_name))
            if normalized & entity_aliases:
                matches.append(entity)
        matches.sort(key=lambda item: (-item.salience, -item.last_turn, item.entity_id))
        return matches

    def resolve_alias(
        self, text: str, *, expected_kind: Optional[EntityKind] = None
    ) -> Resolution:
        alias = normalize_alias(text)
        if not alias:
            return Resolution(None, (), "empty reference")
        matches = self._entities_matching_aliases({alias})
        if expected_kind is not None and expected_kind != EntityKind.UNKNOWN:
            matches = [
                item
                for item in matches
                if item.kind in {expected_kind, EntityKind.UNKNOWN}
            ]
        if not matches:
            return Resolution(None, (), f"no entity matching {text!r}")
        if len(matches) == 1:
            self.mention(matches[0].entity_id)
            return Resolution(matches[0].entity_id)
        first, second = matches[0], matches[1]
        if first.salience - second.salience >= 0.5 or first.last_turn > second.last_turn:
            self.mention(first.entity_id)
            return Resolution(first.entity_id)
        return Resolution(
            None,
            tuple(item.entity_id for item in matches),
            f"ambiguous reference {text!r}",
        )

    def resolve_pronoun(self, pronoun: str) -> Resolution:
        word = pronoun.lower().strip()
        if word in FIRST_PERSON:
            return Resolution("user")
        if word in SECOND_PERSON:
            return Resolution("assistant")

        candidates = [
            item
            for item in self.entities.values()
            if item.entity_id not in {"user", "assistant"}
        ]
        if word in FEMALE_PRONOUNS:
            candidates = [
                item
                for item in candidates
                if item.kind == EntityKind.PERSON and item.gender == Gender.FEMALE
            ]
        elif word in MALE_PRONOUNS:
            candidates = [
                item
                for item in candidates
                if item.kind == EntityKind.PERSON and item.gender == Gender.MALE
            ]
        elif word in PLURAL_PRONOUNS:
            candidates = [
                item
                for item in candidates
                if item.number == GrammaticalNumber.PLURAL
            ]
        elif word in OBJECT_PRONOUNS:
            candidates = [
                item
                for item in candidates
                if item.kind not in {EntityKind.PERSON, EntityKind.TIME}
            ]
        else:
            return self.resolve_alias(word)

        candidates.sort(key=lambda item: (-item.last_turn, -item.salience, item.entity_id))
        if not candidates:
            return Resolution(None, (), f"no compatible antecedent for {word!r}")
        if len(candidates) == 1:
            self.mention(candidates[0].entity_id)
            return Resolution(candidates[0].entity_id)

        first, second = candidates[0], candidates[1]
        if first.last_turn > second.last_turn or first.salience - second.salience >= 0.75:
            self.mention(first.entity_id)
            return Resolution(first.entity_id)
        return Resolution(
            None,
            tuple(item.entity_id for item in candidates),
            f"multiple compatible antecedents for {word!r}",
        )

    def entity_value(self, entity_id: str, display: Optional[str] = None) -> RoleValue:
        entity = self.entities[entity_id]
        return RoleValue(
            kind=ValueKind.ENTITY,
            value=entity_id,
            display=display or entity.display_name,
        )

    @staticmethod
    def text_value(text: str, *, kind: ValueKind = ValueKind.TEXT) -> RoleValue:
        normalized = normalize_alias(text) if kind != ValueKind.TIME else text.lower().strip()
        return RoleValue(kind=kind, value=normalized, display=text.strip())

    def remember_fact(
        self,
        frame,
        *,
        provenance,
        certainty: int = 230,
    ) -> Fact:
        for existing in self.facts:
            if self._frames_equivalent(existing.frame, frame):
                existing.certainty = max(existing.certainty, certainty)
                existing.turn_id = self.turn
                return existing
        fact = Fact(
            fact_id=self._next_fact_id(),
            frame=frame,
            provenance=provenance,
            certainty=max(0, min(255, int(certainty))),
            turn_id=self.turn,
        )
        self.facts.append(fact)
        for value in frame.roles.values():
            if value.kind == ValueKind.ENTITY:
                self.mention(value.value, 0.5)
        return fact

    def _frames_equivalent(self, left, right) -> bool:
        if (
            left.predicate != right.predicate
            or left.polarity != right.polarity
            or left.tense != right.tense
        ):
            return False
        if set(left.roles) != set(right.roles):
            return False
        return all(self.role_values_match(left.roles[role], right.roles[role]) for role in left.roles)

    def role_values_match(self, left: RoleValue, right: RoleValue) -> bool:
        if left.kind == ValueKind.ENTITY and right.kind == ValueKind.ENTITY:
            if left.value == right.value:
                return True
            left_entity = self.entities.get(left.value)
            right_entity = self.entities.get(right.value)
            if left_entity is None or right_entity is None:
                return False
            left_aliases = {normalize_alias(item) for item in left_entity.aliases}
            right_aliases = {normalize_alias(item) for item in right_entity.aliases}
            return bool(left_aliases & right_aliases)
        if left.kind == ValueKind.ENTITY or right.kind == ValueKind.ENTITY:
            entity_value = left if left.kind == ValueKind.ENTITY else right
            literal_value = right if left.kind == ValueKind.ENTITY else left
            entity = self.entities.get(entity_value.value)
            if entity is None:
                return False
            aliases = {normalize_alias(item) for item in entity.aliases}
            aliases.add(normalize_alias(entity.canonical_name))
            aliases.add(normalize_alias(entity.display_name))
            return normalize_alias(literal_value.display) in aliases
        if left.kind == ValueKind.TIME or right.kind == ValueKind.TIME:
            return left.value.lower().strip() == right.value.lower().strip()
        return normalize_alias(left.value) == normalize_alias(right.value)

    def query(
        self,
        question: QuestionFrame,
        *,
        include_opposing_polarity: bool = False,
    ) -> Tuple[Fact, ...]:
        """Return facts compatible with the known portion of a question.

        Open-slot questions may only bind facts that preserve tense, modality,
        and polarity.  A negative fact must never answer a positive ``what``
        question, and a past event must never silently answer a future one.

        Polar questions are the deliberate exception for polarity: both the
        proposition and its explicit negation are retrieved so the answer
        engine can distinguish TRUE, FALSE, UNKNOWN, and conflict states.
        """

        requested = set(question.requested_roles)
        matches: List[Fact] = []
        for fact in self.facts:
            if fact.frame.predicate != question.frame.predicate:
                continue
            if fact.frame.tense != question.frame.tense:
                continue
            if fact.frame.modality != question.frame.modality:
                continue
            if (
                not include_opposing_polarity
                and fact.frame.polarity != question.frame.polarity
            ):
                continue
            if question.frame.repeated and not fact.frame.repeated:
                continue

            compatible = True
            for role, value in question.frame.roles.items():
                if role in requested:
                    continue
                fact_value = fact.frame.roles.get(role)
                if fact_value is None or not self.role_values_match(fact_value, value):
                    compatible = False
                    break
            if compatible:
                matches.append(fact)
        matches.sort(key=lambda item: (-item.certainty, -item.turn_id, item.fact_id))
        return tuple(matches)

    def latest_fact(self, predicate: Optional[str] = None) -> Optional[Fact]:
        values = [fact for fact in self.facts if predicate is None or fact.frame.predicate == predicate]
        if not values:
            return None
        return max(values, key=lambda item: (item.turn_id, item.certainty, item.fact_id))

    def pronoun(self, entity_id: str, *, subject: bool = True) -> str:
        if entity_id == "user":
            return "you"
        if entity_id == "assistant":
            return "I" if subject else "me"
        entity = self.entities[entity_id]
        if entity.number == GrammaticalNumber.PLURAL:
            return "they" if subject else "them"
        if entity.kind != EntityKind.PERSON:
            return "it"
        if entity.gender == Gender.FEMALE:
            return "she" if subject else "her"
        if entity.gender == Gender.MALE:
            return "he" if subject else "him"
        return "they" if subject else "them"

    def possessive(self, entity_id: str) -> str:
        if entity_id == "user":
            return "your"
        if entity_id == "assistant":
            return "my"
        entity = self.entities[entity_id]
        if entity.number == GrammaticalNumber.PLURAL or entity.gender == Gender.UNKNOWN:
            return "their"
        if entity.gender == Gender.FEMALE:
            return "her"
        if entity.gender == Gender.MALE:
            return "his"
        return "its"

    def describe_entity(
        self,
        entity_id: str,
        *,
        subject: bool = True,
        prefer_pronoun: bool = False,
        definite: bool = False,
    ) -> str:
        if prefer_pronoun:
            if entity_id not in {"user", "assistant"}:
                entity = self.entities[entity_id]
                # A proper name with no known gender is safer and clearer than
                # introducing singular "they" without an antecedent.
                if (
                    entity.kind == EntityKind.PERSON
                    and entity.gender == Gender.UNKNOWN
                    and entity.relation_to_user is None
                ):
                    return entity.display_name
            return self.pronoun(entity_id, subject=subject)
        if entity_id == "user":
            return "you"
        if entity_id == "assistant":
            return "I" if subject else "me"
        entity = self.entities[entity_id]
        if entity.relation_to_user:
            return f"your {entity.relation_to_user}"
        name = entity.display_name.strip()
        if entity.kind == EntityKind.PERSON:
            return name
        if name.lower().split()[0] in DETERMINERS:
            return name
        if definite:
            return f"the {name}"
        article = entity.determiner or choose_article(name)
        return f"{article} {name}" if article in {"a", "an"} else f"{article} {name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "entity_counter": self._entity_counter,
            "fact_counter": self._fact_counter,
            "running_state": self.running_state.to_dict(),
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "facts": [fact.to_dict() for fact in self.facts],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversationMemory":
        memory = cls(session_id=str(data.get("session_id", "default")))
        memory.turn = int(data.get("turn", 0))
        memory._entity_counter = int(data.get("entity_counter", 0))
        memory._fact_counter = int(data.get("fact_counter", 0))
        memory.running_state = AffectVector.from_mapping(data.get("running_state", {}))
        memory.entities = {
            entity.entity_id: entity
            for entity in (Entity.from_dict(item) for item in data.get("entities", []))
        }
        if "user" not in memory.entities or "assistant" not in memory.entities:
            previous = dict(memory.entities)
            memory.entities = {}
            memory._install_special_entities()
            memory.entities.update(previous)
        memory.facts = [Fact.from_dict(item) for item in data.get("facts", [])]
        return memory
