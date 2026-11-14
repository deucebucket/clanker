"""Nouns recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
from typing import List, Optional, Sequence
from .. import lexicon
from ..memory import ConversationMemory
from ..model import EntityKind, Gender, GrammaticalNumber, SemanticRef
from .types import NPResult

class NounsRules:
    """Stateless nouns transformations composed by SemanticParser."""

    def _parse_np(
        self,
        tokens: Sequence[lexicon.Token],
        memory: ConversationMemory,
        *,
        expected_kind: EntityKind = EntityKind.UNKNOWN,
        role_hint: str = "other",
        preposition: Optional[str] = None,
    ) -> NPResult:
        items = [token for token in tokens if token.norm not in lexicon.PUNCTUATION and token.norm not in lexicon.INTENSIFIERS]
        if not items:
            return NPResult(None)
        surface = self._surface(items)
        norms = [token.norm for token in items]

        # Typed variable inserted by interrogative transformation.
        if len(items) == 1 and norms[0].startswith("__var_") and norms[0].endswith("__"):
            role = norms[0][len("__var_") : -2]
            return NPResult(SemanticRef.variable(role, expected_kind), surface=surface)

        # Private aliases inserted by structural transformations bind an
        # already-created entity before ordinary possessive/name heuristics.
        if len(items) == 1 and norms[0].startswith(
            ("entityref-", "modified-", "possessed-")
        ):
            resolution = memory.find_by_alias(norms[0], EntityKind.UNKNOWN)
            if resolution.resolved and resolution.entity:
                entity = resolution.entity
                memory.mention(entity.entity_id, role_hint)
                return NPResult(
                    entity.to_ref(surface),
                    entity_ids=[entity.entity_id],
                    surface=surface,
                )
            unresolved = memory.unresolved_from_resolution(
                surface,
                resolution,
                expected_kind,
            )
            return NPResult(None, [unresolved], surface=surface)

        # Numeric determiner/quantity: "three cars", "$50".
        quantity: Optional[SemanticRef] = None
        quantity_prefix: List[str] = []
        for word in norms:
            if word.isdigit() or word.replace(".", "", 1).isdigit() or (word in lexicon.NUMBER_WORDS and word not in lexicon.ARTICLES) or word[:1] in "$€£":
                quantity_prefix.append(word)
            else:
                break
        if quantity_prefix:
            parsed = lexicon.parse_number(quantity_prefix)
            if parsed is not None:
                key = str(int(parsed)) if isinstance(parsed, float) and parsed.is_integer() else str(parsed)
                quantity = SemanticRef.literal(key, " ".join(quantity_prefix), EntityKind.ABSTRACT)
                items = items[len(quantity_prefix) :]
                norms = norms[len(quantity_prefix) :]
                if not items:
                    return NPResult(quantity, quantity=quantity, surface=surface)

        # Fixed participant or third-person pronoun.
        if len(items) == 1 and norms[0] in lexicon.PRONOUN_FEATURES:
            resolution = memory.resolve_pronoun(norms[0], expected_kind)
            if resolution.resolved:
                assert resolution.entity is not None
                memory.mention(resolution.entity.entity_id, role_hint)
                return NPResult(resolution.entity.to_ref(surface), quantity=quantity, entity_ids=[resolution.entity.entity_id], surface=surface)
            unresolved = memory.unresolved_from_resolution(surface, resolution, expected_kind)
            return NPResult(None, [unresolved], quantity=quantity, surface=surface)

        # Genitive noun phrase: ``Sarah's eyes`` / ``John's car``.
        # The tokenizer intentionally keeps apostrophe-s attached, so resolve
        # the owner here and key the possessed entity by owner + head phrase.
        if len(items) >= 2 and (norms[0].endswith("'s") or norms[0].endswith("’s")):
            owner_surface = items[0].text[:-2]
            owner_norm = norms[0][:-2]
            owner_resolution = memory.find_by_alias(owner_norm, EntityKind.PERSON)
            if owner_resolution.resolved and owner_resolution.entity:
                owner = owner_resolution.entity
            else:
                owner = memory.get_or_create_named_entity(
                    owner_surface,
                    kind=EntityKind.PERSON,
                    gender=lexicon.infer_name_gender(owner_norm),
                    aliases=[owner_norm],
                    role_salience=0.4,
                )
            possessed_items = list(items[1:])
            possessed_norms = [token.norm for token in possessed_items if token.norm not in lexicon.ARTICLES]
            canonical = self._surface([token for token in possessed_items if token.norm not in lexicon.ARTICLES])
            alias = f"{owner.entity_id}:{lexicon.normalize_phrase(possessed_norms)}"
            existing = memory.find_by_alias(alias, expected_kind)
            if existing.resolved and existing.entity:
                entity = existing.entity
            else:
                kind = expected_kind if expected_kind != EntityKind.UNKNOWN else lexicon.classify_unknown_noun(possessed_norms, preposition=preposition)
                entity = memory.get_or_create_named_entity(
                    canonical,
                    kind=kind,
                    aliases=[surface, alias, lexicon.normalize_phrase(possessed_norms)],
                    role_salience=0.4,
                )
                entity.owner_id = owner.entity_id
            self._apply_np_attributes(entity, possessed_norms)
            return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[owner.entity_id, entity.entity_id], surface=surface)

        # Possessive relationship: my sister / your mom / his brother.
        if len(items) >= 2 and norms[0] in lexicon.POSSESSIVES and norms[-1] in lexicon.RELATIONS:
            owner_id: Optional[str] = None
            if norms[0] == "my":
                owner_id = "user"
            elif norms[0] == "your":
                owner_id = "assistant"
            elif norms[0] in {"his", "her", "their"}:
                pronoun = {"his": "he", "her": "she", "their": "they"}[norms[0]]
                owner_resolution = memory.resolve_pronoun(pronoun, EntityKind.PERSON)
                if owner_resolution.resolved and owner_resolution.entity:
                    owner_id = owner_resolution.entity.entity_id
                else:
                    unresolved = memory.unresolved_from_resolution(norms[0], owner_resolution, EntityKind.PERSON)
                    return NPResult(None, [unresolved], quantity=quantity, surface=surface)
            if owner_id:
                entity = memory.get_or_create_relation(owner_id, norms[-1], surface=surface, role_salience=0.4)
                return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[entity.entity_id], surface=surface)

        # Possessive common object: my car / her phone.  Resolve owner and create
        # an object entity keyed by owner + head noun.
        if len(items) >= 2 and norms[0] in lexicon.POSSESSIVES:
            owner_id: Optional[str] = None
            if norms[0] == "my":
                owner_id = "user"
            elif norms[0] == "your":
                owner_id = "assistant"
            else:
                pronoun = {"his": "he", "her": "she", "their": "they", "our": "we", "its": "it"}.get(norms[0])
                if pronoun:
                    owner_resolution = memory.resolve_pronoun(pronoun)
                    if owner_resolution.resolved and owner_resolution.entity:
                        owner_id = owner_resolution.entity.entity_id
                    else:
                        unresolved = memory.unresolved_from_resolution(norms[0], owner_resolution, EntityKind.UNKNOWN)
                        return NPResult(None, [unresolved], quantity=quantity, surface=surface)
            canonical = " ".join(norms[1:])
            alias = f"{owner_id}:{canonical}" if owner_id else canonical
            existing = memory.find_by_alias(alias, expected_kind)
            if existing.resolved and existing.entity:
                entity = existing.entity
            else:
                kind = expected_kind if expected_kind != EntityKind.UNKNOWN else lexicon.classify_unknown_noun(norms[1:], preposition=preposition)
                entity = memory.get_or_create_named_entity(canonical, kind=kind, aliases=[surface, alias], role_salience=0.4)
                entity.owner_id = owner_id
            return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[entity.entity_id], surface=surface)

        # Remove articles for entity lookup/creation.
        content_items = [token for token in items if token.norm not in lexicon.ARTICLES]
        content_norms = [token.norm for token in content_items]
        if not content_items:
            return NPResult(None, quantity=quantity, surface=surface)
        normalized = lexicon.normalize_phrase(content_norms)

        # Standalone relation defaults to the user's relation in conversational
        # self-report unless an existing unowned relation is more salient.
        if content_norms[-1] in lexicon.RELATIONS and len(content_norms) <= 2:
            entity = memory.get_or_create_relation("user", content_norms[-1], surface=surface, role_salience=0.4)
            return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[entity.entity_id], surface=surface)

        # Existing alias/coreference before creating a new entity.
        resolution = memory.find_by_alias(normalized, expected_kind)
        if resolution.resolved and resolution.entity:
            resolution.entity.add_alias(surface)
            self._apply_np_attributes(resolution.entity, content_norms)
            memory.mention(resolution.entity.entity_id, role_hint)
            return NPResult(resolution.entity.to_ref(surface), quantity=quantity, entity_ids=[resolution.entity.entity_id], surface=surface)
        if resolution.status == "ambiguous":
            unresolved = memory.unresolved_from_resolution(surface, resolution, expected_kind)
            return NPResult(None, [unresolved], quantity=quantity, surface=surface)

        # Proper names and known names are people; prepositional/location nouns
        # and ordinary noun phrases are typed conservatively.
        looks_name = (
            len(content_items) <= 3
            and any(token.text[:1].isupper() for token in content_items)
            and not all(token.index == 0 for token in content_items)
        ) or content_norms[0] in lexicon.FEMALE_NAMES | lexicon.MALE_NAMES
        if looks_name and expected_kind in {EntityKind.UNKNOWN, EntityKind.PERSON}:
            kind = EntityKind.PERSON
            gender = lexicon.infer_name_gender(content_norms[0])
        else:
            kind = expected_kind if expected_kind != EntityKind.UNKNOWN else lexicon.classify_unknown_noun(content_norms, preposition=preposition)
            gender = Gender.UNKNOWN
        number = (
            GrammaticalNumber.PLURAL
            if "and" in content_norms or (content_norms[-1].endswith("s") and content_norms[-1] not in {"news"})
            else GrammaticalNumber.SINGULAR
        )
        if "and" in content_norms:
            gender = Gender.NEUTRAL
        entity = memory.get_or_create_named_entity(
            self._surface(content_items),
            kind=kind,
            gender=gender,
            number=number,
            aliases=[surface, normalized],
            role_salience=0.4,
        )
        self._apply_np_attributes(entity, content_norms)
        return NPResult(entity.to_ref(surface), quantity=quantity, entity_ids=[entity.entity_id], surface=surface)

    @staticmethod
    def _apply_np_attributes(entity, words: Sequence[str]) -> None:
        """Extract safe, explicit modifier attributes from a noun phrase."""

        if not words:
            return
        for word in words[:-1]:
            if word in lexicon.COLORS:
                entity.attributes["color"] = word
            dimension = lexicon.ADJECTIVE_DIMENSIONS.get(word)
            if dimension:
                entity.attributes[dimension] = word

    @staticmethod
    def _looks_entity_phrase(tokens: Sequence[lexicon.Token]) -> bool:
        if not tokens:
            return False
        if len(tokens) == 1 and tokens[0].norm in lexicon.PRONOUN_FEATURES:
            return True
        return any(token.text[:1].isupper() for token in tokens) or tokens[-1].norm in lexicon.RELATIONS

    @staticmethod
    def _surface(tokens: Sequence[lexicon.Token]) -> str:
        return " ".join(token.text for token in tokens if token.norm not in lexicon.PUNCTUATION).strip()
