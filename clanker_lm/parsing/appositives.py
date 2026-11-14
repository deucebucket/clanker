"""Appositives recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
import hashlib
from typing import List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import AppositiveAttachmentAmbiguity, AppositiveRelation, AppositiveRelationType, EntityKind, EventFrame, ModifierRestriction, RefKind
from .types import AppositiveSplit
from .nouns import NounsRules

class AppositivesRules:
    """Stateless appositives transformations composed by SemanticParser."""

    def _split_appositive_clause(
        self,
        tokens: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[AppositiveSplit], Optional[AppositiveAttachmentAmbiguity]]:
        """Resolve one explicit subject appositive with conservative evidence."""

        items = list(tokens)
        primary_tokens: List[lexicon.Token]
        appositive_tokens: List[lexicon.Token]
        tail: List[lexicon.Token]
        restriction: ModifierRestriction

        comma_indices = [
            index for index, token in enumerate(items) if token.norm == ","
        ]
        if len(comma_indices) >= 2:
            first, second = comma_indices[0], comma_indices[1]
            primary_tokens = [
                token for token in items[:first] if token.norm not in lexicon.PUNCTUATION
            ]
            appositive_tokens = [
                token for token in items[first + 1 : second]
                if token.norm not in lexicon.PUNCTUATION
            ]
            tail = [
                token for token in items[second + 1 :]
                if token.norm not in lexicon.PUNCTUATION
            ]
            restriction = ModifierRestriction.NONRESTRICTIVE
            if (
                not primary_tokens
                or not appositive_tokens
                or not tail
                or appositive_tokens[0].norm in self.RELATIVE_MARKERS
                or self._find_main_verb(primary_tokens) >= 0
                or self._find_main_verb(appositive_tokens) >= 0
                or self._find_main_verb(tail) < 0
            ):
                return None, None
        elif not comma_indices:
            verb_index = self._find_main_verb(items)
            if verb_index <= 2:
                return None, None
            subject = [
                token for token in items[:verb_index]
                if token.norm not in lexicon.PUNCTUATION
            ]
            name_start = next(
                (
                    index
                    for index, token in enumerate(subject[1:], start=1)
                    if self._looks_proper_name([token])
                    and any(
                        item.norm in lexicon.RELATIONS
                        or item.norm in self.APPOSITIVE_ROLE_NOUNS
                        for item in subject[:index]
                    )
                ),
                -1,
            )
            if name_start < 1:
                return None, None
            primary_tokens = subject[:name_start]
            appositive_tokens = subject[name_start:]
            tail = list(items[verb_index:])
            restriction = ModifierRestriction.RESTRICTIVE
        else:
            return None, None

        primary_surface = self._surface(primary_tokens)
        appositive_surface = self._surface(appositive_tokens)
        primary_is_name = self._looks_proper_name(primary_tokens)
        appositive_is_name = self._looks_proper_name(appositive_tokens)

        if appositive_is_name and not primary_is_name:
            canonical_tokens = appositive_tokens
            descriptor_tokens = primary_tokens
        else:
            canonical_tokens = primary_tokens
            descriptor_tokens = appositive_tokens

        relation_type, expected_kind, owner_id, role_name = self._appositive_profile(
            descriptor_tokens,
            descriptor_is_name=self._looks_proper_name(descriptor_tokens),
        )
        canonical = self._parse_np(
            canonical_tokens,
            memory,
            expected_kind=expected_kind,
            role_hint="subject",
        )
        if not canonical.ref or canonical.ref.kind != RefKind.ENTITY:
            candidates = [
                candidate
                for unresolved in canonical.unresolved
                for candidate in unresolved.candidates
            ]
            return None, self._appositive_ambiguity(
                items,
                primary_surface,
                appositive_surface,
                "appositive head could not be resolved",
                candidates,
            )

        entity = memory.get_entity(canonical.ref.key)
        if entity is None:
            return None, self._appositive_ambiguity(
                items,
                primary_surface,
                appositive_surface,
                "appositive head entity is unavailable",
                [],
            )

        descriptor_surface = self._surface(descriptor_tokens)
        binding = memory.bind_appositive_alias(
            entity.entity_id,
            descriptor_surface,
            relation_type=relation_type,
            expected_kind=expected_kind,
            role_owner_id=owner_id,
            role_name=role_name,
        )
        if not binding.resolved:
            return None, self._appositive_ambiguity(
                items,
                primary_surface,
                appositive_surface,
                binding.reason or "appositive identity is ambiguous",
                [candidate.entity_id for candidate in binding.candidates],
            )

        # Both explicit surfaces identify the same entity after safe binding.
        entity.add_alias(primary_surface)
        entity.add_alias(appositive_surface)
        internal_alias = memory.ensure_internal_alias(entity.entity_id)
        head_token = lexicon.Token(
            entity.canonical_name,
            internal_alias,
            -1,
        )
        relation = AppositiveRelation(
            head_entity_id=entity.entity_id,
            primary_surface=primary_surface,
            appositive_surface=appositive_surface,
            relation_type=relation_type,
            restriction=restriction,
            appositive_key=memory.normalize_alias(descriptor_surface),
            role_owner_id=owner_id,
            role_name=role_name,
            diagnostics=[
                f"appositive primary={primary_surface}",
                f"appositive value={appositive_surface}",
                f"appositive type={relation_type.value}",
            ],
        )
        return (
            AppositiveSplit(
                main_tokens=[head_token] + tail,
                relation=relation,
                entity_ids=[entity.entity_id],
                diagnostics=list(relation.diagnostics),
            ),
            None,
        )

    @staticmethod
    def _looks_proper_name(tokens: Sequence[lexicon.Token]) -> bool:
        content = [
            token
            for token in tokens
            if token.norm not in lexicon.DETERMINERS
            and token.norm not in lexicon.POSSESSIVES
            and token.norm not in lexicon.PUNCTUATION
        ]
        return bool(content) and all(
            token.text[:1].isupper()
            and token.norm not in lexicon.RELATIONS
            for token in content
        )

    def _appositive_profile(
        self,
        tokens: Sequence[lexicon.Token],
        *,
        descriptor_is_name: bool,
    ) -> Tuple[AppositiveRelationType, EntityKind, str, str]:
        norms = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.PUNCTUATION
        ]
        if descriptor_is_name:
            return AppositiveRelationType.IDENTITY, EntityKind.PERSON, "", ""
        relation_word = next(
            (
                word
                for word in reversed(norms)
                if word in lexicon.RELATIONS
                or word in self.APPOSITIVE_ROLE_NOUNS
            ),
            "",
        )
        if relation_word:
            canonical, _gender, _number, kind = lexicon.relation_features(
                relation_word
            )
            owner_id = ""
            if norms and norms[0] == "my":
                owner_id = "user"
            elif norms and norms[0] == "your":
                owner_id = "assistant"
            return AppositiveRelationType.ROLE, kind, owner_id, canonical
        location_heads = {
            "city", "country", "county", "hospital", "office", "place",
            "school", "state", "store", "town", "village",
        }
        expected = (
            EntityKind.PLACE
            if any(word in location_heads for word in norms)
            else lexicon.classify_unknown_noun(norms)
        )
        return AppositiveRelationType.DESCRIPTION, expected, "", ""

    @staticmethod
    def _appositive_ambiguity(
        tokens: Sequence[lexicon.Token],
        primary_surface: str,
        appositive_surface: str,
        reason: str,
        candidate_ids: Sequence[str],
    ) -> AppositiveAttachmentAmbiguity:
        normalized = " ".join(token.norm for token in tokens)
        return AppositiveAttachmentAmbiguity(
            primary_surface=primary_surface,
            appositive_surface=appositive_surface,
            clause_surface=NounsRules._surface(tokens),
            reason=reason,
            candidate_entity_ids=list(dict.fromkeys(candidate_ids)),
            ambiguity_id=(
                "appositive-"
                + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
            ),
            diagnostics=["appositive identity unresolved; durable assertion suppressed"],
        )

    @staticmethod
    def _event_signature_key(event: EventFrame) -> str:
        payload = repr(event.signature()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]
