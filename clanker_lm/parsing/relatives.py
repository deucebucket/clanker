"""Relatives recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
import hashlib
from typing import List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import EntityKind, Gender, GrammaticalNumber, ModifierAttachmentAmbiguity, ModifierGapRole, ModifierRestriction, RefKind
from .types import RelativeSplit

class RelativesRules:
    """Stateless relatives transformations composed by SemanticParser."""

    def _split_relative_clause(
        self,
        tokens: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[RelativeSplit], Optional[ModifierAttachmentAmbiguity]]:
        """Split one finite relative modifier without attachment guessing."""

        items = list(tokens)
        marker_indices = [
            index
            for index, token in enumerate(items)
            if token.norm in self.RELATIVE_MARKERS
            and index > 0
            and not (
                token.norm == "that"
                and items[index - 1].norm == "so"
            )
        ]
        if not marker_indices:
            return None, None
        if len(marker_indices) > 1:
            surfaces = []
            for index in marker_indices:
                candidate = self._surface(
                    item for item in items[:index] if item.norm not in {",", ";"}
                )
                if candidate and candidate not in surfaces:
                    surfaces.append(candidate)
            ambiguity = ModifierAttachmentAmbiguity(
                marker=items[marker_indices[0]].norm,
                clause_surface=self._surface(items),
                candidate_head_surfaces=surfaces,
                reason="multiple finite relative markers require attachment resolution",
                ambiguity_id=(
                    "relative-"
                    + hashlib.sha256(" ".join(token.norm for token in items).encode("utf-8")).hexdigest()[:16]
                ),
                diagnostics=["multiple relative markers; durable assertion suppressed"],
            )
            return None, ambiguity

        marker_index = marker_indices[0]
        marker = items[marker_index].norm
        pre = list(items[:marker_index])
        if not pre:
            return None, None

        restriction = (
            ModifierRestriction.NONRESTRICTIVE
            if pre and pre[-1].norm == ","
            else ModifierRestriction.RESTRICTIVE
        )
        pre_without_comma = [item for item in pre if item.norm not in {",", ";"}]
        outer_verb_before = self._find_main_verb(pre_without_comma)

        closing_comma = -1
        if restriction == ModifierRestriction.NONRESTRICTIVE:
            closing_comma = next(
                (
                    index
                    for index in range(marker_index + 1, len(items))
                    if items[index].norm == ","
                ),
                -1,
            )
            if closing_comma < 0:
                return None, ModifierAttachmentAmbiguity(
                    marker=marker,
                    clause_surface=self._surface(items),
                    candidate_head_surfaces=[self._surface(pre_without_comma)],
                    reason="nonrestrictive relative clause lacks a closing comma",
                    ambiguity_id=(
                        "relative-"
                        + hashlib.sha256(" ".join(token.norm for token in items).encode("utf-8")).hexdigest()[:16]
                    ),
                    diagnostics=["unterminated nonrestrictive relative clause"],
                )
            relative_body = [
                item
                for item in items[marker_index + 1 : closing_comma]
                if item.norm not in {",", ";"}
            ]
            post_relative = [
                item
                for item in items[closing_comma + 1 :]
                if item.norm not in {",", ";"}
            ]
        else:
            relative_body = [
                item
                for item in items[marker_index + 1 :]
                if item.norm not in {",", ";"}
            ]
            post_relative = []

        main_tail: List[lexicon.Token] = []
        if outer_verb_before >= 0:
            head_start = outer_verb_before + 1
            if (
                head_start < len(pre_without_comma)
                and pre_without_comma[head_start].norm in lexicon.PREPOSITIONS
            ):
                head_start += 1
            head_tokens = pre_without_comma[head_start:]
            main_prefix = pre_without_comma[:head_start]
            if not head_tokens:
                return None, None
            if self._relative_head_looks_like_complement(head_tokens):
                return None, None
            if restriction == ModifierRestriction.RESTRICTIVE:
                main_tail = []
            else:
                main_tail = post_relative
        else:
            head_tokens = pre_without_comma
            main_prefix = []
            if self._relative_head_looks_like_complement(head_tokens):
                return None, None
            if restriction == ModifierRestriction.RESTRICTIVE:
                verb_indices = self._finite_verb_indices(relative_body)
                if len(verb_indices) < 2:
                    return None, None
                outer_boundary = verb_indices[-1]
                main_tail = relative_body[outer_boundary:]
                relative_body = relative_body[:outer_boundary]
            else:
                main_tail = post_relative

        if not head_tokens or not relative_body or not main_tail and not main_prefix:
            return None, None

        gap_role = self._relative_gap_role(marker, relative_body)
        if gap_role is None:
            return None, ModifierAttachmentAmbiguity(
                marker=marker,
                clause_surface=self._surface(items),
                candidate_head_surfaces=[self._surface(head_tokens)],
                reason="relative clause lacks a resolvable finite gap role",
                ambiguity_id=(
                    "relative-"
                    + hashlib.sha256(
                        " ".join(token.norm for token in items).encode("utf-8")
                    ).hexdigest()[:16]
                ),
                diagnostics=[
                    "relative gap unresolved; durable assertion suppressed"
                ],
            )
        signature = lexicon.normalize_phrase(
            [marker, gap_role.value]
            + [token.norm for token in relative_body]
        )
        head_surface = self._surface(head_tokens)
        kind, gender, number = self._relative_head_features(head_tokens)

        if restriction == ModifierRestriction.RESTRICTIVE:
            head_entity, internal_alias = memory.get_or_create_modified_entity(
                head_surface,
                signature,
                kind=kind,
                gender=gender,
                number=number,
            )
        else:
            head_np = self._parse_np(
                head_tokens,
                memory,
                expected_kind=kind,
                role_hint="subject",
            )
            if not head_np.ref or head_np.ref.kind != RefKind.ENTITY:
                return None, None
            head_entity = memory.get_entity(head_np.ref.key)
            if head_entity is None:
                return None, None
            internal_alias = memory.ensure_internal_alias(head_entity.entity_id)

        head_token = lexicon.Token(head_surface, internal_alias, -1)
        main_tokens = main_prefix + [head_token] + main_tail
        possessed_entity_id = ""

        if gap_role == ModifierGapRole.AGENT:
            modifier_tokens = [head_token] + relative_body
        elif gap_role == ModifierGapRole.PATIENT:
            verb_index = self._find_main_verb(relative_body)
            if verb_index < 0:
                return None, None
            verb_and_tail = list(relative_body[verb_index:])
            insertion = self._object_insertion_index(verb_and_tail)
            modifier_tokens = (
                list(relative_body[:verb_index])
                + verb_and_tail[:insertion]
                + [head_token]
                + verb_and_tail[insertion:]
            )
        else:
            verb_index = self._find_main_verb(relative_body)
            if verb_index <= 0:
                return None, None
            possessed_tokens = relative_body[:verb_index]
            possessed_surface = self._surface(possessed_tokens)
            possessed_kind = lexicon.classify_unknown_noun(
                [token.norm for token in possessed_tokens]
            )
            possessed, possessed_alias = memory.get_or_create_possessed_entity(
                head_entity.entity_id,
                possessed_surface,
                signature,
                kind=possessed_kind,
            )
            possessed_entity_id = possessed.entity_id
            possessed_token = lexicon.Token(
                possessed_surface,
                possessed_alias,
                -1,
            )
            modifier_tokens = [possessed_token] + list(relative_body[verb_index:])

        return (
            RelativeSplit(
                main_tokens=main_tokens,
                modifier_tokens=modifier_tokens,
                head_entity_id=head_entity.entity_id,
                marker=marker,
                gap_role=gap_role,
                restriction=restriction,
                possessed_entity_id=possessed_entity_id,
                diagnostics=[
                    f"relative marker={marker}",
                    f"relative restriction={restriction.value}",
                    f"relative gap={gap_role.value}",
                    f"relative head={head_surface}",
                ],
            ),
            None,
        )

    def _finite_verb_indices(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> List[int]:
        indices: List[int] = []
        for index, token in enumerate(tokens):
            previous = tokens[index - 1].norm if index > 0 else None
            if lexicon.is_probable_verb(token.norm, previous=previous):
                if token.norm not in lexicon.AUXILIARIES or index + 1 == len(tokens):
                    indices.append(index)
        return indices

    def _relative_gap_role(
        self,
        marker: str,
        body: Sequence[lexicon.Token],
    ) -> Optional[ModifierGapRole]:
        if marker == "whose":
            return ModifierGapRole.POSSESSOR
        if marker == "whom":
            return ModifierGapRole.PATIENT
        verb_index = self._find_main_verb(body)
        if verb_index < 0:
            return None
        if verb_index == 0:
            return ModifierGapRole.AGENT
        prefix = [token.norm for token in body[:verb_index]]
        if all(
            word in lexicon.AUXILIARIES
            or word in lexicon.NEGATORS
            or word in lexicon.INTENSIFIERS
            for word in prefix
        ):
            return ModifierGapRole.AGENT
        return ModifierGapRole.PATIENT

    def _relative_head_features(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Tuple[EntityKind, Gender, GrammaticalNumber]:
        content = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.DETERMINERS
            and token.norm not in {",", ";"}
        ]
        head = content[-1] if content else "entity"
        proper_name = bool(
            len(content) == 1
            and any(
                token.text[:1].isupper()
                for token in tokens
                if token.norm == head
            )
        )
        if proper_name or head in self.PERSON_RELATIVE_HEADS or head in lexicon.RELATIONS:
            kind = EntityKind.PERSON
        else:
            kind = lexicon.classify_unknown_noun(head)
        if head in self.FEMALE_RELATIVE_HEADS:
            gender = Gender.FEMALE
        elif head in self.MALE_RELATIVE_HEADS:
            gender = Gender.MALE
        else:
            gender = Gender.UNKNOWN
        number = (
            GrammaticalNumber.PLURAL
            if head.endswith("s") and not head.endswith("ss")
            else GrammaticalNumber.SINGULAR
        )
        return kind, gender, number

    def _relative_head_looks_like_complement(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        content = [
            token.norm
            for token in tokens
            if token.norm not in lexicon.DETERMINERS
        ]
        return bool(content and content[-1] in self.ABSTRACT_COMPLEMENT_HEADS)
