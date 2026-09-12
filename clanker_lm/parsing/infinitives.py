"""Infinitives recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
import hashlib
from typing import List, Optional, Sequence, Tuple
from .. import lexicon
from ..model import EventFrame, InfinitivalAttachmentAmbiguity, InfinitivalRelationType, RefKind
from .types import InfinitivalPredicateProfile, InfinitivalSplit

class InfinitivesRules:
    """Stateless infinitives transformations composed by SemanticParser."""

    def _split_infinitival_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Tuple[
        Optional[InfinitivalSplit],
        Optional[InfinitivalAttachmentAmbiguity],
    ]:
        """Split one catalog-licensed selected ``to`` complement.

        Movement/purpose clauses deliberately remain in the pre-existing
        purpose-adjunct path.  This splitter activates only when a reviewed
        matrix predicate licenses control or raising.
        """

        items = [item for item in tokens if item.norm != ";"]

        # Relation layers can place another finite verb before the selected
        # infinitival matrix predicate (for example, a relative clause in
        # ``the woman who called plans to leave``).  Inspect every licensed
        # predicate that has a later infinitival boundary before relying on
        # the ordinary main-verb heuristic.  Until staged composition exists,
        # such inputs must fail closed instead of letting the earlier modifier
        # verb steal the entire frame.
        licensed_candidates: List[int] = []
        for candidate_index, token in enumerate(items):
            if lexicon.lemma(token.norm) not in self.INFINITIVAL_PREDICATES:
                continue
            if any(
                items[index].norm == "to"
                and index + 1 < len(items)
                and lexicon.is_probable_verb(items[index + 1].norm, "to")
                for index in range(candidate_index + 1, len(items) - 1)
            ):
                licensed_candidates.append(candidate_index)
        for candidate_index in licensed_candidates:
            layered = self._infinitival_layering_ambiguity(
                items,
                candidate_index,
            )
            if layered is not None:
                return None, layered

        verb_index = self._find_main_verb(items)
        if verb_index < 0:
            return None, None
        predicate = lexicon.lemma(items[verb_index].norm)
        profile = self.INFINITIVAL_PREDICATES.get(predicate)
        if profile is None:
            return None, None

        boundaries: List[int] = []
        for index in range(verb_index + 1, len(items) - 1):
            if items[index].norm != "to":
                continue
            next_index = index + 1
            while (
                next_index < len(items)
                and items[next_index].norm
                in lexicon.NEGATORS | lexicon.INTENSIFIERS
            ):
                next_index += 1
            if next_index >= len(items):
                continue
            next_word = items[next_index].norm
            if lexicon.is_probable_verb(next_word, "to"):
                boundaries.append(index)

        if not boundaries:
            return None, None
        if len(boundaries) > 1:
            return None, self._infinitival_ambiguity(
                items,
                boundaries,
                reason="nested infinitival content exceeds the configured depth",
                candidate_types=[profile.relation_type],
            )

        boundary = boundaries[0]
        matrix_tail = [
            item
            for item in items[verb_index + 1 : boundary]
            if item.norm not in lexicon.PUNCTUATION
        ]
        embedded_prefix: List[lexicon.Token] = []
        if matrix_tail and matrix_tail[-1].norm == "not":
            # ``plans not to leave`` and ``told John not to leave`` place
            # negation inside the infinitive rather than on the matrix event.
            embedded_prefix.append(matrix_tail.pop())

        has_object_controller = bool(matrix_tail)
        if has_object_controller:
            if not profile.allows_object_controller:
                return None, self._infinitival_ambiguity(
                    items,
                    [boundary],
                    reason=(
                        f"{predicate} does not license an explicit object "
                        "controller in this parser slice"
                    ),
                    candidate_types=[profile.relation_type],
                )
            if not self._plausible_controller_np(matrix_tail):
                return None, self._infinitival_ambiguity(
                    items,
                    [boundary],
                    reason="object-controller phrase is structurally unsupported",
                    candidate_types=[InfinitivalRelationType.OBJECT_CONTROL],
                )
            relation_type = InfinitivalRelationType.OBJECT_CONTROL
            controller_role = "object"
        else:
            if not profile.allows_subject_controller:
                return None, self._infinitival_ambiguity(
                    items,
                    [boundary],
                    reason=f"{predicate} requires an explicit object controller",
                    candidate_types=[InfinitivalRelationType.OBJECT_CONTROL],
                )
            relation_type = profile.relation_type
            controller_role = "subject"

        selected_profile = InfinitivalPredicateProfile(
            relation_type=relation_type,
            content_status=profile.content_status,
            predicate_family=profile.predicate_family,
            certainty=profile.certainty,
            allows_object_controller=profile.allows_object_controller,
            allows_subject_controller=profile.allows_subject_controller,
        )
        matrix = [
            item
            for item in items[:boundary]
            if item not in embedded_prefix and item.norm not in lexicon.PUNCTUATION
        ]
        complement = [
            *embedded_prefix,
            *[
                item
                for item in items[boundary + 1 :]
                if item.norm not in lexicon.PUNCTUATION
            ],
        ]
        if self._contains_nested_content_clause(complement):
            return None, self._infinitival_ambiguity(
                items,
                [boundary],
                reason=(
                    "infinitival content containing a finite content clause "
                    "exceeds the configured depth"
                ),
                candidate_types=[relation_type],
            )
        return (
            InfinitivalSplit(
                matrix_tokens=matrix,
                complement_tokens=complement,
                marker="to",
                profile=selected_profile,
                controller_role=controller_role,
                certainty=selected_profile.certainty,
                diagnostics=[
                    f"infinitival matrix predicate={predicate}",
                    f"infinitival relation={relation_type.value}",
                    f"infinitival status={profile.content_status.value}",
                    f"infinitival boundary={boundary}",
                ],
            ),
            None,
        )

    def _infinitival_layering_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        matrix_verb_index: int,
    ) -> Optional[InfinitivalAttachmentAmbiguity]:
        norms = [item.norm for item in tokens]
        coordination_markers = [
            index
            for index, norm in enumerate(norms)
            if norm in {"and", "but", "yet", "or", "so"}
        ]
        if coordination_markers:
            return self._infinitival_ambiguity(
                tokens,
                coordination_markers,
                reason=(
                    "infinitival content combined with coordination requires "
                    "staged parsing"
                ),
            )
        relative_markers = [
            index
            for index, norm in enumerate(norms)
            if norm in {"who", "whom", "whose", "which"}
            or (norm == "that" and index < matrix_verb_index)
        ]
        if relative_markers:
            return self._infinitival_ambiguity(
                tokens,
                [matrix_verb_index + 1],
                reason=(
                    "infinitival content combined with a relative clause "
                    "requires staged parsing"
                ),
            )
        commas_before = [
            index for index in range(matrix_verb_index) if norms[index] == ","
        ]
        if len(commas_before) >= 2:
            return self._infinitival_ambiguity(
                tokens,
                [matrix_verb_index + 1],
                reason=(
                    "infinitival content combined with an appositive requires "
                    "staged parsing"
                ),
            )
        for index in range(len(tokens)):
            for marker in self.SUBORDINATE_MARKERS:
                if tuple(norms[index : index + len(marker)]) == marker:
                    return self._infinitival_ambiguity(
                        tokens,
                        [matrix_verb_index + 1],
                        reason=(
                            "infinitival content combined with a subordinate "
                            "clause requires staged parsing"
                        ),
                    )
        return None

    @staticmethod
    def _plausible_controller_np(
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        if not tokens:
            return False
        norms = [item.norm for item in tokens]
        if any(item in lexicon.PREPOSITIONS for item in norms):
            return False
        if any(
            lexicon.is_probable_verb(
                item.norm,
                previous=(tokens[index - 1].norm if index else None),
                following=(
                    tokens[index + 1].norm
                    if index + 1 < len(tokens)
                    else None
                ),
            )
            for index, item in enumerate(tokens)
        ):
            return False
        return True

    @staticmethod
    def _infinitival_controller_entity(
        event: EventFrame,
        controller_role: str,
    ) -> str:
        roles = (
            ("patient", "recipient")
            if controller_role == "object"
            else ("agent", "experiencer", "subject", "possessor", "patient")
        )
        for role in roles:
            ref = event.arguments.get(role)
            if ref is not None and ref.kind == RefKind.ENTITY:
                return ref.key
        return ""

    def _infinitival_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        boundaries: Sequence[int],
        *,
        reason: str,
        candidate_types: Sequence[InfinitivalRelationType] = (),
    ) -> InfinitivalAttachmentAmbiguity:
        first = boundaries[0] if boundaries else len(tokens)
        return InfinitivalAttachmentAmbiguity(
            matrix_surface=self._surface(tokens[:first]),
            complement_surface=self._surface(tokens[first:]),
            clause_surface=self._surface(tokens),
            reason=reason,
            candidate_boundaries=list(boundaries),
            candidate_relation_types=list(candidate_types),
            ambiguity_id=(
                "infinitive-"
                + hashlib.sha256(
                    " ".join(item.norm for item in tokens).encode("utf-8")
                ).hexdigest()[:16]
            ),
            diagnostics=[reason, "unsafe infinitival content suppressed"],
        )
