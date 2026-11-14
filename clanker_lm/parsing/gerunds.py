"""Gerunds recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import EventFrame, GerundAttachmentAmbiguity, GerundRelationType, RefKind
from .types import GerundSplit

class GerundsRules:
    """Stateless gerunds transformations composed by SemanticParser."""

    def _split_gerund_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Tuple[
        Optional[GerundSplit],
        Optional[GerundAttachmentAmbiguity],
    ]:
        """Split one reviewed matrix predicate and selected ``-ing`` event.

        Bare suffix shape is never sufficient.  The matrix predicate must be
        catalog licensed, the complement token must be a supported verbal
        present participle, and any perception controller must be an explicit
        object NP.  Unsupported relation layering is returned as a typed
        ambiguity instead of being flattened into one event.
        """

        items = [item for item in tokens if item.norm != ";"]
        licensed_candidates: List[Tuple[int, List[int]]] = []
        for matrix_index, token in enumerate(items):
            predicate = lexicon.lemma(token.norm)
            if predicate not in self.GERUND_PREDICATES:
                continue
            previous = items[matrix_index - 1].norm if matrix_index else None
            following = (
                items[matrix_index + 1].norm
                if matrix_index + 1 < len(items)
                else None
            )
            if not lexicon.is_probable_verb(
                token.norm,
                previous=previous,
                following=following,
            ):
                continue
            boundaries = [
                index
                for index in range(matrix_index + 1, len(items))
                if self._is_selected_ing_token(items, index)
            ]
            if boundaries:
                licensed_candidates.append((matrix_index, boundaries))

        for matrix_index, boundaries in licensed_candidates:
            layered = self._gerund_layering_ambiguity(
                items,
                matrix_index,
                boundaries,
            )
            if layered is not None:
                return None, layered

        # A comma-delimited clause-edge participle is adjunct material, not a
        # selected complement.  The flat clause parser cannot yet preserve
        # that extra relation without swallowing it into the subject/object,
        # so fail closed before entity creation or event storage.
        comma_indices = [
            index for index, item in enumerate(items) if item.norm == ","
        ]
        if len(comma_indices) == 1:
            comma_index = comma_indices[0]
            left = items[:comma_index]
            right = items[comma_index + 1 :]
            free_boundaries: List[int] = []
            if self._is_independently_finite(right):
                free_boundaries.extend(
                    index
                    for index in range(comma_index)
                    if self._is_selected_ing_token(items, index)
                )
            if self._is_independently_finite(left):
                free_boundaries.extend(
                    index
                    for index in range(comma_index + 1, len(items))
                    if self._is_selected_ing_token(items, index)
                )
            if free_boundaries:
                return None, self._gerund_ambiguity(
                    items,
                    free_boundaries,
                    reason=(
                        "comma-delimited -ing free adjunct requires staged "
                        "parsing"
                    ),
                )

        verb_index = self._find_main_verb(items)
        if verb_index < 0:
            return None, None
        predicate = lexicon.lemma(items[verb_index].norm)
        profile = self.GERUND_PREDICATES.get(predicate)
        if profile is None:
            if predicate == "feel":
                unsupported_boundaries = [
                    index
                    for index in range(verb_index + 1, len(items))
                    if self._is_selected_ing_token(items, index)
                ]
                if unsupported_boundaries:
                    return None, self._gerund_ambiguity(
                        items,
                        unsupported_boundaries,
                        reason=(
                            "feel with an -ing continuation is an unsupported "
                            "perception/controller boundary"
                        ),
                        candidate_types=[
                            GerundRelationType.PERCEPTION_PARTICIPIAL
                        ],
                    )
            return None, None

        boundaries = [
            index
            for index in range(verb_index + 1, len(items))
            if self._is_selected_ing_token(items, index)
        ]
        if not boundaries:
            return None, None
        if len(boundaries) > 1:
            nested = lexicon.lemma(
                items[boundaries[0]].norm
            ) in self.GERUND_PREDICATES
            return None, self._gerund_ambiguity(
                items,
                boundaries,
                reason=(
                    "nested selected -ing complement requires staged parsing"
                    if nested
                    else "multiple -ing complement boundaries remain plausible"
                ),
                candidate_types=[profile.relation_type],
            )

        boundary = boundaries[0]
        matrix_tail = [
            item
            for item in items[verb_index + 1 : boundary]
            if item.norm not in lexicon.PUNCTUATION
        ]

        # A trailing negative sequence belongs to the embedded event:
        # ``did not avoid calling`` negates AVOID, while ``avoided not
        # calling`` negates CALL.  Retain nearby intensification with the
        # embedded negator so it cannot be mistaken for a controller NP.
        embedded_prefix: List[lexicon.Token] = []
        suffix_start = len(matrix_tail)
        while (
            suffix_start > 0
            and matrix_tail[suffix_start - 1].norm
            in lexicon.NEGATORS | lexicon.INTENSIFIERS
        ):
            suffix_start -= 1
        trailing_scope = matrix_tail[suffix_start:]
        if any(item.norm in lexicon.NEGATORS for item in trailing_scope):
            embedded_prefix = trailing_scope
            matrix_tail = matrix_tail[:suffix_start]

        has_object_controller = bool(matrix_tail)
        if has_object_controller:
            if not profile.allows_object_controller:
                return None, self._gerund_ambiguity(
                    items,
                    [boundary],
                    reason=(
                        f"{predicate} does not license an explicit object "
                        "controller for selected -ing content"
                    ),
                    candidate_types=[profile.relation_type],
                )
            if not self._plausible_controller_np(matrix_tail):
                return None, self._gerund_ambiguity(
                    items,
                    [boundary],
                    reason=(
                        "participial object-controller phrase is "
                        "structurally unsupported"
                    ),
                    candidate_types=[profile.relation_type],
                )
            controller_role = "object"
        else:
            if not profile.allows_subject_controller:
                return None, self._gerund_ambiguity(
                    items,
                    [boundary],
                    reason=(
                        f"{predicate} requires an explicit object controller "
                        "before its participial complement"
                    ),
                    candidate_types=[GerundRelationType.PERCEPTION_PARTICIPIAL],
                )
            controller_role = "subject"

        matrix = [
            item
            for item in items[:boundary]
            if item not in embedded_prefix
            and item.norm not in lexicon.PUNCTUATION
        ]
        complement = [
            *embedded_prefix,
            *[
                item
                for item in items[boundary:]
                if item.norm not in lexicon.PUNCTUATION
            ],
        ]
        if self._contains_nested_content_clause(complement):
            return None, self._gerund_ambiguity(
                items,
                [boundary],
                reason=(
                    "selected -ing content containing a finite content "
                    "clause exceeds the configured depth"
                ),
                candidate_types=[profile.relation_type],
            )
        if self._contains_selected_infinitive(complement):
            return None, self._gerund_ambiguity(
                items,
                [boundary],
                reason=(
                    "selected -ing content containing an infinitival "
                    "complement exceeds the configured depth"
                ),
                candidate_types=[profile.relation_type],
            )

        return (
            GerundSplit(
                matrix_tokens=matrix,
                complement_tokens=complement,
                marker="-ing",
                profile=profile,
                controller_role=controller_role,
                certainty=profile.certainty,
                diagnostics=[
                    f"gerund matrix predicate={predicate}",
                    f"gerund relation={profile.relation_type.value}",
                    f"gerund status={profile.content_status.value}",
                    f"gerund boundary={boundary}",
                    f"gerund controller role={controller_role}",
                ],
            ),
            None,
        )

    @staticmethod
    def _is_selected_ing_token(
        tokens: Sequence[lexicon.Token],
        index: int,
    ) -> bool:
        token = tokens[index]
        if not token.norm.endswith("ing") or len(token.norm) < 5:
            return False
        previous = tokens[index - 1].norm if index else None
        following = tokens[index + 1].norm if index + 1 < len(tokens) else None
        if previous in lexicon.DETERMINERS | lexicon.POSSESSIVES:
            return False
        predicate = lexicon.lemma(token.norm)
        return (
            predicate in lexicon.KNOWN_VERBS
            and token.norm == lexicon.gerund_form(predicate)
            and lexicon.is_probable_verb(
                token.norm,
                previous=previous,
                following=following,
            )
        )

    def _gerund_layering_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        matrix_verb_index: int,
        boundaries: Sequence[int],
    ) -> Optional[GerundAttachmentAmbiguity]:
        norms = [item.norm for item in tokens]
        profile = self.GERUND_PREDICATES[lexicon.lemma(
            tokens[matrix_verb_index].norm
        )]
        candidate_types = [profile.relation_type]

        coordination = [
            index
            for index, norm in enumerate(norms)
            if norm in {"and", "but", "yet", "or", "so"}
        ]
        if coordination:
            return self._gerund_ambiguity(
                tokens,
                boundaries,
                reason=(
                    "selected -ing content combined with coordination "
                    "requires staged parsing"
                ),
                candidate_types=candidate_types,
            )

        relative_markers = [
            index
            for index, norm in enumerate(norms)
            if norm in {"who", "whom", "whose", "which"}
            or (norm == "that" and index < matrix_verb_index)
        ]
        if relative_markers:
            return self._gerund_ambiguity(
                tokens,
                boundaries,
                reason=(
                    "selected -ing content combined with a relative clause "
                    "requires staged parsing"
                ),
                candidate_types=candidate_types,
            )

        commas_before = [
            index for index in range(matrix_verb_index) if norms[index] == ","
        ]
        if len(commas_before) >= 2:
            return self._gerund_ambiguity(
                tokens,
                boundaries,
                reason=(
                    "selected -ing content combined with an appositive "
                    "requires staged parsing"
                ),
                candidate_types=candidate_types,
            )

        if self._split_subordinate_clause(tokens) is not None:
            return self._gerund_ambiguity(
                tokens,
                boundaries,
                reason=(
                    "selected -ing content combined with a subordinate "
                    "clause requires staged parsing"
                ),
                candidate_types=candidate_types,
            )

        first_verb = self._find_main_verb(tokens)
        if first_verb != matrix_verb_index:
            return self._gerund_ambiguity(
                tokens,
                boundaries,
                reason=(
                    "selected -ing content following a free adjunct or "
                    "earlier predicate requires staged parsing"
                ),
                candidate_types=candidate_types,
            )
        if any(
            norms[index] == ","
            for index in range(matrix_verb_index + 1, boundaries[0])
        ):
            return self._gerund_ambiguity(
                tokens,
                boundaries,
                reason=(
                    "comma-delimited participial material is a possible free "
                    "adjunct, not safely selected content"
                ),
                candidate_types=candidate_types,
            )
        return None

    @staticmethod
    def _gerund_controller_entity(
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

    @staticmethod
    def _gerund_source_entity(event: EventFrame) -> str:
        for role in ("agent", "experiencer", "subject", "possessor", "source"):
            ref = event.arguments.get(role)
            if ref is not None and ref.kind == RefKind.ENTITY:
                return ref.key
        return ""

    @staticmethod
    def _restore_memory_checkpoint(
        memory: ConversationMemory,
        checkpoint: Dict[str, object],
    ) -> None:
        """Roll back entity, alias, salience, counter, and store mutations."""

        memory.__dict__.clear()
        memory.__dict__.update(checkpoint)

    @staticmethod
    def _is_factual_phase_matrix(
        event: EventFrame,
        complement: EventFrame,
    ) -> bool:
        """Return whether a phase predicate can license an actual inference."""

        return (
            event.polarity
            and event.modality is None
            and event.tense != "future"
            and event.aspect in {"simple", "perfect"}
            and not GerundsRules._has_forward_deictic_time(event)
            and not GerundsRules._has_forward_deictic_time(complement)
        )

    @staticmethod
    def _has_forward_deictic_time(event: EventFrame) -> bool:
        time_ref = event.arguments.get("time")
        if time_ref is None:
            return False
        words = {
            word
            for value in (time_ref.key, time_ref.surface)
            for word in value.lower().split()
        }
        return bool(words & {"tomorrow", "later", "next"})

    def _gerund_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        boundaries: Sequence[int],
        *,
        reason: str,
        candidate_types: Sequence[GerundRelationType] = (),
    ) -> GerundAttachmentAmbiguity:
        first = boundaries[0] if boundaries else len(tokens)
        return GerundAttachmentAmbiguity(
            matrix_surface=self._surface(tokens[:first]),
            complement_surface=self._surface(tokens[first:]),
            clause_surface=self._surface(tokens),
            reason=reason,
            candidate_boundaries=list(boundaries),
            candidate_relation_types=list(candidate_types),
            ambiguity_id=(
                "gerund-"
                + hashlib.sha256(
                    " ".join(item.norm for item in tokens).encode("utf-8")
                ).hexdigest()[:16]
            ),
            diagnostics=[reason, "unsafe selected -ing content suppressed"],
        )
