"""Subordinates recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from .. import lexicon
from ..model import ClauseRelationDirection, ClauseRelationType
from .types import SubordinateSplit

class SubordinatesRules:
    """Stateless subordinates transformations composed by SemanticParser."""

    def _split_subordinate_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Optional[SubordinateSplit]:
        """Split one finite subordinate clause without guessing attachment."""

        items = list(tokens)
        norms = [item.norm for item in items]
        marker_index = -1
        marker_tokens: Tuple[str, ...] = ()
        for index in range(len(items)):
            for candidate in self.SUBORDINATE_MARKERS:
                if tuple(norms[index : index + len(candidate)]) == candidate:
                    marker_index = index
                    marker_tokens = candidate
                    break
            if marker_index >= 0:
                break
        if marker_index < 0:
            return None

        marker = " ".join(marker_tokens)
        after_marker = marker_index + len(marker_tokens)
        if marker_index == 0:
            comma_index = next(
                (
                    index
                    for index in range(after_marker, len(items))
                    if items[index].norm == ","
                ),
                -1,
            )
            if comma_index < 0:
                return None
            subordinate_tokens = [
                item
                for item in items[after_marker:comma_index]
                if item.norm not in {",", ";"}
            ]
            main_tokens = [
                item
                for item in items[comma_index + 1 :]
                if item.norm not in {",", ";"}
            ]
            order = "subordinate-first"
        else:
            main_tokens = [
                item
                for item in items[:marker_index]
                if item.norm not in {",", ";"}
            ]
            subordinate_tokens = [
                item
                for item in items[after_marker:]
                if item.norm not in {",", ";"}
            ]
            order = "main-first"

        if not (
            self._is_independently_finite(main_tokens)
            and self._is_independently_finite(subordinate_tokens)
        ):
            return None

        relation_type, direction, candidates, certainty, rationale = (
            self._classify_subordinate_relation(
                marker,
                main_tokens,
                subordinate_tokens,
            )
        )
        return SubordinateSplit(
            main_tokens=main_tokens,
            subordinate_tokens=subordinate_tokens,
            marker=marker,
            relation_type=relation_type,
            direction=direction,
            candidate_types=candidates,
            certainty=certainty,
            diagnostics=[
                f"subordinate marker={marker}",
                f"subordinate order={order}",
                rationale,
            ],
        )

    def _classify_subordinate_relation(
        self,
        marker: str,
        main_tokens: Sequence[lexicon.Token],
        subordinate_tokens: Sequence[lexicon.Token],
    ) -> Tuple[
        ClauseRelationType,
        ClauseRelationDirection,
        List[ClauseRelationType],
        int,
        str,
    ]:
        main_words = [item.norm for item in main_tokens]
        subordinate_words = [item.norm for item in subordinate_tokens]

        resolved = {
            "because": (
                ClauseRelationType.CAUSE,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "when": (
                ClauseRelationType.TEMPORAL_WHEN,
                ClauseRelationDirection.SYMMETRIC,
            ),
            "before": (
                ClauseRelationType.TEMPORAL_BEFORE,
                ClauseRelationDirection.MAIN_TO_SUBORDINATE,
            ),
            "after": (
                ClauseRelationType.TEMPORAL_AFTER,
                ClauseRelationDirection.MAIN_TO_SUBORDINATE,
            ),
            "until": (
                ClauseRelationType.TEMPORAL_UNTIL,
                ClauseRelationDirection.MAIN_TO_SUBORDINATE,
            ),
            "if": (
                ClauseRelationType.CONDITION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "unless": (
                ClauseRelationType.EXCEPTION_CONDITION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "although": (
                ClauseRelationType.CONCESSION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "though": (
                ClauseRelationType.CONCESSION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
            "even though": (
                ClauseRelationType.CONCESSION,
                ClauseRelationDirection.SUBORDINATE_TO_MAIN,
            ),
        }
        if marker in resolved:
            relation_type, direction = resolved[marker]
            return relation_type, direction, [], 230, "relation resolved by connector"

        if marker == "since":
            if any(word in lexicon.TIME_WORDS for word in subordinate_words):
                return (
                    ClauseRelationType.TEMPORAL_SINCE,
                    ClauseRelationDirection.MAIN_TO_SUBORDINATE,
                    [],
                    210,
                    "since resolved temporally by explicit time anchor",
                )
            return (
                ClauseRelationType.AMBIGUOUS,
                ClauseRelationDirection.UNRESOLVED,
                [ClauseRelationType.CAUSE, ClauseRelationType.TEMPORAL_SINCE],
                128,
                "since remains causally/temporally ambiguous",
            )

        if marker == "while":
            if self._contains_progressive(main_words) and self._contains_progressive(
                subordinate_words
            ):
                return (
                    ClauseRelationType.TEMPORAL_OVERLAP,
                    ClauseRelationDirection.SYMMETRIC,
                    [],
                    210,
                    "while resolved as temporal overlap from paired progressives",
                )
            return (
                ClauseRelationType.AMBIGUOUS,
                ClauseRelationDirection.UNRESOLVED,
                [
                    ClauseRelationType.TEMPORAL_OVERLAP,
                    ClauseRelationType.CONCESSION,
                ],
                128,
                "while remains temporal/concessive ambiguous",
            )

        if marker == "so that":
            if any(word in lexicon.MODALS for word in subordinate_words):
                return (
                    ClauseRelationType.PURPOSE,
                    ClauseRelationDirection.MAIN_TO_SUBORDINATE,
                    [],
                    220,
                    "so that resolved as purpose from subordinate modality",
                )
            if any(
                word in {"became", "got", "happened", "resulted"}
                for word in subordinate_words
            ):
                return (
                    ClauseRelationType.RESULT,
                    ClauseRelationDirection.MAIN_TO_SUBORDINATE,
                    [],
                    205,
                    "so that resolved as result from change-of-state cue",
                )
            return (
                ClauseRelationType.AMBIGUOUS,
                ClauseRelationDirection.UNRESOLVED,
                [ClauseRelationType.PURPOSE, ClauseRelationType.RESULT],
                128,
                "so that remains purpose/result ambiguous",
            )

        return (
            ClauseRelationType.AMBIGUOUS,
            ClauseRelationDirection.UNRESOLVED,
            [],
            96,
            "unclassified subordinate relation",
        )

    @staticmethod
    def _contains_progressive(words: Sequence[str]) -> bool:
        return any(word in lexicon.COPULAS for word in words) and any(
            word.endswith("ing") for word in words
        )
