"""Content recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
import hashlib
from typing import List, Optional, Sequence, Tuple
from .. import lexicon
from ..model import ContentAttachmentAmbiguity, EventFrame, RefKind
from .types import ContentSplit

class ContentRules:
    """Stateless content transformations composed by SemanticParser."""

    def _split_content_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Tuple[Optional[ContentSplit], Optional[ContentAttachmentAmbiguity]]:
        """Split one licensed finite content complement conservatively.

        This bounded slice intentionally refuses to flatten a content clause
        together with another relation layer.  Relative modifiers, appositives,
        and finite subordination are each represented by their own typed
        structures elsewhere in the parser; until staged composition is
        implemented, combining them here would create false source identities
        or swallow an outer matrix predicate.
        """

        items = [item for item in tokens if item.norm not in {";"}]
        layered_ambiguity = self._content_layering_ambiguity(items)
        if layered_ambiguity is not None:
            return None, layered_ambiguity

        verb_index = self._find_main_verb(items)
        if verb_index < 0:
            return None, None
        predicate = lexicon.lemma(items[verb_index].norm)
        profile = self.CONTENT_PREDICATES.get(predicate)
        if profile is None:
            return None, None
        relation_type, family, certainty = profile

        explicit = [
            index
            for index in range(verb_index + 1, len(items))
            if items[index].norm == "that"
        ]
        if explicit:
            marker_index = explicit[0]
            if len(explicit) > 1:
                return None, self._content_ambiguity(
                    items,
                    [index + 1 for index in explicit],
                    reason="multiple complementizer candidates require nested parsing",
                )
            matrix_tail = [
                item
                for item in items[verb_index + 1 : marker_index]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if matrix_tail and predicate not in self.CONTENT_RECIPIENT_PREDICATES:
                # A substantive noun phrase before ``that`` is more likely the
                # head of a relative clause (``reported the idea that ...``).
                return None, None
            content = [
                item
                for item in items[marker_index + 1 :]
                if item.norm not in lexicon.PUNCTUATION
            ]
            matrix = [
                item
                for item in items[:marker_index]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if not self._is_independently_finite(content):
                return None, None
            if self._contains_nested_content_clause(content):
                return None, self._content_ambiguity(
                    items,
                    [marker_index + 1],
                    reason="nested finite content exceeds the configured depth",
                )
            if self._contains_selected_infinitive(content):
                return None, self._content_ambiguity(
                    items,
                    [marker_index + 1],
                    reason=(
                        "finite content containing an infinitival complement "
                        "exceeds the configured depth"
                    ),
                )
            return (
                ContentSplit(
                    matrix_tokens=matrix,
                    content_tokens=content,
                    marker="that",
                    relation_type=relation_type,
                    predicate_family=family,
                    certainty=certainty,
                    diagnostics=[
                        f"finite content predicate={predicate}",
                        "content marker=that",
                    ],
                ),
                None,
            )

        candidates: List[int] = []
        for boundary in range(verb_index + 1, len(items)):
            content = [
                item
                for item in items[boundary:]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if not self._is_independently_finite(content):
                continue
            if not self._plausible_content_subject(content):
                continue
            matrix_tail = [
                item
                for item in items[verb_index + 1 : boundary]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if matrix_tail and predicate not in self.CONTENT_RECIPIENT_PREDICATES:
                continue
            candidates.append(boundary)

        if not candidates:
            return None, None
        if predicate in self.CONTENT_RECIPIENT_PREDICATES:
            # ``tell`` licenses a recipient before zero-marked content. Prefer
            # the boundary that keeps one recipient NP with the matrix clause.
            recipient_candidates = [
                boundary
                for boundary in candidates
                if boundary > verb_index + 1
            ]
            if recipient_candidates:
                candidates = recipient_candidates
        if len(candidates) != 1:
            return None, self._content_ambiguity(
                items,
                candidates,
                reason="multiple finite zero-complementizer boundaries remain plausible",
            )

        boundary = candidates[0]
        matrix = [
            item for item in items[:boundary] if item.norm not in lexicon.PUNCTUATION
        ]
        content = [
            item for item in items[boundary:] if item.norm not in lexicon.PUNCTUATION
        ]
        if self._contains_nested_content_clause(content):
            return None, self._content_ambiguity(
                items,
                [boundary],
                reason="nested finite content exceeds the configured depth",
            )
        if self._contains_selected_infinitive(content):
            return None, self._content_ambiguity(
                items,
                [boundary],
                reason=(
                    "finite content containing an infinitival complement "
                    "exceeds the configured depth"
                ),
            )
        return (
            ContentSplit(
                matrix_tokens=matrix,
                content_tokens=content,
                marker="zero",
                relation_type=relation_type,
                predicate_family=family,
                certainty=max(0, certainty - 10),
                diagnostics=[
                    f"finite content predicate={predicate}",
                    "content marker=zero",
                    f"content boundary={boundary}",
                ],
            ),
            None,
        )

    def _content_layering_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Optional[ContentAttachmentAmbiguity]:
        """Reject unsupported combinations of content and relation layers.

        The check scans for any licensed finite content predicate rather than
        only the first predicate.  That matters for forms such as ``The man
        who left said that John arrived`` and fronted subordinate clauses,
        where the first finite verb belongs to a different relation layer.
        """

        items = list(tokens)
        content_predicates: List[int] = []
        for index, token in enumerate(items):
            predicate = lexicon.lemma(token.norm)
            if predicate not in self.CONTENT_PREDICATES:
                continue
            previous = items[index - 1].norm if index > 0 else None
            if lexicon.is_probable_verb(token.norm, previous=previous):
                content_predicates.append(index)
        if not content_predicates:
            return None

        first_predicate = content_predicates[0]
        norms = [item.norm for item in items]
        relative_markers = [
            index
            for index, norm in enumerate(norms)
            if norm in {"who", "whom", "whose", "which"}
            or (norm == "that" and index < first_predicate)
        ]
        if relative_markers:
            return self._content_ambiguity(
                items,
                [first_predicate + 1],
                reason=(
                    "finite content combined with a relative clause exceeds "
                    "this parser slice"
                ),
            )

        # A paired comma before the content predicate is the structural shape
        # of the supported appositive layer (for example ``Sarah, my
        # supervisor, said ...``).  Parsing it as a flat matrix subject would
        # invent a composite identity, so fail closed until staged composition.
        commas_before = [
            index
            for index in range(first_predicate)
            if norms[index] == ","
        ]
        if len(commas_before) >= 2:
            return self._content_ambiguity(
                items,
                [first_predicate + 1],
                reason=(
                    "finite content combined with an appositive requires "
                    "staged parsing"
                ),
            )

        # Any reviewed subordinate marker in the same segment creates a second
        # finite relation layer.  Preserve it as an explicit ambiguity rather
        # than treating the embedded proposition as a direct object string.
        for index in range(len(items)):
            for marker in self.SUBORDINATE_MARKERS:
                if tuple(norms[index : index + len(marker)]) == marker:
                    return self._content_ambiguity(
                        items,
                        [first_predicate + 1],
                        reason=(
                            "finite content combined with a subordinate clause "
                            "requires staged parsing"
                        ),
                    )
        return None

    def _contains_nested_content_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        verb_index = self._find_main_verb(tokens)
        if verb_index < 0:
            return False
        predicate = lexicon.lemma(tokens[verb_index].norm)
        if predicate not in self.CONTENT_PREDICATES:
            return False
        for index in range(verb_index + 1, len(tokens)):
            if tokens[index].norm == "that":
                suffix = [
                    item
                    for item in tokens[index + 1 :]
                    if item.norm not in lexicon.PUNCTUATION
                ]
                if self._is_independently_finite(suffix):
                    return True
            suffix = [
                item
                for item in tokens[index:]
                if item.norm not in lexicon.PUNCTUATION
            ]
            if self._is_independently_finite(suffix):
                return True
        return False

    def _contains_selected_infinitive(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        items = [item for item in tokens if item.norm not in lexicon.PUNCTUATION]
        verb_index = self._find_main_verb(items)
        if verb_index < 0:
            return False
        predicate = lexicon.lemma(items[verb_index].norm)
        if predicate not in self.INFINITIVAL_PREDICATES:
            return False
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
            if (
                next_index < len(items)
                and lexicon.is_probable_verb(items[next_index].norm, "to")
            ):
                return True
        return False

    def _content_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        boundaries: Sequence[int],
        *,
        reason: str,
    ) -> ContentAttachmentAmbiguity:
        first = boundaries[0] if boundaries else len(tokens)
        return ContentAttachmentAmbiguity(
            matrix_surface=self._surface(tokens[:first]),
            content_surface=self._surface(tokens[first:]),
            clause_surface=self._surface(tokens),
            reason=reason,
            candidate_boundaries=list(boundaries),
            ambiguity_id=(
                "content-"
                + hashlib.sha256(
                    " ".join(token.norm for token in tokens).encode("utf-8")
                ).hexdigest()[:16]
            ),
            diagnostics=[reason, "unsafe attributed content suppressed"],
        )

    def _plausible_content_subject(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> bool:
        verb_index = self._find_main_verb(tokens)
        if verb_index <= 0:
            return False
        subject = [
            item
            for item in tokens[:verb_index]
            if item.norm not in lexicon.AUXILIARIES
            and item.norm not in lexicon.NEGATORS
            and item.norm not in lexicon.INTENSIFIERS
        ]
        if not subject:
            return False
        first = subject[0].norm
        if first in lexicon.PRONOUN_FEATURES and len(subject) > 1:
            return False
        return True

    @staticmethod
    def _content_source_entity(event: EventFrame) -> str:
        for role in ("agent", "experiencer", "subject", "source"):
            ref = event.arguments.get(role)
            if ref is not None and ref.kind == RefKind.ENTITY:
                return ref.key
        return ""
