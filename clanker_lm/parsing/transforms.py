"""Transforms recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from .. import lexicon

class TransformsRules:
    """Stateless transforms transformations composed by SemanticParser."""

    @staticmethod
    def _detect_social(words: Sequence[str]) -> Optional[str]:
        compact = tuple(word for word in words if word not in {",", ":"})
        for pattern, name in lexicon.SOCIAL_QUESTIONS.items():
            if compact[: len(pattern)] == pattern and len(compact) <= len(pattern) + 2:
                return name
        if len(compact) <= 3 and compact[0] in lexicon.GREETINGS:
            return "greeting"
        return None

    def _split_assertion_segments(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> List[Tuple[List[lexicon.Token], Optional[str]]]:
        """Split only independently finite coordinated assertions.

        A conjunction is not enough evidence by itself: compound subjects,
        compound objects, and shared-subject/gapping coordination remain in one
        clause.  Each returned connector belongs to the clause on its right.
        """

        segments: List[Tuple[List[lexicon.Token], Optional[str]]] = []
        current: List[lexicon.Token] = []
        incoming_connector: Optional[str] = None
        index = 0
        while index < len(tokens):
            token = tokens[index]
            norm = token.norm
            connector: Optional[str] = None
            right_start = index + 1

            if norm == ";":
                connector = ";"
            elif norm in {"and", "but", "yet", "or", "so"}:
                # ``so that`` introduces a subordinate purpose/result clause,
                # not the independently finite resultative coordination covered
                # by this slice.
                if norm == "so" and right_start < len(tokens) and tokens[right_start].norm == "that":
                    current.append(token)
                    index += 1
                    continue
                if norm == "and" and right_start < len(tokens) and tokens[right_start].norm == "then":
                    connector = "and then"
                    right_start += 1
                else:
                    connector = norm

            if connector is not None:
                right = self._immediate_right_clause(tokens, right_start)
                if self._is_independently_finite(current) and self._is_independently_finite(right):
                    # Preserve commas until subordinate-clause analysis;
                    # ``_parse_clause`` removes punctuation after structural
                    # boundaries have consumed it.
                    cleaned_current = [
                        item for item in current if item.norm != ";"
                    ]
                    if cleaned_current:
                        segments.append((cleaned_current, incoming_connector))
                    current = []
                    incoming_connector = connector
                    index = right_start
                    continue

            current.append(token)
            index += 1

        cleaned_current = [item for item in current if item.norm != ";"]
        if cleaned_current:
            segments.append((cleaned_current, incoming_connector))
        return segments or [(list(tokens), None)]

    def _split_assertion_clauses(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> List[List[lexicon.Token]]:
        """Compatibility wrapper returning token lists without connectors."""

        return [clause for clause, _connector in self._split_assertion_segments(tokens)]

    @staticmethod
    def _immediate_right_clause(
        tokens: Sequence[lexicon.Token],
        start: int,
    ) -> List[lexicon.Token]:
        """Return lookahead only through the next coordination boundary.

        Without this bound, an early compound-object ``and`` could borrow a
        finite verb from a later ``but John left`` clause and split at the
        wrong connector.
        """

        end = len(tokens)
        for index in range(start, len(tokens)):
            norm = tokens[index].norm
            if norm == ";":
                end = index
                break
            if norm in {"and", "but", "yet", "or", "so"}:
                if (
                    norm == "so"
                    and index + 1 < len(tokens)
                    and tokens[index + 1].norm == "that"
                ):
                    continue
                end = index
                break
        return [
            item
            for item in tokens[start:end]
            if item.norm not in lexicon.PUNCTUATION
        ]

    def _is_independently_finite(self, tokens: Sequence[lexicon.Token]) -> bool:
        """Return whether a segment contains a finite predicate and subject."""

        items = [item for item in tokens if item.norm not in lexicon.PUNCTUATION]
        verb_index = self._find_main_verb(items)
        if verb_index <= 0:
            # A verb-initial segment may be an imperative or shared-subject
            # continuation.  Those require their own deterministic slice and
            # are deliberately not split here.
            return False
        subject_tokens = [
            item
            for item in items[:verb_index]
            if item.norm not in lexicon.AUXILIARIES
            and item.norm not in lexicon.NEGATORS
            and item.norm not in lexicon.INTENSIFIERS
            and item.norm not in {",", ";"}
        ]
        return bool(subject_tokens)

    @staticmethod
    def _rewrite_yoda(tokens: Sequence[lexicon.Token]) -> List[lexicon.Token]:
        """Normalize fronted-predicate forms such as ``Bought it, Sarah did``."""

        items = list(tokens)
        norms = [token.norm for token in items]
        if not items or norms[-1] not in {"did", "does", "will"}:
            return items
        if not lexicon.is_probable_verb(norms[0]):
            return items

        comma_index = norms.index(",") if "," in norms else -1
        if comma_index >= 0:
            front = [token for token in items[:comma_index] if token.norm not in lexicon.PUNCTUATION]
            subject = [token for token in items[comma_index + 1 : -1] if token.norm not in lexicon.PUNCTUATION]
        else:
            # Implicit comma: locate a late possessive/relation/proper-name chunk.
            subject_start = -1
            for idx in range(1, len(items) - 1):
                norm = items[idx].norm
                if norm in lexicon.POSSESSIVES or norm in lexicon.RELATIONS or items[idx].text[:1].isupper():
                    subject_start = idx
                    break
            if subject_start < 0:
                return items
            front = [token for token in items[:subject_start] if token.norm not in lexicon.PUNCTUATION]
            subject = [token for token in items[subject_start:-1] if token.norm not in lexicon.PUNCTUATION]
        if not front or not subject:
            return items
        rewritten = subject + front
        return [lexicon.Token(token.text, token.norm, idx) for idx, token in enumerate(rewritten)]
