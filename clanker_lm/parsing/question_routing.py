"""Question Routing recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import EmbeddedInterrogativeType, EntityKind, EventFrame, HowKind, QuestionFrame, QuestionKind, SemanticRef, SourceKind, UnresolvedReference, WhyKind

class QuestionRoutingRules:
    """Stateless question routing transformations composed by SemanticParser."""

    def _parse_question(
        self,
        tokens: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        items = [token for token in tokens if token.norm not in lexicon.PUNCTUATION]
        words = [token.norm for token in items]
        diagnostics: List[str] = []
        if not words:
            return None, [], [], ["empty question"]

        # Conventional social questions bypass literal query semantics.
        social = self._detect_social(words)
        if social:
            dummy = EventFrame("social", raw_text=raw, source=SourceKind.USER)
            frame = QuestionFrame(
                kind=QuestionKind.HOW if words[0] == "how" else QuestionKind.WHAT,
                event=dummy,
                raw_text=raw,
                social_convention=social,
            )
            return frame, [], [], [f"social convention: {social}"]

        outer_embedded = self._parse_outer_embedded_interrogative(
            items,
            raw,
            memory,
        )
        if outer_embedded is not None:
            return outer_embedded

        if words[:2] == ["what", "happened"] or words[:3] == ["what", "has", "happened"]:
            event = EventFrame("*", {"event": SemanticRef.variable("event", EntityKind.EVENT)}, raw_text=raw)
            return QuestionFrame(
                kind=QuestionKind.WHAT_HAPPENED,
                event=event,
                requested_role="event",
                answer_type=EntityKind.EVENT,
                raw_text=raw,
            ), [], [], ["open event query"]

        def finalize_question(result):
            frame = result[0]
            if frame is not None:
                self._annotate_infinitival_question_scope(frame, items)
                self._annotate_gerund_question_scope(frame, items)
            return result

        first = words[0]
        if first in lexicon.YES_NO_STARTERS:
            return finalize_question(self._parse_yes_no(items, raw, memory))
        if first == "who" or first == "whom":
            return finalize_question(self._parse_who(items, raw, memory))
        if first == "whose":
            return finalize_question(self._parse_whose(items, raw, memory))
        if first == "what":
            return finalize_question(self._parse_what(items, raw, memory))
        if first == "which":
            return finalize_question(self._parse_which(items, raw, memory))
        if first in {"when", "where", "why", "how"}:
            return finalize_question(
                self._parse_adverbial_question(items, raw, memory)
            )
        return None, [], [], ["unrecognized interrogative form"]

    def _parse_outer_embedded_interrogative(
        self,
        items: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Optional[
        Tuple[
            Optional[QuestionFrame],
            List[UnresolvedReference],
            List[str],
            List[str],
        ]
    ]:
        """Preserve an outer question and one mentioned inner question.

        For example, ``Do you remember when the meeting starts?`` is a polar
        question about the assistant's memory.  It is not silently rewritten
        into the direct inner ``when`` query.  The inner frame remains attached
        for evidence lookup and transparent traces.
        """

        if len(items) < 4:
            return None
        marker_set = self.EMBEDDED_WH_MARKERS | self.EMBEDDED_POLAR_MARKERS
        start = 1 if items[0].norm in self.EMBEDDED_WH_MARKERS else 0
        candidates = [
            index
            for index in range(max(1, start), len(items))
            if items[index].norm in marker_set
        ]
        if not candidates:
            return None
        marker_index = candidates[0]
        prefix = list(items[:marker_index])
        if not prefix:
            return None

        first = prefix[0].norm
        if first in lexicon.YES_NO_STARTERS:
            outer = self._parse_yes_no(prefix, raw, memory)
        elif first in {"who", "whom"}:
            outer = self._parse_who(prefix, raw, memory)
        elif first == "whose":
            outer = self._parse_whose(prefix, raw, memory)
        elif first == "what":
            outer = self._parse_what(prefix, raw, memory)
        elif first == "which":
            outer = self._parse_which(prefix, raw, memory)
        elif first in {"when", "where", "why", "how"}:
            outer = self._parse_adverbial_question(prefix, raw, memory)
        else:
            return None
        frame, outer_unresolved, outer_entities, outer_diagnostics = outer
        if frame is None:
            return None
        matrix_predicate = lexicon.lemma(frame.event.predicate)
        if matrix_predicate not in self.EMBEDDED_INTERROGATIVE_PREDICATES:
            return None

        inner, inner_unresolved, inner_entities, inner_diagnostics = (
            self._parse_embedded_question(
                items[marker_index:],
                raw,
                memory,
            )
        )
        if inner is None:
            unresolved = [
                *outer_unresolved,
                UnresolvedReference(
                    surface=self._surface(items[marker_index:]),
                    reason="embedded question could not be parsed",
                ),
                *inner_unresolved,
            ]
            frame.unresolved = unresolved
            return (
                frame,
                unresolved,
                list(dict.fromkeys(outer_entities + inner_entities)),
                [
                    *outer_diagnostics,
                    *inner_diagnostics,
                    "outer question retained; embedded scope unresolved",
                ],
            )

        frame.embedded_question = inner
        frame.embedded_interrogative_type = (
            EmbeddedInterrogativeType.POLAR
            if items[marker_index].norm in self.EMBEDDED_POLAR_MARKERS
            else EmbeddedInterrogativeType.WH
        )
        frame.embedded_marker = items[marker_index].norm
        frame.embedded_matrix_predicate = matrix_predicate
        if len(candidates) > 1:
            ambiguity = UnresolvedReference(
                surface=self._surface(items[marker_index:]),
                reason=(
                    "multiple embedded question markers exceed the "
                    "configured depth"
                ),
            )
            frame.unresolved = [
                *outer_unresolved,
                ambiguity,
                *inner_unresolved,
            ]
        else:
            frame.unresolved = [*outer_unresolved, *inner_unresolved]
        return (
            frame,
            frame.unresolved,
            list(dict.fromkeys(outer_entities + inner_entities)),
            [
                *outer_diagnostics,
                *inner_diagnostics,
                f"outer question matrix={matrix_predicate}",
                f"embedded question marker={frame.embedded_marker}",
                "outer and inner speech acts preserved separately",
            ],
        )

    def _parse_yes_no(
        self,
        items: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[Optional[QuestionFrame], List[UnresolvedReference], List[str], List[str]]:
        aux = items[0]
        rest = list(items[1:])
        if not rest:
            return None, [], [], ["yes/no question lacks proposition"]
        main_idx = self._find_main_verb(rest, start=0)
        if aux.norm in lexicon.COPULAS:
            # Is Sarah a nurse? -> Sarah is a nurse.
            declarative = rest[:1] + [aux] + rest[1:]
        elif main_idx >= 0:
            # Did Sarah buy a car? -> Sarah did buy a car.
            declarative = rest[:main_idx] + [aux] + rest[main_idx:]
        else:
            declarative = rest + [aux]
        matrix_polarity, embedded_polarity = (
            self._question_infinitival_polarities(rest, main_idx)
        )
        clause = self._parse_clause(declarative, raw, memory)
        if not clause.event:
            return None, clause.unresolved, clause.entities, clause.diagnostics + ["failed yes/no proposition parse"]
        if matrix_polarity is not None:
            clause.event.polarity = matrix_polarity
        frame = QuestionFrame(
            kind=QuestionKind.YES_NO,
            event=clause.event,
            raw_text=raw,
            unresolved=clause.unresolved,
            matrix_polarity=matrix_polarity,
            embedded_polarity=embedded_polarity,
        )
        return frame, clause.unresolved, clause.entities, clause.diagnostics + [f"yes/no auxiliary={aux.norm}"]

    def _annotate_infinitival_question_scope(
        self,
        frame: QuestionFrame,
        items: Sequence[lexicon.Token],
    ) -> None:
        """Attach two-level polarity to any licensed infinitival question."""

        predicate = frame.event.predicate
        if predicate not in self.INFINITIVAL_PREDICATES:
            return
        predicate_index = next(
            (
                index
                for index, token in enumerate(items)
                if lexicon.lemma(token.norm) == predicate
            ),
            -1,
        )
        if predicate_index < 0:
            return
        boundary = next(
            (
                index
                for index in range(predicate_index + 1, len(items) - 1)
                if items[index].norm == "to"
                and any(
                    lexicon.is_probable_verb(items[next_index].norm, "to")
                    for next_index in range(
                        index + 1,
                        min(len(items), index + 4),
                    )
                    if items[next_index].norm not in (
                        lexicon.NEGATORS | lexicon.INTENSIFIERS
                    )
                )
            ),
            -1,
        )
        if boundary < 0:
            return
        matrix_negative = any(
            token.norm in lexicon.NEGATORS
            for token in items[:predicate_index]
        )
        embedded_negative = any(
            token.norm in lexicon.NEGATORS
            for token in items[predicate_index + 1 :]
        )
        frame.matrix_polarity = not matrix_negative
        frame.embedded_polarity = not embedded_negative
        frame.event.polarity = frame.matrix_polarity

    def _question_infinitival_polarities(
        self,
        rest: Sequence[lexicon.Token],
        main_idx: int,
    ) -> Tuple[Optional[bool], Optional[bool]]:
        """Return matrix and embedded polarity for a selected infinitive.

        General clause parsing has one polarity bit, but a question such as
        ``Did Sarah plan not to leave?`` contains two independently scoped
        propositions.  Preserve that distinction for the infinitival answer
        contract instead of treating embedded negation as matrix negation.
        """

        if main_idx < 0 or main_idx >= len(rest):
            return None, None
        predicate = lexicon.lemma(rest[main_idx].norm)
        if predicate not in self.INFINITIVAL_PREDICATES:
            return None, None

        boundary = next(
            (
                index
                for index in range(main_idx + 1, len(rest) - 1)
                if rest[index].norm == "to"
                and any(
                    lexicon.is_probable_verb(rest[next_index].norm, "to")
                    for next_index in range(index + 1, min(len(rest), index + 4))
                    if rest[next_index].norm not in lexicon.NEGATORS
                    | lexicon.INTENSIFIERS
                )
            ),
            -1,
        )
        if boundary < 0:
            return None, None

        matrix_negative = any(
            token.norm in lexicon.NEGATORS
            for token in rest[:main_idx]
        )
        embedded_negative = any(
            token.norm in lexicon.NEGATORS
            for token in rest[main_idx + 1 :]
        )
        return not matrix_negative, not embedded_negative

    def _annotate_gerund_question_scope(
        self,
        frame: QuestionFrame,
        items: Sequence[lexicon.Token],
    ) -> None:
        """Preserve matrix and embedded polarity in selected -ing questions."""

        predicate = lexicon.lemma(frame.event.predicate)
        if predicate not in self.GERUND_PREDICATES:
            return
        predicate_index = next(
            (
                index
                for index, token in enumerate(items)
                if lexicon.lemma(token.norm) == predicate
                and lexicon.is_probable_verb(
                    token.norm,
                    previous=(items[index - 1].norm if index else None),
                    following=(
                        items[index + 1].norm
                        if index + 1 < len(items)
                        else None
                    ),
                )
            ),
            -1,
        )
        if predicate_index < 0:
            return
        boundary = next(
            (
                index
                for index in range(predicate_index + 1, len(items))
                if self._is_selected_ing_token(items, index)
            ),
            -1,
        )
        if boundary < 0:
            return

        matrix_negative = any(
            token.norm in lexicon.NEGATORS
            for token in items[:predicate_index]
        )
        embedded_negative = any(
            token.norm in lexicon.NEGATORS
            for token in items[predicate_index + 1 :]
        )
        frame.matrix_polarity = not matrix_negative
        frame.embedded_polarity = not embedded_negative
        frame.event.polarity = frame.matrix_polarity

    @staticmethod
    def _variable_token(role: str) -> lexicon.Token:
        return lexicon.Token(f"?{role}", f"__var_{role}__", -1)

    @staticmethod
    def _object_insertion_index(verb_and_tail: Sequence[lexicon.Token]) -> int:
        """Place an object variable before temporal/prepositional adjuncts."""

        if not verb_and_tail:
            return 0
        for idx in range(1, len(verb_and_tail)):
            norm = verb_and_tail[idx].norm
            if norm in lexicon.PREPOSITIONS or norm in lexicon.TIME_WORDS:
                return idx
        return len(verb_and_tail)

    @staticmethod
    def _classify_why(event: EventFrame) -> WhyKind:
        if event.modality in {"should", "must", "ought", "need"}:
            return WhyKind.JUSTIFICATION
        if event.predicate in {"think", "believe", "know", "claim", "say"}:
            return WhyKind.EVIDENCE
        if event.predicate in lexicon.PHYSICAL_EVENT_VERBS:
            return WhyKind.CAUSE
        if event.predicate in lexicon.PURPOSE_LIKELY_VERBS:
            return WhyKind.PURPOSE
        if event.predicate in lexicon.VOLITIONAL_VERBS:
            return WhyKind.MOTIVE
        return WhyKind.CAUSE

    @staticmethod
    def _classify_how(event: EventFrame) -> HowKind:
        if event.predicate in lexicon.PROCESS_VERBS:
            return HowKind.PROCESS
        if event.predicate in lexicon.PHYSICAL_EVENT_VERBS:
            return HowKind.MECHANISM
        if event.predicate == "be":
            return HowKind.STATE
        return HowKind.METHOD
