"""Embedded recognition and transformation. Shared state is passed explicitly as memory."""

from __future__ import annotations
import hashlib
from typing import List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import EmbeddedInterrogativeAttachmentAmbiguity, EmbeddedInterrogativeType, EntityKind, QuestionFrame, QuestionKind, SemanticRef, UnresolvedReference
from .types import EmbeddedInterrogativeSplit

class EmbeddedRules:
    """Stateless embedded transformations composed by SemanticParser."""

    def _split_embedded_interrogative_clause(
        self,
        tokens: Sequence[lexicon.Token],
    ) -> Tuple[
        Optional[EmbeddedInterrogativeSplit],
        Optional[EmbeddedInterrogativeAttachmentAmbiguity],
    ]:
        """Split one catalog-licensed matrix predicate and embedded question.

        A WH-shaped relative modifier after an ordinary object NP is not enough
        evidence for interrogative content.  Only reviewed matrix predicates
        enter this path, and non-recipient predicates require the marker to
        follow the matrix verb directly.
        """

        items = [
            item
            for item in tokens
            if item.norm not in {";"}
        ]
        if not items:
            return None, None
        marker_set = self.EMBEDDED_WH_MARKERS | self.EMBEDDED_POLAR_MARKERS
        predicate_candidates: List[Tuple[int, str]] = []
        for index, token in enumerate(items):
            predicate = lexicon.lemma(token.norm)
            if predicate not in self.EMBEDDED_INTERROGATIVE_PREDICATES:
                continue
            previous = items[index - 1].norm if index else None
            if not lexicon.is_probable_verb(token.norm, previous=previous):
                continue
            if any(item.norm in marker_set for item in items[index + 1 :]):
                predicate_candidates.append((index, predicate))
        if not predicate_candidates:
            return None, None
        if len(predicate_candidates) > 1:
            return None, self._embedded_interrogative_ambiguity(
                items,
                [index for index, _predicate in predicate_candidates],
                reason=(
                    "multiple embedded-interrogative matrix predicates exceed "
                    "the configured depth"
                ),
            )

        verb_index, predicate = predicate_candidates[0]
        profile = self.EMBEDDED_INTERROGATIVE_PREDICATES[predicate]
        marker_candidates = [
            index
            for index in range(verb_index + 1, len(items))
            if items[index].norm in marker_set
        ]
        if not marker_candidates:
            return None, None
        if len(marker_candidates) > 1:
            return None, self._embedded_interrogative_ambiguity(
                items,
                marker_candidates,
                reason=(
                    "nested or competing embedded interrogative markers "
                    "exceed the configured depth"
                ),
            )

        marker_index = marker_candidates[0]
        marker = items[marker_index].norm
        matrix_tail = [
            item
            for item in items[verb_index + 1 : marker_index]
            if item.norm not in lexicon.PUNCTUATION
        ]
        if matrix_tail and not profile.allows_recipient:
            # ``I know the man who called`` is a relative/object structure, not
            # an embedded question.  Leave it for the ordinary relation layers.
            return None, None
        if matrix_tail:
            if any(
                lexicon.is_probable_verb(
                    item.norm,
                    previous=(matrix_tail[index - 1].norm if index else None),
                )
                for index, item in enumerate(matrix_tail)
            ):
                return None, self._embedded_interrogative_ambiguity(
                    items,
                    [marker_index],
                    reason="matrix recipient contains a competing predicate",
                )
            if len(matrix_tail) > 4:
                return None, self._embedded_interrogative_ambiguity(
                    items,
                    [marker_index],
                    reason=(
                        "embedded-question recipient boundary is not "
                        "structurally bounded"
                    ),
                )

        question_tokens = [
            item
            for item in items[marker_index:]
            if item.norm not in lexicon.PUNCTUATION
        ]
        if len(question_tokens) < 2:
            return None, self._embedded_interrogative_ambiguity(
                items,
                [marker_index],
                reason="embedded interrogative lacks a proposition",
            )
        relation_type = (
            EmbeddedInterrogativeType.POLAR
            if marker in self.EMBEDDED_POLAR_MARKERS
            else EmbeddedInterrogativeType.WH
        )
        imperative = verb_index == 0
        direct_answer_request = bool(
            imperative and profile.allows_direct_answer_request
        )
        return (
            EmbeddedInterrogativeSplit(
                matrix_tokens=[
                    item
                    for item in items[:marker_index]
                    if item.norm not in lexicon.PUNCTUATION
                ],
                question_tokens=question_tokens,
                marker=marker,
                relation_type=relation_type,
                profile=profile,
                direct_answer_request=direct_answer_request,
                diagnostics=[
                    f"embedded interrogative predicate={predicate}",
                    f"embedded marker={marker}",
                    f"embedded type={relation_type.value}",
                ],
            ),
            None,
        )

    def _embedded_interrogative_ambiguity(
        self,
        tokens: Sequence[lexicon.Token],
        boundaries: Sequence[int],
        *,
        reason: str,
    ) -> EmbeddedInterrogativeAttachmentAmbiguity:
        first = boundaries[0] if boundaries else len(tokens)
        markers = [
            tokens[index].norm
            for index in boundaries
            if 0 <= index < len(tokens)
        ]
        return EmbeddedInterrogativeAttachmentAmbiguity(
            matrix_surface=self._surface(tokens[:first]),
            question_surface=self._surface(tokens[first:]),
            clause_surface=self._surface(tokens),
            reason=reason,
            candidate_boundaries=list(boundaries),
            candidate_markers=markers,
            ambiguity_id=(
                "embedded-question-"
                + hashlib.sha256(
                    " ".join(item.norm for item in tokens).encode("utf-8")
                ).hexdigest()[:16]
            ),
            diagnostics=[reason, "unsafe embedded question suppressed"],
        )

    def _parse_embedded_question(
        self,
        tokens: Sequence[lexicon.Token],
        raw: str,
        memory: ConversationMemory,
    ) -> Tuple[
        Optional[QuestionFrame],
        List[UnresolvedReference],
        List[str],
        List[str],
    ]:
        """Parse one indirect WH or polar question without matrix inversion."""

        items = [
            item
            for item in tokens
            if item.norm not in lexicon.PUNCTUATION
        ]
        if len(items) < 2:
            return None, [], [], ["embedded question lacks content"]
        marker = items[0].norm
        body = list(items[1:])
        if marker in self.EMBEDDED_POLAR_MARKERS:
            if body[:2] and [item.norm for item in body[:2]] == ["or", "not"]:
                body = body[2:]
            clause = self._parse_clause(body, raw, memory)
            if clause.event is None:
                return None, clause.unresolved, clause.entities, [
                    *clause.diagnostics,
                    "failed embedded polar proposition parse",
                ]
            frame = QuestionFrame(
                kind=QuestionKind.YES_NO,
                event=clause.event,
                raw_text=raw,
                unresolved=clause.unresolved,
                embedded_interrogative_type=EmbeddedInterrogativeType.POLAR,
                embedded_marker=marker,
            )
            return (
                frame,
                clause.unresolved,
                clause.entities,
                [*clause.diagnostics, f"embedded polar marker={marker}"],
            )

        if marker in {"when", "where", "why", "how"}:
            result = self._parse_adverbial_question(items, raw, memory)
            frame = result[0]
            if frame is not None:
                frame.embedded_interrogative_type = EmbeddedInterrogativeType.WH
                frame.embedded_marker = marker
            return result

        if marker == "whose":
            # Indirect subject form: ``whose car broke`` keeps the matrix
            # predicate (BREAK) and adds a typed possessor gap to the subject
            # entity.  The ordinary top-level WHOSE parser maps ownership
            # questions to OWN, which would otherwise discard ``broke`` here.
            if (
                len(body) >= 2
                and body[0].norm not in lexicon.AUXILIARIES
                and lexicon.is_probable_verb(
                    body[1].norm,
                    previous=body[0].norm,
                )
            ):
                clause = self._parse_clause(body, raw, memory)
                if clause.event is None:
                    return None, clause.unresolved, clause.entities, [
                        *clause.diagnostics,
                        "failed embedded whose proposition parse",
                    ]
                clause.event.arguments["possessor"] = SemanticRef.variable(
                    "possessor",
                    EntityKind.PERSON,
                )
                frame = QuestionFrame(
                    kind=QuestionKind.WHOSE,
                    event=clause.event,
                    requested_role="possessor",
                    answer_type=EntityKind.PERSON,
                    raw_text=raw,
                    unresolved=clause.unresolved,
                    focus_surface=body[0].text,
                    embedded_interrogative_type=EmbeddedInterrogativeType.WH,
                    embedded_marker=marker,
                )
                return (
                    frame,
                    clause.unresolved,
                    clause.entities,
                    [
                        *clause.diagnostics,
                        "embedded whose preserves possessed-subject predicate",
                    ],
                )
            result = self._parse_whose(items, raw, memory)
            frame = result[0]
            if frame is not None:
                frame.embedded_interrogative_type = EmbeddedInterrogativeType.WH
                frame.embedded_marker = marker
            return result

        if marker in {"who", "whom", "what"}:
            subject_gap = bool(
                body
                and lexicon.is_probable_verb(body[0].norm, previous=None)
            )
            if marker == "what" and body and body[0].norm in {"happened", "happen"}:
                result = self._parse_question(items, raw, memory)
                frame = result[0]
                if frame is not None:
                    frame.embedded_interrogative_type = EmbeddedInterrogativeType.WH
                    frame.embedded_marker = marker
                return result
            if subject_gap and marker != "whom":
                helper = self._parse_who if marker == "who" else self._parse_what
                result = helper(items, raw, memory)
                frame = result[0]
                if frame is not None:
                    frame.embedded_interrogative_type = EmbeddedInterrogativeType.WH
                    frame.embedded_marker = marker
                return result

            clause = self._parse_clause(body, raw, memory)
            if clause.event is None:
                return None, clause.unresolved, clause.entities, [
                    *clause.diagnostics,
                    "failed embedded WH proposition parse",
                ]
            requested_role = "patient"
            answer_type = (
                EntityKind.PERSON
                if marker in {"who", "whom"}
                else EntityKind.THING
            )
            clause.event.arguments[requested_role] = SemanticRef.variable(
                requested_role,
                answer_type,
            )
            frame = QuestionFrame(
                kind=(
                    QuestionKind.WHO
                    if marker in {"who", "whom"}
                    else QuestionKind.WHAT
                ),
                event=clause.event,
                requested_role=requested_role,
                answer_type=answer_type,
                raw_text=raw,
                unresolved=clause.unresolved,
                embedded_interrogative_type=EmbeddedInterrogativeType.WH,
                embedded_marker=marker,
            )
            return (
                frame,
                clause.unresolved,
                clause.entities,
                [
                    *clause.diagnostics,
                    f"embedded {marker} requests {requested_role}",
                ],
            )

        if marker == "which":
            if len(body) < 2:
                return None, [], [], ["embedded which lacks selection class"]
            focus = body[0].text
            clause = self._parse_clause(body[1:], raw, memory)
            if clause.event is None:
                return None, clause.unresolved, clause.entities, [
                    *clause.diagnostics,
                    "failed embedded which proposition parse",
                ]
            clause.event.arguments["patient"] = SemanticRef.variable(
                "patient",
                EntityKind.THING,
            )
            frame = QuestionFrame(
                kind=QuestionKind.WHICH,
                event=clause.event,
                requested_role="patient",
                answer_type=EntityKind.THING,
                raw_text=raw,
                unresolved=clause.unresolved,
                focus_surface=focus,
                embedded_interrogative_type=EmbeddedInterrogativeType.WH,
                embedded_marker=marker,
            )
            return (
                frame,
                clause.unresolved,
                clause.entities,
                [*clause.diagnostics, f"embedded selection class={focus}"],
            )

        return None, [], [], [f"unsupported embedded marker={marker}"]
