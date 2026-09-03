"""Deterministic semantic frame and question parser.

The parser is deliberately bounded and auditable.  It does not claim to be a
full English grammar; it recognizes high-value proposition shapes, records
what it could not resolve, and routes ambiguity to a context probe instead of
inventing an antecedent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .memory import ConversationMemory, Resolution
from .models import (
    EntityKind,
    Gender,
    GrammaticalNumber,
    HowType,
    ParsedUtterance,
    QuestionFamily,
    QuestionFrame,
    RoleValue,
    SemanticFrame,
    SemanticRole,
    SpeechAct,
    UnresolvedReference,
    ValueKind,
    WhyType,
)
from .normalize import (
    AUXILIARIES,
    BODY_PARTS,
    CASUAL_MARKERS,
    COPULAS,
    DETERMINERS,
    DO_AUX,
    FEMALE_PRONOUNS,
    FIRST_PERSON,
    HAVE_AUX,
    HIGH_SEVERITY_MARKERS,
    MALE_PRONOUNS,
    MODALS,
    MOTION_VERBS,
    OBJECT_PRONOUNS,
    PHRASAL_VERBS,
    PLURAL_PRONOUNS,
    POSSESSIVES,
    PROFANITY,
    RELATION_ALIASES,
    REPETITION_MARKERS,
    SECOND_PERSON,
    SEVERE_MARKERS,
    TRANSFER_VERBS,
    VOLITIONAL_VERBS,
    PHYSICAL_EVENT_VERBS,
    canonical_relation,
    clean_phrase,
    detect_time_phrase,
    is_known_verb,
    infer_tense,
    lemmatize_verb,
    normalize_alias,
    relation_features,
    tokenize,
)


_WH_WORDS = {"who", "what", "which", "when", "where", "why", "how"}
_POLAR_START = set(AUXILIARIES)
_PUNCTUATION = {"?", "!", ".", ",", ";", ":"}
_LEADING_DISCOURSE = set(CASUAL_MARKERS) | {"well", "okay", "ok", "so", "like"}
_INTENSIFIERS = {"really", "very", "extremely", "super", "seriously", "so"}
_NEGATORS = {"not", "never", "no"}
_PARTICLES = {"off", "up", "out", "away", "down", "in", "on", "through"}
_CAUSE_MARKERS = (("because", "of"), ("due", "to"), ("because",), ("since",))
_PURPOSE_MARKERS = (("in", "order", "to"), ("so", "that"))
_METHOD_MARKERS = {"using", "via", "through", "by", "with"}
_LOCATION_PREPS = {"at", "in", "inside", "outside", "near", "to", "from", "into", "onto"}
_PATIENT_VERBS = {
    "hit",
    "hurt",
    "kill",
    "upset",
    "piss_off",
    "love",
    "hate",
    "help",
    "call",
    "cheat",
}
_STATE_ADJECTIVES = {
    "sick",
    "ill",
    "hurt",
    "angry",
    "mad",
    "sad",
    "happy",
    "fine",
    "okay",
    "safe",
    "dead",
    "missing",
    "tired",
    "scared",
    "worried",
    "pregnant",
    "ready",
    "broken",
    "red",
    "blue",
    "old",
    "new",
}


@dataclass(frozen=True)
class _NPResult:
    value: Optional[RoleValue]
    next_index: int
    unresolved: Optional[UnresolvedReference] = None


@dataclass(frozen=True)
class _PredicateParse:
    frame: Optional[SemanticFrame]
    unresolved: Tuple[UnresolvedReference, ...] = ()


class SemanticParser:
    """Parse statements and questions against a session entity memory."""

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory

    def parse(self, text: str) -> ParsedUtterance:
        lex = tokenize(text)
        raw_words = [token.surface for token in lex if token.norm not in _PUNCTUATION]
        items = [token.norm for token in lex if token.norm not in _PUNCTUATION]
        if not items:
            return ParsedUtterance(raw_text=text, speech_act=SpeechAct.UNKNOWN)

        register_score = self._register_score(items)
        familial = any(canonical_relation(item) for item in items)
        severity_score = self._severity_score(items, familial=familial)
        repeated = any(item in REPETITION_MARKERS for item in items)

        lowered = tuple(items)
        if self._is_greeting(lowered):
            return ParsedUtterance(
                raw_text=text,
                speech_act=SpeechAct.GREETING,
                tokens=lowered,
                register_score=register_score,
                severity_score=severity_score,
                familial=familial,
                repeated=repeated,
            )
        if self._is_social_checkin(lowered):
            question = QuestionFrame(
                family=QuestionFamily.SOCIAL,
                frame=SemanticFrame(predicate="state", surface=text),
                requested_roles=(SemanticRole.ATTRIBUTE,),
                expected_type="social_state",
                wh_word="how",
                how_type=HowType.SOCIAL,
                surface=text,
            )
            return ParsedUtterance(
                raw_text=text,
                speech_act=SpeechAct.SOCIAL_CHECKIN,
                question=question,
                tokens=lowered,
                register_score=register_score,
                severity_score=severity_score,
                familial=familial,
                repeated=repeated,
            )

        is_question = "?" in [token.norm for token in lex] or items[0] in _WH_WORDS | _POLAR_START
        if is_question:
            question, unresolved = self._parse_question(items, raw_words, text)
            return ParsedUtterance(
                raw_text=text,
                speech_act=SpeechAct.QUESTION,
                frame=question.frame if question else None,
                question=question,
                unresolved_references=tuple(unresolved),
                tokens=lowered,
                register_score=register_score,
                severity_score=severity_score,
                familial=familial,
                repeated=repeated,
            )

        statement = self._parse_statement(items, raw_words, text)
        return ParsedUtterance(
            raw_text=text,
            speech_act=SpeechAct.STATEMENT,
            frame=statement.frame,
            unresolved_references=statement.unresolved,
            tokens=lowered,
            register_score=register_score,
            severity_score=severity_score,
            familial=familial,
            repeated=repeated,
        )

    @staticmethod
    def _is_greeting(items: Sequence[str]) -> bool:
        normalized = " ".join(items)
        return normalized in {
            "hi",
            "hello",
            "hey",
            "yo",
            "good morning",
            "good afternoon",
            "good evening",
        }

    @staticmethod
    def _is_social_checkin(items: Sequence[str]) -> bool:
        normalized = " ".join(items)
        return normalized in {
            "how are you",
            "how have you been",
            "how is it going",
            "how are things",
            "you good",
        }

    @staticmethod
    def _register_score(items: Sequence[str]) -> float:
        casual = sum(1 for item in items if item in CASUAL_MARKERS)
        profane = sum(1 for item in items if item in PROFANITY)
        return min(1.0, casual * 0.35 + profane * 0.22)

    @staticmethod
    def _severity_score(items: Sequence[str], *, familial: bool) -> float:
        score = 0.0
        if any(item in HIGH_SEVERITY_MARKERS for item in items):
            score = 0.88
        elif any(item in SEVERE_MARKERS for item in items):
            score = 0.52
        if any(item in _INTENSIFIERS for item in items) and score:
            score += 0.12
        if familial and score:
            score += 0.08
        if "tummy" in items and "hurt" in items:
            score = min(score, 0.38)
        return min(1.0, score)

    def _parse_question(
        self, items: Sequence[str], raw_words: Sequence[str], surface: str
    ) -> tuple[Optional[QuestionFrame], List[UnresolvedReference]]:
        first = items[0]
        if first == "how" and len(items) > 1 and items[1] in {
            "many",
            "much",
            "long",
            "old",
            "tall",
            "fast",
            "far",
            "often",
            "quickly",
            "slowly",
            "well",
            "badly",
            "bad",
        }:
            return self._parse_how_measure(items, raw_words, surface)
        if first in {"why", "how", "when", "where", "what", "who", "which"}:
            return self._parse_wh_question(items, raw_words, surface)
        if first in _POLAR_START:
            return self._parse_polar_question(items, raw_words, surface)
        return None, []

    def _parse_wh_question(
        self, items: Sequence[str], raw_words: Sequence[str], surface: str
    ) -> tuple[Optional[QuestionFrame], List[UnresolvedReference]]:
        wh = items[0]
        unresolved: List[UnresolvedReference] = []

        rhetorical = self._looks_rhetorical(items)
        if rhetorical:
            frame = SemanticFrame(predicate="rhetorical", surface=surface)
            question = QuestionFrame(
                family=QuestionFamily.WHY if wh == "why" else QuestionFamily.WH,
                frame=frame,
                requested_roles=(),
                expected_type="pragmatic_support",
                wh_word=wh,
                why_type=WhyType.RHETORICAL if wh == "why" else WhyType.UNKNOWN,
                rhetorical=True,
                surface=surface,
            )
            return question, unresolved

        # Subject WH: "Who bought the Honda?" / "What broke the window?"
        if len(items) > 1 and items[1] not in AUXILIARIES and is_known_verb(items[1]):
            predicate = lemmatize_verb(items[1])
            tail = list(items[2:])
            raw_tail = list(raw_words[2:]) if len(raw_words) >= 2 else tail
            frame = SemanticFrame(
                predicate=predicate, tense=infer_tense(items[1]), surface=surface
            )
            requested_role = SemanticRole.AGENT
            expected = "person" if wh == "who" else "entity"
            object_parse = self._parse_tail(
                predicate, tail, raw_tail, subject=None, missing_roles=(requested_role,)
            )
            frame.roles.update(object_parse.frame.roles if object_parse.frame else {})
            unresolved.extend(object_parse.unresolved)
            if wh == "what" and predicate in {"cause", "break", "happen"}:
                expected = "cause_or_event"
            question = QuestionFrame(
                family=QuestionFamily.WH,
                frame=frame,
                requested_roles=(requested_role,),
                expected_type=expected,
                wh_word=wh,
                surface=surface,
            )
            return question, unresolved

        if len(items) < 2:
            return None, unresolved

        aux = items[1]
        if aux not in AUXILIARIES:
            return self._parse_copular_wh_without_aux(items, raw_words, surface)

        polarity = True
        cursor = 2
        if cursor < len(items) and items[cursor] == "not":
            polarity = False
            cursor += 1

        # Copular WH: what/who/where/how + is + subject/complement
        if aux in COPULAS:
            if wh == "where":
                subject = self._parse_np(items, raw_words, cursor, len(items), expected_kind=EntityKind.UNKNOWN)
                if subject.unresolved:
                    unresolved.append(subject.unresolved)
                frame = SemanticFrame(predicate="be", polarity=polarity, surface=surface)
                if subject.value:
                    frame.roles[SemanticRole.AGENT] = subject.value
                return (
                    QuestionFrame(
                        family=QuestionFamily.WH,
                        frame=frame,
                        requested_roles=(SemanticRole.LOCATION,),
                        expected_type="place",
                        wh_word=wh,
                        surface=surface,
                    ),
                    unresolved,
                )
            if wh == "when":
                subject = self._parse_np(items, raw_words, cursor, len(items), expected_kind=EntityKind.UNKNOWN)
                if subject.unresolved:
                    unresolved.append(subject.unresolved)
                frame = SemanticFrame(predicate="be", polarity=polarity, surface=surface)
                if subject.value:
                    frame.roles[SemanticRole.AGENT] = subject.value
                return (
                    QuestionFrame(
                        family=QuestionFamily.WH,
                        frame=frame,
                        requested_roles=(SemanticRole.TIME,),
                        expected_type="time",
                        wh_word=wh,
                        surface=surface,
                    ),
                    unresolved,
                )
            if wh == "how":
                subject = self._parse_np(items, raw_words, cursor, len(items), expected_kind=EntityKind.UNKNOWN)
                if subject.unresolved:
                    unresolved.append(subject.unresolved)
                frame = SemanticFrame(predicate="be", polarity=polarity, surface=surface)
                if subject.value:
                    frame.roles[SemanticRole.AGENT] = subject.value
                return (
                    QuestionFrame(
                        family=QuestionFamily.HOW,
                        frame=frame,
                        requested_roles=(SemanticRole.ATTRIBUTE, SemanticRole.VALUE),
                        expected_type="condition",
                        wh_word=wh,
                        how_type=HowType.CONDITION,
                        surface=surface,
                    ),
                    unresolved,
                )
            # "What is the meeting?" / "Who is Sarah?"
            subject = self._parse_np(items, raw_words, cursor, len(items), expected_kind=EntityKind.UNKNOWN)
            if subject.unresolved:
                unresolved.append(subject.unresolved)
            frame = SemanticFrame(predicate="be", polarity=polarity, surface=surface)
            if subject.value:
                frame.roles[SemanticRole.AGENT] = subject.value
            return (
                QuestionFrame(
                    family=QuestionFamily.WH,
                    frame=frame,
                    requested_roles=(SemanticRole.VALUE, SemanticRole.ATTRIBUTE),
                    expected_type="identity_or_attribute",
                    wh_word=wh,
                    surface=surface,
                ),
                unresolved,
            )

        subject, verb_index = self._parse_question_subject(items, raw_words, cursor)
        if subject.unresolved:
            unresolved.append(subject.unresolved)
        if verb_index < len(items) and items[verb_index] == "not":
            polarity = False
            verb_index += 1
        if verb_index >= len(items):
            frame = SemanticFrame(predicate=lemmatize_verb(aux), polarity=polarity, surface=surface)
        else:
            predicate = lemmatize_verb(items[verb_index])
            frame = SemanticFrame(
                predicate=predicate,
                tense="past" if aux == "did" else "future" if aux == "will" else "present",
                polarity=polarity,
                modality=aux if aux in MODALS else None,
                surface=surface,
            )
            if subject.value:
                frame.roles[SemanticRole.AGENT] = subject.value
            tail = list(items[verb_index + 1 :])
            raw_tail = list(raw_words[verb_index + 1 :]) if len(raw_words) > verb_index else tail

            requested_roles: Tuple[SemanticRole, ...]
            family = QuestionFamily.WH
            expected_type = "unknown"
            why_type = WhyType.UNKNOWN
            how_type = HowType.UNKNOWN

            if wh in {"what", "which"}:
                requested_roles = (self._object_role(predicate),)
                expected_type = "entity"
            elif wh == "who":
                # Trailing "to" asks recipient: "Who did she give the book to?"
                if predicate in TRANSFER_VERBS and tail and tail[-1] == "to":
                    requested_roles = (SemanticRole.RECIPIENT,)
                    tail = tail[:-1]
                    raw_tail = raw_tail[:-1]
                else:
                    requested_roles = (self._object_role(predicate),)
                expected_type = "person"
            elif wh == "when":
                requested_roles = (SemanticRole.TIME,)
                expected_type = "time"
            elif wh == "where":
                requested_roles = (SemanticRole.LOCATION,)
                expected_type = "place"
            elif wh == "why":
                family = QuestionFamily.WHY
                why_type, requested_roles = self._classify_why(items, predicate, subject.value)
                expected_type = why_type.value
            elif wh == "how":
                family = QuestionFamily.HOW
                how_type, requested_roles = self._classify_how(items, predicate, aux)
                expected_type = how_type.value
            else:
                requested_roles = (SemanticRole.VALUE,)

            tail_parse = self._parse_tail(
                predicate,
                tail,
                raw_tail,
                subject=subject.value,
                missing_roles=requested_roles,
            )
            if tail_parse.frame:
                for role, value in tail_parse.frame.roles.items():
                    if role not in requested_roles:
                        frame.roles[role] = value
                frame.repeated = tail_parse.frame.repeated
            unresolved.extend(tail_parse.unresolved)
            question = QuestionFrame(
                family=family,
                frame=frame,
                requested_roles=requested_roles,
                expected_type=expected_type,
                wh_word=wh,
                why_type=why_type,
                how_type=how_type,
                surface=surface,
            )
            if unresolved:
                question.unresolved_reference = unresolved[0]
            return question, unresolved

        return None, unresolved

    def _parse_copular_wh_without_aux(
        self, items: Sequence[str], raw_words: Sequence[str], surface: str
    ) -> tuple[Optional[QuestionFrame], List[UnresolvedReference]]:
        # Bare conversational forms such as "where my keys" are intentionally
        # not guessed into facts; they receive an unknown frame.
        frame = SemanticFrame(predicate="unknown", surface=surface)
        return (
            QuestionFrame(
                family=QuestionFamily.WH,
                frame=frame,
                requested_roles=(SemanticRole.VALUE,),
                expected_type="unknown",
                wh_word=items[0],
                surface=surface,
            ),
            [],
        )

    def _parse_how_measure(
        self, items: Sequence[str], raw_words: Sequence[str], surface: str
    ) -> tuple[Optional[QuestionFrame], List[UnresolvedReference]]:
        measure = items[1]
        unresolved: List[UnresolvedReference] = []
        if measure in {"many", "much"}:
            role = SemanticRole.QUANTITY
            how_type = HowType.QUANTITY
        elif measure in {"quickly", "slowly", "well", "badly"}:
            role = SemanticRole.MANNER
            how_type = HowType.MANNER
        else:
            role = SemanticRole.DEGREE
            how_type = HowType.DEGREE

        # Quantity questions carry a constrained object before the auxiliary:
        # "How many books did Sarah buy?" is BUY(agent=Sarah, theme=books,
        # quantity=?).  Reuse the ordinary WH parser for the proposition, then
        # add the measured noun as a known role.
        aux_index = next(
            (index for index in range(2, len(items)) if items[index] in AUXILIARIES),
            None,
        )
        if aux_index is not None:
            measured_items = list(items[2:aux_index])
            measured_raw = list(raw_words[2:aux_index])
            reduced = ["how"] + list(items[aux_index:])
            reduced_raw = ["how"] + list(raw_words[aux_index:])
            question, unresolved = self._parse_wh_question(reduced, reduced_raw, surface)
            if question is not None and measured_items and question.frame.predicate != "unknown":
                measured = self._parse_np(
                    measured_items,
                    measured_raw,
                    0,
                    len(measured_items),
                    expected_kind=EntityKind.OBJECT,
                )
                if measured.value:
                    question.frame.roles[self._object_role(question.frame.predicate)] = measured.value
                if measured.unresolved:
                    unresolved.append(measured.unresolved)
        else:
            # Copular measurement: "How tall is Sarah?"
            reduced = ["how"] + list(items[2:])
            reduced_raw = ["how"] + list(raw_words[2:])
            question, unresolved = self._parse_wh_question(reduced, reduced_raw, surface)

        if question is None:
            question = QuestionFrame(
                family=QuestionFamily.HOW,
                frame=SemanticFrame(predicate="unknown", surface=surface),
                requested_roles=(role,),
                expected_type=measure,
                wh_word="how",
                how_type=how_type,
                surface=surface,
            )
        else:
            question.family = QuestionFamily.HOW
            question.requested_roles = (role,)
            question.expected_type = measure
            question.how_type = how_type
            question.surface = surface
            if unresolved:
                question.unresolved_reference = unresolved[0]
        return question, unresolved

    def _parse_polar_question(
        self, items: Sequence[str], raw_words: Sequence[str], surface: str
    ) -> tuple[Optional[QuestionFrame], List[UnresolvedReference]]:
        aux = items[0]
        cursor = 1
        polarity = True
        unresolved: List[UnresolvedReference] = []
        if cursor < len(items) and items[cursor] == "not":
            polarity = False
            cursor += 1

        if aux in COPULAS:
            # Subject ends before a state adjective or location preposition.
            complement_index = self._find_copula_complement(items, cursor)
            subject = self._parse_np(
                items,
                raw_words,
                cursor,
                complement_index,
                expected_kind=EntityKind.UNKNOWN,
            )
            if subject.unresolved:
                unresolved.append(subject.unresolved)
            frame = SemanticFrame(
                predicate="be",
                tense="past" if aux in {"was", "were"} else "present",
                polarity=polarity,
                surface=surface,
            )
            if subject.value:
                frame.roles[SemanticRole.AGENT] = subject.value
            complement = list(items[complement_index:])
            raw_complement = list(raw_words[complement_index:]) if len(raw_words) >= complement_index else complement
            if complement:
                if complement[0] in _LOCATION_PREPS:
                    value = clean_phrase(raw_complement[1:] or raw_complement)
                    if value:
                        frame.roles[SemanticRole.LOCATION] = self._value_for_phrase(
                            value, expected_kind=EntityKind.PLACE
                        )
                else:
                    value = clean_phrase(raw_complement)
                    if value:
                        frame.roles[SemanticRole.ATTRIBUTE] = self.memory.text_value(value)
        else:
            subject, verb_index = self._parse_question_subject(items, raw_words, cursor)
            if subject.unresolved:
                unresolved.append(subject.unresolved)
            if verb_index < len(items) and items[verb_index] == "not":
                polarity = False
                verb_index += 1
            if verb_index >= len(items):
                return None, unresolved
            predicate = lemmatize_verb(items[verb_index])
            frame = SemanticFrame(
                predicate=predicate,
                tense="past" if aux in {"did", "had"} else "future" if aux == "will" else "present",
                polarity=polarity,
                modality=aux if aux in MODALS else None,
                surface=surface,
            )
            if subject.value:
                frame.roles[SemanticRole.AGENT] = subject.value
            tail = list(items[verb_index + 1 :])
            raw_tail = list(raw_words[verb_index + 1 :]) if len(raw_words) > verb_index else tail
            tail_parse = self._parse_tail(predicate, tail, raw_tail, subject=subject.value)
            if tail_parse.frame:
                frame.roles.update(tail_parse.frame.roles)
                frame.repeated = tail_parse.frame.repeated
            unresolved.extend(tail_parse.unresolved)

        question = QuestionFrame(
            family=QuestionFamily.POLAR,
            frame=frame,
            requested_roles=(),
            expected_type="boolean",
            surface=surface,
        )
        if unresolved:
            question.unresolved_reference = unresolved[0]
        return question, unresolved

    def _parse_statement(
        self, items: Sequence[str], raw_words: Sequence[str], surface: str
    ) -> _PredicateParse:
        if self._looks_like_yoda_inversion(items):
            return self._parse_yoda_inversion(items, raw_words, surface)

        start = 0
        while start < len(items) and items[start] in _LEADING_DISCOURSE:
            start += 1
        if start >= len(items):
            return _PredicateParse(None)

        subject_end = self._find_statement_subject_end(items, start)
        if subject_end <= start or subject_end >= len(items):
            return _PredicateParse(None)
        subject = self._parse_np(
            items,
            raw_words,
            start,
            subject_end,
            expected_kind=self._subject_expected_kind(
                items, raw_words, start, subject_end
            ),
        )
        unresolved: List[UnresolvedReference] = []
        if subject.unresolved:
            unresolved.append(subject.unresolved)

        cursor = subject_end
        polarity = True
        modality: Optional[str] = None
        tense = "present"
        token = items[cursor]

        if token in COPULAS:
            tense = "past" if token in {"was", "were"} else "present"
            cursor += 1
            if cursor < len(items) and items[cursor] == "not":
                polarity = False
                cursor += 1
            frame = SemanticFrame(
                predicate="be",
                tense=tense,
                polarity=polarity,
                surface=surface,
            )
            if subject.value:
                frame.roles[SemanticRole.AGENT] = subject.value
            tail = list(items[cursor:])
            raw_tail = list(raw_words[cursor:]) if len(raw_words) >= cursor else tail
            self._fill_copular_complement(frame, tail, raw_tail)
            return _PredicateParse(frame, tuple(unresolved))

        if token in DO_AUX | HAVE_AUX | MODALS:
            if token == "did" or token == "had":
                tense = "past"
            elif token == "will":
                tense = "future"
            if token in MODALS:
                modality = token
            cursor += 1
            if cursor < len(items) and items[cursor] == "not":
                polarity = False
                cursor += 1
            if cursor >= len(items):
                return _PredicateParse(None, tuple(unresolved))
            token = items[cursor]

        predicate = lemmatize_verb(token)
        if not predicate:
            return _PredicateParse(None, tuple(unresolved))
        if token.endswith("ed") or token in {
            "bought",
            "left",
            "went",
            "gave",
            "saw",
            "ate",
            "made",
            "took",
            "got",
            "found",
            "sent",
            "wrote",
            "broke",
            "hurt",
            "felt",
            "thought",
            "knew",
            "ran",
            "came",
            "paid",
            "spoke",
            "won",
            "lost",
            "lied",
        }:
            tense = "past"
        cursor += 1
        tail = list(items[cursor:])
        raw_tail = list(raw_words[cursor:]) if len(raw_words) >= cursor else tail

        frame = SemanticFrame(
            predicate=predicate,
            tense=tense,
            polarity=polarity,
            modality=modality,
            repeated=any(item in REPETITION_MARKERS for item in items),
            surface=surface,
        )
        if subject.value:
            frame.roles[SemanticRole.AGENT] = subject.value
        tail_parse = self._parse_tail(predicate, tail, raw_tail, subject=subject.value)
        if tail_parse.frame:
            frame.predicate = tail_parse.frame.predicate
            frame.roles.update(tail_parse.frame.roles)
            frame.repeated = frame.repeated or tail_parse.frame.repeated
        # In a normative modal statement, a because-clause supplies the reason
        # the action is warranted, not merely a physical event cause.
        if modality in {"should", "must"} and SemanticRole.CAUSE in frame.roles:
            frame.roles.setdefault(
                SemanticRole.JUSTIFICATION,
                frame.roles.pop(SemanticRole.CAUSE),
            )
        unresolved.extend(tail_parse.unresolved)
        return _PredicateParse(frame, tuple(unresolved))

    def _parse_tail(
        self,
        predicate: str,
        tail: Sequence[str],
        raw_tail: Sequence[str],
        *,
        subject: Optional[RoleValue],
        missing_roles: Sequence[SemanticRole] = (),
    ) -> _PredicateParse:
        frame = SemanticFrame(predicate=predicate)
        unresolved: List[UnresolvedReference] = []
        if not tail:
            return _PredicateParse(frame)

        items = list(tail)
        raws = list(raw_tail) if len(raw_tail) == len(tail) else list(tail)
        repeated = any(item in REPETITION_MARKERS for item in items)
        frame.repeated = repeated

        # Remove discourse-only tokens at the end.
        while items and items[-1] in CASUAL_MARKERS:
            items.pop()
            raws.pop()

        # Capture causal/purpose clauses first so their contents are not treated
        # as the direct object.
        cause_span = self._find_marker(items, _CAUSE_MARKERS)
        if cause_span:
            marker_start, marker_end = cause_span
            cause_text = clean_phrase(raws[marker_end:])
            if cause_text:
                frame.roles[SemanticRole.CAUSE] = self.memory.text_value(cause_text)
            items = items[:marker_start]
            raws = raws[:marker_start]

        purpose_span = self._find_marker(items, _PURPOSE_MARKERS)
        if purpose_span:
            marker_start, marker_end = purpose_span
            purpose_text = clean_phrase(raws[marker_end:])
            if purpose_text:
                frame.roles[SemanticRole.PURPOSE] = self.memory.text_value(purpose_text)
            items = items[:marker_start]
            raws = raws[:marker_start]

        time_text, time_span = detect_time_phrase(items)
        if time_span:
            start, end = time_span
            frame.roles[SemanticRole.TIME] = self.memory.text_value(
                clean_phrase(raws[start:end]), kind=ValueKind.TIME
            )
            del items[start:end]
            del raws[start:end]

        # Strip repeat markers after retaining the repeated flag.
        kept = [
            (item, raw)
            for item, raw in zip(items, raws)
            if item not in REPETITION_MARKERS
        ]
        items = [item for item, _ in kept]
        raws = [raw for _, raw in kept]

        # Phrasal verb may be adjacent or wrap an object: "pissed me off".
        if items and (predicate, items[0]) in PHRASAL_VERBS:
            frame.predicate = PHRASAL_VERBS[(predicate, items[0])]
            items = items[1:]
            raws = raws[1:]
        elif items and (predicate, items[-1]) in PHRASAL_VERBS:
            frame.predicate = PHRASAL_VERBS[(predicate, items[-1])]
            items = items[:-1]
            raws = raws[:-1]
        predicate = frame.predicate

        # Method phrases are explicit and deterministic.
        method_index = next((i for i, item in enumerate(items) if item in _METHOD_MARKERS), None)
        if method_index is not None:
            method_text = clean_phrase(raws[method_index:])
            if method_text:
                frame.roles[SemanticRole.METHOD] = self.memory.text_value(method_text)
            items = items[:method_index]
            raws = raws[:method_index]

        # Motion verbs route prepositional destinations into LOCATION.
        if predicate in MOTION_VERBS:
            location_index = next(
                (i for i, item in enumerate(items) if item in _LOCATION_PREPS), None
            )
            if location_index is not None:
                location_phrase = clean_phrase(raws[location_index + 1 :])
                if location_phrase:
                    frame.roles[SemanticRole.LOCATION] = self._value_for_phrase(
                        location_phrase, expected_kind=EntityKind.PLACE
                    )
                items = items[:location_index]
                raws = raws[:location_index]

        # Transfer verbs can have a recipient introduced by "to".
        if predicate in TRANSFER_VERBS and "to" in items:
            to_index = items.index("to")
            recipient_phrase = clean_phrase(raws[to_index + 1 :])
            if recipient_phrase:
                recipient = self._parse_np(
                    items,
                    raws,
                    to_index + 1,
                    len(items),
                    expected_kind=EntityKind.PERSON,
                )
                if recipient.value:
                    frame.roles[SemanticRole.RECIPIENT] = recipient.value
                if recipient.unresolved:
                    unresolved.append(recipient.unresolved)
            items = items[:to_index]
            raws = raws[:to_index]

        # Infinitive remaining after an object is a purpose.
        if "to" in items:
            to_index = items.index("to")
            if to_index + 1 < len(items) and is_known_verb(items[to_index + 1]):
                purpose_text = clean_phrase(raws[to_index + 1 :])
                if purpose_text:
                    frame.roles[SemanticRole.PURPOSE] = self.memory.text_value(purpose_text)
                items = items[:to_index]
                raws = raws[:to_index]

        # Numeric quantifiers become their own semantic role so the same
        # proposition can answer "how many" without storing "three books" as
        # one opaque object string.
        if items and self._is_quantity_token(items[0]):
            frame.roles[SemanticRole.QUANTITY] = self.memory.text_value(
                raws[0], kind=ValueKind.NUMBER
            )
            items = items[1:]
            raws = raws[1:]

        # English transfer verbs permit a double-object frame:
        # "Sarah gave John the book".  The question's missing role disambiguates
        # the one-NP forms ("What did Sarah give John?" / "Who did Sarah give
        # the book to?").
        if predicate in TRANSFER_VERBS and items and "to" not in items:
            missing = set(missing_roles)
            if SemanticRole.THEME in missing and SemanticRole.RECIPIENT not in missing:
                recipient = self._parse_np(
                    items, raws, 0, len(items), expected_kind=EntityKind.PERSON
                )
                if recipient.value:
                    frame.roles[SemanticRole.RECIPIENT] = recipient.value
                if recipient.unresolved:
                    unresolved.append(recipient.unresolved)
                return _PredicateParse(frame, tuple(unresolved))
            if SemanticRole.RECIPIENT in missing:
                theme = self._parse_np(
                    items, raws, 0, len(items), expected_kind=EntityKind.OBJECT
                )
                if theme.value:
                    frame.roles[SemanticRole.THEME] = theme.value
                if theme.unresolved:
                    unresolved.append(theme.unresolved)
                return _PredicateParse(frame, tuple(unresolved))

            recipient_end = self._transfer_recipient_end(items, raws)
            if 0 < recipient_end < len(items):
                recipient = self._parse_np(
                    items, raws, 0, recipient_end, expected_kind=EntityKind.PERSON
                )
                theme = self._parse_np(
                    items, raws, recipient_end, len(items), expected_kind=EntityKind.OBJECT
                )
                if recipient.value:
                    frame.roles[SemanticRole.RECIPIENT] = recipient.value
                if theme.value:
                    frame.roles[SemanticRole.THEME] = theme.value
                if recipient.unresolved:
                    unresolved.append(recipient.unresolved)
                if theme.unresolved:
                    unresolved.append(theme.unresolved)
                return _PredicateParse(frame, tuple(unresolved))

        object_phrase = clean_phrase(raws)
        if object_phrase:
            object_result = self._parse_np(
                items,
                raws,
                0,
                len(items),
                expected_kind=(
                    EntityKind.PERSON if predicate in _PATIENT_VERBS else EntityKind.UNKNOWN
                ),
            )
            if object_result.value:
                frame.roles[self._object_role(predicate)] = object_result.value
            if object_result.unresolved:
                unresolved.append(object_result.unresolved)

        return _PredicateParse(frame, tuple(unresolved))

    def _fill_copular_complement(
        self, frame: SemanticFrame, tail: Sequence[str], raw_tail: Sequence[str]
    ) -> None:
        if not tail:
            return
        items = list(tail)
        raws = list(raw_tail) if len(raw_tail) == len(tail) else list(tail)
        time_text, time_span = detect_time_phrase(items)
        if time_span and time_span[0] == 0:
            start, end = time_span
            frame.roles[SemanticRole.TIME] = self.memory.text_value(
                clean_phrase(raws[start:end]), kind=ValueKind.TIME
            )
            return
        if items[0] in _LOCATION_PREPS:
            place_text = clean_phrase(raws[1:])
            if place_text:
                frame.roles[SemanticRole.LOCATION] = self._value_for_phrase(
                    place_text, expected_kind=EntityKind.PLACE
                )
            return
        attribute_text = clean_phrase(raws)
        if attribute_text:
            frame.roles[SemanticRole.ATTRIBUTE] = self.memory.text_value(attribute_text)

    def _parse_question_subject(
        self, items: Sequence[str], raw_words: Sequence[str], start: int
    ) -> tuple[_NPResult, int]:
        if start >= len(items):
            return _NPResult(None, start), start
        end = self._subject_phrase_end(items, start)
        if end <= start:
            end = start + 1
        # Generic noun phrases continue until the first lexical verb.
        if end == start + 1 and items[start] in DETERMINERS:
            for index in range(start + 1, len(items)):
                if items[index] == "not" or is_known_verb(items[index]):
                    end = index
                    break
        elif end == start + 1 and items[start] not in (
            FIRST_PERSON | SECOND_PERSON | FEMALE_PRONOUNS | MALE_PRONOUNS | PLURAL_PRONOUNS | OBJECT_PRONOUNS
        ):
            for index in range(start + 1, len(items)):
                if items[index] == "not" or is_known_verb(items[index]):
                    end = index
                    break
        expected = self._subject_expected_kind(items, raw_words, start, end)
        result = self._parse_np(items, raw_words, start, end, expected_kind=expected)
        return result, end

    @staticmethod
    def _subject_expected_kind(
        items: Sequence[str], raw_words: Sequence[str], start: int, end: int
    ) -> EntityKind:
        if end - start == 1 and start < len(raw_words):
            raw = raw_words[start]
            token = items[start]
            if (
                raw[:1].isupper()
                and token not in DETERMINERS
                and token not in {"i"}
                and canonical_relation(token) is None
            ):
                return EntityKind.PERSON
        return EntityKind.UNKNOWN

    def _parse_np(
        self,
        items: Sequence[str],
        raw_words: Sequence[str],
        start: int,
        end: int,
        *,
        expected_kind: EntityKind,
    ) -> _NPResult:
        if start >= end or start >= len(items):
            return _NPResult(None, start)
        end = min(end, len(items))
        phrase_items = list(items[start:end])
        phrase_raw = list(raw_words[start:end]) if len(raw_words) >= end else phrase_items
        if not phrase_items:
            return _NPResult(None, end)

        first = phrase_items[0]
        if len(phrase_items) == 1 and first in (
            FIRST_PERSON
            | SECOND_PERSON
            | FEMALE_PRONOUNS
            | MALE_PRONOUNS
            | PLURAL_PRONOUNS
            | OBJECT_PRONOUNS
        ):
            resolution = self.memory.resolve_pronoun(first)
            if resolution.resolved:
                return _NPResult(self.memory.entity_value(resolution.entity_id), end)
            return _NPResult(
                None,
                end,
                UnresolvedReference(
                    surface=first,
                    expected_kind=expected_kind,
                    compatible_entity_ids=resolution.candidates,
                    reason=resolution.reason,
                ),
            )

        # "my older sister" / "my tummy"
        if first in POSSESSIVES:
            relation_index = next(
                (
                    index
                    for index, item in enumerate(phrase_items[1:], start=1)
                    if canonical_relation(item)
                ),
                None,
            )
            if first == "my" and relation_index is not None:
                relation = canonical_relation(phrase_items[relation_index])
                assert relation is not None
                gender, number = relation_features(relation)
                display = clean_phrase(phrase_raw[1 : relation_index + 1]) or relation
                entity = self.memory.upsert_entity(
                    relation,
                    display_name=relation,
                    kind=EntityKind.PERSON,
                    gender=gender,
                    number=number,
                    relation_to_user=relation,
                    aliases={display, phrase_items[relation_index]},
                )
                return _NPResult(self.memory.entity_value(entity.entity_id), end)
            if first == "my" and len(phrase_items) >= 2 and phrase_items[-1] in BODY_PARTS:
                body = phrase_items[-1]
                entity = self.memory.upsert_entity(
                    body,
                    display_name=body,
                    kind=EntityKind.BODY_PART,
                    gender=Gender.NEUTRAL,
                    relation_to_user=body,
                    aliases={clean_phrase(phrase_items[1:])},
                    metadata={"owner": "user"},
                )
                return _NPResult(self.memory.entity_value(entity.entity_id), end)

        phrase = clean_phrase(phrase_raw)
        normalized = normalize_alias(phrase)
        if not normalized:
            return _NPResult(None, end)

        resolution = self.memory.resolve_alias(normalized, expected_kind=expected_kind)
        if resolution.resolved:
            return _NPResult(self.memory.entity_value(resolution.entity_id, display=phrase), end)
        if resolution.candidates:
            return _NPResult(
                None,
                end,
                UnresolvedReference(
                    surface=phrase,
                    expected_kind=expected_kind,
                    compatible_entity_ids=resolution.candidates,
                    reason=resolution.reason,
                ),
            )

        stripped_items = [item for item in phrase_items if item not in DETERMINERS]
        head = stripped_items[-1] if stripped_items else phrase_items[-1]
        relation = canonical_relation(head)
        if relation:
            gender, number = relation_features(relation)
            entity = self.memory.upsert_entity(
                relation,
                display_name=relation,
                kind=EntityKind.PERSON,
                gender=gender,
                number=number,
                aliases={phrase, head},
            )
            return _NPResult(self.memory.entity_value(entity.entity_id), end)

        kind = expected_kind
        if kind == EntityKind.UNKNOWN:
            kind = EntityKind.OBJECT
        determiner = phrase_items[0] if phrase_items[0] in DETERMINERS else None
        display = clean_phrase(
            phrase_raw[1:] if determiner and len(phrase_raw) > 1 else phrase_raw
        )
        aliases = {phrase, display, head}
        entity = self.memory.upsert_entity(
            normalized,
            display_name=display or phrase,
            kind=kind,
            gender=Gender.NEUTRAL if kind != EntityKind.PERSON else Gender.UNKNOWN,
            number=(
                GrammaticalNumber.PLURAL
                if head.endswith("s") and not head.endswith("ss")
                else GrammaticalNumber.SINGULAR
            ),
            determiner=determiner,
            aliases=aliases,
        )
        return _NPResult(self.memory.entity_value(entity.entity_id), end)

    def _value_for_phrase(self, phrase: str, *, expected_kind: EntityKind) -> RoleValue:
        normalized = normalize_alias(phrase)
        resolution = self.memory.resolve_alias(normalized, expected_kind=expected_kind)
        if resolution.resolved:
            return self.memory.entity_value(resolution.entity_id, display=phrase)
        entity = self.memory.upsert_entity(
            normalized,
            display_name=phrase,
            kind=expected_kind,
            gender=Gender.NEUTRAL,
            aliases={phrase},
        )
        return self.memory.entity_value(entity.entity_id)

    @staticmethod
    def _is_quantity_token(token: str) -> bool:
        return token.isdigit() or token in {
            "zero", "one", "two", "three", "four", "five", "six",
            "seven", "eight", "nine", "ten", "eleven", "twelve",
            "dozen", "hundred", "thousand",
        }

    def _transfer_recipient_end(
        self, items: Sequence[str], raw_items: Sequence[str]
    ) -> int:
        if len(items) < 2:
            return 0
        first = items[0]
        if first in (
            FIRST_PERSON
            | SECOND_PERSON
            | FEMALE_PRONOUNS
            | MALE_PRONOUNS
            | PLURAL_PRONOUNS
        ):
            return 1
        if first in POSSESSIVES:
            return self._subject_phrase_end(items, 0)
        if first in DETERMINERS:
            # "the teacher the book"; keep a relation/adjective+noun together.
            return min(2, len(items) - 1)
        if raw_items and raw_items[0][:1].isupper():
            return 1
        resolution = self.memory.resolve_alias(first, expected_kind=EntityKind.PERSON)
        if resolution.resolved:
            return 1
        return 0

    @staticmethod
    def _find_marker(
        items: Sequence[str], markers: Sequence[Sequence[str]]
    ) -> Optional[tuple[int, int]]:
        for start in range(len(items)):
            for marker in markers:
                marker_tuple = tuple(marker)
                if tuple(items[start : start + len(marker_tuple)]) == marker_tuple:
                    return start, start + len(marker_tuple)
        return None

    @staticmethod
    def _object_role(predicate: str) -> SemanticRole:
        return SemanticRole.PATIENT if predicate in _PATIENT_VERBS else SemanticRole.THEME

    @staticmethod
    def _classify_why(
        items: Sequence[str], predicate: str, subject: Optional[RoleValue]
    ) -> tuple[WhyType, Tuple[SemanticRole, ...]]:
        if any(item in {"should", "must", "ought", "allowed"} for item in items):
            return WhyType.JUSTIFICATION, (SemanticRole.JUSTIFICATION,)
        if predicate in {"think", "believe", "claim", "say", "know"}:
            return WhyType.EVIDENCE, (SemanticRole.EVIDENCE, SemanticRole.CAUSE)
        if predicate in PHYSICAL_EVENT_VERBS:
            return WhyType.CAUSE, (SemanticRole.CAUSE,)
        if predicate in VOLITIONAL_VERBS:
            return WhyType.MOTIVE, (
                SemanticRole.MOTIVE,
                SemanticRole.PURPOSE,
                SemanticRole.CAUSE,
            )
        return WhyType.CAUSE, (SemanticRole.CAUSE, SemanticRole.MOTIVE)

    @staticmethod
    def _classify_how(
        items: Sequence[str], predicate: str, auxiliary: str
    ) -> tuple[HowType, Tuple[SemanticRole, ...]]:
        if predicate in {"work", "calculate", "operate", "function", "learn"} or auxiliary in {"do", "does"}:
            return HowType.PROCESS, (
                SemanticRole.PROCESS,
                SemanticRole.MECHANISM,
                SemanticRole.METHOD,
            )
        if auxiliary == "did" or predicate in VOLITIONAL_VERBS:
            return HowType.METHOD, (SemanticRole.METHOD, SemanticRole.MANNER)
        return HowType.MANNER, (SemanticRole.MANNER, SemanticRole.METHOD)

    @staticmethod
    def _looks_rhetorical(items: Sequence[str]) -> bool:
        if not items or items[0] not in {"why", "how"}:
            return False
        has_modal = any(item in {"would", "could", "should"} for item in items)
        has_indefinite = any(item in {"anyone", "anybody", "someone", "somebody"} for item in items)
        has_self = any(item in {"me", "my", "myself", "i"} for item in items)
        return has_modal and has_indefinite and has_self

    @staticmethod
    def _subject_phrase_end(items: Sequence[str], start: int) -> int:
        if start >= len(items):
            return start
        first = items[start]
        if first in (
            FIRST_PERSON
            | SECOND_PERSON
            | FEMALE_PRONOUNS
            | MALE_PRONOUNS
            | PLURAL_PRONOUNS
            | OBJECT_PRONOUNS
        ):
            return start + 1
        if first in POSSESSIVES:
            for index in range(start + 1, min(len(items), start + 5)):
                if canonical_relation(items[index]) or items[index] in BODY_PARTS:
                    return index + 1
            return min(len(items), start + 2)
        return start + 1

    def _find_statement_subject_end(self, items: Sequence[str], start: int) -> int:
        direct = self._subject_phrase_end(items, start)
        if direct > start + 1 or items[start] in (
            FIRST_PERSON
            | SECOND_PERSON
            | FEMALE_PRONOUNS
            | MALE_PRONOUNS
            | PLURAL_PRONOUNS
            | OBJECT_PRONOUNS
        ):
            return direct
        for index in range(start + 1, len(items)):
            if items[index] in AUXILIARIES or is_known_verb(items[index]):
                return index
        return direct

    @staticmethod
    def _find_copula_complement(items: Sequence[str], start: int) -> int:
        for index in range(start + 1, len(items)):
            if items[index] in _STATE_ADJECTIVES or items[index] in _LOCATION_PREPS:
                return index
        return max(start + 1, len(items) - 1)

    @staticmethod
    def _looks_like_yoda_inversion(items: Sequence[str]) -> bool:
        return (
            len(items) >= 4
            and is_known_verb(items[0])
            and items[-1] == "did"
            and any(canonical_relation(item) for item in items[1:-1])
        )

    def _parse_yoda_inversion(
        self, items: Sequence[str], raw_words: Sequence[str], surface: str
    ) -> _PredicateParse:
        relation_pos = next(
            index for index in range(1, len(items) - 1) if canonical_relation(items[index])
        )
        subject_start = relation_pos - 1 if relation_pos > 0 and items[relation_pos - 1] in POSSESSIVES else relation_pos
        subject = self._parse_np(
            items,
            raw_words,
            subject_start,
            relation_pos + 1,
            expected_kind=EntityKind.PERSON,
        )
        predicate = lemmatize_verb(items[0])
        front_tail = list(items[1:subject_start])
        raw_front = list(raw_words[1:subject_start]) if len(raw_words) >= subject_start else front_tail
        frame = SemanticFrame(predicate=predicate, tense="past", surface=surface)
        if subject.value:
            frame.roles[SemanticRole.AGENT] = subject.value
        tail_parse = self._parse_tail(predicate, front_tail, raw_front, subject=subject.value)
        unresolved: List[UnresolvedReference] = []
        if subject.unresolved:
            unresolved.append(subject.unresolved)
        if tail_parse.frame:
            frame.predicate = tail_parse.frame.predicate
            frame.roles.update(tail_parse.frame.roles)
        unresolved.extend(tail_parse.unresolved)
        return _PredicateParse(frame, tuple(unresolved))
