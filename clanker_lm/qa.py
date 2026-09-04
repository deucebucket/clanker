"""Typed question answering over Clanker-LM's symbolic memory."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import lexicon
from .memory import ConversationMemory, EventMatch
from .model import (
    AnswerContract,
    AnswerStatus,
    EmbeddedInterrogativeRelation,
    EntityKind,
    Evidence,
    EventFrame,
    GerundRelation,
    GerundRelationType,
    HowKind,
    InfinitivalRelation,
    QuestionFrame,
    QuestionKind,
    RefKind,
    SemanticRef,
    SourceKind,
    TruthValue,
    WhyKind,
)


class QuestionAnswerer:
    """Bind a question's typed hole from explicit evidence in memory."""

    WHY_FALLBACKS: Dict[WhyKind, Tuple[str, ...]] = {
        WhyKind.CAUSE: ("cause", "motive", "purpose", "justification", "evidence"),
        WhyKind.MOTIVE: ("motive", "cause", "purpose"),
        WhyKind.PURPOSE: ("purpose", "motive", "cause"),
        WhyKind.JUSTIFICATION: ("justification", "cause", "motive"),
        WhyKind.EVIDENCE: ("evidence", "cause", "justification"),
        WhyKind.UNKNOWN: ("cause", "motive", "purpose", "justification", "evidence"),
    }
    HOW_FALLBACKS: Dict[HowKind, Tuple[str, ...]] = {
        HowKind.METHOD: ("method", "manner", "process", "mechanism"),
        HowKind.MANNER: ("manner", "method"),
        HowKind.MECHANISM: ("mechanism", "cause", "method", "process"),
        HowKind.PROCESS: ("process", "method", "mechanism"),
        HowKind.DEGREE: ("value", "quantity"),
        HowKind.QUANTITY: ("quantity",),
        HowKind.STATE: ("value", "state"),
        HowKind.UNKNOWN: ("method", "manner", "process", "mechanism", "value"),
    }
    QUESTION_ROLE_FALLBACKS: Dict[QuestionKind, Dict[str, Tuple[str, ...]]] = {
        QuestionKind.WHERE: {
            "location": ("location", "destination", "goal"),
            "destination": ("destination", "location", "goal"),
            "source": ("source",),
        },
        QuestionKind.WHEN: {"time": ("time",)},
        QuestionKind.WHOSE: {"possessor": ("possessor", "owner")},
    }

    def answer(self, question: QuestionFrame, memory: ConversationMemory) -> AnswerContract:
        if question.social_convention:
            return AnswerContract(
                status=AnswerStatus.ANSWERED,
                question=question,
                certainty=255,
                source=SourceKind.TRAINED,
                response_goal="social",
                reason=question.social_convention,
            )

        if question.unresolved:
            unresolved = question.unresolved[0]
            if unresolved.candidates:
                return AnswerContract(
                    status=AnswerStatus.AMBIGUOUS_REFERENCE,
                    question=question,
                    certainty=255,
                    source=SourceKind.INFERRED,
                    reason=unresolved.surface,
                    required_slots={
                        "reference": unresolved.surface,
                        "candidate_ids": ",".join(unresolved.candidates),
                    },
                    response_goal="clarify",
                )
            return AnswerContract(
                status=AnswerStatus.MISSING_REFERENCE,
                question=question,
                certainty=255,
                source=SourceKind.INFERRED,
                reason=unresolved.surface,
                required_slots={"reference": unresolved.surface},
                response_goal="clarify",
            )

        if question.kind == QuestionKind.WHAT_HAPPENED:
            return self._answer_event_query(question, memory)
        embedded = self._answer_embedded_interrogative_request(
            question,
            memory,
        )
        if embedded is not None:
            return embedded
        attributed = self._answer_attributed_content_request(question, memory)
        if attributed is not None:
            return attributed
        gerund = self._answer_gerund_request(question, memory)
        if gerund is not None:
            return gerund
        infinitival = self._answer_infinitival_request(question, memory)
        if infinitival is not None:
            return infinitival
        if question.kind == QuestionKind.WHOSE:
            ownership = self._answer_possessor(question, memory)
            if ownership is not None:
                return ownership
        if question.kind == QuestionKind.YES_NO:
            return self._answer_yes_no(question, memory)
        if not question.requested_role:
            return AnswerContract(
                status=AnswerStatus.UNSUPPORTED,
                question=question,
                reason="question has no typed open slot",
                response_goal="clarify",
            )
        return self._answer_open_slot(question, memory)


    EMBEDDED_INTERROGATIVE_QUERY_PREDICATES = {
        "ask", "wonder", "know", "remember", "discover", "determine", "tell",
    }
    EPISTEMIC_SELF_QUERY_PREDICATES = {"know", "remember"}

    def _answer_embedded_interrogative_request(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        """Answer questions about attributed questions without solving them.

        The only exception is an explicit first-party memory/knowledge probe
        such as ``Do you remember when the meeting starts?``.  That path asks
        the ordinary fact store whether the inner question is answerable, but
        still returns a contract for the outer polar question.
        """

        predicate = lexicon.lemma(question.event.predicate)
        if predicate not in self.EMBEDDED_INTERROGATIVE_QUERY_PREDICATES:
            return None

        source_ref = (
            question.event.arguments.get("agent")
            or question.event.arguments.get("experiencer")
            or question.event.arguments.get("subject")
        )
        inner = question.embedded_question

        if (
            inner is not None
            and question.kind == QuestionKind.YES_NO
            and predicate in self.EPISTEMIC_SELF_QUERY_PREDICATES
            and source_ref is not None
            and source_ref.kind == RefKind.ENTITY
            and source_ref.key == "assistant"
        ):
            inner_contract = self.answer(inner, memory)
            known = inner_contract.status in {
                AnswerStatus.ANSWERED,
                AnswerStatus.TRUE,
                AnswerStatus.FALSE,
            }
            return AnswerContract(
                status=AnswerStatus.TRUE if known else AnswerStatus.FALSE,
                question=question,
                proposition=(
                    inner_contract.proposition
                    if known and inner_contract.proposition is not None
                    else question.event
                ),
                values=list(inner_contract.values),
                evidence=list(inner_contract.evidence),
                truth=TruthValue.TRUE if known else TruthValue.FALSE,
                certainty=255,
                source=(
                    inner_contract.source
                    if known
                    else SourceKind.INFERRED
                ),
                reason=(
                    "the embedded question is answerable from stored evidence"
                    if known
                    else "the embedded question is not answerable from stored evidence"
                ),
                response_goal="answer",
                required_slots={
                    "embedded_memory_probe": "true",
                    "embedded_memory_known": "true" if known else "false",
                    "embedded_matrix_predicate": predicate,
                    "embedded_question_kind": inner.kind.value,
                },
                forbidden_claims=[
                    "invent_inner_answer",
                    "collapse_outer_and_inner_question",
                ],
                diagnostics=[
                    f"inner answer status={inner_contract.status.value}",
                    "outer epistemic question retained",
                ],
            )

        if inner is not None:
            relations = self._matching_embedded_interrogatives(
                question,
                memory,
            )
            if question.kind == QuestionKind.YES_NO:
                desired = question.event.polarity
                supporting = [
                    item
                    for item in relations
                    if item[1].polarity == desired
                ]
                opposing = [
                    item
                    for item in relations
                    if item[1].polarity != desired
                ]
                if supporting and opposing:
                    evidence = self._embedded_interrogative_evidence(
                        [item[0] for item in relations],
                        memory,
                    )
                    return AnswerContract(
                        status=AnswerStatus.CONFLICT,
                        question=question,
                        proposition=supporting[0][1],
                        evidence=evidence,
                        truth=TruthValue.CONFLICT,
                        certainty=min(item[0].certainty for item in relations),
                        source=SourceKind.ATTRIBUTED,
                        reason="positive and negated question attributions conflict",
                        response_goal="warn",
                        required_slots={
                            "embedded_interrogative": "true",
                            "relation_ids": ",".join(
                                item[0].relation_id for item in relations
                            ),
                        },
                    )
                selected = supporting or opposing
                if selected:
                    relation, matrix_event, question_event = selected[0]
                    positive = bool(supporting)
                    return AnswerContract(
                        status=AnswerStatus.TRUE if positive else AnswerStatus.FALSE,
                        question=question,
                        proposition=matrix_event,
                        evidence=self._embedded_interrogative_evidence(
                            [relation],
                            memory,
                        ),
                        truth=TruthValue.TRUE if positive else TruthValue.FALSE,
                        certainty=relation.certainty,
                        source=SourceKind.ATTRIBUTED,
                        reason=(
                            "matching attributed question relation is stored"
                            if positive
                            else "only the opposite matrix polarity is stored"
                        ),
                        response_goal="answer",
                        required_slots=self._embedded_interrogative_slots(
                            [relation]
                        ),
                        forbidden_claims=[
                            "assert_inner_answer",
                            "promote_mentioned_question_to_fact",
                        ],
                    )
                return AnswerContract(
                    status=AnswerStatus.UNKNOWN,
                    question=question,
                    proposition=question.event,
                    truth=TruthValue.UNKNOWN,
                    certainty=0,
                    source=SourceKind.ATTRIBUTED,
                    reason="no matching attributed question relation is stored",
                    response_goal="answer",
                    required_slots={
                        "embedded_interrogative_query": "true",
                        "embedded_matrix_predicate": predicate,
                    },
                    forbidden_claims=["invent_question_attribution"],
                )

            if relations:
                relation_ids = [item[0].relation_id for item in relations]
                values: List[SemanticRef] = []
                seen: set[str] = set()
                for relation, _matrix, _question_event in relations:
                    if relation.source_entity_id in seen:
                        continue
                    seen.add(relation.source_entity_id)
                    entity = memory.get_entity(relation.source_entity_id)
                    if entity is not None:
                        values.append(entity.to_ref(entity.canonical_name))
                relation, matrix_event, question_event = relations[0]
                return AnswerContract(
                    status=AnswerStatus.ANSWERED,
                    question=question,
                    proposition=question_event,
                    values=values,
                    evidence=self._embedded_interrogative_evidence(
                        [item[0] for item in relations],
                        memory,
                    ),
                    certainty=min(item[0].certainty for item in relations),
                    source=SourceKind.ATTRIBUTED,
                    reason="bound outer question through typed question attribution",
                    response_goal="answer",
                    required_slots=self._embedded_interrogative_slots(
                        [item[0] for item in relations]
                    ),
                    forbidden_claims=[
                        "assert_inner_answer",
                        "promote_mentioned_question_to_fact",
                    ],
                )
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                reason="no matching attributed question relation is stored",
                response_goal="answer",
                forbidden_claims=["invent_question_attribution"],
            )

        if (
            question.kind == QuestionKind.WHAT
            and question.requested_role in {"patient", "content", "value"}
        ):
            relations = list(memory.embedded_interrogatives)
            if source_ref is not None and not source_ref.is_variable:
                if source_ref.kind != RefKind.ENTITY:
                    return None
                relations = [
                    item
                    for item in relations
                    if item.source_entity_id == source_ref.key
                ]
            relations = [
                item
                for item in relations
                if lexicon.lemma(item.matrix_predicate) == predicate
            ]
            relations = [item for item in relations if item.licensed]
            if not relations:
                negated = [
                    item
                    for item in memory.embedded_interrogatives
                    if lexicon.lemma(item.matrix_predicate) == predicate
                    and (
                        source_ref is None
                        or source_ref.is_variable
                        or (
                            source_ref.kind == RefKind.ENTITY
                            and item.source_entity_id == source_ref.key
                        )
                    )
                    and not item.licensed
                ]
                if negated:
                    return AnswerContract(
                        status=AnswerStatus.UNKNOWN,
                        question=question,
                        reason="only a negated question attribution is stored",
                        response_goal="answer",
                        forbidden_claims=["invert_negated_question_attribution"],
                    )
                return None
            latest_turn = max(
                memory.get_event(item.matrix_event_id).turn_index
                for item in relations
                if memory.get_event(item.matrix_event_id) is not None
            )
            latest = [
                item
                for item in relations
                if memory.get_event(item.matrix_event_id) is not None
                and memory.get_event(item.matrix_event_id).turn_index == latest_turn
            ]
            relation = max(latest, key=lambda item: item.certainty)
            question_event = memory.get_event(relation.question_event_id)
            if question_event is None:
                return None
            return AnswerContract(
                status=AnswerStatus.ANSWERED,
                question=question,
                proposition=question_event,
                values=[
                    SemanticRef.event(
                        question_event.event_id,
                        self._embedded_question_phrase(relation, question_event, memory),
                    )
                ],
                evidence=self._embedded_interrogative_evidence(
                    [relation],
                    memory,
                ),
                certainty=relation.certainty,
                source=SourceKind.ATTRIBUTED,
                reason="bound question content through an explicit attribution relation",
                response_goal="answer",
                required_slots=self._embedded_interrogative_slots([relation]),
                forbidden_claims=[
                    "assert_inner_answer",
                    "promote_mentioned_question_to_fact",
                ],
            )
        return None

    def _matching_embedded_interrogatives(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> List[Tuple[EmbeddedInterrogativeRelation, EventFrame, EventFrame]]:
        inner = question.embedded_question
        if inner is None:
            return []
        predicate = lexicon.lemma(question.event.predicate)
        source_ref = (
            question.event.arguments.get("agent")
            or question.event.arguments.get("experiencer")
            or question.event.arguments.get("subject")
        )
        matches: List[
            Tuple[EmbeddedInterrogativeRelation, EventFrame, EventFrame]
        ] = []
        for relation in memory.embedded_interrogatives:
            if lexicon.lemma(relation.matrix_predicate) != predicate:
                continue
            if source_ref is not None and not source_ref.is_variable:
                if (
                    source_ref.kind != RefKind.ENTITY
                    or relation.source_entity_id != source_ref.key
                ):
                    continue
            matrix_event = memory.get_event(relation.matrix_event_id)
            question_event = memory.get_event(relation.question_event_id)
            if matrix_event is None or question_event is None:
                continue
            if question.event.tense and matrix_event.tense != question.event.tense:
                continue
            if question.event.modality != matrix_event.modality:
                continue
            matrix_roles_match = True
            for role, expected in question.event.arguments.items():
                if role in {"agent", "experiencer", "subject", "source"}:
                    continue
                if expected.is_variable:
                    continue
                actual = matrix_event.arguments.get(role)
                if actual is None or not memory.refs_equal(expected, actual):
                    matrix_roles_match = False
                    break
            if not matrix_roles_match:
                continue
            if relation.question_kind != inner.kind:
                continue
            if relation.requested_role != inner.requested_role:
                continue
            if question_event.predicate != inner.event.predicate:
                continue
            if question_event.polarity != inner.event.polarity:
                continue
            if inner.event.tense and question_event.tense != inner.event.tense:
                continue
            if relation.focus_surface and inner.focus_surface:
                if (
                    memory.normalize_alias(relation.focus_surface)
                    != memory.normalize_alias(inner.focus_surface)
                ):
                    continue
            fixed_match = True
            for role, expected in inner.event.arguments.items():
                if expected.is_variable:
                    continue
                actual = question_event.arguments.get(role)
                if actual is None or not memory.refs_equal(expected, actual):
                    fixed_match = False
                    break
            if fixed_match:
                matches.append((relation, matrix_event, question_event))
        matches.sort(
            key=lambda item: (
                item[1].turn_index,
                item[0].certainty,
            ),
            reverse=True,
        )
        return matches

    @staticmethod
    def _embedded_interrogative_slots(
        relations: Sequence[EmbeddedInterrogativeRelation],
    ) -> Dict[str, str]:
        first = relations[0]
        return {
            "embedded_interrogative": "true",
            "relation_ids": ",".join(item.relation_id for item in relations),
            "matrix_event_ids": ",".join(item.matrix_event_id for item in relations),
            "question_event_ids": ",".join(item.question_event_id for item in relations),
            "source_entity_ids": ",".join(item.source_entity_id for item in relations),
            "matrix_predicate": first.matrix_predicate,
            "question_kind": first.question_kind.value,
            "embedded_marker": first.marker,
        }

    @staticmethod
    def _embedded_interrogative_evidence(
        relations: Sequence[EmbeddedInterrogativeRelation],
        memory: ConversationMemory,
    ) -> List[Evidence]:
        evidence: List[Evidence] = []
        seen: set[str] = set()
        for relation in relations:
            for event_id, role, score in (
                (relation.matrix_event_id, "matrix", 1.0),
                (relation.question_event_id, "interrogative", 0.8),
            ):
                if event_id in seen:
                    continue
                seen.add(event_id)
                event = memory.get_event(event_id)
                if event is not None:
                    evidence.append(
                        Evidence(event, matched_roles=[role], score=score)
                    )
        return evidence

    @staticmethod
    def _embedded_question_phrase(
        relation: EmbeddedInterrogativeRelation,
        event: EventFrame,
        memory: ConversationMemory,
    ) -> str:
        marker = relation.marker
        subject = (
            event.arguments.get("agent")
            or event.arguments.get("experiencer")
            or event.arguments.get("subject")
        )
        subject_text = ""
        if subject is not None and not subject.is_variable:
            if subject.kind == RefKind.ENTITY:
                subject_text = memory.describe_entity(subject.key).lower()
            else:
                subject_text = (subject.surface or subject.key).lower()
        verb = (
            lexicon.past_form(event.predicate)
            if event.tense == "past"
            else lexicon.present_form(
                event.predicate,
                third_person_singular=bool(subject_text),
            )
        )
        if relation.requested_role in {"agent", "subject", "experiencer"}:
            return " ".join([marker, verb]).strip()
        terms = [marker]
        if relation.focus_surface:
            terms.append(relation.focus_surface)
        if subject_text:
            terms.append(subject_text)
        terms.append(verb)
        for role in (
            "patient", "recipient", "destination", "location", "time",
            "cause", "motive", "purpose", "method", "manner",
        ):
            value = event.arguments.get(role)
            if value is None or value.is_variable:
                continue
            terms.append((value.surface or value.key).lower())
        return " ".join(" ".join(terms).split())

    CONTENT_QUERY_PREDICATES = {
        "say", "tell", "report", "claim",
        "think", "believe", "know",
        "notice", "hear",
    }
    INFINITIVAL_QUERY_PREDICATES = {
        "plan", "intend", "hope", "want", "tell", "ask", "seem", "appear",
    }
    GERUND_QUERY_PREDICATES = {
        "enjoy", "avoid", "start", "begin", "stop", "keep", "continue",
        "see", "watch", "hear", "notice",
    }

    def _answer_gerund_request(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        """Answer through a selected gerund/participial matrix relation.

        The answer may expose the selected content, its controller, or one of
        its open arguments, but the embedded frame remains qualified by the
        matrix predicate.  In particular, enjoying, avoiding, or perceiving an
        event never stores that event as an unqualified direct fact.
        """

        predicate = lexicon.lemma(question.event.predicate)
        if predicate not in self.GERUND_QUERY_PREDICATES:
            return None

        source_ref = (
            question.event.arguments.get("agent")
            or question.event.arguments.get("experiencer")
            or question.event.arguments.get("subject")
        )
        relations = list(memory.gerunds)
        if source_ref is not None and not source_ref.is_variable:
            if source_ref.kind != RefKind.ENTITY:
                return None
            relations = [
                item for item in relations
                if item.source_entity_id == source_ref.key
            ]
        relations = [
            item for item in relations
            if item.matrix_predicate == predicate
            and self._gerund_relation_matches_question(item, question, memory)
        ]
        if not relations:
            return None

        if question.kind == QuestionKind.YES_NO:
            # A plain matrix question such as ``Did I see Sarah?`` belongs to
            # ordinary event QA.  This path is only for a selected -ing
            # proposition such as ``Did I see Sarah leaving?``.
            if not any(
                self._gerund_question_selects_complement(
                    question,
                    memory.get_event(item.complement_event_id),
                )
                for item in relations
            ):
                return None

            supporting: List[GerundRelation] = []
            contradicting: List[GerundRelation] = []
            for relation in relations:
                complement = memory.get_event(relation.complement_event_id)
                if complement is None:
                    continue
                matrix_polarity, embedded_polarity = (
                    self._gerund_question_polarities(question, complement)
                )
                exact = relation.licensed == matrix_polarity
                if embedded_polarity is not None:
                    exact = exact and complement.polarity == embedded_polarity
                (supporting if exact else contradicting).append(relation)

            if supporting and contradicting:
                combined = supporting + contradicting
                matrix_events = [
                    event
                    for item in combined
                    if (event := memory.get_event(item.matrix_event_id)) is not None
                ]
                return AnswerContract(
                    status=AnswerStatus.CONFLICT,
                    question=question,
                    proposition=memory.get_event(supporting[0].matrix_event_id),
                    truth=TruthValue.CONFLICT,
                    evidence=self._gerund_evidence(combined, memory),
                    certainty=min(item.certainty for item in combined),
                    source=self._combine_sources(matrix_events),
                    reason=(
                        "stored matrix/embedded polarity pairs conflict with "
                        "the selected -ing proposition"
                    ),
                    response_goal="warn",
                    required_slots={
                        "gerund": "true",
                        "relation_ids": ",".join(
                            item.relation_id for item in combined
                        ),
                    },
                    forbidden_claims=[
                        "promote_gerund_to_unqualified_event",
                    ],
                )

            winner_pool = supporting or contradicting
            if not winner_pool:
                return None
            winner = max(
                winner_pool,
                key=lambda item: (
                    memory.get_event(item.matrix_event_id).turn_index
                    if memory.get_event(item.matrix_event_id) is not None
                    else -1,
                    item.certainty,
                ),
            )
            matrix = memory.get_event(winner.matrix_event_id)
            if matrix is None:
                return None
            truth = bool(supporting)
            return AnswerContract(
                status=AnswerStatus.TRUE if truth else AnswerStatus.FALSE,
                question=question,
                proposition=matrix,
                truth=TruthValue.TRUE if truth else TruthValue.FALSE,
                evidence=self._gerund_evidence([winner], memory),
                certainty=winner.certainty,
                source=matrix.source,
                reason=(
                    "typed gerund/participial proposition match"
                    if truth
                    else "typed opposite gerund/participial proposition match"
                ),
                response_goal="answer",
                required_slots=self._gerund_slots(winner),
                forbidden_claims=[
                    "promote_gerund_to_unqualified_event",
                ],
            )

        licensed = [item for item in relations if item.licensed]
        if not licensed:
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                certainty=0,
                source=SourceKind.ATTRIBUTED,
                reason="only a negated gerund/participial relation is stored",
                response_goal="answer",
                required_slots={
                    "unknown_object": "content",
                    "gerund": "true",
                    "matrix_predicate": predicate,
                },
                forbidden_claims=[
                    "invert_negated_gerund_relation",
                    "promote_gerund_to_unqualified_event",
                ],
            )

        requested = question.requested_role or ""
        tail = self._gerund_question_tail(question)
        if (
            question.kind == QuestionKind.WHAT
            and requested in {"patient", "content"}
            and (
                not tail
                or self._gerund_tail_is_content_placeholder(tail)
            )
        ):
            latest = self._latest_gerund_relations(licensed, memory)
            phrases: Dict[str, Tuple[GerundRelation, SemanticRef]] = {}
            for relation in latest:
                complement = memory.get_event(relation.complement_event_id)
                if complement is None:
                    continue
                phrase = self._gerund_event_phrase(complement, relation, memory)
                phrases.setdefault(
                    phrase,
                    (
                        relation,
                        SemanticRef.literal(phrase, phrase, EntityKind.ABSTRACT),
                    ),
                )
            if not phrases:
                return None
            if len(phrases) > 1:
                selected = [item[0] for item in phrases.values()]
                return AnswerContract(
                    status=AnswerStatus.MULTIPLE_MATCHES,
                    question=question,
                    values=[item[1] for item in phrases.values()],
                    evidence=self._gerund_evidence(selected, memory),
                    certainty=min(item.certainty for item in selected),
                    source=SourceKind.ATTRIBUTED,
                    reason=(
                        "multiple qualified -ing contents are stored for the "
                        "latest turn"
                    ),
                    response_goal="clarify",
                    required_slots={
                        "gerund": "true",
                        "relation_ids": ",".join(
                            item.relation_id for item in selected
                        ),
                    },
                    forbidden_claims=[
                        "promote_gerund_to_unqualified_event",
                    ],
                )
            relation, value = next(iter(phrases.values()))
            complement = memory.get_event(relation.complement_event_id)
            if complement is None:
                return None
            return AnswerContract(
                status=AnswerStatus.ANSWERED,
                question=question,
                proposition=complement,
                values=[value],
                evidence=self._gerund_evidence([relation], memory),
                certainty=relation.certainty,
                source=SourceKind.ATTRIBUTED,
                reason=(
                    "bound selected -ing content through its qualified "
                    "matrix relation"
                ),
                response_goal="answer",
                required_slots=self._gerund_slots(relation),
                forbidden_claims=[
                    "promote_gerund_to_unqualified_event",
                ],
            )

        # Bind WH/open slots through the typed relation rather than through
        # the nonassertive complement in the ordinary fact store.
        bound: List[Tuple[GerundRelation, SemanticRef, str]] = []
        for relation in licensed:
            selected = self._gerund_open_slot_value(
                relation,
                question,
                memory,
            )
            if selected is not None:
                value, actual_role = selected
                bound.append((relation, value, actual_role))
        if not bound:
            return None

        grouped: Dict[Tuple[str, str], List[Tuple[GerundRelation, SemanticRef, str]]] = defaultdict(list)
        for item in bound:
            grouped[(item[1].kind.value, item[1].key)].append(item)
        if len(grouped) > 1:
            representatives = [items[0] for items in grouped.values()]
            selected_relations = [item[0] for item in representatives]
            return AnswerContract(
                status=AnswerStatus.MULTIPLE_MATCHES,
                question=question,
                values=[item[1] for item in representatives],
                evidence=self._gerund_evidence(selected_relations, memory),
                certainty=min(item.certainty for item in selected_relations),
                source=SourceKind.ATTRIBUTED,
                reason="multiple values satisfy the qualified -ing open slot",
                response_goal="clarify",
                required_slots={
                    "gerund": "true",
                    "relation_ids": ",".join(
                        item.relation_id for item in selected_relations
                    ),
                },
                forbidden_claims=[
                    "promote_gerund_to_unqualified_event",
                ],
            )

        candidates = next(iter(grouped.values()))
        relation, value, actual_role = max(
            candidates,
            key=lambda item: (
                memory.get_event(item[0].matrix_event_id).turn_index
                if memory.get_event(item[0].matrix_event_id) is not None
                else -1,
                item[0].certainty,
            ),
        )
        complement = memory.get_event(relation.complement_event_id)
        if complement is None:
            return None
        slots = self._gerund_slots(relation)
        slots.update({
            "requested_role": requested,
            "actual_role": actual_role,
        })
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=complement,
            values=[value],
            evidence=self._gerund_evidence(
                [item[0] for item in candidates],
                memory,
            ),
            certainty=max(item[0].certainty for item in candidates),
            source=SourceKind.ATTRIBUTED,
            reason="bound open slot through a qualified -ing relation",
            response_goal="answer",
            required_slots=slots,
            forbidden_claims=[
                "promote_gerund_to_unqualified_event",
            ],
        )

    def _gerund_relation_matches_question(
        self,
        relation: GerundRelation,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> bool:
        matrix = memory.get_event(relation.matrix_event_id)
        complement = memory.get_event(relation.complement_event_id)
        if matrix is None or complement is None:
            return False
        if question.event.tense and matrix.tense != question.event.tense:
            return False
        if question.event.aspect != matrix.aspect:
            return False
        if question.event.modality != matrix.modality:
            return False

        tail = self._gerund_question_tail(question)
        selects_complement = self._gerund_question_selects_complement(
            question,
            complement,
        )
        for role, expected in question.event.arguments.items():
            if expected.is_variable:
                continue
            if role in {"agent", "experiencer", "subject", "possessor"}:
                actual = (
                    matrix.arguments.get(role)
                    or matrix.arguments.get("agent")
                    or matrix.arguments.get("experiencer")
                    or matrix.arguments.get("subject")
                    or matrix.arguments.get("possessor")
                )
                if actual is None or not memory.refs_equal(expected, actual):
                    return False
                continue
            if role in {"patient", "recipient"} and selects_complement:
                # The ordinary clause parser represents the selected surface
                # as a nominal matrix object in questions.  Match it against
                # the typed complement below, not against that placeholder.
                continue
            actual = matrix.arguments.get(role)
            if actual is None or not memory.refs_equal(expected, actual):
                return False

        if not tail:
            return True
        if not selects_complement:
            return False
        actual_phrase = self._gerund_event_phrase(complement, relation, memory)
        requested = " ".join(tail)
        if self._gerund_tail_is_content_placeholder(tail):
            focus = " ".join(tail[:-1]).strip()
            return not focus or (
                actual_phrase == focus
                or actual_phrase.startswith(focus + " ")
            )
        # Embedded polarity is compared independently by the polar-answer
        # branch.  Surface negation therefore must not prevent the underlying
        # selected proposition from reaching that comparison.
        requested_content = " ".join(
            word for word in requested.split()
            if word not in lexicon.NEGATORS
        )
        actual_content = " ".join(
            word for word in actual_phrase.split()
            if word not in lexicon.NEGATORS
        )
        return (
            requested_content == actual_content
            or actual_content.startswith(requested_content + " ")
            or (
                relation.relation_type
                == GerundRelationType.PERCEPTION_PARTICIPIAL
                and actual_content.endswith(" " + requested_content)
            )
        )

    @staticmethod
    def _latest_gerund_relations(
        relations: Sequence[GerundRelation],
        memory: ConversationMemory,
    ) -> List[GerundRelation]:
        latest_turn = max(
            (
                memory.get_event(item.matrix_event_id).turn_index
                if memory.get_event(item.matrix_event_id) is not None
                else -1
            )
            for item in relations
        )
        return [
            item for item in relations
            if (
                memory.get_event(item.matrix_event_id).turn_index
                if memory.get_event(item.matrix_event_id) is not None
                else -1
            ) == latest_turn
        ]

    @staticmethod
    def _gerund_question_tail(question: QuestionFrame) -> List[str]:
        items = [
            token
            for token in lexicon.tokenize(question.raw_text)
            if token.norm not in lexicon.PUNCTUATION
        ]
        predicate = lexicon.lemma(question.event.predicate)
        predicate_index = next(
            (
                index for index, token in enumerate(items)
                if lexicon.lemma(token.norm) == predicate
            ),
            -1,
        )
        if predicate_index < 0:
            return []
        return [token.norm for token in items[predicate_index + 1 :]]

    @staticmethod
    def _gerund_tail_is_content_placeholder(tail: Sequence[str]) -> bool:
        return bool(tail) and tail[-1] in {
            "do", "doing", "something", "anything",
        }

    def _gerund_question_selects_complement(
        self,
        question: QuestionFrame,
        complement: Optional[EventFrame],
    ) -> bool:
        if complement is None:
            return False
        tail = self._gerund_question_tail(question)
        if self._gerund_tail_is_content_placeholder(tail):
            return True
        return any(
            lexicon.lemma(word) == complement.predicate
            and (word.endswith("ing") or word == complement.predicate)
            for word in tail
        )

    def _gerund_question_polarities(
        self,
        question: QuestionFrame,
        complement: EventFrame,
    ) -> Tuple[bool, Optional[bool]]:
        matrix_polarity = (
            question.matrix_polarity
            if question.matrix_polarity is not None
            else question.event.polarity
        )
        if question.embedded_polarity is not None:
            return matrix_polarity, question.embedded_polarity

        tail = self._gerund_question_tail(question)
        predicate_index = next(
            (
                index for index, word in enumerate(tail)
                if lexicon.lemma(word) == complement.predicate
            ),
            -1,
        )
        if predicate_index >= 0 and any(
            word in lexicon.NEGATORS for word in tail[:predicate_index]
        ):
            return True, False
        return matrix_polarity, None

    def _gerund_open_slot_value(
        self,
        relation: GerundRelation,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[Tuple[SemanticRef, str]]:
        requested = question.requested_role or ""
        if requested in {"agent", "subject", "experiencer", "possessor"}:
            entity = memory.get_entity(relation.source_entity_id)
            return (entity.to_ref(), "source") if entity is not None else None

        if (
            relation.relation_type
            == GerundRelationType.PERCEPTION_PARTICIPIAL
            and requested in {"patient", "recipient"}
        ):
            entity = memory.get_entity(relation.controller_entity_id)
            return (entity.to_ref(), "controller") if entity is not None else None

        complement = memory.get_event(relation.complement_event_id)
        if complement is None:
            return None
        for role in self._candidate_roles(question):
            value = complement.arguments.get(role)
            if value is not None and not value.is_variable:
                if (
                    value.kind == RefKind.ENTITY
                    and value.key == relation.controller_entity_id
                ):
                    continue
                return value, role
        return None

    @staticmethod
    def _gerund_event_phrase(
        event: EventFrame,
        relation: GerundRelation,
        memory: ConversationMemory,
    ) -> str:
        terms: List[str] = []
        if relation.relation_type == GerundRelationType.PERCEPTION_PARTICIPIAL:
            controller = memory.get_entity(relation.controller_entity_id)
            if controller is not None:
                terms.append(controller.canonical_name.lower())
        if not event.polarity:
            terms.append("not")
        terms.append(lexicon.gerund_form(event.predicate))
        for role in (
            "patient", "recipient", "destination", "location", "time",
            "method", "manner",
        ):
            value = event.arguments.get(role)
            if value is None or value.is_variable:
                continue
            if (
                value.kind == RefKind.ENTITY
                and value.key == relation.controller_entity_id
            ):
                continue
            if value.kind == RefKind.ENTITY:
                entity = memory.get_entity(value.key)
                terms.append(
                    entity.canonical_name.lower()
                    if entity is not None
                    else (value.surface or value.key).lower()
                )
            else:
                terms.append((value.surface or value.key).lower())
        return " ".join(" ".join(terms).split())

    @staticmethod
    def _gerund_slots(relation: GerundRelation) -> Dict[str, str]:
        return {
            "gerund": "true",
            "relation_id": relation.relation_id,
            "matrix_event_id": relation.matrix_event_id,
            "complement_event_id": relation.complement_event_id,
            "source_entity_id": relation.source_entity_id,
            "controller_entity_id": relation.controller_entity_id,
            "embedded_subject_entity_id": relation.embedded_subject_entity_id,
            "matrix_predicate": relation.matrix_predicate,
            "relation_type": relation.relation_type.value,
            "content_status": relation.content_status.value,
            "predicate_family": relation.predicate_family,
            "entailed": "true" if relation.entailed else "false",
        }

    @staticmethod
    def _gerund_evidence(
        relations: Sequence[GerundRelation],
        memory: ConversationMemory,
    ) -> List[Evidence]:
        evidence: List[Evidence] = []
        seen: set[Tuple[str, str]] = set()
        for relation in relations:
            matrix = memory.get_event(relation.matrix_event_id)
            complement = memory.get_event(relation.complement_event_id)
            if matrix is not None and (matrix.event_id, "matrix") not in seen:
                evidence.append(
                    Evidence(matrix, matched_roles=["matrix"], score=1.0)
                )
                seen.add((matrix.event_id, "matrix"))
            if (
                complement is not None
                and (complement.event_id, "gerund") not in seen
            ):
                evidence.append(
                    Evidence(
                        complement,
                        matched_roles=["gerund"],
                        score=0.8,
                    )
                )
                seen.add((complement.event_id, "gerund"))
        return evidence

    def _answer_infinitival_request(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        """Answer questions about a typed control/raising relation.

        The matrix event is the asserted fact. The complement remains a
        planned, desired, requested, directed, hoped, or evidential event and
        is never promoted to an accomplished event by this path.
        """

        predicate = question.event.predicate
        if predicate not in self.INFINITIVAL_QUERY_PREDICATES:
            return None

        source_ref = (
            question.event.arguments.get("agent")
            or question.event.arguments.get("experiencer")
            or question.event.arguments.get("subject")
        )
        requested = question.requested_role or ""
        relations = list(memory.infinitivals)
        if source_ref is not None and not source_ref.is_variable:
            if source_ref.kind != RefKind.ENTITY:
                return None
            relations = [
                item
                for item in relations
                if item.source_entity_id == source_ref.key
            ]
        relations = [
            item for item in relations if item.matrix_predicate == predicate
        ]
        relations = [
            item
            for item in relations
            if self._infinitival_relation_matches_question(
                item,
                question,
                memory,
            )
        ]
        if not relations:
            return None

        if question.kind == QuestionKind.YES_NO:
            query_matrix_polarity = (
                question.matrix_polarity
                if question.matrix_polarity is not None
                else question.event.polarity
            )
            query_embedded_polarity = question.embedded_polarity
            supporting: List[InfinitivalRelation] = []
            contradicting: List[InfinitivalRelation] = []
            for relation in relations:
                complement = memory.get_event(relation.complement_event_id)
                if complement is None:
                    continue
                exact = relation.licensed == query_matrix_polarity
                if query_embedded_polarity is not None:
                    exact = exact and (
                        complement.polarity == query_embedded_polarity
                    )
                (supporting if exact else contradicting).append(relation)

            if supporting and contradicting:
                evidence = self._infinitival_evidence(
                    supporting + contradicting,
                    memory,
                )
                return AnswerContract(
                    status=AnswerStatus.CONFLICT,
                    question=question,
                    proposition=memory.get_event(
                        supporting[0].matrix_event_id
                    ),
                    truth=TruthValue.CONFLICT,
                    evidence=evidence,
                    certainty=min(item.certainty for item in relations),
                    source=SourceKind.USER,
                    reason=(
                        "stored matrix/embedded polarity pairs conflict with "
                        "the infinitival proposition"
                    ),
                    response_goal="warn",
                    required_slots={
                        "infinitival": "true",
                        "relation_ids": ",".join(
                            item.relation_id for item in relations
                        ),
                    },
                )
            winner_pool = supporting or contradicting
            if not winner_pool:
                return None
            winner = max(
                winner_pool,
                key=lambda item: (
                    memory.get_event(item.matrix_event_id).turn_index
                    if memory.get_event(item.matrix_event_id) is not None
                    else -1,
                    item.certainty,
                ),
            )
            matrix = memory.get_event(winner.matrix_event_id)
            if matrix is None:
                return None
            truth = bool(supporting)
            return AnswerContract(
                status=AnswerStatus.TRUE if truth else AnswerStatus.FALSE,
                question=question,
                proposition=matrix,
                truth=TruthValue.TRUE if truth else TruthValue.FALSE,
                evidence=self._infinitival_evidence([winner], memory),
                certainty=winner.certainty,
                source=SourceKind.USER,
                reason=(
                    "typed infinitival proposition match"
                    if truth
                    else "typed opposite infinitival proposition match"
                ),
                response_goal="answer",
                required_slots={
                    "infinitival": "true",
                    "relation_id": winner.relation_id,
                },
            )

        if question.kind == QuestionKind.WHO and requested in {
            "agent", "subject", "patient", "recipient"
        }:
            licensed_all = [item for item in relations if item.licensed]
            if not licensed_all:
                return AnswerContract(
                    status=AnswerStatus.UNKNOWN,
                    question=question,
                    certainty=0,
                    source=SourceKind.ATTRIBUTED,
                    reason="only negated infinitival relations are stored",
                    response_goal="answer",
                    forbidden_claims=["invert_negated_infinitival_relation"],
                )
            grouped: Dict[str, List[InfinitivalRelation]] = defaultdict(list)
            for relation in licensed_all:
                entity_id = (
                    relation.source_entity_id
                    if requested in {"agent", "subject"}
                    else relation.controller_entity_id
                )
                grouped[entity_id].append(relation)
            if len(grouped) > 1:
                return AnswerContract(
                    status=AnswerStatus.MULTIPLE_MATCHES,
                    question=question,
                    values=[
                        memory.get_entity(entity_id).to_ref()
                        for entity_id in grouped
                        if memory.get_entity(entity_id) is not None
                    ],
                    evidence=self._infinitival_evidence(
                        licensed_all,
                        memory,
                    ),
                    certainty=min(item.certainty for item in licensed_all),
                    source=SourceKind.USER,
                    reason="multiple controllers satisfy the infinitival question",
                    response_goal="clarify",
                )
            relation = max(
                licensed_all,
                key=lambda item: (
                    memory.get_event(item.matrix_event_id).turn_index
                    if memory.get_event(item.matrix_event_id) is not None
                    else -1,
                    item.certainty,
                ),
            )
            entity_id = next(iter(grouped))
            entity = memory.get_entity(entity_id)
            complement = memory.get_event(relation.complement_event_id)
            if entity is None or complement is None:
                return None
            return AnswerContract(
                status=AnswerStatus.ANSWERED,
                question=question,
                proposition=complement,
                values=[entity.to_ref()],
                evidence=self._infinitival_evidence([relation], memory),
                certainty=relation.certainty,
                source=SourceKind.ATTRIBUTED,
                reason="bound controller through a typed infinitival relation",
                response_goal="answer",
                required_slots=self._infinitival_slots(relation),
            )

        latest_turn = max(
            (
                memory.get_event(item.matrix_event_id).turn_index
                if memory.get_event(item.matrix_event_id) is not None
                else -1
            )
            for item in relations
        )
        latest = [
            item
            for item in relations
            if (
                memory.get_event(item.matrix_event_id).turn_index
                if memory.get_event(item.matrix_event_id) is not None
                else -1
            )
            == latest_turn
        ]
        licensed = [item for item in latest if item.licensed]
        if not licensed:
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                certainty=0,
                source=SourceKind.ATTRIBUTED,
                reason="only a negated infinitival relation is stored",
                response_goal="answer",
                required_slots={
                    "unknown_object": "content",
                    "infinitival": "true",
                    "matrix_predicate": predicate,
                },
                forbidden_claims=["invert_negated_infinitival_relation"],
            )

        if (
            question.kind == QuestionKind.WHAT
            and requested in {"patient", "content"}
        ):
            signatures = {
                memory.get_event(item.complement_event_id).signature()
                for item in licensed
                if memory.get_event(item.complement_event_id) is not None
            }
            if len(signatures) > 1:
                values = [
                    SemanticRef.literal(
                        self._infinitival_event_phrase(event, relation, memory),
                        self._infinitival_event_phrase(event, relation, memory),
                        EntityKind.ABSTRACT,
                    )
                    for relation in licensed
                    for event in [
                        memory.get_event(relation.complement_event_id)
                    ]
                    if event is not None
                ]
                return AnswerContract(
                    status=AnswerStatus.MULTIPLE_MATCHES,
                    question=question,
                    values=values,
                    evidence=self._infinitival_evidence(licensed, memory),
                    certainty=min(item.certainty for item in licensed),
                    source=SourceKind.ATTRIBUTED,
                    reason="multiple infinitival contents are stored for the latest turn",
                    response_goal="clarify",
                )
            relation = max(licensed, key=lambda item: item.certainty)
            complement = memory.get_event(relation.complement_event_id)
            if complement is None:
                return None
            return AnswerContract(
                status=AnswerStatus.ANSWERED,
                question=question,
                proposition=complement,
                values=[
                    SemanticRef.literal(
                        self._infinitival_event_phrase(
                            complement,
                            relation,
                            memory,
                        ),
                        self._infinitival_event_phrase(
                            complement,
                            relation,
                            memory,
                        ),
                        EntityKind.ABSTRACT,
                    )
                ],
                evidence=self._infinitival_evidence([relation], memory),
                certainty=relation.certainty,
                source=SourceKind.ATTRIBUTED,
                reason="bound selected infinitival content through its matrix relation",
                response_goal="answer",
                required_slots=self._infinitival_slots(relation),
                forbidden_claims=["promote_infinitive_to_accomplished_event"],
            )
        return None

    def _infinitival_relation_matches_question(
        self,
        relation: InfinitivalRelation,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> bool:
        matrix = memory.get_event(relation.matrix_event_id)
        complement = memory.get_event(relation.complement_event_id)
        if matrix is None or complement is None:
            return False
        if question.event.tense and matrix.tense != question.event.tense:
            return False
        if question.kind != QuestionKind.YES_NO:
            if (
                question.matrix_polarity is not None
                and relation.licensed != question.matrix_polarity
            ):
                return False
            if (
                question.embedded_polarity is not None
                and complement.polarity != question.embedded_polarity
            ):
                return False

        for role, expected in question.event.arguments.items():
            if expected.is_variable or role == "purpose":
                continue
            if role in {"patient", "recipient"}:
                actual = matrix.arguments.get(role)
                if actual is None:
                    actual = (
                        matrix.arguments.get("recipient")
                        or matrix.arguments.get("patient")
                    )
            else:
                actual = matrix.arguments.get(role)
            if actual is None or not memory.refs_equal(expected, actual):
                return False

        purpose = question.event.arguments.get("purpose")
        if purpose is None or purpose.is_variable:
            return True
        requested = " ".join((purpose.surface or purpose.key).lower().split())
        if requested in {"do", "something", "anything"}:
            return True
        actual = self._infinitival_event_phrase(complement, relation, memory)
        return (
            requested == actual
            or actual.startswith(requested + " ")
            or requested.startswith(actual + " ")
            or requested.split()[0] == complement.predicate
        )

    @staticmethod
    def _infinitival_event_phrase(
        event: EventFrame,
        relation: InfinitivalRelation,
        memory: ConversationMemory,
    ) -> str:
        terms = [event.predicate]
        for role in (
            "patient", "recipient", "destination", "location", "method", "manner"
        ):
            value = event.arguments.get(role)
            if value is None:
                continue
            if value.kind == RefKind.ENTITY:
                entity = memory.get_entity(value.key)
                terms.append(
                    entity.canonical_name.lower()
                    if entity is not None
                    else (value.surface or value.key).lower()
                )
            else:
                terms.append((value.surface or value.key).lower())
        return " ".join(" ".join(terms).split())

    @staticmethod
    def _infinitival_slots(
        relation: InfinitivalRelation,
    ) -> Dict[str, str]:
        return {
            "infinitival": "true",
            "relation_id": relation.relation_id,
            "matrix_event_id": relation.matrix_event_id,
            "complement_event_id": relation.complement_event_id,
            "source_entity_id": relation.source_entity_id,
            "controller_entity_id": relation.controller_entity_id,
            "matrix_predicate": relation.matrix_predicate,
            "relation_type": relation.relation_type.value,
            "content_status": relation.content_status.value,
            "predicate_family": relation.predicate_family,
        }

    @staticmethod
    def _infinitival_evidence(
        relations: Sequence[InfinitivalRelation],
        memory: ConversationMemory,
    ) -> List[Evidence]:
        evidence: List[Evidence] = []
        for relation in relations:
            matrix = memory.get_event(relation.matrix_event_id)
            complement = memory.get_event(relation.complement_event_id)
            if matrix is not None:
                evidence.append(
                    Evidence(matrix, matched_roles=["matrix"], score=1.0)
                )
            if complement is not None:
                evidence.append(
                    Evidence(
                        complement,
                        matched_roles=["infinitive"],
                        score=0.8,
                    )
                )
        return evidence

    def _answer_attributed_content_request(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        """Answer ``What did X say/think/...`` from typed content links."""

        if (
            question.kind != QuestionKind.WHAT
            or question.requested_role not in {"patient", "content"}
            or question.event.predicate not in self.CONTENT_QUERY_PREDICATES
        ):
            return None
        source_ref = (
            question.event.arguments.get("agent")
            or question.event.arguments.get("experiencer")
            or question.event.arguments.get("subject")
        )
        if source_ref is None or source_ref.kind != RefKind.ENTITY:
            return None
        relations = memory.content_relations_for_source(
            source_ref.key,
            matrix_predicate=question.event.predicate,
        )
        if not relations:
            negated = memory.content_relations_for_source(
                source_ref.key,
                matrix_predicate=question.event.predicate,
                include_negated=True,
            )
            if negated:
                return AnswerContract(
                    status=AnswerStatus.UNKNOWN,
                    question=question,
                    certainty=0,
                    source=SourceKind.ATTRIBUTED,
                    reason="only a negated content attribution is stored",
                    response_goal="answer",
                    required_slots={
                        "unknown_object": "content",
                        "source_entity_id": source_ref.key,
                        "matrix_predicate": question.event.predicate,
                    },
                    forbidden_claims=["invert_negated_attribution"],
                )
            return None

        resolved = []
        for relation in relations:
            content_event = memory.get_event(relation.content_event_id)
            matrix_event = memory.get_event(relation.matrix_event_id)
            if content_event is None or matrix_event is None:
                continue
            if question.event.tense and matrix_event.tense != question.event.tense:
                continue
            resolved.append((relation, matrix_event, content_event))
        if not resolved:
            return None

        latest_turn = max(matrix.turn_index for _relation, matrix, _content in resolved)
        latest = [item for item in resolved if item[1].turn_index == latest_turn]
        signatures = {
            item[2].signature()
            for item in latest
        }
        if len(signatures) > 1:
            values = [
                SemanticRef.event(content.event_id, content.raw_text)
                for _relation, _matrix, content in latest
            ]
            return AnswerContract(
                status=AnswerStatus.MULTIPLE_MATCHES,
                question=question,
                proposition=latest[0][2],
                values=values,
                evidence=[Evidence(content, score=1.0) for _relation, _matrix, content in latest],
                certainty=min(relation.certainty for relation, _matrix, _content in latest),
                source=SourceKind.ATTRIBUTED,
                reason="multiple attributed contents are stored for the latest turn",
                response_goal="clarify",
                required_slots={
                    "attributed": "true",
                    "source_entity_id": source_ref.key,
                    "matrix_predicate": question.event.predicate,
                },
            )

        relation, matrix_event, content_event = max(
            latest,
            key=lambda item: (item[0].certainty, item[2].certainty),
        )
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=content_event,
            values=[SemanticRef.event(content_event.event_id, content_event.raw_text)],
            evidence=[Evidence(content_event, matched_roles=["content"], score=1.0)],
            certainty=min(relation.certainty, content_event.certainty),
            source=SourceKind.ATTRIBUTED,
            reason="bound finite content through an explicit attribution relation",
            response_goal="answer",
            required_slots={
                "attributed": "true",
                "source_entity_id": relation.source_entity_id,
                "matrix_predicate": relation.matrix_predicate,
                "matrix_tense": matrix_event.tense,
                "relation_type": relation.relation_type.value,
            },
        )

    def _answer_possessor(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        patient = question.event.arguments.get("patient")
        if patient is None or patient.kind != RefKind.ENTITY:
            return None
        entity = memory.get_entity(patient.key)
        if entity is None or not entity.owner_id:
            return None
        owner = memory.get_entity(entity.owner_id)
        if owner is None:
            return None
        proposition = EventFrame(
            predicate="own",
            arguments={
                "possessor": owner.to_ref(),
                "patient": entity.to_ref(),
            },
            source=SourceKind.USER,
            certainty=230,
            turn_index=max(entity.last_mentioned_turn, owner.last_mentioned_turn),
        )
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=proposition,
            values=[owner.to_ref()],
            evidence=[Evidence(proposition, matched_roles=["patient"], score=1.0)],
            certainty=230,
            source=SourceKind.USER,
            reason="possessed entity stores an explicit owner relation",
            response_goal="answer",
        )

    def _answer_event_query(self, question: QuestionFrame, memory: ConversationMemory) -> AnswerContract:
        candidates = [
            event
            for event in memory.events
            if event.discourse_role
            not in memory.NONASSERTIVE_DISCOURSE_ROLES
        ]
        fixed = question.event.fixed_arguments()
        if fixed:
            filtered: List[EventFrame] = []
            for event in candidates:
                if all(
                    role in event.arguments and memory.refs_equal(expected, event.arguments[role])
                    for role, expected in fixed.items()
                ):
                    filtered.append(event)
            candidates = filtered
        if not candidates:
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                reason="no event in memory matches the requested subject",
                response_goal="answer",
                forbidden_claims=["invent_event"],
            )
        latest_turn = max(event.turn_index for event in candidates)
        latest = [event for event in candidates if event.turn_index == latest_turn]
        if len(latest) > 1:
            return AnswerContract(
                status=AnswerStatus.MULTIPLE_MATCHES,
                question=question,
                proposition=latest[0],
                values=[SemanticRef.event(event.event_id, event.raw_text) for event in latest],
                evidence=[Evidence(event, score=1.0) for event in latest],
                certainty=min(event.certainty for event in latest),
                source=self._combine_sources(latest),
                reason="multiple events occurred in the latest turn",
                response_goal="clarify",
            )
        event = latest[0]
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=event,
            values=[SemanticRef.event(event.event_id, event.raw_text)],
            evidence=[Evidence(event, score=1.0)],
            certainty=event.certainty,
            source=event.source,
            response_goal="answer",
        )

    def _answer_yes_no(self, question: QuestionFrame, memory: ConversationMemory) -> AnswerContract:
        query = question.event
        matches = [
            match
            for match in memory.match_events(
                query,
                include_opposite_polarity=True,
            )
            if self._event_aspects_compatible(query, match.event)
        ]
        same = [match for match in matches if match.event.polarity == query.polarity]
        opposite = [match for match in matches if match.event.polarity != query.polarity]

        if same and opposite:
            evidence = [self._to_evidence(item) for item in same + opposite]
            return AnswerContract(
                status=AnswerStatus.CONFLICT,
                question=question,
                truth=TruthValue.CONFLICT,
                evidence=evidence,
                certainty=min(item.event.certainty for item in same + opposite),
                source=self._combine_sources([item.event for item in same + opposite]),
                reason="both proposition and negation are stored",
                response_goal="warn",
            )
        if same:
            winner = same[0]
            return AnswerContract(
                status=AnswerStatus.TRUE,
                question=question,
                proposition=winner.event,
                truth=TruthValue.TRUE,
                evidence=[self._to_evidence(item) for item in same],
                certainty=max(item.event.certainty for item in same),
                source=self._combine_sources([item.event for item in same]),
                reason="direct proposition match",
                response_goal="answer",
            )
        if opposite:
            winner = opposite[0]
            return AnswerContract(
                status=AnswerStatus.FALSE,
                question=question,
                proposition=winner.event,
                truth=TruthValue.FALSE,
                evidence=[self._to_evidence(item) for item in opposite],
                certainty=max(item.event.certainty for item in opposite),
                source=self._combine_sources([item.event for item in opposite]),
                reason="explicit negated proposition match",
                response_goal="answer",
            )

        attributed = memory.match_events(
            query,
            include_opposite_polarity=True,
            include_attributed_content=True,
        )
        attributed = [
            match
            for match in attributed
            if self._event_aspects_compatible(query, match.event)
            and match.event.discourse_role == "content"
            and any(
                relation.attributed
                and relation.content_event_id == match.event.event_id
                for relation in memory.content_relations_for_event(match.event.event_id)
            )
        ]
        if attributed:
            sources: List[str] = []
            matrix_predicates: List[str] = []
            for match in attributed:
                for relation in memory.content_relations_for_event(match.event.event_id):
                    if relation.content_event_id != match.event.event_id:
                        continue
                    if relation.source_entity_id not in sources:
                        sources.append(relation.source_entity_id)
                    if relation.matrix_predicate not in matrix_predicates:
                        matrix_predicates.append(relation.matrix_predicate)
            polarities = {match.event.polarity for match in attributed}
            reason = (
                "attributed sources disagree about the proposition"
                if len(polarities) > 1
                else "only attributed content supports the proposition"
            )
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                proposition=query,
                truth=TruthValue.UNKNOWN,
                evidence=[self._to_evidence(item) for item in attributed],
                certainty=0,
                source=SourceKind.ATTRIBUTED,
                reason=reason,
                response_goal="answer",
                required_slots={
                    "attributed": "true",
                    "source_entity_ids": ",".join(sources),
                    "matrix_predicates": ",".join(matrix_predicates),
                    "attributed_polarity_count": str(len(polarities)),
                },
                forbidden_claims=[
                    "promote_attributed_content_to_unqualified_fact",
                    "convert_attributed_content_to_truth",
                ],
                diagnostics=[reason],
            )

        gerund = self._answer_gerund_truth_boundary(
            question,
            memory,
        )
        if gerund is not None:
            return gerund

        infinitival = self._answer_infinitival_truth_boundary(
            question,
            memory,
        )
        if infinitival is not None:
            return infinitival

        # A related fact is useful context but is not logically equivalent to a
        # negation.  "Sarah bought it" does not prove that Mary had no role.
        related = self._loosely_related(query, memory)
        diagnostics: List[str] = []
        if related:
            diagnostics.append("related facts exist, but none prove or disprove the proposition")
        return AnswerContract(
            status=AnswerStatus.UNKNOWN,
            question=question,
            truth=TruthValue.UNKNOWN,
            evidence=[self._to_evidence(item) for item in related[:3]],
            certainty=0,
            source=SourceKind.UNKNOWN,
            reason="the stored evidence does not establish this proposition",
            response_goal="answer",
            forbidden_claims=["convert_absence_of_evidence_to_false"],
            diagnostics=diagnostics,
        )

    def _answer_gerund_truth_boundary(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        """Keep selected ``-ing`` content qualified in direct-event QA.

        Reviewed aspectual relations can license a derived phase inference.
        Enjoyment, avoidance, and perception only supply qualified context and
        therefore leave the unqualified proposition unknown.
        """

        query = question.event
        matches: List[GerundRelation] = []
        for relation in memory.gerunds:
            if not relation.licensed:
                continue
            complement = memory.get_event(relation.complement_event_id)
            if complement is None or complement.predicate != query.predicate:
                continue
            fixed = query.fixed_arguments()
            if any(
                not self._gerund_argument_matches(
                    role,
                    expected,
                    complement,
                    memory,
                )
                for role, expected in fixed.items()
            ):
                continue
            matches.append(relation)
        if not matches:
            return None

        statuses = sorted({item.content_status.value for item in matches})
        sources = sorted({item.source_entity_id for item in matches})
        relation_ids = ",".join(item.relation_id for item in matches)
        required_slots = {
            "gerund": "true",
            "gerund_evidence": "true",
            "relation_ids": relation_ids,
            "source_entity_ids": ",".join(sources),
            "content_statuses": ",".join(statuses),
        }
        forbidden = [
            "promote_gerund_to_unqualified_event",
            "erase_gerund_phase_or_attribution_qualification",
        ]

        entailed = [
            item for item in matches
            if item.entailed
            and self._gerund_phase_query_supported(item, query, memory)
        ]
        if entailed:
            same = [
                item for item in entailed
                if (
                    memory.get_event(item.complement_event_id) is not None
                    and memory.get_event(item.complement_event_id).polarity
                    == query.polarity
                )
            ]
            opposite = [item for item in entailed if item not in same]
            if same and opposite:
                return AnswerContract(
                    status=AnswerStatus.CONFLICT,
                    question=question,
                    proposition=query,
                    truth=TruthValue.CONFLICT,
                    evidence=self._gerund_evidence(entailed, memory),
                    certainty=min(item.certainty for item in entailed),
                    source=SourceKind.INFERRED,
                    reason=(
                        "qualified aspectual phase inferences support opposing "
                        "event polarities"
                    ),
                    response_goal="warn",
                    required_slots=required_slots,
                    forbidden_claims=forbidden,
                    diagnostics=[
                        "phase entailment is derived from a typed aspectual relation",
                    ],
                )

            supports = same or opposite
            truth = bool(same)
            required_slots.update({
                "gerund": "true",
                "derived_phase_inference": "true",
                "relation_id": supports[0].relation_id,
                "relation_type": supports[0].relation_type.value,
                "content_status": supports[0].content_status.value,
            })
            return AnswerContract(
                status=AnswerStatus.TRUE if truth else AnswerStatus.FALSE,
                question=question,
                proposition=query,
                truth=TruthValue.TRUE if truth else TruthValue.FALSE,
                evidence=self._gerund_evidence(supports, memory),
                certainty=min(item.certainty for item in supports),
                source=SourceKind.INFERRED,
                reason=(
                    "typed aspectual relation supports a qualified derived "
                    "phase inference"
                ),
                response_goal="answer",
                required_slots=required_slots,
                forbidden_claims=forbidden,
                diagnostics=[
                    "the event is supported only as an aspectually qualified inference",
                ],
            )

        polarities = {
            memory.get_event(item.complement_event_id).polarity
            for item in matches
            if memory.get_event(item.complement_event_id) is not None
        }
        reason = (
            "qualified -ing sources contain opposing embedded polarities"
            if len(polarities) > 1
            else (
                "only perception-qualified participial content supports the proposition"
                if any(
                    item.relation_type
                    == GerundRelationType.PERCEPTION_PARTICIPIAL
                    for item in matches
                )
                else "only non-entailed gerund content supports the proposition"
            )
        )
        return AnswerContract(
            status=AnswerStatus.UNKNOWN,
            question=question,
            proposition=query,
            truth=TruthValue.UNKNOWN,
            evidence=self._gerund_evidence(matches, memory),
            certainty=0,
            source=SourceKind.ATTRIBUTED,
            reason=reason,
            response_goal="answer",
            required_slots=required_slots,
            forbidden_claims=forbidden,
            diagnostics=[reason],
        )

    @staticmethod
    def _gerund_phase_query_supported(
        relation: GerundRelation,
        query: EventFrame,
        memory: ConversationMemory,
    ) -> bool:
        """Limit phase entailment to the reviewed prior-activity reading.

        A past-simple event question can ask whether the activity occurred.
        It cannot establish a present progressive state, and a negated
        embedded phase requires temporal semantics beyond this slice.
        """

        complement = memory.get_event(relation.complement_event_id)
        if complement is None or not complement.polarity:
            return False
        return query.tense == "past" and query.aspect == "simple"

    @staticmethod
    def _gerund_argument_matches(
        role: str,
        expected: SemanticRef,
        complement: EventFrame,
        memory: ConversationMemory,
    ) -> bool:
        actual = complement.arguments.get(role)
        if actual is None and role in {
            "agent", "experiencer", "subject", "possessor"
        }:
            actual = (
                complement.arguments.get("agent")
                or complement.arguments.get("experiencer")
                or complement.arguments.get("subject")
                or complement.arguments.get("possessor")
            )
        if actual is None and role in {"patient", "recipient"}:
            actual = (
                complement.arguments.get("patient")
                or complement.arguments.get("recipient")
            )
        return actual is not None and memory.refs_equal(expected, actual)

    def _answer_infinitival_truth_boundary(
        self,
        question: QuestionFrame,
        memory: ConversationMemory,
    ) -> Optional[AnswerContract]:
        """Qualify plans/desires/requests instead of asserting completion."""

        query = question.event
        matches: List[InfinitivalRelation] = []
        for relation in memory.infinitivals:
            if not relation.licensed or relation.entailed:
                continue
            complement = memory.get_event(relation.complement_event_id)
            if complement is None or complement.predicate != query.predicate:
                continue
            fixed = query.fixed_arguments()
            if any(
                role not in complement.arguments
                or not memory.refs_equal(expected, complement.arguments[role])
                for role, expected in fixed.items()
            ):
                continue
            matches.append(relation)
        if not matches:
            return None

        statuses = sorted({item.content_status.value for item in matches})
        sources = sorted({item.source_entity_id for item in matches})
        polarities = {
            memory.get_event(item.complement_event_id).polarity
            for item in matches
            if memory.get_event(item.complement_event_id) is not None
        }
        reason = (
            "infinitival sources contain opposing intended polarities"
            if len(polarities) > 1
            else "only non-entailed infinitival content supports the proposition"
        )
        return AnswerContract(
            status=AnswerStatus.UNKNOWN,
            question=question,
            proposition=query,
            truth=TruthValue.UNKNOWN,
            evidence=self._infinitival_evidence(matches, memory),
            certainty=0,
            source=SourceKind.ATTRIBUTED,
            reason=reason,
            response_goal="answer",
            required_slots={
                "infinitival_evidence": "true",
                "relation_ids": ",".join(item.relation_id for item in matches),
                "source_entity_ids": ",".join(sources),
                "content_statuses": ",".join(statuses),
            },
            forbidden_claims=[
                "promote_infinitive_to_accomplished_event",
                "convert_plan_or_desire_to_truth",
            ],
            diagnostics=[reason],
        )

    def _answer_open_slot(self, question: QuestionFrame, memory: ConversationMemory) -> AnswerContract:
        requested = question.requested_role or ""
        role_candidates = self._candidate_roles(question)
        query = question.event

        # Attribute questions are normalized against both explicit attribute
        # frames and ordinary copular facts carrying an attribute marker.
        if query.predicate == "attribute":
            matches = self._match_attribute_query(question, memory)
        else:
            matches = [
                match
                for match in memory.related_events(query, requested)
                if self._event_aspects_compatible(query, match.event)
            ]

        if not matches:
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                reason="no matching proposition is stored",
                response_goal="answer",
                forbidden_claims=["invent_missing_fact"],
            )

        positive_matches = [match for match in matches if match.event.polarity]
        negative_matches = [match for match in matches if not match.event.polarity]
        if positive_matches and negative_matches and self._same_anchor_sets(positive_matches, negative_matches, requested):
            return AnswerContract(
                status=AnswerStatus.CONFLICT,
                question=question,
                evidence=[self._to_evidence(item) for item in matches],
                certainty=min(item.event.certainty for item in matches),
                source=self._combine_sources([item.event for item in matches]),
                reason="matching positive and negative facts conflict",
                response_goal="warn",
            )

        usable: List[Tuple[EventMatch, str, SemanticRef]] = []
        for match in positive_matches or matches:
            for role in role_candidates:
                value = match.event.arguments.get(role)
                if value is not None and not value.is_variable:
                    usable.append((match, role, value))
                    break

        if not usable:
            # The base event is known, but the requested relation is not.
            best = matches[0]
            return AnswerContract(
                status=AnswerStatus.UNKNOWN,
                question=question,
                proposition=best.event,
                evidence=[self._to_evidence(item) for item in matches[:3]],
                certainty=max(item.event.certainty for item in matches),
                source=self._combine_sources([item.event for item in matches]),
                reason="base proposition known; requested slot absent",
                response_goal="answer",
                required_slots={"missing_role": requested},
                forbidden_claims=[f"invent_{requested}"],
            )

        grouped: Dict[Tuple[str, str], List[Tuple[EventMatch, str, SemanticRef]]] = defaultdict(list)
        for item in usable:
            value = item[2]
            grouped[(value.kind.value, value.key)].append(item)

        if len(grouped) > 1:
            values = [items[0][2] for items in grouped.values()]
            evidence = [self._to_evidence(item[0]) for items in grouped.values() for item in items]
            return AnswerContract(
                status=AnswerStatus.MULTIPLE_MATCHES,
                question=question,
                proposition=usable[0][0].event,
                values=values,
                evidence=evidence,
                certainty=min(item[0].event.certainty for item in usable),
                source=self._combine_sources([item[0].event for item in usable]),
                reason="more than one distinct value satisfies the open slot",
                response_goal="clarify",
            )

        selected_items = next(iter(grouped.values()))
        selected_match, actual_role, value = max(
            selected_items,
            key=lambda item: (item[0].score, item[0].event.certainty, item[0].event.turn_index),
        )
        proposition = selected_match.event
        diagnostics: List[str] = []
        if actual_role != requested:
            diagnostics.append(f"bound {requested} through compatible role {actual_role}")
        return AnswerContract(
            status=AnswerStatus.ANSWERED,
            question=question,
            proposition=proposition,
            values=[value],
            evidence=[self._to_evidence(item[0]) for item in selected_items],
            certainty=max(item[0].event.certainty for item in selected_items),
            source=self._combine_sources([item[0].event for item in selected_items]),
            reason=f"bound open slot {requested}",
            response_goal="answer",
            required_slots={"requested_role": requested, "actual_role": actual_role},
            diagnostics=diagnostics,
        )

    def _match_attribute_query(self, question: QuestionFrame, memory: ConversationMemory) -> List[EventMatch]:
        query = question.event
        subject = query.arguments.get("subject")
        attribute = query.arguments.get("attribute")
        matches: List[EventMatch] = []
        for event in memory.events:
            if event.discourse_role in memory.NONASSERTIVE_DISCOURSE_ROLES:
                continue
            if event.predicate not in {"attribute", "be"}:
                continue
            if subject:
                actual_subject = event.arguments.get("subject") or event.arguments.get("agent")
                if not actual_subject or not memory.refs_equal(subject, actual_subject):
                    continue
            if attribute:
                actual_attribute = event.arguments.get("attribute")
                if actual_attribute and not memory.refs_equal(attribute, actual_attribute):
                    continue
                # No attribute tag means we cannot safely claim that a generic
                # copular value answers a specific color/age/etc. question.
                if actual_attribute is None and attribute.key not in {"state", "identity"}:
                    continue
            value = event.arguments.get("value")
            if value is None:
                continue
            matches.append(EventMatch(event, ["subject", "attribute"], 25.0 + event.certainty / 128.0))

        # Noun-phrase modifiers are stored directly on their entity (for
        # example ``a red car`` or ``blue eyes``).  Expose those explicit
        # attributes as a synthetic evidence frame rather than discarding the
        # information or inventing a value at answer time.
        if subject and subject.kind == RefKind.ENTITY and attribute:
            entity = memory.get_entity(subject.key)
            if entity:
                stored_value = entity.attributes.get(attribute.key)
                if stored_value:
                    synthetic = EventFrame(
                        "attribute",
                        {
                            "subject": subject,
                            "attribute": attribute,
                            "value": SemanticRef.literal(stored_value, stored_value, EntityKind.ABSTRACT),
                        },
                        tense="present",
                        source=SourceKind.USER,
                        certainty=230,
                        turn_index=entity.last_mentioned_turn,
                        inferred=False,
                    )
                    matches.append(EventMatch(synthetic, ["subject", "attribute"], 27.0))
        matches.sort(key=lambda item: (item.score, item.event.turn_index), reverse=True)
        return matches

    def _candidate_roles(self, question: QuestionFrame) -> Tuple[str, ...]:
        requested = question.requested_role or ""
        if question.kind == QuestionKind.WHY:
            return self.WHY_FALLBACKS.get(question.why_kind, (requested,))
        if question.kind == QuestionKind.HOW:
            return self.HOW_FALLBACKS.get(question.how_kind, (requested,))
        by_kind = self.QUESTION_ROLE_FALLBACKS.get(question.kind, {})
        if requested in by_kind:
            return by_kind[requested]
        if requested == "event":
            return ("event",)
        return (requested,)

    @staticmethod
    def _event_aspects_compatible(query: EventFrame, event: EventFrame) -> bool:
        """Keep direct QA inside the stored event's aspect boundary.

        Simple, progressive, perfect, and perfect-progressive propositions are
        distinct facts.  Any licensed implication between them must come from
        an explicit typed relation rather than the general event matcher.
        """

        return query.aspect == event.aspect

    @staticmethod
    def _to_evidence(match: EventMatch) -> Evidence:
        return Evidence(match.event, list(match.matched_roles), match.score)

    @staticmethod
    def _combine_sources(events: Sequence[EventFrame]) -> SourceKind:
        if not events:
            return SourceKind.UNKNOWN
        sources = {event.source for event in events}
        if len(sources) == 1:
            return next(iter(sources))
        if SourceKind.VERIFIED in sources:
            return SourceKind.VERIFIED
        return SourceKind.INFERRED

    @staticmethod
    def _same_anchor_sets(
        positive: Sequence[EventMatch],
        negative: Sequence[EventMatch],
        requested_role: str,
    ) -> bool:
        pos = {item.event.signature(exclude_roles={requested_role}) for item in positive}
        neg = {item.event.signature(exclude_roles={requested_role}) for item in negative}
        return bool(pos & neg)

    @staticmethod
    def _loosely_related(query: EventFrame, memory: ConversationMemory) -> List[EventMatch]:
        """Find facts sharing predicate and at least one argument.

        These are diagnostic evidence only; they never convert UNKNOWN to FALSE.
        """

        results: List[EventMatch] = []
        fixed = query.fixed_arguments()
        for event in memory.events:
            if event.discourse_role in memory.NONASSERTIVE_DISCOURSE_ROLES:
                continue
            if event.predicate != query.predicate:
                continue
            matched = [
                role for role, expected in fixed.items()
                if role in event.arguments and memory.refs_equal(expected, event.arguments[role])
            ]
            if matched:
                results.append(EventMatch(event, matched, len(matched) * 5.0))
        results.sort(key=lambda item: (item.score, item.event.turn_index), reverse=True)
        return results
