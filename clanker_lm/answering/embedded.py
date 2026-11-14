"""Embedded responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import AnswerContract, AnswerStatus, EmbeddedInterrogativeRelation, Evidence, EventFrame, QuestionFrame, QuestionKind, RefKind, SemanticRef, SourceKind, TruthValue

class EmbeddedComponent:
    """Embedded behavior composed by the stable public QuestionAnswerer API."""

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
