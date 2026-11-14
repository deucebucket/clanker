"""Infinitives responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Sequence
from ..memory import ConversationMemory
from ..model import AnswerContract, AnswerStatus, EntityKind, Evidence, EventFrame, InfinitivalRelation, QuestionFrame, QuestionKind, RefKind, SemanticRef, SourceKind, TruthValue

class InfinitivesComponent:
    """Infinitives behavior composed by the stable public QuestionAnswerer API."""

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
