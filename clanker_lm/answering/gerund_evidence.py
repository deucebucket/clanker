"""Gerund Evidence responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
from .. import lexicon
from ..memory import ConversationMemory
from ..model import AnswerContract, AnswerStatus, Evidence, EventFrame, GerundRelation, GerundRelationType, QuestionFrame, RefKind, SemanticRef, SourceKind, TruthValue

class GerundEvidenceComponent:
    """Gerund Evidence behavior composed by the stable public QuestionAnswerer API."""

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
