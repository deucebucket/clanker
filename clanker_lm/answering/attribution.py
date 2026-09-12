"""Attribution responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import Optional
from ..memory import ConversationMemory
from ..model import AnswerContract, AnswerStatus, Evidence, QuestionFrame, QuestionKind, RefKind, SemanticRef, SourceKind

class AttributionComponent:
    """Attribution behavior composed by the stable public QuestionAnswerer API."""

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
