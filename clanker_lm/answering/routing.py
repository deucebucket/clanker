"""Routing responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from ..memory import ConversationMemory
from ..model import AnswerContract, AnswerStatus, QuestionFrame, QuestionKind, SourceKind

class RoutingComponent:
    """Routing behavior composed by the stable public QuestionAnswerer API."""

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
