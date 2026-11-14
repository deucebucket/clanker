"""Embedded responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import Any, List
from .. import lexicon
from ..model import AnswerContract, EmbeddedInterrogativeRelation, EventFrame, QuestionKind

class EmbeddedComponent:
    """Embedded behavior composed by the stable public SurfaceRealizer API."""

    @staticmethod
    def _finite_content_verb(predicate: str, tense: str) -> str:
        if tense == "past":
            return lexicon.past_form(predicate)
        if tense == "future":
            return "will " + predicate
        return lexicon.present_form(predicate, third_person_singular=True)

    def _embedded_interrogative_relations_from_contract(
        self,
        contract: AnswerContract,
    ) -> List[EmbeddedInterrogativeRelation]:
        relation_ids = [
            item
            for item in contract.required_slots.get("relation_ids", "").split(",")
            if item
        ]
        if relation_ids:
            by_id = {
                item.relation_id: item
                for item in self.memory.embedded_interrogatives
            }
            return [
                by_id[relation_id]
                for relation_id in relation_ids
                if relation_id in by_id
            ]

        matrix_ids = {
            item
            for item in contract.required_slots.get("matrix_event_ids", "").split(",")
            if item
        }
        question_ids = {
            item
            for item in contract.required_slots.get("question_event_ids", "").split(",")
            if item
        }
        return [
            item
            for item in self.memory.embedded_interrogatives
            if (not matrix_ids or item.matrix_event_id in matrix_ids)
            and (not question_ids or item.question_event_id in question_ids)
        ]

    def render_embedded_interrogative_relation(
        self,
        relation: EmbeddedInterrogativeRelation,
        *,
        capitalize: bool = False,
    ) -> str:
        matrix = self.memory.get_event(relation.matrix_event_id)
        question_event = self.memory.get_event(relation.question_event_id)
        if matrix is None or question_event is None:
            return ""
        matrix_clause = self.render_event(matrix, capitalize=False)
        question_clause = self.render_embedded_question(
            relation,
            question_event,
            capitalize=False,
        )
        return self._finish_clause(
            " ".join(item for item in (matrix_clause, question_clause) if item),
            capitalize,
        )

    def render_embedded_question(
        self,
        relation: EmbeddedInterrogativeRelation,
        event: EventFrame,
        *,
        capitalize: bool = False,
    ) -> str:
        frame = relation.to_question_frame(event)
        return self.render_embedded_question_frame(
            frame,
            marker=relation.marker,
            capitalize=capitalize,
        )

    def render_embedded_question_frame(
        self,
        question: Any,
        *,
        marker: str = "",
        capitalize: bool = False,
    ) -> str:
        event = question.event
        operator = marker or getattr(question, "embedded_marker", "")
        if not operator:
            operator = {
                QuestionKind.WHO: "who",
                QuestionKind.WHAT: "what",
                QuestionKind.WHEN: "when",
                QuestionKind.WHERE: "where",
                QuestionKind.WHY: "why",
                QuestionKind.HOW: "how",
                QuestionKind.WHICH: "which",
                QuestionKind.WHOSE: "whose",
                QuestionKind.YES_NO: "whether",
            }.get(question.kind, "whether")

        requested_role = question.requested_role
        if question.kind == QuestionKind.YES_NO:
            clause = self.render_event(event, capitalize=False)
            return self._finish_clause(f"{operator} {clause}", capitalize)

        if requested_role == "possessor":
            focus = getattr(question, "focus_surface", "")
            clause = self.render_event(event, capitalize=False)
            tail = clause
            normalized_focus = focus.lower().strip()
            for prefix in (
                f"the {normalized_focus} ",
                f"a {normalized_focus} ",
                f"an {normalized_focus} ",
                f"{normalized_focus} ",
            ):
                if normalized_focus and clause.lower().startswith(prefix):
                    tail = clause[len(prefix):]
                    break
            terms = [operator]
            if focus:
                terms.append(focus)
            terms.append(tail)
            return self._finish_clause(" ".join(terms), capitalize)

        if requested_role in {"agent", "subject", "experiencer"}:
            tail = self._predicate_tail_for_question(event)
            focus = getattr(question, "focus_surface", "")
            terms = [operator]
            if focus and focus.lower() != operator.lower():
                terms.append(focus)
            terms.append(tail)
            return self._finish_clause(" ".join(terms), capitalize)

        clause = self.render_event(event, capitalize=False)
        focus = getattr(question, "focus_surface", "")
        terms = [operator]
        if focus and question.kind in {QuestionKind.WHICH, QuestionKind.WHOSE}:
            terms.append(focus)
        terms.append(clause)
        return self._finish_clause(" ".join(terms), capitalize)

    def render_embedded_interrogative_query(
        self,
        question: Any,
        *,
        capitalize: bool = False,
    ) -> str:
        matrix_clause = self.render_event(question.event, capitalize=False)
        inner = question.embedded_question
        if inner is None:
            return self._finish_clause(matrix_clause, capitalize)
        inner_clause = self.render_embedded_question_frame(
            inner,
            marker=question.embedded_marker,
            capitalize=False,
        )
        return self._finish_clause(
            f"{matrix_clause} {inner_clause}",
            capitalize,
        )
