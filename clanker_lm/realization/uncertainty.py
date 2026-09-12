"""Uncertainty responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import List, Tuple
from .. import lexicon
from ..model import AnswerContract, CandidateResponse, GateDecision, QuestionKind
from .types import Part

class UncertaintyComponent:
    """Uncertainty behavior composed by the stable public SurfaceRealizer API."""

    def _unknown(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        question = contract.question
        period = self._atom("punct.period")
        i_atom = self._atom("pronoun.i")
        dont = self._atom("neg.dont")
        know = self._atom("cognition.know")
        plan = self._rule_plan("reply:unknown")

        if (
            contract.required_slots.get("direct_embedded_answer_request") == "true"
            and question is not None
        ):
            question_phrase = self.render_embedded_question_frame(
                question,
                capitalize=False,
            )
            return [
                self._candidate(
                    [i_atom, dont, know, question_phrase, period],
                    candidate_id="compose.unknown.direct_embedded_request",
                    semantic_plan=[
                        *plan,
                        f"INNER_QUESTION:{question.kind.value}",
                        "TRUTH:UNKNOWN",
                    ],
                    priority=122,
                )
            ]

        if (
            contract.required_slots.get("embedded_interrogative_query") == "true"
            and question is not None
        ):
            query_phrase = self.render_embedded_interrogative_query(
                question,
                capitalize=False,
            )
            return [
                self._candidate(
                    [
                        i_atom,
                        dont,
                        know,
                        self._atom("question.whether"),
                        query_phrase,
                        period,
                    ],
                    candidate_id="compose.unknown.embedded_interrogative",
                    semantic_plan=[
                        *plan,
                        f"MATRIX_PREDICATE:{question.event.predicate}",
                        "TRUTH:QUESTION_ATTRIBUTION_UNKNOWN",
                    ],
                    priority=120,
                )
            ]

        if (
            contract.required_slots.get("infinitival_evidence") == "true"
            and question is not None
        ):
            relation_ids = [
                item
                for item in contract.required_slots.get(
                    "relation_ids",
                    "",
                ).split(",")
                if item
            ]
            relations = [
                relation
                for relation_id in relation_ids[:3]
                for relation in [
                    next(
                        (
                            item
                            for item in self.memory.infinitivals
                            if item.relation_id == relation_id
                        ),
                        None,
                    )
                ]
                if relation is not None
            ]
            if relations:
                context_parts: List[Part] = []
                for index, relation in enumerate(relations):
                    if index:
                        context_parts.append(self._atom("link.but"))
                    context_parts.append(
                        self.render_infinitival_relation(
                            relation,
                            capitalize=(index == 0),
                        )
                    )
                query_clause = self.render_event(
                    question.event,
                    capitalize=False,
                )
                context_parts.extend(
                    [
                        period,
                        i_atom,
                        dont,
                        know,
                        self._atom("question.whether"),
                        query_clause,
                        period,
                    ]
                )
                return [
                    self._candidate(
                        context_parts,
                        candidate_id="compose.unknown.infinitival_boundary",
                        semantic_plan=[
                            *plan,
                            "EVIDENCE:NONENTAILED_INFINITIVE",
                            f"QUERY_FRAME:{question.event.predicate}",
                        ],
                        priority=120,
                    )
                ]

        if (
            contract.required_slots.get("gerund_evidence") == "true"
            and question is not None
        ):
            relation_ids = [
                item
                for item in contract.required_slots.get(
                    "relation_ids",
                    "",
                ).split(",")
                if item
            ]
            by_id = {
                relation.relation_id: relation
                for relation in self.memory.gerunds
            }
            relations = [
                by_id[relation_id]
                for relation_id in relation_ids[:3]
                if relation_id in by_id
            ]
            if relations:
                context_parts = []
                for index, relation in enumerate(relations):
                    if index:
                        context_parts.append(self._atom("link.but"))
                    context_parts.append(
                        self.render_gerund_relation(
                            relation,
                            capitalize=(index == 0),
                        )
                    )
                query_clause = self.render_event(
                    question.event,
                    capitalize=False,
                )
                context_parts.extend(
                    [
                        period,
                        i_atom,
                        dont,
                        know,
                        self._atom("question.whether"),
                        query_clause,
                        period,
                    ]
                )
                return [
                    self._candidate(
                        context_parts,
                        candidate_id="compose.unknown.gerund_boundary",
                        semantic_plan=[
                            *plan,
                            *[
                                f"GERUND_RELATION:{item.relation_type.value}"
                                for item in relations
                            ],
                            *[
                                f"CONTENT_STATUS:{item.content_status.value}"
                                for item in relations
                            ],
                            "EVIDENCE:QUALIFIED_GERUND_CONTENT",
                            "TRUTH:EMBEDDED_EVENT_UNKNOWN",
                            f"QUERY_FRAME:{question.event.predicate}",
                        ],
                        priority=121,
                    )
                ]

        if contract.proposition is not None and question is not None and question.requested_role:
            omitted = self._reason_roles(question)
            known_clause = self.render_event(contract.proposition, omit_roles=omitted, capitalize=False)
            role_surface, role_atom = self._role_surface(question.requested_role)
            parts: List[Part] = [
                i_atom,
                know,
                known_clause,
                self._atom("punct.comma"),
                self._atom("link.but"),
                self._atom("pronoun.you"),
                self._atom("neg.havent"),
                lexicon.participle_form(self._atom("communication.tell").lemma),
                self._atom("pronoun.me"),
                self._atom("det.the"),
                role_atom or role_surface,
                period,
            ]
            return [
                self._candidate(
                    parts,
                    candidate_id="compose.unknown.known_base",
                    semantic_plan=[*plan, f"KNOWN_FRAME:{contract.proposition.predicate}", f"MISSING_ROLE:{question.requested_role}"],
                    priority=112,
                )
            ]

        if question is not None and question.kind == QuestionKind.YES_NO:
            query_clause = self.render_event(question.event, capitalize=False)
            if contract.required_slots.get("attributed") == "true":
                attributed_parts: List[Part] = []
                seen: set[Tuple[str, str]] = set()
                for evidence in contract.evidence:
                    relations = self.memory.content_relations_for_event(
                        evidence.event.event_id
                    )
                    relation = next(
                        (
                            item
                            for item in relations
                            if item.content_event_id == evidence.event.event_id
                        ),
                        None,
                    )
                    if relation is None:
                        continue
                    key = (relation.source_entity_id, evidence.event.event_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    source = self.memory.describe_entity(relation.source_entity_id)
                    verb = self._finite_content_verb(
                        relation.matrix_predicate,
                        self.memory.get_event(relation.matrix_event_id).tense
                        if self.memory.get_event(relation.matrix_event_id) is not None
                        else "present",
                    )
                    clause = self.render_event(evidence.event, capitalize=False)
                    if attributed_parts:
                        attributed_parts.append(self._atom("link.but"))
                    attributed_parts.extend([source, verb, clause])
                    if len(seen) >= 3:
                        break
                if attributed_parts:
                    attributed_parts.extend(
                        [
                            period,
                            i_atom,
                            dont,
                            know,
                            self._atom("question.whether"),
                            query_clause,
                            period,
                        ]
                    )
                    return [
                        self._candidate(
                            attributed_parts,
                            candidate_id="compose.unknown.attributed_polar",
                            semantic_plan=[
                                *plan,
                                "EVIDENCE:ATTRIBUTED_ONLY",
                                f"QUERY_FRAME:{question.event.predicate}",
                            ],
                            priority=118,
                        )
                    ]
            return [
                self._candidate(
                    [i_atom, dont, know, self._atom("question.whether"), query_clause, period],
                    candidate_id="compose.unknown.polar",
                    semantic_plan=[*plan, f"QUERY_FRAME:{question.event.predicate}"],
                    priority=108,
                )
            ]

        unknown_object = contract.required_slots.get("unknown_object")
        if not unknown_object and question is not None:
            unknown_object = question.requested_role or "answer"
        unknown_object = unknown_object or "answer"
        role_surface, role_atom = self._role_surface(unknown_object)
        return [
            self._candidate(
                [i_atom, dont, know, self._atom("det.the"), role_atom or role_surface, period],
                candidate_id="compose.unknown.slot",
                semantic_plan=[*plan, f"UNKNOWN_SLOT:{unknown_object}"],
                priority=100,
            )
        ]
