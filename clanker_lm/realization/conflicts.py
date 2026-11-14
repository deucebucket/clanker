"""Conflicts responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import List
from ..model import AnswerContract, CandidateResponse, GateDecision, GerundRelation
from .types import Part

class ConflictsComponent:
    """Conflicts behavior composed by the stable public SurfaceRealizer API."""

    def _conflict(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        if (
            contract.required_slots.get("gerund") == "true"
            or contract.required_slots.get("gerund_evidence") == "true"
            or any(
                evidence.event.discourse_role == "gerund"
                for evidence in contract.evidence
            )
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
            relations: List[GerundRelation] = []
            seen_ids: set[str] = set()
            for relation_id in relation_ids[:4]:
                relation = by_id.get(relation_id)
                if relation is None or relation_id in seen_ids:
                    continue
                relations.append(relation)
                seen_ids.add(relation_id)

            clauses = [
                clause
                for relation in relations
                for clause in [
                    self.render_gerund_relation(
                        relation,
                        capitalize=False,
                    )
                ]
                if clause
            ]
            summary = self.join_phrases(
                clauses,
                conjunction=self._atom("link.but").surface,
            )
            parts: List[Part] = [
                self._atom("pronoun.i"),
                self._atom("aux.have"),
                self._atom("meta.conflicting"),
                self._atom("meta.information"),
            ]
            if summary:
                parts.extend([self._atom("punct.colon"), summary])
            parts.append(self._atom("punct.period"))
            return [
                self._candidate(
                    parts,
                    candidate_id="compose.conflict.gerund_relations",
                    semantic_plan=[
                        *self._rule_plan("reply:unknown"),
                        *[
                            f"GERUND_RELATION:{item.relation_type.value}"
                            for item in relations
                        ],
                        *[
                            f"CONTENT_STATUS:{item.content_status.value}"
                            for item in relations
                        ],
                        "EVIDENCE:CONTRADICTORY_QUALIFIED_GERUND_RELATIONS",
                        "TRUTH:EMBEDDED_EVENTS_REMAIN_MATRIX_QUALIFIED",
                        (
                            "RELATION_LOOKUP:BOUND"
                            if relations
                            else "RELATION_LOOKUP:FAILED_CLOSED"
                        ),
                    ],
                    priority=124,
                )
            ]

        if contract.required_slots.get("embedded_interrogative") == "true":
            relations = self._embedded_interrogative_relations_from_contract(contract)
            clauses = [
                self.render_embedded_interrogative_relation(
                    relation,
                    capitalize=False,
                )
                for relation in relations[:4]
            ]
            summary = self.join_phrases(
                clauses,
                conjunction=self._atom("link.but").surface,
            )
            return [
                self._candidate(
                    [
                        self._atom("pronoun.i"),
                        self._atom("aux.have"),
                        self._atom("meta.conflicting"),
                        self._atom("meta.information"),
                        self._atom("punct.colon"),
                        summary,
                        self._atom("punct.period"),
                    ],
                    candidate_id="compose.conflict.embedded_interrogative",
                    semantic_plan=[
                        *self._rule_plan("reply:unknown"),
                        "EVIDENCE:CONTRADICTORY_QUESTION_ATTRIBUTION",
                    ],
                    priority=122,
                )
            ]

        clauses: List[str] = []
        seen: set[str] = set()
        for evidence in contract.evidence:
            clause = self.render_event(evidence.event, capitalize=False)
            if clause and clause not in seen:
                clauses.append(clause)
                seen.add(clause)
            if len(clauses) >= 3:
                break
        summary = self.join_phrases(clauses, conjunction=self._atom("link.but").surface)
        parts: List[Part] = [
            self._atom("pronoun.i"),
            self._atom("aux.have"),
            self._atom("meta.conflicting"),
            self._atom("meta.information"),
        ]
        if summary:
            parts.extend([self._atom("punct.colon"), summary])
        parts.append(self._atom("punct.period"))
        return [
            self._candidate(
                parts,
                candidate_id="compose.conflict.evidence",
                semantic_plan=[*self._rule_plan("reply:unknown"), "EVIDENCE:CONTRADICTORY"],
                priority=115,
            )
        ]

    def _multiple_matches(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        values = [self.render_ref(value, definite=False) for value in contract.values]
        listing = self.join_phrases(values, conjunction=self._atom("link.and").surface)
        first = self._candidate(
            [
                self._atom("pronoun.i"),
                self._atom("verb.find"),
                self._atom("meta.more"),
                self._atom("prep.than"),
                self._atom("meta.one"),
                self._atom("meta.answer"),
                self._atom("punct.colon"),
                listing,
                self._atom("punct.period"),
                self._atom("question.which"),
                self._atom("meta.one"),
                self._atom("aux.do"),
                self._atom("pronoun.you"),
                self._atom("communication.mean"),
                self._atom("punct.question"),
            ],
            candidate_id="compose.probe.multiple_values",
            semantic_plan=[*self._rule_plan("reply:probe"), *[f"VALUE:{value.key}" for value in contract.values]],
            priority=118,
        )
        return [first]
