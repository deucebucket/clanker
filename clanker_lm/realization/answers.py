"""Answers responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import List
from ..model import AnswerContract, CandidateResponse, GateDecision, QuestionKind

class AnswersComponent:
    """Answers behavior composed by the stable public SurfaceRealizer API."""

    def _answered(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        if contract.response_goal == "social":
            return self._social(contract, gates)
        if contract.proposition is None:
            return self._unsupported(contract, gates)
        if contract.required_slots.get("gerund") == "true":
            relation = self._gerund_relation_from_contract(contract)
            if relation is None:
                return self._unsupported(contract, gates)
            clause = self.render_gerund_relation(
                relation,
                capitalize=True,
            )
            return [
                self._candidate(
                    [clause, self._atom("punct.period")],
                    candidate_id="compose.answer.gerund_relation",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        f"GERUND_RELATION:{relation.relation_type.value}",
                        f"CONTENT_STATUS:{relation.content_status.value}",
                        f"MATRIX_PREDICATE:{relation.matrix_predicate}",
                        f"CONTROLLER:{relation.controller_entity_id}",
                        "TRUTH:SELECTED_CONTENT_QUALIFIED_BY_MATRIX",
                    ],
                    priority=119,
                )
            ]
        if contract.required_slots.get("embedded_interrogative") == "true":
            relations = self._embedded_interrogative_relations_from_contract(contract)
            if not relations:
                return self._unsupported(contract, gates)
            clauses = [
                self.render_embedded_interrogative_relation(
                    relation,
                    capitalize=(index == 0),
                )
                for index, relation in enumerate(relations[:4])
            ]
            clause = self.join_phrases(
                clauses,
                conjunction=self._atom("link.and").surface,
            )
            return [
                self._candidate(
                    [clause, self._atom("punct.period")],
                    candidate_id="compose.answer.embedded_interrogative",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        *[
                            f"EMBEDDED_INTERROGATIVE:{item.relation_type.value}"
                            for item in relations[:4]
                        ],
                        *[
                            f"MATRIX_PREDICATE:{item.matrix_predicate}"
                            for item in relations[:4]
                        ],
                        "TRUTH:QUESTION_ATTRIBUTION_ONLY",
                    ],
                    priority=120,
                )
            ]
        if contract.required_slots.get("infinitival") == "true":
            relation = self._infinitival_relation_from_contract(contract)
            if relation is None:
                return self._unsupported(contract, gates)
            clause = self.render_infinitival_relation(
                relation,
                capitalize=True,
            )
            return [
                self._candidate(
                    [clause, self._atom("punct.period")],
                    candidate_id="compose.answer.infinitival_relation",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        f"INFINITIVAL_RELATION:{relation.relation_type.value}",
                        f"CONTENT_STATUS:{relation.content_status.value}",
                        f"MATRIX_PREDICATE:{relation.matrix_predicate}",
                        f"CONTROLLER:{relation.controller_entity_id}",
                    ],
                    priority=118,
                )
            ]
        if contract.required_slots.get("attributed") == "true":
            source_id = contract.required_slots.get("source_entity_id", "")
            predicate = contract.required_slots.get("matrix_predicate", "say")
            tense = contract.required_slots.get("matrix_tense", "present")
            source = self.memory.describe_entity(source_id)
            content = self.render_event(contract.proposition, capitalize=False)
            verb = self._finite_content_verb(predicate, tense)
            return [
                self._candidate(
                    [source, verb, content, self._atom("punct.period")],
                    candidate_id="compose.answer.attributed_content",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        f"ATTRIBUTION_SOURCE:{source_id}",
                        f"ATTRIBUTION_PREDICATE:{predicate}",
                        f"CONTENT_FRAME:{contract.proposition.predicate}",
                    ],
                    priority=116,
                )
            ]
        if (
            contract.question is not None
            and contract.question.kind == QuestionKind.WHOSE
            and contract.values
        ):
            owner = self.render_ref(
                contract.values[0],
                case="subject",
                definite=True,
                capitalize=True,
            )
            return [
                self._candidate(
                    [owner, self._atom("punct.period")],
                    candidate_id="compose.answer.possessor",
                    semantic_plan=[
                        *self._rule_plan("reply:answer"),
                        "ROLE:possessor",
                        f"VALUE:{contract.values[0].key}",
                    ],
                    priority=108,
                )
            ]
        clause = self.render_event(contract.proposition, capitalize=True)
        return [
            self._candidate(
                [clause, self._atom("punct.period")],
                candidate_id="compose.answer.proposition",
                semantic_plan=[*self._rule_plan("reply:answer"), f"FRAME:{contract.proposition.predicate}"],
                priority=100,
            )
        ]

    def _polar(self, contract: AnswerContract, gates: GateDecision, *, positive: bool) -> List[CandidateResponse]:
        if contract.proposition is None:
            return self._unknown(contract, gates)
        category = "affirmation" if positive else "denial"
        preferred = (
            ("polarity.yeah", "polarity.yes")
            if positive and gates.register == "casual"
            else ("polarity.yes", "polarity.yeah")
            if positive
            else ("polarity.nope", "polarity.no")
            if gates.register == "casual"
            else ("polarity.no", "polarity.nope")
        )
        particles = self._atoms(category, register=gates.register, preferred=preferred)
        semantic_plan_details: List[str] = []
        if contract.required_slots.get("embedded_memory_probe") == "true":
            known = contract.required_slots.get("embedded_memory_known") == "true"
            if known:
                clause = self.render_event(contract.proposition, capitalize=True)
                semantic_plan_details.extend([
                    "OUTER_EPISTEMIC_QUERY:true",
                    "INNER_QUESTION_ANSWERABLE:true",
                ])
            else:
                inner = contract.question.embedded_question if contract.question else None
                inner_phrase = (
                    self.render_embedded_question_frame(inner, capitalize=False)
                    if inner is not None
                    else self.render_event(contract.proposition, capitalize=False)
                )
                clause, _ = self._compose([
                    self._atom("pronoun.i"),
                    self._atom("aux.do"),
                    self._atom("neg.not"),
                    self._atom("cognition.know"),
                    inner_phrase,
                ])
                semantic_plan_details.extend([
                    "OUTER_EPISTEMIC_QUERY:true",
                    "INNER_QUESTION_ANSWERABLE:false",
                ])
        elif contract.required_slots.get("embedded_interrogative") == "true":
            relations = self._embedded_interrogative_relations_from_contract(contract)
            if not relations:
                return self._unknown(contract, gates)
            clauses = [
                self.render_embedded_interrogative_relation(
                    relation,
                    capitalize=(index == 0),
                )
                for index, relation in enumerate(relations[:4])
            ]
            clause = self.join_phrases(
                clauses,
                conjunction=self._atom("link.but").surface,
            )
            semantic_plan_details.extend([
                *[
                    f"EMBEDDED_INTERROGATIVE:{item.relation_type.value}"
                    for item in relations[:4]
                ],
                "TRUTH:QUESTION_ATTRIBUTION_ONLY",
            ])
        else:
            gerund_relation = (
                self._gerund_relation_from_contract(contract)
                if contract.required_slots.get("gerund") == "true"
                else None
            )
            if contract.required_slots.get("gerund") == "true":
                if gerund_relation is None:
                    return self._unknown(contract, gates)
                clause = self.render_gerund_relation(
                    gerund_relation,
                    capitalize=True,
                )
                semantic_plan_details.append(
                    f"GERUND_RELATION:{gerund_relation.relation_type.value}"
                )
                semantic_plan_details.append(
                    f"CONTENT_STATUS:{gerund_relation.content_status.value}"
                )
            else:
                relation = (
                    self._infinitival_relation_from_contract(contract)
                    if contract.required_slots.get("infinitival") == "true"
                    else None
                )
                clause = (
                    self.render_infinitival_relation(relation, capitalize=True)
                    if relation is not None
                    else self.render_event(contract.proposition, capitalize=True)
                )
                if relation is not None:
                    semantic_plan_details.append(
                        f"INFINITIVAL_RELATION:{relation.relation_type.value}"
                    )
        candidates: List[CandidateResponse] = []
        for index, particle in enumerate(particles[:2]):
            candidates.append(
                self._candidate(
                    [particle, self._atom("punct.period"), clause, self._atom("punct.period")],
                    candidate_id=f"compose.polar.{particle.atom_id}",
                    semantic_plan=[
                        *self._rule_plan(f"reply:{'true' if positive else 'false'}", {"polarity": positive}),
                        f"FRAME:{contract.proposition.predicate}",
                        *semantic_plan_details,
                    ],
                    priority=105 - index,
                )
            )
        return candidates
