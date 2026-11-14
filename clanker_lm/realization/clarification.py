"""Clarification responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import List
from .. import lexicon
from ..model import AnswerContract, CandidateResponse, GateDecision
from .types import Part

class ClarificationComponent:
    """Clarification behavior composed by the stable public SurfaceRealizer API."""

    def _missing_reference(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        reference = contract.required_slots.get("reference", contract.reason or "it")
        operator_id = "question.who" if self._reference_is_person(reference) else "question.what"
        parts: List[Part] = [
            self._atom(operator_id),
            self._atom("aux.do"),
            self._atom("pronoun.you"),
            self._atom("communication.mean"),
            self._atom("prep.by"),
            reference,
            self._atom("punct.question"),
        ]
        return [
            self._candidate(
                parts,
                candidate_id="compose.probe.missing_reference",
                semantic_plan=[*self._rule_plan("reply:probe"), f"UNRESOLVED:{reference}"],
                priority=120,
            )
        ]

    def _ambiguous_reference(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        ids = [item for item in contract.required_slots.get("candidate_ids", "").split(",") if item]
        labels = [self.memory.describe_entity(entity_id) for entity_id in ids]
        alternatives = self.join_phrases(labels, conjunction=self._atom("link.or").surface)
        return [
            self._candidate(
                [
                    self._atom("aux.do"),
                    self._atom("pronoun.you"),
                    self._atom("communication.mean"),
                    alternatives or contract.required_slots.get("reference", "that"),
                    self._atom("punct.question"),
                ],
                candidate_id="compose.probe.ambiguous_reference",
                semantic_plan=[*self._rule_plan("reply:probe"), *[f"CANDIDATE:{item}" for item in ids]],
                priority=120,
            )
        ]

    def _lexical_probe(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        term = contract.required_slots.get("term", contract.reason or "word")
        axis = contract.required_slots.get("probe_axis", "definition")
        plan = [*self._rule_plan("reply:probe"), f"LEXEME:{term}", f"PROBE_AXIS:{axis}"]
        if axis in {"definition", "meaning"}:
            # A user asking for an unknown definition gets an explicit
            # uncertainty statement followed by the smallest useful probe.
            # Every element is still composed from atoms, inflection, and the
            # dynamic term; no completed response sentence is stored.
            if contract.required_slots.get("semantic_position") == "definition_query":
                means = lexicon.present_form(
                    self._atom("communication.mean").lemma,
                    third_person_singular=True,
                )
                parts: List[Part] = [
                    self._atom("pronoun.i"),
                    self._atom("aux.do"),
                    self._atom("neg.not"),
                    self._atom("cognition.know"),
                    self._atom("question.what"),
                    term,
                    means,
                    self._atom("punct.period"),
                    self._atom("learn.example"),
                    self._atom("punct.question"),
                ]
                return [
                    self._candidate(
                        parts,
                        candidate_id="compose.lexical.unknown_definition_probe",
                        semantic_plan=[*plan, "TRUTH:UNKNOWN", "ACT:REQUEST_EXAMPLE"],
                        priority=130,
                    )
                ]
            parts = [
                self._atom("question.what"),
                self._atom("aux.does"),
                term,
                self._atom("communication.mean"),
                self._atom("punct.question"),
            ]
            return [self._candidate(parts, candidate_id="compose.lexical.definition_probe", semantic_plan=plan, priority=125)]
        atom_id = {
            "polarity": "learn.polarity",
            "semantic_class": "learn.meaning",
            "intensity": "learn.intense",
            "example": "learn.example",
        }.get(axis, "learn.meaning")
        atom = self.store.atom_by_id(atom_id)
        if atom is None and axis == "polarity":
            atom = self._atom("learn.positive")
        assert atom is not None
        return [
            self._candidate(
                [atom, self._atom("punct.question")],
                candidate_id=f"compose.lexical.one_word.{axis}",
                semantic_plan=plan,
                priority=125,
            )
        ]

    def _lexical_learned(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        if contract.proposition is not None:
            return self._answered(contract, gates)
        return self._acknowledge(contract, gates)

    def _unsupported(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        return [
            self._candidate(
                [
                    self._atom("pronoun.i"),
                    self._atom("neg.cannot"),
                    self._atom("verb.determine"),
                    self._atom("det.the"),
                    self._atom("meta.answer"),
                    self._atom("punct.period"),
                ],
                candidate_id="compose.unsupported",
                semantic_plan=["ROUTE:FAIL_CLOSED", "EPISTEMIC:UNSUPPORTED"],
                priority=1,
            )
        ]
