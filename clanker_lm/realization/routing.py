"""Routing responsibilities; extracted without changing decision rules."""

from __future__ import annotations
from typing import List
from ..database import LanguageStore
from ..memory import ConversationMemory
from ..model import AnswerContract, AnswerStatus, CandidateResponse, GateDecision

class RoutingComponent:
    """Routing behavior composed by the stable public SurfaceRealizer API."""

    def __init__(self, memory: ConversationMemory, store: LanguageStore) -> None:
        self.memory = memory
        self.store = store
        self.store.assert_template_free()

    def realize(self, contract: AnswerContract, gates: GateDecision) -> List[CandidateResponse]:
        status = contract.status
        if status == AnswerStatus.ANSWERED:
            candidates = self._answered(contract, gates)
        elif status == AnswerStatus.TRUE:
            candidates = self._polar(contract, gates, positive=True)
        elif status == AnswerStatus.FALSE:
            candidates = self._polar(contract, gates, positive=False)
        elif status == AnswerStatus.UNKNOWN:
            candidates = self._unknown(contract, gates)
        elif status == AnswerStatus.CONFLICT:
            candidates = self._conflict(contract, gates)
        elif status == AnswerStatus.MISSING_REFERENCE:
            candidates = self._missing_reference(contract, gates)
        elif status == AnswerStatus.AMBIGUOUS_REFERENCE:
            candidates = self._ambiguous_reference(contract, gates)
        elif status == AnswerStatus.MULTIPLE_MATCHES:
            candidates = self._multiple_matches(contract, gates)
        elif status == AnswerStatus.LEXICAL_PROBE:
            candidates = self._lexical_probe(contract, gates)
        elif status == AnswerStatus.LEXICAL_LEARNED:
            candidates = self._lexical_learned(contract, gates)
        elif status == AnswerStatus.ACKNOWLEDGED:
            candidates = self._acknowledge(contract, gates)
        else:
            candidates = self._unsupported(contract, gates)

        valid_candidates: List[CandidateResponse] = []
        for candidate in candidates:
            valid, reason = self._validate_surface(candidate.text, contract)
            candidate.semantic_valid = candidate.semantic_valid and valid
            candidate.semantic_reason = reason if not valid else candidate.semantic_reason or reason
            valid_candidates.append(candidate)
        if valid_candidates:
            return valid_candidates
        return self._unsupported(contract, gates)
