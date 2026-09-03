"""Conformance tests for deterministic response-act planning."""

from __future__ import annotations

from itertools import product

import pytest

from clanker_lm.affect import AffectBackend, HeuristicAffectBackend
from clanker_lm.database import LanguageStore
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AffectReading,
    AffectVector,
    AnswerContract,
    AnswerStatus,
    EventFrame,
    GateDecision,
    ParseResult,
    SpeechAct,
)
from clanker_lm.realize import SurfaceRealizer
from clanker_lm.response_policy import ResponseActPlanner
from clanker_lm.runtime import ClankerLM


SUBJECTS = ("Sarah", "My sister", "The technician", "Our neighbor")
NEUTRAL_VERBS = (
    ("recorded", "record"),
    ("moved", "move"),
    ("opened", "open"),
    ("called", "call"),
    ("scheduled", "schedule"),
)
OBJECTS = ("the meeting", "a Honda", "the door", "the package", "an appointment")
NEUTRAL_CASES = tuple(product(SUBJECTS, NEUTRAL_VERBS, OBJECTS))


class BoundaryMisreadBackend(AffectBackend):
    """Return a moderate boundary vector for every sentence."""

    name = "boundary-misread"

    def analyze(self, text: str) -> AffectReading:
        return AffectReading(
            vector=AffectVector(v=100, a=181, d=124, u=0, g=120, w=128, i=136),
            backend=self.name,
        )

    def transition(self, state: AffectVector, message: AffectVector) -> AffectVector:
        values = {
            axis: round(getattr(state, axis) * 0.6 + getattr(message, axis) * 0.4)
            for axis in ("v", "a", "d", "u", "g", "w", "i")
        }
        return AffectVector(**values)


@pytest.mark.parametrize("subject,verb_pair,obj", NEUTRAL_CASES)
def test_neutral_fact_conformance_overrides_moderate_boundary(
    subject: str,
    verb_pair: tuple[str, str],
    obj: str,
) -> None:
    surface_verb, predicate = verb_pair
    text = f"{subject} {surface_verb} {obj}."
    parse = ParseResult(
        speech_act=SpeechAct.ASSERT,
        raw_text=text,
        events=[EventFrame(predicate=predicate)],
    )
    decision = ResponseActPlanner().plan(
        text,
        parse,
        BoundaryMisreadBackend().analyze(text),
        answer_status=AnswerStatus.ACKNOWLEDGED,
        initial_severity="moderate",
        initial_register="neutral",
    )
    assert decision.response_act == "neutral_acknowledge"
    assert decision.severity == "low"
    assert any("neutrality" in reason or "factual" in reason for reason in decision.rationale)


def test_runtime_honda_fact_is_not_recast_as_distress() -> None:
    runtime = ClankerLM(affect_backend=BoundaryMisreadBackend())
    try:
        result = runtime.process("My sister bought a used Honda yesterday.")
        assert result.gates.response_act == "neutral_acknowledge"
        assert result.gates.severity == "low"
        assert "?" not in result.response
        assert not {"rough", "serious", "sorry"} & set(result.response.lower().split())
        assert all(
            "RESPONSE_ACT:neutral_acknowledge" in candidate.semantic_plan
            for candidate in result.candidates
            if candidate.semantic_valid
        )
    finally:
        runtime.close()


def test_real_v8_honda_fact_uses_neutral_response_act() -> None:
    runtime = ClankerLM()
    try:
        result = runtime.process("My sister bought a used Honda yesterday.")
        assert result.gates.response_act == "neutral_acknowledge"
        assert "?" not in result.response
    finally:
        runtime.close()


def test_positive_outcome_uses_positive_composition() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        result = runtime.process("I passed my exam.")
        assert result.gates.response_act == "positive_acknowledge"
        assert "good" in result.response.lower()
        assert all(
            "RESPONSE_ACT:positive_acknowledge" in candidate.semantic_plan
            for candidate in result.candidates
            if candidate.semantic_valid
        )
    finally:
        runtime.close()


def test_completed_loss_does_not_force_a_followup_question() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        result = runtime.process("My dad died yesterday.")
        assert result.gates.response_act == "empathic_acknowledge"
        assert "?" not in result.response
    finally:
        runtime.close()


def test_serious_family_disclosure_retains_followup_and_masking() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        result = runtime.process("Bruh, my mom is really sick.")
        assert result.gates.response_act == "serious_followup"
        assert result.gates.severity in {"high", "critical"}
        assert result.gates.masking
        assert "?" in result.response
    finally:
        runtime.close()


def test_question_truth_contract_cannot_be_overridden_by_affect() -> None:
    runtime = ClankerLM(affect_backend=BoundaryMisreadBackend())
    try:
        runtime.process("Sarah opened the door.")
        result = runtime.process("Did Sarah open the door?")
        assert result.gates.response_act == "answer"
        assert result.contract.status in {AnswerStatus.TRUE, AnswerStatus.FALSE, AnswerStatus.CONFLICT}
    finally:
        runtime.close()


def test_neutral_acknowledgments_rotate_without_randomness() -> None:
    memory = ConversationMemory()
    with LanguageStore() as store:
        realizer = SurfaceRealizer(memory, store)
        contract = AnswerContract(status=AnswerStatus.ACKNOWLEDGED)
        gates = GateDecision(
            register="neutral",
            severity="low",
            response_act="neutral_acknowledge",
            max_sentences=1,
        )
        outputs = []
        for _ in range(3):
            memory.begin_turn()
            candidates = realizer.realize(contract, gates)
            assert len(candidates) == 1
            outputs.append(candidates[0].text)
        assert len(set(outputs)) == 3


def test_safe_is_positive_but_not_safe_is_serious() -> None:
    planner = ResponseActPlanner()
    parse = ParseResult(
        speech_act=SpeechAct.ASSERT,
        raw_text="I am safe.",
        events=[EventFrame(predicate="be")],
    )
    reading = HeuristicAffectBackend().analyze("I am safe.")
    safe = planner.plan(
        "I am safe.",
        parse,
        reading,
        answer_status=AnswerStatus.ACKNOWLEDGED,
        initial_severity="low",
        initial_register="neutral",
    )
    unsafe = planner.plan(
        "I am not safe.",
        parse,
        reading,
        answer_status=AnswerStatus.ACKNOWLEDGED,
        initial_severity="low",
        initial_register="neutral",
    )
    assert safe.response_act == "positive_acknowledge"
    assert unsafe.response_act == "serious_followup"
