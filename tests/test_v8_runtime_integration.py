from __future__ import annotations

import importlib.util

import pytest

from clanker_lm import ClankerLM
from clanker_lm.model import AnswerStatus


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("engine") is None,
    reason="existing Clanker engine is absent from isolated package test",
)


def assert_v8_turn(result, expected_status: AnswerStatus) -> None:
    assert result.input_affect.backend == "clanker-v8"
    assert result.contract.status is expected_status
    assert result.response.strip()
    assert any(
        candidate.text == result.response and candidate.semantic_valid
        for candidate in result.candidates
    )
    for state in (result.observed_state, result.target_state, result.predicted_state):
        assert all(0 <= value <= 255 for value in state.to_dict().values())


def test_real_v8_backend_drives_complete_question_answer_dialogue() -> None:
    with ClankerLM() as runtime:
        assert runtime.affect_backend_name == "clanker-v8"

        assert_v8_turn(
            runtime.process("My sister bought a used Honda yesterday."),
            AnswerStatus.ACKNOWLEDGED,
        )
        assert_v8_turn(
            runtime.process("Who bought the Honda?"),
            AnswerStatus.ANSWERED,
        )
        assert_v8_turn(
            runtime.process("What did she buy?"),
            AnswerStatus.ANSWERED,
        )
        assert_v8_turn(
            runtime.process("Why did she buy it?"),
            AnswerStatus.UNKNOWN,
        )
        assert_v8_turn(
            runtime.process("She bought it because her old car broke down."),
            AnswerStatus.ACKNOWLEDGED,
        )
        final = runtime.process("Why did she buy it?")
        assert_v8_turn(final, AnswerStatus.ANSWERED)
        assert "broke down" in final.response.lower()
        assert runtime.memory.events


def test_real_v8_backend_preserves_collision_masking_gate() -> None:
    with ClankerLM() as runtime:
        result = runtime.process("Bruh, my mom is really sick.")
        assert_v8_turn(result, AnswerStatus.ACKNOWLEDGED)
        assert result.gates.masking
        assert result.gates.register == "casual"
        assert {"formal", "humor", "slang", "high_severity"}.issubset(
            result.gates.locked_pools
        )
