"""Integration tests against the repository's real Clanker VADUGWI kernel."""

from clanker_lm import AnswerStatus, ClankerAffectAdapter, ClankerLM


def test_strict_adapter_uses_real_clanker_engine():
    affect = ClankerAffectAdapter(strict=True)
    assert affect.using_clanker is True

    reading = affect.analyze("My sister pissed me off again.")
    assert 0 <= reading.vector.v <= 255
    assert 0 <= reading.vector.a <= 255
    assert 0 <= reading.vector.d <= 255
    assert 0 <= reading.vector.u <= 255
    assert 0 <= reading.vector.g <= 255
    assert 0 <= reading.vector.w <= 255
    assert 0 <= reading.vector.i <= 255
    assert reading.source == "clanker"


def test_real_clanker_scores_but_cannot_override_semantic_truth():
    bot = ClankerLM(strict_clanker=True)
    bot.reply("My sister bought a used Honda yesterday.")
    result = bot.process("What did she buy?")

    assert result.answer and result.answer.status == AnswerStatus.ANSWERED
    assert "Honda" in result.response
    selected = next(item for item in result.candidate_scores if item.selected)
    assert selected.semantic_valid is True
    assert selected.hard_rejected is False
    assert result.response_affect.source == "clanker"
