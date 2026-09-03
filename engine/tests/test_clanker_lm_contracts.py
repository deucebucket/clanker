"""Semantic contract, provenance, and candidate-gating tests."""

from clanker_lm import AnswerStatus, ClankerLM, RuleAffectAdapter
from clanker_lm.answers import SemanticValidator
from clanker_lm.models import ResponseCandidate


def _bot():
    return ClankerLM(affect=RuleAffectAdapter())


def test_unknown_is_not_treated_as_false_closed_world_answer():
    bot = _bot()
    bot.reply("My sister bought a Honda.")
    result = bot.process("Did my mother buy the Honda?")

    assert result.answer and result.answer.status == AnswerStatus.UNKNOWN
    assert result.answer.truth_value is None


def test_user_fact_carries_user_provenance_and_explicit_certainty():
    bot = _bot()
    bot.process("My sister bought a Honda.", certainty=207)
    result = bot.process("What did she buy?")

    assert result.answer and result.answer.status == AnswerStatus.ANSWERED
    assert result.answer.certainty == 207
    assert result.answer.provenance.value == "user"


def test_multiple_distinct_bindings_produce_ambiguous_answer():
    bot = _bot()
    bot.reply("My sister bought a Honda.")
    bot.reply("My sister bought a Toyota.")
    result = bot.process("What did she buy?")

    assert result.answer and result.answer.status == AnswerStatus.AMBIGUOUS
    assert any(marker in result.response.lower() for marker in ("conflicting", "multiple", "more than one"))


def test_semantic_validator_rejects_assertive_unknown_candidate():
    bot = _bot()
    bot.reply("My sister bought a Honda.")
    contract = bot.process("Why did she buy it?").answer
    assert contract is not None

    valid, reasons = SemanticValidator().validate(
        ResponseCandidate("bad", "She bought it because it was cheap."), contract
    )

    # PARTIAL_UNKNOWN is permitted only as an uncertainty response.  This
    # candidate has no uncertainty marker and must not be selected.
    assert valid is False
    assert reasons


def test_all_selected_factual_candidates_pass_semantic_gate():
    bot = _bot()
    bot.reply("My sister bought a Honda yesterday.")
    result = bot.process("When did she buy it?")

    selected = next(item for item in result.candidate_scores if item.selected)
    assert selected.semantic_valid is True
    assert selected.hard_rejected is False
