"""SQLite session persistence and isolation tests."""

from clanker_lm import AnswerStatus, ClankerLM, RuleAffectAdapter


def _bot(path, session):
    return ClankerLM(
        session_id=session,
        db_path=path,
        affect=RuleAffectAdapter(),
    )


def test_entities_facts_and_running_state_survive_restart(tmp_path):
    db = tmp_path / "clanker-lm.db"
    bot = _bot(db, "alpha")
    bot.reply("My sister bought a blue car yesterday.")
    state_before = bot.memory.running_state.to_dict()
    bot.close()

    reopened = _bot(db, "alpha")
    assert reopened.memory.running_state.to_dict() == state_before
    result = reopened.process("What did she buy?")
    reopened.close()

    assert result.answer and result.answer.status == AnswerStatus.ANSWERED
    assert result.response == "She bought a blue car yesterday."


def test_sessions_are_isolated(tmp_path):
    db = tmp_path / "clanker-lm.db"
    alpha = _bot(db, "alpha")
    alpha.reply("My sister bought a blue car.")
    alpha.close()

    beta = _bot(db, "beta")
    result = beta.process("What did she buy?")
    beta.close()

    assert result.answer and result.answer.status == AnswerStatus.NEEDS_CONTEXT
    assert result.response == "Who do you mean by she?"


def test_reset_removes_persisted_semantic_state(tmp_path):
    db = tmp_path / "clanker-lm.db"
    bot = _bot(db, "alpha")
    bot.reply("My sister bought a blue car.")
    bot.reset()
    bot.close()

    reopened = _bot(db, "alpha")
    result = reopened.process("What did she buy?")
    reopened.close()

    assert result.answer and result.answer.status == AnswerStatus.NEEDS_CONTEXT
