"""Boundary regressions found by the automated review of #47."""

from __future__ import annotations

from clanker_lm.affect import HeuristicAffectBackend
from clanker_lm.model import AnswerStatus, EventFrame, ParseResult, SpeechAct
from clanker_lm.response_policy import ResponseActPlanner


def _plan(text: str):
    parse = ParseResult(
        speech_act=SpeechAct.ASSERT,
        raw_text=text,
        events=[EventFrame(predicate="be")],
    )
    return ResponseActPlanner().plan(
        text,
        parse,
        HeuristicAffectBackend().analyze(text),
        answer_status=AnswerStatus.ACKNOWLEDGED,
        initial_severity="low",
        initial_register="neutral",
    )


def test_knot_safe_does_not_match_not_safe_substring() -> None:
    decision = _plan("This is a knot safe for climbing.")
    assert decision.response_act == "positive_acknowledge"


def test_explicit_not_safe_remains_serious() -> None:
    decision = _plan("I am not safe.")
    assert decision.response_act == "serious_followup"
