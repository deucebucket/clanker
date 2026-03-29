"""Tests for BookendParser -- relational geometry from first + last word."""

import pytest

from demo.bookend import BookendParser


@pytest.fixture
def parser():
    return BookendParser()


def test_question(parser):
    """'Can you help me' -> first=can(question), last=me(self), pattern=question."""
    result = parser.parse("Can you help me")
    assert result["first_word"] == "can"
    assert result["first_role"] == "question"
    assert result["last_word"] == "me"
    assert result["last_role"] == "self"
    assert result["pattern"] == "question"


def test_declaration(parser):
    """'I love you' -> self->target, pattern=declaration."""
    result = parser.parse("I love you")
    assert result["first_word"] == "i"
    assert result["first_role"] == "self"
    assert result["last_word"] == "you"
    assert result["last_role"] == "target"
    assert result["direction"] == "self\u2192target"
    assert result["pattern"] == "declaration"


def test_isolation(parser):
    """'Nobody cares about me' -> absence->self, pattern=isolation."""
    result = parser.parse("Nobody cares about me")
    assert result["first_word"] == "nobody"
    assert result["first_role"] == "absence"
    assert result["last_word"] == "me"
    assert result["last_role"] == "self"
    assert result["pattern"] == "isolation"


def test_tail_modifier_stripped(parser):
    """'Nobody cares about me lol' -> tail_modifier=lol, last=me NOT lol."""
    result = parser.parse("Nobody cares about me lol")
    assert result["tail_modifier"] == "lol"
    assert result["last_word"] == "me"
    assert result["last_role"] == "self"
    assert result["pattern"] == "isolation"


def test_self_loop(parser):
    """'I hate myself' -> self->self, pattern=self_loop (DANGER)."""
    result = parser.parse("I hate myself")
    assert result["first_role"] == "self"
    assert result["last_role"] == "self"
    assert result["direction"] == "self\u2192self"
    assert result["pattern"] == "self_loop"


def test_plea_please_not_stripped(parser):
    """'Help me please' -> please is NOT tail noise; last=please, pattern=plea."""
    result = parser.parse("Help me please")
    assert result["first_word"] == "help"
    assert result["first_role"] == "action"
    assert result["last_word"] == "please"
    assert result["last_role"] == "action"
    assert result["pattern"] == "plea"


def test_accusation(parser):
    """'You always do this to me' -> target->self, pattern=accusation."""
    result = parser.parse("You always do this to me")
    assert result["first_word"] == "you"
    assert result["first_role"] == "target"
    assert result["last_word"] == "me"
    assert result["last_role"] == "self"
    assert result["pattern"] == "accusation"


def test_overwhelm(parser):
    """'Everything is my fault' -> totality + last=fault(None), pattern=overwhelm."""
    result = parser.parse("Everything is my fault")
    assert result["first_word"] == "everything"
    assert result["first_role"] == "totality"
    assert result["last_word"] == "fault"
    assert result["last_role"] is None
    assert result["pattern"] == "overwhelm"


def test_empty_string(parser):
    """Empty string -> all None."""
    result = parser.parse("")
    assert result["first_word"] == ""
    assert result["last_word"] == ""
    assert result["tail_modifier"] is None
    assert result["first_role"] is None
    assert result["last_role"] is None
    assert result["direction"] is None
    assert result["pattern"] is None
