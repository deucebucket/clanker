"""Finite-verb and independently coordinated assertion conformance."""

from __future__ import annotations

from itertools import product

import pytest

from clanker_lm import lexicon
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import SpeechAct
from clanker_lm.parser import SemanticParser


FINITE_CASES = (
    ("My dad died yesterday.", "die"),
    ("Sarah tried the task.", "try"),
    ("Sarah cried yesterday.", "cry"),
    ("Sarah studied the book.", "study"),
    ("Sarah planned the trip.", "plan"),
    ("Sarah stopped the car.", "stop"),
    ("Sarah went to the store.", "go"),
    ("Sarah came home.", "come"),
    ("Sarah left the store.", "leave"),
    ("Sarah saw the car.", "see"),
    ("Sarah heard the noise.", "hear"),
    ("Sarah felt the fabric.", "feel"),
    ("Sarah thought about the plan.", "think"),
    ("Sarah wrote the note.", "write"),
    ("Sarah read the note yesterday.", "read"),
    ("Sarah took the key.", "take"),
    ("Sarah gave John the key.", "give"),
    ("Sarah made the table.", "make"),
    ("The window broke yesterday.", "break"),
    ("Sarah brought the package.", "bring"),
    ("Sarah caught the ball.", "catch"),
    ("Sarah found the key.", "find"),
    ("Sarah told John the story.", "tell"),
    ("Sarah sold the car.", "sell"),
    ("Sarah sent John the letter.", "send"),
    ("Sarah built the table.", "build"),
    ("Sarah ran yesterday.", "run"),
    ("Sarah recorded the meeting.", "record"),
    ("Sarah scheduled the appointment.", "schedule"),
)

LEFT_CLAUSES = (
    ("Sarah opened the door", "open"),
    ("John bought the car", "buy"),
    ("My sister called Mary", "call"),
    ("The technician built the table", "build"),
)
RIGHT_CLAUSES = (
    ("Mary closed the window", "close"),
    ("John sold the truck", "sell"),
    ("Sarah wrote the note", "write"),
    ("The driver moved the chair", "move"),
    ("My brother sent the letter", "send"),
)
CONNECTORS = ("and", "but", "yet", "or", "so")
COORDINATE_CASES = tuple(product(CONNECTORS, LEFT_CLAUSES, RIGHT_CLAUSES))


def parse(text: str, memory: ConversationMemory | None = None):
    memory = memory or ConversationMemory()
    memory.begin_turn()
    return SemanticParser().parse(text, memory)


@pytest.mark.parametrize("text,predicate", FINITE_CASES)
def test_common_finite_forms_produce_past_event_frames(text: str, predicate: str) -> None:
    result = parse(text)
    assert result.speech_act == SpeechAct.ASSERT, result.diagnostics
    assert result.events, result.diagnostics
    assert result.events[0].predicate == predicate
    assert result.events[0].tense == "past"


def test_died_retains_silent_e_lemma_and_unaccusative_patient() -> None:
    result = parse("My dad died yesterday.")
    event = result.events[0]
    assert event.predicate == "die"
    assert "patient" in event.arguments
    assert event.arguments["patient"].surface.lower() == "my dad"


@pytest.mark.parametrize("connector,left,right", COORDINATE_CASES)
def test_independently_finite_coordinate_matrix(
    connector: str,
    left: tuple[str, str],
    right: tuple[str, str],
) -> None:
    left_text, left_predicate = left
    right_text, right_predicate = right
    result = parse(f"{left_text} {connector} {right_text}.")
    assert [event.predicate for event in result.events] == [
        left_predicate,
        right_predicate,
    ], result.diagnostics
    assert [event.discourse_role for event in result.events] == [
        "main",
        "coordinate",
    ]
    assert f"coordinate connector={connector}" in result.diagnostics


def test_semicolon_reaches_assertion_splitter() -> None:
    result = parse("Sarah opened the door; John closed the window.")
    assert [event.predicate for event in result.events] == ["open", "close"]
    assert [event.discourse_role for event in result.events] == ["main", "coordinate"]
    assert "coordinate connector=;" in result.diagnostics


@pytest.mark.parametrize(
    "text",
    (
        "Sarah and Mary opened the door.",
        "Sarah bought bread and milk.",
        "Sarah opened the door and closed the window.",
        "The black and white dog ran yesterday.",
    ),
)
def test_non_independent_conjunctions_remain_one_clause(text: str) -> None:
    result = parse(text)
    assert len(result.events) == 1, result.diagnostics
    assert result.events[0].discourse_role == "main"


def test_unknown_suffix_is_not_a_bare_verb_guess() -> None:
    result = parse("Sarah florbed the device yesterday.")
    assert not result.events
    assert result.speech_act == SpeechAct.UNKNOWN


def test_auxiliary_can_license_an_unknown_predicate() -> None:
    result = parse("Sarah did florb the device.")
    assert result.events
    assert result.events[0].predicate == "florb"
    assert result.events[0].tense == "past"


@pytest.mark.parametrize(
    "surface,expected",
    (
        ("died", "die"),
        ("tied", "tie"),
        ("lied", "lie"),
        ("vied", "vie"),
        ("tried", "try"),
        ("cried", "cry"),
        ("studied", "study"),
        ("planned", "plan"),
        ("stopped", "stop"),
        ("records", "record"),
        ("schedules", "schedule"),
    ),
)
def test_deterministic_morphology(surface: str, expected: str) -> None:
    assert lexicon.lemma(surface) == expected


def test_yoda_rewrite_preserves_supported_buy_signature() -> None:
    memory = ConversationMemory()
    normal = parse("Sarah bought a Honda.", memory)
    yoda = parse("Bought a Honda, Sarah did.", memory)
    assert normal.events and yoda.events
    assert normal.events[0].predicate == yoda.events[0].predicate == "buy"
    assert set(normal.events[0].arguments) == set(yoda.events[0].arguments)


def test_compound_object_does_not_borrow_later_clause_verb() -> None:
    result = parse("Sarah bought bread and milk, but John left.")
    assert [event.predicate for event in result.events] == ["buy", "leave"]
    assert [event.discourse_role for event in result.events] == ["main", "coordinate"]
    assert "coordinate connector=but" in result.diagnostics
    assert "coordinate connector=and" not in result.diagnostics


def test_chained_coordinate_connectors_preserve_local_boundaries() -> None:
    result = parse(
        "Sarah opened the door, but John closed the window, so Mary left."
    )
    assert [event.predicate for event in result.events] == ["open", "close", "leave"]
    assert [event.discourse_role for event in result.events] == [
        "main",
        "coordinate",
        "coordinate",
    ]
    assert "coordinate connector=but" in result.diagnostics
    assert "coordinate connector=so" in result.diagnostics


def test_right_hand_compound_object_remains_inside_coordinate_clause() -> None:
    result = parse("Sarah called Mary, but John bought bread and milk.")
    assert [event.predicate for event in result.events] == ["call", "buy"]
    assert "coordinate connector=but" in result.diagnostics
    assert "coordinate connector=and" not in result.diagnostics
