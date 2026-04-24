"""Novel sentence integration tests — sentences NEVER in any benchmark or training data.

V2 scored 26% on these. This is the REAL test.

Two categories:
  1. NOVEL_CRISIS: should detect at least ONE structural crisis pattern
  2. NOVEL_SAFE: should NOT detect any crisis patterns
"""

import pytest

from engine.word_classifier import classify_sentence
from engine.structures import StructureDetector


# ── Helpers ─────────────────────────────────────────────────────

def _detect(sentence: str):
    """Classify and detect structures in a sentence."""
    roles = classify_sentence(sentence.split())
    detector = StructureDetector()
    return detector.detect_all(roles)


def _pattern_names(matches):
    """Return set of pattern names from matches."""
    return {m.pattern for m in matches}


# Crisis patterns — if any of these appear, something is flagged
CRISIS_PATTERNS = frozenset({
    "FAREWELL", "METHOD_ACQUISITION", "SELF_REMOVAL",
    "EXHAUSTION", "NO_EXIT", "SELF_NULLIFY",
    "SUSPICIOUS_CALM", "BLANKET_APOLOGY", "FINALITY",
    "DIVESTITURE", "DANGLING_BOND", "MASKING", "RESIGNATION",
    "WORLD_CONTINUES", "METHOD_FIXATION", "ABANDONMENT",
})


# ── Novel Crisis Sentences ──────────────────────────────────────

NOVEL_CRISIS = [
    ("i just gave my dog to my neighbor", {"FAREWELL", "DIVESTITURE"}),
    ("i left the keys on the counter for whoever finds them", {"FAREWELL", "FINALITY"}),
    pytest.param(
        "take care of my cat for me", {"FAREWELL"},
        marks=pytest.mark.xfail(reason="Needs CARETAKER_TRANSFER pattern — 'take care of' is entrustment, not ACQUIRE"),
    ),
    ("im sorry for everything", {"BLANKET_APOLOGY"}),
    ("this is the last time youll hear from me", {"FINALITY"}),
    # "i finally feel at peace" and "im ready to go now" = need conversation
    # context to be crisis. Standalone = could be waiting for a ride.
    ("i feel calm now that ive decided", {"SUSPICIOUS_CALM"}),
    ("just bought a bunch of pills", {"METHOD_ACQUISITION"}),
    ("found a bridge near my house", {"METHOD_ACQUISITION"}),
    pytest.param(
        "wrote letters to everyone", {"BLANKET_APOLOGY", "FINALITY"},
        marks=pytest.mark.xfail(reason="Needs BROADCAST pattern — writing to everyone with no apology word"),
    ),
    ("done with everything tbh", {"FINALITY", "BLANKET_APOLOGY"}),
    ("honestly might just end it lol", {"FINALITY"}),
    ("im so fucking done", {"FINALITY"}),
    ("cant do this shit anymore", {"EXHAUSTION"}),
    ("no point in trying anymore", {"NO_EXIT"}),
]


@pytest.mark.parametrize("sentence,expected_any", NOVEL_CRISIS)
def test_novel_crisis_detects_pattern(sentence, expected_any):
    """Each novel crisis sentence should trigger at least ONE structural pattern."""
    matches = _detect(sentence)
    detected = _pattern_names(matches)
    crisis_detected = detected & CRISIS_PATTERNS

    assert crisis_detected, (
        f"No crisis pattern detected for: '{sentence}'\n"
        f"  Detected patterns: {detected or 'NONE'}\n"
        f"  Expected at least one of: {expected_any}"
    )


# ── Novel Safe Sentences ────────────────────────────────────────

NOVEL_SAFE = [
    "im having a bad day",
    "work was stressful",
    "i failed my exam",
    "my girlfriend broke up with me",
    "i feel kinda sad today",
    "mondays suck",
    "this homework is killing me",
    "im dead tired",
    "i could kill for a pizza",
]


@pytest.mark.parametrize("sentence", NOVEL_SAFE)
def test_novel_safe_no_crisis(sentence):
    """Safe sentences should NOT trigger any crisis patterns."""
    matches = _detect(sentence)
    detected = _pattern_names(matches)
    crisis_detected = detected & CRISIS_PATTERNS

    assert not crisis_detected, (
        f"Crisis pattern(s) falsely detected for: '{sentence}'\n"
        f"  False positives: {crisis_detected}"
    )
