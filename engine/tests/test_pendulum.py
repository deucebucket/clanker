"""Tests for V3 Fixed Physics Layer — pendulum.py.

Verifies that structural analysis drives VADUG computation correctly.
Tests cover: neutral baseline, positive/negative shifts, negation,
amplification, structure detection passthrough, and safe sentences.
"""

import pytest

from engine.pendulum import compute_vadug, CENTER
from engine.shared import VADUG


# ── Helper ───────────────────────────────────────────────────────

def _vadug(text: str) -> VADUG:
    """Get VADUGWI for text, discard trace."""
    result, _ = compute_vadug(text)
    return result


def _trace(text: str) -> dict:
    """Get trace dict for text."""
    _, trace = compute_vadug(text)
    return trace


# ── Neutral input ───────────────────────────────────────────────

class TestNeutral:

    def test_neutral_v_near_center(self):
        """Neutral sentence should have V near 128."""
        r = _vadug("the meeting is at three")
        assert 118 <= r.v <= 138, f"V={r.v}, expected near 128"

    def test_neutral_all_dimensions_near_center(self):
        """Neutral sentence should keep all dimensions near baseline."""
        r = _vadug("the meeting is at three")
        assert 118 <= r.v <= 138
        assert 118 <= r.a <= 138
        assert 118 <= r.d <= 138
        assert r.u <= 20  # urgency stays low
        assert 118 <= r.g <= 138


# ── Positive input ──────────────────────────────────────────────

class TestPositive:

    def test_positive_v_above_140(self):
        """Positive sentence should push V above 140."""
        r = _vadug("I am very happy")
        assert r.v > 140, f"V={r.v}, expected > 140"

    def test_positive_g_above_center(self):
        """Happy should feel lighter -- G above center."""
        r = _vadug("I am very happy")
        assert r.g > 128, f"G={r.g}, expected > 128"


# ── Negative input ──────────────────────────────────────────────

class TestNegative:

    def test_negative_v_below_115(self):
        """Negative sentence should push V below 115."""
        r = _vadug("I am very sad")
        assert r.v < 115, f"V={r.v}, expected < 115"

    def test_negative_g_below_center(self):
        """Sad should feel heavier -- G below center."""
        r = _vadug("I am very sad")
        assert r.g < 128, f"G={r.g}, expected < 128"


# ── Negation ────────────────────────────────────────────────────

class TestNegation:

    def test_not_happy_lower_than_happy(self):
        """'not happy' should have lower V than 'happy'."""
        v_happy = _vadug("happy").v
        v_not_happy = _vadug("not happy").v
        assert v_not_happy < v_happy, (
            f"not happy V={v_not_happy} should be < happy V={v_happy}"
        )

    def test_negation_flips_direction(self):
        """Negation should flip V direction on moderate words."""
        # Use moderate word to avoid ceiling effects
        v_good = _vadug("good").v
        v_not_good = _vadug("not good").v
        # "not good" should be below center, "good" above
        assert v_good > 128
        assert v_not_good < 128


# ── Amplification ──────────────────────────────────────────────

class TestAmplification:

    def test_amplifier_increases_displacement(self):
        """Amplifier should increase displacement from center on moderate words."""
        # Use a moderate word — "good" (dV=50) saturates before amplification helps
        v_glad = _vadug("glad").v
        v_very_glad = _vadug("very glad").v
        dist_glad = abs(v_glad - 128)
        dist_very = abs(v_very_glad - 128)
        assert dist_very > dist_glad, (
            f"very glad dist={dist_very} should be > glad dist={dist_glad}"
        )


# ── Structure detection passthrough ─────────────────────────────

class TestStructures:

    def test_farewell_detected(self):
        """'I gave my dog to my neighbor' should detect DIVESTITURE (giving away possessions)."""
        trace = _trace("I gave my dog to my neighbor")
        patterns = [s.pattern for s in trace["structures"]]
        assert "DIVESTITURE" in patterns or "FAREWELL" in patterns, f"Expected DIVESTITURE or FAREWELL, got {patterns}"

    def test_method_acquisition_detected(self):
        """'just bought a bunch of pills' should detect METHOD_ACQUISITION."""
        trace = _trace("just bought a bunch of pills")
        patterns = [s.pattern for s in trace["structures"]]
        assert "METHOD_ACQUISITION" in patterns, (
            f"Expected METHOD_ACQUISITION, got {patterns}"
        )

    def test_safe_no_crisis_structures(self):
        """'I had a bad day at work' should have no crisis structures."""
        trace = _trace("I had a bad day at work")
        crisis = {"FAREWELL", "METHOD_ACQUISITION", "BLANKET_APOLOGY",
                  "SELF_REMOVAL", "SUSPICIOUS_CALM", "EXHAUSTION",
                  "NO_EXIT", "SELF_NULLIFY"}
        found = [s.pattern for s in trace["structures"] if s.pattern in crisis]
        assert not found, f"Safe sentence falsely flagged: {found}"


# ── VADUGWI object ──────────────────────────────────────────────

class TestVADUGObject:

    def test_returns_vadug_type(self):
        """compute_vadug should return a VADUGWI object."""
        result, _ = compute_vadug("hello world")
        assert isinstance(result, VADUG)

    def test_all_five_dimensions(self):
        """VADUGWI should have all 7 dimensions."""
        r = _vadug("I am very happy")
        assert hasattr(r, "v")
        assert hasattr(r, "a")
        assert hasattr(r, "d")
        assert hasattr(r, "u")
        assert hasattr(r, "g")

    def test_dimensions_in_range(self):
        """All dimensions should be 0-255."""
        r = _vadug("I am extremely terribly devastatingly sad")
        assert 0 <= r.v <= 255
        assert 0 <= r.a <= 255
        assert 0 <= r.d <= 255
        assert 0 <= r.u <= 255
        assert 0 <= r.g <= 255


# ── Trace dict ──────────────────────────────────────────────────

class TestTrace:

    def test_trace_has_per_word_entries(self):
        """Trace should have one entry per word."""
        trace = _trace("I am happy")
        assert len(trace["trace"]) == 3

    def test_trace_entry_fields(self):
        """Each trace entry should have word, role, coeff, v, a, d, u, g."""
        trace = _trace("happy")
        entry = trace["trace"][0]
        for key in ("word", "role", "coeff", "v", "a", "d", "u", "g"):
            assert key in entry, f"Missing key: {key}"

    def test_word_count(self):
        """Trace should report correct word count."""
        trace = _trace("the quick brown fox")
        assert trace["word_count"] == 4

    def test_empty_input(self):
        """Empty string should return neutral VADUGWI and empty trace."""
        r, trace = compute_vadug("")
        assert r.v == 128
        assert r.w == 128
        assert trace["word_count"] == 0
        assert trace["trace"] == []


# ── V4: Self-Worth (W) dimension ─────────────────────────────────

class TestSelfWorth:
    """W tracks the user's self-assessment of their own value."""

    def test_default_w_is_neutral(self):
        """Default VADUGWI has W=128 (stable self-worth)."""
        assert VADUG().w == 128

    def test_neutral_sentence_w_unchanged(self):
        """Non-self-referential sentences should not move W."""
        r = _vadug("the meeting is at three")
        assert r.w == 128

    def test_no_self_ref_w_unchanged(self):
        """Sentences without SELF_REF should keep W at 128."""
        r = _vadug("she walked her dog")
        assert r.w == 128

    def test_self_nullify_drops_w(self):
        """'i am worthless' should produce W < 100."""
        r = _vadug("i am worthless")
        assert r.w < 100, f"W={r.w} should be < 100 for self-nullification"

    def test_self_nothing_drops_w(self):
        """'i am nothing' should produce W < 110."""
        r = _vadug("i am nothing")
        assert r.w < 110, f"W={r.w} should be < 110"

    def test_self_proud_raises_w(self):
        """'i am proud of myself' should produce W > 128."""
        r = _vadug("i am proud of myself")
        assert r.w > 128, f"W={r.w} should be > 128 for self-pride"

    def test_self_deserve_raises_w(self):
        """'i deserve to be happy' should produce W > 128."""
        r = _vadug("i deserve to be happy")
        assert r.w > 128, f"W={r.w} should be > 128"

    def test_told_useless_drops_w(self):
        """'he told me i was useless' should drop W below neutral."""
        r = _vadug("he told me i was useless")
        assert r.w < 128, f"W={r.w} should be < 128"

    def test_w_amplifies_negative_v(self):
        """Low W should amplify negative V (self-worth lens effect)."""
        # "i am worthless" has low W AND low V
        # The V should be lower than it would be without W coefficient
        r = _vadug("i am worthless")
        assert r.v < 80, f"V={r.v} should be deeply negative with W amplification"

    def test_burden_drops_w(self):
        """'im a burden' triggers SELF_NULLIFY which drops W."""
        r = _vadug("im a burden to everyone")
        assert r.w < 110, f"W={r.w} should be < 110 for self-as-burden"
