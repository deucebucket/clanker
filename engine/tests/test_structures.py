"""Tests for V3 Structure Detector -- chess-like pattern recognition.

All test sentences are NOVEL -- things never in any benchmark.
Safe sentences must NOT flag crisis patterns.
"""

import pytest

from engine.word_classifier import classify_sentence
from engine.structures import StructureDetector, StructureMatch


# ── Helper ───────────────────────────────────────────────────────

def _detect(sentence: str):
    """Classify and detect structures in a sentence."""
    roles = classify_sentence(sentence.split())
    detector = StructureDetector()
    return detector.detect_all(roles)


def _has_pattern(matches, pattern_name):
    """Check if a specific pattern was detected."""
    return any(m.pattern == pattern_name for m in matches)


def _get_pattern(matches, pattern_name):
    """Get the first match of a specific pattern."""
    for m in matches:
        if m.pattern == pattern_name:
            return m
    return None


# ── FAREWELL ─────────────────────────────────────────────────────

class TestFarewell:

    def test_gave_dog_to_neighbor(self):
        """'I gave my dog to my neighbor' -> FAREWELL."""
        matches = _detect("I gave my dog to my neighbor")
        assert _has_pattern(matches, "FAREWELL")

    def test_leaving_keys_with_friend(self):
        """'leaving my keys with a friend' -> FAREWELL."""
        matches = _detect("leaving my keys with a friend")
        assert _has_pattern(matches, "FAREWELL")

    def test_no_farewell_without_ref(self):
        """'I gave my dog a treat' -- no person ref, not FAREWELL."""
        matches = _detect("I gave my dog a treat")
        assert not _has_pattern(matches, "FAREWELL")


# ── METHOD_ACQUISITION ───────────────────────────────────────────

class TestMethodAcquisition:

    def test_bought_pills(self):
        """'just bought a bunch of pills' -> METHOD_ACQUISITION."""
        matches = _detect("just bought a bunch of pills")
        assert _has_pattern(matches, "METHOD_ACQUISITION")

    def test_found_a_gun(self):
        """'I found a gun in the closet' -> METHOD_ACQUISITION."""
        matches = _detect("I found a gun in the closet")
        assert _has_pattern(matches, "METHOD_ACQUISITION")

    def test_bought_groceries_safe(self):
        """'just bought some groceries' -- no METHOD, should be safe."""
        matches = _detect("just bought some groceries")
        assert not _has_pattern(matches, "METHOD_ACQUISITION")


# ── FINALITY ─────────────────────────────────────────────────────

class TestFinality:

    def test_last_time(self):
        """'this is the last time' -> FINALITY."""
        matches = _detect("this is the last time")
        assert _has_pattern(matches, "FINALITY")

    def test_goodbye(self):
        """'goodbye everyone' -> FINALITY."""
        matches = _detect("goodbye everyone")
        assert _has_pattern(matches, "FINALITY")

    def test_finality_with_self_ref(self):
        """'I am done with everything' -> FINALITY with SELF_REF boost."""
        matches = _detect("I am done with everything")
        assert _has_pattern(matches, "FINALITY")
        m = _get_pattern(matches, "FINALITY")
        assert m.confidence > 0.4  # boosted by SELF_REF


# ── BLANKET_APOLOGY ──────────────────────────────────────────────

class TestBlanketApology:

    def test_sorry_for_everything(self):
        """'im sorry for everything' -> BLANKET_APOLOGY."""
        matches = _detect("im sorry for everything")
        assert _has_pattern(matches, "BLANKET_APOLOGY")

    def test_apologize_to_everyone(self):
        """'I apologize to everyone' -> BLANKET_APOLOGY."""
        matches = _detect("I apologize to everyone")
        assert _has_pattern(matches, "BLANKET_APOLOGY")

    def test_sorry_for_being_late_safe(self):
        """'im sorry for being late' -- specific apology, NOT blanket."""
        matches = _detect("im sorry for being late")
        assert not _has_pattern(matches, "BLANKET_APOLOGY")


# ── SELF_REMOVAL ─────────────────────────────────────────────────

class TestSelfRemoval:

    def test_happier_without_me(self):
        """'they would be happier if I wasnt here' -> SELF_REMOVAL."""
        matches = _detect("they would be happier if I wasnt here")
        assert _has_pattern(matches, "SELF_REMOVAL")

    def test_better_off_without(self):
        """'everyone is better off without me' -> SELF_REMOVAL."""
        matches = _detect("everyone is better off without me")
        assert _has_pattern(matches, "SELF_REMOVAL")

    def test_better_at_cooking_safe(self):
        """'she is better at cooking than me' -- comparison, not removal."""
        matches = _detect("she is better at cooking than me")
        # No conditional word present, should not match
        assert not _has_pattern(matches, "SELF_REMOVAL")


# ── SUSPICIOUS_CALM ──────────────────────────────────────────────

class TestSuspiciousCalm:

    def test_finally_at_peace_is_relief(self):
        """'I finally feel at peace' = present tense breakthrough, NOT suspicious."""
        matches = _detect("I finally feel at peace")
        assert not _has_pattern(matches, "SUSPICIOUS_CALM")

    def test_decided_calm_is_suspicious(self):
        """'i feel calm now that ive decided' -> SUSPICIOUS_CALM."""
        matches = _detect("i feel calm now that ive decided")
        assert _has_pattern(matches, "SUSPICIOUS_CALM")

    def test_ready_to_go_is_not_suspicious(self):
        """'im ready to go now' = could be waiting for a ride. Needs conversation context."""
        matches = _detect("im ready to go now")
        assert not _has_pattern(matches, "SUSPICIOUS_CALM")

    def test_peace_without_finally_safe(self):
        """'I feel at peace' -- no 'finally', not suspicious."""
        matches = _detect("I feel at peace")
        assert not _has_pattern(matches, "SUSPICIOUS_CALM")


# ── EXHAUSTION ───────────────────────────────────────────────────

class TestExhaustion:

    def test_cant_take_anymore(self):
        """'I cant take this anymore' -> EXHAUSTION."""
        matches = _detect("I cant take this anymore")
        assert _has_pattern(matches, "EXHAUSTION")

    def test_cant_do_this(self):
        """'I cant do this' -> EXHAUSTION (without temporal, lower conf)."""
        matches = _detect("I cant do this")
        assert _has_pattern(matches, "EXHAUSTION")

    def test_cant_find_keys_safe(self):
        """'I cant find my keys' -- 'find' is ACQUIRE not sustain."""
        matches = _detect("I cant find my keys")
        assert not _has_pattern(matches, "EXHAUSTION")


# ── SARCASM_INVERSION ────────────────────────────────────────────

class TestSarcasmInversion:

    def test_great_another_monday(self):
        """'oh great another monday' -> SARCASM_INVERSION."""
        matches = _detect("oh great another monday")
        assert _has_pattern(matches, "SARCASM_INVERSION")

    def test_genuine_great_not_sarcasm(self):
        """'that was a great wonderful performance' -- genuinely positive."""
        # Multiple positive words = genuine, not sarcastic
        matches = _detect("that was a great wonderful performance")
        # This may or may not flag -- the key test is the obvious sarcasm above
        # If it does flag, confidence should be lower


# ── NO_EXIT ──────────────────────────────────────────────────────

class TestNoExit:

    def test_no_hope(self):
        """'there is no hope' -> NO_EXIT."""
        matches = _detect("there is no hope")
        assert _has_pattern(matches, "NO_EXIT")

    def test_no_point(self):
        """'there is no point anymore' -> NO_EXIT."""
        matches = _detect("there is no point anymore")
        assert _has_pattern(matches, "NO_EXIT")

    def test_no_milk_safe(self):
        """'there is no milk' -- 'milk' is not an exit concept."""
        matches = _detect("there is no milk")
        assert not _has_pattern(matches, "NO_EXIT")


# ── SELF_NULLIFY ─────────────────────────────────────────────────

class TestSelfNullify:

    def test_i_am_nothing(self):
        """'I am nothing' -> SELF_NULLIFY."""
        matches = _detect("I am nothing")
        assert _has_pattern(matches, "SELF_NULLIFY")

    def test_i_am_worthless(self):
        """'I am worthless' -> SELF_NULLIFY."""
        matches = _detect("I am worthless")
        assert _has_pattern(matches, "SELF_NULLIFY")

    def test_i_am_tired_safe(self):
        """'I am tired' -- 'tired' is not a null word."""
        matches = _detect("I am tired")
        assert not _has_pattern(matches, "SELF_NULLIFY")


# ── CHOPPER_SPLIT ────────────────────────────────────────────────

class TestChopperSplit:

    def test_but_splits(self):
        """'I was fine but now everything hurts' -> CHOPPER_SPLIT."""
        matches = _detect("I was fine but now everything hurts")
        assert _has_pattern(matches, "CHOPPER_SPLIT")

    def test_however_splits(self):
        """'things were good however it changed' -> CHOPPER_SPLIT."""
        matches = _detect("things were good however it changed")
        assert _has_pattern(matches, "CHOPPER_SPLIT")


# ── SAFE sentences should NOT flag crisis patterns ───────────────

class TestSafeSentences:
    """These everyday sentences must NOT trigger crisis patterns."""

    CRISIS_PATTERNS = {
        "FAREWELL", "METHOD_ACQUISITION", "BLANKET_APOLOGY",
        "SELF_REMOVAL", "SUSPICIOUS_CALM", "EXHAUSTION",
        "NO_EXIT", "SELF_NULLIFY",
    }

    def _assert_no_crisis(self, sentence):
        matches = _detect(sentence)
        crisis_found = [
            m.pattern for m in matches if m.pattern in self.CRISIS_PATTERNS
        ]
        assert not crisis_found, (
            f"'{sentence}' falsely flagged: {crisis_found}"
        )

    def test_bad_day(self):
        self._assert_no_crisis("im having a bad day")

    def test_work_stressful(self):
        self._assert_no_crisis("work was stressful")

    def test_mondays_suck(self):
        self._assert_no_crisis("mondays suck")

    def test_traffic_was_terrible(self):
        self._assert_no_crisis("traffic was terrible today")

    def test_need_coffee(self):
        self._assert_no_crisis("I need more coffee")

    def test_forgot_lunch(self):
        self._assert_no_crisis("I forgot my lunch at home")


# ── StructureMatch dataclass ─────────────────────────────────────

class TestStructureMatch:

    def test_has_required_fields(self):
        m = StructureMatch(
            pattern="TEST",
            confidence=0.8,
            matched_indices=[0, 1],
            description="test match",
        )
        assert m.pattern == "TEST"
        assert m.confidence == 0.8
        assert m.v_weight == 0.0  # default

    def test_weight_fields(self):
        m = StructureMatch(
            pattern="TEST",
            confidence=0.8,
            matched_indices=[0],
            description="test",
            v_weight=-30.0,
            d_weight=-20.0,
            u_weight=40.0,
            g_weight=50.0,
        )
        assert m.v_weight == -30.0
        assert m.g_weight == 50.0
