"""Tests for Layer 0 intent detection (demo.intent)."""

import os
import sys

import pytest

# Ensure the demo package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from demo.intent import (
    IntentDetector,
    GREETING, QUESTION, REQUEST, VENTING, CASUAL, EMOTIONAL, STATEMENT,
    ALL_MODES, PROCESSING_PATHS,
)


@pytest.fixture
def det():
    return IntentDetector()


# ── Greeting ──────────────────────────────────────────────────────────

class TestGreeting:
    def test_single_word(self, det):
        for word in ("hey", "hi", "hello", "sup", "yo", "heya", "hiya", "howdy"):
            mode, conf = det.detect(word)
            assert mode == GREETING, f"{word!r} should be GREETING, got {mode}"
            assert conf >= 0.8

    def test_greeting_phrase(self, det):
        for phrase in ("good morning", "good evening", "good afternoon",
                       "what's up", "how's it going", "how are you"):
            mode, conf = det.detect(phrase)
            assert mode == GREETING, f"{phrase!r} should be GREETING, got {mode}"

    def test_greeting_with_name(self, det):
        mode, conf = det.detect("hey dude")
        assert mode == GREETING

    def test_long_sentence_not_greeting(self, det):
        mode, _ = det.detect("hey can you help me fix this server issue")
        assert mode != GREETING  # too many words for greeting


# ── Question ──────────────────────────────────────────────────────────

class TestQuestion:
    def test_question_mark(self, det):
        mode, conf = det.detect("Is this working?")
        assert mode == QUESTION
        assert conf >= 0.8

    def test_question_starter(self, det):
        for start in ("Why is the sky blue", "How does this work",
                       "What time is it", "Where are my keys",
                       "When does it start", "Who said that"):
            mode, _ = det.detect(start)
            assert mode == QUESTION, f"{start!r} should be QUESTION, got {mode}"

    def test_question_starter_with_qmark(self, det):
        mode, conf = det.detect("Can you see this?")
        assert mode == QUESTION
        assert conf >= 0.85

    def test_single_why(self, det):
        # "why" alone — question starter but only 1 word, needs >= 2 for starter-only
        mode, conf = det.detect("why?")
        assert mode == QUESTION

    def test_rhetorical_question_with_emotion_goes_emotional(self, det):
        """Crisis-adjacent content with '?' should still be EMOTIONAL."""
        mode, _ = det.detect("What's the point of living?")
        assert mode == EMOTIONAL


# ── Request ───────────────────────────────────────────────────────────

class TestRequest:
    def test_imperative(self, det):
        mode, _ = det.detect("fix this bug")
        assert mode == REQUEST

    def test_please_verb(self, det):
        mode, _ = det.detect("please help me")
        assert mode == REQUEST

    def test_help_me_verb(self, det):
        mode, conf = det.detect("help me understand this")
        assert mode == REQUEST
        assert conf >= 0.8

    def test_can_you_verb(self, det):
        mode, _ = det.detect("can you check the logs")
        assert mode == REQUEST

    def test_please_anywhere(self, det):
        mode, _ = det.detect("I need you to fix it please")
        assert mode == REQUEST


# ── Casual ────────────────────────────────────────────────────────────

class TestCasual:
    def test_short_casual(self, det):
        mode, _ = det.detect("lol")
        assert mode == CASUAL

    def test_multiple_markers(self, det):
        mode, conf = det.detect("lol bruh that's wild")
        assert mode == CASUAL
        assert conf >= 0.7

    def test_high_density(self, det):
        mode, _ = det.detect("omg lol wtf bruh ngl")
        assert mode == CASUAL

    def test_single_idk(self, det):
        mode, _ = det.detect("idk")
        assert mode == CASUAL


# ── Emotional ─────────────────────────────────────────────────────────

class TestEmotional:
    def test_i_feel(self, det):
        mode, _ = det.detect("I feel so lost and alone")
        assert mode == EMOTIONAL

    def test_emotional_markers(self, det):
        mode, _ = det.detect("I'm scared and anxious about everything")
        assert mode == EMOTIONAL

    def test_strong_emotion_short(self, det):
        mode, _ = det.detect("I'm devastated")
        assert mode == EMOTIONAL

    def test_emotional_overrides_question(self, det):
        """'Can you help me I'm so scared' — emotional content wins."""
        mode, _ = det.detect("Can you help me I'm so scared")
        assert mode == EMOTIONAL

    def test_crisis_question(self, det):
        """Crisis content must go EMOTIONAL even with question marks."""
        mode, conf = det.detect("What's the point of living?")
        assert mode == EMOTIONAL
        assert conf >= 0.9

    def test_want_to_die(self, det):
        mode, conf = det.detect("I want to die")
        assert mode == EMOTIONAL
        assert conf >= 0.9

    def test_kill_myself(self, det):
        mode, conf = det.detect("I want to kill myself")
        assert mode == EMOTIONAL
        assert conf >= 0.9


# ── Venting ───────────────────────────────────────────────────────────

class TestVenting:
    def test_venting_first_person(self, det):
        mode, _ = det.detect("Everything sucks and I hate my job")
        assert mode == VENTING

    def test_venting_needs_length(self, det):
        """Short angry statements should not be VENTING."""
        mode, _ = det.detect("ugh hate")
        assert mode != VENTING

    def test_venting_no_question(self, det):
        """With a question mark it should be QUESTION, not VENTING."""
        mode, _ = det.detect("Why does everything suck so much?")
        assert mode != VENTING

    def test_venting_high_negativity(self, det):
        mode, conf = det.detect("I am so frustrated and angry and tired of this stupid thing")
        assert mode in (VENTING, EMOTIONAL)
        assert conf >= 0.7


# ── Statement ─────────────────────────────────────────────────────────

class TestStatement:
    def test_neutral_fact(self, det):
        mode, conf = det.detect("The meeting is at 3pm")
        assert mode == STATEMENT
        assert conf == 0.5

    def test_neutral_longer(self, det):
        mode, _ = det.detect("Python 3.12 was released last October")
        assert mode == STATEMENT


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self, det):
        mode, conf = det.detect("")
        assert mode == STATEMENT
        assert conf < 0.5

    def test_whitespace_only(self, det):
        mode, _ = det.detect("   ")
        assert mode == STATEMENT

    def test_single_word_ugh(self, det):
        """'ugh' — short, no clear category. Should not crash."""
        mode, _ = det.detect("ugh")
        assert mode in ALL_MODES

    def test_mixed_case(self, det):
        mode, _ = det.detect("HEY What's Up")
        assert mode == GREETING

    def test_extra_whitespace(self, det):
        mode, _ = det.detect("  hello  there  ")
        assert mode == GREETING


# ── Confidence thresholds ─────────────────────────────────────────────

class TestConfidence:
    def test_greeting_high(self, det):
        _, conf = det.detect("hi")
        assert conf >= 0.8

    def test_statement_low(self, det):
        _, conf = det.detect("The train departs at noon")
        assert conf <= 0.6

    def test_crisis_very_high(self, det):
        _, conf = det.detect("I want to kill myself")
        assert conf >= 0.9

    def test_confidence_range(self, det):
        """All confidences must be in [0, 1]."""
        samples = [
            "hey", "why?", "fix this", "lol", "I feel sad",
            "everything is terrible and I hate it all",
            "The cat sat on the mat", "", "   ",
        ]
        for text in samples:
            _, conf = det.detect(text)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range for {text!r}"


# ── Processing paths ──────────────────────────────────────────────────

class TestProcessingPaths:
    def test_all_modes_have_paths(self):
        for mode in ALL_MODES:
            assert mode in PROCESSING_PATHS

    def test_emotional_full_pipeline(self, det):
        path = det.get_processing_path(EMOTIONAL)
        assert path["pendulum"] is True
        assert path["chunker"] is True
        assert path["grader"] is True
        assert path["response"] == "full_empathy"

    def test_greeting_skips_pipeline(self, det):
        path = det.get_processing_path(GREETING)
        assert path["pendulum"] is False
        assert path["chunker"] is False
        assert path["grader"] is False

    def test_returns_copy(self, det):
        """Mutating the returned dict should not affect the original."""
        path = det.get_processing_path(GREETING)
        path["pendulum"] = True
        assert PROCESSING_PATHS[GREETING]["pendulum"] is False

    def test_unknown_mode_falls_back(self, det):
        path = det.get_processing_path("NONEXISTENT")
        assert path == PROCESSING_PATHS[STATEMENT]
