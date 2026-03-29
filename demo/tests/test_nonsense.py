"""Tests for Layer -1 nonsense / gibberish detection (demo.nonsense)."""

import os
import sys

import pytest

# Ensure the demo package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from demo.nonsense import NonsenseDetector, NONSENSE, NON_EMOTIONAL, VALID


@pytest.fixture
def det():
    return NonsenseDetector()


# ── Keyboard smashes ─────────────────────────────────────────────────

class TestKeyboardSmash:
    def test_asdfghjkl(self, det):
        is_nonsense, reason = det.detect("asdfghjkl")
        assert is_nonsense, "asdfghjkl should be nonsense (keyboard smash)"

    def test_qwertyuiop(self, det):
        is_nonsense, reason = det.detect("qwertyuiop")
        assert is_nonsense, "qwertyuiop should be nonsense (keyboard row)"

    def test_zxcvbnm(self, det):
        is_nonsense, reason = det.detect("zxcvbnm")
        assert is_nonsense, "zxcvbnm should be nonsense (keyboard row)"

    def test_mixed_smash(self, det):
        is_nonsense, _ = det.detect("asjkdfhlg qweurytp")
        assert is_nonsense, "mixed keyboard smash should be nonsense"


# ── Repeated characters ──────────────────────────────────────────────

class TestRepeatedChars:
    def test_repeated_letter(self, det):
        is_nonsense, _ = det.detect("aaaaaaaaaa")
        assert is_nonsense, "aaaaaaaaaa should be nonsense"

    def test_repeated_punctuation(self, det):
        is_nonsense, _ = det.detect("!!!!!!")
        assert is_nonsense, "!!!!!! should be nonsense (no alphanumeric)"

    def test_repeated_question_marks(self, det):
        is_nonsense, _ = det.detect("????????")
        assert is_nonsense, "???????? should be nonsense"


# ── Single character spam ────────────────────────────────────────────

class TestSingleCharSpam:
    def test_spaced_single_chars(self, det):
        is_nonsense, _ = det.detect("a a a a a a")
        assert is_nonsense, "'a a a a a a' should be nonsense (single char spam)"


# ── Empty / whitespace ───────────────────────────────────────────────

class TestEmpty:
    def test_empty_string(self, det):
        is_nonsense, reason = det.detect("")
        assert is_nonsense
        assert "empty" in reason.lower()

    def test_whitespace_only(self, det):
        is_nonsense, _ = det.detect("   \t  \n  ")
        assert is_nonsense

    def test_none_like(self, det):
        """Empty string after stripping."""
        is_nonsense, _ = det.detect("   ")
        assert is_nonsense


# ── URLs and code ────────────────────────────────────────────────────

class TestNonEmotional:
    def test_url_https(self, det):
        is_nonsense, reason = det.detect("https://example.com")
        assert is_nonsense
        assert "url" in reason.lower() or "non-emotional" in reason.lower()

    def test_url_localhost(self, det):
        is_nonsense, reason = det.detect("localhost:3000")
        assert is_nonsense
        assert "non-emotional" in reason.lower() or "url" in reason.lower()

    def test_code_fragment(self, det):
        is_nonsense, _ = det.detect("function foo() { return bar; }")
        assert is_nonsense


# ── Valid inputs (should NOT be flagged) ─────────────────────────────

class TestValidInputs:
    def test_feeling_sad(self, det):
        is_nonsense, _ = det.detect("I'm feeling sad")
        assert not is_nonsense, "'I'm feeling sad' should NOT be nonsense"

    def test_lol_bruh(self, det):
        is_nonsense, _ = det.detect("lol bruh")
        assert not is_nonsense, "'lol bruh' should NOT be nonsense (known slang)"

    def test_hahahahaha(self, det):
        """Repeated 'ha' deduplicates to 'haha' which is a known word."""
        is_nonsense, _ = det.detect("hahahahaha")
        assert not is_nonsense, "'hahahahaha' should NOT be nonsense"

    def test_the(self, det):
        is_nonsense, _ = det.detect("the")
        assert not is_nonsense, "'the' should NOT be nonsense"

    def test_short_real_sentence(self, det):
        is_nonsense, _ = det.detect("I hate everything about this")
        assert not is_nonsense

    def test_emotional_statement(self, det):
        is_nonsense, _ = det.detect("I feel so lost and alone")
        assert not is_nonsense

    def test_casual_slang(self, det):
        is_nonsense, _ = det.detect("ngl that was pretty wild tbh")
        assert not is_nonsense

    def test_single_ok(self, det):
        is_nonsense, _ = det.detect("ok")
        assert not is_nonsense

    def test_greeting(self, det):
        is_nonsense, _ = det.detect("hey")
        assert not is_nonsense


# ── Borderline cases ─────────────────────────────────────────────────

class TestBorderline:
    def test_xkcd_short_no_vowels(self, det):
        """4-char word with no vowels but short — should NOT trigger."""
        is_nonsense, _ = det.detect("xkcd")
        assert not is_nonsense, "'xkcd' is short and known — should not trigger"

    def test_no_vowels_long(self, det):
        """Long consonant-only string should be nonsense."""
        is_nonsense, _ = det.detect("bcdfghjk")
        assert is_nonsense

    def test_mixed_valid_and_gibberish(self, det):
        """Mostly real words with one oddball should pass."""
        is_nonsense, _ = det.detect("I feel really asdfg today")
        # 4 real words, 1 gibberish → avg should be below threshold
        assert not is_nonsense

    def test_all_gibberish_words(self, det):
        is_nonsense, _ = det.detect("qwerty asdfgh zxcvbn")
        assert is_nonsense
