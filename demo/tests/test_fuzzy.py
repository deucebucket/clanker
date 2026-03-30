"""Tests for the fuzzy word matcher (demo.fuzzy) — V2 version.

Tests against EMOTIONAL_VOCABULARY (2.3K curated words), not legacy WORD_FORCES.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from demo.fuzzy import fuzzy_match, clear_cache, TEXT_SPEAK
from demo.forces_curated import EMOTIONAL_VOCABULARY


@pytest.fixture(autouse=True)
def _clear():
    """Clear cache before each test."""
    clear_cache()


# ── Deduplication ───────────────────────────────────────────────────

class TestDeduplication:
    def test_happy_elongated(self):
        assert fuzzy_match("happyyyy") == "happy"

    def test_yes_elongated(self):
        assert fuzzy_match("yesssss") == "yes"

    def test_cool_elongated(self):
        assert fuzzy_match("coooool") == "cool"

    def test_never_elongated(self):
        assert fuzzy_match("neverrr") == "never"

    def test_sad_elongated(self):
        assert fuzzy_match("saaaad") == "sad"

    def test_exact_match_not_fuzzied(self):
        """Words already in vocabulary should not get fuzzied."""
        result = fuzzy_match("happy")
        assert result is None


# ── Text speak ──────────────────────────────────────────────────────

class TestTextSpeak:
    def test_luv_to_love(self):
        assert fuzzy_match("luv") == "love"

    def test_gr8_to_great(self):
        assert fuzzy_match("gr8") == "great"

    def test_thx_to_thanks(self):
        assert fuzzy_match("thx") == "thanks"

    def test_h8_to_hate(self):
        assert fuzzy_match("h8") == "hate"

    def test_pls_to_please(self):
        assert fuzzy_match("pls") == "please"

    def test_tbh_to_honestly(self):
        assert fuzzy_match("tbh") == "honestly"

    def test_smh_to_disappointed(self):
        assert fuzzy_match("smh") == "disappointed"

    def test_depresed_misspelling(self):
        assert fuzzy_match("depresed") == "depressed"

    def test_fustrated_misspelling(self):
        assert fuzzy_match("fustrated") == "frustrated"

    def test_text_speak_targets_in_vocab(self):
        """All text speak targets should map to words in EMOTIONAL_VOCABULARY."""
        for abbrev, target in TEXT_SPEAK.items():
            if target in EMOTIONAL_VOCABULARY:
                result = fuzzy_match(abbrev)
                assert result == target, f"{abbrev} should map to {target}"


# ── Cambridge effect ─────────────────────────────────────────────────

class TestCambridge:
    def test_sickening_scramble(self):
        assert fuzzy_match("scikening") == "sickening"

    def test_frustrated_scramble(self):
        assert fuzzy_match("frsutrated") == "frustrated"

    def test_short_word_no_match(self):
        """Short words (< 6 chars) should NOT use Cambridge matching."""
        assert fuzzy_match("hpapy") is None  # 5 chars, below threshold


# ── Cache ───────────────────────────────────────────────────────────

class TestCache:
    def test_cache_returns_same_result(self):
        first = fuzzy_match("happyyyy")
        second = fuzzy_match("happyyyy")
        assert first == second == "happy"

    def test_cache_returns_none_consistently(self):
        first = fuzzy_match("xyzzy")
        second = fuzzy_match("xyzzy")
        assert first is None and second is None


# ── Performance ─────────────────────────────────────────────────────

class TestPerformance:
    def test_1000_lookups_fast(self):
        """1000 fuzzy lookups should complete in < 1 second."""
        words = ["happyyyy", "tbh", "xyzzy", "luv", "saaaad"] * 200
        start = time.time()
        for w in words:
            fuzzy_match(w)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"1000 lookups took {elapsed:.2f}s"
