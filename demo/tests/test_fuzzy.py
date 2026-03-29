"""Tests for the fuzzy word matcher (demo.fuzzy)."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from demo.fuzzy import fuzzy_match, clear_cache, TEXT_SPEAK
from demo.forces import WORD_FORCES


@pytest.fixture(autouse=True)
def _clear():
    """Clear cache before each test."""
    clear_cache()


# ── Deduplication ───────────────────────────────────────────────────

class TestDeduplication:
    def test_happy_elongated(self):
        assert fuzzy_match("happyyyy") == "happy"

    def test_no_elongated(self):
        assert fuzzy_match("nooooo") == "no"

    def test_yes_elongated(self):
        assert fuzzy_match("yesssss") == "yes"

    def test_cool_elongated(self):
        assert fuzzy_match("coooool") == "cool"

    def test_never_elongated(self):
        assert fuzzy_match("neverrr") == "never"

    def test_exact_match_not_fuzzied(self):
        """Words already in WORD_FORCES should not get edit-distance matched elsewhere."""
        # "happy" is in WORD_FORCES — edit distance should not fire
        result = fuzzy_match("happy")
        # Should be None (no fuzzy needed) since dedup won't trigger (no 3+ runs)
        # and text speak doesn't have it, and edit distance skips exact matches
        assert result is None


# ── Text speak ──────────────────────────────────────────────────────

class TestTextSpeak:
    def test_u_to_you(self):
        assert fuzzy_match("u") == "you"

    def test_luv_to_love(self):
        assert fuzzy_match("luv") == "love"

    def test_gr8_to_great(self):
        assert fuzzy_match("gr8") == "great"

    def test_thx_to_thanks(self):
        assert fuzzy_match("thx") == "thanks"

    def test_h8_to_hate(self):
        assert fuzzy_match("h8") == "hate"

    def test_4ever_to_forever(self):
        assert fuzzy_match("4ever") == "forever"

    def test_pls_to_please(self):
        assert fuzzy_match("pls") == "please"

    def test_tbh_to_honestly(self):
        assert fuzzy_match("tbh") == "honestly"


# ── Edit distance ───────────────────────────────────────────────────

class TestEditDistance:
    def test_terrified_typo(self):
        # "terriifed" is edit distance ~2 from "terrified" due to transposition
        # Use a simpler typo
        assert fuzzy_match("terriied") == "terrified"

    def test_beautiful_typo(self):
        assert fuzzy_match("beautful") == "beautiful"

    def test_overwhelmed_typo(self):
        # "overwhelmd" is edit distance 1 from "overwhelmed"
        assert fuzzy_match("overwhelmd") == "overwhelmed"

    def test_no_false_positive_short_word(self):
        """Short words (< 5 chars) should NOT use edit distance."""
        # "cat" is 3 chars, should not fuzzy match to anything
        assert fuzzy_match("cat") is None

    def test_no_false_positive_bat(self):
        """'bat' should not match 'bad' — too short for edit distance."""
        assert fuzzy_match("bat") is None


# ── Cache ───────────────────────────────────────────────────────────

class TestCache:
    def test_cache_returns_same_result(self):
        first = fuzzy_match("happyyyy")
        second = fuzzy_match("happyyyy")
        assert first == second == "happy"

    def test_cache_returns_none_consistently(self):
        first = fuzzy_match("xyzzy")
        second = fuzzy_match("xyzzy")
        assert first is None
        assert second is None


# ── Performance ─────────────────────────────────────────────────────

class TestPerformance:
    def test_1000_lookups_fast(self):
        """1000 cached lookups should complete well under 10ms."""
        # Prime the cache with a mix of hits and misses
        test_words = (
            ["happyyyy", "u", "luv", "gr8", "beautful", "xyzzy", "cat",
             "nooooo", "thx", "pls"] * 100
        )
        # First pass primes cache
        for w in test_words:
            fuzzy_match(w)

        # Timed pass — all cached
        start = time.perf_counter()
        for w in test_words:
            fuzzy_match(w)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, f"1000 cached lookups took {elapsed_ms:.2f}ms (> 10ms)"

    def test_cold_lookups_reasonable(self):
        """Cold lookups (uncached, after index built) should be reasonable."""
        # Prime the index with one lookup, then clear cache
        fuzzy_match("warmup12345")
        clear_cache()
        words = [f"test{i}word" for i in range(100)] + ["happyyyy"] * 100
        start = time.perf_counter()
        for w in words:
            fuzzy_match(w)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # With index already built, 200 lookups should be under 2s
        assert elapsed_ms < 2000, f"200 cold lookups took {elapsed_ms:.2f}ms"
