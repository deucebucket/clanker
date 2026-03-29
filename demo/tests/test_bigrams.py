"""Tests for bigram pair scoring."""

import pytest
from demo.bigrams import BigramDetector


@pytest.fixture
def detector():
    return BigramDetector()


class TestBigramDetector:
    """Test bigram pair detection."""

    def test_every_time_detected(self, detector):
        """('every', 'time') detected in 'every time I go'."""
        words = ["every", "time", "i", "go"]
        result = detector.detect(words, 0)
        assert result is not None
        assert result[-1] == "temporal: recurring pattern"

    def test_fall_love_with_bridge_word(self, detector):
        """('fall', 'love') detected in 'I fell in love' — words within 3 positions."""
        words = ["i", "fall", "in", "love"]
        result = detector.detect(words, 1)
        assert result is not None
        assert result[-1] == "romance: falling in love"

    def test_give_up_returns_surrender(self, detector):
        """('give', 'up') returns surrender force."""
        words = ["i", "give", "up"]
        result = detector.detect(words, 1)
        assert result is not None
        assert result == (-35, -10, -35, 5, -30, "surrender: quitting")

    def test_life_death_returns_existential(self, detector):
        """('life', 'death') returns existential force."""
        words = ["life", "and", "death"]
        result = detector.detect(words, 0)
        assert result is not None
        assert result[-1] == "existential: life and death"

    def test_calm_down_returns_deescalation(self, detector):
        """('calm', 'down') returns deescalation."""
        words = ["please", "calm", "down"]
        result = detector.detect(words, 1)
        assert result is not None
        assert result[-1] == "deescalation: calming"

    def test_no_pair_returns_none(self, detector):
        """Words not in any pair return None."""
        words = ["the", "cat", "sat"]
        result = detector.detect(words, 1)
        assert result is None

    def test_bridge_word_detection(self, detector):
        """Pair detection works with bridge words between: 'fall in love' (1 word gap)."""
        words = ["fall", "in", "love"]
        result = detector.detect(words, 0)
        assert result is not None
        assert result[-1] == "romance: falling in love"

    def test_beyond_three_positions_fails(self, detector):
        """Pair detection fails beyond 3 positions."""
        words = ["fall", "x", "y", "z", "love"]
        result = detector.detect(words, 0)
        assert result is None
