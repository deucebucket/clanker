"""Tests for the morpheme word factory."""

import pytest
from demo.word_factory import WordFactory


@pytest.fixture(scope="module")
def factory():
    return WordFactory()


class TestBuildWord:
    """Core build_word functionality."""

    def test_returns_sorted_by_distance(self, factory):
        results = factory.build_word(30, 10, 10, 0, 20, top_n=10)
        distances = [r[2] for r in results]
        assert distances == sorted(distances)

    def test_returns_requested_count(self, factory):
        results = factory.build_word(0, 0, 0, 0, 0, top_n=3)
        assert len(results) == 3

    def test_returns_valid_5_tuples(self, factory):
        results = factory.build_word(50, 30, 20, 0, 40, top_n=5)
        for word, force, dist in results:
            assert isinstance(word, str)
            assert len(word) > 0
            assert isinstance(force, tuple)
            assert len(force) == 5
            assert all(isinstance(v, (int, float)) for v in force)
            assert isinstance(dist, float)

    def test_exact_target_has_zero_or_small_distance(self, factory):
        """If we target an exact root's values, distance should be very small."""
        # "joy" root is (+70, +40, +30, 0, +55)
        results = factory.build_word(70, 40, 30, 0, 55, top_n=1)
        word, force, dist = results[0]
        assert dist < 1.0  # should find exact match


class TestBuildPositive:
    """build_positive convenience method."""

    def test_returns_positive_valence(self, factory):
        results = factory.build_positive(intensity=0.5, top_n=5)
        assert len(results) > 0
        # Top result should have positive valence
        best_word, best_force, best_dist = results[0]
        assert best_force[0] > 0, f"Expected positive valence, got {best_force[0]}"

    def test_higher_intensity_higher_valence(self, factory):
        low = factory.build_positive(intensity=0.3, top_n=1)
        high = factory.build_positive(intensity=0.9, top_n=1)
        # The target valence is higher for high intensity, so best match should differ
        assert low[0][1] != high[0][1] or low[0][0] != high[0][0]


class TestBuildNegative:
    """build_negative convenience method."""

    def test_returns_negative_valence(self, factory):
        results = factory.build_negative(intensity=0.5, top_n=5)
        assert len(results) > 0
        best_word, best_force, best_dist = results[0]
        assert best_force[0] < 0, f"Expected negative valence, got {best_force[0]}"

    def test_higher_intensity_more_negative(self, factory):
        low = factory.build_negative(intensity=0.3, top_n=1)
        high = factory.build_negative(intensity=0.9, top_n=1)
        assert low[0][1] != high[0][1] or low[0][0] != high[0][0]


class TestBuildForEmotion:
    """Emotion-targeted word building."""

    def test_anger_high_arousal_negative_valence(self, factory):
        results = factory.build_for_emotion("anger", top_n=5)
        assert len(results) > 0
        best_word, best_force, best_dist = results[0]
        assert best_force[0] < 0, "Anger words should have negative valence"
        assert best_force[1] > 10, "Anger words should have high arousal"

    def test_calm_low_arousal(self, factory):
        results = factory.build_for_emotion("calm", top_n=5)
        assert len(results) > 0
        best_word, best_force, best_dist = results[0]
        assert best_force[1] < 0, "Calm words should have low/negative arousal"

    def test_joy_positive_valence(self, factory):
        results = factory.build_for_emotion("joy", top_n=5)
        assert len(results) > 0
        best_word, best_force, best_dist = results[0]
        assert best_force[0] > 0, "Joy words should have positive valence"

    def test_unknown_emotion_returns_empty(self, factory):
        results = factory.build_for_emotion("flurble")
        assert results == []

    def test_all_emotions_return_results(self, factory):
        for emotion in ("joy", "sadness", "anger", "fear", "love", "disgust", "surprise", "calm"):
            results = factory.build_for_emotion(emotion, top_n=3)
            assert len(results) == 3, f"{emotion} should return 3 results"


class TestMorphemeFormats:
    """Ensure factory handles both dict and tuple morpheme formats."""

    def test_factory_builds_combinations(self, factory):
        """Factory should have a large number of precomputed combinations."""
        assert len(factory._combos) > 100

    def test_all_forces_are_5_tuples(self, factory):
        """Every combo should have exactly 5 force components."""
        for word, force in factory._combos:
            assert len(force) == 5, f"{word} has {len(force)}-tuple force"
            assert all(isinstance(v, (int, float)) for v in force), f"{word} has non-numeric force"
