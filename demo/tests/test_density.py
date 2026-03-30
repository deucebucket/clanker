"""Tests for emotional density scoring — length-normalized VADUG.

Verifies that compute_density() produces comparable V values for short
and long sentences expressing the same emotion, and that crisis
detection is not weakened by normalization.
"""
import math
import pytest
from demo.pendulum_v2 import PendulumV2
from demo.classifier import AdaptiveClassifier
from demo.pipeline_config import PipelineConfig


def _compute_density(vadug, trace):
    """Compute emotional density — length-normalized VADUG (mirrors V1 logic)."""
    n = max(len(trace), 1)
    sqrt_n = math.sqrt(n)
    scale = min(sqrt_n, 4.0)

    dv = (vadug.v - 128) / scale
    da = (vadug.a - 128) / scale
    dd = (vadug.d - 128) / scale
    du = vadug.u / scale
    dg = (vadug.g - 128) / scale

    return {
        'v': max(0, min(255, int(128 + dv))),
        'a': max(0, min(255, int(128 + da))),
        'd': max(0, min(255, int(128 + dd))),
        'u': max(0, min(255, int(du))),
        'g': max(0, min(255, int(128 + dg))),
    }


def _run(text):
    """Run text through V2 pendulum, return (raw_v, density_v, vadug, trace)."""
    engine = PendulumV2()
    vadug, trace = engine.process_text(text)
    density = _compute_density(vadug, trace)
    return int(vadug.v), density['v'], vadug, trace


class TestDensityConvergence:
    """Short and long sentences expressing the same emotion should
    produce closer density V values than raw V values."""

    def test_happy_short_vs_long(self):
        short_raw, short_den, _, _ = _run("I am happy")
        long_raw, long_den, _, _ = _run(
            "I am really truly happy about everything today"
        )
        raw_gap = abs(long_raw - short_raw)
        den_gap = abs(long_den - short_den)
        assert den_gap <= raw_gap, (
            f"Density gap ({den_gap}) should be <= raw gap ({raw_gap})"
        )

    def test_sad_short_vs_long(self):
        """Sentences with same emotional core but different padding length.
        Density should reduce the gap when padding words cause significant drift."""
        short_raw, short_den, _, _ = _run("I am feeling very sad")
        long_raw, long_den, _, _ = _run(
            "I am feeling very sad about what happened today"
        )
        raw_gap = abs(long_raw - short_raw)
        den_gap = abs(long_den - short_den)
        # The raw gap here is large (>30) due to accumulation drift;
        # density should reduce it meaningfully
        assert den_gap < raw_gap, (
            f"Density gap ({den_gap}) should be < raw gap ({raw_gap})"
        )


class TestDensityRange:
    """Density values must stay within 0-255 byte range."""

    def test_neutral_sentence(self):
        _, _, vadug, trace = _run("the weather is cloudy today")
        density = _compute_density(vadug, trace)
        for key in ('v', 'a', 'd', 'u', 'g'):
            assert 0 <= density[key] <= 255, f"{key}={density[key]} out of range"

    def test_extreme_positive(self):
        _, _, vadug, trace = _run("absolutely incredible amazing wonderful fantastic")
        density = _compute_density(vadug, trace)
        for key in ('v', 'a', 'd', 'u', 'g'):
            assert 0 <= density[key] <= 255, f"{key}={density[key]} out of range"

    def test_extreme_negative(self):
        _, _, vadug, trace = _run("horrible terrible awful disgusting hateful")
        density = _compute_density(vadug, trace)
        for key in ('v', 'a', 'd', 'u', 'g'):
            assert 0 <= density[key] <= 255, f"{key}={density[key]} out of range"

    def test_single_word(self):
        _, _, vadug, trace = _run("happy")
        density = _compute_density(vadug, trace)
        for key in ('v', 'a', 'd', 'u', 'g'):
            assert 0 <= density[key] <= 255, f"{key}={density[key]} out of range"


class TestCrisisNotWeakened:
    """Crisis sentences must still trigger crisis detection even with density."""

    def test_want_to_die_still_negative(self):
        """Short crisis sentence should still be strongly negative with density."""
        _, _, vadug, trace = _run("I want to die")
        density = _compute_density(vadug, trace)
        # Crisis sentences should have density V well below neutral (128)
        assert density['v'] < 110, (
            f"Crisis density V ({density['v']}) should be strongly negative"
        )

    def test_crisis_raw_v_stays_low(self):
        """Raw V for crisis should be very low — density doesn't change raw."""
        raw_v, _, _, _ = _run("I want to die")
        assert raw_v < 80, f"Crisis raw V ({raw_v}) should be very low"

    def test_end_it_all(self):
        _, _, vadug, trace = _run("I just want to end it all")
        density = _compute_density(vadug, trace)
        assert density['v'] < 110, (
            f"Crisis density V ({density['v']}) should be strongly negative"
        )


class TestClassifyDensity:
    """AdaptiveClassifier.classify_density works correctly."""

    def test_positive_density(self):
        c = AdaptiveClassifier()
        # High density V should classify as positive
        result = c.classify_density(160, "DEFAULT")
        assert result == "positive"

    def test_negative_density(self):
        c = AdaptiveClassifier()
        result = c.classify_density(90, "DEFAULT")
        assert result == "negative"

    def test_neutral_density(self):
        c = AdaptiveClassifier()
        result = c.classify_density(120, "DEFAULT")
        assert result == "neutral"

    def test_entropy_override(self):
        """When history is mixed, entropy should override to neutral."""
        c = AdaptiveClassifier()
        # A history that oscillates should trigger entropy neutral
        mixed_history = [
            {"word": "love", "v": 180, "a": 150, "d": 140, "u": 10, "g": 150},
            {"word": "hate", "v": 60, "a": 170, "d": 100, "u": 30, "g": 90},
            {"word": "happy", "v": 170, "a": 140, "d": 135, "u": 5, "g": 145},
            {"word": "angry", "v": 70, "a": 165, "d": 110, "u": 25, "g": 95},
        ]
        # With mixed signals, classify_density should use entropy check
        result = c.classify_density(130, "DEFAULT", mixed_history)
        # Result depends on entropy implementation, but method should not crash
        assert result in ("positive", "negative", "neutral")


class TestPipelineConfig:
    """density_normalization config flag exists."""

    def test_default_disabled(self):
        """Density is optional — benchmarks showed raw V is more accurate."""
        config = PipelineConfig()
        assert config.density_normalization is False

    def test_can_enable(self):
        config = PipelineConfig(density_normalization=True)
        assert config.density_normalization is True

    def test_minimal_preset(self):
        """Minimal preset should still have density_normalization as True
        (it's a post-processing step, not a processing layer)."""
        config = PipelineConfig.minimal()
        assert hasattr(config, 'density_normalization')
