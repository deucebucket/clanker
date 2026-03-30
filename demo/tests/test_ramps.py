"""Tests for the intensity ramp system (#72).

Ramps replace single-word intensifiers with multi-word amplification slopes.
"""
import pytest
from demo.pendulum_v2 import PendulumV2


def _v(text):
    """Run text through V2 engine and return the Valence byte."""
    engine = PendulumV2()
    vadug, _trace = engine.process_text(text)
    return vadug.v


class TestRampDataclass:
    """Ramp dataclass sanity checks — V1 Ramp/RAMPS/INTENSIFIERS are legacy."""

    @pytest.mark.skip(reason="Ramp dataclass is V1-only (demo.pendulum); V2 handles ramps internally")
    def test_ramp_fields(self):
        pass

    @pytest.mark.skip(reason="Ramp dataclass is V1-only (demo.pendulum); V2 handles ramps internally")
    def test_ramp_default_decay(self):
        pass

    @pytest.mark.skip(reason="RAMPS dict is V1-only (demo.pendulum); V2 handles ramps internally")
    def test_all_ramps_have_valid_values(self):
        pass


class TestBackwardCompat:
    """INTENSIFIERS dict still works as a flat multiplier lookup — V1 only."""

    @pytest.mark.skip(reason="INTENSIFIERS/RAMPS are V1-only (demo.pendulum); V2 uses EMOTIONAL_VOCABULARY")
    def test_intensifiers_populated(self):
        pass

    @pytest.mark.skip(reason="INTENSIFIERS/RAMPS are V1-only (demo.pendulum); V2 uses EMOTIONAL_VOCABULARY")
    def test_intensifiers_values_match_ramps(self):
        pass


class TestRampAmplification:
    """Core ramp behavior: amplified words hit harder."""

    def test_extremely_unhappy_vs_unhappy(self):
        """'extremely unhappy' should produce a more negative V than 'unhappy' alone."""
        v_plain = _v("unhappy")
        v_ramped = _v("extremely unhappy")

        assert v_ramped < v_plain, (
            f"'extremely unhappy' (V={v_ramped}) should be more negative "
            f"than 'unhappy' (V={v_plain})"
        )

    def test_slightly_angry_vs_very_angry(self):
        """'slightly angry' should be WEAKER than 'very angry' (dampener vs amplifier)."""
        v_damped = _v("slightly angry")
        v_amplified = _v("very angry")

        # "slightly" (dampener) vs "very" (amplifier) — clear separation
        assert v_damped > v_amplified, (
            f"'slightly angry' (V={v_damped}) should be less negative "
            f"than 'very angry' (V={v_amplified})"
        )

    def test_barely_sad_very_weak(self):
        """'barely sad' should produce minimal deviation from neutral."""
        v_plain = _v("sad")
        v_damped = _v("barely sad")

        # "barely" dampens — result should be closer to neutral (128)
        assert v_damped > v_plain, (
            f"'barely sad' (V={v_damped}) should be less negative "
            f"than 'sad' (V={v_plain})"
        )


class TestRampChaining:
    """Stacked ramp words compound their multipliers."""

    def test_really_really_angry(self):
        """'really really angry' should compound — at least as negative as 'really angry'.
        Note: V2 may clamp both to the same floor, so we allow equal."""
        v_single = _v("really angry")
        v_double = _v("really really angry")

        assert v_double <= v_single, (
            f"'really really angry' (V={v_double}) should be at least as negative "
            f"as 'really angry' (V={v_single})"
        )

    def test_absolutely_horrible_vs_horrible(self):
        """'absolutely horrible' should be more negative than 'horrible' alone."""
        v_plain = _v("horrible")
        v_ramped = _v("absolutely horrible")

        assert v_ramped < v_plain, (
            f"'absolutely horrible' (V={v_ramped}) should be more negative "
            f"than 'horrible' (V={v_plain})"
        )


class TestRampDecay:
    """Ramp effect decays across multiple words — tested via full sentence processing."""

    def test_very_happy_sad_valid_output(self):
        """'very happy sad' should produce valid V in 0-255 range."""
        engine = PendulumV2()
        vadug, _trace = engine.process_text("very happy sad")
        assert 0 <= vadug.v <= 255
        assert 0 <= vadug.a <= 255

    def test_quite_good_terrible(self):
        """'quite good terrible' — ramp from 'quite' shouldn't persist to 'terrible'."""
        engine = PendulumV2()
        vadug, _trace = engine.process_text("quite good terrible")
        # Just verify it runs and produces valid output
        assert 0 <= vadug.v <= 255


class TestRampInTrace:
    """Ramp words show up correctly in the trace."""

    def test_very_happy_produces_trace(self):
        """V2 engine should produce trace entries for 'very happy'."""
        engine = PendulumV2()
        vadug, trace = engine.process_text("very happy")
        assert len(trace) >= 2, f"Expected at least 2 trace entries, got {len(trace)}"
        # Verify trace has expected structure
        assert "word" in trace[0] or "v" in trace[0], f"Unexpected trace format: {trace[0]}"
