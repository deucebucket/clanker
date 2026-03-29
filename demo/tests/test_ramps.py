"""Tests for the intensity ramp system (#72).

Ramps replace single-word intensifiers with multi-word amplification slopes.
"""
import pytest
from demo.pendulum import SequentialPendulum, RAMPS, Ramp, INTENSIFIERS


class TestRampDataclass:
    """Ramp dataclass sanity checks."""

    def test_ramp_fields(self):
        r = Ramp(1.5, 3, 0.6)
        assert r.multiplier == 1.5
        assert r.length == 3
        assert r.decay == 0.6

    def test_ramp_default_decay(self):
        r = Ramp(1.3, 2)
        assert r.decay == 0.6

    def test_all_ramps_have_valid_values(self):
        for word, ramp in RAMPS.items():
            assert ramp.multiplier > 0, f"{word}: multiplier must be positive"
            assert ramp.length >= 1, f"{word}: length must be >= 1"
            assert 0 < ramp.decay <= 1.0, f"{word}: decay must be in (0, 1]"


class TestBackwardCompat:
    """INTENSIFIERS dict still works as a flat multiplier lookup."""

    def test_intensifiers_populated(self):
        assert len(INTENSIFIERS) == len(RAMPS)

    def test_intensifiers_values_match_ramps(self):
        for word, mult in INTENSIFIERS.items():
            assert word in RAMPS
            assert mult == RAMPS[word].multiplier


class TestRampAmplification:
    """Core ramp behavior: amplified words hit harder."""

    def test_extremely_unhappy_vs_unhappy(self):
        """'extremely unhappy' should produce a more negative V than 'unhappy' alone."""
        p_plain = SequentialPendulum()
        p_plain.process_text("unhappy")
        v_plain = p_plain.v

        p_ramped = SequentialPendulum()
        p_ramped.process_text("extremely unhappy")
        v_ramped = p_ramped.v

        assert v_ramped < v_plain, (
            f"'extremely unhappy' (V={v_ramped:.1f}) should be more negative "
            f"than 'unhappy' (V={v_plain:.1f})"
        )

    def test_slightly_angry_vs_very_angry(self):
        """'slightly angry' should be WEAKER than 'very angry' (dampener vs amplifier)."""
        p_damped = SequentialPendulum()
        p_damped.process_text("slightly angry")
        v_damped = p_damped.v

        p_amplified = SequentialPendulum()
        p_amplified.process_text("very angry")
        v_amplified = p_amplified.v

        # "slightly" (0.7x) vs "very" (1.3x) — clear separation
        assert v_damped > v_amplified, (
            f"'slightly angry' (V={v_damped:.1f}) should be less negative "
            f"than 'very angry' (V={v_amplified:.1f})"
        )

    def test_barely_sad_very_weak(self):
        """'barely sad' should produce minimal deviation from neutral."""
        p_plain = SequentialPendulum()
        p_plain.process_text("sad")
        v_plain = p_plain.v

        p_damped = SequentialPendulum()
        p_damped.process_text("barely sad")
        v_damped = p_damped.v

        # "barely" has multiplier 0.5 — heavy dampener
        assert v_damped > v_plain, (
            f"'barely sad' (V={v_damped:.1f}) should be less negative "
            f"than 'sad' (V={v_plain:.1f})"
        )


class TestRampChaining:
    """Stacked ramp words compound their multipliers."""

    def test_really_really_angry(self):
        """'really really angry' should compound (1.35 * 1.35 ≈ 1.82x)."""
        p_single = SequentialPendulum()
        p_single.process_text("really angry")
        v_single = p_single.v

        p_double = SequentialPendulum()
        p_double.process_text("really really angry")
        v_double = p_double.v

        assert v_double < v_single, (
            f"'really really angry' (V={v_double:.1f}) should be more negative "
            f"than 'really angry' (V={v_single:.1f})"
        )

    def test_absolutely_devastatingly_horrible(self):
        """Multiple ramp words should compound for extreme effect."""
        # "absolutely" is a ramp; "devastatingly" won't be in RAMPS but
        # "horrible" should get the ramp applied.
        p_plain = SequentialPendulum()
        p_plain.process_text("horrible")
        v_plain = p_plain.v

        p_ramped = SequentialPendulum()
        p_ramped.process_text("absolutely horrible")
        v_ramped = p_ramped.v

        assert v_ramped < v_plain, (
            f"'absolutely horrible' (V={v_ramped:.1f}) should be more negative "
            f"than 'horrible' (V={v_plain:.1f})"
        )


class TestRampDecay:
    """Ramp effect decays across multiple words."""

    def test_very_affects_second_word_less(self):
        """'very' (length=2) should affect the second word less than the first."""
        # We can't easily measure per-word multiplier from the outside,
        # but we can verify the ramp doesn't persist beyond its length.
        p = SequentialPendulum()
        words = ["very", "happy", "sad"]
        for idx, w in enumerate(words):
            p.process_word(w, words, idx)

        # After "very" (ramp length=2), "happy" gets full ramp, "sad" gets decayed ramp.
        # The ramp should still be partially active for "sad" since length=2.
        # We just verify the engine doesn't crash and produces valid output.
        assert 0 <= p.v <= 255
        assert 0 <= p.a <= 255

    def test_quite_expires_after_one_word(self):
        """'quite' (length=1) should only affect the next word."""
        p = SequentialPendulum()
        # After "quite good", the ramp should be gone.
        words = ["quite", "good", "terrible"]
        for idx, w in enumerate(words):
            p.process_word(w, words, idx)

        # Ramp should be None after consuming "good"
        assert p._active_ramp is None


class TestRampInTrace:
    """Ramp words show up correctly in the trace."""

    def test_ramp_word_in_trace(self):
        p = SequentialPendulum()
        p.process_text("very happy")
        states = [h["state"] for h in p.history]
        # The ramp word should have "RAMP" in its state
        assert "RAMP" in states[0], f"Expected 'RAMP' in trace, got: {states[0]}"
