"""Tests for the Battleship probe system."""

import pytest

from engine.shared import VADUG
from engine.battleship import (
    PROBES,
    fire_probe,
    triangulate,
    state_transition,
    NEUTRAL,
)


class TestProbes:
    """Verify probes are loaded with real VADUGWI values."""

    def test_probes_loaded(self):
        assert len(PROBES) == 5

    def test_probe_names(self):
        names = [p.name for p in PROBES]
        assert "minimal_ack" in names
        assert "slight_validation" in names
        assert "clarification" in names
        assert "light_redirect" in names
        assert "direct_check" in names

    def test_probes_have_vadug(self):
        for probe in PROBES:
            assert isinstance(probe.vadug, VADUG)

    def test_probes_have_tests_for(self):
        for probe in PROBES:
            assert len(probe.tests_for) > 0


class TestStateTransition:
    """Verify the local state transition function."""

    def test_neutral_plus_neutral(self):
        result = state_transition(NEUTRAL, NEUTRAL)
        assert result.v == 128
        assert result.d == 128
        assert result.g == 128

    def test_transition_blends(self):
        crisis = VADUG(v=30, a=180, d=40, u=200, g=30)
        result = state_transition(crisis, NEUTRAL)
        # Should be between crisis and neutral
        assert result.v > 30 and result.v < 128
        assert result.d > 40 and result.d < 128


class TestFireProbe:
    """Test firing individual probes against known states."""

    def test_crisis_high_vibration_on_minimal_ack(self):
        """Crisis state should produce high vibration on minimal_ack."""
        crisis = VADUG(v=30, a=180, d=40, u=200, g=30)
        minimal_ack = [p for p in PROBES if p.name == "minimal_ack"][0]
        result = fire_probe(minimal_ack, crisis)
        # Crisis deviates heavily from neutral — vibration should be high
        assert result.vibration > 20, f"Expected high vibration for crisis, got {result.vibration}"

    def test_neutral_low_vibration(self):
        """Neutral state should produce low vibration on any probe."""
        neutral = VADUG(v=128, a=128, d=128, u=0, g=128)
        for probe in PROBES:
            result = fire_probe(probe, neutral)
            assert result.vibration < 5, (
                f"Expected low vibration for neutral on {probe.name}, "
                f"got {result.vibration}"
            )

    def test_joy_positive_vibration(self):
        """Joy state should produce measurable vibration."""
        joy = VADUG(v=200, a=180, d=170, u=0, g=200)
        minimal_ack = [p for p in PROBES if p.name == "minimal_ack"][0]
        result = fire_probe(minimal_ack, joy)
        # Joy deviates from neutral — should see vibration
        assert result.vibration > 10, f"Expected positive vibration for joy, got {result.vibration}"

    def test_probe_result_has_zones(self):
        """ProbeResult should include zone classification."""
        state = VADUG(v=30, a=180, d=40, u=200, g=30)
        result = fire_probe(PROBES[0], state)
        assert isinstance(result.estimated_zone, str)
        assert 0.0 <= result.zone_confidence <= 1.0


class TestTriangulate:
    """Test triangulation across multiple probes."""

    def test_crisis_high_total_vibration(self):
        """Crisis state should produce high total vibration."""
        crisis = VADUG(v=30, a=180, d=40, u=200, g=30)
        result = triangulate(crisis, num_probes=3)
        assert result["total_vibration"] > 60, (
            f"Expected high total vibration for crisis, got {result['total_vibration']}"
        )

    def test_crisis_confidence_above_threshold(self):
        """Crisis triangulation should have confidence > 0.5."""
        crisis = VADUG(v=30, a=180, d=40, u=200, g=30)
        result = triangulate(crisis, num_probes=3)
        assert result["confidence"] > 0.5, (
            f"Expected confidence > 0.5 for crisis, got {result['confidence']}"
        )

    def test_neutral_low_total_vibration(self):
        """Neutral state should produce low total vibration."""
        neutral = VADUG(v=128, a=128, d=128, u=0, g=128)
        result = triangulate(neutral, num_probes=3)
        assert result["total_vibration"] < 15, (
            f"Expected low total vibration for neutral, got {result['total_vibration']}"
        )

    def test_triangulate_returns_expected_keys(self):
        """Triangulation result should have all expected keys."""
        state = VADUG(v=100, a=150, d=80, u=50, g=90)
        result = triangulate(state, num_probes=3)
        assert "estimated_zone" in result
        assert "confidence" in result
        assert "total_vibration" in result
        assert "probe_results" in result
        assert len(result["probe_results"]) == 3

    def test_triangulate_all_probes(self):
        """Can fire all 5 probes."""
        crisis = VADUG(v=30, a=180, d=40, u=200, g=30)
        result = triangulate(crisis, num_probes=5)
        assert len(result["probe_results"]) == 5
