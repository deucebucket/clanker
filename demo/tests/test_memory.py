"""Tests for VADUG conversation memory (demo.memory)."""

import os
import sys
import json
import tempfile

import pytest

# Ensure the demo package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from demo.shared import VADUG
from demo.memory import TurnMemory, SessionMemory, ConversationMemory


# ── Helpers ──────────────────────────────────────────────────────────

def _vadug(v=128, a=128, d=128, u=0, g=128):
    return VADUG(v=v, a=a, d=d, u=u, g=g)


# ── SessionMemory ────────────────────────────────────────────────────

class TestSessionGravityWell:
    def test_empty_session_returns_neutral(self):
        s = SessionMemory()
        well = s.gravity_well
        assert well.v == 128
        assert well.g == 128

    def test_single_turn(self):
        s = SessionMemory()
        s.add_turn(_vadug(v=200, a=100, d=150, u=50, g=80),
                   _vadug(), "STATEMENT", "B")
        well = s.gravity_well
        assert well.v == 200
        assert well.a == 100
        assert well.d == 150
        assert well.u == 50
        assert well.g == 80

    def test_averages_correctly(self):
        s = SessionMemory()
        s.add_turn(_vadug(v=100, a=100, d=100, u=100, g=100),
                   _vadug(), "STATEMENT", "C")
        s.add_turn(_vadug(v=200, a=200, d=200, u=200, g=200),
                   _vadug(), "STATEMENT", "C")
        well = s.gravity_well
        assert well.v == 150
        assert well.a == 150
        assert well.d == 150
        assert well.u == 150
        assert well.g == 150


class TestEmotionalArc:
    def test_insufficient_data_single_turn(self):
        s = SessionMemory()
        s.add_turn(_vadug(v=50), _vadug(), "STATEMENT", "C")
        assert s.emotional_arc == "insufficient_data"

    def test_insufficient_data_empty(self):
        s = SessionMemory()
        assert s.emotional_arc == "insufficient_data"

    def test_ascending(self):
        s = SessionMemory()
        # First half: low valence
        s.add_turn(_vadug(v=50), _vadug(), "VENTING", "D")
        s.add_turn(_vadug(v=60), _vadug(), "VENTING", "D")
        # Second half: high valence (diff > 15)
        s.add_turn(_vadug(v=120), _vadug(), "STATEMENT", "B")
        s.add_turn(_vadug(v=130), _vadug(), "STATEMENT", "B")
        assert s.emotional_arc == "ascending"

    def test_descending(self):
        s = SessionMemory()
        s.add_turn(_vadug(v=200), _vadug(), "GREETING", "A")
        s.add_turn(_vadug(v=190), _vadug(), "STATEMENT", "A")
        s.add_turn(_vadug(v=80), _vadug(), "VENTING", "D")
        s.add_turn(_vadug(v=70), _vadug(), "VENTING", "D")
        assert s.emotional_arc == "descending"

    def test_stable(self):
        s = SessionMemory()
        s.add_turn(_vadug(v=128), _vadug(), "STATEMENT", "C")
        s.add_turn(_vadug(v=130), _vadug(), "STATEMENT", "C")
        s.add_turn(_vadug(v=126), _vadug(), "STATEMENT", "C")
        s.add_turn(_vadug(v=132), _vadug(), "STATEMENT", "C")
        assert s.emotional_arc == "stable"


class TestCrisisCount:
    def test_no_crises(self):
        s = SessionMemory()
        s.add_turn(_vadug(), _vadug(), "STATEMENT", "C")
        s.add_turn(_vadug(), _vadug(), "STATEMENT", "B")
        assert s.crisis_count == 0

    def test_counts_f_grades(self):
        s = SessionMemory()
        s.add_turn(_vadug(), _vadug(), "VENTING", "F")
        s.add_turn(_vadug(), _vadug(), "VENTING", "F-")
        s.add_turn(_vadug(), _vadug(), "VENTING", "F+")
        s.add_turn(_vadug(), _vadug(), "STATEMENT", "C")
        assert s.crisis_count == 3

    def test_empty_session(self):
        assert SessionMemory().crisis_count == 0


class TestToBytes:
    def test_byte_count(self):
        s = SessionMemory()
        s.add_turn(_vadug(), _vadug(), "STATEMENT", "C")
        s.add_turn(_vadug(), _vadug(), "STATEMENT", "C")
        s.add_turn(_vadug(), _vadug(), "STATEMENT", "C")
        assert len(s.to_bytes()) == 15  # 3 turns * 5 bytes

    def test_byte_size_property(self):
        s = SessionMemory()
        assert s.byte_size == 0
        s.add_turn(_vadug(), _vadug(), "STATEMENT", "C")
        assert s.byte_size == 5
        s.add_turn(_vadug(), _vadug(), "STATEMENT", "C")
        assert s.byte_size == 10

    def test_byte_values(self):
        s = SessionMemory()
        s.add_turn(_vadug(v=10, a=20, d=30, u=40, g=50),
                   _vadug(), "STATEMENT", "C")
        raw = s.to_bytes()
        assert raw == bytes([10, 20, 30, 40, 50])

    def test_empty_session_bytes(self):
        assert SessionMemory().to_bytes() == b""


# ── ConversationMemory ───────────────────────────────────────────────

class TestConversationMemory:
    def test_empty_memory(self):
        cm = ConversationMemory()
        rg = cm.relationship_gravity
        assert rg.v == 128
        assert rg.g == 128

    def test_single_session_gravity(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(v=100, g=100), _vadug())
        cm.add_turn(_vadug(v=200, g=200), _vadug())
        rg = cm.relationship_gravity
        assert rg.v == 150
        assert rg.g == 150

    def test_new_session_archives(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(v=100), _vadug())
        cm.new_session()
        assert len(cm.sessions) == 1
        assert len(cm.current_session.turns) == 0

    def test_new_session_empty_no_archive(self):
        cm = ConversationMemory()
        cm.new_session()
        assert len(cm.sessions) == 0


class TestRelationshipGravityWeighting:
    def test_recent_sessions_weighted_higher(self):
        cm = ConversationMemory()
        # Session 1: low valence (weight=1)
        cm.add_turn(_vadug(v=50), _vadug())
        cm.new_session()
        # Session 2: high valence (weight=2)
        cm.add_turn(_vadug(v=200), _vadug())
        cm.new_session()
        # Weighted: (50*1 + 200*2) / 3 = 450/3 = 150
        rg = cm.relationship_gravity
        assert rg.v == 150

    def test_three_sessions_weighting(self):
        cm = ConversationMemory()
        # Session 1 (weight=1): v=60
        cm.add_turn(_vadug(v=60), _vadug())
        cm.new_session()
        # Session 2 (weight=2): v=120
        cm.add_turn(_vadug(v=120), _vadug())
        cm.new_session()
        # Session 3 (weight=3): v=180
        cm.add_turn(_vadug(v=180), _vadug())
        # Weighted: (60*1 + 120*2 + 180*3) / 6 = (60+240+540)/6 = 140
        rg = cm.relationship_gravity
        assert rg.v == 140


class TestBackgroundModifier:
    def test_neutral_gravity(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(g=128), _vadug())
        mod = cm.get_background_modifier()
        assert mod == 1.0

    def test_heavy_history(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(g=0), _vadug())
        mod = cm.get_background_modifier()
        assert mod > 1.0
        assert mod == pytest.approx(1.5, abs=0.01)

    def test_light_history(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(g=255), _vadug())
        mod = cm.get_background_modifier()
        assert mod < 1.0
        # (128 - 255) / 256 = -127/256 ≈ -0.496
        assert mod == pytest.approx(0.504, abs=0.01)

    def test_empty_memory_neutral(self):
        cm = ConversationMemory()
        mod = cm.get_background_modifier()
        assert mod == 1.0


class TestEmotionalProfile:
    def test_carries_pain(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(v=50, a=80, d=60, g=60), _vadug())
        profile = cm.emotional_profile
        assert "carries_pain" in profile["traits"]
        assert "heavy_history" in profile["traits"]
        assert "calm_presence" in profile["traits"]
        assert "feels_helpless" in profile["traits"]

    def test_generally_positive(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(v=200, a=200, d=200, g=200), _vadug())
        profile = cm.emotional_profile
        assert "generally_positive" in profile["traits"]
        assert "emotionally_buoyant" in profile["traits"]
        assert "high_intensity" in profile["traits"]
        assert "in_control" in profile["traits"]

    def test_crisis_history_in_traits(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(v=20), _vadug(), grade="F")
        cm.add_turn(_vadug(v=30), _vadug(), grade="F-")
        profile = cm.emotional_profile
        assert "crisis_history(2)" in profile["traits"]

    def test_session_and_turn_counts(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(), _vadug())
        cm.add_turn(_vadug(), _vadug())
        cm.new_session()
        cm.add_turn(_vadug(), _vadug())
        profile = cm.emotional_profile
        assert profile["sessions"] == 2
        assert profile["total_turns"] == 3

    def test_total_bytes(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(), _vadug())
        cm.add_turn(_vadug(), _vadug())
        cm.new_session()
        cm.add_turn(_vadug(), _vadug())
        profile = cm.emotional_profile
        assert profile["total_bytes"] == 15  # 3 turns * 5 bytes

    def test_empty_profile(self):
        cm = ConversationMemory()
        profile = cm.emotional_profile
        assert profile["sessions"] == 0
        assert profile["total_turns"] == 0
        assert profile["total_bytes"] == 0


class TestSave:
    def test_save_to_json(self):
        cm = ConversationMemory()
        cm.add_turn(_vadug(v=100, a=150), _vadug(), "VENTING", "D")
        cm.new_session()
        cm.add_turn(_vadug(v=200, a=100), _vadug(), "GREETING", "A")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            path = f.name

        try:
            cm.save(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data["sessions"]) == 1
            assert data["current_turns"] == 1
            assert "relationship_gravity" in data
            assert "profile" in data
        finally:
            os.unlink(path)
