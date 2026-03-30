import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def test_vadug_import():
    from engine.shared import VADUG
    v = VADUG(v=100, a=150, d=128, u=50, g=90)
    assert v.v == 100
    assert v.g == 90

def test_vocabulary_import():
    from engine.vocabulary import VOCABULARY
    assert len(VOCABULARY) > 2000
    assert "happy" in VOCABULARY
    assert "sad" in VOCABULARY

def test_zones_import():
    from engine.zones import ZONES
    assert "JOY" in ZONES
    assert "CRISIS" in ZONES

def test_personality_import():
    from engine.personality import PersonalityVector
    p = PersonalityVector()
    assert hasattr(p, "emotional_sensitivity")

def test_fuzzy_import():
    from engine.fuzzy import fuzzy_match
    assert fuzzy_match("happyyyy") == "happy"
    assert fuzzy_match("tbh") == "honestly"
