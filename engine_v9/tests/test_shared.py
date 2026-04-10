from engine_v9.shared import VADUG, PersonalityVector

def test_vadug_defaults_to_neutral():
    v = VADUG()
    assert v.v == 128
    assert v.a == 128
    assert v.d == 128
    assert v.u == 0
    assert v.g == 128
    assert v.w == 128
    assert v.i == 128

def test_vadug_clamps():
    v = VADUG(v=300, a=-10)
    assert v.v == 255
    assert v.a == 0

def test_vadug_to_bytes():
    v = VADUG(v=100, a=200, d=50, u=30, g=180, w=90, i=160)
    b = v.to_bytes()
    assert len(b) == 7
    assert b[0] == 100

def test_personality_sensitivity_range():
    p = PersonalityVector()
    assert 0.5 <= p.emotional_sensitivity <= 2.0
