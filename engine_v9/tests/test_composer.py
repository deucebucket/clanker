"""Tests for the V9 composer — charge resolution from decomposed equation."""

from engine_v9.composer import compose
from engine_v9.decomposer import decompose

CENTER = 128


def test_positive_sentence():
    eq = decompose("I am happy")
    result = compose(eq)
    assert result.v > CENTER


def test_negative_sentence():
    eq = decompose("I am sad")
    result = compose(eq)
    assert result.v < CENTER


def test_negation_flips():
    eq_pos = decompose("I am happy")
    eq_neg = decompose("I am not happy")
    v_pos = compose(eq_pos)
    v_neg = compose(eq_neg)
    assert v_pos.v > CENTER
    assert v_neg.v < CENTER


def test_intensifier_amplifies():
    eq_base = decompose("I am happy")
    eq_intense = decompose("I am very happy")
    v_base = compose(eq_base)
    v_intense = compose(eq_intense)
    assert v_intense.v > v_base.v


def test_strong_negative_event():
    eq = decompose("I just got laid off from work")
    result = compose(eq)
    assert result.v < 90
    assert result.d < CENTER


def test_yoda_same_result():
    eq1 = decompose("I just got laid off from work")
    eq2 = decompose("Laid off from work I just got")
    v1 = compose(eq1)
    v2 = compose(eq2)
    assert abs(v1.v - v2.v) <= 15


def test_neutral_sentence():
    eq = decompose("the cat sat on the mat")
    result = compose(eq)
    assert 100 < result.v < 156


def test_result_is_clamped():
    eq = decompose("I am absolutely devastated destroyed shattered")
    result = compose(eq)
    assert 0 <= result.v <= 255
    assert 0 <= result.a <= 255
    assert 0 <= result.d <= 255


def test_context_weighted_less_than_nucleus():
    eq1 = decompose("I am happy")
    eq2 = decompose("I am devastated but the weather is happy")
    v1 = compose(eq1)
    v2 = compose(eq2)
    assert v2.v < CENTER  # devastated as nucleus should dominate
