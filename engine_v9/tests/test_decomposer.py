"""Tests for V9 decomposer — gravity-based nucleus detection + role assignment."""

from engine_v9.decomposer import decompose, Equation
from engine_v9.roots import RootCategory


def test_simple_negative_event():
    eq = decompose("I just got laid off from work")
    assert eq.nucleus.root.name == "EMPLOYMENT_LOSS"


def test_simple_positive():
    eq = decompose("I am so happy")
    assert eq.nucleus.root.category == RootCategory.POSITIVE_STATE


def test_yoda_order_same_nucleus():
    eq1 = decompose("I just got laid off from work")
    eq2 = decompose("Laid off from work I just got")
    assert eq1.nucleus.root.name == eq2.nucleus.root.name


def test_subject_is_self():
    eq = decompose("I am happy")
    assert eq.subject.root.category == RootCategory.SELF_REF


def test_implicit_self():
    eq = decompose("feeling sad today")
    assert eq.subject.root.category == RootCategory.SELF_REF


def test_negator_in_operators():
    eq = decompose("I am not happy")
    operator_cats = [a.root.category for a in eq.operators]
    assert RootCategory.NEGATOR in operator_cats


def test_context_atoms():
    eq = decompose("I got laid off from work yesterday")
    assert len(eq.context) >= 1


def test_tie_breaks_to_later_atom():
    eq = decompose("I am happy but devastated")
    assert eq.nucleus.root.charge[0] < 0


def test_equation_has_all_parts():
    eq = decompose("I am really happy")
    assert eq.subject is not None
    assert eq.nucleus is not None
    assert isinstance(eq.context, list)
    assert isinstance(eq.operators, list)
