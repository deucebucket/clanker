"""
Tests for engine_v9.root_map — three-tier word-to-root lookup.
"""

from engine_v9.root_map import map_to_root
from engine_v9.roots import RootCategory


def test_direct_lookup():
    root = map_to_root("happy")
    assert root.name == "HAPPY"


def test_case_insensitive():
    root = map_to_root("HAPPY")
    assert root.name == "HAPPY"


def test_morphological_strip():
    root = map_to_root("happiness")
    assert root.category == RootCategory.POSITIVE_STATE


def test_ly_suffix():
    root = map_to_root("sadly")
    assert root.category == RootCategory.NEGATIVE_STATE


def test_unknown_word_returns_generic():
    root = map_to_root("splendiferous")
    assert root.name == "OBJECT_GENERIC"
    assert root.phase == "GAS"
    assert root.charge[0] == 0


def test_self_ref():
    root = map_to_root("i")
    assert root.category == RootCategory.SELF_REF


def test_negator():
    root = map_to_root("not")
    assert root.category == RootCategory.NEGATOR


def test_intensifier():
    root = map_to_root("very")
    assert root.category == RootCategory.INTENSIFIER


def test_compound_event():
    root = map_to_root("laidoff")
    assert root.category == RootCategory.COMPOUND_EVENT
    assert root.charge[0] < -40
