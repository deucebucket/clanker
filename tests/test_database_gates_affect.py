from __future__ import annotations

import importlib.util

import pytest

from clanker_lm.affect import AffectController, ClankerAffectBackend, HeuristicAffectBackend
from clanker_lm.database import LanguageStore
from clanker_lm.gates import ContextGate
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AffectReading,
    AffectVector,
    AnswerContract,
    AnswerStatus,
    CandidateResponse,
    ParseResult,
    SourceKind,
    SpeechAct,
)
from clanker_lm.parser import SemanticParser


def test_language_database_is_atomic_and_template_free():
    with LanguageStore() as store:
        summary = store.schema_summary()
        assert summary["atoms"] >= 90
        assert summary["grammar_rules"] >= 10
        assert summary["gate_rules"] >= 3
        assert summary["template_tables"] == 0
        tables = {
            row["name"]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert not ({"constructions", "construction_slots", "graph_edges"} & tables)


def test_atoms_are_single_tokens_and_grammar_contains_only_symbols():
    with LanguageStore() as store:
        store.assert_template_free()
        surfaces = [row["surface"] for row in store.connection.execute("SELECT surface FROM atoms")]
        assert surfaces
        assert all(surface and not any(char.isspace() for char in surface) for surface in surfaces)
        rules = store.grammar_rules("reply:answer")
        assert rules and rules[0].children == ("DECLARATIVE_CLAUSE",)
        assert not hasattr(rules[0], "template")


def test_gate_rules_are_independent_from_surface_generation():
    with LanguageStore() as store:
        rules = store.applicable_gate_rules({
            "severity_level": 2,
            "register": "neutral",
            "masking": False,
            "familial": True,
            "profanity": False,
        })
        assert any(rule["id"] == "high_severity_no_humor" for rule in rules)
        assert all("template" not in rule for rule in rules)


def gate_for(text: str):
    store = LanguageStore()
    memory = ConversationMemory()
    memory.begin_turn()
    parser = SemanticParser()
    parse = parser.parse(text, memory)
    affect = HeuristicAffectBackend().analyze(text)
    gate = ContextGate(store).decide(text, parse, affect, memory, answer_status=AnswerStatus.ACKNOWLEDGED)
    return store, gate


def test_low_severity_slang_locks_formal_and_high_severity_pools():
    store, gate = gate_for("My tummy hurts bruh.")
    try:
        assert gate.register == "casual"
        assert gate.severity == "moderate"
        assert "formal" in gate.locked_pools
        assert "high_severity" in gate.locked_pools
    finally:
        store.close()


def test_severe_family_content_locks_humor_and_slang():
    store, gate = gate_for("My mom is really sick.")
    try:
        assert gate.severity == "high"
        assert {"humor", "slang"}.issubset(gate.locked_pools)
    finally:
        store.close()


def test_collision_masking_preserves_casual_distance_but_locks_slang_pool():
    store, gate = gate_for("Bruh, my mom is really sick.")
    try:
        assert gate.masking
        assert gate.register == "casual"
        assert "casual_serious" in gate.allowed_pools
        assert {"formal", "humor", "slang", "high_severity"}.issubset(gate.locked_pools)
    finally:
        store.close()


def test_unresolved_reference_forces_probe_gate():
    store = LanguageStore()
    try:
        memory = ConversationMemory(); memory.begin_turn()
        parsed = SemanticParser().parse("She left.", memory)
        gate = ContextGate(store).decide(
            "She left.", parsed, HeuristicAffectBackend().analyze("She left."), memory,
            answer_status=AnswerStatus.MISSING_REFERENCE,
        )
        assert gate.requires_probe
        assert gate.response_act == "probe"
    finally:
        store.close()


def test_fallback_affect_backend_is_deterministic():
    backend = HeuristicAffectBackend()
    one = backend.analyze("Bruh, my mom is really sick.")
    two = backend.analyze("Bruh, my mom is really sick.")
    assert one.to_dict() == two.to_dict()
    assert one.vector.v < 128
    assert "MASKING" in one.structures


def test_candidate_ranker_never_selects_semantically_invalid_candidate():
    controller = AffectController(HeuristicAffectBackend())
    observed = AffectVector()
    target = AffectVector(v=150, a=110, d=150, i=185)
    invalid = CandidateResponse("A perfect sounding lie.", "bad", semantic_valid=False, priority=999)
    valid = CandidateResponse("I don't know.", "good", semantic_valid=True, priority=1)
    selected, scored = controller.rank_candidates([invalid, valid], observed, target)
    assert selected.construction_id == "good"
    assert next(item for item in scored if item.construction_id == "bad").score > 9000


def test_target_for_unknown_is_bounded_and_nonassertive():
    controller = AffectController(HeuristicAffectBackend())
    contract = AnswerContract(status=AnswerStatus.UNKNOWN, source=SourceKind.UNKNOWN)
    from clanker_lm.model import GateDecision
    target = controller.target_for(AffectVector(v=80, a=190, d=80, g=70), contract, GateDecision())
    assert 118 <= target.v <= 145
    assert target.d >= 145
    assert target.i >= 172


@pytest.mark.skipif(importlib.util.find_spec("engine") is None, reason="existing Clanker engine is absent from isolated package test")
def test_real_clanker_backend_integrates_with_existing_v8_engine():
    backend = ClankerAffectBackend()
    reading = backend.analyze("whatever makes you happy")
    assert reading.backend == "clanker-v8"
    assert reading.vector.v < 128
    transitioned = backend.transition(AffectVector(), reading.vector)
    assert isinstance(transitioned, AffectVector)
