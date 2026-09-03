"""Construction graph traversal tests."""

from clanker_lm.constructions import ConstructionGraph
from clanker_lm.models import GateProfile, ResponsePlan


def _gate(register="neutral", severity="low", locked=()):
    return GateProfile(
        register=register,
        severity=severity,
        collision_masking=False,
        locked_pools=tuple(locked),
    )


def test_graph_traversal_respects_register_and_required_slots():
    graph = ConstructionGraph()
    plan = ResponsePlan(
        act="conflict_support",
        slots={"agent_pronoun": "she", "repeat_phrase": " again"},
        gate=_gate(),
    )

    candidates = graph.traverse(plan)
    texts = {item.text for item in candidates}

    assert "I'm sorry she did this again. What happened?" in texts
    assert "That sounds frustrating. What happened?" in texts
    assert "That sucks. What happened?" not in texts


def test_locked_pool_removes_tagged_constructions():
    graph = ConstructionGraph()
    plan = ResponsePlan(
        act="celebrate",
        slots={},
        gate=_gate(register="casual", locked=("playful",)),
    )

    assert graph.traverse(plan) == ()


def test_context_probe_does_not_emit_empty_option_template():
    graph = ConstructionGraph()
    plan = ResponsePlan(
        act="context_probe",
        slots={"reference": "she", "options": ""},
        gate=_gate(),
    )

    candidates = graph.traverse(plan)
    assert candidates
    assert all("Do you mean ?" not in item.text for item in candidates)
