"""Restrictive and nonrestrictive finite relative-modifier conformance."""

from __future__ import annotations

from itertools import product

import pytest

from clanker_lm import lexicon
from clanker_lm.affect import HeuristicAffectBackend
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AnswerStatus,
    EntityKind,
    EntityModifierRelation,
    ModifierGapRole,
    ModifierRestriction,
)
from clanker_lm.parser import SemanticParser
from clanker_lm.runtime import ClankerLM


HEADS = ("woman", "man", "teacher", "nurse", "driver")
MODIFIERS = (
    ("called Sarah", "call"),
    ("bought a car", "buy"),
    ("wrote the note", "write"),
    ("opened the door", "open"),
    ("sent the letter", "send"),
)
OUTERS = (
    ("left yesterday", "leave"),
    ("arrived today", "arrive"),
    ("called Mary", "call"),
)
MARKERS = ("who", "that")
CONFORMANCE_CASES = tuple(product(HEADS, MODIFIERS, OUTERS, MARKERS))


def parse(text: str):
    memory = ConversationMemory()
    memory.begin_turn()
    return SemanticParser().parse(text, memory)


@pytest.mark.parametrize("head,modifier,outer,marker", CONFORMANCE_CASES)
def test_restrictive_subject_relative_matrix(head, modifier, outer, marker) -> None:
    modifier_text, modifier_predicate = modifier
    outer_text, outer_predicate = outer
    result = parse(f"The {head} {marker} {modifier_text} {outer_text}.")
    assert [event.predicate for event in result.events] == [
        outer_predicate,
        modifier_predicate,
    ], result.diagnostics
    assert [event.discourse_role for event in result.events] == [
        "main",
        "modifier",
    ]
    assert len(result.modifiers) == 1
    relation = result.modifiers[0]
    assert relation.marker == marker
    assert relation.gap_role == ModifierGapRole.AGENT
    assert relation.restriction == ModifierRestriction.RESTRICTIVE
    assert relation.modifier_event_index == 1
    modifier_event = result.events[1]
    assert modifier_event.arguments["agent"].key == relation.head_entity_id


def test_whom_maps_head_to_modifier_patient() -> None:
    result = parse("The woman whom Sarah called left.")
    relation = result.modifiers[0]
    assert relation.gap_role == ModifierGapRole.PATIENT
    assert result.events[1].arguments["patient"].key == relation.head_entity_id


@pytest.mark.parametrize(
    "text,head_kind",
    (
        ("The book that Sarah bought arrived.", "book"),
        ("The car which Sarah sold arrived.", "car"),
        ("The package that Mary sent arrived.", "package"),
    ),
)
def test_inanimate_object_gap_relative(text: str, head_kind: str) -> None:
    result = parse(text)
    relation = result.modifiers[0]
    assert relation.gap_role == ModifierGapRole.PATIENT
    assert result.events[1].arguments["patient"].key == relation.head_entity_id
    assert head_kind in result.events[0].arguments["patient"].surface.lower()


def test_whose_creates_possessed_entity_and_possessor_relation() -> None:
    result = parse("The woman whose car broke called Mary.")
    relation = result.modifiers[0]
    assert relation.gap_role == ModifierGapRole.POSSESSOR
    assert relation.possessed_entity_id
    assert any(
        ref.key == relation.possessed_entity_id
        for ref in result.events[1].arguments.values()
    )


def test_nonrestrictive_named_head_reuses_identity() -> None:
    memory = ConversationMemory()
    memory.begin_turn()
    parser = SemanticParser()
    result = parser.parse("Sarah, who called Mary, left.", memory)
    assert len(result.modifiers) == 1
    relation = result.modifiers[0]
    assert relation.restriction == ModifierRestriction.NONRESTRICTIVE
    sarah = memory.find_by_alias("Sarah")
    assert sarah.resolved
    assert relation.head_entity_id == sarah.entity.entity_id


def test_nonrestrictive_object_gap() -> None:
    result = parse("The car, which Sarah bought, arrived.")
    relation = result.modifiers[0]
    assert relation.restriction == ModifierRestriction.NONRESTRICTIVE
    assert relation.gap_role == ModifierGapRole.PATIENT


def test_relative_modifier_on_object_of_main_clause() -> None:
    result = parse("Sarah called the woman who bought a car.")
    assert [event.predicate for event in result.events] == ["call", "buy"]
    relation = result.modifiers[0]
    assert result.events[0].arguments["patient"].key == relation.head_entity_id
    assert result.events[1].arguments["agent"].key == relation.head_entity_id


def test_complementizer_that_is_not_a_relative_modifier() -> None:
    result = parse("I think that Sarah left.")
    assert not result.modifiers


def test_abstract_content_head_is_deferred_to_complement_slice() -> None:
    result = parse("I know the fact that Sarah left.")
    assert not result.modifiers


def test_multiple_relative_markers_create_explicit_ambiguity() -> None:
    result = parse("The woman who called Sarah who left arrived.")
    assert not result.modifiers
    assert len(result.modifier_ambiguities) == 1
    ambiguity = result.modifier_ambiguities[0]
    assert ambiguity.candidate_head_surfaces
    assert "multiple" in ambiguity.reason


def test_modified_generic_heads_remain_distinct() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("The woman who called Sarah left.")
        runtime.process("The woman who called Mary arrived.")
        modifiers = runtime.memory.modifiers
        assert len(modifiers) == 2
        assert modifiers[0].head_entity_id != modifiers[1].head_entity_id
        generic = runtime.memory.find_by_alias("woman")
        assert generic.status == "ambiguous"
    finally:
        runtime.close()


def test_modifier_event_and_main_event_are_both_queryable() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("The woman who called Sarah left.")
        called = runtime.process("Who called Sarah?")
        assert called.contract.status == AnswerStatus.ANSWERED
        assert "woman" in called.response.lower()

        left = runtime.process("Who left?")
        assert left.contract.status == AnswerStatus.ANSWERED
        assert "woman" in left.response.lower()
    finally:
        runtime.close()


def test_modifier_relation_round_trip() -> None:
    relation = EntityModifierRelation(
        head_entity_id="woman_1",
        modifier_event_index=1,
        marker="who",
        gap_role=ModifierGapRole.AGENT,
        restriction=ModifierRestriction.RESTRICTIVE,
        diagnostics=["test modifier"],
    )
    assert EntityModifierRelation.from_dict(relation.to_dict()).to_dict() == relation.to_dict()


def test_memory_snapshot_binds_and_restores_modifier_ids() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        result = runtime.process("The woman who called Sarah left.")
        assert len(result.parse.modifiers) == 1
        stored = runtime.memory.modifiers[0]
        assert stored.relation_id
        assert stored.modifier_event_id
        snapshot = runtime.memory.dumps(indent=None)
    finally:
        runtime.close()

    restored = ConversationMemory.loads(snapshot)
    assert restored.SNAPSHOT_VERSION == ConversationMemory.SNAPSHOT_VERSION
    assert len(restored.modifiers) == 1
    modifier = restored.modifiers[0]
    assert restored.modifiers_for_entity(modifier.head_entity_id) == [modifier]


def test_version_two_snapshot_remains_loadable() -> None:
    memory = ConversationMemory()
    legacy = memory.to_dict()
    legacy["snapshot_version"] = 2
    legacy.pop("modifiers", None)
    legacy.pop("modifier_counter", None)
    restored = ConversationMemory.from_dict(legacy)
    assert restored.modifiers == []


def test_repeated_relative_assertion_does_not_duplicate_modifier() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("The woman who called Sarah left.")
        runtime.process("The woman who called Sarah left.")
        assert len(runtime.memory.modifiers) == 1
    finally:
        runtime.close()


def test_parse_result_serializes_modifier_trace() -> None:
    result = parse("The woman who called Sarah left.")
    payload = result.to_dict()
    assert payload["modifiers"][0]["gap_role"] == "agent"
    assert payload["modifier_ambiguities"] == []


def test_participant_pronouns_precede_nominal_modifier_ambiguity() -> None:
    memory = ConversationMemory()
    memory.begin_turn()
    memory.get_or_create_modified_entity(
        "woman", "who called", kind=EntityKind.PERSON
    )
    memory.get_or_create_modified_entity(
        "woman", "who left", kind=EntityKind.PERSON
    )
    assert memory.find_by_alias("woman").status == "ambiguous"
    assert memory.find_by_alias("I").entity.entity_id == "user"
    assert memory.find_by_alias("you").entity.entity_id == "assistant"


def test_modifier_binding_rejects_wrong_event_index_with_signature() -> None:
    parser = SemanticParser()
    memory = ConversationMemory()
    memory.begin_turn()
    parsed = parser.parse("The woman who called Sarah left.", memory)
    stored = [memory.add_event(event) for event in reversed(parsed.events)]
    bound = memory.add_entity_modifier_relations(parsed.modifiers, stored)
    assert bound[0].modifier_event_id == next(
        event.event_id for event in stored if event.discourse_role == "modifier"
    )


def test_relative_gap_role_does_not_default_without_a_predicate() -> None:
    parser = SemanticParser()
    body = lexicon.tokenize("very tall", include_punctuation=False)
    assert parser._relative_gap_role("who", body) is None


def test_abstract_heads_are_deferred_for_every_relative_marker() -> None:
    parser = SemanticParser()
    for marker in ("that", "which"):
        memory = ConversationMemory()
        memory.begin_turn()
        parsed = parser.parse(
            f"The claim {marker} shocked Sarah was false.", memory
        )
        assert not parsed.modifiers


def test_runtime_snapshot_version_matches_memory_generation() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        assert runtime.SNAPSHOT_VERSION == ConversationMemory.SNAPSHOT_VERSION == 6
        snapshot = runtime.to_dict()
        assert snapshot["snapshot_version"] == 6
        assert snapshot["memory"]["snapshot_version"] == 6
        snapshot["snapshot_version"] = 2
        restored = ClankerLM.from_dict(
            snapshot, affect_backend=HeuristicAffectBackend()
        )
        restored.close()
    finally:
        runtime.close()
