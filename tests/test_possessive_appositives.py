"""Possessive-relative and appositive identity conformance for issue #84."""

from __future__ import annotations

from itertools import product

import pytest

from clanker_lm.affect import HeuristicAffectBackend
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AnswerStatus,
    AppositiveRelation,
    AppositiveRelationType,
    ModifierGapRole,
    ModifierRestriction,
)
from clanker_lm.parser import SemanticParser
from clanker_lm.runtime import ClankerLM


POSSESSORS = ("man", "woman", "teacher", "driver", "doctor")
POSSESSED = (
    ("car", "broke", "break"),
    ("sister", "left", "leave"),
    ("office", "moved", "move"),
    ("phone", "failed", "fail"),
    ("package", "fell", "fall"),
)
OUTERS = (
    ("called Mary", "call"),
    ("cried", "cry"),
    ("left", "leave"),
    ("arrived", "arrive"),
    ("laughed", "laugh"),
    ("waited", "wait"),
)
POSSESSIVE_CASES = tuple(product(POSSESSORS, POSSESSED, OUTERS))


def parse(text: str):
    memory = ConversationMemory()
    memory.begin_turn()
    return memory, SemanticParser().parse(text, memory)


@pytest.mark.parametrize("head,possessed,outer", POSSESSIVE_CASES)
def test_generated_possessive_relative_identity(head, possessed, outer) -> None:
    noun, relative_verb, relative_predicate = possessed
    outer_text, outer_predicate = outer
    memory, result = parse(
        f"The {head} whose {noun} {relative_verb} {outer_text}."
    )
    assert [event.predicate for event in result.events] == [
        outer_predicate,
        relative_predicate,
    ], result.diagnostics
    assert len(result.modifiers) == 1
    relation = result.modifiers[0]
    assert relation.gap_role == ModifierGapRole.POSSESSOR
    assert relation.possessed_entity_id
    possessed_entity = memory.get_entity(relation.possessed_entity_id)
    assert possessed_entity is not None
    assert possessed_entity.owner_id == relation.head_entity_id
    assert any(
        ref.key == relation.possessed_entity_id
        for ref in result.events[1].arguments.values()
    )


def test_nonrestrictive_possessive_reuses_head_and_possessed_identity() -> None:
    memory, result = parse("My supervisor, whose office moved, called.")
    relation = result.modifiers[0]
    assert result.events[0].arguments["agent"].key == relation.head_entity_id
    assert result.events[1].arguments["agent"].key == relation.possessed_entity_id
    assert memory.get_entity(relation.possessed_entity_id).owner_id == relation.head_entity_id


def test_whose_question_reads_owner_metadata() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("The man whose car broke called me.")
        answer = runtime.process("Whose car broke?")
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert "man" in answer.response.lower()
    finally:
        runtime.close()


def test_whose_relation_noun_prefers_contextual_possessed_entity() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("The woman whose sister left cried.")
        answer = runtime.process("Whose sister left?")
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert "woman" in answer.response.lower()
        assert "you" not in answer.response.lower()
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "text,relation_type,expected_name",
    (
        ("Sarah, my supervisor, called.", AppositiveRelationType.ROLE, "Sarah"),
        ("My supervisor Sarah called.", AppositiveRelationType.ROLE, "Sarah"),
        ("My friend, Sarah, called.", AppositiveRelationType.ROLE, "Sarah"),
        ("Chicago, the largest city in Illinois, is crowded.", AppositiveRelationType.DESCRIPTION, "Chicago"),
    ),
)
def test_appositive_subject_identity(text, relation_type, expected_name) -> None:
    memory, result = parse(text)
    assert len(result.events) == 1, result.diagnostics
    assert len(result.appositives) == 1
    relation = result.appositives[0]
    assert relation.relation_type == relation_type
    event_entity = next(
        ref.key for ref in result.events[0].arguments.values() if ref.kind.value == "entity"
    )
    assert event_entity == relation.head_entity_id
    entity = memory.get_entity(relation.head_entity_id)
    assert entity is not None
    assert expected_name.lower() in entity.canonical_name.lower()


def test_appositive_role_alias_is_queryable() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah, my supervisor, called.")
        who = runtime.process("Who called?")
        assert who.contract.status == AnswerStatus.ANSWERED
        assert any(
            label in who.response.lower()
            for label in ("sarah", "supervisor")
        )
        sarah = runtime.memory.find_by_alias("Sarah")
        supervisor = runtime.memory.find_by_alias("my supervisor")
        assert sarah.resolved and supervisor.resolved
        assert sarah.entity.entity_id == supervisor.entity.entity_id

        yes = runtime.process("Did my supervisor call?")
        assert yes.contract.status == AnswerStatus.TRUE
    finally:
        runtime.close()


def test_conflicting_appositive_role_is_explicitly_ambiguous() -> None:
    memory = ConversationMemory()
    memory.begin_turn()
    existing = memory.get_or_create_relation("user", "supervisor", surface="my supervisor")
    memory.begin_turn()
    result = SemanticParser().parse("Sarah, my supervisor, called.", memory)
    assert result.appositive_ambiguities
    assert result.unresolved
    assert not result.events
    assert memory.find_by_alias("Sarah").status in {"resolved", "missing"}
    assert memory.get_entity(existing.entity_id).entity_id == existing.entity_id


def test_appositive_relation_round_trip_and_snapshot() -> None:
    relation = AppositiveRelation(
        head_entity_id="sarah_1",
        primary_surface="Sarah",
        appositive_surface="my supervisor",
        relation_type=AppositiveRelationType.ROLE,
        restriction=ModifierRestriction.NONRESTRICTIVE,
        appositive_key="my supervisor",
        role_owner_id="user",
        role_name="supervisor",
    )
    assert AppositiveRelation.from_dict(relation.to_dict()).to_dict() == relation.to_dict()

    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah, my supervisor, called.")
        snapshot = runtime.memory.dumps(indent=None)
    finally:
        runtime.close()
    restored = ConversationMemory.loads(snapshot)
    assert len(restored.appositives) == 1
    stored = restored.appositives[0]
    assert stored.relation_id
    assert restored.appositives_for_entity(stored.head_entity_id) == [stored]


def test_repeated_appositive_does_not_duplicate_relation() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah, my supervisor, called.")
        runtime.process("Sarah, my supervisor, called.")
        assert len(runtime.memory.appositives) == 1
    finally:
        runtime.close()


def test_parse_result_serializes_appositive_trace() -> None:
    _memory, result = parse("Sarah, my supervisor, called.")
    payload = result.to_dict()
    assert payload["appositives"][0]["relation_type"] == "role"
    assert payload["appositive_ambiguities"] == []


def test_version_three_snapshot_remains_loadable() -> None:
    memory = ConversationMemory()
    legacy = memory.to_dict()
    legacy["snapshot_version"] = 3
    legacy.pop("appositives", None)
    legacy.pop("appositive_counter", None)
    restored = ConversationMemory.from_dict(legacy)
    assert restored.appositives == []
