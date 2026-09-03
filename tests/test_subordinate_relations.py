"""Typed finite subordinate-clause relation conformance."""

from __future__ import annotations

from itertools import product

import pytest

from clanker_lm.affect import HeuristicAffectBackend
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AnswerStatus,
    ClauseRelation,
    ClauseRelationDirection,
    ClauseRelationType,
)
from clanker_lm.parser import SemanticParser
from clanker_lm.runtime import ClankerLM


MAIN_CLAUSES = (
    ("Sarah left", "leave"),
    ("Mary opened the door", "open"),
    ("John called the office", "call"),
    ("The technician moved the box", "move"),
    ("My sister wrote the note", "write"),
)
SUBORDINATE_CLAUSES = (
    ("John called", "call"),
    ("Mary opened the window", "open"),
    ("The driver left", "leave"),
    ("My brother sent the letter", "send"),
    ("John bought the car", "buy"),
)
RELATION_SPECS = (
    ("because", ClauseRelationType.CAUSE, ClauseRelationDirection.SUBORDINATE_TO_MAIN),
    ("when", ClauseRelationType.TEMPORAL_WHEN, ClauseRelationDirection.SYMMETRIC),
    ("before", ClauseRelationType.TEMPORAL_BEFORE, ClauseRelationDirection.MAIN_TO_SUBORDINATE),
    ("after", ClauseRelationType.TEMPORAL_AFTER, ClauseRelationDirection.MAIN_TO_SUBORDINATE),
    ("until", ClauseRelationType.TEMPORAL_UNTIL, ClauseRelationDirection.MAIN_TO_SUBORDINATE),
    ("if", ClauseRelationType.CONDITION, ClauseRelationDirection.SUBORDINATE_TO_MAIN),
    ("unless", ClauseRelationType.EXCEPTION_CONDITION, ClauseRelationDirection.SUBORDINATE_TO_MAIN),
    ("although", ClauseRelationType.CONCESSION, ClauseRelationDirection.SUBORDINATE_TO_MAIN),
    ("though", ClauseRelationType.CONCESSION, ClauseRelationDirection.SUBORDINATE_TO_MAIN),
    ("even though", ClauseRelationType.CONCESSION, ClauseRelationDirection.SUBORDINATE_TO_MAIN),
    ("since", ClauseRelationType.AMBIGUOUS, ClauseRelationDirection.UNRESOLVED),
    ("while", ClauseRelationType.AMBIGUOUS, ClauseRelationDirection.UNRESOLVED),
)
CONFORMANCE_CASES = tuple(
    product(RELATION_SPECS, MAIN_CLAUSES, SUBORDINATE_CLAUSES, ("main", "front"))
)


def parse(text: str):
    memory = ConversationMemory()
    memory.begin_turn()
    return SemanticParser().parse(text, memory)


@pytest.mark.parametrize("spec,main,subordinate,order", CONFORMANCE_CASES)
def test_subordinate_relation_matrix(spec, main, subordinate, order) -> None:
    marker, expected_type, expected_direction = spec
    main_text, main_predicate = main
    subordinate_text, subordinate_predicate = subordinate
    if order == "main":
        text = f"{main_text} {marker} {subordinate_text}."
    else:
        text = f"{marker.capitalize()} {subordinate_text}, {main_text}."

    result = parse(text)
    assert [event.predicate for event in result.events] == [
        main_predicate,
        subordinate_predicate,
    ], result.diagnostics
    assert [event.discourse_role for event in result.events] == [
        "main",
        "subordinate",
    ]
    assert len(result.relations) == 1
    relation = result.relations[0]
    assert relation.main_event_index == 0
    assert relation.subordinate_event_index == 1
    assert relation.marker == marker
    assert relation.relation_type == expected_type
    assert relation.direction == expected_direction
    if expected_type == ClauseRelationType.AMBIGUOUS:
        assert relation.candidate_types
        assert relation.certainty < 200


def test_cause_relation_preserves_existing_why_anchor() -> None:
    result = parse("Sarah left because John called.")
    main = result.events[0]
    assert main.arguments["cause"].surface == "John called"
    assert result.relations[0].relation_type == ClauseRelationType.CAUSE


def test_since_with_explicit_time_anchor_resolves_temporally() -> None:
    result = parse("Sarah left since John called yesterday.")
    relation = result.relations[0]
    assert relation.relation_type == ClauseRelationType.TEMPORAL_SINCE
    assert relation.direction == ClauseRelationDirection.MAIN_TO_SUBORDINATE
    assert not relation.candidate_types


def test_paired_progressives_resolve_while_as_overlap() -> None:
    result = parse("Sarah was reading while John was writing.")
    relation = result.relations[0]
    assert relation.relation_type == ClauseRelationType.TEMPORAL_OVERLAP
    assert relation.direction == ClauseRelationDirection.SYMMETRIC


def test_so_that_modality_resolves_purpose() -> None:
    result = parse("Sarah opened the door so that John could leave.")
    relation = result.relations[0]
    assert relation.relation_type == ClauseRelationType.PURPOSE
    assert result.events[0].arguments["purpose"].surface == "John could leave"


def test_so_that_change_of_state_resolves_result() -> None:
    result = parse("Sarah closed the door so that the room got quiet.")
    relation = result.relations[0]
    assert relation.relation_type == ClauseRelationType.RESULT


def test_so_that_without_discriminator_is_explicitly_ambiguous() -> None:
    result = parse("Sarah closed the door so that John called.")
    relation = result.relations[0]
    assert relation.relation_type == ClauseRelationType.AMBIGUOUS
    assert set(relation.candidate_types) == {
        ClauseRelationType.PURPOSE,
        ClauseRelationType.RESULT,
    }


@pytest.mark.parametrize(
    "text",
    (
        "Sarah called John after Monday.",
        "Sarah left before dinner.",
        "Sarah called John since Tuesday.",
        "Sarah waited until noon.",
    ),
)
def test_lexical_temporal_prepositions_do_not_create_clause_relations(text: str) -> None:
    result = parse(text)
    assert len(result.events) == 1, result.diagnostics
    assert not result.relations


def test_clause_relation_round_trip() -> None:
    relation = ClauseRelation(
        relation_type=ClauseRelationType.CONDITION,
        main_event_index=0,
        subordinate_event_index=1,
        marker="if",
        direction=ClauseRelationDirection.SUBORDINATE_TO_MAIN,
        candidate_types=[],
        certainty=220,
        diagnostics=["test relation"],
    )
    assert ClauseRelation.from_dict(relation.to_dict()).to_dict() == relation.to_dict()


def test_memory_snapshot_binds_and_restores_relation_ids() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        result = runtime.process("Sarah left because John called.")
        assert len(result.parse.relations) == 1
        assert len(runtime.memory.relations) == 1
        stored = runtime.memory.relations[0]
        assert stored.main_event_id
        assert stored.subordinate_event_id
        snapshot = runtime.memory.dumps(indent=None)
    finally:
        runtime.close()

    restored = ConversationMemory.loads(snapshot)
    assert restored.SNAPSHOT_VERSION == 2
    assert len(restored.relations) == 1
    relation = restored.relations[0]
    assert restored.relation_between(
        relation.main_event_id,
        relation.subordinate_event_id,
    ) is not None
    assert len(restored.relations_for_event(relation.main_event_id)) == 1


def test_version_one_memory_snapshot_remains_loadable() -> None:
    memory = ConversationMemory()
    legacy = memory.to_dict()
    legacy["snapshot_version"] = 1
    legacy.pop("relations", None)
    legacy.pop("relation_counter", None)
    restored = ConversationMemory.from_dict(legacy)
    assert restored.relations == []


def test_relation_is_not_duplicated_for_repeated_statement() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah left because John called.")
        runtime.process("Sarah left because John called.")
        assert len(runtime.memory.relations) == 1
    finally:
        runtime.close()


def test_q_and_a_can_bind_events_from_both_clauses() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah left because John called.")
        subordinate = runtime.process("Who called?")
        assert subordinate.contract.status == AnswerStatus.ANSWERED
        assert "John" in subordinate.response

        main = runtime.process("Who left?")
        assert main.contract.status == AnswerStatus.ANSWERED
        assert "Sarah" in main.response

        why = runtime.process("Why did Sarah leave?")
        assert why.contract.status == AnswerStatus.ANSWERED
        assert "John called" in why.response
        assert len(runtime.memory.relations) == 1
    finally:
        runtime.close()


def test_coordinate_main_clause_can_retain_subordinate_relation() -> None:
    result = parse(
        "Mary opened the door, but Sarah left because John called."
    )
    assert [event.discourse_role for event in result.events] == [
        "main",
        "coordinate",
        "subordinate",
    ]
    relation = result.relations[0]
    assert relation.main_event_index == 1
    assert relation.subordinate_event_index == 2
    assert relation.relation_type == ClauseRelationType.CAUSE
    assert "coordinate connector=but" in result.diagnostics
