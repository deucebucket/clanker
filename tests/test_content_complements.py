"""Finite attributed-content complement conformance for issue #86."""

from __future__ import annotations

from itertools import product

import pytest

from clanker_lm.affect import HeuristicAffectBackend
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AnswerStatus,
    ContentRelation,
    ContentRelationType,
    SourceKind,
)
from clanker_lm.parser import SemanticParser
from clanker_lm.runtime import ClankerLM


MATRIX_SUBJECTS = ("Sarah", "David")
MATRIX_PREDICATES = (
    ("said", "say", ContentRelationType.REPORTED, "speech"),
    ("reported", "report", ContentRelationType.REPORTED, "speech"),
    ("claimed", "claim", ContentRelationType.REPORTED, "speech"),
    ("thought", "think", ContentRelationType.BELIEVED, "belief"),
    ("believed", "believe", ContentRelationType.BELIEVED, "belief"),
    ("knew", "know", ContentRelationType.KNOWN, "knowledge"),
    ("noticed", "notice", ContentRelationType.PERCEIVED, "perception"),
    ("heard", "hear", ContentRelationType.PERCEIVED, "perception"),
)
CONTENT_CLAUSES = (
    ("John left", "leave"),
    ("Mary arrived", "arrive"),
    ("the car broke", "break"),
    ("the store closed", "close"),
    ("the dog died", "die"),
)
MARKERS = (("that ", "that"), ("", "zero"))
CONTENT_CASES = tuple(
    product(MATRIX_SUBJECTS, MATRIX_PREDICATES, CONTENT_CLAUSES, MARKERS)
)


def parse(text: str):
    memory = ConversationMemory()
    memory.begin_turn()
    result = SemanticParser().parse(text, memory)
    return memory, result


@pytest.mark.parametrize("subject,matrix,content,marker", CONTENT_CASES)
def test_generated_finite_content_complements(subject, matrix, content, marker) -> None:
    matrix_surface, matrix_predicate, relation_type, family = matrix
    content_surface, content_predicate = content
    marker_surface, marker_name = marker
    memory, result = parse(
        f"{subject} {matrix_surface} {marker_surface}{content_surface}."
    )

    assert [event.predicate for event in result.events] == [
        matrix_predicate,
        content_predicate,
    ], result.diagnostics
    assert [event.discourse_role for event in result.events] == ["main", "content"]
    assert result.events[0].source == SourceKind.USER
    assert result.events[1].source == SourceKind.ATTRIBUTED
    assert not result.modifiers
    assert len(result.contents) == 1
    relation = result.contents[0]
    assert relation.relation_type == relation_type
    assert relation.marker == marker_name
    assert relation.matrix_predicate == matrix_predicate
    assert relation.predicate_family == family
    assert relation.source_entity_id == result.events[0].arguments["agent"].key
    assert memory.get_entity(relation.source_entity_id) is not None


def test_tell_preserves_recipient_before_explicit_content() -> None:
    _memory, result = parse("Sarah told me that John left.")
    assert [event.predicate for event in result.events] == ["tell", "leave"]
    assert result.contents[0].marker == "that"
    assert result.events[0].arguments["patient"].surface.lower() == "me"
    assert not result.modifiers


def test_tell_preserves_recipient_before_zero_content() -> None:
    _memory, result = parse("Sarah told me John left.")
    assert [event.predicate for event in result.events] == ["tell", "leave"]
    assert result.contents[0].marker == "zero"
    assert result.events[0].arguments["patient"].surface.lower() == "me"


def test_relative_that_after_direct_object_is_not_stolen() -> None:
    _memory, result = parse("Sarah noticed the man that left.")
    assert not result.contents
    assert len(result.modifiers) == 1
    assert [event.discourse_role for event in result.events] == ["main", "modifier"]


def test_abstract_direct_object_that_relative_is_not_content() -> None:
    _memory, result = parse("Sarah reported the idea that changed everything.")
    assert not result.contents
    assert len(result.events) == 1
    assert result.events[0].predicate == "report"


def test_ordinary_direct_object_is_not_promoted_to_content() -> None:
    _memory, result = parse("Sarah believes the report.")
    assert not result.contents
    assert len(result.events) == 1
    assert result.events[0].arguments["patient"].surface == "the report"


def test_nested_content_fails_explicitly_at_configured_depth() -> None:
    _memory, result = parse("Sarah thinks John said Mary left.")
    assert not result.events
    assert not result.contents
    assert result.content_ambiguities
    assert "nested finite content" in result.content_ambiguities[0].reason
    assert result.unresolved


def test_unresolved_content_source_suppresses_storage() -> None:
    _memory, result = parse("She said that John left.")
    assert not result.events
    assert result.content_ambiguities
    assert any("source" in item.reason for item in result.content_ambiguities)


def test_content_relation_round_trip_and_memory_snapshot() -> None:
    relation = ContentRelation(
        relation_type=ContentRelationType.REPORTED,
        matrix_event_index=0,
        content_event_index=1,
        marker="that",
        matrix_predicate="say",
        source_entity_id="sarah_1",
        predicate_family="speech",
    )
    assert ContentRelation.from_dict(relation.to_dict()).to_dict() == relation.to_dict()

    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah said that John left.")
        snapshot = runtime.memory.dumps(indent=None)
    finally:
        runtime.close()
    restored = ConversationMemory.loads(snapshot)
    assert len(restored.contents) == 1
    stored = restored.contents[0]
    assert stored.relation_id
    assert stored.matrix_event_id
    assert stored.content_event_id
    assert restored.get_event(stored.content_event_id).source == SourceKind.ATTRIBUTED


def test_content_is_excluded_from_unqualified_event_matching() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah said that John left.")
        query = runtime.parser.parse("Did John leave?", runtime.memory).question.event
        assert runtime.memory.match_events(query) == []
        attributed = runtime.memory.match_events(
            query,
            include_attributed_content=True,
        )
        assert len(attributed) == 1
        assert attributed[0].event.discourse_role == "content"
    finally:
        runtime.close()


def test_what_did_source_say_reads_typed_content_relation() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah said that John left.")
        answer = runtime.process("What did Sarah say?")
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert answer.contract.source == SourceKind.ATTRIBUTED
        assert answer.contract.required_slots["source_entity_id"]
        assert "sarah" in answer.response.lower()
        assert "john left" in answer.response.lower()
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,verb",
    (
        ("Sarah thought John left.", "What did Sarah think?", "thought"),
        ("Sarah believed John left.", "What did Sarah believe?", "believed"),
        ("Sarah knew John left.", "What did Sarah know?", "knew"),
        ("Sarah noticed John left.", "What did Sarah notice?", "noticed"),
        ("Sarah heard John left.", "What did Sarah hear?", "heard"),
        ("Sarah reported John left.", "What did Sarah report?", "reported"),
    ),
)
def test_attributed_content_query_preserves_matrix_predicate(statement, question, verb) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert verb in answer.response.lower()
        assert "john left" in answer.response.lower()
    finally:
        runtime.close()


def test_direct_truth_question_qualifies_attributed_only_evidence() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah said that John left.")
        answer = runtime.process("Did John leave?")
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.source == SourceKind.ATTRIBUTED
        assert "promote_attributed_content_to_unqualified_fact" in answer.contract.forbidden_claims
        assert "sarah said" in answer.response.lower()
        assert "don't know whether john left" in answer.response.lower()
    finally:
        runtime.close()


def test_direct_user_fact_outranks_conflicting_attributed_content() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah said that John did not leave.")
        runtime.process("John left.")
        answer = runtime.process("Did John leave?")
        assert answer.contract.status == AnswerStatus.TRUE
        assert answer.contract.source == SourceKind.USER
    finally:
        runtime.close()


def test_contradictory_speakers_remain_separate_attributed_sources() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah said that John left.")
        runtime.process("Mary said that John did not leave.")
        assert len(runtime.memory.contents) == 2
        assert len({item.source_entity_id for item in runtime.memory.contents}) == 2

        answer = runtime.process("Did John leave?")
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.reason == "attributed sources disagree about the proposition"
        assert len(answer.contract.evidence) == 2
        assert "sarah" in answer.response.lower()
        assert "mary" in answer.response.lower()
    finally:
        runtime.close()


def test_parse_result_serializes_content_trace() -> None:
    _memory, result = parse("Sarah said that John left.")
    payload = result.to_dict()
    assert payload["contents"][0]["relation_type"] == "reported"
    assert payload["content_ambiguities"] == []


def test_repeated_content_relation_deduplicates() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah said that John left.")
        runtime.process("Sarah said that John left.")
        assert len(runtime.memory.contents) == 1
    finally:
        runtime.close()


def test_legacy_snapshot_without_content_fields_remains_loadable() -> None:
    memory = ConversationMemory()
    legacy = memory.to_dict()
    legacy.pop("contents", None)
    legacy.pop("content_counter", None)
    restored = ConversationMemory.from_dict(legacy)
    assert restored.contents == []


def test_matching_direct_assertion_is_not_deduplicated_into_attributed_content() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah said that John left.")
        runtime.process("John left.")
        john_events = [
            event
            for event in runtime.memory.events
            if event.predicate == "leave" and event.polarity
        ]
        assert {event.discourse_role for event in john_events} == {"main", "content"}
        answer = runtime.process("Did John leave?")
        assert answer.contract.status == AnswerStatus.TRUE
        assert answer.contract.source == SourceKind.USER
    finally:
        runtime.close()


def test_negated_matrix_does_not_license_content_as_evidence() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        parsed = runtime.parser.parse(
            "Sarah did not say that John left.",
            runtime.memory,
        )
        assert len(parsed.contents) == 1
        assert parsed.events[0].polarity is False
        assert parsed.contents[0].attributed is False

        runtime.process("Sarah did not say that John left.")
        direct = runtime.process("Did John leave?")
        assert direct.contract.status == AnswerStatus.UNKNOWN
        assert direct.contract.source == SourceKind.UNKNOWN
        assert direct.contract.evidence == []

        content = runtime.process("What did Sarah say?")
        assert content.contract.status == AnswerStatus.UNKNOWN
    finally:
        runtime.close()

@pytest.mark.parametrize(
    "text,reason",
    (
        (
            "The man who said that John left called Mary.",
            "relative clause exceeds this parser slice",
        ),
        (
            "The man who left said that John arrived.",
            "relative clause exceeds this parser slice",
        ),
        (
            "Because Mary called, Sarah said that John left.",
            "subordinate clause requires staged parsing",
        ),
        (
            "Sarah, my supervisor, said that John left.",
            "appositive requires staged parsing",
        ),
        (
            "Sarah said that John left because Mary called.",
            "subordinate clause requires staged parsing",
        ),
        (
            "Sarah said that the man who called left.",
            "relative clause exceeds this parser slice",
        ),
    ),
)
def test_content_combined_with_other_relation_layers_fails_closed(
    text: str,
    reason: str,
) -> None:
    _memory, result = parse(text)
    assert not result.events
    assert not result.contents
    assert result.content_ambiguities
    assert reason in result.content_ambiguities[0].reason
    assert result.unresolved
    assert "unsafe attributed content suppressed" in result.diagnostics
