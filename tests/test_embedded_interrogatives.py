"""Embedded WH/polar content and question-attribution conformance for issue #89."""

from __future__ import annotations

from itertools import product

import pytest

from clanker_lm.affect import HeuristicAffectBackend
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AnswerStatus,
    EmbeddedInterrogativeRelation,
    EmbeddedInterrogativeStatus,
    EmbeddedInterrogativeType,
    QuestionKind,
    SourceKind,
    SpeechAct,
)
from clanker_lm.parser import SemanticParser
from clanker_lm.runtime import ClankerLM


SUBJECTS = ("Sarah", "David")
MATRIX_FORMS = (
    ("asked", "ask", "past", EmbeddedInterrogativeStatus.ASKED, "questioning"),
    ("asks", "ask", "present", EmbeddedInterrogativeStatus.ASKED, "questioning"),
    ("wondered", "wonder", "past", EmbeddedInterrogativeStatus.WONDERED, "uncertain_cognition"),
    ("wonders", "wonder", "present", EmbeddedInterrogativeStatus.WONDERED, "uncertain_cognition"),
    ("knew", "know", "past", EmbeddedInterrogativeStatus.KNOWN, "knowledge"),
    ("knows", "know", "present", EmbeddedInterrogativeStatus.KNOWN, "knowledge"),
    ("remembered", "remember", "past", EmbeddedInterrogativeStatus.REMEMBERED, "memory"),
    ("remembers", "remember", "present", EmbeddedInterrogativeStatus.REMEMBERED, "memory"),
)
WH_CONTENT = (
    ("who called", "who", QuestionKind.WHO, "call", "agent", ""),
    ("what Mary bought", "what", QuestionKind.WHAT, "buy", "patient", ""),
    ("when the meeting starts", "when", QuestionKind.WHEN, "start", "time", ""),
    ("where John went", "where", QuestionKind.WHERE, "go", "destination", ""),
    ("why Mary left", "why", QuestionKind.WHY, "leave", "motive", ""),
    ("how David opened the box", "how", QuestionKind.HOW, "open", "method", ""),
    ("which car Mary bought", "which", QuestionKind.WHICH, "buy", "patient", "car"),
    ("whose car broke", "whose", QuestionKind.WHOSE, "break", "possessor", "car"),
)
WH_CASES = tuple(product(SUBJECTS, MATRIX_FORMS, WH_CONTENT))

POLAR_MATRIX_FORMS = (
    ("asked", "ask", EmbeddedInterrogativeStatus.ASKED),
    ("wondered", "wonder", EmbeddedInterrogativeStatus.WONDERED),
    ("knew", "know", EmbeddedInterrogativeStatus.KNOWN),
    ("remembered", "remember", EmbeddedInterrogativeStatus.REMEMBERED),
)
POLAR_CONTENT = (
    ("whether Mary left", "whether", "leave"),
    ("if John called", "if", "call"),
    ("whether the store closed", "whether", "close"),
    ("if the car broke", "if", "break"),
)
POLAR_CASES = tuple(product(SUBJECTS, POLAR_MATRIX_FORMS, POLAR_CONTENT))

# 128 WH + 32 polar = 160 generated conformance cases.
assert len(WH_CASES) == 128
assert len(POLAR_CASES) == 32


def parse(text: str):
    memory = ConversationMemory()
    memory.begin_turn()
    result = SemanticParser().parse(text, memory)
    return memory, result


@pytest.mark.parametrize("subject,matrix,content", WH_CASES)
def test_generated_embedded_wh_content(subject, matrix, content) -> None:
    surface, predicate, tense, status, family = matrix
    inner, marker, kind, inner_predicate, requested_role, focus = content
    memory, result = parse(f"{subject} {surface} {inner}.")

    assert result.speech_act == SpeechAct.ASSERT
    assert not result.embedded_interrogative_ambiguities, result.diagnostics
    assert len(result.events) == 2
    assert [event.discourse_role for event in result.events] == ["main", "interrogative"]
    assert result.events[0].predicate == predicate
    assert result.events[0].tense == tense
    assert result.events[0].source == SourceKind.USER
    assert result.events[1].predicate == inner_predicate
    assert result.events[1].source == SourceKind.ATTRIBUTED
    assert len(result.embedded_interrogatives) == 1

    relation = result.embedded_interrogatives[0]
    source = result.events[0].arguments.get("agent") or result.events[0].arguments.get("experiencer")
    assert source is not None
    assert relation.relation_type == EmbeddedInterrogativeType.WH
    assert relation.content_status == status
    assert relation.marker == marker
    assert relation.matrix_predicate == predicate
    assert relation.source_entity_id == source.key
    assert relation.predicate_family == family
    assert relation.question_kind == kind
    assert relation.requested_role == requested_role
    assert relation.focus_surface == focus
    assert relation.licensed
    assert not relation.direct_answer_request
    assert result.events[1].arguments[requested_role].is_variable
    assert memory.get_entity(source.key) is not None


@pytest.mark.parametrize("subject,matrix,content", POLAR_CASES)
def test_generated_embedded_polar_content(subject, matrix, content) -> None:
    surface, predicate, status = matrix
    inner, marker, inner_predicate = content
    _memory, result = parse(f"{subject} {surface} {inner}.")

    assert result.speech_act == SpeechAct.ASSERT
    assert not result.embedded_interrogative_ambiguities, result.diagnostics
    assert len(result.events) == 2
    assert result.events[0].predicate == predicate
    assert result.events[1].predicate == inner_predicate
    assert result.events[1].discourse_role == "interrogative"
    assert result.events[1].source == SourceKind.ATTRIBUTED
    assert not any(ref.is_variable for ref in result.events[1].arguments.values())

    relation = result.embedded_interrogatives[0]
    assert relation.relation_type == EmbeddedInterrogativeType.POLAR
    assert relation.content_status == status
    assert relation.marker == marker
    assert relation.question_kind == QuestionKind.YES_NO
    assert relation.requested_role is None
    assert relation.licensed


def test_question_attribution_does_not_answer_inner_question() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah asked who called.")
        result = runtime.process("Who called?")
        assert result.contract.status == AnswerStatus.UNKNOWN
        assert "don't know" in result.response.lower()
        query = runtime.parser.parse("Who called?", runtime.memory).question.event
        assert runtime.memory.match_events(query) == []
        explicit = runtime.memory.match_events(query, include_interrogative_content=True)
        assert len(explicit) == 1
        assert explicit[0].event.discourse_role == "interrogative"
    finally:
        runtime.close()


def test_source_query_preserves_inner_question_in_surface() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah asked who called.")
        result = runtime.process("Who asked who called?")
        assert result.contract.status == AnswerStatus.ANSWERED
        assert result.response == "Sarah asked who called."
        assert result.contract.required_slots["embedded_interrogative"] == "true"
    finally:
        runtime.close()


def test_content_query_preserves_attribution_without_asserting_answer() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah asked who called.")
        result = runtime.process("What did Sarah ask?")
        assert result.contract.status == AnswerStatus.ANSWERED
        assert result.response == "Sarah asked who called."
        assert "called" in result.response.lower()
    finally:
        runtime.close()


def test_polar_attribution_realizes_full_matrix_and_inner_content() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("John wondered whether Mary left.")
        result = runtime.process("Did John wonder whether Mary left?")
        assert result.contract.status == AnswerStatus.TRUE
        assert result.response == "Yes. John wondered whether Mary left."
        inner = runtime.process("Did Mary leave?")
        assert inner.contract.status == AnswerStatus.UNKNOWN
    finally:
        runtime.close()


def test_negated_matrix_yields_false_without_inverting_inner_truth() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah did not ask whether Mary left.")
        result = runtime.process("Did Sarah ask whether Mary left?")
        assert result.contract.status == AnswerStatus.FALSE
        assert result.response == "No. Sarah did not ask whether Mary left."
        inner = runtime.process("Did Mary leave?")
        assert inner.contract.status == AnswerStatus.UNKNOWN
    finally:
        runtime.close()


def test_matrix_and_embedded_negation_remain_separate() -> None:
    _memory, result = parse("Sarah asked whether Mary did not leave.")
    relation = result.embedded_interrogatives[0]
    assert result.events[0].polarity is True
    assert result.events[1].polarity is False
    assert relation.licensed is True

    _memory, result = parse("Sarah did not ask whether Mary left.")
    relation = result.embedded_interrogatives[0]
    assert result.events[0].polarity is False
    assert result.events[1].polarity is True
    assert relation.licensed is False


def test_conflicting_question_attributions_are_rendered_as_relations() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah asked who called.")
        runtime.process("Sarah did not ask who called.")
        result = runtime.process("Did Sarah ask who called?")
        assert result.contract.status == AnswerStatus.CONFLICT
        assert "conflicting information" in result.response.lower()
        assert "asked who called" in result.response.lower()
        assert "did not ask who called" in result.response.lower()
    finally:
        runtime.close()


def test_multiple_sources_are_not_silently_collapsed() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah asked who called.")
        runtime.process("David asked who called.")
        result = runtime.process("Who asked who called?")
        assert result.contract.status == AnswerStatus.ANSWERED
        assert "Sarah" in result.response
        assert "David" in result.response
        assert len(result.contract.values) == 2
    finally:
        runtime.close()


def test_recipient_is_preserved_before_embedded_question() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah asked John who called.")
        result = runtime.process("Did Sarah ask John who called?")
        assert result.contract.status == AnswerStatus.TRUE
        assert "Sarah asked John who called" in result.response
    finally:
        runtime.close()


def test_if_and_whether_are_equivalent_for_matching_but_preserve_evidence_surface() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("John wondered if Mary left.")
        result = runtime.process("Did John wonder whether Mary left?")
        assert result.contract.status == AnswerStatus.TRUE
        assert "John wondered if Mary left" in result.response
    finally:
        runtime.close()


def test_direct_embedded_answer_request_returns_fact_when_known() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("John went to Chicago.")
        result = runtime.process("Tell me where John went.")
        assert result.contract.status == AnswerStatus.ANSWERED
        assert result.response == "John went to Chicago."
        assert result.contract.required_slots["direct_embedded_answer_request"] == "true"
    finally:
        runtime.close()


def test_direct_embedded_answer_request_is_explicit_when_unknown() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        result = runtime.process("Tell me where John went.")
        assert result.contract.status == AnswerStatus.UNKNOWN
        assert result.response == "I don't know where John went."
    finally:
        runtime.close()


def test_outer_epistemic_question_remains_polar_and_returns_inner_fact() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("The meeting starts at 3 PM.")
        parsed = runtime.parser.parse(
            "Do you remember when the meeting starts?",
            runtime.memory,
        )
        assert parsed.question.kind == QuestionKind.YES_NO
        assert parsed.question.embedded_question.kind == QuestionKind.WHEN

        result = runtime.process("Do you remember when the meeting starts?")
        assert result.contract.status == AnswerStatus.TRUE
        assert result.response.startswith("Yes.")
        assert "3 PM" in result.response
    finally:
        runtime.close()


def test_outer_epistemic_question_unknown_does_not_claim_memory() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        result = runtime.process("Do you remember when the meeting starts?")
        assert result.contract.status == AnswerStatus.FALSE
        assert result.response.startswith("No.")
        assert "do not know when" in result.response.lower()
    finally:
        runtime.close()


def test_outer_reported_question_and_inner_question_remain_distinct() -> None:
    _memory, result = parse("Who asked who called?")
    assert result.speech_act == SpeechAct.ASK
    assert result.question.kind == QuestionKind.WHO
    assert result.question.embedded_question is not None
    assert result.question.embedded_question.kind == QuestionKind.WHO
    assert result.question.event.predicate == "ask"
    assert result.question.embedded_question.event.predicate == "call"


def test_relative_marker_after_object_is_not_stolen_as_question_content() -> None:
    _memory, result = parse("I know the man who called.")
    assert not result.embedded_interrogatives
    assert result.embedded_interrogative_ambiguities == []


def test_nested_embedded_interrogatives_fail_closed() -> None:
    _memory, result = parse("Sarah asked who wondered whether Mary left.")
    assert not result.events
    assert not result.embedded_interrogatives
    assert result.embedded_interrogative_ambiguities
    assert result.unresolved
    assert "configured depth" in result.embedded_interrogative_ambiguities[0].reason


def test_relation_round_trip_and_memory_snapshot_version_six() -> None:
    relation = EmbeddedInterrogativeRelation(
        relation_type=EmbeddedInterrogativeType.WH,
        content_status=EmbeddedInterrogativeStatus.ASKED,
        matrix_event_index=0,
        question_event_index=1,
        marker="who",
        matrix_predicate="ask",
        source_entity_id="sarah_1",
        predicate_family="questioning",
        question_kind=QuestionKind.WHO,
        requested_role="agent",
    )
    assert EmbeddedInterrogativeRelation.from_dict(relation.to_dict()).to_dict() == relation.to_dict()

    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah asked who called.")
        snapshot = runtime.dumps(indent=None)
    finally:
        runtime.close()

    restored = ClankerLM.loads(snapshot, affect_backend=HeuristicAffectBackend())
    try:
        assert restored.SNAPSHOT_VERSION == 6
        assert restored.memory.SNAPSHOT_VERSION == 6
        assert len(restored.memory.embedded_interrogatives) == 1
        stored = restored.memory.embedded_interrogatives[0]
        assert stored.relation_id
        assert restored.memory.get_event(stored.matrix_event_id) is not None
        assert restored.memory.get_event(stored.question_event_id).discourse_role == "interrogative"
        answer = restored.process("Who asked who called?")
        assert answer.response == "Sarah asked who called."
    finally:
        restored.close()


def test_trusted_learning_preserves_question_attribution_and_nonassertion() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        stored = runtime.learn("Sarah asked who called.", source=SourceKind.RETRIEVED)
        assert stored[0].source == SourceKind.RETRIEVED
        assert stored[1].source == SourceKind.ATTRIBUTED
        assert len(runtime.memory.embedded_interrogatives) == 1
        attribution = runtime.process("Who asked who called?")
        assert attribution.contract.status == AnswerStatus.ANSWERED
        inner = runtime.process("Who called?")
        assert inner.contract.status == AnswerStatus.UNKNOWN
    finally:
        runtime.close()


def test_version_four_snapshot_migrates_with_empty_embedded_relation_store() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        snapshot = runtime.to_dict()
    finally:
        runtime.close()
    snapshot["snapshot_version"] = 4
    snapshot["memory"]["snapshot_version"] = 4
    snapshot["memory"].pop("embedded_interrogatives", None)
    snapshot["memory"].pop("embedded_interrogative_counter", None)

    restored = ClankerLM.from_dict(
        snapshot,
        affect_backend=HeuristicAffectBackend(),
    )
    try:
        assert restored.memory.embedded_interrogatives == []
        assert restored.memory.SNAPSHOT_VERSION == 6
    finally:
        restored.close()


def test_matrix_recipient_must_match_exactly() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah asked John who called.")
        mismatch = runtime.process("Did Sarah ask Mary who called?")
        assert mismatch.contract.status == AnswerStatus.UNKNOWN
        exact = runtime.process("Did Sarah ask John who called?")
        assert exact.contract.status == AnswerStatus.TRUE
    finally:
        runtime.close()
