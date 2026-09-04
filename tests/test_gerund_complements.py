"""Gerund, participial, and aspectual conformance for issue #90."""

from __future__ import annotations

import json
from itertools import product

import pytest

from clanker_lm.affect import HeuristicAffectBackend
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AnswerStatus,
    GerundAttachmentAmbiguity,
    GerundContentStatus,
    GerundRelation,
    GerundRelationType,
    SourceKind,
)
from clanker_lm.parser import SemanticParser
from clanker_lm.runtime import ClankerLM


SUBJECTS = ("Sarah", "David")
SUBJECT_CONTROL_FORMS = (
    (
        "enjoys",
        "enjoy",
        "present",
        GerundRelationType.GERUND_CONTENT,
        GerundContentStatus.ENJOYED,
        False,
    ),
    (
        "enjoyed",
        "enjoy",
        "past",
        GerundRelationType.GERUND_CONTENT,
        GerundContentStatus.ENJOYED,
        False,
    ),
    (
        "avoids",
        "avoid",
        "present",
        GerundRelationType.GERUND_CONTENT,
        GerundContentStatus.AVOIDED,
        False,
    ),
    (
        "avoided",
        "avoid",
        "past",
        GerundRelationType.GERUND_CONTENT,
        GerundContentStatus.AVOIDED,
        False,
    ),
    (
        "starts",
        "start",
        "present",
        GerundRelationType.ASPECTUAL_START,
        GerundContentStatus.BEGUN,
        True,
    ),
    (
        "started",
        "start",
        "past",
        GerundRelationType.ASPECTUAL_START,
        GerundContentStatus.BEGUN,
        True,
    ),
    (
        "stops",
        "stop",
        "present",
        GerundRelationType.ASPECTUAL_STOP,
        GerundContentStatus.STOPPED,
        True,
    ),
    (
        "stopped",
        "stop",
        "past",
        GerundRelationType.ASPECTUAL_STOP,
        GerundContentStatus.STOPPED,
        True,
    ),
    (
        "continues",
        "continue",
        "present",
        GerundRelationType.ASPECTUAL_CONTINUATION,
        GerundContentStatus.CONTINUED,
        True,
    ),
    (
        "continued",
        "continue",
        "past",
        GerundRelationType.ASPECTUAL_CONTINUATION,
        GerundContentStatus.CONTINUED,
        True,
    ),
    (
        "keeps",
        "keep",
        "present",
        GerundRelationType.ASPECTUAL_CONTINUATION,
        GerundContentStatus.CONTINUED,
        True,
    ),
    (
        "kept",
        "keep",
        "past",
        GerundRelationType.ASPECTUAL_CONTINUATION,
        GerundContentStatus.CONTINUED,
        True,
    ),
)
SUBJECT_CONTROL_COMPLEMENTS = (
    ("reading", "read"),
    ("calling Mary", "call"),
    ("buying groceries", "buy"),
    ("opening the door", "open"),
    ("helping John", "help"),
)
SUBJECT_CONTROL_CASES = tuple(
    product(SUBJECTS, SUBJECT_CONTROL_FORMS, SUBJECT_CONTROL_COMPLEMENTS)
)

PERCEPTION_FORMS = (
    ("sees", "see", "present"),
    ("saw", "see", "past"),
)
PERCEPTION_CONTROLLERS = ("John", "Mary")
PERCEPTION_COMPLEMENTS = (
    ("leaving", "leave"),
    ("calling David", "call"),
    ("buying groceries", "buy"),
    ("opening the door", "open"),
    ("talking", "talk"),
)
PERCEPTION_CASES = tuple(
    product(
        SUBJECTS,
        PERCEPTION_FORMS,
        PERCEPTION_CONTROLLERS,
        PERCEPTION_COMPLEMENTS,
    )
)

# 120 subject-controlled + 40 perception/object-controlled = 160 generated
# conformance cases. Focused tests below are deliberately not counted here.
assert len(SUBJECT_CONTROL_CASES) == 120
assert len(PERCEPTION_CASES) == 40
assert len(SUBJECT_CONTROL_CASES) + len(PERCEPTION_CASES) == 160


def parse(text: str):
    memory = ConversationMemory()
    memory.begin_turn()
    result = SemanticParser().parse(text, memory)
    return memory, result


def entity_argument(
    event,
    roles=("agent", "subject", "experiencer", "possessor"),
):
    for role in roles:
        ref = event.arguments.get(role)
        if ref is not None:
            return ref
    raise AssertionError(f"No entity role in event {event.to_dict()}")


def durable_memory_signature(memory: ConversationMemory):
    """Return durable semantic state, excluding turn/salience bookkeeping."""

    payload = memory.to_dict()
    counters = (
        "revision",
        "entity_counter",
        "event_counter",
        "relation_counter",
        "modifier_counter",
        "appositive_counter",
        "content_counter",
        "embedded_interrogative_counter",
        "infinitival_counter",
        "gerund_counter",
    )
    stores = (
        "events",
        "relations",
        "modifiers",
        "appositives",
        "contents",
        "embedded_interrogatives",
        "infinitivals",
        "gerunds",
    )
    return {
        **{key: payload[key] for key in counters},
        **{key: payload[key] for key in stores},
        "entity_ids": tuple(item["entity_id"] for item in payload["entities"]),
        "entity_aliases": tuple(
            (item["entity_id"], tuple(item["aliases"]))
            for item in payload["entities"]
        ),
    }


def test_parser_and_memory_gerund_catalogs_have_semantic_parity() -> None:
    parser_catalog = {
        predicate: (
            profile.relation_type,
            profile.content_status,
            profile.phase_entailing,
        )
        for predicate, profile in SemanticParser.GERUND_PREDICATES.items()
    }
    memory_catalog = ConversationMemory._GERUND_PREDICATE_CATALOG

    assert parser_catalog.keys() == memory_catalog.keys()
    assert parser_catalog == memory_catalog


@pytest.mark.parametrize(
    "subject,matrix,complement",
    SUBJECT_CONTROL_CASES,
)
def test_generated_subject_controlled_gerunds(
    subject,
    matrix,
    complement,
) -> None:
    (
        matrix_surface,
        matrix_predicate,
        matrix_tense,
        relation_type,
        content_status,
        entailed,
    ) = matrix
    complement_surface, complement_predicate = complement
    memory, result = parse(
        f"{subject} {matrix_surface} {complement_surface}."
    )

    assert not result.gerund_ambiguities, result.diagnostics
    assert [event.predicate for event in result.events] == [
        matrix_predicate,
        complement_predicate,
    ]
    assert [event.discourse_role for event in result.events] == ["main", "gerund"]
    assert result.events[0].tense == matrix_tense
    assert result.events[0].aspect == "simple"
    assert result.events[0].source == SourceKind.USER
    assert result.events[1].tense == "nonfinite"
    assert result.events[1].aspect == "gerund"
    assert result.events[1].source == SourceKind.ATTRIBUTED
    assert len(result.gerunds) == 1

    relation = result.gerunds[0]
    matrix_source = entity_argument(result.events[0])
    embedded_subject = entity_argument(result.events[1])
    assert relation.relation_type == relation_type
    assert relation.content_status == content_status
    assert relation.matrix_event_index == 0
    assert relation.complement_event_index == 1
    assert relation.marker == "-ing"
    assert relation.matrix_predicate == matrix_predicate
    assert relation.predicate_family
    assert relation.source_entity_id == matrix_source.key
    assert relation.controller_entity_id == matrix_source.key
    assert relation.embedded_subject_entity_id == embedded_subject.key
    assert embedded_subject.key == matrix_source.key
    assert relation.licensed
    assert relation.entailed is entailed
    assert memory.get_entity(matrix_source.key) is not None


@pytest.mark.parametrize(
    "perceiver,matrix,controller,complement",
    PERCEPTION_CASES,
)
def test_generated_perception_participials(
    perceiver,
    matrix,
    controller,
    complement,
) -> None:
    matrix_surface, matrix_predicate, matrix_tense = matrix
    complement_surface, complement_predicate = complement
    memory, result = parse(
        f"{perceiver} {matrix_surface} {controller} {complement_surface}."
    )

    assert not result.gerund_ambiguities, result.diagnostics
    assert [event.predicate for event in result.events] == [
        matrix_predicate,
        complement_predicate,
    ]
    assert [event.discourse_role for event in result.events] == [
        "main",
        "participle",
    ]
    assert result.events[0].tense == matrix_tense
    assert result.events[0].source == SourceKind.USER
    assert result.events[1].tense == "nonfinite"
    assert result.events[1].aspect == "participle"
    assert result.events[1].source == SourceKind.ATTRIBUTED
    assert len(result.gerunds) == 1

    relation = result.gerunds[0]
    matrix_source = entity_argument(result.events[0])
    matrix_controller = entity_argument(
        result.events[0],
        roles=("patient", "recipient"),
    )
    embedded_subject = entity_argument(result.events[1])
    assert relation.relation_type == GerundRelationType.PERCEPTION_PARTICIPIAL
    assert relation.content_status == GerundContentStatus.PERCEIVED
    assert relation.matrix_event_index == 0
    assert relation.complement_event_index == 1
    assert relation.marker == "-ing"
    assert relation.matrix_predicate == matrix_predicate
    assert relation.predicate_family
    assert relation.source_entity_id == matrix_source.key
    assert relation.controller_entity_id == matrix_controller.key
    assert relation.embedded_subject_entity_id == embedded_subject.key
    assert embedded_subject.key == matrix_controller.key
    assert relation.controller_entity_id != relation.source_entity_id
    assert relation.licensed
    assert not relation.entailed
    assert memory.get_entity(matrix_controller.key).canonical_name.lower() == (
        controller.lower()
    )


@pytest.mark.parametrize(
    "statement,matrix_question,matrix_predicate",
    (
        ("Sarah heard John leaving.", "Did Sarah hear John leaving?", "hear"),
        ("Sarah watched John leaving.", "Did Sarah watch John leaving?", "watch"),
        (
            "Sarah noticed John leaving.",
            "Did Sarah notice John leaving?",
            "notice",
        ),
    ),
)
def test_reviewed_perception_catalog_entries_parse_and_answer_end_to_end(
    statement,
    matrix_question,
    matrix_predicate,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        assertion = runtime.process(statement)
        assert [event.predicate for event in assertion.parse.events] == [
            matrix_predicate,
            "leave",
        ]
        assert assertion.parse.events[1].discourse_role == "participle"
        relation = assertion.parse.gerunds[0]
        assert relation.relation_type == GerundRelationType.PERCEPTION_PARTICIPIAL
        assert relation.content_status == GerundContentStatus.PERCEIVED
        assert relation.controller_entity_id == relation.embedded_subject_entity_id
        assert not relation.entailed

        matrix = runtime.process(matrix_question)
        assert matrix.contract.status == AnswerStatus.TRUE
        assert matrix.contract.required_slots["gerund"] == "true"
        assert matrix.contract.required_slots["matrix_predicate"] == matrix_predicate

        embedded = runtime.process("Did John leave?")
        assert embedded.contract.status == AnswerStatus.UNKNOWN
        assert embedded.contract.source == SourceKind.ATTRIBUTED
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "text,predicate,tense,status",
    (
        ("Sarah begins working.", "begin", "present", GerundContentStatus.BEGUN),
        ("Sarah began working.", "begin", "past", GerundContentStatus.BEGUN),
        (
            "David continues talking.",
            "continue",
            "present",
            GerundContentStatus.CONTINUED,
        ),
        (
            "David continued talking.",
            "continue",
            "past",
            GerundContentStatus.CONTINUED,
        ),
    ),
)
def test_begin_and_continue_catalog_entries_align_end_to_end(
    text,
    predicate,
    tense,
    status,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        result = runtime.process(text)
        assert [event.discourse_role for event in result.parse.events] == [
            "main",
            "gerund",
        ]
        relation = result.parse.gerunds[0]
        assert relation.matrix_predicate == predicate
        assert relation.content_status == status
        assert relation.licensed
        assert relation.entailed
        stored = runtime.memory.gerunds[0]
        assert stored.relation_id == "gerund_1"
        assert runtime.memory.get_event(stored.matrix_event_id).tense == tense
        assert runtime.memory.get_event(stored.complement_event_id).predicate in {
            "work",
            "talk",
        }
    finally:
        runtime.close()


def test_contextual_plural_controller_supports_keep_relation_and_qa() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah and John arrived.")
        assertion = runtime.process("They kept talking.")
        assert [event.predicate for event in assertion.parse.events] == [
            "keep",
            "talk",
        ]
        relation = assertion.parse.gerunds[0]
        assert relation.relation_type == GerundRelationType.ASPECTUAL_CONTINUATION
        assert relation.content_status == GerundContentStatus.CONTINUED
        assert relation.controller_entity_id == relation.embedded_subject_entity_id
        controller = runtime.memory.get_entity(relation.controller_entity_id)
        assert controller is not None
        assert controller.canonical_name.lower() == "sarah and john"

        content = runtime.process("What did they keep doing?")
        assert content.contract.status == AnswerStatus.ANSWERED
        assert content.contract.required_slots["gerund"] == "true"
        assert "sarah and john kept talking" in content.response.lower()

        activity = runtime.process("Did they talk?")
        assert activity.contract.status == AnswerStatus.TRUE
        assert activity.contract.source == SourceKind.INFERRED
        assert "sarah and john kept talking" in activity.response.lower()
    finally:
        runtime.close()


def test_standalone_plural_controller_fails_closed() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        before = durable_memory_signature(runtime.memory)
        result = runtime.process("They kept talking.")
        assert result.contract.status == AnswerStatus.MISSING_REFERENCE
        assert not result.parse.events
        assert not result.parse.gerunds
        assert result.parse.unresolved
        assert durable_memory_signature(runtime.memory) == before
        assert set(runtime.memory.entities) == {"user", "assistant"}
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "text,predicate,tense",
    (
        ("Sarah is reading.", "read", "present"),
        ("Sarah was reading.", "read", "past"),
        ("David is calling Mary.", "call", "present"),
    ),
)
def test_progressive_aspect_is_not_a_selected_gerund_complement(
    text,
    predicate,
    tense,
) -> None:
    _memory, result = parse(text)
    assert not result.gerunds
    assert not result.gerund_ambiguities
    assert len(result.events) == 1
    assert result.events[0].predicate == predicate
    assert result.events[0].tense == tense
    assert result.events[0].aspect == "progressive"
    assert result.events[0].discourse_role == "main"
    assert result.events[0].source == SourceKind.USER


def test_selected_gerund_remains_distinct_from_matrix_progressive() -> None:
    _memory, result = parse("Sarah enjoys reading.")
    assert [event.aspect for event in result.events] == ["simple", "gerund"]
    assert [event.predicate for event in result.events] == ["enjoy", "read"]
    assert len(result.gerunds) == 1


@pytest.mark.parametrize(
    "statement,question,stored_aspect,query_aspect",
    (
        ("Sarah reads.", "Is Sarah reading?", "simple", "progressive"),
        ("Sarah is reading.", "Does Sarah read?", "progressive", "simple"),
    ),
)
def test_simple_and_progressive_events_do_not_directly_match(
    statement,
    question,
    stored_aspect,
    query_aspect,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        query_parse = runtime.parser.parse(question, runtime.memory)
        assert runtime.memory.events[0].aspect == stored_aspect
        assert query_parse.question.event.aspect == query_aspect
        assert runtime.memory.match_events(query_parse.question.event) == []

        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.reason != "direct proposition match"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "text,expected_predicate",
    (
        ("The meeting is at three.", "be"),
        ("Reading is fun.", "be"),
        ("Sarah enjoys the painting.", "enjoy"),
        ("Sarah likes painting.", "like"),
        ("Sarah owns a painting.", "own"),
    ),
)
def test_nominal_ing_forms_do_not_become_selected_events(
    text,
    expected_predicate,
) -> None:
    _memory, result = parse(text)
    assert not result.gerunds
    assert not result.gerund_ambiguities
    assert len(result.events) == 1
    assert result.events[0].predicate == expected_predicate
    assert result.events[0].discourse_role == "main"


@pytest.mark.parametrize(
    "text",
    (
        "Walking home, Sarah called John.",
        "Sarah called John, walking home.",
        "Reading quietly, David waited.",
    ),
)
def test_free_ing_adjuncts_are_not_selected_as_gerund_content(text) -> None:
    _memory, result = parse(text)
    assert not result.events
    assert not result.gerunds
    assert result.gerund_ambiguities
    reason = result.gerund_ambiguities[0].reason.lower()
    assert "free adjunct" in reason
    assert "staged" in reason
    assert result.unresolved


@pytest.mark.parametrize(
    "text,reason_fragment",
    (
        ("Sarah saw leaving.", "controller"),
        ("Sarah enjoys Mary reading.", "controller"),
        ("Sarah enjoys reading and writing.", "coordination"),
        ("Sarah started enjoying reading.", "staged"),
    ),
)
def test_unsafe_gerund_boundaries_fail_closed_with_typed_ambiguity(
    text,
    reason_fragment,
) -> None:
    _memory, result = parse(text)
    assert not result.events
    assert not result.gerunds
    assert result.gerund_ambiguities
    ambiguity = result.gerund_ambiguities[0]
    assert reason_fragment in ambiguity.reason.lower()
    assert ambiguity.clause_surface
    assert ambiguity.candidate_boundaries
    assert result.unresolved


@pytest.mark.parametrize(
    "text",
    (
        "The woman who called me enjoys reading.",
        "Sarah, my supervisor, enjoys reading.",
        "Sarah enjoys reading because John called.",
    ),
)
def test_unstaged_relation_layers_with_gerunds_fail_closed(text) -> None:
    _memory, result = parse(text)
    assert not result.events
    assert not result.gerunds
    assert result.gerund_ambiguities
    assert "staged" in result.gerund_ambiguities[0].reason.lower()
    assert result.unresolved


def test_unsupported_feel_ing_fails_without_partial_memory_mutation() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        before = durable_memory_signature(runtime.memory)
        result = runtime.process("Sarah felt John leaving.")
        assert not result.parse.events
        assert not result.parse.gerunds
        assert result.parse.gerund_ambiguities
        assert result.parse.unresolved
        assert durable_memory_signature(runtime.memory) == before
    finally:
        runtime.close()


def test_missing_perception_controller_fails_atomically() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        before = durable_memory_signature(runtime.memory)
        result = runtime.process("Sarah saw leaving.")
        assert not result.parse.events
        assert not result.parse.gerunds
        assert result.parse.gerund_ambiguities
        assert "controller" in result.parse.gerund_ambiguities[0].reason.lower()
        assert durable_memory_signature(runtime.memory) == before
    finally:
        runtime.close()


def test_late_unresolved_gerund_controller_failure_is_atomic() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        before = durable_memory_signature(runtime.memory)
        before_turn = runtime.memory.turn_index
        failed = runtime.process("Sarah enjoyed calling them.")
        assert not failed.parse.events
        assert not failed.parse.gerunds
        assert failed.parse.unresolved
        assert durable_memory_signature(runtime.memory) == before
        assert runtime.memory.turn_index == before_turn + 1
        assert all(
            "sarah" not in alias and not alias.startswith("entityref-")
            for entity in runtime.memory.entities.values()
            for alias in entity.aliases
        )

        followup = runtime.process("She reads.")
        assert followup.contract.status == AnswerStatus.MISSING_REFERENCE
        assert followup.parse.unresolved
        assert all(
            ref.key != "sarah_1"
            for event in followup.parse.events
            for ref in event.arguments.values()
        )
        assert not runtime.memory.events
        assert durable_memory_signature(runtime.memory) == before
        assert runtime.memory.turn_index == before_turn + 2
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "text,matrix_polarity,embedded_polarity,licensed,entailed",
    (
        ("Sarah enjoys not reading.", True, False, True, False),
        ("Sarah did not enjoy reading.", False, True, False, False),
        ("Sarah started not working.", True, False, True, True),
        ("Sarah did not start working.", False, True, False, False),
        ("David saw Sarah not leaving.", True, False, True, False),
        ("David did not see Sarah leaving.", False, True, False, False),
    ),
)
def test_matrix_and_embedded_gerund_negation_scopes_remain_separate(
    text,
    matrix_polarity,
    embedded_polarity,
    licensed,
    entailed,
) -> None:
    _memory, result = parse(text)
    assert len(result.events) == 2, result.diagnostics
    assert result.events[0].polarity is matrix_polarity
    assert result.events[1].polarity is embedded_polarity
    relation = result.gerunds[0]
    assert relation.licensed is licensed
    assert relation.entailed is entailed


def test_gerund_models_round_trip() -> None:
    relation = GerundRelation(
        relation_type=GerundRelationType.PERCEPTION_PARTICIPIAL,
        content_status=GerundContentStatus.PERCEIVED,
        matrix_event_index=0,
        complement_event_index=1,
        marker="-ing",
        matrix_predicate="see",
        source_entity_id="david_1",
        controller_entity_id="sarah_2",
        embedded_subject_entity_id="sarah_2",
        predicate_family="perception",
        certainty=215,
        relation_id="gerund_1",
        matrix_event_id="event_1",
        complement_event_id="event_2",
        licensed=True,
        entailed=False,
        diagnostics=["licensed perception participial"],
    )
    restored = GerundRelation.from_dict(relation.to_dict())
    assert restored.to_dict() == relation.to_dict()
    assert restored.signature() == relation.signature()

    ambiguity = GerundAttachmentAmbiguity(
        matrix_surface="Sarah enjoys",
        complement_surface="reading and writing",
        clause_surface="Sarah enjoys reading and writing",
        reason="gerund coordination requires staged parsing",
        candidate_boundaries=[2],
        candidate_relation_types=[GerundRelationType.GERUND_CONTENT],
        ambiguity_id="gerund-example",
        diagnostics=["unsafe gerund content suppressed"],
    )
    assert (
        GerundAttachmentAmbiguity.from_dict(ambiguity.to_dict()).to_dict()
        == ambiguity.to_dict()
    )


def test_parse_result_serializes_typed_gerund_trace() -> None:
    _memory, result = parse("David saw Sarah leaving.")
    payload = result.to_dict()
    assert payload["gerunds"][0]["relation_type"] == "perception_participial"
    assert payload["gerunds"][0]["content_status"] == "perceived"
    assert payload["gerunds"][0]["controller_entity_id"]
    assert payload["gerund_ambiguities"] == []


def test_memory_v6_snapshot_preserves_stable_gerund_links() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("David saw Sarah leaving.")
        payload = runtime.memory.dumps(indent=None)
    finally:
        runtime.close()

    assert json.loads(payload)["snapshot_version"] == 6
    restored = ConversationMemory.loads(payload)
    assert len(restored.gerunds) == 1
    relation = restored.gerunds[0]
    assert relation.relation_id
    assert relation.matrix_event_id
    assert relation.complement_event_id
    matrix = restored.get_event(relation.matrix_event_id)
    complement = restored.get_event(relation.complement_event_id)
    assert matrix is not None and matrix.predicate == "see"
    assert complement is not None and complement.predicate == "leave"
    assert complement.discourse_role == "participle"
    assert complement.tense == "nonfinite"
    assert complement.aspect == "participle"
    assert complement.source == SourceKind.ATTRIBUTED
    assert relation.controller_entity_id == relation.embedded_subject_entity_id


def test_runtime_and_memory_snapshots_share_v6_generation() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah enjoyed reading.")
        snapshot = runtime.to_dict()
        assert snapshot["snapshot_version"] == 6
        assert snapshot["memory"]["snapshot_version"] == 6
        assert snapshot["memory"]["gerund_counter"] == 1
    finally:
        runtime.close()


def test_legacy_v5_snapshot_without_gerund_fields_remains_loadable() -> None:
    memory = ConversationMemory()
    legacy = memory.to_dict()
    legacy["snapshot_version"] = 5
    legacy.pop("gerunds", None)
    legacy.pop("gerund_counter", None)
    restored = ConversationMemory.from_dict(legacy)
    assert restored.gerunds == []
    assert restored.to_dict()["snapshot_version"] == 6


def test_legacy_v5_runtime_snapshot_without_gerunds_migrates_to_v6() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        legacy = runtime.to_dict()
    finally:
        runtime.close()

    legacy["snapshot_version"] = 5
    legacy["memory"]["snapshot_version"] = 5
    legacy["memory"].pop("gerunds", None)
    legacy["memory"].pop("gerund_counter", None)
    restored = ClankerLM.from_dict(
        legacy,
        affect_backend=HeuristicAffectBackend(),
    )
    try:
        migrated = restored.to_dict()
        assert migrated["snapshot_version"] == 6
        assert migrated["memory"]["snapshot_version"] == 6
        assert migrated["memory"]["gerunds"] == []
        assert migrated["memory"]["gerund_counter"] == 0
    finally:
        restored.close()


def test_restored_gerund_counter_allocates_the_next_stable_id() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah enjoyed reading.")
        snapshot = runtime.dumps(indent=None)
    finally:
        runtime.close()

    restored = ClankerLM.loads(
        snapshot,
        affect_backend=HeuristicAffectBackend(),
    )
    try:
        restored.process("David avoided calling Mary.")
        assert [relation.relation_id for relation in restored.memory.gerunds] == [
            "gerund_1",
            "gerund_2",
        ]
        assert restored.memory.to_dict()["gerund_counter"] == 2
    finally:
        restored.close()


@pytest.mark.parametrize(
    "corruption",
    ("counter_below_max", "duplicate_relation_id", "empty_relation_id"),
)
def test_v6_snapshot_rejects_gerund_id_corruption_before_realization(
    corruption,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah enjoyed reading.")
        runtime.process("David avoided calling Mary.")
        snapshot = runtime.to_dict()
    finally:
        runtime.close()

    gerunds = snapshot["memory"]["gerunds"]
    assert [item["relation_id"] for item in gerunds] == [
        "gerund_1",
        "gerund_2",
    ]
    if corruption == "counter_below_max":
        snapshot["memory"]["gerund_counter"] = 1
    elif corruption == "duplicate_relation_id":
        gerunds[1]["relation_id"] = gerunds[0]["relation_id"]
    else:
        gerunds[0]["relation_id"] = ""

    # Reject before a duplicate/empty ID can bind a contract to the wrong
    # relation and cause its matrix qualification to be realized incorrectly.
    with pytest.raises(ValueError):
        ClankerLM.from_dict(
            snapshot,
            affect_backend=HeuristicAffectBackend(),
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "excessive_certainty",
        "forged_predicate_family",
        "equal_local_indices",
        "negative_matrix_index",
        "negative_complement_index",
    ),
)
def test_v6_snapshot_rejects_forged_gerund_relation_fields(corruption) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah enjoyed reading.")
        snapshot = runtime.to_dict()
    finally:
        runtime.close()

    relation = snapshot["memory"]["gerunds"][0]
    if corruption == "excessive_certainty":
        relation["certainty"] = 255
    elif corruption == "forged_predicate_family":
        relation["predicate_family"] = "forged-family"
    elif corruption == "equal_local_indices":
        relation["complement_event_index"] = relation["matrix_event_index"]
    elif corruption == "negative_matrix_index":
        relation["matrix_event_index"] = -1
    else:
        relation["complement_event_index"] = -1

    with pytest.raises(ValueError):
        ConversationMemory.from_dict(snapshot["memory"])


def test_repeated_gerund_relation_deduplicates() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah enjoyed reading.")
        runtime.process("Sarah enjoyed reading.")
        assert len(runtime.memory.gerunds) == 1
        assert len(
            [event for event in runtime.memory.events if event.predicate == "enjoy"]
        ) == 1
        assert len(
            [event for event in runtime.memory.events if event.predicate == "read"]
        ) == 1
        assert runtime.memory.to_dict()["gerund_counter"] == 1
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "first,second",
    (
        ("Sarah might start working.", "Sarah starts working."),
        ("Sarah starts working.", "Sarah might start working."),
        ("Sarah will start working.", "Sarah starts working."),
        ("Sarah starts working.", "Sarah will start working."),
    ),
)
def test_nonfactual_and_factual_phase_relations_remain_distinct_in_both_orders(
    first,
    second,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(first)
        runtime.process(second)
        matrix_events = [
            event
            for event in runtime.memory.events
            if event.predicate == "start" and event.discourse_role == "main"
        ]
        assert len(matrix_events) == 2
        assert len(runtime.memory.gerunds) == 2
        assert {relation.entailed for relation in runtime.memory.gerunds} == {
            False,
            True,
        }
        assert len(
            {relation.matrix_event_id for relation in runtime.memory.gerunds}
        ) == 2
        assert len(
            {relation.relation_id for relation in runtime.memory.gerunds}
        ) == 2

        activity = runtime.process("Did Sarah work?")
        assert activity.contract.status == AnswerStatus.TRUE
        assert activity.contract.source == SourceKind.INFERRED
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,query_text,embedded_predicate",
    (
        ("Sarah enjoyed reading.", "Did Sarah read?", "read"),
        ("Sarah avoided calling John.", "Did Sarah call John?", "call"),
        ("Mary started working.", "Did Mary work?", "work"),
        ("John stopped smoking.", "Did John smoke?", "smoke"),
        ("David continued talking.", "Did David talk?", "talk"),
        ("David saw Sarah leaving.", "Did Sarah leave?", "leave"),
    ),
)
def test_gerund_event_is_excluded_from_unqualified_memory_matching(
    statement,
    query_text,
    embedded_predicate,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        query = runtime.parser.parse(query_text, runtime.memory).question.event
        assert query.predicate == embedded_predicate
        complement = runtime.memory.get_event(
            runtime.memory.gerunds[0].complement_event_id
        )
        assert complement is not None
        nonfinite_query = query.copy(
            tense=complement.tense,
            aspect=complement.aspect,
        )
        assert runtime.memory.match_events(nonfinite_query) == []
        attributed = runtime.memory.match_events(
            nonfinite_query,
            include_gerund_content=True,
        )
        assert len(attributed) == 1
        assert attributed[0].event.discourse_role in {"gerund", "participle"}
        assert attributed[0].event.source == SourceKind.ATTRIBUTED
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,expected",
    (
        (
            "Sarah enjoyed reading.",
            "What did Sarah enjoy doing?",
            "sarah enjoyed reading",
        ),
        (
            "David avoided calling Mary.",
            "What did David avoid doing?",
            "david avoided calling mary",
        ),
        (
            "Mary started opening the door.",
            "What did Mary start doing?",
            "mary started opening the door",
        ),
        (
            "John stopped buying groceries.",
            "What did John stop doing?",
            "john stopped buying groceries",
        ),
        (
            "David continued helping Sarah.",
            "What did David continue doing?",
            "david continued helping sarah",
        ),
        (
            "Sarah saw John calling Mary.",
            "What did Sarah see John doing?",
            "sarah saw john calling mary",
        ),
    ),
)
def test_gerund_wh_questions_bind_and_realize_compositionally(
    statement,
    question,
    expected,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert answer.contract.source == SourceKind.ATTRIBUTED
        assert answer.contract.required_slots["gerund"] == "true"
        assert answer.contract.required_slots["content_status"]
        assert expected in answer.response.lower()
        assert "promote_gerund_to_unqualified_event" in (
            answer.contract.forbidden_claims
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,completion_question,matrix_qualification",
    (
        (
            "Sarah enjoyed reading.",
            "Did Sarah read?",
            "sarah enjoyed reading",
        ),
        (
            "Sarah avoided calling John.",
            "Did Sarah call John?",
            "sarah avoided calling john",
        ),
        (
            "David saw Sarah leaving.",
            "Did Sarah leave?",
            "david saw sarah leaving",
        ),
    ),
)
def test_nonphase_gerund_content_never_proves_unqualified_completion(
    statement,
    completion_question,
    matrix_qualification,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(completion_question)
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.source == SourceKind.ATTRIBUTED
        assert answer.contract.required_slots["gerund"] == "true"
        assert "promote_gerund_to_unqualified_event" in (
            answer.contract.forbidden_claims
        )
        assert matrix_qualification in answer.response.lower()
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,completion_question,matrix_qualification,status",
    (
        (
            "Mary started working.",
            "Did Mary work?",
            "mary started working",
            "begun",
        ),
        (
            "John stopped smoking.",
            "Did John smoke?",
            "john stopped smoking",
            "stopped",
        ),
        (
            "David continued talking.",
            "Did David talk?",
            "david continued talking",
            "continued",
        ),
    ),
)
def test_phase_relations_support_only_qualified_derived_answers(
    statement,
    completion_question,
    matrix_qualification,
    status,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(completion_question)
        assert answer.contract.status == AnswerStatus.TRUE
        assert answer.contract.source == SourceKind.INFERRED
        assert answer.contract.required_slots["gerund"] == "true"
        assert answer.contract.required_slots["content_status"] == status
        assert "promote_gerund_to_unqualified_event" in (
            answer.contract.forbidden_claims
        )
        assert matrix_qualification in answer.response.lower()
        assert answer.response.lower().strip() not in {"yes", "yes."}
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,matrix_predicate,embedded_predicate,modality",
    (
        ("Sarah might start working.", "Did Sarah work?", "start", "work", "might"),
        ("Sarah may start working.", "Did Sarah work?", "start", "work", "may"),
        ("Sarah will start working.", "Did Sarah work?", "start", "work", "will"),
        ("John might stop smoking.", "Did John smoke?", "stop", "smoke", "might"),
        ("John may stop smoking.", "Did John smoke?", "stop", "smoke", "may"),
        ("John will stop smoking.", "Did John smoke?", "stop", "smoke", "will"),
        (
            "David might continue talking.",
            "Did David talk?",
            "continue",
            "talk",
            "might",
        ),
        (
            "David may continue talking.",
            "Did David talk?",
            "continue",
            "talk",
            "may",
        ),
        (
            "David will continue talking.",
            "Did David talk?",
            "continue",
            "talk",
            "will",
        ),
    ),
)
def test_modal_phase_matrices_do_not_license_activity_inference(
    statement,
    question,
    matrix_predicate,
    embedded_predicate,
    modality,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        turn = runtime.process(statement)
        assert [event.predicate for event in turn.parse.events] == [
            matrix_predicate,
            embedded_predicate,
        ]
        assert turn.parse.events[0].modality == modality
        relation = turn.parse.gerunds[0]
        assert relation.licensed
        assert not relation.entailed

        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.source != SourceKind.INFERRED
        assert answer.contract.required_slots["gerund"] == "true"
        assert "promote_gerund_to_unqualified_event" in (
            answer.contract.forbidden_claims
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,matrix_aspect",
    (
        ("Sarah is starting working.", "Did Sarah work?", "progressive"),
        ("John is stopping smoking.", "Did John smoke?", "progressive"),
        ("David is continuing talking.", "Did David talk?", "progressive"),
    ),
)
def test_progressive_phase_matrices_do_not_license_activity_inference(
    statement,
    question,
    matrix_aspect,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        turn = runtime.process(statement)
        assert turn.parse.events[0].aspect == matrix_aspect
        relation = turn.parse.gerunds[0]
        assert relation.licensed
        assert not relation.entailed

        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.source != SourceKind.INFERRED
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,time_surface",
    (
        ("Mary starts working tomorrow.", "tomorrow"),
        ("Mary starts working later.", "later"),
        ("Mary starts working next week.", "next week"),
    ),
)
def test_scheduled_future_phase_matrices_do_not_license_activity_inference(
    statement,
    time_surface,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        turn = runtime.process(statement)
        complement = turn.parse.events[1]
        assert complement.arguments["time"].surface.lower() == time_surface
        relation = turn.parse.gerunds[0]
        assert relation.licensed
        assert not relation.entailed

        answer = runtime.process("Did Mary work?")
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.source != SourceKind.INFERRED
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,content_status",
    (
        ("Sarah starts working.", "Did Sarah work?", "begun"),
        ("Sarah started working.", "Did Sarah work?", "begun"),
        ("John stops smoking.", "Did John smoke?", "stopped"),
        ("John stopped smoking.", "Did John smoke?", "stopped"),
        ("David continues talking.", "Did David talk?", "continued"),
        ("David continued talking.", "Did David talk?", "continued"),
    ),
)
def test_factual_nonprogressive_phase_matrices_preserve_reviewed_inference(
    statement,
    question,
    content_status,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        turn = runtime.process(statement)
        assert turn.parse.events[0].modality is None
        relation = turn.parse.gerunds[0]
        assert relation.licensed
        assert relation.entailed

        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.TRUE
        assert answer.contract.source == SourceKind.INFERRED
        assert answer.contract.required_slots["content_status"] == content_status
        assert answer.contract.required_slots["derived_phase_inference"] == "true"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,completion_question",
    (
        ("Mary did not start working.", "Did Mary work?"),
        ("John did not stop smoking.", "Did John smoke?"),
        ("David did not continue talking.", "Did David talk?"),
    ),
)
def test_negated_phase_matrix_does_not_license_derived_activity(
    statement,
    completion_question,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        relation = runtime.memory.gerunds[0]
        assert not relation.licensed
        assert not relation.entailed
        answer = runtime.process(completion_question)
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.source != SourceKind.INFERRED
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,status,response_fragment",
    (
        (
            "Sarah enjoyed not reading.",
            "Did Sarah enjoy reading?",
            AnswerStatus.FALSE,
            "enjoyed not reading",
        ),
        (
            "Sarah enjoyed not reading.",
            "Did Sarah enjoy not reading?",
            AnswerStatus.TRUE,
            "enjoyed not reading",
        ),
        (
            "Sarah enjoyed reading.",
            "Did Sarah enjoy not reading?",
            AnswerStatus.FALSE,
            "enjoyed reading",
        ),
        (
            "Sarah did not enjoy reading.",
            "Did Sarah not enjoy reading?",
            AnswerStatus.TRUE,
            "did not enjoy reading",
        ),
        (
            "Sarah enjoyed reading.",
            "Did Sarah not enjoy reading?",
            AnswerStatus.FALSE,
            "enjoyed reading",
        ),
    ),
)
def test_polar_questions_compare_matrix_and_embedded_gerund_negation(
    statement,
    question,
    status,
    response_fragment,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(question)
        assert answer.contract.status == status
        assert answer.contract.required_slots["gerund"] == "true"
        assert response_fragment in answer.response.lower()
        frame = answer.contract.question
        assert frame.matrix_polarity is not None
        assert frame.embedded_polarity is not None
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,more_specific_question",
    (
        ("Sarah avoided calling.", "Did Sarah avoid calling Mary?"),
        ("Sarah enjoyed reading.", "Did Sarah enjoy reading books?"),
    ),
)
def test_underspecified_gerund_content_does_not_prove_specific_matrix_query(
    statement,
    more_specific_question,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(more_specific_question)
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.status != AnswerStatus.TRUE
        assert answer.response.lower().strip() not in {"yes", "yes."}
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,less_specific_question,qualification",
    (
        (
            "Sarah avoided calling Mary.",
            "Did Sarah avoid calling?",
            "sarah avoided calling mary",
        ),
        (
            "Sarah enjoyed reading books.",
            "Did Sarah enjoy reading?",
            "sarah enjoyed reading books",
        ),
    ),
)
def test_specific_gerund_content_supports_less_specific_matrix_query(
    statement,
    less_specific_question,
    qualification,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(less_specific_question)
        assert answer.contract.status == AnswerStatus.TRUE
        assert answer.contract.required_slots["gerund"] == "true"
        assert qualification in answer.response.lower()
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "source,inferred",
    (
        (SourceKind.RETRIEVED, False),
        (SourceKind.VERIFIED, False),
        (SourceKind.INFERRED, True),
    ),
)
def test_learned_gerund_matrix_qa_preserves_trusted_provenance(
    source,
    inferred,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        stored = runtime.learn(
            "Sarah enjoyed reading.",
            source=source,
            inferred=inferred,
        )
        assert [event.source for event in stored] == [
            source,
            SourceKind.ATTRIBUTED,
        ]
        relation = runtime.memory.gerunds[0]
        assert relation.matrix_event_id == stored[0].event_id
        assert relation.complement_event_id == stored[1].event_id

        matrix_answer = runtime.process("Did Sarah enjoy reading?")
        assert matrix_answer.contract.status == AnswerStatus.TRUE
        assert matrix_answer.contract.source == source
        assert matrix_answer.contract.source != SourceKind.USER

        embedded_answer = runtime.process("Did Sarah read?")
        assert embedded_answer.contract.status == AnswerStatus.UNKNOWN
        assert embedded_answer.contract.source == SourceKind.ATTRIBUTED
    finally:
        runtime.close()


def test_learned_gerund_conflict_does_not_invent_user_provenance() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.learn(
            "Sarah enjoyed reading.",
            source=SourceKind.RETRIEVED,
        )
        runtime.learn(
            "Sarah did not enjoy reading.",
            source=SourceKind.VERIFIED,
        )
        answer = runtime.process("Did Sarah enjoy reading?")
        assert answer.contract.status == AnswerStatus.CONFLICT
        assert answer.contract.source != SourceKind.USER
        assert {
            evidence.event.source
            for evidence in answer.contract.evidence
            if evidence.event.discourse_role == "main"
        } == {SourceKind.RETRIEVED, SourceKind.VERIFIED}
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statements,question,relation_phrases",
    (
        (
            (
                "Sarah enjoyed reading.",
                "Sarah did not enjoy reading.",
            ),
            "Did Sarah enjoy reading?",
            ("sarah enjoyed reading", "sarah did not enjoy reading"),
        ),
        (
            (
                "Sarah enjoyed reading.",
                "Sarah enjoyed not reading.",
            ),
            "Did Sarah enjoy reading?",
            ("sarah enjoyed reading", "sarah enjoyed not reading"),
        ),
    ),
)
def test_gerund_conflict_realizes_whole_relations_without_content_leak(
    statements,
    question,
    relation_phrases,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        for statement in statements:
            runtime.process(statement)
        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.CONFLICT
        response = answer.response.lower()
        for phrase in relation_phrases:
            assert phrase in response
        assert "sarah reads" not in response
        assert "sarah does not read" not in response
        assert "sarah did not read" not in response
    finally:
        runtime.close()


def test_stale_gerund_conflict_relation_ids_fail_closed_in_realization() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah enjoyed reading.")
        runtime.process("Sarah enjoyed not reading.")
        conflict = runtime.process("Did Sarah enjoy reading?")
        assert conflict.contract.status == AnswerStatus.CONFLICT
        conflict.contract.required_slots["relation_ids"] = (
            "gerund_missing_1,gerund_missing_2"
        )

        candidates = runtime.realizer.realize(conflict.contract, conflict.gates)
        assert candidates
        assert {candidate.text for candidate in candidates} == {
            "I have conflicting information."
        }
        assert all("sarah reads" not in item.text.lower() for item in candidates)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,modality,matrix_aspect",
    (
        (
            "Sarah might enjoy reading.",
            "Does Sarah enjoy reading?",
            "might",
            "simple",
        ),
        (
            "David might see Sarah leaving.",
            "Does David see Sarah leaving?",
            "might",
            "simple",
        ),
        (
            "Sarah is enjoying reading.",
            "Does Sarah enjoy reading?",
            None,
            "progressive",
        ),
        (
            "I was seeing John leaving.",
            "Did I see John leaving?",
            None,
            "progressive",
        ),
    ),
)
def test_nonfactual_nonphase_relation_does_not_prove_factual_matrix_query(
    statement,
    question,
    modality,
    matrix_aspect,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        turn = runtime.process(statement)
        assert turn.parse.events[0].modality == modality
        assert turn.parse.events[0].aspect == matrix_aspect
        assert not turn.parse.gerunds[0].entailed
        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.status != AnswerStatus.TRUE
    finally:
        runtime.close()


def test_question_trace_serializes_both_gerund_polarities() -> None:
    memory = ConversationMemory()
    memory.begin_turn()
    parsed = SemanticParser().parse("Did Sarah enjoy not reading?", memory)
    payload = parsed.to_dict()["question"]
    assert payload["matrix_polarity"] is True
    assert payload["embedded_polarity"] is False
    assert payload["event"]["polarity"] is True


def test_direct_activity_fact_outranks_qualified_phase_inference() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Mary started working.")
        runtime.process("Mary worked.")
        answer = runtime.process("Did Mary work?")
        assert answer.contract.status == AnswerStatus.TRUE
        assert answer.contract.source == SourceKind.USER
    finally:
        runtime.close()


def test_runtime_snapshot_round_trip_preserves_gerund_qa() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah avoided calling John.")
        snapshot = runtime.dumps(indent=None)
    finally:
        runtime.close()

    restored = ClankerLM.loads(
        snapshot,
        affect_backend=HeuristicAffectBackend(),
    )
    try:
        answer = restored.process("What did Sarah avoid doing?")
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert answer.contract.required_slots["gerund"] == "true"
        assert "sarah avoided calling john" in answer.response.lower()
    finally:
        restored.close()
