"""Infinitival control, raising, and non-entailment conformance for issue #88."""

from __future__ import annotations

from itertools import product

import pytest

from clanker_lm.affect import HeuristicAffectBackend
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    AnswerStatus,
    InfinitivalAttachmentAmbiguity,
    InfinitivalContentStatus,
    InfinitivalRelation,
    InfinitivalRelationType,
    SourceKind,
)
from clanker_lm.parser import SemanticParser
from clanker_lm.runtime import ClankerLM


SUBJECTS = ("Sarah", "David")
SUBJECT_CONTROL_FORMS = (
    ("plans", "plan", "present", InfinitivalContentStatus.PLANNED, "planning"),
    ("planned", "plan", "past", InfinitivalContentStatus.PLANNED, "planning"),
    ("intends", "intend", "present", InfinitivalContentStatus.INTENDED, "intention"),
    ("intended", "intend", "past", InfinitivalContentStatus.INTENDED, "intention"),
    ("hopes", "hope", "present", InfinitivalContentStatus.HOPED, "hope"),
    ("hoped", "hope", "past", InfinitivalContentStatus.HOPED, "hope"),
    ("wants", "want", "present", InfinitivalContentStatus.DESIRED, "desire"),
    ("wanted", "want", "past", InfinitivalContentStatus.DESIRED, "desire"),
)
SUBJECT_CONTROL_COMPLEMENTS = (
    ("leave", "leave"),
    ("call Mary", "call"),
    ("buy groceries", "buy"),
    ("open the door", "open"),
    ("help John", "help"),
)
SUBJECT_CONTROL_CASES = tuple(
    product(SUBJECTS, SUBJECT_CONTROL_FORMS, SUBJECT_CONTROL_COMPLEMENTS)
)

OBJECT_CONTROL_FORMS = (
    ("tells", "tell", "present", InfinitivalContentStatus.DIRECTED, "directive"),
    ("told", "tell", "past", InfinitivalContentStatus.DIRECTED, "directive"),
    ("asks", "ask", "present", InfinitivalContentStatus.REQUESTED, "request"),
    ("asked", "ask", "past", InfinitivalContentStatus.REQUESTED, "request"),
    ("wants", "want", "present", InfinitivalContentStatus.DESIRED, "desire"),
    ("wanted", "want", "past", InfinitivalContentStatus.DESIRED, "desire"),
)
CONTROLLERS = ("John", "Mary")
OBJECT_CONTROL_COMPLEMENTS = (
    ("leave", "leave"),
    ("call David", "call"),
    ("buy groceries", "buy"),
    ("open the door", "open"),
)
OBJECT_CONTROL_CASES = tuple(
    product(
        SUBJECTS,
        OBJECT_CONTROL_FORMS,
        CONTROLLERS,
        OBJECT_CONTROL_COMPLEMENTS,
    )
)

RAISING_FORMS = (
    ("seems", "seem", "present"),
    ("seemed", "seem", "past"),
    ("appears", "appear", "present"),
    ("appeared", "appear", "past"),
)
RAISING_COMPLEMENTS = (
    ("know the answer", "know"),
    ("be tired", "be"),
    ("need help", "need"),
)
RAISING_CASES = tuple(product(SUBJECTS, RAISING_FORMS, RAISING_COMPLEMENTS))

# 80 subject-control + 96 object-control + 24 raising = 200 generated cases.
assert len(SUBJECT_CONTROL_CASES) + len(OBJECT_CONTROL_CASES) + len(RAISING_CASES) == 200


def parse(text: str):
    memory = ConversationMemory()
    memory.begin_turn()
    result = SemanticParser().parse(text, memory)
    return memory, result


def entity_argument(event, roles=("agent", "subject", "experiencer")):
    for role in roles:
        ref = event.arguments.get(role)
        if ref is not None:
            return ref
    raise AssertionError(f"No entity role in event {event.to_dict()}")


@pytest.mark.parametrize("subject,matrix,complement", SUBJECT_CONTROL_CASES)
def test_generated_subject_control(subject, matrix, complement) -> None:
    surface, predicate, tense, status, family = matrix
    complement_surface, complement_predicate = complement
    memory, result = parse(f"{subject} {surface} to {complement_surface}.")

    assert not result.infinitival_ambiguities, result.diagnostics
    assert [event.predicate for event in result.events] == [
        predicate,
        complement_predicate,
    ]
    assert [event.discourse_role for event in result.events] == ["main", "infinitive"]
    assert result.events[0].tense == tense
    assert result.events[0].source == SourceKind.USER
    assert result.events[1].tense == "infinitive"
    assert result.events[1].source == SourceKind.ATTRIBUTED
    assert len(result.infinitivals) == 1

    relation = result.infinitivals[0]
    source = entity_argument(result.events[0])
    embedded = entity_argument(result.events[1])
    assert relation.relation_type == InfinitivalRelationType.SUBJECT_CONTROL
    assert relation.content_status == status
    assert relation.predicate_family == family
    assert relation.matrix_predicate == predicate
    assert relation.source_entity_id == source.key
    assert relation.controller_entity_id == source.key
    assert relation.embedded_subject_entity_id == embedded.key == source.key
    assert relation.licensed
    assert not relation.entailed
    assert memory.get_entity(source.key) is not None


@pytest.mark.parametrize(
    "subject,matrix,controller,complement",
    OBJECT_CONTROL_CASES,
)
def test_generated_object_control(subject, matrix, controller, complement) -> None:
    surface, predicate, tense, status, family = matrix
    complement_surface, complement_predicate = complement
    memory, result = parse(
        f"{subject} {surface} {controller} to {complement_surface}."
    )

    assert not result.infinitival_ambiguities, result.diagnostics
    assert [event.predicate for event in result.events] == [
        predicate,
        complement_predicate,
    ]
    assert [event.discourse_role for event in result.events] == ["main", "infinitive"]
    assert result.events[0].tense == tense
    assert result.events[1].tense == "infinitive"
    assert result.events[1].source == SourceKind.ATTRIBUTED
    assert len(result.infinitivals) == 1

    relation = result.infinitivals[0]
    matrix_source = entity_argument(result.events[0])
    matrix_controller = entity_argument(
        result.events[0],
        roles=("patient", "recipient"),
    )
    embedded = entity_argument(result.events[1])
    assert relation.relation_type == InfinitivalRelationType.OBJECT_CONTROL
    assert relation.content_status == status
    assert relation.predicate_family == family
    assert relation.source_entity_id == matrix_source.key
    assert relation.controller_entity_id == matrix_controller.key
    assert relation.embedded_subject_entity_id == embedded.key == matrix_controller.key
    assert relation.controller_entity_id != relation.source_entity_id
    assert relation.licensed
    assert not relation.entailed
    assert memory.get_entity(matrix_controller.key).canonical_name.lower() == controller.lower()


@pytest.mark.parametrize("subject,matrix,complement", RAISING_CASES)
def test_generated_raising(subject, matrix, complement) -> None:
    surface, predicate, tense = matrix
    complement_surface, complement_predicate = complement
    _memory, result = parse(f"{subject} {surface} to {complement_surface}.")

    assert not result.infinitival_ambiguities, result.diagnostics
    assert [event.predicate for event in result.events] == [predicate, complement_predicate]
    assert result.events[0].tense == tense
    assert result.events[1].tense == "infinitive"
    assert result.events[1].source == SourceKind.ATTRIBUTED
    assert len(result.infinitivals) == 1

    relation = result.infinitivals[0]
    matrix_subject = entity_argument(result.events[0])
    embedded_subject = entity_argument(result.events[1])
    assert relation.relation_type == InfinitivalRelationType.RAISING
    assert relation.content_status == InfinitivalContentStatus.EVIDENTIAL
    assert relation.predicate_family == "appearance"
    assert relation.source_entity_id == matrix_subject.key
    assert relation.controller_entity_id == embedded_subject.key == matrix_subject.key
    assert relation.licensed
    assert not relation.entailed


def test_purpose_adjunct_does_not_become_selected_infinitival_complement() -> None:
    _memory, result = parse("Sarah went to buy groceries.")
    assert not result.infinitivals
    assert not result.infinitival_ambiguities
    assert len(result.events) == 1
    assert result.events[0].predicate == "go"
    assert result.events[0].arguments["purpose"].surface == "buy groceries"


@pytest.mark.parametrize(
    "text,matrix_polarity,embedded_polarity,licensed",
    (
        ("Sarah plans not to leave.", True, False, True),
        ("Sarah did not plan to leave.", False, True, False),
        ("Sarah told John not to leave.", True, False, True),
        ("Sarah did not tell John to leave.", False, True, False),
    ),
)
def test_matrix_and_embedded_negation_scopes_remain_separate(
    text,
    matrix_polarity,
    embedded_polarity,
    licensed,
) -> None:
    _memory, result = parse(text)
    assert len(result.events) == 2, result.diagnostics
    assert result.events[0].polarity is matrix_polarity
    assert result.events[1].polarity is embedded_polarity
    assert result.infinitivals[0].licensed is licensed
    assert not result.infinitivals[0].entailed


@pytest.mark.parametrize(
    "text,reason_fragment",
    (
        (
            "Sarah plans John to leave.",
            "does not license an explicit object controller",
        ),
        ("Sarah told to leave.", "requires an explicit object controller"),
        ("Sarah asked to call Mary.", "requires an explicit object controller"),
        ("Sarah plans to try to leave.", "nested infinitival content"),
        (
            "Sarah plans to say John left.",
            "infinitival content containing a finite content clause",
        ),
    ),
)
def test_unsupported_infinitival_boundaries_fail_explicitly(
    text,
    reason_fragment,
) -> None:
    _memory, result = parse(text)
    assert not result.events
    assert not result.infinitivals
    assert result.infinitival_ambiguities
    assert reason_fragment in result.infinitival_ambiguities[0].reason
    assert result.unresolved


def test_finite_content_containing_infinitive_fails_at_configured_depth() -> None:
    _memory, result = parse("Sarah said John plans to leave.")
    assert not result.events
    assert not result.contents
    assert not result.infinitivals
    assert result.content_ambiguities
    assert "finite content containing an infinitival complement" in result.content_ambiguities[0].reason


@pytest.mark.parametrize(
    "text",
    (
        "The woman who called me plans to leave.",
        "Sarah, my supervisor, plans to leave.",
        "Sarah plans to leave because John called.",
    ),
)
def test_unstaged_infinitive_relation_layers_fail_closed(text) -> None:
    _memory, result = parse(text)
    assert not result.events
    assert not result.infinitivals
    assert result.infinitival_ambiguities
    assert "requires staged parsing" in result.infinitival_ambiguities[0].reason


def test_infinitival_models_round_trip() -> None:
    relation = InfinitivalRelation(
        relation_type=InfinitivalRelationType.OBJECT_CONTROL,
        content_status=InfinitivalContentStatus.DIRECTED,
        matrix_event_index=0,
        complement_event_index=1,
        marker="to",
        matrix_predicate="tell",
        source_entity_id="sarah_1",
        controller_entity_id="john_2",
        embedded_subject_entity_id="john_2",
        predicate_family="directive",
        certainty=210,
        relation_id="infinitive_1",
        matrix_event_id="event_1",
        complement_event_id="event_2",
        licensed=True,
        entailed=False,
        diagnostics=["licensed object control"],
    )
    restored = InfinitivalRelation.from_dict(relation.to_dict())
    assert restored.to_dict() == relation.to_dict()
    assert restored.signature() == relation.signature()

    ambiguity = InfinitivalAttachmentAmbiguity(
        matrix_surface="Sarah plans",
        complement_surface="to try to leave",
        clause_surface="Sarah plans to try to leave",
        reason="nested infinitival content exceeds the configured depth",
        candidate_boundaries=[2, 4],
        candidate_relation_types=[InfinitivalRelationType.SUBJECT_CONTROL],
        ambiguity_id="infinitive-example",
        diagnostics=["unsafe content suppressed"],
    )
    assert (
        InfinitivalAttachmentAmbiguity.from_dict(ambiguity.to_dict()).to_dict()
        == ambiguity.to_dict()
    )


def test_memory_snapshot_preserves_stable_infinitival_links() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah told John to call Mary.")
        payload = runtime.memory.dumps(indent=None)
    finally:
        runtime.close()

    restored = ConversationMemory.loads(payload)
    assert len(restored.infinitivals) == 1
    relation = restored.infinitivals[0]
    assert relation.relation_id
    assert relation.matrix_event_id
    assert relation.complement_event_id
    matrix = restored.get_event(relation.matrix_event_id)
    complement = restored.get_event(relation.complement_event_id)
    assert matrix is not None and matrix.predicate == "tell"
    assert complement is not None and complement.predicate == "call"
    assert complement.discourse_role == "infinitive"
    assert complement.source == SourceKind.ATTRIBUTED
    assert relation.controller_entity_id == relation.embedded_subject_entity_id


def test_legacy_snapshot_without_infinitival_fields_remains_loadable() -> None:
    memory = ConversationMemory()
    legacy = memory.to_dict()
    legacy["snapshot_version"] = 3
    legacy.pop("infinitivals", None)
    legacy.pop("infinitival_counter", None)
    restored = ConversationMemory.from_dict(legacy)
    assert restored.infinitivals == []


def test_repeated_infinitival_relation_deduplicates() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah planned to leave.")
        runtime.process("Sarah planned to leave.")
        assert len(runtime.memory.infinitivals) == 1
        assert len([event for event in runtime.memory.events if event.predicate == "plan"]) == 1
        assert len([event for event in runtime.memory.events if event.predicate == "leave"]) == 1
    finally:
        runtime.close()


def test_infinitive_is_excluded_from_unqualified_event_matching() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah planned to leave.")
        query = runtime.parser.parse("Did Sarah leave?", runtime.memory).question.event
        assert runtime.memory.match_events(query) == []
        attributed = runtime.memory.match_events(
            query,
            include_infinitival_content=True,
        )
        assert len(attributed) == 1
        assert attributed[0].event.discourse_role == "infinitive"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,expected",
    (
        ("Sarah planned to leave.", "What did Sarah plan?", "sarah planned to leave"),
        ("Sarah planned to leave.", "What did Sarah plan to do?", "sarah planned to leave"),
        ("Sarah planned to leave.", "Who planned to leave?", "sarah planned to leave"),
        ("Sarah told John to call Mary.", "Who did Sarah tell to call Mary?", "sarah told john to call mary"),
        ("Sarah told John to call Mary.", "What did Sarah tell John to do?", "sarah told john to call mary"),
        ("Sarah wants John to leave.", "What does Sarah want John to do?", "sarah wants john to leave"),
        ("Sarah seems to know the answer.", "What does Sarah seem to know?", "sarah seems to know the answer"),
        ("Sarah seems to know the answer.", "Who seems to know the answer?", "sarah seems to know the answer"),
    ),
)
def test_infinitival_wh_questions_bind_typed_relation(statement, question, expected) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(question)
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert answer.contract.source == SourceKind.ATTRIBUTED
        assert answer.contract.required_slots["infinitival"] == "true"
        assert expected in answer.response.lower()
        assert "promote_infinitive_to_accomplished_event" in answer.contract.forbidden_claims or answer.contract.question.kind.value == "who"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,question,status,fragment",
    (
        ("Sarah planned to leave.", "Did Sarah plan to leave?", AnswerStatus.TRUE, "yes"),
        ("Sarah did not plan to leave.", "Did Sarah plan to leave?", AnswerStatus.FALSE, "no"),
        ("Sarah told John to call Mary.", "Did Sarah tell John to call Mary?", AnswerStatus.TRUE, "yes"),
        ("Sarah seems to know the answer.", "Does Sarah seem to know the answer?", AnswerStatus.TRUE, "yes"),
    ),
)
def test_infinitival_polar_questions_preserve_matrix_truth(
    statement,
    question,
    status,
    fragment,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(question)
        assert answer.contract.status == status
        assert fragment in answer.response.lower()
        assert answer.contract.required_slots["infinitival"] == "true"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "statement,completion_question,matrix_fragment,completion_fragment",
    (
        ("Sarah planned to leave.", "Did Sarah leave?", "sarah planned to leave", "don't know whether sarah left"),
        ("Sarah wanted John to leave.", "Did John leave?", "sarah wanted john to leave", "don't know whether john left"),
        ("Sarah told John to call Mary.", "Did John call Mary?", "sarah told john to call mary", "don't know whether john called mary"),
        ("Sarah asked John to call Mary.", "Did John call Mary?", "sarah asked john to call mary", "don't know whether john called mary"),
        ("Sarah seems to know the answer.", "Did Sarah know the answer?", "sarah seems to know the answer", "don't know whether sarah knew the answer"),
    ),
)
def test_infinitival_content_never_proves_completion(
    statement,
    completion_question,
    matrix_fragment,
    completion_fragment,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(completion_question)
        assert answer.contract.status == AnswerStatus.UNKNOWN
        assert answer.contract.source == SourceKind.ATTRIBUTED
        assert "promote_infinitive_to_accomplished_event" in answer.contract.forbidden_claims
        assert matrix_fragment in answer.response.lower()
        assert completion_fragment in answer.response.lower()
    finally:
        runtime.close()


def test_direct_completion_fact_outranks_nonentailed_plan() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah planned to leave.")
        runtime.process("Sarah left.")
        answer = runtime.process("Did Sarah leave?")
        assert answer.contract.status == AnswerStatus.TRUE
        assert answer.contract.source == SourceKind.USER
    finally:
        runtime.close()


def test_negated_matrix_does_not_license_positive_infinitival_content() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah did not plan to leave.")
        relation = runtime.memory.infinitivals[0]
        assert not relation.licensed

        content = runtime.process("What did Sarah plan?")
        assert content.contract.status == AnswerStatus.UNKNOWN
        assert "only a negated infinitival relation" in content.contract.reason

        completion = runtime.process("Did Sarah leave?")
        assert completion.contract.status == AnswerStatus.UNKNOWN
        assert completion.contract.source != SourceKind.ATTRIBUTED
    finally:
        runtime.close()


def test_conflicting_positive_and_negated_matrix_relations_remain_conflict() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah planned to leave.")
        runtime.process("Sarah did not plan to leave.")
        answer = runtime.process("Did Sarah plan to leave?")
        assert answer.contract.status == AnswerStatus.CONFLICT
        assert len(answer.contract.evidence) >= 2
    finally:
        runtime.close()


def test_parse_result_serializes_infinitival_trace() -> None:
    _memory, result = parse("Sarah told John to leave.")
    payload = result.to_dict()
    assert payload["infinitivals"][0]["relation_type"] == "object_control"
    assert payload["infinitivals"][0]["content_status"] == "directed"
    assert payload["infinitival_ambiguities"] == []


def test_runtime_snapshot_round_trip_preserves_infinitival_qa() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah intends to call Mary.")
        snapshot = runtime.dumps(indent=None)
    finally:
        runtime.close()

    restored = ClankerLM.loads(
        snapshot,
        affect_backend=HeuristicAffectBackend(),
    )
    try:
        answer = restored.process("What does Sarah intend to do?")
        assert answer.contract.status == AnswerStatus.ANSWERED
        assert "sarah intends to call mary" in answer.response.lower()
    finally:
        restored.close()

@pytest.mark.parametrize(
    "statement,question,status,response_fragment",
    (
        (
            "Sarah planned not to leave.",
            "Did Sarah plan to leave?",
            AnswerStatus.FALSE,
            "planned to not leave",
        ),
        (
            "Sarah planned not to leave.",
            "Did Sarah plan not to leave?",
            AnswerStatus.TRUE,
            "planned to not leave",
        ),
        (
            "Sarah planned to leave.",
            "Did Sarah plan not to leave?",
            AnswerStatus.FALSE,
            "planned to leave",
        ),
        (
            "Sarah did not plan to leave.",
            "Did Sarah not plan to leave?",
            AnswerStatus.TRUE,
            "did not plan to leave",
        ),
        (
            "Sarah planned to leave.",
            "Did Sarah not plan to leave?",
            AnswerStatus.FALSE,
            "planned to leave",
        ),
        (
            "Sarah wants John not to leave.",
            "Does Sarah want John to leave?",
            AnswerStatus.FALSE,
            "wants john to not leave",
        ),
        (
            "Sarah wants John not to leave.",
            "Does Sarah want John not to leave?",
            AnswerStatus.TRUE,
            "wants john to not leave",
        ),
    ),
)
def test_polar_questions_compare_matrix_and_embedded_negation_independently(
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
        assert response_fragment in answer.response.lower()
        frame = answer.contract.question
        assert frame.matrix_polarity is not None
        assert frame.embedded_polarity is not None
    finally:
        runtime.close()


def test_question_trace_serializes_both_infinitival_polarities() -> None:
    memory = ConversationMemory()
    memory.begin_turn()
    parsed = SemanticParser().parse("Did Sarah plan not to leave?", memory)
    payload = parsed.to_dict()["question"]
    assert payload["matrix_polarity"] is True
    assert payload["embedded_polarity"] is False
    assert payload["event"]["polarity"] is True


def test_who_question_reports_multiple_infinitival_sources() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah planned to leave.")
        runtime.process("Mary planned to leave.")
        answer = runtime.process("Who planned to leave?")
        assert answer.contract.status == AnswerStatus.MULTIPLE_MATCHES
        assert {item.surface.lower() for item in answer.contract.values} == {
            "sarah",
            "mary",
        }
    finally:
        runtime.close()


def test_object_control_question_reports_multiple_controllers() -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process("Sarah wanted John to leave.")
        runtime.process("Sarah wanted Mary to leave.")
        answer = runtime.process("Who did Sarah want to leave?")
        assert answer.contract.status == AnswerStatus.MULTIPLE_MATCHES
        assert {item.surface.lower() for item in answer.contract.values} == {
            "john",
            "mary",
        }
    finally:
        runtime.close()

@pytest.mark.parametrize(
    "text",
    (
        "Sarah told John and Mary to leave.",
        "Sarah plans to leave and Mary calls John.",
        "Sarah plans to leave and call Mary.",
        "Sarah and David plan to leave.",
    ),
)
def test_infinitival_coordination_fails_closed_until_staged_composition(text) -> None:
    _memory, result = parse(text)
    assert not result.events
    assert not result.infinitivals
    assert result.infinitival_ambiguities
    assert "coordination requires staged parsing" in result.infinitival_ambiguities[0].reason
    assert result.unresolved


@pytest.mark.parametrize(
    "statement,question,expected_status",
    (
        ("Sarah planned not to leave.", "Who planned not to leave?", AnswerStatus.ANSWERED),
        ("Sarah planned to leave.", "Who planned not to leave?", AnswerStatus.UNKNOWN),
        ("Sarah planned to leave.", "Who planned to leave?", AnswerStatus.ANSWERED),
        ("Sarah did not plan to leave.", "Who planned to leave?", AnswerStatus.UNKNOWN),
    ),
)
def test_wh_infinitival_questions_respect_scoped_polarity(
    statement,
    question,
    expected_status,
) -> None:
    runtime = ClankerLM(affect_backend=HeuristicAffectBackend())
    try:
        runtime.process(statement)
        answer = runtime.process(question)
        assert answer.contract.status == expected_status
    finally:
        runtime.close()
