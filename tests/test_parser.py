from __future__ import annotations

import pytest

from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (
    EntityKind,
    HowKind,
    QuestionKind,
    RefKind,
    SpeechAct,
    WhyKind,
)
from clanker_lm.parser import SemanticParser


def parse(text: str, memory: ConversationMemory | None = None):
    memory = memory or ConversationMemory()
    memory.begin_turn()
    return SemanticParser().parse(text, memory), memory


def first_event(text: str):
    result, memory = parse(text)
    assert result.events, result.diagnostics
    return result.events[0], result, memory


def test_basic_transitive_statement_has_typed_roles():
    event, result, memory = first_event("My sister bought a used Honda yesterday.")
    assert result.speech_act == SpeechAct.ASSERT
    assert event.predicate == "buy"
    assert event.tense == "past"
    assert event.arguments["agent"].kind == RefKind.ENTITY
    sister = memory.get_entity(event.arguments["agent"].key)
    assert sister is not None and sister.relation == "sister"
    assert event.arguments["patient"].surface == "a used Honda"
    assert event.arguments["time"].key == "yesterday"


def test_yoda_form_normalizes_to_same_semantic_frame():
    normal, _, _ = first_event("My sister pissed me off.")
    yoda, _, _ = first_event("Pissed me off, my sister did.")
    assert normal.predicate == yoda.predicate == "anger"
    assert set(normal.arguments) == set(yoda.arguments)
    assert normal.arguments["agent"].surface.lower().endswith("sister")
    assert yoda.arguments["agent"].surface.lower().endswith("sister")
    assert normal.arguments["patient"].key == yoda.arguments["patient"].key == "user"


def test_phrasal_predicate_removes_particle_from_object_and_tracks_again():
    event, _, _ = first_event("My sister pissed me off again.")
    assert event.predicate == "anger"
    assert event.arguments["patient"].key == "user"
    assert event.arguments["time"].key == "again"


def test_passive_voice_recovers_agent_and_patient():
    event, _, memory = first_event("The coat was bought by Sarah.")
    assert event.predicate == "buy"
    assert event.arguments["patient"].value_type == EntityKind.THING
    assert memory.get_entity(event.arguments["agent"].key).canonical_name == "Sarah"


def test_negative_copula_sets_polarity_false():
    event, _, _ = first_event("Sarah was not a nurse.")
    assert event.predicate == "be"
    assert event.polarity is False


def test_negative_do_support_sets_polarity_false():
    event, _, _ = first_event("Sarah did not buy the car.")
    assert event.predicate == "buy"
    assert event.polarity is False


def test_present_perfect_is_not_reduced_to_simple_past():
    event, _, _ = first_event("Sarah has bought the car.")
    assert event.predicate == "buy"
    assert event.aspect == "perfect"
    assert event.tense == "present"


def test_finite_causal_subclause_becomes_second_event():
    result, memory = parse("Sarah left because the argument upset her.")
    assert [event.predicate for event in result.events] == ["leave", "upset"]
    leave, upset = result.events
    assert leave.arguments["motive"].key == "the argument upset her"
    assert upset.arguments["patient"].key == leave.arguments["agent"].key


def test_nonfinite_purpose_stays_purpose_not_asserted_event():
    result, _ = parse("Sarah went to the store to buy groceries.")
    assert len(result.events) == 1
    event = result.events[0]
    assert event.predicate == "go"
    assert event.arguments["destination"].surface == "the store"
    assert event.arguments["purpose"].key == "buy groceries"


def test_location_and_its_preposition_are_preserved():
    event, _, _ = first_event("Sarah lives in Chicago.")
    assert event.arguments["location"].surface == "Chicago"
    assert event.arguments["location_preposition"].key == "in"


def test_method_and_its_preposition_are_preserved():
    event, _, _ = first_event("Sarah opened the door with a key.")
    assert event.arguments["method"].surface == "a key"
    assert event.arguments["method_preposition"].key == "with"


def test_clock_time_is_not_misparsed_as_location_or_quantity():
    event, _, _ = first_event("The meeting is at three.")
    assert event.predicate == "be"
    assert event.arguments["time"].key == "three"
    assert event.arguments["time_preposition"].key == "at"
    assert "quantity" not in event.arguments


def test_have_is_normalized_to_ownership_and_modifiers_are_attributes():
    event, _, memory = first_event("Sarah has blue eyes.")
    assert event.predicate == "own"
    eyes = memory.get_entity(event.arguments["patient"].key)
    assert eyes is not None
    assert eyes.owner_id == event.arguments["possessor"].key
    assert eyes.attributes["color"] == "blue"


def test_belongs_to_is_normalized_to_ownership():
    event, _, _ = first_event("The red car belongs to Sarah.")
    assert event.predicate == "own"
    assert set(event.arguments) >= {"patient", "possessor"}


def test_quantity_is_a_separate_typed_slot():
    event, _, _ = first_event("Sarah bought three cars.")
    assert event.arguments["quantity"].key == "3"
    assert event.arguments["patient"].surface == "three cars"


def test_independent_coordinated_clauses_are_split():
    result, _ = parse("Sarah bought a car and John bought a bike.")
    assert len(result.events) == 2
    assert [event.predicate for event in result.events] == ["buy", "buy"]
    assert result.events[0].arguments["agent"].key != result.events[1].arguments["agent"].key


def test_compound_subject_is_not_split_as_two_clauses():
    result, memory = parse("Sarah and Mary bought cars.")
    assert len(result.events) == 1
    entity = memory.get_entity(result.events[0].arguments["agent"].key)
    assert entity is not None and entity.number.value == "plural"


def test_missing_pronoun_is_explicitly_unresolved():
    result, _ = parse("She bought a car.")
    assert result.unresolved
    assert result.unresolved[0].surface == "She"


def test_who_subject_question_opens_agent_slot():
    result, _ = parse("Who bought the car?")
    q = result.question
    assert q and q.kind == QuestionKind.WHO
    assert q.requested_role == "agent"
    assert q.event.arguments["agent"].is_variable


def test_who_object_question_opens_patient_slot():
    result, _ = parse("Who did Sarah call?")
    q = result.question
    assert q and q.requested_role == "patient"
    assert q.event.arguments["patient"].is_variable


def test_who_recipient_question_opens_recipient_slot():
    result, _ = parse("Who did Sarah give the book to?")
    q = result.question
    assert q and q.requested_role == "recipient"


def test_who_did_the_work_treats_did_as_lexical_predicate():
    result, _ = parse("Who did the work?")
    q = result.question
    assert q and q.event.predicate == "do"
    assert q.requested_role == "agent"


def test_what_subject_question_opens_nonhuman_agent_slot():
    result, _ = parse("What broke the window?")
    q = result.question
    assert q and q.requested_role == "agent"
    assert q.answer_type == EntityKind.THING


def test_what_object_question_opens_patient_slot():
    result, _ = parse("What did Sarah buy?")
    q = result.question
    assert q and q.requested_role == "patient"


def test_what_is_question_opens_value_slot():
    result, _ = parse("What is Sarah?")
    q = result.question
    assert q and q.event.predicate == "be"
    assert q.requested_role == "value"


def test_what_did_subject_do_is_event_query():
    result, _ = parse("What did Sarah do?")
    q = result.question
    assert q and q.kind == QuestionKind.WHAT_HAPPENED
    assert q.requested_role == "event"


def test_yes_no_question_contains_closed_proposition():
    result, _ = parse("Did Sarah buy the car?")
    q = result.question
    assert q and q.kind == QuestionKind.YES_NO
    assert not q.event.variable_roles()


def test_where_movement_question_requests_destination():
    result, _ = parse("Where did Sarah go?")
    q = result.question
    assert q and q.requested_role == "destination"


def test_where_from_question_requests_source():
    result, _ = parse("Where did Sarah travel from?")
    q = result.question
    assert q and q.requested_role == "source"


def test_where_static_question_requests_location():
    result, _ = parse("Where does Sarah work?")
    q = result.question
    assert q and q.requested_role == "location"


def test_why_subtypes_are_typed():
    physical, _ = parse("Why did the window break?")
    purpose, _ = parse("Why did Sarah go to the store?")
    normative, _ = parse("Why should I apologize?")
    assert physical.question and physical.question.why_kind == WhyKind.CAUSE
    assert purpose.question and purpose.question.why_kind == WhyKind.PURPOSE
    assert normative.question and normative.question.why_kind == WhyKind.JUSTIFICATION


def test_how_subtypes_are_typed():
    method, _ = parse("How did Sarah open the door?")
    process, _ = parse("How does the engine calculate valence?")
    degree, _ = parse("How tall is Sarah?")
    assert method.question and method.question.how_kind == HowKind.METHOD
    assert process.question and process.question.how_kind == HowKind.PROCESS
    assert degree.question and degree.question.how_kind == HowKind.DEGREE


def test_how_many_opens_quantity_slot():
    result, _ = parse("How many cars did Sarah buy?")
    q = result.question
    assert q and q.kind == QuestionKind.HOW_MANY
    assert q.requested_role == "quantity"


def test_attribute_question_has_dimension_and_value_hole():
    result, _ = parse("What color is the car?")
    q = result.question
    assert q and q.event.predicate == "attribute"
    assert q.event.arguments["attribute"].key == "color"
    assert q.event.arguments["value"].is_variable


def test_genitive_attribute_subject_is_owner_scoped():
    memory = ConversationMemory()
    memory.begin_turn()
    parser = SemanticParser()
    asserted = parser.parse("Sarah has blue eyes.", memory)
    for event in asserted.events:
        memory.add_event(event)
    memory.begin_turn()
    result = parser.parse("What color are Sarah's eyes?", memory)
    q = result.question
    assert q
    eyes = memory.get_entity(q.event.arguments["subject"].key)
    assert eyes is not None and eyes.owner_id is not None


def test_which_question_retains_selection_class():
    result, _ = parse("Which car did Sarah buy?")
    q = result.question
    assert q and q.kind == QuestionKind.WHICH
    assert q.focus_surface.lower() == "car"


def test_whose_question_maps_to_ownership():
    result, _ = parse("Whose car is this?")
    q = result.question
    assert q and q.kind == QuestionKind.WHOSE
    assert q.event.predicate == "own"
    assert q.requested_role == "possessor"


def test_social_question_is_pragmatic_not_literal():
    result, _ = parse("How are you?")
    assert result.question and result.question.social_convention == "wellbeing_check"


def test_greeting_is_speech_act_without_fake_event():
    result, _ = parse("Hello")
    assert result.speech_act == SpeechAct.GREET
    assert not result.events
