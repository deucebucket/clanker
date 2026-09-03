from __future__ import annotations

import pytest

from clanker_lm import ClankerLM, HeuristicAffectBackend
from clanker_lm.model import AnswerStatus, SourceKind, TruthValue


def say(runtime: ClankerLM, *messages: str):
    return [runtime.process(message) for message in messages]


def test_basic_fact_can_be_queried_from_each_open_slot(runtime):
    say(runtime, "My sister bought a used Honda yesterday.")
    who, what, when = say(runtime, "Who bought the Honda?", "What did she buy?", "When did she buy it?")
    assert who.contract.status == AnswerStatus.ANSWERED
    assert what.contract.status == AnswerStatus.ANSWERED
    assert when.contract.status == AnswerStatus.ANSWERED
    assert "sister" in who.response.lower()
    assert "honda" in what.response.lower()
    assert "yesterday" in when.response.lower()


def test_question_reconstruction_preserves_full_proposition(runtime):
    say(runtime, "Sarah gave John a book yesterday.")
    answers = say(
        runtime,
        "Who gave John a book?",
        "What did Sarah give John?",
        "Who did Sarah give the book to?",
        "When did Sarah give John the book?",
    )
    for answer in answers:
        assert answer.contract.status == AnswerStatus.ANSWERED
        lower = answer.response.lower()
        assert "sarah" in lower and "john" in lower and "book" in lower and "yesterday" in lower


def test_yes_no_true_false_unknown_are_not_collapsed(runtime):
    say(runtime, "Sarah did not buy the car.")
    false_answer = runtime.process("Did Sarah buy the car?")
    true_answer = runtime.process("Did Sarah not buy the car?")
    unknown_answer = runtime.process("Did Mary buy the car?")
    assert false_answer.contract.status == AnswerStatus.FALSE
    assert false_answer.contract.truth == TruthValue.FALSE
    assert false_answer.response.startswith("No.")
    assert true_answer.contract.status == AnswerStatus.TRUE
    assert true_answer.response.startswith("Yes.")
    assert unknown_answer.contract.status == AnswerStatus.UNKNOWN
    assert "don't know" in unknown_answer.response.lower()


def test_absence_of_evidence_never_becomes_false(runtime):
    say(runtime, "Sarah bought the car.")
    result = runtime.process("Did Mary buy the car?")
    assert result.contract.status == AnswerStatus.UNKNOWN
    assert "convert_absence_of_evidence_to_false" in result.contract.forbidden_claims


def test_explicit_conflict_is_reported(runtime):
    say(runtime, "Sarah bought the car.", "Sarah did not buy the car.")
    result = runtime.process("Did Sarah buy the car?")
    assert result.contract.status == AnswerStatus.CONFLICT
    assert result.contract.truth == TruthValue.CONFLICT
    assert "conflict" in result.response.lower()


def test_unknown_reason_reports_known_base_without_invention(runtime):
    say(runtime, "Sarah bought the car.")
    result = runtime.process("Why did Sarah buy the car?")
    assert result.contract.status == AnswerStatus.UNKNOWN
    assert result.contract.proposition is not None
    assert "haven't told" in result.response.lower()
    assert any(item.startswith("invent_") for item in result.contract.forbidden_claims)


def test_reason_becomes_answerable_when_explicitly_supplied(runtime):
    say(runtime, "Sarah bought the car because her old car broke down.")
    result = runtime.process("Why did Sarah buy the car?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "old car broke down" in result.response.lower()


def test_finite_cause_supports_questions_about_subordinate_event(runtime):
    say(runtime, "Sarah left because the argument upset her.")
    cause = runtime.process("What upset Sarah?")
    patient = runtime.process("Who did the argument upset?")
    assert cause.contract.status == AnswerStatus.ANSWERED
    assert patient.contract.status == AnswerStatus.ANSWERED
    assert "argument" in cause.response.lower()
    assert "sarah" in patient.response.lower()


def test_where_binds_destination_location_and_source(runtime):
    say(runtime, "Sarah traveled from Chicago to St Louis on Monday.")
    source = runtime.process("Where did Sarah travel from?")
    destination = runtime.process("Where did Sarah travel?")
    when = runtime.process("When did Sarah travel?")
    assert all(item.contract.status == AnswerStatus.ANSWERED for item in (source, destination, when))
    assert "chicago" in source.response.lower()
    assert "st louis" in destination.response.lower()
    assert "monday" in when.response.lower()


def test_how_method_preserves_preposition(runtime):
    say(runtime, "Sarah opened the door with a key.")
    result = runtime.process("How did Sarah open the door?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "with a key" in result.response.lower()
    assert "openned" not in result.response.lower()


def test_how_process_uses_explicit_process_evidence(runtime):
    say(runtime, "The engine calculates valence through nine stages.")
    result = runtime.process("How does the engine calculate valence?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "through nine stages" in result.response.lower()


def test_quantity_is_realized_exactly_once(runtime):
    say(runtime, "Sarah bought three cars.")
    result = runtime.process("How many cars did Sarah buy?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert result.response.lower().count("three") == 1
    assert "three cars" in result.response.lower()


def test_attribute_from_copula(runtime):
    say(runtime, "Sarah is tall.")
    result = runtime.process("How tall is Sarah?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "tall" in result.response.lower()


def test_attribute_from_noun_modifier(runtime):
    say(runtime, "Sarah bought a red car.")
    result = runtime.process("What color is the car?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert result.contract.values[0].key == "red"
    assert "red" in result.response.lower()


def test_owner_scoped_body_part_attribute(runtime):
    say(runtime, "Sarah has blue eyes.")
    result = runtime.process("What color are Sarah's eyes?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "blue" in result.response.lower()


def test_ownership_queries_normalize_have_own_and_belong(runtime):
    say(runtime, "The red car belongs to Sarah.")
    whose = runtime.process("Whose car is the red car?")
    yes_no = runtime.process("Does Sarah have the red car?")
    assert whose.contract.status == AnswerStatus.ANSWERED
    assert "sarah" in whose.response.lower()
    assert yes_no.contract.status == AnswerStatus.TRUE


def test_passive_and_active_questions_share_event_semantics(runtime):
    say(runtime, "The coat was bought by Sarah.")
    who = runtime.process("Who bought the coat?")
    what = runtime.process("What was bought by Sarah?")
    assert who.contract.status == AnswerStatus.ANSWERED
    assert what.contract.status == AnswerStatus.ANSWERED
    assert "sarah" in who.response.lower()
    assert "coat" in what.response.lower()


def test_coordinated_assertions_become_independently_queryable(runtime):
    say(runtime, "Sarah bought a car and John bought a bike.")
    sarah = runtime.process("What did Sarah buy?")
    john = runtime.process("What did John buy?")
    assert "car" in sarah.response.lower() and "bike" not in sarah.response.lower()
    assert "bike" in john.response.lower() and "car" not in john.response.lower()


def test_multiple_values_trigger_probe_not_arbitrary_choice(runtime):
    say(runtime, "Sarah bought a car.", "Sarah bought a bike.")
    result = runtime.process("What did Sarah buy?")
    assert result.contract.status == AnswerStatus.MULTIPLE_MATCHES
    assert "more than one" in result.response.lower()
    assert "car" in result.response.lower() and "bike" in result.response.lower()


def test_missing_pronoun_halts_normal_storage_and_probes(runtime):
    result = runtime.process("She bought a car.")
    assert result.contract.status == AnswerStatus.MISSING_REFERENCE
    assert result.gates.requires_probe
    assert result.response.endswith("?")
    assert not runtime.memory.events


def test_ambiguous_pronoun_names_candidates(runtime):
    say(runtime, "Sarah met Mary.")
    result = runtime.process("She left.")
    assert result.contract.status == AnswerStatus.AMBIGUOUS_REFERENCE
    assert "sarah" in result.response.lower() and "mary" in result.response.lower()
    assert not any(event.predicate == "leave" for event in runtime.memory.events)


def test_recent_antecedent_resolves_deterministically(runtime):
    say(runtime, "Sarah called John.", "Mary arrived.", "She left.")
    result = runtime.process("Who left?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "mary" in result.response.lower()


def test_what_happened_returns_latest_event(runtime):
    say(runtime, "Sarah arrived.", "John left.")
    result = runtime.process("What happened?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "john left" in result.response.lower()


def test_what_did_subject_do_uses_event_query(runtime):
    say(runtime, "Sarah did the work.")
    result = runtime.process("What did Sarah do?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "did the work" in result.response.lower()


def test_lexical_did_who_question(runtime):
    say(runtime, "Sarah did the work.")
    result = runtime.process("Who did the work?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "sarah" in result.response.lower()


def test_present_perfect_realization(runtime):
    say(runtime, "Sarah has bought the car.")
    result = runtime.process("Has Sarah bought the car?")
    assert result.contract.status == AnswerStatus.TRUE
    assert "has bought" in result.response.lower()


def test_social_conventions_do_not_query_memory(runtime):
    wellbeing = runtime.process("How are you?")
    activity = runtime.process("What are you doing?")
    assert wellbeing.contract.response_goal == "social"
    assert activity.contract.response_goal == "social"
    assert "ready to work" in wellbeing.response.lower()
    assert "working" in activity.response.lower()


def test_greeting_is_handled_without_fake_fact(runtime):
    result = runtime.process("Hello")
    assert result.contract.status == AnswerStatus.ACKNOWLEDGED
    assert result.contract.response_goal == "social"
    assert not runtime.memory.events


def test_external_learning_preserves_provenance_and_certainty(runtime):
    learned = runtime.learn("The launch is on Monday.", source=SourceKind.RETRIEVED, certainty=211)
    assert learned and learned[0].source == SourceKind.RETRIEVED
    assert learned[0].certainty == 211
    result = runtime.process("When is the launch?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert result.contract.source == SourceKind.RETRIEVED
    assert result.contract.certainty == 211


def test_learning_rejects_unresolved_or_unsupported_text(runtime):
    with pytest.raises(ValueError):
        runtime.learn("She bought it.")
    with pytest.raises(ValueError):
        runtime.learn("hmm")


def test_runtime_snapshot_restores_semantic_and_affective_state(runtime, tmp_path):
    say(runtime, "Sarah bought the car.")
    path = tmp_path / "session.json"
    runtime.save(path)
    loaded = ClankerLM.load(path, affect_backend=HeuristicAffectBackend())
    try:
        result = loaded.process("Who bought the car?")
        assert result.contract.status == AnswerStatus.ANSWERED
        assert "sarah" in result.response.lower()
        assert loaded.observed_state != loaded.predicted_state or loaded.last_result is not None
    finally:
        loaded.close()


def test_explain_trace_contains_parse_contract_gates_and_candidate_scores(runtime):
    runtime.process("Sarah bought the car.")
    runtime.process("Who bought the car?")
    trace = runtime.explain_last()
    assert trace["parse"]["question"]["requested_role"] == "agent"
    assert trace["answer_contract"]["status"] == "answered"
    assert trace["candidate_ranking"]
    assert all("score" in item for item in trace["candidate_ranking"])


def test_resolved_question_pronouns_refresh_discourse_focus(runtime):
    say(runtime, "My sister bought a used Honda yesterday.")
    say(runtime, "She bought it because her old car broke down.")
    first = runtime.process("Why did she buy it?")
    assert first.contract.status == AnswerStatus.ANSWERED
    second = runtime.process("Did my mother buy it?")
    assert second.contract.status == AnswerStatus.UNKNOWN
    assert second.contract.status != AnswerStatus.AMBIGUOUS_REFERENCE
