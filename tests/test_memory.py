from __future__ import annotations

from clanker_lm.memory import ConversationMemory
from clanker_lm.model import EntityKind, EventFrame, Gender, SemanticRef, SourceKind


def test_relation_entities_are_deduplicated_per_owner():
    memory = ConversationMemory()
    memory.begin_turn()
    one = memory.get_or_create_relation("user", "mother", surface="my mother")
    two = memory.get_or_create_relation("user", "mom", surface="my mom")
    assert one.entity_id == two.entity_id
    assert one.relation == "mom"


def test_alias_head_resolves_modified_object():
    memory = ConversationMemory()
    memory.begin_turn()
    honda = memory.get_or_create_named_entity("used Honda", kind=EntityKind.THING)
    resolved = memory.find_by_alias("the Honda", EntityKind.THING)
    assert resolved.resolved and resolved.entity.entity_id == honda.entity_id


def test_pronoun_without_antecedent_is_missing():
    memory = ConversationMemory()
    memory.begin_turn()
    result = memory.resolve_pronoun("she", EntityKind.PERSON)
    assert result.status == "missing"


def test_same_turn_compatible_people_are_ambiguous():
    memory = ConversationMemory()
    memory.begin_turn()
    memory.get_or_create_named_entity("Sarah", gender=Gender.FEMALE, role_salience=1.0)
    memory.get_or_create_named_entity("Mary", gender=Gender.FEMALE, role_salience=1.0)
    result = memory.resolve_pronoun("she", EntityKind.PERSON)
    assert result.status == "ambiguous"


def test_recent_turn_beats_accumulated_old_salience_for_pronouns():
    memory = ConversationMemory()
    memory.begin_turn()
    sarah = memory.get_or_create_named_entity("Sarah", gender=Gender.FEMALE, role_salience=10.0)
    memory.begin_turn()
    mary = memory.get_or_create_named_entity("Mary", gender=Gender.FEMALE, role_salience=0.5)
    result = memory.resolve_pronoun("she", EntityKind.PERSON)
    assert result.resolved and result.entity.entity_id == mary.entity_id


def test_exact_repeated_fact_updates_instead_of_duplicating():
    memory = ConversationMemory()
    memory.begin_turn()
    sarah = memory.get_or_create_named_entity("Sarah")
    car = memory.get_or_create_named_entity("car", kind=EntityKind.THING)
    event = EventFrame("buy", {"agent": sarah.to_ref(), "patient": car.to_ref()}, tense="past")
    one = memory.add_event(event)
    two = memory.add_event(event.copy(certainty=250))
    assert one.event_id == two.event_id
    assert len(memory.events) == 1
    assert memory.events[0].certainty == 250


def test_positive_and_negative_propositions_are_both_retained():
    memory = ConversationMemory()
    memory.begin_turn()
    sarah = memory.get_or_create_named_entity("Sarah")
    car = memory.get_or_create_named_entity("car", kind=EntityKind.THING)
    args = {"agent": sarah.to_ref(), "patient": car.to_ref()}
    memory.add_event(EventFrame("buy", args, polarity=True))
    memory.add_event(EventFrame("buy", args, polarity=False))
    assert len(memory.events) == 2


def test_open_slot_event_matching_ignores_requested_role_only():
    memory = ConversationMemory()
    memory.begin_turn()
    sarah = memory.get_or_create_named_entity("Sarah")
    car = memory.get_or_create_named_entity("car", kind=EntityKind.THING)
    memory.add_event(EventFrame("buy", {"agent": sarah.to_ref(), "patient": car.to_ref()}, tense="past"))
    query = EventFrame("buy", {"agent": sarah.to_ref(), "patient": SemanticRef.variable("patient")}, tense="past")
    matches = memory.related_events(query, "patient")
    assert len(matches) == 1
    assert matches[0].event.arguments["patient"].key == car.entity_id


def test_memory_snapshot_round_trip_preserves_entities_events_and_counters(tmp_path):
    memory = ConversationMemory()
    memory.begin_turn()
    sarah = memory.get_or_create_named_entity("Sarah", gender=Gender.FEMALE)
    event = memory.add_event(EventFrame("arrive", {"patient": sarah.to_ref()}, source=SourceKind.RETRIEVED, certainty=210))
    path = tmp_path / "memory.json"
    memory.save(path)
    loaded = ConversationMemory.load(path)
    assert loaded.to_dict() == memory.to_dict()
    loaded.begin_turn()
    next_entity = loaded.get_or_create_named_entity("John")
    assert next_entity.entity_id != sarah.entity_id
    assert loaded.events[0].event_id == event.event_id


from clanker_lm.model import AnswerStatus


def test_subordinate_clause_entities_do_not_steal_main_object_pronoun(runtime):
    runtime.process("My sister bought a used Honda yesterday.")
    runtime.process("She bought it because her old car broke down.")
    result = runtime.process("Why did she buy it?")
    assert result.contract.status == AnswerStatus.ANSWERED
    assert "broke down" in result.response.lower()
