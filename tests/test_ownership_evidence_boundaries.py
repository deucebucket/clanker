"""Native ownership truth boundaries; no decoder or appraisal-graph dependency.

All sentences are authored development inputs. Expected roles/statuses are
scenario assertions, never whole-response strings or model-generated labels.
"""
from __future__ import annotations

import pytest

from clanker_lm import ClankerLM
from clanker_lm.lexicon import detect_tense
from clanker_lm.model import RefKind


def object_named(runtime: ClankerLM, name: str):
    return next(entity for entity in runtime.memory.entities.values()
                if entity.canonical_name == name)


@pytest.mark.parametrize("person", ["Jordan", "Morgan", "Taylor", "Alice"])
@pytest.mark.parametrize("item", ["bag", "coat", "camera", "book"])
def test_an_ownership_question_does_not_assign_the_owner(person, item):
    with ClankerLM() as runtime:
        runtime.process(f"{person} borrowed a {item} yesterday.")
        entity = object_named(runtime, item)
        before = entity.owner_id
        answer = runtime.process(f"Does {person} own the {item}?")
        assert answer.contract.status.value == "unknown"
        assert entity.owner_id == before is None
        proposed_owner = answer.contract.question.event.arguments["possessor"]
        assert proposed_owner.kind == RefKind.ENTITY
        assert f"{proposed_owner.key}:{item}" not in entity.aliases
        followup = runtime.process(f"Whose {item} is the {item}?")
        assert followup.contract.status.value == "unknown"
        assert not followup.contract.values


@pytest.mark.parametrize("text", [
    "Alex does not own the bag.", "Does Alex own the bag?",
    "Sarah said Alex owns the bag.", "Alex might own the bag.",
    "If Alex owns the bag, Sarah leaves.", "Alex owned the bag.",
])
def test_only_asserted_current_outer_possession_can_update_association(text):
    with ClankerLM() as runtime:
        runtime.parser.parse(text, runtime.memory)
        entity = object_named(runtime, "bag")
        assert entity.owner_id is None
        assert not any(alias.startswith("alex_") and ":bag" in alias
                       for alias in entity.aliases)


@pytest.mark.parametrize("person", ["Alex", "Morgan", "Robin"])
def test_supported_ownership_uses_evidence_even_after_another_ownership_question(person):
    with ClankerLM() as runtime:
        runtime.process(f"{person} owns the bag.")
        actual_owner = object_named(runtime, "bag").owner_id
        runtime.process("Does Jordan own the bag?")
        assert object_named(runtime, "bag").owner_id == actual_owner
        answer = runtime.process("Who owns the bag?")
        assert answer.contract.status.value == "answered"
        assert answer.contract.proposition.predicate == "own"
        assert answer.contract.evidence
        assert runtime.memory.get_entity(answer.contract.values[0].key).canonical_name == person


@pytest.mark.parametrize("assertion,question", [
    ("Alex does not own the bag.", "Who owns the bag?"),
    ("Alex did not buy the bag.", "What did Alex buy?"),
    ("Alex did not open the bag.", "What did Alex open?"),
])
def test_negative_only_evidence_cannot_fill_a_positive_wh_request(assertion, question):
    with ClankerLM() as runtime:
        runtime.process(assertion)
        answer = runtime.process(question)
        assert answer.contract.status.value == "unknown"
        assert not answer.contract.values
        assert answer.contract.evidence


@pytest.mark.parametrize("denier,status", [("Alex", "conflict"), ("Jordan", "answered")])
def test_wh_conflict_requires_the_same_closed_proposition(denier, status):
    with ClankerLM() as runtime:
        runtime.process("Alex owns the bag.")
        runtime.process(f"{denier} does not own the bag.")
        answer = runtime.process("Who owns the bag?")
        assert answer.contract.status.value == status
        if status == "answered":
            assert runtime.memory.get_entity(answer.contract.values[0].key).canonical_name == "Alex"


@pytest.mark.parametrize("word,auxiliary,tense", [
    ("has", None, "present"), ("does", None, "present"),
    ("goes", None, "present"), ("had", None, "past"),
    ("read", "does", "present"), ("read", "will", "future"),
    ("read", "did", "past"),
])
def test_finite_auxiliary_and_present_irregulars_determine_tense(word, auxiliary, tense):
    assert detect_tense(word, auxiliary) == tense


def test_positive_body_part_association_and_scope_survive_snapshot():
    with ClankerLM() as runtime:
        out = runtime.process("Sarah has blue eyes.")
        event = out.parse.events[0]
        assert event.tense == "present"
        eyes = runtime.memory.get_entity(event.arguments["patient"].key)
        assert eyes.owner_id == event.arguments["possessor"].key
        runtime.process("Does John own the eyes?")
        frozen = runtime.dumps()
        owner = eyes.owner_id
    with ClankerLM.loads(frozen) as restored:
        assert object_named(restored, "blue eyes").owner_id == owner
        answer = restored.process("Who owns the eyes?")
        assert answer.contract.status.value == "answered"
        assert answer.contract.values[0].key == owner
