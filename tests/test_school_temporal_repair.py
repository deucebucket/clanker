"""A narrow parser candidate must improve roles, not just echo expected words."""
from __future__ import annotations

import pytest

from clanker_lm import ClankerLM
from clanker_lm.memory import ConversationMemory
from clanker_lm import lexicon
from experiments.school.core import ref_name
from experiments.school.temporal_repair import TemporalSuffixParser, TemporalAttachmentAmbiguity


def runtime():
    r = ClankerLM()
    r.parser = TemporalSuffixParser()
    return r


@pytest.mark.parametrize("name", ["Alex", "John", "Casey", "Morgan Lee"])
@pytest.mark.parametrize("when", ["yesterday", "today", "tomorrow", "tonight"])
def test_source_and_time_are_separate(name, when):
    with runtime() as r:
        result = r.process(f"Jordan borrowed a bicycle from {name} {when}.")
        event = result.contract.proposition
        assert ref_name(r, event.arguments["source"]) == name.lower()
        assert event.arguments["time"].key == when
        assert not any(e.canonical_name.lower() == f"{name.lower()} {when}"
                       for e in r.memory.entities.values())
        result = r.process("When did Jordan borrow the bicycle?")
        assert result.contract.status.value == "answered"
        assert result.contract.values[0].key == when
        result = r.process("Who did Jordan borrow the bicycle from?")
        assert result.contract.status.value == "answered"
        assert ref_name(r, result.contract.values[0]) == name.lower()


@pytest.mark.parametrize("text,role,value", [
    ("Sarah traveled to Chicago yesterday.", "destination", "chicago"),
    ("Sarah worked in London today.", "location", "london"),
    ("Sarah opened the door with a key yesterday.", "method", "a key"),
    ("Sarah gave the book to John yesterday.", "recipient", "john"),
    ("The door was opened by Sarah yesterday.", "agent", "sarah"),
])
def test_other_pp_roles_share_the_same_boundary_fix(text, role, value):
    with runtime() as r:
        out = r.process(text)
        e = out.contract.proposition
        assert e is not None
        assert ref_name(r, e.arguments[role]) == value
        assert e.arguments["time"].key in {"yesterday", "today"}


@pytest.mark.parametrize("phrase", ["from Yesterday", "from Club Yesterday", 'from "Alex yesterday"',
                                      "from the club today report", "from Alex next week"])
def test_candidate_does_not_strip_proper_names_or_unsupported_boundaries(phrase):
    # Call the shared attachment operation directly to check the proposed
    # boundary, rather than depending on unrelated top-level quote parsing.
    parser = TemporalSuffixParser()
    args, unresolved, entities, diagnostics = {}, [], [], []
    parser._parse_predicate_tail("borrow", lexicon.tokenize("a bike " + phrase), args,
                                 ConversationMemory(), unresolved, entities, diagnostics, passive=False)
    assert not any(d.startswith("school-candidate:") for d in diagnostics)


def test_temporal_pp_is_not_stolen():
    with runtime() as r:
        out = r.process("Sarah worked until tomorrow.")
        assert not any(d.startswith("school-candidate:") for d in out.parse.diagnostics)


def test_negative_event_preserves_polarity_and_snapshot_bindings():
    with runtime() as r:
        out = r.process("Sarah did not borrow a coat from John yesterday.")
        assert out.contract.proposition.polarity is False
        snapshot = r.dumps()
    with ClankerLM.loads(snapshot) as resumed:
        resumed.parser = TemporalSuffixParser()
        result = resumed.process("Did Sarah borrow a coat from John yesterday?")
        assert result.contract.status.value == "false"
        event = resumed.memory.events[0]
        assert ref_name(resumed, event.arguments["source"]) == "john"
        assert event.arguments["time"].key == "yesterday"


def test_incompatible_time_evidence_fails_explicitly():
    parser = TemporalSuffixParser()
    with pytest.raises(TemporalAttachmentAmbiguity):
        parser._parse_predicate_tail("borrow", lexicon.tokenize("a bike on Monday from Alex yesterday"),
            {}, ConversationMemory(), [], [], [], passive=False)
