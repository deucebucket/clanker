"""Native recurrence contracts: inspect bindings, not canned response strings."""
from __future__ import annotations
import copy
import pytest
from clanker_lm import ClankerLM
from clanker_lm import lexicon
from clanker_lm.model import EventFrame, SemanticRef, EntityKind
from clanker_lm.recurrence import (RECURRENCE_ROLE, RECURRENCE_KIND, recurrence_of,
                                  scan_recurrence)
from clanker_lm.state_descriptions import state_components,original_state_rows,answer_from_appraisals


@pytest.mark.parametrize('who',['Jordan','Morgan','Riley'])
@pytest.mark.parametrize('description',['is angry again','is again angry','again feels angry',
                                        'feels angry again','was very angry again yesterday'])
def test_operator_is_resolved_before_participant_and_state_parsing(who,description):
    with ClankerLM() as r:
        parsed=r.parser.parse(who+' '+description+'.',r.memory)
        assert parsed.understood and len(parsed.events)==1
        e=parsed.events[0];parts=state_components(e)
        assert recurrence_of(e)=='again' and len(parts)==1
        assert parts[0].report.state_term=='angry'
        assert r.memory.get_entity(parts[0].report.entity_id).canonical_name==who
        assert 'again' not in (parts[0].event.arguments.get('state') or parts[0].event.arguments['value']).key
        assert all('again' not in n.canonical_name for n in r.memory.entities.values())
        assert not (e.arguments.get('time') and e.arguments['time'].key=='again')
        assert any('typed state recurrence' in d for d in parsed.diagnostics)


@pytest.mark.parametrize('text',['Jordan is not angry again.','Jordan is no longer angry again.',
    'Jordan is angry again again.','Jordan again again feels angry.','Jordan is again.',
    'Jordan again is angry.'])
def test_unsupported_operator_scope_is_not_flattened_into_a_positive_state(text):
    with ClankerLM() as r:
        parsed=r.parser.parse(text,r.memory)
        assert not parsed.understood
        assert not any(state_components(e) for e in parsed.events)


def test_capitalized_word_or_quoted_name_is_not_removed_as_an_operator():
    for text in ['Jordan is Again','Jordan feels Again','Again is Jordan']:
        items=lexicon.tokenize(text)
        vi=next(i for i,t in enumerate(items) if lexicon.lemma(t.norm) in {'be','feel'})
        scan=scan_recurrence(items,lexicon.lemma(items[vi].norm),vi)
        assert not scan.marker and scan.tokens==tuple(items)


@pytest.mark.parametrize('mutation',['kind','spelling','negative','predicate','aspect','cessation'])
def test_recurrence_metadata_is_validated_at_each_semantic_boundary(mutation):
    e=EventFrame('be',{'subject':SemanticRef.entity('j','Jordan'),
       'value':SemanticRef.literal('angry','angry',EntityKind.ABSTRACT),
       'attribute':SemanticRef.literal('state','state'),
       RECURRENCE_ROLE:SemanticRef.literal(RECURRENCE_KIND,'again',EntityKind.ABSTRACT)})
    if mutation=='kind': e.arguments[RECURRENCE_ROLE]=SemanticRef.entity('recurred','again')
    if mutation=='spelling': e.arguments[RECURRENCE_ROLE]=SemanticRef.literal('recurred','still')
    if mutation=='negative':e.polarity=False
    if mutation=='predicate':e.predicate='buy'
    if mutation=='aspect':e.aspect='progressive'
    if mutation=='cessation':e.arguments['state_change']=SemanticRef.literal('ceased','no_longer')
    with pytest.raises(ValueError):recurrence_of(e)
    assert not state_components(e)


@pytest.mark.parametrize('text,scope,source',[
    ('Jordan was angry again yesterday.','historical','user'),
    ('Jordan might be angry again.','projected_or_modal','user'),
    ('Jordan will be angry again tomorrow.','projected_or_modal','user'),
    ('Sarah said Jordan was angry again yesterday.','historical','attributed')])
def test_recurrence_preserves_history_modality_and_reporter_scope(text,scope,source):
    with ClankerLM() as r:
        p=r.parser.parse(text,r.memory)
        e=next(e for e in p.events if e.predicate in {'be','feel'})
        c=state_components(e)[0]
        assert c.report.temporal_scope==scope and e.source.value==source
        assert recurrence_of(e)=='again'
        if source=='attributed':assert e.discourse_role=='content'


def test_every_state_observation_keeps_its_original_time_and_content():
    # Exercise native parser+store without invoking the legacy generator on
    # new recurrence roles. Current production activation is a separate gate.
    with ClankerLM() as r:
        originals=[]
        for text in ['Jordan is angry.','Jordan is no longer angry.','Jordan is angry again.',
                     'Jordan is no longer angry.','Jordan is angry again.','Jordan is angry.']:
            r.memory.begin_turn()
            p=r.parser.parse(text,r.memory)
            stored=r.memory.add_event(p.events[0])
            assert all(r.memory.get_event(e['event_id']).to_dict()==e for e in originals)
            originals.append(copy.deepcopy(stored.to_dict()))
        assert len(r.memory.events)==6 and len({e.event_id for e in r.memory.events})==6
        rows=original_state_rows(r.memory)
        assert len(rows)==6 and sum(row['state_recurrence']=='again' for row in rows)==2
        assert [e.turn_index for e in r.memory.events]==list(range(1,7))
        restored=type(r.memory).from_dict(r.memory.to_dict())
        assert restored.to_dict()==r.memory.to_dict()


def test_same_turn_state_storage_remains_idempotent():
    with ClankerLM() as r:
        r.memory.begin_turn();p=r.parser.parse('Jordan is angry again.',r.memory)
        one=r.memory.add_event(p.events[0]);two=r.memory.add_event(p.events[0])
        assert one.event_id==two.event_id and len(r.memory.events)==1


def test_nonstate_fact_deduplication_is_unchanged():
    with ClankerLM() as r:
        for text in ['Sarah bought a Honda.','Sarah bought a Honda.']:
            r.memory.begin_turn();p=r.parser.parse(text,r.memory);r.memory.add_event(p.events[0])
        assert len(r.memory.events)==1


def test_a_recurrent_report_does_not_invent_a_prior_event():
    with ClankerLM() as r:
        r.memory.begin_turn();p=r.parser.parse('Jordan is angry again.',r.memory)
        r.memory.add_event(p.events[0])
        assert len(r.memory.events)==1
        assert len(original_state_rows(r.memory))==1


def test_conjunction_recurrence_is_not_distributed_without_scope_resolution():
    with ClankerLM() as r:
        p=r.parser.parse('Jordan is angry and sad again.',r.memory)
        assert any(RECURRENCE_ROLE in e.arguments for e in p.events)
        assert not any(state_components(e) for e in p.events)


def test_an_explicit_old_observation_id_cannot_be_reused_for_a_new_turn():
    with ClankerLM() as r:
        r.memory.begin_turn()
        p=r.parser.parse('Jordan is angry again.',r.memory)
        old=r.memory.add_event(p.events[0]); original=copy.deepcopy(old.to_dict())
        r.memory.begin_turn()
        with pytest.raises(ValueError,match='identity cannot be reused'):
            r.memory.add_event(old.copy(turn_index=r.memory.turn_index))
        assert old.to_dict()==original and len(r.memory.events)==1
