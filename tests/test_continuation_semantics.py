"""Inspect phase bindings, qualification and source; do not grade fluent echoes."""
from __future__ import annotations
import copy
import pytest
from clanker_lm import ClankerLM,lexicon
from clanker_lm.model import EventFrame,SemanticRef,EntityKind
from clanker_lm.continuation import (CONTINUATION_ROLE,CONTINUATION_KIND,DISCOURSE_ROLE,
                                    continuation_of,discourse_of,scan_continuation)
from clanker_lm.recurrence import RECURRENCE_ROLE,recurrence_of
from clanker_lm.state_descriptions import state_components,original_state_rows


@pytest.mark.parametrize('who',['Jordan','Morgan','Riley'])
@pytest.mark.parametrize('phrase',['is still angry','still feels sad','feels still calm',
                                   'was still really angry yesterday'])
def test_native_continuation_binds_before_entity_and_state_resolution(who,phrase):
    with ClankerLM() as r:
        p=r.parser.parse(who+' '+phrase+'.',r.memory)
        assert p.understood and len(p.events)==1
        e=p.events[0]; parts=state_components(e)
        assert continuation_of(e)=='still' and len(parts)==1
        assert recurrence_of(e) is None and discourse_of(e) is None
        assert r.memory.get_entity(parts[0].report.entity_id).canonical_name==who
        assert 'still' not in (parts[0].event.arguments.get('state') or parts[0].event.arguments['value']).key
        assert not any('still' in n.canonical_name.lower() for n in r.memory.entities.values())
        assert all(t.key!='still' for k,t in e.arguments.items() if k=='time')


@pytest.mark.parametrize('marker,reading',[('Again','recurrence_or_reiteration'),('Still','concession')])
@pytest.mark.parametrize('who',['Jordan','Morgan','Riley'])
def test_fronted_operator_preserved_without_inventing_temporal_reading(marker,reading,who):
    with ClankerLM() as r:
        p=r.parser.parse(f'{marker}, {who} is angry.',r.memory)
        assert p.understood and len(p.events)==1
        e=p.events[0]; part=state_components(e)[0]
        assert e.arguments[DISCOURSE_ROLE].key==reading
        assert discourse_of(e)==marker.lower()
        assert not continuation_of(e) and not recurrence_of(e)
        assert r.memory.get_entity(part.report.entity_id).canonical_name==who
        assert e.arguments['subject'].surface==who
        assert e.raw_text==f'{marker}, {who} is angry.'


@pytest.mark.parametrize('text',[
 'Jordan is not still angry.','Jordan is still not angry.','Jordan is no longer still angry.',
 'Jordan is still angry again.','Jordan is still still angry.',
 'Again, Jordan is angry and Sarah is sad.','Again, Sarah said Jordan is angry.',
 'Again, Jordan is angry if Sarah calls.','Again, is Jordan angry?',
 'Still, Jordan is still angry.','Again, Jordan is no longer angry.',
 'Jordan is angry still.'
])
def test_unsupported_scope_does_not_collapse_to_simple_eligible_state(text):
    with ClankerLM() as r:
        p=r.parser.parse(text,r.memory)
        assert not p.understood
        assert not any(state_components(e) for e in p.events)


@pytest.mark.parametrize('text', ['Jordan stood still','Jordan is Still','Still is Jordan'])
def test_nonstate_and_proper_name_uses_are_not_removed(text):
    words=lexicon.tokenize(text)
    predicate='stand' if 'stood' in text else 'be'
    vi=1
    result=scan_continuation(words,predicate,vi)
    assert not result.marker and result.tokens==tuple(words)


@pytest.mark.parametrize('text,scope,source',[
 ('Jordan was still angry yesterday.','historical','user'),
 ('Jordan might still feel angry.','projected_or_modal','user'),
 ('Jordan will still be angry tomorrow.','projected_or_modal','user'),
 ('Sarah said Jordan is still angry.','current','attributed')])
def test_continuation_does_not_erase_source_time_or_modality(text,scope,source):
    with ClankerLM() as r:
        p=r.parser.parse(text,r.memory)
        e=next(e for e in p.events if e.predicate in {'be','feel'})
        c=state_components(e)[0]
        assert c.report.temporal_scope==scope and e.source.value==source
        assert continuation_of(e)=='still'
        if source=='attributed':assert e.discourse_role=='content'


@pytest.mark.parametrize('mutation',['kind','surface','negative','predicate','aspect','recurrence','cessation','front'])
def test_metadata_is_validated_at_component_boundary(mutation):
    e=EventFrame('be',{'subject':SemanticRef.entity('j','Jordan'),
      'value':SemanticRef.literal('angry','angry',EntityKind.ABSTRACT),
      'attribute':SemanticRef.literal('state','state'),
      CONTINUATION_ROLE:SemanticRef.literal(CONTINUATION_KIND,'still',EntityKind.ABSTRACT)})
    if mutation=='kind':e.arguments[CONTINUATION_ROLE]=SemanticRef.entity('continued','still')
    if mutation=='surface':e.arguments[CONTINUATION_ROLE]=SemanticRef.literal('continued','again')
    if mutation=='negative':e.polarity=False
    if mutation=='predicate':e.predicate='buy'
    if mutation=='aspect':e.aspect='progressive'
    if mutation=='recurrence':e.arguments[RECURRENCE_ROLE]=SemanticRef.literal('recurred','again')
    if mutation=='cessation':e.arguments['state_change']=SemanticRef.literal('ceased','no_longer')
    if mutation=='front':e.arguments[DISCOURSE_ROLE]=SemanticRef.literal('concession','still')
    with pytest.raises(ValueError):continuation_of(e)
    assert not state_components(e)


@pytest.mark.parametrize('text',['Jordan is still angry.','Again, Jordan is angry.','Still, Jordan is angry.'])
def test_original_observations_and_snapshot_preserve_all_operator_fields(text):
    with ClankerLM() as r:
        for _ in range(3):
            before=copy.deepcopy([e.to_dict() for e in r.memory.events])
            r.memory.begin_turn();p=r.parser.parse(text,r.memory);r.memory.add_event(p.events[0])
            assert [e.to_dict() for e in r.memory.events][:len(before)]==before
        assert len(r.memory.events)==3
        rows=original_state_rows(r.memory);assert len(rows)==3
        assert all(row.get('state_continuation') or row.get('state_discourse') for row in rows)
        restored=type(r.memory).from_dict(r.memory.to_dict())
        assert original_state_rows(restored)==rows


def test_live_question_does_not_create_continuation_observation():
    with ClankerLM() as r:
        r.memory.begin_turn();p=r.parser.parse('Jordan is angry.',r.memory);r.memory.add_event(p.events[0])
        old=copy.deepcopy([e.to_dict() for e in r.memory.events])
        p=r.parser.parse('Is Jordan still angry?',r.memory)
        assert p.question and continuation_of(p.question.event)=='still'
        assert [e.to_dict() for e in r.memory.events]==old


@pytest.mark.parametrize('text',['Jordan is still.','Jordan is still here.','Are you still there?'])
def test_nonaffective_still_stays_on_its_original_semantic_path(text):
    with ClankerLM() as r:
        p=r.parser.parse(text,r.memory)
        candidates=p.events+([p.question.event] if p.question else [])
        assert candidates
        assert all(CONTINUATION_ROLE not in e.arguments for e in candidates)
        assert not any(state_components(e) for e in candidates)
