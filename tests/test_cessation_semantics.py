"""Native parser/state phase tests: no new graph or decoder dependency."""
from __future__ import annotations
import copy
import pytest
from clanker_lm import ClankerLM
from clanker_lm import lexicon
from clanker_lm.cessation import marker_of,scan_cessation,withdrawal_matches
from clanker_lm.state_descriptions import state_components,direct_state_evidence

FORMS=[
    ('Jordan is no longer angry.','be','no_longer'),
    ('Jordan no longer feels angry.','feel','no_longer'),
    ('Jordan is not angry anymore.','be','not_anymore'),
    ('Jordan does not feel angry anymore.','feel','not_anymore'),
    ('Jordan is not angry any more.','be','not_any_more'),
    ('Jordan is not any longer angry.','be','not_any_longer'),
]
@pytest.mark.parametrize('text,pred,marker',FORMS)
def test_phase_operator_separates_from_person_and_state(text,pred,marker):
    with ClankerLM() as r:
        p=r.parser.parse(text,r.memory)
        assert len(p.events)==1
        e=p.events[0];part=state_components(e)[0]
        subject=e.arguments.get('subject') or e.arguments['experiencer']
        assert r.memory.get_entity(subject.key).canonical_name=='Jordan'
        assert not any('longer' in n.canonical_name for n in r.memory.entities.values())
        assert e.predicate==pred and e.polarity is False
        assert marker_of(e)==marker
        assert part.report.state_term=='angry'
        assert (part.event.arguments.get('state') or part.event.arguments['value']).key=='angry'
        assert part.report.temporal_scope=='current'
        assert e.raw_text==text
        assert r.memory.events==[] # parsing is not asserting these events

@pytest.mark.parametrize('modifier',['really','very','quite','extremely'])
def test_degree_is_not_deleted(modifier):
    with ClankerLM() as r:
        e=r.parser.parse(f'Jordan is no longer {modifier} angry.',r.memory).events[0]
        p=state_components(e)[0]
        assert p.event.arguments['value'].key==modifier+' angry'
        assert not withdrawal_matches(modifier+' angry','angry')
        assert withdrawal_matches('angry',modifier+' angry')

@pytest.mark.parametrize('text',[
    'Jordan is not no longer angry.','Jordan is never no longer angry.',
    'Jordan is no longer not angry.','Jordan is no longer angry anymore.',
    'Jordan is no longer no longer angry.','Jordan is angry anymore.',
])
def test_double_or_missing_negation_is_not_collapsed(text):
    with ClankerLM() as r:
        p=r.parser.parse(text,r.memory)
        assert not p.events
        assert any('cessation' in d or 'state change' in d for d in p.diagnostics)

@pytest.mark.parametrize('complement',['angry and sad','angry or sad','angry but calm'])
def test_no_negation_distribution(complement):
    with ClankerLM() as r:
        e=r.parser.parse('Jordan is no longer '+complement+'.',r.memory).events[0]
        assert not state_components(e)

@pytest.mark.parametrize('text,expected',[
    ('Jordan was no longer angry yesterday.','historical'),
    ('Jordan might no longer be angry.','projected_or_modal'),
    ('Jordan will no longer feel angry.','projected_or_modal'),
])
def test_tense_and_modality_are_preserved(text,expected):
    with ClankerLM() as r:
        e=r.parser.parse(text,r.memory).events[0]
        assert state_components(e)[0].report.temporal_scope==expected

@pytest.mark.parametrize('text',[
    'If Jordan is no longer angry, Sarah calls.',
    'Jordan is no longer angry if Sarah calls.',
    'Jordan is no longer angry unless Sarah calls.',
])
def test_condition_never_becomes_unqualified_revision(text):
    with ClankerLM() as r:
        r.process(text)
        phase=[e for e in r.memory.events if 'state_change' in e.arguments]
        assert phase and all(not direct_state_evidence(e,r.memory) for e in phase)

@pytest.mark.parametrize('text',[
    'I am no longer sad.','I am not happy anymore.','Jordan no longer feels sad.',
])
def test_native_response_policy_consumes_the_same_denied_state(text):
    with ClankerLM() as r:
        out=r.process(text)
        assert out.gates.response_act=='neutral_acknowledge'
        assert any('negated state' in x for x in out.gates.rationale)


def test_applying_operator_does_not_weaken_severe_native_gates():
    with ClankerLM() as r:
        o=r.process('I am no longer safe.')
        assert o.gates.severity in {'high','critical'}
        assert o.gates.response_act in {'serious_followup','safety_probe'}


def test_snapshot_roundtrip_keeps_operator_and_original_evidence():
    with ClankerLM() as r:
        r.process('Jordan is no longer angry.')
        snap=r.dumps();original=r.memory.events[0].to_dict()
    with ClankerLM.loads(snap) as restored:
        assert restored.memory.events[0].to_dict()==original
        assert marker_of(restored.memory.events[0])=='no_longer'


def test_unrelated_comparative_is_not_a_phase_operator():
    toks=lexicon.tokenize('the table is longer')
    result=scan_cessation(toks,'be',2)
    assert result.marker is None and result.error is None
