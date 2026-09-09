"""Scoped affective evidence: no canned-reply assertions or emotion ground truth.

This covers the native runtime; no experimental word decoder is required.
"""
import pytest
from clanker_lm import ClankerLM
from clanker_lm.model import EventFrame, SemanticRef, EntityKind, SourceKind
from clanker_lm.state_scope import state_report


@pytest.mark.parametrize('subject', ['I', 'Sarah'])
@pytest.mark.parametrize('state', ['sad', 'angry', 'happy', 'proud'])
def test_denied_affect_is_not_asserted_or_inverted_by_response_policy(subject, state):
    with ClankerLM() as lm:
        result = lm.process(f'{subject} {"am" if subject == "I" else "is"} not {state}.')
        assert result.parse.understood
        assert result.gates.response_act == 'neutral_acknowledge'
        assert any('negated state' in reason for reason in result.gates.rationale)
        assert not result.response.endswith('?')


@pytest.mark.parametrize('tense,polarity,modality,scope,current', [
    ('present', True, None, 'current', True),
    ('past', True, None, 'historical', False),
    ('future', True, None, 'projected_or_modal', False),
    ('present', False, None, 'current', False),
    ('past', False, None, 'historical', False),
    ('present', True, 'might', 'projected_or_modal', False),
])
def test_typed_report_retains_scope_instead_of_pretending_every_axis_was_observed(tense, polarity, modality, scope, current):
    event = EventFrame('feel', arguments={
        'experiencer': SemanticRef.entity('user','I'),
        'state': SemanticRef.literal('sad','sad',EntityKind.ABSTRACT),
    }, tense=tense, polarity=polarity, modality=modality, source=SourceKind.USER)
    report = state_report(event, {'sad'})
    assert report is not None
    assert report.temporal_scope == scope
    assert report.current_self_report is current
    assert report.to_dict()['polarity'] is polarity


@pytest.mark.parametrize('subject,source,role', [
    ('sarah_1', SourceKind.USER, 'main'),
    ('user', SourceKind.ATTRIBUTED, 'content'),
    ('user', SourceKind.RETRIEVED, 'main'),
])
def test_current_self_evidence_does_not_transfer_across_participant_or_source(subject, source, role):
    event = EventFrame('be', arguments={
        'subject': SemanticRef.entity(subject,subject),
        'value': SemanticRef.literal('angry','angry',EntityKind.ABSTRACT),
    },source=source,discourse_role=role)
    assert not state_report(event,{'angry'}).current_self_report


@pytest.mark.parametrize('value', ['sad but happy','a teacher','sad because John left'])
def test_scope_adapter_abstains_on_unsupported_state_complements(value):
    event = EventFrame('be', arguments={
        'subject': SemanticRef.entity('user','I'),
        'value': SemanticRef.literal(value,value,EntityKind.ABSTRACT),
    })
    assert state_report(event,{'sad','happy'}) is None


def test_negated_safety_does_not_weaken_existing_serious_gate():
    with ClankerLM() as lm:
        result=lm.process('I am not safe.')
        assert result.gates.severity in {'high','critical'}
        assert result.gates.response_act in {'serious_followup','safety_probe'}


def test_unnegated_positive_report_still_has_positive_acknowledgment():
    with ClankerLM() as lm:
        assert lm.process('I am happy.').gates.response_act=='positive_acknowledge'


def test_continuous_native_conversation_survives_scope_correction():
    with ClankerLM() as lm:
        lm.process('My sister borrowed my car yesterday.')
        lm.process('I am not sad.')
        answer = lm.process('Who borrowed the car?')
        assert answer.contract.status.value == 'answered'
        assert 'sister' in answer.response.lower()


@pytest.mark.parametrize('time_text', ['on Monday','in 2030','at six'])
def test_unresolved_calendar_reference_is_not_current_evidence(time_text):
    event=EventFrame('feel', arguments={
        'experiencer': SemanticRef.entity('user','I'),
        'state': SemanticRef.literal('sad','sad',EntityKind.ABSTRACT),
        'time': SemanticRef.literal(time_text,time_text,EntityKind.TIME),
    })
    report=state_report(event,{'sad'})
    assert report.temporal_scope=='unspecified'
    assert not report.current_self_report
