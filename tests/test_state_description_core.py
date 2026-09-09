"""Native state-description contracts; no graph or new decoder is required."""
from __future__ import annotations

import copy
import pytest
from clanker_lm import ClankerLM
from clanker_lm.model import EntityKind, EventFrame, SemanticRef, SourceKind
from clanker_lm.state_descriptions import (state_components, resolve_reported_state,
                                         state_question_scope)


def event(text, *, tense="present", polarity=True, modality=None, time=None):
    args={"subject":SemanticRef.entity("jordan", "Jordan"),
          "value":SemanticRef.literal(text,text,EntityKind.ABSTRACT),
          "attribute":SemanticRef.literal("state","state",EntityKind.ABSTRACT)}
    if time: args["time"]=SemanticRef.literal(time,time,EntityKind.TIME)
    return EventFrame("be",args,tense=tense,polarity=polarity,modality=modality)


@pytest.mark.parametrize("text,labels", [
    ("angry",["angry"]), ("very angry",["angry"]),
    ("angry and sad",["angry","sad"]),
    ("really angry and quite sad",["angry","sad"]),
    ("angry and sad and tired and anxious",["angry","sad","tired","anxious"]),
])
def test_components_preserve_full_supported_qualification(text,labels):
    original=event(text); before=copy.deepcopy(original.to_dict())
    parts=state_components(original)
    assert [p.report.state_term for p in parts]==labels
    assert original.to_dict()==before
    assert " and ".join(p.event.arguments['value'].surface for p in parts)==text


@pytest.mark.parametrize("text", ["angry or sad", "angry but happy", "angry and", "and sad",
    "angry and angry", "not angry", "angry; happy", "angry yesterday and sad today",
    "sad because John left", "happy and sad and tired and angry and calm", "a teacher", "sad? happy"])
def test_unsupported_structure_cannot_be_flattened_into_a_state(text):
    assert not state_components(event(text))


@pytest.mark.parametrize("polarity,modal", [(False,None),(True,"might"),(False,"could")])
def test_scoped_negation_and_modality_do_not_distribute_over_conjunction(polarity,modal):
    assert not state_components(event("angry and sad",polarity=polarity,modality=modal))


@pytest.mark.parametrize("text,tense,scope,cue", [
    ("angry yesterday","past","historical","yesterday"),
    ("really angry yesterday","past","historical","yesterday"),
    ("sad now","present","current","now"),
    ("calm today","present","current","today"),
    ("sad tomorrow","future","projected_or_modal","tomorrow"),
    ("angry and sad yesterday","past","historical","yesterday"),
])
def test_shared_time_is_extracted_without_losing_modifiers(text,tense,scope,cue):
    parts=state_components(event(text,tense=tense))
    assert parts and all(p.temporal_cue==cue and p.report.temporal_scope==scope for p in parts)
    assert all(p.event.arguments['time'].key==cue for p in parts)
    assert all(cue not in p.event.arguments['value'].surface for p in parts)


def test_disagreement_between_time_argument_and_suffix_is_not_overwritten():
    assert not state_components(event("sad yesterday",time="today"))
    assert not state_components(event("sad tomorrow",tense="past"))
    assert not state_components(event("sad",time="in 2030"))


def test_negated_anymore_is_preserved_not_turned_into_its_opposite():
    part=state_components(event("angry anymore",polarity=False))[0]
    assert not part.report.polarity
    assert part.event.arguments['value'].surface=='angry anymore'
    assert not state_components(event("angry anymore"))


@pytest.mark.parametrize("statement,question", [
    ('Sarah said Jordan was angry.','What did Sarah say?'),
    ('Sarah said Jordan was not angry.','What did Sarah say?'),
    ('Sarah believes Jordan is sad.','What does Sarah believe?'),
    ('Sarah said Jordan was angry yesterday.','What did Sarah say?'),
])
def test_native_attributed_answer_has_a_verified_source_chain(statement,question):
    with ClankerLM() as r:
        r.process(statement)
        parsed=r.parser.parse(question,r.memory)
        answer=r.answerer.answer(parsed.question,r.memory)
        relation,matrix,content=resolve_reported_state(answer,r.memory)
        assert relation.matrix_event_id==matrix.event_id
        assert content.event_id==answer.proposition.event_id
        assert content.source==SourceKind.ATTRIBUTED
        assert matrix.arguments['agent'].key==relation.source_entity_id


@pytest.mark.parametrize("mutation",['source','matrix_tense','relation_type','value','certainty','content','relation'])
def test_native_source_binding_rejects_semantic_tampering(mutation):
    with ClankerLM() as r:
        r.process('Sarah said Jordan was angry.')
        p=r.parser.parse('What did Sarah say?',r.memory)
        answer=r.answerer.answer(p.question,r.memory)
        if mutation=='source': answer.required_slots['source_entity_id']='user'
        if mutation=='matrix_tense': answer.required_slots['matrix_tense']='present'
        if mutation=='relation_type': answer.required_slots['relation_type']='believed'
        if mutation=='value': answer.values=[]
        if mutation=='certainty': answer.certainty=255
        if mutation=='content': answer.proposition=answer.proposition.copy(polarity=False)
        if mutation=='relation': r.memory.contents[0].attributed=False
        with pytest.raises(ValueError):resolve_reported_state(answer,r.memory)


def test_legacy_bare_feel_entity_is_projected_but_not_a_name_or_an_object():
    e=EventFrame('feel', {'experiencer':SemanticRef.entity('user','I'),
                         'state':SemanticRef.entity('better_1','better')})
    before=copy.deepcopy(e.to_dict())
    parts=state_components(e)
    assert parts and parts[0].report.state_term=='better'
    assert parts[0].event.arguments['state'].kind.value=='literal'
    assert e.to_dict()==before
    for surface in ['Happy','the car','my sad','angry or sad']:
        assert not state_components(e.copy(arguments={**e.arguments,
            'state':SemanticRef.entity('x',surface)}))


def indexed_case(r, *, historical=False):
    """Typed graph-index seam; no claim that native main contains MemoryWeb."""
    from clanker_lm.model import HowKind,QuestionFrame,QuestionKind
    from clanker_lm.state_descriptions import content_hash
    r.process('Jordan was angry yesterday.' if historical else 'Jordan is angry and sad.')
    e=r.memory.events[-1]
    parts=state_components(e)
    rows=[]
    for p in parts:
        s=p.report
        rows.append({'id':f'appraisal:{e.turn_index}:{e.event_id}'+(f':component:{p.index}' if p.index else ''),
                     'event_id':e.event_id,'event_signature_hash':content_hash(e.to_dict()),
                     'experiencer_id':s.entity_id,'turn':e.turn_index,'label':s.state_term,
                     'component_index':p.index,'temporal_cue':p.temporal_cue,
                     'polarity':s.polarity,'tense':s.tense,'source_kind':e.source.value,
                     'reporter_ids':['user']})
    args={'experiencer':SemanticRef.entity(parts[0].report.entity_id,'Jordan'),
          'state':SemanticRef.variable('state',EntityKind.ABSTRACT)}
    if historical:args['time']=SemanticRef.literal('yesterday','yesterday',EntityKind.TIME)
    q=QuestionFrame(QuestionKind.HOW,EventFrame('feel',args,tense='past' if historical else 'present'),
                    requested_role='state',answer_type=EntityKind.ABSTRACT,how_kind=HowKind.STATE)
    return q,rows


@pytest.mark.parametrize('historical',[False,True])
def test_indexed_answer_contract_roundtrip_keeps_every_component(historical):
    import json
    from clanker_lm.state_descriptions import answer_from_appraisals,resolve_state_bundle
    with ClankerLM() as r:
        q,rows=indexed_case(r,historical=historical)
        answer=answer_from_appraisals(q,r.memory,rows)
        assert answer.status.value=='answered'
        assert {v.key for v in answer.values}==({'angry'} if historical else {'angry','sad'})
        replayed=copy.deepcopy(answer)
        replayed.required_slots['state_bundle']=json.dumps(json.loads(answer.required_slots['state_bundle']))
        with ClankerLM.loads(r.dumps()) as restored:
            projected=resolve_state_bundle(replayed,restored.memory)
        assert len(projected)==(1 if historical else 2)
        assert all(e.tense==('past' if historical else 'present') for e in projected)


@pytest.mark.parametrize('field,value',[('experiencer_id','user'),('event_signature_hash','bad'),
    ('source_kind','verified'),('reporter_ids',['assistant']),('component_index',9)])
def test_indexed_rows_require_matching_live_bindings(field,value):
    from clanker_lm.state_descriptions import answer_from_appraisals
    with ClankerLM() as r:
        q,rows=indexed_case(r,historical=True)
        rows[0][field]=value
        assert answer_from_appraisals(q,r.memory,rows).status.value=='unknown'


@pytest.mark.parametrize('mutation',['missing_value','appraisal_ids','duplicate_component','wrong_proposition'])
def test_native_bundle_boundary_rejects_changed_selection(mutation):
    import json
    from clanker_lm.state_descriptions import answer_from_appraisals,resolve_state_bundle
    with ClankerLM() as r:
        q,rows=indexed_case(r)
        answer=answer_from_appraisals(q,r.memory,rows)
        if mutation=='missing_value':answer.values=answer.values[:1]
        if mutation=='appraisal_ids':answer.required_slots['retrieved_appraisal_ids']='wrong'
        if mutation=='wrong_proposition':answer.proposition.polarity=False
        if mutation=='duplicate_component':
            data=json.loads(answer.required_slots['state_bundle'])
            data['entries'][1]=data['entries'][0]
            answer.required_slots['state_bundle']=json.dumps(data)
            answer.required_slots['retrieved_appraisal_ids']=','.join(x['appraisal_id'] for x in data['entries'])
        with pytest.raises(ValueError):resolve_state_bundle(answer,r.memory)
