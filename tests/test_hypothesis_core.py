"""Portable tests use explicit typed source rows, not a simulated chat pass."""
from copy import deepcopy
from types import SimpleNamespace
from dataclasses import replace
import random
import pytest
from clanker_lm.activation import fingerprint
from clanker_lm.hypotheses import Claim, Limits, Pattern, ProposalRule, propose, assess
from clanker_lm.hypotheses.schemas import MATERIAL, SITUATION
from clanker_lm.hypotheses.evidence import EvidenceView, record_body, observation_id
from clanker_lm.hypotheses.review import verify_assessment


def view(claims, *, origins=None):
    nodes={}
    for i,c in enumerate(claims):
        body=record_body('scope',c,evidence_id=f'e{i}',origins=((origins[i] if origins else f's{i}'),),
                         source_kind='document_report',content_hash=fingerprint([i,c.to_dict()]))
        key=observation_id('scope',f'e{i}')
        nodes[key]={'id':key,'kind':'study_observation','record':body}
        for a in c.arguments:nodes[a]={'id':a,'kind':'concept','label':a}
    return EvidenceView(SimpleNamespace(nodes=nodes),'scope')


def seed(a='paper',b='wood',p='flammable',frame='lab'):
    return [Claim('made_from',(a,b),frame),Claim('has_property',(a,p),frame)]


@pytest.mark.parametrize('words', [('paper','wood','flammable'),('p','q','r'),('sheet','material','brittle'),
                                   ('tool','metal','conductive'),('x','y','transparent')])
def test_only_structure_determines_candidate_not_name(words):
    a,b,p=words;v=view(seed(*words));result=propose(v,(MATERIAL,))
    assert len(result['candidates'])==1
    assert result['candidates'][0]['claim']==Claim('has_property',(b,p),'lab').to_dict()
    r=assess(result['candidates'],v,(MATERIAL,))
    assert r['status']=='proposed' and r['probability'] is None


@pytest.mark.parametrize('mutation',['relation_direction','property_subject','frame','condition','negation'])
def test_incompatible_patterns_do_not_match(mutation):
    a,b=seed()
    if mutation=='relation_direction':a=replace(a,arguments=('wood','paper'))
    if mutation=='property_subject':b=replace(b,arguments=('unrelated','flammable'))
    if mutation=='frame':b=replace(b,frame='elsewhere')
    if mutation=='condition':b=replace(b,conditions=(('phase','liquid'),))
    if mutation=='negation':b=replace(b,positive=False)
    assert not propose(view([a,b]),(MATERIAL,))['candidates']


def test_raw_cardinality_is_not_independent_support_or_confidence():
    claims=seed();candidate=propose(view(claims),(MATERIAL,))['candidates']
    target=Claim('has_property',('wood','flammable'),'lab')
    v=view(claims+[target]*100,origins=['s0','s1']+['one-original']*100)
    r=assess(candidate,v,(MATERIAL,))
    assert len(r['supporting_evidence'])==100
    assert r['support_origin_groups']==[['one-original']] and r['probability'] is None
    assert not r['is_proof'] and not r['authorized_for_factual_inference']


def test_positive_and_negative_support_remain_contested_not_majority_vote():
    claims=seed();candidate=propose(view(claims),(MATERIAL,))['candidates']
    target=Claim('has_property',('wood','flammable'),'lab')
    v=view(claims+[target]*12+[target.opposite()])
    r=assess(candidate,v,(MATERIAL,))
    assert r['status']=='contested' and len(r['opposing_evidence'])==1


def test_changed_source_content_breaks_existing_motivation():
    claims=seed();candidate=propose(view(claims),(MATERIAL,))['candidates']
    v=view([claims[0],replace(claims[1],arguments=('paper','opaque'))])
    with pytest.raises(ValueError,match='source changed'):assess(candidate,v,(MATERIAL,))


def test_recomputed_digest_cannot_bless_an_unbound_mapping():
    v=view(seed());candidate=propose(v,(MATERIAL,))['candidates']
    candidate[0]['claim']['arguments'][0]='different_material'
    candidate[0]['digest']=fingerprint({k:x for k,x in candidate[0].items() if k!='digest'})
    with pytest.raises(ValueError):assess(candidate,v,(MATERIAL,))


def test_source_withdrawal_does_not_allow_tampered_route_to_escape_validation():
    v=view(seed());candidate=propose(v,(MATERIAL,))['candidates']
    v.active={}
    candidate[0]['bindings']['?material']='different'
    candidate[0]['digest']=fingerprint({k:x for k,x in candidate[0].items() if k!='digest'})
    with pytest.raises(ValueError):assess(candidate,v,(MATERIAL,))


def test_rule_version_is_pinned():
    v=view(seed());candidate=propose(v,(MATERIAL,))['candidates']
    changed=replace(MATERIAL,obligations=('different_condition',))
    with pytest.raises(ValueError):assess(candidate,v,(changed,))


def test_discovery_budget_stops_explicitly():
    claims=[]
    for i in range(20):claims+=seed('paper'+str(i),'wood'+str(i),'flammable')
    r=propose(view(claims),(MATERIAL,),Limits(max_checks=3))
    assert r['status']=='incomplete' and r['match_checks']==3


def test_no_recursion_from_the_candidate_itself():
    v=view(seed());a=propose(v,(MATERIAL,));b=propose(v,(MATERIAL,))
    assert a==b and len(v.active)==2
    result=assess(a['candidates'],v,(MATERIAL,))
    verify_assessment(result,a['candidates'],v,(MATERIAL,))
    assert not result['supporting_evidence']


def test_reordered_input_has_identical_bindings_with_stable_evidence_ids():
    v=view(seed());expected=propose(v,(MATERIAL,))
    v.active=dict(reversed(list(v.active.items())))
    assert propose(v,(MATERIAL,))==expected


def test_generated_property_projection_matches_an_independent_join():
    rng=random.Random(172)
    for _ in range(15):
        claims=[];materials=[];properties=[]
        for i in range(5):
            for j in range(5):
                if i!=j and rng.random()<.2:
                    materials.append((f'm{i}',f'm{j}'))
            if rng.random()<.7:properties.append((f'm{i}',f'p{i%2}'))
        for a,b in materials:claims.append(Claim('made_from',(a,b),'f'))
        for a,p in properties:claims.append(Claim('has_property',(a,p),'f'))
        expected={(b,p) for a,b in materials for x,p in properties if x==a}
        actual=propose(view(claims),(MATERIAL,))
        assert {tuple(c['claim']['arguments']) for c in actual['candidates']}==expected


def test_context_reference_is_not_a_material_provenance_edge():
    a,b=seed();a=replace(a,predicate='context_reference')
    assert not propose(view([a,b]),(MATERIAL,))['candidates']
