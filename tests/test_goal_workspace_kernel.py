"""Portable tests of the exact goal kernel; no delivered runtime is assumed.

Rows are author-specified test evidence, not a model or a response corpus.
The full conversational adapter is tested separately on the integrated branch.
"""
from __future__ import annotations
import copy
import pytest
from clanker_lm.reasoning_core import Goal, ReasoningLimits, solve, verify, fingerprint


def rows(*triples):
    return [dict(kind='assert',scope='test-context',relation=r,subject=a,target=b,
                 positive=positive,evidence_id=f'premise:{i}')
            for i,(r,a,b,positive) in enumerate(triples)]


def execute(evidence, *, scope='test-context', budget=1024, bridge=False):
    return solve(evidence,scope,fingerprint(evidence),Goal('member','mira_1','device'),
                 limits=ReasoningLimits(max_work=budget),find_bridge=bridge)


def chain():
    return rows(('member','mira_1','dax',True),('subclass','dax','tool',True),
                ('subclass','tool','device',True))


def test_three_premises_produce_a_replayable_goal_derivation():
    evidence=chain();proof=execute(evidence)
    assert proof['status']=='true'
    assert proof['witness']['premises']==['premise:0','premise:1','premise:2']
    assert proof['witness']['rules']==['membership_inheritance']*2
    assert any(s['operation']=='EXPAND_TARGET_SUBGOAL' for s in proof['steps'])
    assert proof['work_used']==len(proof['steps'])
    verify(proof,evidence,'test-context',fingerprint(evidence))


@pytest.mark.parametrize('negative',[('subclass','dax','tool',False),
    ('subclass','dax','device',False),('member','mira_1','tool',False),('member','mira_1','device',False)])
def test_contrary_evidence_is_preserved_at_intermediate_and_final_steps(negative):
    evidence=chain();row=rows(negative)[0];row['evidence_id']='denial';evidence.append(row)
    proof=execute(evidence)
    assert proof['status']=='conflict'
    assert 'denial' in {r['evidence_id'] for r in proof['evidence']}


def test_hypothetical_subgoal_is_not_promoted_to_an_existing_fact():
    evidence=rows(('member','mira_1','dax',True),('subclass','tool','device',True))
    before=copy.deepcopy(evidence);proof=execute(evidence,bridge=True)
    assert proof['status']=='unknown'
    assert proof['bridge']['goal']==['subclass','dax','tool']
    assert proof['bridge']['hypothetical_status']=='true'
    assert proof['bridge']['accepted_as_fact'] is False
    assert evidence==before


def test_subclass_denial_is_not_disjointness_or_negative_membership():
    evidence=rows(('member','mira_1','dax',True),('subclass','dax','device',False))
    assert execute(evidence)['status']=='unknown'


@pytest.mark.parametrize('budget',[1,2,5,8])
def test_execution_budget_limits_actual_steps_and_does_not_prove_false(budget):
    proof=execute(chain(),budget=budget)
    assert proof['status']=='unknown' and proof['truncated']
    assert proof['work_used']<=budget and len(proof['steps'])<=budget


@pytest.mark.parametrize('alteration',['operation','status','scope','evidence'])
def test_rehashing_cannot_hide_a_changed_computation(alteration):
    evidence=chain();proof=execute(evidence)
    if alteration=='operation':proof['steps'][0]['operation']='invented'
    if alteration=='status':proof['status']='false'
    if alteration=='scope':proof['scope']='other-account'
    if alteration=='evidence':proof['evidence']=[]
    proof['sha256']=fingerprint({k:v for k,v in proof.items() if k!='sha256'})
    with pytest.raises(ValueError):verify(proof,evidence,'test-context',fingerprint(evidence))


def test_other_scope_rows_are_rejected_before_search():
    with pytest.raises(ValueError):execute(chain(),scope='other-account')


def test_revised_evidence_invalidates_the_old_proof():
    evidence=chain();proof=execute(evidence);evidence.pop(1)
    with pytest.raises(ValueError):verify(proof,evidence,'test-context',fingerprint(evidence))
    assert execute(evidence)['status']=='unknown'


def test_cycle_has_no_power_to_create_a_missing_membership():
    evidence=rows(('subclass','dax','tool',True),('subclass','tool','dax',True),
                  ('subclass','tool','device',True))
    assert execute(evidence)['status']=='unknown'


@pytest.mark.parametrize('value',[0,-1,True,1.5,16385])
def test_work_limit_is_not_silently_coerced(value):
    with pytest.raises(ValueError):ReasoningLimits(max_work=value)
