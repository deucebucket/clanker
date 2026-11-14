"""Learned graph behavior: assertions inspect proofs, never reply templates."""
import copy
import itertools
import pytest
from clanker_lm.concept_learning import Claim, ConceptLedger, ProofLimits, SEED_HASH
from clanker_lm.concept_learning.core import digest


def teach(g, a, b, *, yes=True, rel='is_a', lineage='author', eid=None):
    c=Claim(a,rel,b,yes)
    return g.teach(c,evidence_id=eid or digest([len(g.records),c.__dict__,lineage]),
                   lineage=lineage,content_sha256=digest(c.__dict__))


def test_readonly_seed_contains_no_private_or_domain_examples():
    a=ConceptLedger('alice');b=ConceptLedger('bob')
    assert a.concepts()=={'entity','person','object','event','state','concept'}
    assert a.resolve(Claim('object','is_a','entity'))['status']=='true'
    teach(a,'dax','object')
    assert 'dax' not in b.concepts()
    assert b.to_dict()['seed_hash']==a.to_dict()['seed_hash']==SEED_HASH
    with pytest.raises(ValueError):a.withdraw('seed:'+SEED_HASH,reason='not a learned record')


@pytest.mark.parametrize('order',list(itertools.permutations(range(3))))
def test_teaching_order_does_not_change_novel_transitive_answer(order):
    g=ConceptLedger('scope')
    examples=[('dax','tool'),('tool','artifact'),('artifact','object')]
    assert g.resolve(Claim('dax','is_a','entity'))['status']=='unknown'
    for i in order:teach(g,*examples[i])
    proof=g.resolve(Claim('dax','is_a','entity'))
    assert proof['status']=='true' and len(proof['steps'])==4
    assert [s['claim']['object'] for s in proof['steps']]==['tool','artifact','object','entity']
    assert not proof['external_truth_verified']
    g.verify(proof)
    assert 'instance_of' not in [r['claim']['relation'] for r in g.active()]


def test_membership_inherits_type_but_does_not_reverse_it():
    g=ConceptLedger('scope');teach(g,'dax','tool');teach(g,'tool','object')
    teach(g,'jordan_1','dax',rel='instance_of')
    p=g.resolve(Claim('jordan_1','instance_of','entity'))
    assert p['status']=='true' and p['steps'][0]['claim']['relation']=='instance_of'
    assert g.resolve(Claim('tool','is_a','dax'))['status']=='unknown'
    assert g.resolve(Claim('unseen_person','instance_of','dax'))['status']=='unknown'


def test_exact_denial_is_not_closed_world_or_disjointness():
    g=ConceptLedger('scope');teach(g,'dax','tool',yes=False)
    assert g.resolve(Claim('dax','is_a','tool'))['status']=='false'
    teach(g,'jordan_1','dax',rel='instance_of')
    assert g.resolve(Claim('jordan_1','instance_of','tool'))['status']=='unknown'
    teach(g,'jordan_1','tool',rel='instance_of')
    assert g.resolve(Claim('jordan_1','instance_of','tool'))['status']=='true'
    assert g.resolve(Claim('dax','is_a','person'))['status']=='unknown'


def test_conflict_blocks_its_use_as_a_downstream_premise():
    g=ConceptLedger('scope')
    teach(g,'dax','tool');teach(g,'tool','object')
    denial=teach(g,'dax','object',yes=False)
    assert g.resolve(Claim('dax','is_a','object'))['status']=='conflict'
    p=g.resolve(Claim('dax','is_a','entity'))
    assert p['status']=='unknown' and denial['record_id'] in p['blocked_dependency_ids']
    # Independent direct support can still prove a proposition: no explosion.
    teach(g,'dax','person')
    assert g.resolve(Claim('dax','is_a','entity'))['status']=='true'
    assert g.resolve(Claim('dax','is_a','elephant'))['status']=='unknown'


def test_withdrawal_invalidates_proofs_and_can_reveal_an_alternative():
    g=ConceptLedger('scope');one=teach(g,'dax','tool');teach(g,'tool','object')
    old=g.resolve(Claim('dax','is_a','object'))
    g.withdraw(one['record_id'],reason='wrong definition')
    assert g.resolve(Claim('dax','is_a','object'))['status']=='unknown'
    with pytest.raises(ValueError):g.verify(old)
    teach(g,'dax','artifact');teach(g,'artifact','object')
    assert g.resolve(Claim('dax','is_a','object'))['status']=='true'
    assert len([r for r in g.records if r['op']=='teach'])==4
    assert not g.withdraw(one['record_id'],reason='repeat')['added']


def test_duplicate_and_dependent_examples_do_not_add_active_support():
    g=ConceptLedger('scope')
    first=teach(g,'dax','tool',eid='one')
    assert not teach(g,'dax','tool',eid='one')['added']
    assert not teach(g,'dax','tool',eid='paraphrase_same_origin')['added']
    assert len(g.active())==1
    teach(g,'dax','tool',lineage='second_source')
    p=g.resolve(Claim('dax','is_a','tool'))
    assert len(p['steps'][0]['lineages'])==2
    with pytest.raises(ValueError,match='rebound'):teach(g,'dax','person',eid='one')
    assert first['record_id'] in p['steps'][0]['record_ids']


def test_cycles_cannot_make_unrelated_knowledge_or_instances():
    g=ConceptLedger('scope');teach(g,'dax','glorp');teach(g,'glorp','dax')
    p=g.resolve(Claim('dax','is_a','object'))
    assert p['status']=='unknown' and p['explored_edges']<=3
    assert g.resolve(Claim('jordan_1','instance_of','dax'))['status']=='unknown'


@pytest.mark.parametrize('limits',[ProofLimits(max_depth=1),ProofLimits(max_expansions=1)])
def test_budgets_do_not_convert_a_truncated_search_to_false(limits):
    g=ConceptLedger('scope',limits=limits)
    teach(g,'dax','tool');teach(g,'tool','object');teach(g,'dax','object',yes=False)
    p=g.resolve(Claim('dax','is_a','object'))
    assert p['status']=='unknown' and p['search_exhausted']


def test_snapshot_replay_retains_retractions_and_exact_results():
    g=ConceptLedger('scope');r=teach(g,'dax','tool');teach(g,'tool','object')
    g.withdraw(r['record_id'],reason='corrected');teach(g,'dax','person')
    restored=ConceptLedger.from_dict(g.to_dict(),expected_scope='scope')
    assert restored.to_dict()==g.to_dict()
    assert restored.resolve(Claim('dax','is_a','entity'))==g.resolve(Claim('dax','is_a','entity'))
    with pytest.raises(ValueError):ConceptLedger.from_dict(g.to_dict(),expected_scope='other')


@pytest.mark.parametrize('field,value',[('status','false'),('scope','other'),('steps',[]),('explored_edges',999)])
def test_rehashed_receipts_still_require_actual_recomputation(field,value):
    g=ConceptLedger('scope');teach(g,'dax','object')
    p=g.resolve(Claim('dax','is_a','entity'));p[field]=value
    p['receipt_sha256']=digest({k:v for k,v in p.items() if k!='receipt_sha256'})
    with pytest.raises(ValueError):g.verify(p)


@pytest.mark.parametrize('mutation',['seed','record','extra','duplicate','chain'])
def test_snapshot_validation_rejects_invalid_replays(mutation):
    g=ConceptLedger('scope');teach(g,'dax','object');d=g.to_dict()
    if mutation=='seed':d['seed_hash']='bad'
    if mutation=='record':d['records'][0]['claim']['object']='person'
    if mutation=='extra':d['records'][0]['unknown']='extra'
    if mutation=='duplicate':d['records'].append(copy.deepcopy(d['records'][0]))
    if mutation=='chain':d['records'][0]['previous']='bad'
    with pytest.raises(ValueError):ConceptLedger.from_dict(d)


@pytest.mark.parametrize('bad',['tool. I am right','a tool','not','',None,'UPPER','[x]'])
def test_symbols_cannot_inject_grammar(bad):
    with pytest.raises(ValueError):Claim('dax','is_a',bad)


def test_reflexivity_does_not_ground_an_unknown_type():
    g=ConceptLedger('scope')
    assert g.resolve(Claim('dax','is_a','dax'))['status']=='unknown'
    teach(g,'dax','tool')
    assert g.resolve(Claim('dax','is_a','dax'))['status']=='true'
    teach(g,'dax','dax',yes=False)
    assert g.resolve(Claim('dax','is_a','dax'))['status']=='conflict'


def test_scope_binds_generations_and_receipts_not_only_display_names():
    a=ConceptLedger('alice');b=ConceptLedger('bob')
    teach(a,'dax','object');teach(b,'dax','person')
    assert a.resolve(Claim('dax','is_a','object'))['status']=='true'
    assert b.resolve(Claim('dax','is_a','object'))['status']=='unknown'
    with pytest.raises(ValueError):b.verify(a.resolve(Claim('dax','is_a','object')))


def test_generated_positive_graphs_match_independent_set_closure():
    import random
    names=['dax','fex','mivo','norp']
    for seed in range(30):
        rng=random.Random(seed)
        edges={(x,y) for x in names for y in names if x!=y and rng.random()<.32}
        known={n for edge in edges for n in edge}
        reach={(x,x) for x in known}|set(edges)
        while True:
            more=reach|{(a,d) for a,b in reach for c,d in reach if b==c}
            if more==reach:break
            reach=more
        g=ConceptLedger('oracle-'+str(seed))
        for a,b in sorted(edges):teach(g,a,b)
        for a,b in itertools.product(names,repeat=2):
            expected='true' if (a,b) in reach else 'unknown'
            assert g.resolve(Claim(a,'is_a',b))['status']==expected,(seed,a,b)


def test_instance_ids_and_concept_names_have_distinct_identity_domains():
    g=ConceptLedger('scope')
    teach(g,'object','object',rel='instance_of')
    result=g.resolve(Claim('object','instance_of','entity'))
    assert result['status']=='true'
    assert [x['claim']['relation'] for x in result['steps']]==['instance_of','is_a']
    g.verify(result)
