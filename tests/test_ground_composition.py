"""Authored calculus tests, including nonnumeric transfer and negative controls."""
from dataclasses import replace
import copy
import json
import random
import pytest
from clanker_lm.composition import Operator, Term, atom, apply, EquationLedger, Limits, derive, verify
from clanker_lm.composition.numeric import NATURAL_ADD, parse_addition, validate_numeric_lesson
from clanker_lm.composition.terms import digest


def fact(ledger, lhs, rhs, eid='e1', origin='external_lesson'):
    return ledger.teach(lhs, rhs, evidence_id=eid, source_id='source', source_hash=digest([eid]), origin=origin)


def numeric(assoc=True):
    ledger=EquationLedger('s',(replace(NATURAL_ADD, associative=assoc),))
    fact(ledger,parse_addition('2+2'),atom('natural','4'))
    fact(ledger,parse_addition('4+4'),atom('natural','8'),'e2')
    return ledger


def test_main_example_reuses_one_premise_twice_without_claiming_three_sources():
    l=numeric();before=l.to_dict();r=derive(parse_addition('2+2+2+2'),l,scope='s')
    assert r['status']=='proved' and r['result']['symbol']=='8'
    assert r['supporting_evidence']==['e1','e2']
    assert [s.get('evidence_id') for s in r['proof']].count('e1')==2
    assert any(s['rule'].startswith('associate') for s in r['proof'])
    assert not r['target_calculator_used'] and before==l.to_dict()


def test_associativity_is_not_inferred_from_two_ground_equalities():
    l=numeric(False)
    assert derive(parse_addition('2+2+2+2'),l,scope='s')['status']=='unknown'
    assert derive(parse_addition('(2+2)+(2+2)'),l,scope='s')['result']['symbol']=='8'


@pytest.mark.parametrize('expression',['2+2+2+2','(2+2)+(2+2)','2+(2+(2+2))','(2+(2+2))+2'])
def test_different_groupings_are_solved_by_the_same_laws(expression):
    assert derive(parse_addition(expression),numeric(),scope='s')['result']['symbol']=='8'


@pytest.mark.parametrize('base',[1,3,5,7,9,11,17,23])
def test_unseen_numeric_instances_without_target_calculation(base):
    l=EquationLedger('s',(NATURAL_ADD,))
    fact(l,parse_addition(f'{base}+{base}'),atom('natural',str(base*2)))
    fact(l,parse_addition(f'{base*2}+{base*2}'),atom('natural',str(base*4)),'e2')
    q=parse_addition('+'.join([str(base)]*4))
    assert derive(q,l,scope='s')['result']['symbol']==str(base*4)


def test_same_algorithm_handles_abstract_ordered_sequence_composition():
    op=Operator('join','sequence',True)
    a,b,c,d,ab,cd,whole=[atom('sequence',x) for x in ['a','b','c','d','ab','cd','abcd']]
    l=EquationLedger('symbolic',(op,))
    fact(l,apply(op,a,b),ab,'left');fact(l,apply(op,c,d),cd,'right')
    fact(l,apply(op,ab,cd),whole,'joined')
    q=apply(op,apply(op,apply(op,a,b),c),d)
    r=derive(q,l,scope='symbolic');assert r['result']==whole.to_dict()
    assert derive(apply(op,b,a),l,scope='symbolic')['status']=='unknown'  # no commutativity


def test_sort_boundary_prevents_unit_or_domain_transfer():
    op=Operator('group','count:stones',True)
    with pytest.raises(ValueError):apply(op,atom('count:stones','2'),atom('count:minutes','2'))
    l=numeric()
    q=apply(Operator('other.add','other',True),atom('other','2'),atom('other','2'))
    with pytest.raises(ValueError):derive(q,l,scope='s')


@pytest.mark.parametrize('origin',['hypothesis','self_output','context_reference','analogy'])
def test_proposed_associations_cannot_become_equation_premises(origin):
    l=EquationLedger('s',(NATURAL_ADD,))
    with pytest.raises(ValueError):fact(l,parse_addition('2+2'),atom('natural','4'),origin=origin)
    assert not l.active()


def test_withdrawal_changes_answer_and_invalidates_old_receipt():
    l=numeric();q=parse_addition('2+2+2+2');r=derive(q,l,scope='s')
    verify(r,q,l,scope='s');l.withdraw('e1',reason='lesson withdrawn')
    assert derive(q,l,scope='s')['status']=='unknown'
    with pytest.raises(ValueError):verify(r,q,l,scope='s')
    assert len(l.to_dict()['rows'])==3


def test_conflicting_abstract_equations_are_not_outvoted_or_ignored():
    l=numeric();fact(l,parse_addition('2+2'),atom('natural','5'),'wrong')
    r=derive(parse_addition('2+2+2+2'),l,scope='s')
    assert r['status']=='conflict' and r['result'] is None
    assert r['conflicting_evidence']==['e1','wrong']
    l.withdraw('wrong',reason='test correction')
    assert derive(parse_addition('2+2+2+2'),l,scope='s')['status']=='proved'
    # Numeric runtime ingress independently rejects this false lesson.
    with pytest.raises(ValueError):validate_numeric_lesson(parse_addition('2+2'),atom('natural','5'))


def test_unrelated_conflicting_equation_does_not_poison_every_goal():
    l=numeric()
    fact(l,parse_addition('9+9'),atom('natural','18'),'a')
    fact(l,parse_addition('9+9'),atom('natural','19'),'b')
    assert derive(parse_addition('2+2+2+2'),l,scope='s')['status']=='proved'


@pytest.mark.parametrize('limits',[Limits(max_states=1),Limits(max_steps=1),Limits(max_proof_depth=1)])
def test_budget_exhaustion_is_not_unknown_or_a_verified_result(limits):
    r=derive(parse_addition('2+2+2+2'),numeric(),scope='s',limits=limits)
    assert r['status']=='incomplete' and r['result'] is None and r['stop_reasons']


def test_duplicate_evidence_id_is_idempotent_and_cannot_change_meaning():
    l=numeric();before=l.to_dict();fact(l,parse_addition('2+2'),atom('natural','4'))
    assert l.to_dict()==before
    with pytest.raises(ValueError):fact(l,parse_addition('2+2'),atom('natural','5'))


def test_active_alternative_remains_after_one_premise_withdrawal():
    l=numeric();fact(l,parse_addition('2+2'),atom('natural','4'),'alternative')
    l.withdraw('e1',reason='one source withdrawn')
    r=derive(parse_addition('2+2+2+2'),l,scope='s')
    assert r['status']=='proved' and 'e1' not in r['supporting_evidence']


def test_json_roundtrip_and_tampered_lineage():
    l=numeric();l.withdraw('e2',reason='withdraw')
    l.frozen=True;restored=EquationLedger.from_dict(json.loads(json.dumps(l.to_dict())))
    assert restored.to_dict()==l.to_dict() and restored.generation==l.generation
    bad=l.to_dict();bad['rows'][0]['rhs']['symbol']='99'
    with pytest.raises(ValueError):EquationLedger.from_dict(bad)
    with pytest.raises(ValueError):restored.withdraw('e1',reason='frozen')


def test_rehashed_bad_proof_and_other_scope_are_rejected():
    l=numeric();q=parse_addition('2+2+2+2');r=derive(q,l,scope='s')
    r['result']['symbol']='999';r['digest']=digest({k:v for k,v in r.items() if k!='digest'})
    with pytest.raises(ValueError):verify(r,q,l,scope='s')
    with pytest.raises(PermissionError):derive(q,l,scope='other')


@pytest.mark.parametrize('text',['__import__("os")','2*2','2-2','2/2','1.5+2.5','True+True','-2+4','1e3','f(2)','x+2'])
def test_numeric_ingress_declines_unlicensed_semantics(text):
    with pytest.raises(ValueError):parse_addition(text)


def test_generated_proved_results_agree_with_independent_integer_interpreter():
    rng=random.Random(71)
    def evaluate(t):
        return int(t.symbol) if t.atomic else sum(evaluate(a) for a in t.arguments)
    for _ in range(30):
        l=EquationLedger('s',(NATURAL_ADD,))
        for j in range(8):
            a,b=rng.randint(0,6),rng.randint(0,6)
            fact(l,parse_addition(f'{a}+{b}'),atom('natural',str(a+b)),f'e{j}')
        nums=[rng.randint(0,6) for _ in range(4)]
        q=parse_addition('+'.join(map(str,nums)))
        r=derive(q,l,scope='s')
        if r['status']=='proved':assert int(r['result']['symbol'])==evaluate(q)
        assert r['status'] in {'unknown','proved'}


def test_twenty_generated_complete_proofs_have_independent_numeric_oracle():
    rng=random.Random(190)
    passed=0
    for case in range(20):
        a,b,c,d=[rng.randint(1,20) for _ in range(4)]
        l=EquationLedger('s',(NATURAL_ADD,))
        fact(l,parse_addition(f'{a}+{b}'),atom('natural',str(a+b)),'left')
        fact(l,parse_addition(f'{c}+{d}'),atom('natural',str(c+d)),'right')
        fact(l,parse_addition(f'{a+b}+{c+d}'),atom('natural',str(a+b+c+d)),'top')
        query=parse_addition(f'{a}+{b}+{c}+{d}')
        r=derive(query,l,scope='s')
        assert r['status']=='proved',case
        assert int(r['result']['symbol'])==sum([a,b,c,d])
        passed+=1
    assert passed==20
