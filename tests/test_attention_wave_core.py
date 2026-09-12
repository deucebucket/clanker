"""Salience is not truth. Test actual selection and policy before weighting."""
from dataclasses import replace
import random
import pytest
from clanker_lm.activation import ActivationIndex, Node, Link, LinkKind, Plane, Policy, Request, Task, fingerprint
from clanker_lm.attention import WaveConfig, WaveIndex


def graph(nodes, edges):
    return ActivationIndex(nodes, edges, context='scope', content_generation='test-v1')


def setup():
    ns=[Node('person',Plane.PERSONAL),Node('memory',Plane.PERSONAL),
        Node('term',Plane.SEMANTIC),Node('fact',Plane.SEMANTIC),Node('secret',Plane.AUXILIARY)]
    es=[Link('a','person','memory'),Link('b','memory','term',LinkKind.CONTEXT_REFERENCE),
        Link('c','term','fact'),Link('d','person','secret')]
    return WaveIndex(graph(ns,es)),Policy('scope',allowed_tasks=tuple(Task))


@pytest.mark.parametrize('task,root,active,refs',[
    (Task.PERSONAL_RECALL,'person',{'person','memory'},{'term'}),
    (Task.CONCEPT_EXPLANATION,'term',{'term','fact'},set()),
    (Task.CROSS_DOMAIN_COMPARISON,'person',{'person','memory','term','fact'},set()),
])
def test_task_gates_precede_math(task,root,active,refs):
    w,p=setup();r=w.select(Request((root,),task,max_depth=8),p)
    assert set(r['active'])==active and set(r['reference_only'])==refs
    assert not r['context_edges_are_proof_premises'] and not r['salience_is_confidence']
    assert 'secret' not in r['active']
    assert r['work']['payloads_read']==0


def test_no_reverse_gray_traversal_even_in_comparison():
    w,p=setup();r=w.select(Request(('term',),Task.CROSS_DOMAIN_COMPARISON,max_depth=8),p)
    assert set(r['active'])=={'term','fact'}


@pytest.mark.parametrize('bad',[-1,1025,True,1.0,float('nan'),'1'])
def test_invalid_strengths_fail(bad):
    w,_=setup()
    with pytest.raises(ValueError):WaveIndex(w.graph,strengths={'a':bad})


@pytest.mark.parametrize('kw',[{'decay':0},{'decay':1024},{'decay':True},
                               {'minimum_salience':0},{'minimum_salience':1025}])
def test_invalid_configuration(kw):
    with pytest.raises(ValueError):WaveConfig(**kw)


def test_policy_rejected_before_root_existence():
    w,p=setup()
    with pytest.raises(PermissionError):w.select(Request(('missing',),Task.PERSONAL_RECALL),Policy('other'))
    with pytest.raises(PermissionError):w.select(Request(('missing',),Task.CONCEPT_EXPLANATION),Policy('scope'))


def test_late_sorted_strong_branch_survives_node_cap():
    g=graph([Node(k,Plane.PERSONAL) for k in ('root','a','b','z')],
            [Link(k,'root',k) for k in ('a','b','z')])
    w=WaveIndex(g,strengths={'a':256,'b':512,'z':1024})
    r=w.select(Request(('root',),Task.PERSONAL_RECALL,max_nodes=2,max_depth=1),Policy('scope'))
    assert set(r['active'])=={'root','z'}
    assert r['status']=='incomplete' and 'node_budget' in r['stop_reasons']
    assert r['salience']['z']==896


def test_cycles_and_duplicate_edges_cannot_sum_into_confidence():
    ns=[Node(k,Plane.PERSONAL) for k in ('r','a','b')]
    es=[Link('x','r','a'),Link('y','a','b'),Link('z','b','r'),Link('duplicate','r','a')]
    w=WaveIndex(graph(ns,es));r=w.select(Request(('r',),Task.PERSONAL_RECALL,max_depth=8),Policy('scope'))
    assert r['salience']=={'r':1024,'a':896,'b':896}
    assert r['work']['expansions']==3
    assert r['work']['edge_pings']==8


def test_shallower_weaker_path_can_reach_a_child_with_remaining_depth():
    ns=[Node(k,Plane.PERSONAL) for k in ('r','a','b','c')]
    es=[Link('strong','r','a'),Link('bridge','a','b'),Link('weak','r','b'),Link('child','b','c')]
    w=WaveIndex(graph(ns,es),strengths={'weak':512})
    r=w.select(Request(('r',),Task.PERSONAL_RECALL,max_depth=2),Policy('scope'))
    assert 'c' in r['active'] and r['salience']['b']==784
    assert r['salience']['c']==392


def test_duplicate_order_does_not_change_decision_receipts():
    ns=[Node(k,Plane.PERSONAL) for k in ('r','a','b')]
    es=[Link('ab','a','b'),Link('ra','r','a')]
    r=Request(('r',),Task.PERSONAL_RECALL);p=Policy('scope')
    assert WaveIndex(graph(ns,es)).select(r,p)==WaveIndex(graph(reversed(ns),reversed(es))).select(r,p)


def test_small_random_graphs_agree_with_independent_depth_dynamic_program():
    rng=random.Random(149)
    for case in range(40):
        ns=[Node(str(i),Plane.PERSONAL) for i in range(7)]
        es=[Link(f'{a}-{b}',str(a),str(b)) for a in range(7) for b in range(a+1,7) if rng.random()<.3]
        weights={e.key:rng.randint(300,1024) for e in es}
        cfg=WaveConfig(minimum_salience=1);w=WaveIndex(graph(ns,es),strengths=weights)
        req=Request(('0',),Task.PERSONAL_RECALL,max_depth=4,max_nodes=100,max_edges=4096)
        out=w.select(req,Policy('scope'),cfg)
        best={'0':1024};layer={'0':1024}
        for depth in range(4):
            new={}
            for source,score in layer.items():
                for e in es:
                    if source not in (e.source,e.target):continue
                    dest=e.target if source==e.source else e.source
                    val=score*weights[e.key]*cfg.decay//(1024*1024)
                    if val>=cfg.minimum_salience:new[dest]=max(new.get(dest,0),val)
            for n,s in new.items():best[n]=max(best.get(n,0),s)
            layer=new
        assert out['salience']==best,case


def test_budgeted_work_is_explicit_and_finite():
    w,p=setup();req=Request(('person',),Task.PERSONAL_RECALL,max_edges=1)
    result=w.select(req,p)
    assert result['work']['edge_pings']==1 and result['status']=='incomplete'
    assert 'edge_budget' in result['stop_reasons']


def test_rehashed_receipt_tampering_is_not_accepted():
    w,p=setup();req=Request(('person',),Task.PERSONAL_RECALL)
    r=w.select(req,p);w.verify(r,req,p)
    r['active'].append('fact');r['digest']=fingerprint({k:v for k,v in r.items() if k!='digest'})
    with pytest.raises(ValueError):w.verify(r,req,p)


def test_changed_weights_and_graph_generation_invalidate_receipt():
    w,p=setup();req=Request(('person',),Task.PERSONAL_RECALL);r=w.select(req,p)
    with pytest.raises(ValueError):WaveIndex(w.graph,strengths={'a':512}).verify(r,req,p)
    g=ActivationIndex(w.graph.nodes.values(),w.graph.links.values(),context='scope',content_generation='test-v2')
    with pytest.raises(ValueError):WaveIndex(g).verify(r,req,p)
