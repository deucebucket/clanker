"""Independent metadata graphs test policy, traversal work and replay.

These are authored development tests, not human recognition/calibration results.
"""
from copy import deepcopy
import pytest
from clanker_lm.activation import (ActivationIndex, Node, Link, Plane, LinkKind,
    Policy, Request, Task, fingerprint)

ALL = tuple(Task)


def fixture(extra=0):
    nodes = [Node("episode", Plane.PERSONAL), Node("trevor", Plane.PERSONAL),
             Node("appraisal", Plane.PERSONAL), Node("term", Plane.SEMANTIC),
             Node("definition", Plane.SEMANTIC), Node("topic", Plane.SEMANTIC)]
    edges = [Link("a", "episode", "trevor"), Link("b", "episode", "appraisal"),
             Link("c", "episode", "term", LinkKind.CONTEXT_REFERENCE),
             Link("d", "term", "definition"), Link("e", "definition", "topic")]
    for i in range(extra):
        key = f"textbook-{i}"
        nodes.append(Node(key, Plane.SEMANTIC))
        edges.append(Link(f"academic-{i}", "term", key))
    return ActivationIndex(nodes, edges, context="scope-A", content_generation="v1")


def policy(**kwargs):
    return Policy("scope-A", allowed_tasks=ALL, **kwargs)


def test_personal_recall_only_inspects_reference_not_its_academic_neighborhood():
    idx = fixture(1000)
    r = idx.select(Request(("episode",), Task.PERSONAL_RECALL), policy())
    assert r["active"] == ["appraisal", "episode", "trevor"]
    assert r["reference_only"] == ["term"]
    assert r["work"]["expanded_nodes"] == 3
    assert r["work"]["inspected_edges"] == 5
    assert r["status"] == "complete"
    assert all(s["source"] != "term" for s in r["steps"])
    assert not r["context_edges_are_proof_premises"]
    assert not r["authentication_or_identity_inferred"]


def test_concept_explanation_expands_knowledge_but_never_reverse_personal_edges():
    idx = fixture()
    r = idx.select(Request(("term",), Task.CONCEPT_EXPLANATION), policy())
    assert r["active"] == ["definition", "term", "topic"]
    assert not r["reference_only"]
    assert all("episode" not in (s["source"], s["target"]) for s in r["steps"])


def test_explicit_comparison_changes_selected_evidence_not_just_trace():
    r = fixture().select(Request(("episode",), Task.CROSS_DOMAIN_COMPARISON, max_depth=4), policy())
    assert r["active"] == ["appraisal", "definition", "episode", "term", "topic", "trevor"]
    assert not r["reference_only"] and r["status"] == "complete"


def test_multiple_roots_can_upgrade_a_reference_to_an_explicit_active_concept():
    r = fixture().select(Request(("term", "episode"), Task.CROSS_DOMAIN_COMPARISON), policy())
    assert "term" in r["active"] and "term" not in r["reference_only"]
    assert r["work"]["selected_nodes"] == 6


def test_no_bridge_means_no_context_retrieval_even_with_same_label_outside_core():
    idx = fixture()
    reduced = ActivationIndex(idx.nodes.values(), [e for e in idx.links.values() if e.key!="c"],
                              context="scope-A", content_generation="v2")
    r = reduced.select(Request(("episode",), Task.PERSONAL_RECALL), policy())
    assert not r["reference_only"] and "term" not in r["active"]


def test_cross_plane_plain_association_cannot_bypass_the_gray_gate():
    idx = fixture()
    hacked = ActivationIndex(idx.nodes.values(), [*idx.links.values(), Link("x", "episode", "topic")],
                             context="scope-A", content_generation="v2")
    r = hacked.select(Request(("episode",), Task.PERSONAL_RECALL), policy())
    assert "topic" not in r["active"] and "topic" not in r["reference_only"]
    assert any(s["reason"] == "cross_plane_association_requires_a_typed_bridge" for s in r["steps"])


@pytest.mark.parametrize("task,root",[(Task.PERSONAL_RECALL,"term"),
    (Task.CONCEPT_EXPLANATION,"episode")])
def test_wrong_root_plane_does_not_override_task(task,root):
    with pytest.raises(ValueError): fixture().select(Request((root,),task),policy())


def test_unsupported_planes_are_never_silently_personal_or_academic():
    for plane in (Plane.AUXILIARY, Plane.SIMULATION):
        idx=ActivationIndex([Node("n",plane)],[],context="scope-A",content_generation="v1")
        for task in Task:
            with pytest.raises(ValueError): idx.select(Request(("n",),task),policy())


@pytest.mark.parametrize("key,value,reason",[("max_depth",0,"depth_budget"),
    ("max_nodes",1,"node_budget"),("max_edges",1,"edge_budget")])
def test_exhaustion_is_incomplete_with_actual_bounded_work(key,value,reason):
    request=Request(("episode",),Task.PERSONAL_RECALL,**{key:value})
    r=fixture().select(request,policy())
    assert r["status"]=="incomplete" and reason in r["stop_reasons"]
    assert r["work"]["inspected_edges"]<=request.max_edges
    assert r["work"]["selected_nodes"]<=request.max_nodes


def test_cycle_and_self_loop_terminate_and_still_report_complete_selection():
    idx=fixture()
    cyclic=ActivationIndex(idx.nodes.values(),[*idx.links.values(),Link("loop","episode","episode"),
        Link("cycle","appraisal","trevor")],context="scope-A",content_generation="v2")
    r=cyclic.select(Request(("episode",),Task.PERSONAL_RECALL),policy())
    assert r["status"]=="complete" and r["work"]["expanded_nodes"]==3


def test_policy_denial_precedes_even_root_existence_lookup():
    for root in ("episode","not-present"):
        with pytest.raises(PermissionError):
            fixture().select(Request((root,),Task.PERSONAL_RECALL),Policy("scope-B"))
        with pytest.raises(PermissionError):
            fixture().select(Request((root,),Task.PERSONAL_RECALL),
                Policy("scope-A",allowed_tasks=(Task.CONCEPT_EXPLANATION,)))


def test_same_metadata_order_replays_and_index_cannot_be_mutated():
    a=fixture()
    b=ActivationIndex(reversed(tuple(a.nodes.values())),reversed(tuple(a.links.values())),
                      context="scope-A",content_generation="v1")
    request=Request(("episode",),Task.PERSONAL_RECALL)
    r=a.select(request,policy());b.verify(r,request,policy())
    with pytest.raises(TypeError):a.nodes["new"]=Node("new",Plane.PERSONAL)


@pytest.mark.parametrize("mutation",["active","steps","generation","scope","policy"])
def test_rehashed_receipt_tampering_does_not_pass_reexecution(mutation):
    idx=fixture();req=Request(("episode",),Task.PERSONAL_RECALL);p=policy()
    r=deepcopy(idx.select(req,p))
    if mutation=="active":r["active"].append("topic")
    if mutation=="steps":r["steps"]=[]
    if mutation=="generation":r["graph_generation"]="other"
    if mutation=="scope":r["context"]="scope-B"
    if mutation=="policy":r["policy"]["revision"]=2
    r.pop("digest");r["digest"]=fingerprint(r)
    with pytest.raises(ValueError):idx.verify(r,req,p)


def test_policy_or_content_revision_invalidates_old_selection():
    idx=fixture();req=Request(("episode",),Task.PERSONAL_RECALL)
    receipt=idx.select(req,policy())
    with pytest.raises(ValueError):idx.verify(receipt,req,policy(revision=2))
    changed=ActivationIndex(idx.nodes.values(),idx.links.values(),context="scope-A",content_generation="v2")
    with pytest.raises(ValueError):changed.verify(receipt,req,policy())


@pytest.mark.parametrize("kwargs",[{"max_depth":True},{"max_nodes":0},{"max_edges":4097},
    {"max_depth":9},{"max_nodes":301},{"max_edges":0}])
def test_invalid_search_budgets_rejected(kwargs):
    with pytest.raises(ValueError):Request(("episode",),Task.PERSONAL_RECALL,**kwargs)


def test_invalid_records_rejected_before_selection():
    n=Node("p",Plane.PERSONAL);c=Node("c",Plane.SEMANTIC)
    for nodes,edges in [([n,n],[]),([n],[Link("dangling","p","absent")]),
                         ([n,c],[Link("back","c","p",LinkKind.CONTEXT_REFERENCE)]),
                         ([n,c],[Link("a","p","c"),Link("a","p","c")])]:
        with pytest.raises(ValueError):ActivationIndex(nodes,edges,context="s",content_generation="v1")
    for field in ("", " bad", "x\n", "x"*513):
        with pytest.raises(ValueError):Node(field,Plane.PERSONAL)
    with pytest.raises(ValueError):Node("p","personal")
    with pytest.raises(ValueError):Request(("p","p"),Task.PERSONAL_RECALL)
    with pytest.raises(ValueError):Policy("s",allowed_tasks=(Task.PERSONAL_RECALL,)*2)


def test_ordinary_same_plane_search_matches_independent_reachability():
    import random
    rng=random.Random(84017)
    for _ in range(60):
        ns=[Node(str(i),Plane.PERSONAL) for i in range(7)]
        es=[Link(f"{a}-{b}",str(a),str(b)) for a in range(7) for b in range(a+1,7) if rng.random()<.28]
        reachable={"0"}
        while True:
            expanded=reachable|{e.target for e in es if e.source in reachable}|{e.source for e in es if e.target in reachable}
            if expanded==reachable:break
            reachable=expanded
        idx=ActivationIndex(ns,es,context="scope-A",content_generation="random-fixture")
        got=idx.select(Request(("0",),Task.PERSONAL_RECALL,max_depth=8),policy())
        assert set(got["active"])==reachable and got["status"]=="complete"
