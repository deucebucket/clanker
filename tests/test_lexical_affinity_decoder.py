"""Open-development conformance tests; no held-out data used for tuning."""
from __future__ import annotations

import copy
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from clanker_lm.affect import AffectController, HeuristicAffectBackend
from clanker_lm.database import LanguageStore
from clanker_lm.memory import ConversationMemory
from clanker_lm.model import (AffectReading, AffectVector, AnswerContract, AnswerStatus,
    CandidateResponse, EventFrame, GateDecision, SemanticRef, SourceKind)
from experiments.lexical_decoder import AffinityStore, DecodeError, DecoderConfig, LexicalDecoder, ReceiptChat
from experiments.lexical_decoder.affinity import canonical, digest, tokens
from experiments.lexical_decoder.grammar import (Clause, Grammar, ResponsePlan, Slot, Word,
    atom, compose, fixed, predicate)
from experiments.lexical_decoder.runtime import FrontierController, development_pack


ROOT = Path(__file__).resolve().parents[1]
CLOCK = lambda: datetime(2026, 9, 9, 12, 34, tzinfo=timezone.utc)


def backend_read(text):
    # A transparent deterministic fixture, not a claimed human response effect.
    return AffectReading(AffectVector(v=230 if 'good' in text.lower() else 40 if 'rough' in text.lower() else 128), backend='oracle-fixture')


def direct_transition(before, response):
    return response


def decode_kwargs():
    return dict(read=backend_read, transition=direct_transition,
                runtime_hash='a'*64, context_hash='b'*64, backend_name='oracle-fixture')


def tiny_plan():
    return ResponsePlan('evaluation', 'c'*64, (
        Slot('adjective','assessment',(Word('good','good','assessment'),Word('rough','rough','assessment'))),
        atom('.', 'punctuation'),
    ), ('INTERJECTION',), {})


@pytest.fixture
def pack():
    model = AffinityStore.compile(['good good good', 'rough'], source_id='test-only-original')
    yield model
    model.close()


def test_count_denominator_backoff_and_unknown_class():
    p = AffinityStore.compile(['go to the park', 'go to the school', 'go to work'], source_id='counts-example')
    try:
        r = p.probability(('go','to'), 'the')
        assert (r['count'],r['context_total']) == (2,3)
        assert r['probability'] == r['numerator']/r['denominator']
        assert r['probability'] != 2/3  # smoothing is declared, not concealed
        assert p.probability(('never','observed'),'the')['backoff_steps'] == 2
        assert p.probability(('go','to'),'dax')['probability_kind'] == 'unknown_class'
        assert p.probability(('go','to'),'dax')['probability'] > 0
    finally:
        p.close()


@pytest.mark.parametrize('purpose',['heldout','evaluation','private','training','promotion'])
def test_wrong_data_purpose_rejected(purpose):
    with pytest.raises(ValueError):
        AffinityStore.compile(['example'],source_id='bad',purpose=purpose)


def test_sqlite_snapshot_and_pack_hash_roundtrip(pack, tmp_path):
    path = tmp_path/'counts.sqlite'
    pack.save(path)
    loaded = AffinityStore.load(path)
    try:
        assert pack.pack_hash == loaded.pack_hash
        assert pack.probability(('good',),'good') == loaded.probability(('good',),'good')
        with pytest.raises(FileExistsError):
            loaded.save(path)
    finally:
        loaded.close()


def test_pack_digest_rejects_tampering(pack):
    payload = pack.to_payload()
    payload['rows'][0][2] += 1
    with pytest.raises(ValueError, match='digest'):
        AffinityStore.from_payload(payload)


def test_pack_content_not_sqlite_layout_defines_identity(pack):
    payload=pack.to_payload()
    payload['rows'].reverse()
    other=AffinityStore.from_payload(payload)
    try:
        assert other.pack_hash==pack.pack_hash
    finally:
        other.close()


def test_fixture_recompiles_without_loading_any_evaluation_data():
    path=ROOT/'experiments/lexical_decoder/fixtures/development.txt'
    compiled=AffinityStore.compile((s.strip() for s in path.read_text().splitlines() if s.strip()),source_id='clanker-original-affinity-smoke-v1')
    default=development_pack()
    try:
        assert compiled.to_payload()==default.to_payload()
    finally:
        compiled.close();default.close()


def test_both_affinity_and_vadugwi_can_change_real_selected_word(pack):
    plan=tiny_plan();gates=GateDecision();observed=AffectVector()
    lexical=LexicalDecoder(pack,DecoderConfig(affect_weight=0))
    assert lexical.decode(plan,gates,observed,AffectVector(v=40),**decode_kwargs()).selected.text=='Good.'
    affect=LexicalDecoder(pack,DecoderConfig(affect_weight=100))
    assert affect.decode(plan,gates,observed,AffectVector(v=40),**decode_kwargs()).selected.text=='Rough.'
    assert affect.decode(plan,gates,observed,AffectVector(v=230),**decode_kwargs()).selected.text=='Good.'
    other=AffinityStore.compile(['rough']*30+['good'],source_id='other-development')
    try:
        assert LexicalDecoder(other,DecoderConfig(affect_weight=0)).decode(plan,gates,observed,AffectVector(),**decode_kwargs()).selected.text=='Rough.'
    finally:
        other.close()


def test_invalid_semantics_and_pools_never_compete(pack):
    choices=(Word('bad','good','wrong'),Word('locked','rough','right',('humor',)),Word('legal','okay','right',('neutral',)))
    plan=ResponsePlan('test','c'*64,(Slot('adjective','right',choices),atom('.','punct')),(),{})
    gates=GateDecision(locked_pools=['humor'],allowed_pools=['neutral'])
    decoded=LexicalDecoder(pack).decode(plan,gates,AffectVector(),AffectVector(v=230),**decode_kwargs())
    assert decoded.selected.text=='Okay.'
    records=decoded.receipt['narrowing'][0]['decisions']
    assert {r['word_id']:r['rejected_by'] for r in records}=={'bad':['semantic_concept'],'locked':['locked_pool','pool_not_enabled'],'legal':[]}
    assert not any(n['word_id'] in {'bad','locked'} for n in decoded.receipt['nodes'])


def test_all_invalid_returns_error_not_best_invalid(pack):
    plan=ResponsePlan('test','c'*64,(Slot('x','right',(Word('bad','good','wrong'),)),),(),{})
    with pytest.raises(DecodeError,match='no_eligible_token'):
        LexicalDecoder(pack).decode(plan,GateDecision(),AffectVector(),AffectVector(),**decode_kwargs())


def test_invalid_extreme_priority_cannot_win_new_controller():
    with ReceiptChat(affect_backend=HeuristicAffectBackend()) as c:
        bad=CandidateResponse('Wrong.', 'bad',semantic_valid=False,priority=10**12)
        good=CandidateResponse('Okay.', 'good',semantic_valid=True)
        winner, candidates=c.controller.rank_candidates([bad,good],AffectVector(),AffectVector())
        assert winner is good and candidates==[good]
        with pytest.raises(DecodeError,match='no_valid_candidate'):
            c.controller.rank_candidates([bad],AffectVector(),AffectVector())


@pytest.mark.parametrize('field,value', [('beam_width',0),('max_nodes',True),('max_tokens',65),('affect_weight',-1),('lexical_weight',float('nan')),('max_receipt_bytes',0)])
def test_invalid_search_config_rejected(field,value):
    with pytest.raises(ValueError):
        DecoderConfig(**{field:value})


@pytest.mark.parametrize('config,reason',[(DecoderConfig(max_nodes=1),'node_budget'),(DecoderConfig(max_tokens=1),'token_budget'),(DecoderConfig(max_receipt_bytes=100),'receipt_budget')])
def test_budget_exhaustion_never_emits_a_partial_reply(pack,config,reason):
    with pytest.raises(DecodeError,match=reason):
        LexicalDecoder(pack,config).decode(tiny_plan(),GateDecision(),AffectVector(),AffectVector(),**decode_kwargs())


def test_tiny_exhaustive_oracle_matches_search(pack):
    choices=tuple(Word(x,x,'color') for x in ('blue','red','green'))
    plan=ResponsePlan('oracle','c'*64,(Slot('a','color',choices),Slot('b','color',choices),atom('.','punct')),(),{})
    config=DecoderConfig(beam_width=16,affect_weight=7)
    outcome=LexicalDecoder(pack,config).decode(plan,GateDecision(),AffectVector(),AffectVector(v=230),**decode_kwargs())
    results=[]
    for a,b in itertools.product(choices,repeat=2):
        words=(a.surface,b.surface,'.');ids=(a.identity,b.identity,plan.slots[-1].options[0].identity)
        lexical=sum(pack.probability(words[:i],word)['cost_micros'] for i,word in enumerate(words))
        affect=round(backend_read(compose(words)).vector.distance(AffectVector(v=230))/255*1_000_000)
        results.append((lexical+config.affect_weight*affect,ids,compose(words)))
    best=min(results)
    assert (outcome.receipt['winning_cost'],tuple(outcome.receipt['selected_word_ids']),outcome.selected.text)==best
    assert not any(p['pruned_search_budget_not_semantic_rejection'] for p in outcome.receipt['pruning'])


def test_pruning_is_separate_from_hard_rejection(pack):
    decoded=LexicalDecoder(pack,DecoderConfig(beam_width=1)).decode(tiny_plan(),GateDecision(),AffectVector(),AffectVector(),**decode_kwargs())
    assert any(p['pruned_search_budget_not_semantic_rejection'] for p in decoded.receipt['pruning'])
    assert all(d['eligible'] for d in decoded.receipt['narrowing'][0]['decisions'])


def test_receipt_replays_and_changed_inputs_do_not(pack):
    decoder=LexicalDecoder(pack);args=(tiny_plan(),GateDecision(),AffectVector(),AffectVector())
    receipt=decoder.decode(*args,**decode_kwargs()).receipt
    assert decoder.replay(receipt,*args,**decode_kwargs()).receipt==receipt
    corrupt=copy.deepcopy(receipt);corrupt['nodes'][0]['total_cost']+=1
    with pytest.raises(DecodeError,match='digest'):
        decoder.replay(corrupt,*args,**decode_kwargs())
    corrupt['receipt_sha256']=digest({k:v for k,v in corrupt.items() if k!='receipt_sha256'})
    with pytest.raises(DecodeError,match='reproduce'):
        decoder.replay(corrupt,*args,**decode_kwargs())
    with pytest.raises(DecodeError,match='reproduce'):
        decoder.replay(receipt,tiny_plan(),GateDecision(),AffectVector(),AffectVector(v=1),**decode_kwargs())


def test_affinity_receipt_raw_and_constrained_probability_are_distinct(pack):
    decoded=LexicalDecoder(pack).decode(tiny_plan(),GateDecision(),AffectVector(),AffectVector(),**decode_kwargs())
    terminal=decoded.receipt['nodes'][-1]
    assert terminal['conditional_probability_among_slot_choices']==1.0
    assert terminal['lexical']['probability']<1.0
    assert terminal['lookahead_word_ids']==[]


def test_neither_corpus_sentences_nor_legacy_realization_generate_reply(monkeypatch):
    from clanker_lm.realize import SurfaceRealizer
    def fail(*args,**kwargs):
        raise AssertionError('legacy completed-candidate realization was called')
    monkeypatch.setattr(SurfaceRealizer,'realize',fail)
    monkeypatch.setattr(SurfaceRealizer,'render_event',fail)
    with ReceiptChat() as c:
        assert c.process('My sister bought a Honda yesterday.').response
        assert 'Honda' in c.process('Who bought the Honda?').response
        assert c.receipt['backend']=='clanker-v8'
        assert c.receipt['grammar_rules_executed']
        assert c.replay_last()


@pytest.mark.parametrize('text,required', [('Hello.','Hello'),('I am sad.','?'),('I won the game.','sounds'),('Thanks.','welcome'),('Goodbye.','Goodbye')])
def test_responding_not_just_fact_querying(text,required):
    with ReceiptChat() as c:
        r=c.process(text)
        assert required.lower() in r.response.lower()
        assert c.receipt['coverage']=='supported'
        assert ''.join(c.committed_chunks())==r.response
        assert c.replay_last()


@pytest.mark.parametrize('subject',['Sarah','John','My sister','My brother'])
@pytest.mark.parametrize('negative',[False,True])
def test_real_runtime_preserves_subject_patient_tense_and_polarity(subject,negative):
    with ReceiptChat() as c:
        c.process(f'{subject} '+('did not buy' if negative else 'bought')+' a Honda yesterday.')
        r=c.process(f'Did {subject.lower() if subject.startswith("My") else subject} buy a Honda yesterday?')
        assert r.contract.status == (AnswerStatus.FALSE if negative else AnswerStatus.TRUE)
        assert c.receipt['coverage']=='supported'
        assert ('not' in tokens(r.response))==negative
        assert 'Honda' in r.response
        assert ('your '+subject[3:]).lower() in r.response.lower() if subject.startswith('My') else subject in r.response


def test_unknown_is_not_a_common_word_completion():
    with ReceiptChat() as c:
        r=c.process('Did Sarah buy a helicopter?')
        assert r.contract.status==AnswerStatus.UNKNOWN
        assert 'not know' in r.response
        assert 'bought' not in r.response


def test_attributed_or_complex_content_declines_instead_of_flattening():
    with ReceiptChat() as c:
        c.process('Sarah planned to leave.')
        r=c.process('What did Sarah plan?')
        assert c.receipt['coverage'].startswith('declined')
        assert r.contract.status==AnswerStatus.UNSUPPORTED
        assert r.response.endswith('?')
        assert 'Sarah left' not in r.response


def test_live_commands_and_learning_have_real_receipts():
    with ReceiptChat(clock=CLOCK) as c:
        r=c.process('What time is it in Tokyo?')
        assert '9:34 PM' in r.response
        assert r.resolver and c.receipt['coverage']=='supported'
        assert c.process('That game was glorp.').contract.status==AnswerStatus.LEXICAL_PROBE
        c.process('Glorp means negative and disappointing.')
        before=c.runtime.store.connection.execute('SELECT count(*) FROM lexical_evidence').fetchone()[0]
        r=c.process('What does glorp mean?')
        assert 'may mean' in r.response
        assert c.replay_last()
        after=c.runtime.store.connection.execute('SELECT count(*) FROM lexical_evidence').fetchone()[0]
        assert after==before  # decoding/replay never self-trains on output


def test_snapshot_continuation_matches_uninterrupted_runtime():
    with ReceiptChat(clock=CLOCK) as original, ReceiptChat(clock=CLOCK) as restored:
        original.process('My sister called yesterday.')
        restored.restore(original.snapshot())
        a=original.process('Who called yesterday?');b=restored.process('Who called yesterday?')
        assert a.response==b.response
        assert a.observed_state==b.observed_state
        assert a.predicted_state==b.predicted_state
        assert original.receipt==restored.receipt


def test_failed_turn_rolls_back_both_memory_and_sqlite(monkeypatch):
    with ReceiptChat() as c:
        c.process('My sister called yesterday.')
        before=c.snapshot()
        def fail(*args,**kwargs):
            raise DecodeError('injected failure')
        monkeypatch.setattr(c.controller.decoder,'decode',fail)
        with pytest.raises(DecodeError):
            c.process('Sarah bought a Honda yesterday.')
        assert c.snapshot()==before
        r=c.process('Who bought a Honda yesterday?')
        assert r.contract.status==AnswerStatus.UNKNOWN


def test_user_bound_punctuation_cannot_append_an_assertion():
    with LanguageStore() as store:
        g=Grammar(ConversationMemory(),store)
        with pytest.raises(DecodeError):
            g.ref(SemanticRef.literal('bad','Sarah. John left.'),'subject')


def test_clause_grammar_executes_agreement_and_auxiliary_rules():
    with LanguageStore() as store:
        g=Grammar(ConversationMemory(),store)
        def output(c):
            return compose(s.options[0].surface for s in g.clause(c))
        assert output(Clause(fixed('I','s'),(predicate('be'),),'be',fixed('ready','c'),agreement='first'))=='I am ready.'
        assert output(Clause(fixed('they','s'),(predicate('buy'),),'buy',fixed('books','o'),tense='past',polarity=False,agreement='plural'))=='They did not buy books.'
        assert output(Clause(fixed('Sarah','s'),(predicate('buy'),),'buy',fixed('books','o'),tense='past',mode='polar'))=='Did Sarah buy books?'


def test_cli_runs_stateful_chat_and_writes_executed_receipt(tmp_path):
    receipt=tmp_path/'receipt.json';snapshot=tmp_path/'session.json'
    result=subprocess.run([sys.executable,'-m','experiments.lexical_decoder','--receipt',str(receipt),'--snapshot',str(snapshot)],
        input='My sister bought a Honda yesterday.\nWho bought the Honda?\nThanks.\n/quit\n',text=True,capture_output=True,cwd=ROOT,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'Honda' in result.stdout and 'welcome' in result.stdout
    recorded=json.loads(receipt.read_text());assert recorded['stop_reason']=='complete'
    assert json.loads(snapshot.read_text())['has_response'] is True


def test_receipts_are_hashseed_stable():
    script='''from experiments.lexical_decoder import ReceiptChat
from datetime import datetime,timezone
with ReceiptChat(clock=lambda:datetime(2026,9,9,tzinfo=timezone.utc)) as c:
 c.process("I am sad.")
 print(c.receipt["receipt_sha256"])
'''
    outputs=[]
    for seed in ('0','1','42'):
        result=subprocess.run([sys.executable,'-c',script],text=True,capture_output=True,cwd=ROOT,env={**os.environ,'PYTHONHASHSEED':seed},timeout=20)
        assert result.returncode==0,result.stderr
        outputs.append(result.stdout)
    assert len(set(outputs))==1


def test_real_v8_target_changes_next_word_selection(pack):
    from clanker_lm.affect import ClankerAffectBackend
    backend = ClankerAffectBackend()
    observed = AffectVector()
    decoder = LexicalDecoder(pack, DecoderConfig(affect_weight=1000))
    results = []
    for word in ("Good.", "Rough."):
        target = backend.transition(observed, backend.analyze(word).vector)
        result = decoder.decode(tiny_plan(), GateDecision(), observed, target,
            read=backend.analyze, transition=backend.transition, runtime_hash="a"*64,
            context_hash="b"*64, backend_name=backend.name)
        results.append(result.selected.text)
    assert results == ["Good.", "Rough."]
