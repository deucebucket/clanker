"""Open development assertions for issues #126/#127; no sealed corpus imports."""
import copy
from datetime import datetime, timezone

import pytest

from clanker_lm.model import SemanticRef, EntityKind, RefKind, SourceKind
from experiments.lexical_decoder import ReceiptChat, DecoderConfig, DecodeError
from experiments.lexical_decoder.repairs import DefinitionLearner, PerspectiveRealizer
from experiments.lexical_decoder.usage import UsageLedger

CLOCK = lambda: datetime(2026, 9, 9, 12, 34, tzinfo=timezone.utc)

@pytest.mark.parametrize("noun", ["car", "bike", "book", "phone"])
def test_owned_object_uses_identity_not_source_my(noun):
    with ReceiptChat() as c:
        c.process(f"My sister borrowed my {noun} yesterday.")
        r = c.process("What did my sister borrow?")
        assert r.contract.status.value == "answered"
        assert f"your {noun}" in r.response.lower()
        assert f"my {noun}" not in r.response.lower()
        ref = r.contract.values[0]
        assert c.runtime.memory.get_entity(ref.key).owner_id == "user"
        assert c.replay_last()


def test_literal_quotation_is_not_globally_rewritten():
    with ReceiptChat() as c:
        np = PerspectiveRealizer(c.runtime.memory, c.runtime.store)
        raw = SemanticRef.literal("q", 'Sarah said "my car"', EntityKind.ABSTRACT)
        assert np.render_ref(raw) == 'Sarah said "my car"'
        r = c.process('Sarah said John bought a car.')
        assert c.receipt['coverage'].startswith('declined') or r.contract.status.value == 'acknowledged'


@pytest.mark.parametrize("definition", ["not good", "not bad", "not positive", "never good", "not very good"])
def test_negated_hint_is_not_asserted_as_a_definition(definition):
    assert DefinitionLearner._infer_semantic_class(definition) == ("unknown", 0)
    with ReceiptChat() as c:
        r = c.process("Glorp means " + definition + ".")
        assert r.contract.status.value == "lexical_probe"
        senses = c.runtime.store.learned_senses("glorp")
        assert all(s.semantic_class == "unknown" and s.confidence < 0.68 for s in senses)


@pytest.mark.parametrize("definition,label", [
    ("not bad, actually excellent", "positive_evaluation"),
    ("negative, not positive", "negative_evaluation"),
    ("not good but terrible", "negative_evaluation"),
])
def test_contrast_has_positive_evidence_for_its_actual_meaning(definition,label):
    assert DefinitionLearner._infer_semantic_class(definition)[0] == label


def test_pending_probe_does_not_swallow_new_fact():
    with ReceiptChat() as c:
        c.process("That movie was glorp.")
        r = c.process("My sister borrowed my car yesterday.")
        assert r.contract.status.value == "acknowledged"
        assert c.runtime.learner.pending.normalized == "glorp"
        assert "your car" in c.process("What did my sister borrow?").response.lower()
        r = c.process("It means disappointing.")
        assert r.contract.status.value == "lexical_learned"
        assert c.runtime.learner.pending is None


def test_answered_question_changes_actual_act_and_preserves_episode_ids():
    with ReceiptChat() as c:
        initial = c.process("My sister pissed me off again.")
        assert initial.response.endswith("?")
        c.process("She borrowed my car yesterday.")
        c.process("She returned it late.")
        r = c.process("I was angry.")
        assert not r.response.endswith("?")
        assert r.gates.response_act == "empathic_acknowledge"
        assert c.dialogue.elaboration_answered
        assert c.dialogue.episode_events
        assert all(c.runtime.memory.get_event(x) is not None for x in c.dialogue.episode_events)
        assert any(x.startswith("dialogue:") for x in r.gates.rationale)
        assert c.replay_last()


def test_new_unrelated_participant_can_open_a_new_episode():
    with ReceiptChat() as c:
        c.process("My sister pissed me off again.")
        c.process("She borrowed my car yesterday.")
        r = c.process("My brother pissed me off.")
        assert r.gates.response_act == "empathic_followup"
        assert r.response.endswith("?")


def test_answer_focus_does_not_bind_it_to_an_adverb_phrase():
    with ReceiptChat() as c:
        for text in ["My sister pissed me off again.", "She borrowed my car yesterday.",
                     "She returned it late.", "I was angry.", "What did my sister borrow?"]:
            c.process(text)
        r = c.process("When did she borrow it?")
        assert r.contract.status.value == "answered", r.response
        assert "yesterday" in r.response.lower()
        assert not any(e.canonical_name == "it late" for e in c.runtime.memory.entities.values())


def test_stop_preference_is_an_instruction_and_survives_reload():
    with ReceiptChat(clock=CLOCK) as c, ReceiptChat(clock=CLOCK) as restored:
        c.process("My sister pissed me off again.")
        r = c.process("Can you stop asking what happened?")
        assert r.contract.status.value == "acknowledged"
        assert "not ask" in r.response.lower()
        assert c.dialogue.suppress_elaboration
        restored.restore(c.snapshot())
        for text in ["I am sad.", "My brother pissed me off.", "Thanks for listening.", "Bye."]:
            a,b = c.process(text), restored.process(text)
            assert a.response == b.response
            assert c.receipt == restored.receipt
            assert c.dialogue.to_dict() == restored.dialogue.to_dict()
            assert not a.response.endswith("?")


@pytest.mark.parametrize("message,act", [("Thanks for listening.","gratitude"),("Bye.","closure")])
def test_conventional_turn_does_not_become_an_unknown_word(message,act):
    with ReceiptChat() as c:
        r=c.process(message)
        assert r.contract.status.value == 'acknowledged'
        assert c.runtime.parser.convention == act


def test_dialogue_state_rolls_back_with_a_failed_turn(monkeypatch):
    with ReceiptChat(clock=CLOCK) as c:
        c.process("My sister pissed me off again.")
        before=c.snapshot()
        def fail(*a,**kw): raise DecodeError('test failure')
        monkeypatch.setattr(c.controller.decoder, 'decode', fail)
        with pytest.raises(DecodeError): c.process("She borrowed my car yesterday.")
        assert c.snapshot()==before


EXPOSURES = ["That sounds frustrating today.", "That sounds frustrating sometimes.",
             "That sounds frustrating again.", "That sounds frustrating indeed.",
             "That sounds frustrating certainly."]


def teach(c):
    for i,text in enumerate(EXPOSURES):
        assert c.observe_usage(text,evidence_id=f"e{i}",source_id="speaker",consent=True)


def test_observed_counts_change_actual_response_then_retract():
    with ReceiptChat(usage_scope='one', config=DecoderConfig(affect_weight=0), clock=CLOCK) as c:
        before=c.process("I am sad.").response
        raw_before=c.controller.decoder.pack.probability(("that","sounds"),"frustrating")
        initial_count = raw_before['count']
        teach(c)
        after=c.process("I am sad.").response
        raw_after=c.controller.decoder.pack.probability(("that","sounds"),"frustrating")
        assert raw_after['count']==initial_count+5
        assert raw_after['probability']>raw_before['probability']
        assert before!=after and 'frustrating' in after.lower()
        assert c.receipt['source_manifest']['ledger_sha256']==c.usage.hash
        assert c.replay_last()
        for i in range(len(EXPOSURES)):
            c.retract_usage(f'e{i}',reason='Correction: example not suitable for this speaker')
        reverted=c.process("I am sad.").response
        assert reverted==before
        assert c.controller.decoder.pack.probability(("that","sounds"),"frustrating")['count']==initial_count
        assert len(c.usage.observations)==5 and len(c.usage.retractions)==5
        assert c.replay_last()


def test_no_automatic_self_training_and_default_is_disabled():
    with ReceiptChat() as c:
        with pytest.raises(DecodeError): c.observe_usage('example',evidence_id='e',source_id='s',consent=True)
    with ReceiptChat(usage_scope='one') as c:
        for message in ['Hello.','I am sad.','Thanks.']:
            c.process(message)
        assert c.usage.observations==[]


@pytest.mark.parametrize('kwargs', [dict(source_role='assistant'),dict(purpose='heldout'),dict(purpose='evaluation'),
                                   dict(purpose='simulation'),dict(consent=False),dict(quoted=True)])
def test_ineligible_learning_is_rejected(kwargs):
    with ReceiptChat(usage_scope='one') as c:
        options=dict(evidence_id='e',source_id='s',source_role='user',purpose='session_usage',consent=True)
        options.update(kwargs)
        before=c.usage.hash
        with pytest.raises(DecodeError): c.observe_usage('example',**options)
        assert c.usage.hash==before


def test_idempotence_and_evidence_ids_cannot_be_rewritten():
    l=UsageLedger('one')
    kw=dict(evidence_id='e',source_id='s',source_role='user',purpose='session_usage',consent=True)
    assert l.observe('go to school',**kw)
    h=l.hash
    assert not l.observe('go to school',**kw)
    assert not l.observe('go to school',**dict(kw,evidence_id='new-label'))
    assert l.hash==h
    with pytest.raises(DecodeError): l.observe('go to work',**kw)
    assert l.hash==h


def test_learning_scope_snapshot_and_replay_are_stable():
    with ReceiptChat(usage_scope='one',clock=CLOCK) as c, ReceiptChat(usage_scope='one',clock=CLOCK) as restored, ReceiptChat(usage_scope='two',clock=CLOCK) as other:
        teach(c)
        c.process('I am sad.')
        snap=c.snapshot()
        restored.restore(snap)
        with pytest.raises(DecodeError,match='scope'): other.restore(snap)
        for message in ['Hello.','I am sad.','Bye.']:
            assert c.process(message).response==restored.process(message).response
            assert c.receipt==restored.receipt
        assert other.usage.observations==[]


def test_corrupt_learning_snapshot_fails_without_mutation():
    with ReceiptChat(usage_scope='one') as c:
        teach(c)
        before=c.snapshot()
        corrupt=copy.deepcopy(before)
        corrupt['usage']['observations'][0]['rows'][0][2]+=999
        with pytest.raises(ValueError): c.restore(corrupt)
        assert c.snapshot()==before


def test_count_snapshot_cannot_be_exported_as_reviewed_global_pack(tmp_path):
    with ReceiptChat(usage_scope='one') as c:
        c.process('Hello.')
        with pytest.raises(DecodeError): c.controller.decoder.pack.save(tmp_path/'promoted.sqlite')


def test_previously_absent_word_context_edge_is_created_without_adding_facts():
    with ReceiptChat(usage_scope='one') as c:
        c.process('Hello.')
        assert c.controller.decoder.pack.probability(('to','the'),'microdax')['count']==0
        before_events = len(c.runtime.memory.events)
        c.observe_usage('Go to the microdax.', evidence_id='new-word', source_id='speaker', consent=True)
        c.process('Hello.')
        assert c.controller.decoder.pack.probability(('to','the'),'microdax')['count']==1
        assert c.controller.decoder.pack.probability(('to','the'),'microdax')['probability_kind']=='word'
        assert len(c.runtime.memory.events)==before_events
        assert 'microdax' not in c.controller.last.selected.text


def test_frequent_wrong_content_and_agreement_do_not_replace_bound_facts():
    with ReceiptChat(usage_scope='one',config=DecoderConfig(affect_weight=0)) as c:
        c.process('Sarah bought a Honda yesterday.')
        for i in range(12):
            c.observe_usage(f'Sarah buy helicopters number {i}.', evidence_id=f'e{i}', source_id='speaker', consent=True)
        r=c.process('What did Sarah buy?')
        assert r.contract.status.value=='answered'
        assert 'bought' in r.response and 'Honda' in r.response
        assert 'helicopter' not in r.response


def test_explicit_object_correction_updates_answer_without_deleting_evidence():
    with ReceiptChat(clock=CLOCK) as c, ReceiptChat(clock=CLOCK) as restored:
        c.process('My sister borrowed my car yesterday.')
        old=c.runtime.memory.events[-1].event_id
        r=c.process('Actually, she borrowed my bike, not my car.')
        assert r.contract.status.value=='acknowledged'
        assert not r.response.endswith('?')
        assert c.runtime.memory.get_event(old) is not None
        assert c.dialogue.corrections[0]['old_event']==old
        restored.restore(c.snapshot())
        for text in ['What did she borrow?', 'When did she borrow my bike?']:
            a,b=c.process(text),restored.process(text)
            assert a.response==b.response
            assert 'bike' in a.response and 'car' not in a.response
            assert 'yesterday' in a.response
            assert c.receipt==restored.receipt


def test_ambiguous_correction_does_not_supersede_either_event():
    with ReceiptChat() as c:
        c.process('My sister borrowed my car yesterday.')
        c.process('My sister borrowed my car today.')
        count=len(c.runtime.memory.events)
        c.process('Actually, she borrowed my bike, not my car.')
        assert c.dialogue.corrections==[]
        assert len(c.runtime.memory.events)==count


def test_ordinary_conflicting_assertion_is_not_a_correction():
    with ReceiptChat() as c:
        c.process('Sarah opened the door.')
        c.process('Sarah did not open the door.')
        r=c.process('Did Sarah open the door?')
        assert r.contract.status.value=='conflict'
        assert c.dialogue.corrections==[]


@pytest.mark.parametrize('text', ['Actually, Sarah called yesterday, not my car.',
                                  'Actually, Sarah may borrow my bike, not my car.',
                                  'Actually, Sarah said John borrowed my bike, not my car.'])
def test_unsupported_correction_abstains_without_creating_facts(text):
    with ReceiptChat() as c:
        c.process('Sarah borrowed my car yesterday.')
        count=len(c.runtime.memory.events)
        c.process(text)
        assert c.dialogue.corrections==[]
        assert len(c.runtime.memory.events)==count


def test_cli_explicit_hearing_and_retraction(tmp_path):
    import subprocess,sys,json
    snap=tmp_path/'session.json'
    process=subprocess.run([sys.executable,'-m','experiments.lexical_decoder',
                            '--usage-scope','local','--snapshot',str(snap)],
        input='/hear e1 Go to the microdax.\nHello.\n/retract e1 bad example\n/quit\n',
        text=True,capture_output=True)
    assert process.returncode==0,process.stderr
    payload=json.loads(snap.read_text())
    assert len(payload['usage']['observations'])==1
    assert len(payload['usage']['retractions'])==1
    assert 'Hello.' in process.stdout


def test_recovery_qualification_does_not_open_a_topic_named_angry_anymore():
    with ReceiptChat() as c:
        c.process('My sister pissed me off again.')
        c.process('She borrowed my car yesterday.')
        r=c.process("I'm still a little annoyed, but I'm not angry anymore.")
        assert r.gates.response_act=='empathic_acknowledge'
        assert not r.response.endswith('?')


@pytest.mark.parametrize('text', ['Intense pain started today.', 'Negative results arrived today.'])
def test_adjective_initial_complete_fact_is_not_consumed_as_a_definition(text):
    with ReceiptChat() as c:
        c.process('That movie was glorp.')
        r=c.process(text)
        assert r.contract.status.value=='acknowledged'
        assert c.runtime.learner.pending.normalized=='glorp'
        assert r.parse.events


def test_fixed_semantic_target_full_vadugwi_ablation_changes_only_usage_weights():
    from experiments.lexical_decoder import LexicalDecoder
    with ReceiptChat(usage_scope='ablation') as c:
        before=c.process('I am sad.')
        args,kw=c.controller.replay_context
        assert c.config.affect_weight>0 and c.receipt['backend']=='clanker-v8'
        teach(c)
        newer=c.usage.compile(c.pack)
        try:
            after=LexicalDecoder(newer,c.config).decode(*args,**kw)
            assert before.response!=after.selected.text
            assert 'frustrating' in after.selected.text
            assert c.receipt['observed']==after.receipt['observed']
            assert c.receipt['target']==after.receipt['target']
            assert c.receipt['plan_sha256']==after.receipt['plan_sha256']
        finally:
            newer.close()
