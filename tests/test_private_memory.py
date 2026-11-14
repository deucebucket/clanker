"""Principal isolation at the service boundary; native runtime, no graph required.

All principals and conversations are authored fixtures. The test supplies trusted
server identities, not an implemented login provider or a browser auth claim.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import subprocess
import sys
import os
from pathlib import Path
import time

import pytest
from clanker_lm import ClankerLM, LanguageStore
from experiments.private_memory import PrivateMemoryService, ScopeBusy, ScopeError, canonical

KEY = bytes(range(32))  # Test-only key; never a production default.


def service(root, **kw):
    return PrivateMemoryService(root, secret=KEY, configuration_id="native-test-v1", **kw)


def state(s, principal, compartment="personal"):
    return json.loads(s.export_snapshot(principal, compartment=compartment))["state"]


def names(s, principal, compartment="personal"):
    return {n["label"]: n["reference"] for n in s.entities(principal, compartment=compartment)["entities"]}


@pytest.mark.parametrize("key", [None, b"", b"short", "a"*32, b"x"*31])
def test_service_has_no_implicit_or_weak_server_key(tmp_path,key):
    with pytest.raises(ScopeError):
        PrivateMemoryService(tmp_path/'private', secret=key, configuration_id='test')


@pytest.mark.parametrize('principal',['', ' ', None, 7, 'x'*257, 'x\n'])
def test_principal_is_explicit_bounded_server_identity(tmp_path,principal):
    with pytest.raises(ScopeError):service(tmp_path/'private').namespace(principal)


def test_namespaces_are_not_names_and_differ_by_user_and_compartment(tmp_path):
    s=service(tmp_path/'private')
    a=s.namespace('tenant1/auth-user-A');b=s.namespace('tenant1/auth-user-B')
    assert a != b and len(a)==64
    assert a==service(tmp_path/'private').namespace('tenant1/auth-user-A')
    assert a != s.namespace('tenant1/auth-user-A', 'work')
    assert s.namespace('USER') != s.namespace('user')
    assert 'auth-user' not in a


def test_two_jordans_have_same_local_spelling_but_different_resolvable_ids(tmp_path):
    s=service(tmp_path/'private')
    for p,item in [('A','book'),('B','coat')]:
        s.process(p, f'Jordan bought a {item} yesterday.')
    for p,item in [('A','book'),('B','coat'),('A','book')]:
        r=s.process(p,'What did Jordan buy?')
        assert r['turn']['contract']['status']=='answered'
        assert r['turn']['contract']['values'][0]['surface'].endswith(item)
        assert s.verify_response(p,r)
        assert not s.verify_response('B' if p=='A' else 'A',r)
    ar,br=names(s,'A')['Jordan'],names(s,'B')['Jordan']
    assert ar!=br
    assert s.entity('A',ar)['record']['entity_id']==s.entity('B',br)['record']['entity_id']=='jordan_1'
    before=s.export_snapshot('B')
    with pytest.raises(ScopeError):s.entity('B',ar)
    assert s.export_snapshot('B')==before


def test_snapshot_from_other_user_rejected_before_restore_callback(tmp_path):
    s=service(tmp_path/'private');s.process('A','Jordan bought a coat.')
    raw=s.export_snapshot('A');calls=[]
    other=service(tmp_path/'private',restore_factory=lambda data:calls.append(data))
    with pytest.raises(ScopeError):other.restore_snapshot('B',raw)
    assert calls==[]
    assert not (tmp_path/'private'/(s.namespace('B')+'.json')).exists()


@pytest.mark.parametrize('alteration',['namespace','payload','mac','config','runtime','generation','nonfinite'])
def test_relabelled_or_edited_envelope_cannot_be_rehashed_into_another_user(tmp_path,alteration):
    s=service(tmp_path/'private');s.process('A','Jordan bought a coat.')
    raw=json.loads(s.export_snapshot('A'))
    before=s.export_snapshot('A')
    if alteration=='namespace':raw['namespace']=s.namespace('B')
    if alteration=='payload':raw['state']['memory']['events'][0]['predicate']='steal'
    if alteration=='mac':raw['mac']='0'*64
    if alteration=='config':raw['configuration_id']='new'
    if alteration=='runtime':raw['runtime_id']='new'
    if alteration=='generation':raw['generation']+=1
    if alteration=='nonfinite':raw['state']['observed_state']['v']=float('nan')
    with pytest.raises(ValueError):s.restore_snapshot('A',json.dumps(raw).encode())
    assert s.export_snapshot('A')==before


def test_unsigned_legacy_snapshot_has_no_implicit_owner_migration(tmp_path):
    s=service(tmp_path/'private')
    with ClankerLM() as r:
        r.process('Jordan bought a bike.')
        with pytest.raises(ScopeError):s.restore_snapshot('A',r.dumps().encode())


def test_same_user_resume_and_separate_compartment(tmp_path):
    s=service(tmp_path/'private')
    s.process('A','Jordan bought a bike.',compartment='personal')
    s.process('A','Jordan bought a book.',compartment='work')
    restarted=service(tmp_path/'private')
    for compartment,item in [('personal','bike'),('work','book')]:
        result=restarted.process('A','What did Jordan buy?',compartment=compartment)
        assert result['turn']['contract']['status']=='answered'
        assert result['turn']['contract']['values'][0]['surface'].endswith(item)
    raw=restarted.export_snapshot('A')
    with pytest.raises(ScopeError):restarted.restore_snapshot('A',raw,compartment='work')


def test_different_word_senses_are_in_different_private_sql_overlays(tmp_path):
    s=service(tmp_path/'private')
    s.process('A','Glorp means negative and disappointing.')
    s.process('B','Glorp means positive and wonderful.')
    la=s.lexicon('A')['terms'];lb=s.lexicon('B')['terms']
    assert la[0]['senses'][0]['semantic_class']=='negative_evaluation'
    assert lb[0]['senses'][0]['semantic_class']=='positive_evaluation'
    assert s.lexicon('C')['terms']==[]
    for p in ('A','B'):
        ss=state(s,p)
        assert ss['learner']['scope_id']==s.namespace(p)
        assert all(row['scope_id']==s.namespace(p) for row in ss['language_overlay']['tables']['learned_senses'])


def test_shared_language_database_factory_is_refused(tmp_path):
    db=tmp_path/'shared.sqlite3'
    s=service(tmp_path/'private',factory=lambda ns:ClankerLM(language_store=LanguageStore(str(db)),learning_scope=ns))
    with pytest.raises(ScopeError,match='shared databases'):
        s.process('A','Hello.')
    assert not (tmp_path/'private'/(s.namespace('A')+'.json')).exists()


def test_factory_must_use_assigned_learning_scope(tmp_path):
    s=service(tmp_path/'private',factory=lambda ns:ClankerLM(learning_scope='default'))
    with pytest.raises(ScopeError,match='scope'):s.process('A','Hello.')


def test_failed_turn_does_not_commit_partial_mutations(tmp_path):
    s=service(tmp_path/'private');s.process('A','Jordan bought a bike.')
    before=s.export_snapshot('A')
    def failing_restore(data):
        lm=ClankerLM.from_dict(data)
        def broken(_text):
            lm.memory.begin_turn()
            lm.memory.events.clear()
            raise RuntimeError('synthetic failure after mutation')
        lm.process=broken
        return lm
    failed=service(tmp_path/'private',restore_factory=failing_restore)
    with pytest.raises(RuntimeError):failed.process('A','Who bought the bike?')
    assert s.export_snapshot('A')==before
    assert s.process('A','Who bought the bike?')['turn']['contract']['status']=='answered'


def test_reset_only_clears_own_compartment_and_old_backup_cannot_undo_it(tmp_path):
    s=service(tmp_path/'private')
    s.process('A','Jordan bought a coat.');old=s.export_snapshot('A')
    s.process('B','Jordan bought a book.');b=s.export_snapshot('B')
    s.reset('A')
    assert state(s,'A')['memory']['events']==[]
    assert s.export_snapshot('B')==b
    with pytest.raises(ScopeError,match='stale'):s.restore_snapshot('A',old)


def test_restore_to_private_new_host_with_same_key_identity_and_config(tmp_path):
    source=service(tmp_path/'first');source.process('A','Jordan bought a coat.')
    raw=source.export_snapshot('A')
    target=service(tmp_path/'second');target.restore_snapshot('A',raw)
    assert target.export_snapshot('A')==raw
    for text in ['Who bought the coat?','When did Jordan buy it?','Hello.']:
        a=source.process('A',text);b=target.process('A',text)
        # Database wall-clock text columns do not participate in these turn results.
        assert a==b


def test_old_generation_restore_refused_but_current_export_idempotent(tmp_path):
    s=service(tmp_path/'private');s.process('A','Jordan called Sarah.');old=s.export_snapshot('A')
    s.process('A','Who called Sarah?')
    latest=s.export_snapshot('A');s.restore_snapshot('A',latest)
    assert latest==s.export_snapshot('A')
    with pytest.raises(ScopeError):s.restore_snapshot('A',old)


def test_message_and_turn_limits_do_not_modify_other_state(tmp_path):
    s=service(tmp_path/'private',max_turns=1);s.process('A','Hello.')
    before=s.export_snapshot('A')
    with pytest.raises(ScopeError):s.process('A','Hello again.')
    with pytest.raises(ScopeError):s.process('B','x'*8193)
    assert s.export_snapshot('A')==before


def test_user_ids_cannot_be_used_as_paths(tmp_path):
    s=service(tmp_path/'private')
    for p in ['../../outside','user/A','user:B']:
        s.process(p,'Hello.')
    assert all(len(p.stem)==64 for p in (tmp_path/'private').iterdir())
    assert set(p.name for p in tmp_path.iterdir())=={'private'}


def test_storage_files_are_private_and_symlinks_are_not_followed(tmp_path):
    s=service(tmp_path/'private');s.process('A','Hello.')
    assert all(p.stat().st_mode & 0o077 == 0 for p in (tmp_path/'private').iterdir())
    target=tmp_path/'outside';target.write_text('untouched')
    (tmp_path/'private'/(s.namespace('B')+'.json')).symlink_to(target)
    with pytest.raises(OSError):s.process('B','Hello.')
    assert target.read_text()=='untouched'


def test_insecure_existing_directory_refused(tmp_path):
    root=tmp_path/'open';root.mkdir(mode=0o755)
    with pytest.raises(ScopeError):service(root)


def test_concurrent_threads_do_not_lose_turns_or_share_people(tmp_path):
    s=service(tmp_path/'private')
    work=[(p,n) for n in range(6) for p in ('A','B','C')]
    def run(item):
        p,n=item
        return service(tmp_path/'private').process(p, f'Person{p} called Contact{n}.')
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(run,work))
    assert len(rows)==18
    for p in ('A','B','C'):
        ss=state(s,p)
        assert ss['memory']['turn_index']==6
        labels=[e['canonical_name'] for e in ss['memory']['entities']]
        assert f'Person{p}' in labels
        assert not any(f'Person{x}' in labels for x in ('A','B','C') if x!=p)


def test_process_workers_serialize_same_private_compartment(tmp_path):
    root=tmp_path/'private';service(root)
    repo=str(Path(__file__).resolve().parent.parent)
    script = ("import sys;sys.path.insert(0,sys.argv[1]);"
              "from experiments.private_memory import PrivateMemoryService;"
              "s=PrivateMemoryService(sys.argv[2],secret=bytes(range(32)),configuration_id='native-test-v1');"
              "s.process('A','Jordan called Contact'+sys.argv[3]+'.')")
    workers=[subprocess.Popen([sys.executable,'-S','-c',script,repo,str(root),str(n)],
                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for n in range(6)]
    try:
        for child in workers:
            stdout,stderr=child.communicate(timeout=20)
            assert child.returncode==0,stdout+stderr
    finally:
        for child in workers:
            if child.poll() is None:
                child.kill();child.wait(timeout=5)
    assert state(service(root),'A')['memory']['turn_index']==6


def test_lock_wait_is_bounded(tmp_path):
    s=service(tmp_path/'private',lock_timeout=.02)
    with s._locked(s.namespace('A')):
        with pytest.raises(ScopeBusy):s.process('A','Hello.')


def test_no_graph_is_explicit_not_a_fake_empty_shared_graph(tmp_path):
    s=service(tmp_path/'private')
    with pytest.raises(ScopeError,match='no associative graph'):s.graph('A')


def test_modified_response_receipt_or_foreign_reference_is_rejected(tmp_path):
    s=service(tmp_path/'private');r=s.process('A','Hello.')
    altered=deepcopy(r);altered['turn']['response']='invented answer'
    assert not s.verify_response('A',altered)
    assert not s.verify_response('B',r)
    assert not s.verify_response('A',r,compartment='other')
    with pytest.raises(ScopeError):s.entity('A','jordan_1')


def test_wrong_key_or_config_refuses_import(tmp_path):
    s=service(tmp_path/'private');s.process('A','Hello.');payload=s.export_snapshot('A')
    for key,config in [(b'B'*32,'native-test-v1'),(KEY,'new-policy')]:
        other=PrivateMemoryService(tmp_path/'new',secret=key,configuration_id=config)
        with pytest.raises(ScopeError):other.restore_snapshot('A',payload)
