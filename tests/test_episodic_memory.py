import json
from pathlib import Path
import pytest
from vlm_tamp.episodic_memory import client_source,validate_procedure,store_hashes


def test_memory_retrieval_persists_and_empty_store_has_no_episode(tmp_path):
    store=tmp_path/'library';store.mkdir();work=tmp_path/'session';work.mkdir()
    namespace={'__file__':str(work/'memory.py')}
    exec(client_source(store),namespace)
    assert namespace['recall']('open the microwave')==[]
    entry=store/'experience';entry.mkdir()
    (entry/'entry.json').write_text(json.dumps({'id':'e1','task':'open the microwave','robot':'Panda'}))
    fresh={'__file__':str(work/'memory.py')};exec(client_source(store),fresh)
    result=fresh['recall']('open the microwave')
    assert result[0]['id']=='e1' and result[0]['score']==1
    assert fresh['recall']('close drawer')==[]
    assert fresh['recall']('open the microwave',robot='other robot')==[]
    assert len((work/'memory-retrieval.jsonl').read_text().splitlines())==4


def test_evidence_validation_rejects_nonexistent_or_missing_evidence():
    procedure={k:[] for k in ['task','robot','observed_failure','successful_correction','applicability',
        'procedure','do_not_repeat','limitations','attribution']}
    procedure['procedure']=[{'instruction':'Use recorded correction','evidence_events':[2]}]
    with pytest.raises(ValueError,match='Unknown evidence'):validate_procedure(procedure,[{'sequence':1}])
    procedure['procedure'][0]['evidence_events']=[1]
    assert validate_procedure(procedure,[{'sequence':1}])['event_references_valid']
    procedure['procedure'][0]['evidence_events']=[]
    with pytest.raises(ValueError,match='needs'):validate_procedure(procedure,[{'sequence':1}])


def test_memory_manifest_refuses_external_links(tmp_path):
    (tmp_path/'external').symlink_to('/etc/passwd')
    with pytest.raises(ValueError,match='symlinks'):store_hashes(tmp_path)


@pytest.mark.integration
def test_shared_library_is_readable_but_not_writable(tmp_path):
    import subprocess,shutil
    from vlm_tamp.subscription_agent import clean_environment,permission_arguments
    if not shutil.which('codex'):pytest.skip('Codex required')
    public=tmp_path/'session';public.mkdir();library=tmp_path/'library';library.mkdir()
    canary=library/'entry.json';canary.write_text('{"public":true}')
    code='''import pathlib,sys,json
p=pathlib.Path(sys.argv[1]);read=json.loads(p.read_text())
try:p.write_text('changed');writable=True
except OSError as e:
 assert e.errno in (1,13,30)
 writable=False
print(json.dumps({'read':read,'writable':writable}))
'''
    command=[shutil.which('codex'),'sandbox','-P','research-agent',
        *permission_arguments(readonly_paths=[library]),'-C',str(public),'/usr/bin/python3','-c',code,str(canary)]
    result=subprocess.run(command,env=clean_environment(),capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(result.stdout)=={'read':{'public':True},'writable':False}


@pytest.mark.integration
def test_staged_memory_real_retrieval_preflight(tmp_path):
    from vlm_tamp.episodic_memory import verify_memory_access
    library=tmp_path/'episodic_library';library.mkdir()
    (tmp_path/'memory.py').write_text(client_source(library))
    assert verify_memory_access(tmp_path)['retrieved_entries']==0
    e=library/'e1';e.mkdir();(e/'evidence').mkdir()
    (e/'entry.json').write_text(json.dumps({'id':'e1','task':'open the microwave','robot':'Panda'}))
    (e/'procedure.json').write_text('{}')
    assert verify_memory_access(tmp_path)['retrieved_entries']==1
    assert not (tmp_path/'memory-retrieval.jsonl').exists()
