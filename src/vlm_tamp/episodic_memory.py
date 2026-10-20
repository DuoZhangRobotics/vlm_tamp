"""Public experience storage and retrieval, without embedded manipulation policy."""
import hashlib
import json
from pathlib import Path


def store_hashes(root):
    root=Path(root).resolve()
    result={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Memory must not contain symlinks')
        if p.is_file():result[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def validate_procedure(procedure, events):
    """Check provenance links and completeness, not the truth of learned claims."""
    required=['task','robot','observed_failure','successful_correction','applicability',
              'procedure','do_not_repeat','limitations','attribution']
    for key in required:
        if key not in procedure:raise ValueError('Missing procedure field: '+key)
    known={e['sequence'] for e in events}
    steps=procedure['procedure']
    if not isinstance(steps,list) or not steps:raise ValueError('Empty procedure')
    for step in steps:
        if not step.get('instruction') or not step.get('evidence_events'):
            raise ValueError('Procedure step needs an instruction and evidence')
        if any(type(n) is not int or n not in known for n in step['evidence_events']):
            raise ValueError('Unknown evidence event')
    return {'event_references_valid':True,'steps':len(steps),
            'semantic_validation':'Agent interpretation; transfer must be measured separately'}


def client_source(store):
    """Standalone client visible to Luna; indexes only supplied public episodes."""
    return '''import json, re, time
from pathlib import Path
STORE=Path('''+repr(str(Path(store).resolve()))+''')
def recall(instruction, robot="Panda", limit=3):
    query=set(re.findall(r"[a-z0-9]+",str(instruction).lower()))
    matches=[]
    for path in sorted(STORE.glob("*/entry.json")):
        row=json.loads(path.read_text())
        if str(row.get("robot","")).lower()!=str(robot).lower(): continue
        words=set(re.findall(r"[a-z0-9]+",row.get("task"," ").lower()))
        score=len(query & words)/max(1,len(query | words))
        if score<=0: continue
        matches.append({"id":row["id"],"score":score,"task":row["task"],
          "entry":str(path),"procedure":str(path.parent/"procedure.json"),
          "evidence":str(path.parent/"evidence"),"support":row.get("support")})
    matches=sorted(matches,key=lambda x:(-x["score"],x["id"]))[:max(0,min(10,int(limit)))]
    with (Path(__file__).resolve().parent/"memory-retrieval.jsonl").open("a") as out:
        out.write(json.dumps({"time":time.time(),"query":instruction,"robot":robot,"results":matches})+"\\n")
    return matches
if __name__=="__main__":
    import sys
    print(json.dumps(recall(sys.argv[1]),indent=2))
'''


def verify_memory_access(workspace):
    """Exercise actual retrieval and read-only behavior before any simulator reset."""
    import shutil,subprocess
    from .subscription_agent import permission_arguments,clean_environment
    work=Path(workspace).resolve();library=work/'episodic_library'
    expected=len(list(library.glob('*/entry.json')))
    code='''import json,pathlib,errno
from memory import recall,STORE
result=recall("open the microwave",robot="Panda")
try:
 (STORE/"write-probe.tmp").write_text("public probe")
 writable=True
except OSError as error:
 if error.errno not in (errno.EPERM,errno.EACCES,errno.EROFS):raise
 writable=False
for entry in result:
 json.loads(pathlib.Path(entry["procedure"]).read_text())
 list(pathlib.Path(entry["evidence"]).iterdir())
print(json.dumps({"retrieved_entries":len(result),"read_only":not writable}))
'''
    command=[shutil.which('codex') or 'codex','sandbox','-P','research-agent',
        *permission_arguments(readonly_paths=[library]),'-C',str(work),'/usr/bin/python3','-c',code]
    try:
        run=subprocess.run(command,env=clean_environment(),capture_output=True,text=True,check=True,timeout=30)
        result=json.loads(run.stdout)
        if result!={'retrieved_entries':expected,'read_only':True}:
            raise RuntimeError('Memory preflight failed: '+str(result))
        return {**result,'passed':True,'before_any_physical_reset':True}
    finally:
        (work/'memory-retrieval.jsonl').unlink(missing_ok=True)
        (library/'write-probe.tmp').unlink(missing_ok=True)
