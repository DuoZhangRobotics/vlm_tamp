"""Let Luna preserve a concrete failed/successful experience as a reusable procedure."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.subscription_agent import clean_environment,codex_arguments,verify_isolation
from vlm_tamp.episodic_memory import validate_procedure,store_hashes
from run_subscription_agent_pilot import ledger_hashes

PROMPT='''Create persistent episodic memory from the robot agent's recorded experience.
This is offline review, not robot execution. Read corpus/tool-events.jsonl and
corpus/workspace/findings.txt; inspect the relevant images, recorded plans,
measurements and analysis code under corpus/workspace/. Only public sensor data
and prior agent outputs are available. Prior explanations are claims to check.
There is no native evaluator, project implementation, or other history here.

Preserve the concrete failed attempt AND the corrected attempt. This memory must
help a fresh agent reuse a useful procedure instead of rediscovering a known
failure. Exact old tool arguments and recorded plans are legitimate episode
evidence; label them with their observation/frame context. Explain how to check
whether the current scene matches, and which quantities must be measured again
when it does not. Do not treat old coordinates, plan ids, or image-side labels
as universal rules. Include tool sequence, conditional branches, what not to
repeat, remaining uncertainties, and what was supplied by the human example
or inherited tools. Do not invent missing evidence or causal guarantees.

Write procedure.json with fields: task (text), robot (Panda), observed_failure,
successful_correction (observed evidence, not hidden evaluation), applicability,
procedure (list of {instruction, evidence_events: [integer sequence ids]}),
do_not_repeat, limitations, attribution. Additional fields are allowed. Cite
the actual sequence field in tool-events.jsonl, not line indices. Write a useful
operational memory, not just general advice. You may additionally create a
reusable procedure.py if helpful, but its decisions must come from the recorded
experience and it must obtain fresh plans/observations via the documented tools.
Write findings.txt explaining which evidence you inspected and any uncertainty.
Do not edit corpus or access files outside this public workspace. No additional
model API calls or network access. Finish when the memory artifacts are saved.
'''

def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--audit-existing',help='Independently audit this public procedure without human rewriting')
    a=p.parse_args()
    source=Path(a.source).resolve();out=Path(a.output).resolve();out.mkdir(exist_ok=False)
    work=Path(tempfile.mkdtemp(prefix='vlm-episodic-author-'));corpus=work/'corpus';corpus.mkdir()
    shutil.copytree(source/'agent-workspace',corpus/'workspace')
    shutil.copyfile(source/'tool-events.jsonl',corpus/'tool-events.jsonl')
    prompt=PROMPT
    if a.audit_existing:
        shutil.copyfile(a.audit_existing,work/'draft-procedure.json')
        prompt+='''\nA previous fresh agent wrote draft-procedure.json. Independently audit it
against corpus, with particular care about who made each decision: a human, the
acting agent, or an inherited tool. Correct inaccurate attribution and factual
claims, preserve the useful concrete procedure where the evidence supports it,
and do not invent new operational strategies. Write the reviewed procedure.json
and an audit-review.json with checked claims, changes and unresolved uncertainty.
The draft must remain available for comparison.\n'''
    hashes=store_hashes(corpus);write(out/'public-source-hashes.json',hashes)
    before=ledger_hashes();write(out/'api-ledger-checkpoint.json',before)
    env=clean_environment();status=subprocess.run(['codex','login','status'],env=env,text=True,capture_output=True,check=True)
    if 'Logged in using ChatGPT' not in status.stdout+status.stderr:raise RuntimeError('ChatGPT authentication required')
    write(out/'isolation.json',verify_isolation(work,[ROOT/'src/vlm_tamp/coupled_pipeline.py',source/'evaluation-private.json',
        Path('/mnt/c/Users/duo/.codex/memories/MEMORY.md')],readonly_paths=[corpus]))
    (out/'prompt.txt').write_text(prompt);args=codex_arguments(work,out/'agent-last.txt',readonly_paths=[corpus])
    write(out/'command.json',args);code=None;started=time.time();completed=False
    try:
        with (out/'agent-events.jsonl').open('w') as log,(out/'agent-stderr.log').open('w') as err:
            proc=subprocess.Popen(args,stdin=subprocess.PIPE,text=True,stdout=log,stderr=err,env=env)
            write(out/'process.json',{'pid':proc.pid,'workspace':str(work),'started':started})
            try:proc.communicate(prompt,timeout=a.seconds);code=proc.returncode
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=30);raise
        if code:raise RuntimeError('Memory-author session failed; no automatic API/model fallback')
        events=[json.loads(x) for x in (corpus/'tool-events.jsonl').read_text().splitlines()]
        procedure=json.loads((work/'procedure.json').read_text())
        write(out/'validation.json',validate_procedure(procedure,events))
        if store_hashes(corpus)!=hashes:raise RuntimeError('Public source changed')
        entry=out/'library'/'episode-0001';entry.mkdir(parents=True)
        shutil.copytree(corpus,entry/'evidence')
        for name in ['procedure.json','procedure.py','findings.txt','audit-review.json']:
            if (work/name).exists():shutil.copyfile(work/name,entry/name)
        write(entry/'entry.json',{'id':'episode-0001','task':procedure['task'],'robot':procedure['robot'],
            'support':'One recorded two-attempt episode, with observations and agent claims; applicability needs checking',
            'private_evaluation_included':False,'human_authored_operational_rule_added':False,
            'evidence_path_convention':'References to corpus/ in the authored artifacts resolve to evidence/ next to this entry. workspace/ paths are relative to evidence/.'})
        write(out/'library-hashes.json',store_hashes(out/'library'))
        completed=True
    finally:
        archive=out/'agent-workspace';archive.mkdir(exist_ok=True)
        for path in work.iterdir():
            if path.is_file() and not path.is_symlink():shutil.copyfile(path,archive/path.name)
        removed=sum(f.stat().st_size for f in work.rglob('*') if f.is_file())
        shutil.rmtree(work)
        write(out/'completion.json',{'completed':completed,'returncode':code,'wall_seconds':time.time()-started,
            'temporary_bytes_removed':removed,'api_ledgers_unchanged':ledger_hashes()==before,
            'separately_billed_api_calls':0,'temporary_workspace_removed':not work.exists()})
    print(json.dumps({'completed':completed,'library':str(out/'library')}),flush=True)

if __name__=='__main__':main()
