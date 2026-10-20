"""Use isolated Luna to remove scene-specific numbers without adding strategies."""
import argparse, hashlib, json, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.subscription_agent import clean_environment,codex_arguments,verify_isolation
from vlm_tamp.episodic_memory import store_hashes,validate_procedure
from run_subscription_agent_pilot import ledger_hashes

PROMPT='''You are editing the prior agent's existing learned memory, not inventing
new robotics rules. Read source/procedure.json, source/guide.json and the prior
categorical event trace. Produce a coordinate-free procedure.json with the same
required fields and provenance event ids. Preserve only supported conditional
decisions, applicability, failures, corrections and limitations already present.
Remove ALL historical pixel boxes, metric coordinates, poses, joint paths, plan
ids, numeric action schedules, fit statistics and exact tool argument values.
Do not merely spell numbers out. Do not add new numerical thresholds, strategies,
completion criteria, task exceptions, or information from other experiments.
Keep qualitative relationships, such as checking the cabinet-connected edge
opposite the observed handle. Do not turn a historical screen side into a rule.
Use tool names without historical parameter values; fresh values must come from
the current observation and documented tool interface. The old coordinates are
not being made accessible to the future acting agent by any other memory files.

Also write guide.json: the supplied common guide, with case-specific numerical
examples removed, preserving its existing method and conditional advice. Both
experimental conditions will receive this same guide. This is an editorial
abstraction only, not an opportunity to improve its strategy.

All narrative strings in these outputs must contain no digits. The only numeric
values permitted in procedure.json are integer evidence_events ids referring to
the supplied categorical trace. Do not include executable code, source paths,
raw images, observation identifiers or links to omitted files. Write review.txt
explaining what was removed and which limitations remain. The supplied input
directory is read-only. No network/model calls, repository source, private
evaluation or other session history is available. Use this Luna/high session.
'''

def write(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
def validate_abstract(o,path=()):
    if isinstance(o,str):
        if re.search(r'\d',o):raise ValueError('Numeric text remains at '+'.'.join(path))
        if any(x in o for x in ['corpus/','workspace/','.npz','.npy','/home/','/tmp/']):raise ValueError('Raw evidence reference remains')
    elif isinstance(o,dict):
        for k,v in o.items():validate_abstract(v,path+(k,))
    elif isinstance(o,list):
        for v in o:validate_abstract(v,path)
    elif isinstance(o,(int,float)) and not isinstance(o,bool):
        if not path or path[-1]!='evidence_events' or type(o) is not int:raise ValueError('Numeric payload outside provenance')
    elif o is not None and not isinstance(o,bool):raise ValueError('Unsupported abstraction type')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--review-existing');a=ap.parse_args()
    out=(ROOT/a.output).resolve();out.mkdir(exist_ok=False)
    prior=ROOT/'data/agent_memory/luna-microwave-v1'
    source=prior/'library/episode-0001'
    events=[json.loads(x) for x in (source/'evidence/tool-events.jsonl').read_text().splitlines()]
    categorical=[{'sequence':e['sequence'],'operation':e['request']['operation'],'tool_ok':e['response']['ok'],
        'plan_available':e['response'].get('result',{}).get('planned'),
        'record_scope':'Historical tool outcome only; scene-specific arguments and arrays omitted.'} for e in events]
    work=Path(tempfile.mkdtemp(prefix='vlm-memory-abstraction-'));inputs=work/'source';inputs.mkdir()
    shutil.copyfile(source/'procedure.json',inputs/'procedure.json')
    shutil.copyfile(prior/'guidelines.json',inputs/'guide.json')
    write(inputs/'events.json',categorical)
    prompt=PROMPT
    if a.review_existing:
        draft=Path(a.review_existing)
        shutil.copyfile(draft/'library/episode-0001/procedure.json',inputs/'draft-procedure.json')
        shutil.copyfile(draft/'guidelines.json',inputs/'draft-guide.json')
        prompt+='''\nAudit the draft abstraction for fidelity to the original procedure and guide.
Preserve the original action ordering and attribution without inventing changes.
In particular, check whether the draft unambiguously preserves open fingers
during the approach and subsequent physical closure, and whether attribution
still identifies the actual human-supplied method. Correct editorial losses or
ambiguities using only the original source. Preserve all coordinate redaction.
Write the final reviewed procedure.json, guide.json and review.txt.\n'''
    before=store_hashes(inputs);ledgers=ledger_hashes();env=clean_environment()
    auth=subprocess.run(['codex','login','status'],env=env,text=True,capture_output=True,check=True)
    if 'Logged in using ChatGPT' not in auth.stdout+auth.stderr:raise RuntimeError('Subscription login required')
    write(out/'isolation.json',verify_isolation(work,[ROOT/'src/vlm_tamp/rigid_hinge_tracking.py',
        ROOT/'results/subscription-memory-challenge-20261019/summary.json',
        Path('/mnt/c/Users/duo/.codex/memories/MEMORY.md')],readonly_paths=[inputs]))
    args=codex_arguments(work,out/'agent-last.txt',readonly_paths=[inputs]);write(out/'command.json',args)
    (out/'prompt.txt').write_text(prompt);started=time.time();completed=False
    try:
        with (out/'agent-events.jsonl').open('w') as log,(out/'agent-stderr.log').open('w') as err:
            proc=subprocess.Popen(args,stdin=subprocess.PIPE,text=True,env=env,stdout=log,stderr=err)
            write(out/'process.json',{'pid':proc.pid,'workspace':str(work)})
            try:proc.communicate(prompt,timeout=1200)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=30);raise
        if proc.returncode:raise RuntimeError('Author failed; no automatic retry or fallback')
        procedure=json.loads((work/'procedure.json').read_text());guide=json.loads((work/'guide.json').read_text())
        validate_abstract(procedure);validate_abstract(guide);validation=validate_procedure(procedure,events)
        if store_hashes(inputs)!=before:raise RuntimeError('Read-only inputs changed')
        entry=out/'library/episode-0001';(entry/'evidence').mkdir(parents=True)
        write(entry/'procedure.json',procedure);write(out/'guidelines.json',guide)
        write(entry/'evidence/events.json',categorical)
        write(entry/'entry.json',{'id':'episode-0001','task':procedure['task'],'robot':procedure['robot'],
            'support':'Abstraction of one prior failed and corrected episode; observations were not native evaluation.',
            'representation':'Conditional prose with categorical event provenance; no historical scene geometry or executable paths.',
            'raw_episode_accessible':False,'private_evaluation_included':False})
        write(out/'validation.json',validation)
        write(out/'source-provenance.json',{'source_library_hashes':store_hashes(prior/'library'),
            'new_library_hashes':store_hashes(out/'library'),'source_unchanged':store_hashes(inputs)==before,
            'no_new_task_rules_authorized':True,'author_model':'gpt-6-luna','reasoning':'high'})
        completed=True
    finally:
        for name in ['procedure.json','guide.json','review.txt']:
            if (work/name).exists():shutil.copyfile(work/name,out/('author-'+name))
        shutil.rmtree(work)
        write(out/'completion.json',{'completed':completed,'seconds':time.time()-started,
            'api_ledgers_unchanged':ledgers==ledger_hashes(),'separately_billed_api_calls':0,
            'temporary_workspace_removed':not work.exists()})
    print(json.dumps({'completed':completed,'output':str(out)}))

if __name__=='__main__':main()
