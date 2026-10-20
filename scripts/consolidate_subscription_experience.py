"""Two isolated Luna sessions consolidate and audit public experience only."""
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
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.subscription_agent import clean_environment,codex_arguments,verify_isolation
from run_subscription_agent_pilot import ledger_hashes

COMMON='''You are reviewing your robotics agent's past public experiences, not running
a robot. All available evidence is in corpus/. Read corpus/manifest.json first.
Only public RGB-D observations, your prior analysis code, reported findings and
guidelines, documented tools, and tool request/response traces are provided.
No native simulator scores, private evaluation, implementation source or human
debugging advice is supplied. Network and external model calls are unavailable.
Use this session's tools to inspect evidence and write/run analysis if needed.
Do not modify corpus/. A previous agent's explanation is a claim, not a fact.
Each case's public files are under corpus/<case>/workspace/, including
workspace/example/readme.txt, tools.txt, findings.txt and guidelines.json.
Use the integer sequence field inside each tool event as its event identifier.
Never substitute zero-based line indices. The first recorded sequence is 1.
Distinguish directly observed events, plausible hypotheses, and causal claims
that the records cannot establish. Respect per-attempt event context; equal
timestamps or step counters alone do not imply identical state or provenance.
Preserve attribution to the supplied human example and inherited numerical tools.
Do not inspect anything outside this workspace or ask a human for task hints.
Produce a compact guide transferable to changed scenes of the same task. Avoid
memorized pixel/world coordinates, fixed image-side choices, asset identifiers,
case-specific thresholds or exact action sequences as operational rules. Numerical
tool conventions may be described. Rules should express conditions, diagnostic
choices and evidence requirements, not guarantee a particular planner outcome.
Include actionable conditional hypotheses supported by experiences, with their
uncertainty and counterexamples. Lack of a controlled causal ablation does not
require discarding a useful heuristic; label its support level accurately.
Use at most eight rules and at most 1400 words in the entire guide. JSON format:
{"source": "...", "use": "...", "event_id_convention": "sequence", "guidelines": [{"id": "...",
"condition": "...", "guideline": "...", "origin": "...",
"evidence": [{"case": "corpus case id", "events": [integer event numbers],
"claim": "narrow observation supported by those events"}],
"limitations": "counterexamples, assumptions or uncertainty"}]}.
Evidence ids refer to corpus/<case>/tool-events.jsonl. Do not claim that a
case succeeded according to hidden evaluation or that a rule caused success.
'''
DRAFT=COMMON+'''\nReview all ten cases, including failed attempts and contrary evidence.
Write draft-guidelines.json, claim-review.json describing supported/unsupported
past claims and their evidence, and findings.txt. Do not copy a strategy just
because it appeared in a previous guideline. Finish with these saved artifacts.
'''
AUDIT=COMMON+'''\nA separate fresh session produced draft-guidelines.json and claim-review.json.
Independently audit their factual and procedural claims against corpus/. Inspect
the actual relevant tool events and, where useful, images or agent analysis.
Revise or remove unsupported claims, overly specific rules and unjustified causal
interpretations. Do not trust the draft's self-assigned confidence. Write the
final guidelines.json and audit-review.json with checked claims, evidence event
numbers, corrections and remaining limitations. Preserve the draft for comparison.
Do not introduce new operational claims without evidence. Finish when saved.
'''

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def validate_guide(path,corpus):
    guide=json.loads(path.read_text());rules=guide['guidelines']
    if guide.get('event_id_convention')!='sequence':raise ValueError('Use recorded sequence event IDs')
    if not isinstance(rules,list) or not 1<=len(rules)<=8:raise ValueError('Expected 1..8 rules')
    if len(json.dumps(guide).split())>1600:raise ValueError('Guide is too long')
    for r in rules:
        for key in ('id','condition','guideline','origin','evidence','limitations'):
            if key not in r:raise ValueError('Missing guide field: '+key)
        if not r['evidence']:raise ValueError('Rule without evidence')
        for e in r['evidence']:
            case=e['case'];folder=(corpus/case).resolve()
            if not folder.is_relative_to(corpus.resolve()):raise ValueError('Evidence path escape')
            events={x['sequence'] for x in map(json.loads,(folder/'tool-events.jsonl').read_text().splitlines())}
            if not e['events'] or any(type(n) is not int or n not in events for n in e['events']):
                raise ValueError('Missing cited event')
    return {'rules':len(rules),'evidence_references_exist':True,'semantic_review':'Luna audit-review.json; not an independent causal validation'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seconds',type=int,default=1200);a=p.parse_args()
    out=(ROOT/a.output).resolve();out.mkdir(exist_ok=False)
    source=ROOT/'results/subscription-dino-study-20261019'
    env=clean_environment();login=subprocess.run(['codex','login','status'],env=env,capture_output=True,text=True,check=True)
    if 'Logged in using ChatGPT' not in login.stdout+login.stderr:raise RuntimeError('Subscription login required')
    before=ledger_hashes();write(out/'api-ledger-checkpoint.json',before)
    work=Path(tempfile.mkdtemp(prefix='vlm-luna-consolidation-'));corpus=work/'corpus';corpus.mkdir()
    manifest={'source':'Public experiences from ten completed isolated agent sessions',
        'private_evaluation_included':False,'implementation_source_included':False,'cases':[]}
    for run in sorted((source/'runs').iterdir()):
        case=corpus/run.name;case.mkdir()
        shutil.copytree(run/'agent-workspace',case/'workspace')
        shutil.copyfile(run/'tool-events.jsonl',case/'tool-events.jsonl')
        manifest['cases'].append({'id':run.name,'workspace':f'{run.name}/workspace/',
            'example':f'{run.name}/workspace/example/readme.txt','tools':f'{run.name}/workspace/tools.txt',
            'events':f'{run.name}/tool-events.jsonl','event_id_field':'sequence'})
    write(corpus/'manifest.json',manifest)
    hashes={str(f.relative_to(corpus)):sha(f) for f in corpus.rglob('*') if f.is_file()}
    write(out/'public-corpus-hashes.json',hashes);write(out/'corpus-manifest.json',manifest)
    write(out/'protocol.json',{'model':'gpt-6-luna','reasoning':'high','authentication':'ChatGPT subscription',
        'phases':['draft','audit'],'seconds_per_phase':a.seconds,'workspace':str(work),'source':str(source),
        'no_private_evaluator_or_repository_access':True,'separate_api_calls_allowed':False})
    write(out/'isolation.json',verify_isolation(work,[ROOT/'src/vlm_tamp/rigid_hinge_tracking.py',
        source/'summary.json',Path('/mnt/c/Users/duo/.codex/memories/MEMORY.md'),Path('/home/duo/.codex/keys/vlm-tamp.env')]))
    completed=False
    try:
        for name,prompt in [('draft',DRAFT),('audit',AUDIT)]:
            args=codex_arguments(work,out/(name+'-last.txt'),readonly_paths=[corpus])
            (out/(name+'-prompt.txt')).write_text(prompt);write(out/(name+'-command.json'),args)
            start=time.time()
            with (out/(name+'-events.jsonl')).open('w') as log,(out/(name+'-stderr.log')).open('w') as err:
                proc=subprocess.Popen(args,stdin=subprocess.PIPE,text=True,stdout=log,stderr=err,env=env)
                write(out/(name+'-process.json'),{'pid':proc.pid,'started':start})
                try:
                    proc.communicate(prompt,timeout=a.seconds);code=proc.returncode
                except subprocess.TimeoutExpired:
                    proc.terminate();proc.wait(timeout=30);raise
                finally:write(out/(name+'-exit.json'),{'returncode':proc.poll(),'wall_seconds':time.time()-start})
            if code:raise RuntimeError(name+' session failed; no API fallback')
            required='draft-guidelines.json' if name=='draft' else 'guidelines.json'
            write(out/(name+'-validation.json'),validate_guide(work/required,corpus))
            if any(sha(corpus/f)!=h for f,h in hashes.items()):raise RuntimeError('Corpus changed')
            print(json.dumps({'phase':name,'complete':True}),flush=True)
        if ledger_hashes()!=before:raise RuntimeError('API ledger changed')
        completed=True
    finally:
        archive=out/'agent-workspace';archive.mkdir(exist_ok=True)
        for f in work.rglob('*'):
            rel=f.relative_to(work)
            if rel.parts[0]=='corpus' or '__pycache__' in rel.parts:continue
            if f.is_file() and not f.is_symlink():
                dst=archive/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dst)
        size=sum(p.stat().st_size for p in work.rglob('*') if p.is_file() and not p.is_symlink())
        shutil.rmtree(work)
        write(out/'completion.json',{'completed':completed,'temporary_corpus_bytes_removed':size,
            'original_public_evidence_retained_at':str(source),'separately_billed_api_calls':0,
            'api_ledgers_unchanged':ledger_hashes()==before,'temporary_workspace_removed':not work.exists()})

if __name__=='__main__':main()
