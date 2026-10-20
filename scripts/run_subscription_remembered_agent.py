"""Start a one-attempt microwave agent with persistent episodic retrieval enabled.

This entry point retains the existing numerical backend. It loads only the
explicitly registered, hash-verified public library; it adds no motion policy.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/vlm_tamp').is_dir())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.episodic_memory import store_hashes
from run_subscription_dino_study import collect,write
from replay_subscription_episode import compress_observations

def command(args):
    profile=json.loads((ROOT/args.memory_profile).read_text())
    if profile.get('completion_checks') and profile.get('completion_protocol_version')!=2:
        raise RuntimeError('This completion profile requires its earlier source snapshot; use the current registered profile for the revised measurement interface')
    library=(ROOT/profile['public_library']).resolve()
    if hashlib.sha256(json.dumps(store_hashes(library),sort_keys=True).encode()).hexdigest()!=profile['library_manifest_sha256']:
        raise RuntimeError('Registered public memory changed; review and version it before use')
    for key in ['prompt','common_guide']:
        if hashlib.sha256((ROOT/profile[key]).read_bytes()).hexdigest()!=profile[key+'_sha256']:
            raise RuntimeError('Registered prompt or common guide changed')
    output=(ROOT/args.output).resolve()
    if output==ROOT/'results' or not output.is_relative_to(ROOT/'results'):
        raise ValueError('Use a new results/<run> output directory')
    return [sys.executable,str(ROOT/'scripts/run_subscription_agent_pilot.py'),
        '--output',str(output),'--state',str(args.state),'--scene-variant',args.scene_variant,
        '--memory-store',str(library),'--prompt-file',str(ROOT/profile['prompt']),
        '--guidelines-source',str(ROOT/profile['common_guide']),'--attempts','1',
        '--fit-diagnostics','--dino-tools','--agent-seconds',str(args.agent_seconds),
        *(['--completion-checks'] if profile.get('completion_checks') else []),
        *(['--abstract-experience'] if profile.get('abstract_experience') else [])],output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--state',required=True,type=int)
    p.add_argument('--scene-variant',default='nominal');p.add_argument('--agent-seconds',type=int,default=1800)
    p.add_argument('--memory-profile',default='configs/subscription_memory.json')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not 0<=a.state<50:raise ValueError('Official state must be 0..49')
    if a.agent_seconds<=0:raise ValueError('Use a positive session time budget')
    args,out=command(a)
    if a.dry_run:print(json.dumps({'command':args,'physical_attempts':1,'memory_enabled':True},indent=2));return
    if out.exists():raise ValueError('Use a new output directory')
    start=time.perf_counter();result=subprocess.run(args,cwd=ROOT)
    record=collect({'case_id':out.name,'state':a.state,'scene_variant':a.scene_variant,'condition':'with_memory'},out,result.returncode,time.perf_counter()-start)
    record['final_task_success']=bool(record['attempts'] and record['attempts'][0]['final_task_success'])
    record['lossless_compression']=compress_observations(out)
    write(out/'remembered-agent-result.json',record)
    print(json.dumps({'run':str(out),'native_goal_reached':record['first_attempt_success'],
        'native_goal_at_end':record['final_task_success'],'physical_attempts':record['physical_attempts_used'],
        'separately_billed_api_calls':record['api_calls']},indent=2))
    if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
