"""Freeze tested sources and run the four unseen states per task without retuning."""
import argparse,hashlib,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from try_rekep_bowl_v3 import save
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.coupled_storage import compact_episode,write_json


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--development-runs',nargs=2,required=True)
    parser.add_argument('--parallel',type=int,choices=[1,2],default=2)
    args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    references=[ROOT/p for p in args.development_runs]
    protocols=[read(p/'protocol.json') for p in references]
    hashes=protocols[0]['sources_sha256']
    if any(p['sources_sha256']!=hashes or p['config']!=protocols[0]['config'] for p in protocols):
        raise ValueError('Development runs must use the same frozen implementation and configuration')
    if any(digest(ROOT/name)!=value for name,value in hashes.items()):
        raise ValueError('Current source differs from the tested development implementation')
    cases=[]
    for root in references:
        manifest=read(root/'manifest.json')
        if len(manifest['planned_episodes'])!=1 or manifest['planned_episodes'][0]['state']!=1:
            raise ValueError('Provide one completed state-1 run per task')
        item=manifest['planned_episodes'][0]
        if read(root/'summary.json')['completed']!=1:raise ValueError('A development run is incomplete')
        cases.append({'task':item['task'],'state':1,'split':'development',
            'run':str(root.relative_to(ROOT)),'reused_development_measurement':True})
    tasks=[c['task'] for c in cases]
    if set(tasks)!={'open_microwave','close_microwave'}:raise ValueError('Both microwave tasks are required')
    for state in [11,21,31,41]:
        for task in tasks:
            cases.append({'task':task,'state':state,'split':'held_out',
                'run':str((out/'runs'/f'{task}-state{state:03d}').relative_to(ROOT)),
                'reused_development_measurement':False})
    for name in hashes:
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    save(out/'protocol.json',{'sources_sha256':hashes,'config':protocols[0]['config'],
        'frozen_before_unseen_observations':True,'seed':0,'maximum_control_steps':600,
        'states':[1,11,21,31,41],'development_states':[1],'unseen_states':[11,21,31,41],
        'automatic_api_retries':0,'api_budget_usd':2.,'api_snapshot_before':api_snapshot(),
        'simulator_scene_geometry_used_for_policy':False,'per_state_retuning_permitted':False})
    save(out/'manifest.json',{'planned_episodes':cases,'all_failures_in_denominator':True,
        'development_measurements_reused_with_identical_sources':True})
    queue=[c for c in cases if c['split']=='held_out'];active=[];done={}
    def verify_sources():
        if any(digest(ROOT/n)!=h for n,h in hashes.items()):raise ValueError('Frozen evaluation source changed')
    def collect(case):
        run=ROOT/case['run'];entry=run/f'{case["task"]}-state{case["state"]:03d}'/'summary.json'
        return {**case,**read(entry)}
    for case in cases:
        if case['split']=='development':done[(case['task'],case['state'])]=collect(case)
    def checkpoint():
        rows=[done[(c['task'],c['state'])] for c in cases if (c['task'],c['state']) in done]
        ledger=api_snapshot()
        save(out/'summary.json',{'planned':len(cases),'completed':len(rows),'episodes':rows,
            'native_successes':sum(bool(r.get('official_task_success')) for r in rows),
            'held_out_planned':8,'held_out_completed':sum(r['split']=='held_out' for r in rows),
            'held_out_successes':sum(r['split']=='held_out' and bool(r.get('official_task_success')) for r in rows),
            'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],
            'remaining_api_budget_usd':ledger['remaining_usd'],'frozen_sources_match':True})
    checkpoint()
    environment=os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while queue or active:
        verify_sources()
        while queue and len(active)<args.parallel:
            case=queue.pop(0);log=out/f'{case["task"]}-state{case["state"]:03d}.log';stream=log.open('w')
            command=[sys.executable,str(ROOT/'scripts/run_coupled_door.py'),'--tasks',case['task'],
                '--states',str(case['state']),'--output',case['run'],'--frozen-protocol',str(out/'protocol.json')]
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((case,process,stream,time.monotonic()))
        for item in active[:]:
            case,process,stream,start=item
            if time.monotonic()-start>3600 and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
            if process.poll() is None:continue
            stream.close();run=ROOT/case['run'];episode=run/f'{case["task"]}-state{case["state"]:03d}'
            if not (episode/'summary.json').exists():
                episode.mkdir(parents=True,exist_ok=True)
                actions=read(episode/'initial/actions.json') if (episode/'initial/actions.json').exists() else []
                row={'task':case['task'],'initial_state':case['state'],'split':'held_out','development_only':False,
                    'status':'failed','failure_stage':'runner_process','failure':f'Runner exited with status {process.returncode}',
                    'control_steps':len(actions),'official_task_success':any(a['evaluation_task_success'] for a in actions),
                    'final_task_success':bool(actions and actions[-1]['evaluation_task_success']),'new_api_calls':0,
                    'final_state_capture_confirmed':False}
                save(episode/'summary.json',row)
            if not (run/'summary.json').exists():
                row=read(episode/'summary.json');save(run/'summary.json',{'planned':1,'completed':1,'episodes':[row]})
            if not (episode/'retention.json').exists():
                row=read(episode/'summary.json');row['storage']=compact_episode(episode)
                write_json(episode/'summary.json',row)
            done[(case['task'],case['state'])]=collect(case);active.remove(item)
            checkpoint();print(json.dumps(done[(case['task'],case['state'])]),flush=True)
        if active:time.sleep(1)
    verify_sources();checkpoint()
    save(out/'report-runs.json',{'runs':[c['run'] for c in cases]})


if __name__=='__main__':main()
