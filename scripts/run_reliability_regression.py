"""Fresh selection regression for sensor-derived contact and completion repairs."""
import argparse,hashlib,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp import reliable_mechanisms as engine
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import ProfiledResources,compress_retained_observations
from vlm_tamp.profiled_sim_client import ProfiledClient,execute_batch
from run_all_prior_tasks import TASKS,configuration


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--tasks',nargs='+',default=['close_top_drawer','turn_on_stove']);p.add_argument('--states',nargs='+',type=int,default=[1,11,21,31,41])
    p.add_argument('--development',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    tasks=[t for t in TASKS if t['label'] in a.tasks]
    if len(tasks)!=len(a.tasks) or any(t['adapter'] not in {'line','control rotation'} for t in tasks):raise ValueError('Unsupported task')
    out=(ROOT/a.output).resolve();config=configuration();config.update(version='success-focused-1',maximum_candidate_grasps=100,maximum_total_ik_postures=256)
    manifest=[{'task':t['label'],'state':s} for s in a.states for t in tasks]
    if a.resume:
        protocol=read_json(out/'protocol.json');sources=protocol['sources_sha256']
        assert read_json(out/'manifest.json')['episodes']==manifest
        assert config==protocol['configuration']
    else:
        out.mkdir(parents=True,exist_ok=False)
        unused={'reliable_handle.py','reliable_bowl.py','reliable_arc_scene.py'}
        files=[p for p in (ROOT/'src/vlm_tamp').rglob('*.py') if p.name not in unused]+[Path(__file__).resolve(),ROOT/'scripts/run_all_prior_tasks.py',ROOT/'src/vlm_tamp/native/mcvamp_mechanism.cc']
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        for n in sources:
            target=out/'source-snapshot'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
        write_json(out/'manifest.json',{'episodes':manifest,'maximum_control_steps':600,'settling_actions':5,'seed':0,'all_failures_in_denominator':True})
        write_json(out/'protocol.json',{'sources_sha256':sources,'configuration':config,'development':a.development,
            'observed_geometry_only':True,'execution_vetoes':False,'full_issued_trajectory_execution':True,
            'baseline':'results/mcvamp-drawer-stove-report/evaluation.json','fresh_grasp_selection':True,
            'api_before':api_snapshot(),'grounding':'observation-validated cached Luna high','automatic_api_retries':0,
            'held_out_generalization':False,'speed_optimization_experiment':False})
    timeline=Timeline();resources=ProfiledResources(out,timeline,fresh=False);resources.warm_models();rows=[]
    engine.execute=execute_batch
    engine.CoupledClient=lambda *args,**kw:ProfiledClient(*args,**kw,fast=True,timeline=timeline)
    for entry in manifest:
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in sources.items()),'Frozen sources changed'
        task=next(t for t in tasks if t['label']==entry['task']);folder=out/f'{task["label"]}-state{entry["state"]:03d}'
        timeline.context=entry
        if (folder/'summary.json').exists():result=read_json(folder/'summary.json')
        else:
            print({'event':'started',**entry},flush=True)
            result=engine.run_episode(task,entry['state'],folder,config,resources)
            compress_retained_observations(folder);result=read_json(folder/'summary.json')
        rows.append(result);timeline.save(out/'timing.json')
        write_json(out/'summary.json',{'planned':len(manifest),'completed':len(rows),'native_successes':sum(r['official_task_success'] for r in rows),'episodes':rows,'api':api_snapshot()})
        print({'event':'completed',**entry,'success':result['official_task_success'],'failure':result.get('failure'),'steps':result['control_steps'],'seconds':result['episode_wall_seconds']},flush=True)
if __name__=='__main__':main()
