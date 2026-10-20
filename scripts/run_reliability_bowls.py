"""Success-focused bowl and microwave development/regression trials."""
import argparse,hashlib,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp import reliable_bowl_fallback as bowl
import reliable_microwave as microwave
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import ProfiledResources,compress_retained_observations
from vlm_tamp.profiled_sim_client import ProfiledClient,execute_batch
from run_all_prior_tasks import TASKS,configuration


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--states',nargs='+',type=int,required=True);p.add_argument('--development',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    tasks=[t for t in TASKS if t['label'] in a.tasks]
    if len(tasks)!=len(a.tasks) or any(t['label'] not in {'bowl_on_plate','open_microwave','close_microwave'} for t in tasks):raise ValueError('Unsupported task')
    out=(ROOT/a.output).resolve();config=configuration();config.update(maximum_candidate_grasps=100,maximum_total_ik_postures=800,maximum_grasp_search_seconds=600.)
    manifest=[{'task':t['label'],'state':i} for i in a.states for t in tasks]
    if a.resume:
        protocol=read_json(out/'protocol.json');sources=protocol['sources_sha256'];assert read_json(out/'manifest.json')['episodes']==manifest
    else:
        out.mkdir(parents=True,exist_ok=False)
        files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/reliable_microwave.py',ROOT/'scripts/run_all_prior_tasks.py',ROOT/'src/vlm_tamp/native/mcvamp_microwave.cc',ROOT/'src/vlm_tamp/native/mcvamp_contact_refined.cc']
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        for n in sources:
            dst=out/'source-snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
        write_json(out/'manifest.json',{'episodes':manifest,'maximum_control_steps':600,'settling_actions':5,'seed':0,'all_failures_in_denominator':True})
        write_json(out/'protocol.json',{'sources_sha256':sources,'configuration':config,'development':a.development,
            'observed_geometry_only':True,'execution_vetoes':False,'full_issued_trajectory_execution':True,
            'baseline':'results/all-prior-tasks-frozen-v3-20261015','fresh_grasp_selection':True,
            'api_before':api_snapshot(),'grounding':'validated cached Luna high','automatic_api_retries':0,
            'held_out_generalization':False,'microwave_open_gripper_endpoint_hold_steps':60,
            'microwave_planner_algorithm_unchanged':True,'native_build':read_json(ROOT/'.runtime/mcvamp-contact-refined/build.json'),'microwave_collision_geometry':'configured convex arm volumes; 6 mm bounded mount association tolerance','bowl_change':'original source-view proposals first; fused observed-body fallback; selected plate finger contact'})
    trace=Timeline();resources=ProfiledResources(out,trace,fresh=False);resources.warm_models();rows=[]
    for engine in [bowl,microwave]:
        engine.execute=execute_batch
        engine.CoupledClient=lambda *args,**kw:ProfiledClient(*args,**kw,fast=True,timeline=trace)
    for entry in manifest:
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in sources.items()),'Frozen sources changed'
        task=next(t for t in tasks if t['label']==entry['task']);folder=out/f'{task["label"]}-state{entry["state"]:03d}';trace.context=entry
        if (folder/'summary.json').exists():result=read_json(folder/'summary.json')
        else:
            print({'event':'started',**entry},flush=True);fn=bowl.run_episode if task['label']=='bowl_on_plate' else microwave.episode
            result=fn(task,entry['state'],folder,config,resources)
            resources.cleanup_proposals();compress_retained_observations(folder);result=read_json(folder/'summary.json')
        rows.append(result);trace.save(out/'timing.json');write_json(out/'summary.json',{'planned':len(manifest),'completed':len(rows),'native_successes':sum(r['official_task_success'] for r in rows),'episodes':rows,'api':api_snapshot()})
        print({'event':'completed',**entry,'success':result['official_task_success'],'failure':result.get('failure'),'steps':result['control_steps'],'seconds':result['episode_wall_seconds']},flush=True)
if __name__=='__main__':main()
