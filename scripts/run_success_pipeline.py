"""Unified success-focused runner for the seven previously evaluated LIBERO tasks."""
import argparse,hashlib,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp import reliable_mechanisms as mechanisms,reliable_bowl_fallback as bowl
import reliable_microwave_final as microwave
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import ProfiledResources,compress_retained_observations
from vlm_tamp.profiled_sim_client import ProfiledClient,execute_batch
from run_all_prior_tasks import TASKS,configuration


class SceneBoundaryClient(ProfiledClient):
    def observe(self,*args,**kwargs):
        result=super().observe(*args,**kwargs)
        # These engines observe only between completed planning/execution calls.
        # Old CAPT trees cannot help a changed depth scene and otherwise grow
        # across every short segment. Keep reuse within each planning stage.
        from vlm_tamp.mcvamp_mechanism import clear_native_cache as clear_mechanisms
        from vlm_tamp.mcvamp_contact_numerical import clear_native_cache as clear_microwave
        clear_mechanisms();clear_microwave()
        return result


def task_configuration(task):
    config=configuration()
    if task['label']=='bowl_on_plate':
        config.update(maximum_candidate_grasps=100,maximum_total_ik_postures=800,maximum_grasp_search_seconds=600.)
        return bowl.run_episode,config,'source-view GraspGen, fused-view fallback, local placement contacts'
    if task['adapter']=='MC-VAMP hinge':
        config.update(microwave_approach_speed_rad_s=.30,minimum_manipulation_steps_reserved=200,closing_seating_trigger_deg=3.,closing_seating_allowance_deg=2.,
            maximum_candidate_grasps=100,maximum_total_ik_postures=800,maximum_grasp_search_seconds=600.)
        return microwave.episode,config,'MC-VAMP hinge; calibrated arm; budget-feasible IK with joint reserve; complete slower approach'
    config.update(version='success-focused-1',maximum_candidate_grasps=100,maximum_total_ik_postures=256)
    return mechanisms.run_episode,config,'MC-VAMP line/control; observed face ownership and measured targets'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output')
    parser.add_argument('--tasks',nargs='+',choices=[t['label'] for t in TASKS]);parser.add_argument('--states',nargs='+',type=int,default=[1,11,21,31,41])
    parser.add_argument('--resume',action='store_true');parser.add_argument('--describe',action='store_true');a=parser.parse_args()
    tasks=[t for t in TASKS if not a.tasks or t['label'] in a.tasks]
    if len(set(a.states))!=len(a.states) or any(i not in {1,11,21,31,41} for i in a.states):raise ValueError('Choose unique official regression states 1,11,21,31,41')
    configs={t['label']:task_configuration(t)[1] for t in tasks}
    if a.describe:
        print({'tasks':[{'task':t['label'],'pipeline':task_configuration(t)[2]} for t in tasks],
            'states':a.states,'planned_episodes':len(tasks)*len(a.states),'maximum_control_steps':600,
            'execution_vetoes':False,'grounding':'RGB-validated cache; shared $2 ledger for any needed request','model':'gpt-6-luna','reasoning':'high'});return
    if not a.output:parser.error('--output is required unless --describe is used')
    out=(ROOT/a.output).resolve();planned=[{'task':t['label'],'state':i} for i in a.states for t in tasks]
    if a.resume:
        protocol=read_json(out/'protocol.json');sources=protocol['sources_sha256'];assert protocol['task_configurations']==configs
        assert read_json(out/'manifest.json')['episodes']==planned
    else:
        out.mkdir(parents=True,exist_ok=False)
        files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[ROOT/n for n in ['scripts/reliable_microwave_final.py','scripts/run_all_prior_tasks.py','scripts/try_rekep_bowl_v3.py','scripts/run_coupled_door.py','src/vlm_tamp/native/mcvamp_mechanism.cc','src/vlm_tamp/native/mcvamp_contact_numerical.cc','configs/articulated_libero.json']]+[Path(__file__).resolve()]
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        for n in sources:
            destination=out/'source-snapshot'/n;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,destination)
        write_json(out/'manifest.json',{'episodes':planned,'all_failures_in_denominator':True,'maximum_control_steps':600,'seed':0,'settling_actions':5})
        write_json(out/'protocol.json',{'sources_sha256':sources,'task_configurations':configs,'api_before':api_snapshot(),
            'native_builds':{n:read_json(ROOT/'.runtime'/n/'build.json') for n in ['mcvamp-mechanism','mcvamp-contact-numerical']},
            'frozen_before_first_episode':True,'held_out_generalization':False,'fixed_grasp_replay':False,
            'physical_gripper_attachment_only':True,'execution_vetoes':False,'all_issued_references_completed':True,
            'simulator_object_geometry_or_joint_state_used_for_policy':False,'native_success_is_evaluation_only':True,'obsolete_native_scene_indices_released_at_observation_boundaries':True})
    trace=Timeline();resources=ProfiledResources(out,trace,fresh=False);resources.warm_models();rows=[]
    for engine in [mechanisms,bowl,microwave]:
        engine.execute=execute_batch
        engine.CoupledClient=lambda *args,**kw:SceneBoundaryClient(*args,**kw,fast=True,timeline=trace)
    for entry in planned:
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in sources.items()),'Frozen sources changed'
        task=next(t for t in tasks if t['label']==entry['task']);folder=out/f'{task["label"]}-state{entry["state"]:03d}';trace.context=entry
        if (folder/'summary.json').exists():result=read_json(folder/'summary.json')
        else:
            print({'event':'started',**entry},flush=True);fn,config,_=task_configuration(task)
            result=fn(task,entry['state'],folder,config,resources)
            resources.cleanup_proposals();compress_retained_observations(folder);result=read_json(folder/'summary.json')
        rows.append(result);trace.save(out/'timing.json')
        write_json(out/'summary.json',{'planned':len(planned),'completed':len(rows),'native_successes':sum(r['official_task_success'] for r in rows),'episodes':rows,'api':api_snapshot()})
        print({'event':'completed',**entry,'success':result['official_task_success'],'failure':result.get('failure'),'steps':result['control_steps']},flush=True)
if __name__=='__main__':main()
