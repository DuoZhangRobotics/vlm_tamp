"""Cross-family regression with native success termination and optional full motion.

Uses established RGB-grounding and mechanism adapters. This is not evidence that
the microwave acting agent or its learned memory transfers to other families.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from run_success_pipeline import task_configuration,SceneBoundaryClient
from run_all_prior_tasks import TASKS
from vlm_tamp import reliable_mechanisms as mechanisms,reliable_bowl_fallback as bowl
import reliable_microwave_final as microwave
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_pipeline import ProfiledResources,compress_retained_observations
from vlm_tamp.profiled_sim_client import execute_batch
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.run_api_budget import api_run_scope,ledger_for_output
from vlm_tamp.coupled_resources import api_snapshot


class NativeSuccess(Exception):
    pass


class PairedResources(ProfiledResources):
    """Reuse only the exact predictor output for an identical current cloud."""
    source_case=None

    def grasps(self,cloud,output):
        if self.source_case is None:return super().grasps(cloud,output)
        import yaml
        from vlm_tamp.coupled_resources import sha
        config=ROOT/'.runtime/graspgen-models/checkpoints/graspgen_franka_panda.yml'
        raw=yaml.safe_load(config.read_text());pins={'config_sha256':sha(config)}
        for key in ['eval','discriminator']:
            checkpoint=Path(raw[key]['checkpoint'])
            if not checkpoint.is_absolute():checkpoint=config.parent/checkpoint
            pins[key+'_checkpoint_sha256']=sha(checkpoint)
        xyz=np.asarray(cloud,np.float32)
        for file in self.source_case.rglob('input-provenance.json'):
            folder=file.parent
            if not (folder/'input-cloud.npy').exists() or not (folder/'grasp_candidates.npz').exists():continue
            provenance=read_json(file)
            if provenance.get('checkpoint_pins')!=pins or provenance.get('grasp_seed',0)!=getattr(self,'grasp_seed',0):continue
            if not np.array_equal(np.load(folder/'input-cloud.npy'),xyz):continue
            shutil.copytree(folder,output)
            write_json(Path(output)/'paired-proposal-provenance.json',{
                'source':str(folder),'identical_current_cloud':True,'checkpoint_pins_match':True,
                'grasp_or_trajectory_replayed':False,
                'candidate_arrays_sha256':sha(folder/'grasp_candidates.npz')})
            with np.load(Path(output)/'grasp_candidates.npz') as arrays:
                return {k:arrays[k].copy() for k in ['poses_world_base','scores','retained_indices']}
        raise ValueError('No exactly matching proposal input in the paired source; no silent regeneration')


class SuccessClient(SceneBoundaryClient):
    def __init__(self,log_path,*args,**kwargs):
        super().__init__(log_path,*args,**kwargs)
        self.folder=Path(log_path).parent;self.segments=0

    def request(self,operation,**kwargs):
        if operation=='reset':kwargs['stop_on_success']=True
        return super().request(operation,**kwargs)


def execute_to_success(sim,state,references,*,gripper=1,event_sink=None,**kwargs):
    folder=sim.folder/'execution-segments'/f'{sim.segments:03d}'
    folder.mkdir(parents=True);sim.segments+=1
    np.save(folder/'requested.npy',references)
    state,result=execute_batch(sim,state,references,gripper=gripper,event_sink=event_sink,stop_on_success=True,**kwargs)
    np.save(folder/'issued.npy',np.asarray(references)[:result['issued_commands']])
    write_json(folder/'result.json',{**result,'gripper':gripper})
    if result.get('terminated_on_simulator_success'):
        sim.observe(sim.folder/'final')
        write_json(sim.folder/'success-termination.json',{'reason':'simulator_success',
            'control_step':state['control_steps'],'post_success_motion':False})
        # Legacy engines catch this and finalize their evidence. The experiment
        # wrapper distinguishes this terminal event from a genuine failure.
        raise NativeSuccess('simulator_success')
    return state,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--tasks',nargs='+',choices=[t['label'] for t in TASKS],
        default=['bowl_on_plate','open_middle_drawer','open_top_drawer','close_top_drawer','turn_on_stove'])
    p.add_argument('--states',nargs='+',type=int,default=[1,11,21,31,41])
    p.add_argument('--horizon',choices=['baseline','remaining'],default='baseline')
    p.add_argument('--proposal-source',help='Earlier cohort supplying paired, input-validated GraspGen proposals')
    p.add_argument('--full-grasp-probe',action='store_true',help='Require a candidate grasp to support the task motion before approach')
    p.add_argument('--grasp-seed',type=int,default=0)
    p.add_argument('--adaptive-horizon',action='store_true',help='Prefer a full-motion grasp and retain a short-probe fallback')
    p.add_argument('--payload-boundary',action='store_true',help='Require robot–held-object nonpenetration without extra positive clearance')
    args=p.parse_args();out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'results'):raise ValueError('Use a new results directory')
    if any(s not in {1,11,21,31,41} for s in args.states):raise ValueError('Use the existing five regression states')
    out.mkdir()
    selected=[next(t for t in TASKS if t['label']==label) for label in args.tasks]
    manifest=[{'task':t['label'],'state':s} for s in args.states for t in selected]
    ledger=ledger_for_output(out)
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__),ROOT/'scripts/reliable_microwave_final.py',ROOT/'scripts/run_success_pipeline.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in hashes:
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_json(out/'protocol.json',{'manifest':manifest,'sources_sha256':hashes,
        'horizon':args.horizon,'native_success_ends_episode':True,'execution_vetoes':False,
        'proposal_source':args.proposal_source,
        'full_grasp_probe':args.full_grasp_probe,
        'grasp_seed':args.grasp_seed,
        'adaptive_horizon':args.adaptive_horizon,
        'payload_boundary_only':args.payload_boundary,
        'task_specific_parameter_overrides_added':False,'source': 'Established family adapters and observed-RGB-validated grounding cache',
        'inherited_family_priors_retained':True,'fresh_acting_agent_evaluation':False,
        'api_run_id':ledger.run_id,'api_cap_usd':2.,'seed':0,'maximum_steps':600,
        'all_manifest_episodes_in_denominator':True})
    timeline=Timeline();resources=PairedResources(out,timeline,fresh=False);rows=[]
    resources.grasp_seed=args.grasp_seed
    resources.warm_models()
    for engine in [mechanisms,bowl,microwave]:
        engine.execute=execute_to_success
        engine.CoupledClient=lambda *a,**kw:SuccessClient(*a,**kw,fast=True,timeline=timeline)
    with api_run_scope(ledger.run_id):
        for entry in manifest:
            assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items()),'Sources changed during cohort'
            task=next(t for t in selected if t['label']==entry['task']);folder=out/f'{task["label"]}-state{entry["state"]:03d}'
            resources.source_case=(ROOT/args.proposal_source/f'{task["label"]}-state{entry["state"]:03d}'
                                   if args.proposal_source else None)
            timeline.context=entry;began=time.perf_counter()
            import torch
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
            print(json.dumps({'event':'started',**entry}),flush=True)
            fn,config,_=task_configuration(task)
            config['mechanism_horizon']=args.horizon
            config['grasp_probe_full_task']=args.full_grasp_probe
            config['adaptive_grasp_horizon']=args.adaptive_horizon
            if args.payload_boundary:config['robot_payload_clearance_m']=0.
            result=fn(task,entry['state'],folder,config,resources)
            if (folder/'success-termination.json').exists():
                result.update(status='task_success',pipeline_status='success_terminated',
                    termination_reason='simulator_success')
                if result.get('failure')=='simulator_success':
                    result.pop('failure',None);result.pop('failure_stage',None)
                    (folder/'error.txt').unlink(missing_ok=True)
            actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
            first=next((i for i,a in enumerate(actions) if a['evaluation_task_success']),None)
            result['control_steps_after_first_success']=len(actions)-first-1 if first is not None else 0
            result['successful_termination_recorded']=(folder/'success-termination.json').exists()
            segments=[read_json(p) for p in sorted((folder/'execution-segments').glob('*/result.json'))]
            result['requested_trajectories']=len(segments)
            result['all_issued_trajectories_completed']=all(s.get('all_commands_issued',False) for s in segments)
            result['all_trajectories_completed_or_success_terminated']=all(
                s.get('all_commands_issued',False) or s.get('terminated_on_simulator_success',False) for s in segments)
            result['horizon']=args.horizon
            write_json(folder/'summary.json',result)
            resources.cleanup_proposals();compress_retained_observations(folder)
            rows.append(result);timeline.save(out/'timing.json')
            write_json(out/'summary.json',{'planned':len(manifest),'completed':len(rows),
                'native_successes':sum(r['official_task_success'] for r in rows),'episodes':rows,
                'api':api_snapshot()})
            print(json.dumps({'event':'completed',**entry,'success':result['official_task_success'],
                'steps':result['control_steps'],'failure':result.get('failure'),'seconds':time.perf_counter()-began}),flush=True)


if __name__=='__main__':main()
