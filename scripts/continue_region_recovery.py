"""Replay the eight-decision prefix, then permit four more VLM decisions.

Eligibility uses only the prior tool's observed progress and remaining control
budget. Native object state is never used to select or steer a continuation.
"""
import argparse
import hashlib
from pathlib import Path
import random
import shutil
import sys
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from run_strategy_feedback import Resources,TASKS
from vlm_tamp.region_multiview import SeparateRegionTools,SeparateLunaStrategy
from vlm_tamp.strategy_episode import run_episode
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.run_api_budget import api_run_scope,ledger_for_output,snapshot
from vlm_tamp.cli import load_credentials

SOURCE=ROOT/'results/region-recovery-separate-20261018'
OUT=ROOT/'results/region-recovery-continuation-20261018'

class PrefixResources(Resources):
    def grasps(self,cloud,output):
        if not getattr(self,'prefix',False):return super().grasps(cloud,output)
        xyz=np.asarray(cloud,np.float32)
        for path in self.source.rglob('input-cloud.npy'):
            if not (path.parent/'grasp_candidates.npz').exists():continue
            if np.array_equal(np.load(path),xyz):
                shutil.copytree(path.parent,output)
                write_json(Path(output)/'prefix-replay.json',{'source':str(path.parent),'identical_input_cloud':True})
                with np.load(Path(output)/'grasp_candidates.npz') as a:
                    return {k:a[k].copy() for k in ('poses_world_base','scores','retained_indices')}
        raise ValueError('Replayed prefix has a new grasp input; no silent replacement')

def tracking_history_matches(actual,stored,control_step):
    # run_episode takes one final observation after its decision loop. That
    # post-episode bookkeeping frame was never part of the policy prefix.
    extras=len(stored)-len(actual)
    if extras not in (0,1):return False,False
    ignored=extras==1
    if ignored and stored[-1]['control_step']!=control_step:return False,False
    equal=[r['tracking'] for r in actual]==[r['tracking'] for r in stored[:len(actual)]]
    return equal,ignored

def check_endpoint(session,source):
    expected=read_json(source/'final/metadata.json');actual=session.state['metadata'];errors={}
    for key in ('joint_positions','eef_pose_world_xyzw','gripper_qpos'):
        errors[key]=float(np.max(np.abs(np.asarray(actual[key])-expected[key])))
    images={name:bool(np.array_equal(session.views[name]['rgb'],np.asarray(Image.open(source/'final'/f'{name}.png'))))
        for name in ('agentview','frontview')}
    old=read_json(source/'current-task-spec.json')
    errors['axis']=float(np.max(np.abs(np.asarray(session.spec['joint_axis_world'])-old['joint_axis_world'])))
    errors['target_progress']=abs(session.spec['target_progress']-old['target_progress'])
    old_progress=read_json(source/'observed-progress.json')['records'][-1]
    errors['observed_progress']=abs(session.progress.observed_progress-old_progress['observed_progress_m'])
    errors['working_progress']=abs(session.progress.progress-old_progress['progress_m'])
    tracking=read_json(source/'feedback-depth-tracking.json')
    same_tracking,ignored=tracking_history_matches(session.sim.depth_records,tracking,session.state['control_steps'])
    steps=read_json(source/'summary.json')['control_steps']==session.state['control_steps']
    return {'accepted':bool(all(v<=1e-6 for v in errors.values()) and all(images.values()) and same_tracking and steps),
        'errors':errors,'RGB_identical':images,'depth_tracking_history_identical':same_tracking,
        'post_episode_bookkeeping_observation_ignored':ignored,
        'same_control_step':steps,'simulator_object_state_used':False,'simulator_state_injection_used':False}

class PrefixPolicy:
    def __init__(self,source,live,resources):
        self.source=source;self.live=live;self.resources=resources
        self.history=read_json(source/'strategy-history.json');self.number=0;self.session=None
    @property
    def calls(self):return self.live.calls
    def choose(self,packet,image,folder,scene_id):
        number=self.number;self.number+=1
        if number<len(self.history):
            choice=self.history[number]['decision'];write_json(Path(folder)/'decision.json',choice)
            write_json(Path(folder)/'decision-origin.json',{'source':'replayed prior VLM choice','original_round':number})
            return choice
        if number==len(self.history):
            match=check_endpoint(self.session,self.source);write_json(self.session.folder/'continuation-match.json',match)
            if not match['accepted']:raise ValueError('Prefix reconstruction mismatch; no new model call or continuation')
            self.resources.prefix=False
        return self.live.choose(packet,image,folder,scene_id)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(OUT))
    args=parser.parse_args();OUT=Path(args.output).resolve()
    if not OUT.is_relative_to(ROOT/'results'):raise ValueError('Output must be in results')
    OUT.mkdir(exist_ok=False)
    entries=read_json(SOURCE/'manifest.json')['episodes'];planned=[]
    for entry in entries:
        folder=SOURCE/f"{entry['task']}-state{entry['state']:03d}-{entry['policy']}"
        history=read_json(folder/'strategy-history.json');last=history[-1]['result'];s=read_json(folder/'summary.json')
        eligible=(last['status']=='motion_completed' and last.get('tracking_accepted') and
            last.get('object_progress_delta_m',0)>.003 and not last.get('estimated_goal_reached',False) and s['control_steps']<550)
        if eligible:planned.append({'task':entry['task'],'state':entry['state'],'policy':'continuation','source':str(folder)})
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__),ROOT/'configs/prompts/strategy_feedback_v3.txt',
        ROOT/'scripts/run_strategy_feedback.py',ROOT/'scripts/run_shared_contact_feedback.py',ROOT/'scripts/run_success_pipeline.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in hashes:
        dest=OUT/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write_json(OUT/'manifest.json',{'episodes':planned,'eligibility':'last tool completed, accepted tracking, >3mm observed progress, estimated goal not reached, >50 control steps remain'})
    config=read_json(SOURCE/'protocol.json')['configuration']
    config={**config,'strategy_tool_version':'v3','separate_camera_inputs':True}
    write_json(OUT/'protocol.json',{'sources_sha256':hashes,'configuration':config,'maximum_decision_rounds':12,
        'maximum_control_steps':600,'maximum_new_model_decisions':4,'seed':0,
        'original_scores_unchanged':True,'comparison':'conditional continuation from observed-progress prefixes; not a fresh policy success-rate benchmark',
        'simulator_object_state_used_for_eligibility_or_control':False,'shared_api_run_id':'vlm-region-recovery-20261018'})
    ledger=ledger_for_output(OUT,'vlm-region-recovery-20261018');load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
    resources=PrefixResources(OUT,Timeline());resources.allow_paid=False;resources.warm_models();rows=[]
    with api_run_scope(ledger.run_id):
        for entry in planned:
            import torch
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
            assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
            resources.prefix=True;resources.source=Path(entry['source'])
            policy=PrefixPolicy(resources.source,SeparateLunaStrategy(ledger,allow_paid=True),resources)
            class TrackedTools(SeparateRegionTools):
                def __init__(self,*a,**kw):super().__init__(*a,**kw);policy.session=self
            folder=OUT/f"{entry['task']}-state{entry['state']:03d}-continuation"
            print({'event':'started','state':entry['state']},flush=True)
            with patch('vlm_tamp.region_multiview.SeparateRegionTools',TrackedTools):
                run_episode(TASKS[entry['task']],entry['state'],folder,config,resources,policy,maximum_rounds=12)
            compress_retained_observations(folder);result=read_json(folder/'summary.json')
            result.update(strategy_policy='continuation',source_prefix=entry['source'],conditional_continuation=True)
            write_json(folder/'summary.json',result);rows.append(result)
            write_json(OUT/'summary.json',{'planned':len(planned),'completed':len(rows),'episodes':rows,'api':snapshot(ledger)})
            resources.cleanup_case();print({'event':'completed','state':entry['state'],'success':result['official_task_success'],
                'steps':result['control_steps'],'failure':result.get('failure')},flush=True)
        resources.cleanup_case();resources.case_cache.rmdir()

if __name__=='__main__':main()
