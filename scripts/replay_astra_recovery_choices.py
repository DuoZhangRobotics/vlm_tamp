"""Physically reconstruct saved decision points, then test each unique choice.

The existing planning/control methods are inherited unchanged. A validation-only
subclass exposes robot proprioception for matching; no simulator state is set.
"""
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from run_strategy_feedback import Resources,TASKS
from vlm_tamp.strategy_episode import run_episode
from vlm_tamp.recovery_episode import RecoveryTools
from vlm_tamp.recovery_decision import parse_decision
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.rekep_libero.api import ApiError

DIAGNOSTIC=ROOT/'results/astra-recovery-diagnostic-20261018'
SOURCE=ROOT/'results/vlm-recovery-frozen-20261018'
OUT=ROOT/'results/astra-recovery-physical-20261018'
PARAMETERS=('version','operation','contact_mode','approach_tilt_deg','motion_extent','step_mm','posture_rank','retreat_mm')

def signature(decision):return {k:decision[k] for k in PARAMETERS}

def compare_state(packet,reference,joints,expected_joints,image,reference_image):
    a=packet['observed_state'];b=reference['observed_state']
    image=np.asarray(image,dtype=np.int16);reference_image=np.asarray(reference_image,dtype=np.int16)
    diff=np.abs(image-reference_image)
    errors={
        'maximum_joint_error_rad':float(np.max(np.abs(np.asarray(joints)-expected_joints))),
        'maximum_tcp_pose_component_error':float(np.max(np.abs(np.asarray(a['eef_pose_world_xyzw'])-b['eef_pose_world_xyzw']))),
        'maximum_finger_error_m':float(np.max(np.abs(np.asarray(a['gripper_finger_positions_m'])-b['gripper_finger_positions_m']))),
        'estimated_progress_error_m':abs(a['estimated_object_progress_m']-b['estimated_object_progress_m']),
        'working_progress_error_m':abs(a['working_progress_m']-b['working_progress_m']),
        'image_mean_channel_error':float(diff.mean()),'image_max_channel_error':int(diff.max()),
        'same_control_step':packet['budget']['control_steps_remaining']==reference['budget']['control_steps_remaining']}
    geometry={k:float(np.max(np.abs(np.asarray(packet['mechanism_estimate'][k])-reference['mechanism_estimate'][k])))
        for k in ('joint_axis_world','panel_normal_world','source_anchor_world_m','target_progress')}
    accepted=(all(errors[k]<=1e-6 for k in ('maximum_joint_error_rad','maximum_tcp_pose_component_error',
        'maximum_finger_error_m','estimated_progress_error_m','working_progress_error_m'))
        and all(v<=1e-6 for v in geometry.values()) and errors['same_control_step']
        and errors['image_mean_channel_error']<=.25 and errors['image_max_channel_error']<=16)
    return {'accepted':bool(accepted),'errors':errors,'geometry_errors':geometry,
        'simulator_state_restoration_used':False,'native_success_used_for_matching':False}

class ReplayResources(Resources):
    source=None
    def api_ledger(self,*args,**kwargs):raise ApiError('Physical diagnostic may not make API calls')
    def grasps(self,cloud,output):
        if self.pins is None:
            import yaml
            from vlm_tamp.coupled_resources import sha
            config=ROOT/'.runtime/graspgen-models/checkpoints/graspgen_franka_panda.yml'
            raw=yaml.safe_load(config.read_text());self.pins={'config_sha256':sha(config)}
            for name in ('eval','discriminator'):
                checkpoint=Path(raw[name]['checkpoint'])
                if not checkpoint.is_absolute():checkpoint=config.parent/checkpoint
                self.pins[name+'_checkpoint_sha256']=sha(checkpoint)
        original=self.source/'grasps';provenance=read_json(original/'input-provenance.json')
        if not np.array_equal(np.load(original/'input-cloud.npy'),np.asarray(cloud,np.float32)):
            raise ValueError('Reconstructed grasp cloud differs from saved input; no substitution')
        if provenance['checkpoint_pins']!=self.pins:raise ValueError('GraspGen checkpoint provenance differs')
        shutil.copytree(original,output)
        write_json(Path(output)/'replay-provenance.json',{'source':str(original),'input_cloud_identical':True,
            'checkpoint_pins_identical':True,'cached_predictions_only':True})
        with np.load(Path(output)/'grasp_candidates.npz') as a:
            return {k:a[k].copy() for k in ('poses_world_base','scores','retained_indices')}

class ReplayPolicy:
    calls=0
    def __init__(self,source,round_id,choice):
        self.source=source;self.branch=int(round_id.rsplit('-',1)[1]);self.choice=choice
        self.history=read_json(source/'strategy-history.json');self.number=0;self.session=None
        self.round_id=round_id
    def choose(self,packet,image,folder,scene_id):
        number=self.number;self.number+=1
        if number<self.branch:
            choice=self.history[number]['decision'];origin='replayed prior Luna choice'
        elif number==self.branch:
            reference=read_json(self.source/'decisions'/self.round_id/'feedback.json')
            actions=read_json(self.source/'initial/actions.json')
            count=600-reference['budget']['control_steps_remaining']
            expected=actions[count-1]['joint_positions']
            state=self.session.state['metadata']
            match=compare_state(packet,reference,state['joint_positions'],expected,
                Image.open(image).convert('RGB'),Image.open(self.source/'decisions'/self.round_id/'feedback-sheet.png').convert('RGB'))
            write_json(self.session.folder/'decision-point-match.json',match)
            if not match['accepted']:raise ValueError('Reconstructed decision point differs; no counterfactual action executed')
            choice=self.choice;origin='new shared Luna/Astra diagnostic choice'
        else:
            choice={'version':'strategy-v2','operation':'finish','contact_mode':None,'approach_tilt_deg':None,
                'motion_extent':None,'step_mm':None,'posture_rank':None,'retreat_mm':None,
                'evidence':'The predeclared single-decision diagnostic is complete.',
                'expected_observation':'Report this tool result without claiming a full new policy rollout.'}
            origin='scripted end of single-decision diagnostic'
        write_json(Path(folder)/'decision.json',choice)
        write_json(Path(folder)/'decision-origin.json',{'source':origin,'original_round':number})
        return parse_decision(json.dumps(choice))

def main():
    OUT.mkdir(exist_ok=False)
    diagnostic=read_json(DIAGNOSTIC/'summary.json');protocol=read_json(DIAGNOSTIC/'protocol.json')
    groups=[]
    for key,episode,round_id in protocol['cases']:
        rows=[r for r in diagnostic['results'] if r['case']==key]
        if not all(r['decision_valid'] for r in rows):raise ValueError('All planned model decisions must be valid before physical comparison')
        # Do not count two identical command choices as two physical samples.
        if not all(signature(r['decision'])==signature(rows[0]['decision']) for r in rows):
            raise ValueError('This runner handles the observed shared choices; distinct choices need separate predeclared branches')
        groups.append({'case':key,'source_episode':episode,'source_round':round_id,'decision':rows[0]['decision'],
            'shared_by_models':[r['model'] for r in rows]})
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__),ROOT/'scripts/run_strategy_feedback.py',
        ROOT/'scripts/run_shared_contact_feedback.py',ROOT/'scripts/run_success_pipeline.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in hashes:
        dest=OUT/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write_json(OUT/'protocol.json',{'sources_sha256':hashes,'maximum_control_steps':600,'maximum_decision_rounds':5,
        'seed':0,'new_model_decisions_per_replay':1,'replayed_prefix':'prior model choices through unchanged tools',
        'planning_behavior_changed':False,'simulator_state_restoration_used':False,'new_API_calls_allowed':False,
        'study':'physical feasibility of shared recovery choices; not separate model episode success rates',
        'matching_tolerances':{'joint_rad':1e-6,'tcp_component':1e-6,'geometry_m':1e-6,'RGB_mean':.25,'RGB_max':16}})
    manifest=[]
    for group in groups:
        old=read_json(SOURCE/group['source_episode']/'summary.json')
        manifest.append({'task':old['task'],'state':old['initial_state'],'policy':'shared','case':group['case']})
    write_json(OUT/'manifest.json',{'episodes':manifest,'unique_choices':groups})
    resources=ReplayResources(OUT,Timeline());resources.allow_paid=False;resources.warm_models();results=[]
    for group,entry in zip(groups,manifest):
        import torch
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        source=SOURCE/group['source_episode'];resources.source=source
        policy=ReplayPolicy(source,group['source_round'],group['decision'])
        class TrackedTools(RecoveryTools):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);policy.session=self
        config=read_json(SOURCE/'protocol.json')['configuration'][entry['task']]
        folder=OUT/f"{entry['task']}-state{entry['state']:03d}-shared"
        print(json.dumps({'event':'started','case':group['case']}),flush=True)
        with patch('vlm_tamp.recovery_episode.RecoveryTools',TrackedTools):
            result=run_episode(TASKS[entry['task']],entry['state'],folder,config,resources,policy)
        compress_retained_observations(folder);result=read_json(folder/'summary.json')
        result.update(strategy_policy='shared',diagnostic_case=group['case'],shared_by_models=group['shared_by_models'],
            single_decision_followup=True,new_API_calls=0)
        write_json(folder/'summary.json',result);results.append(result)
        write_json(OUT/'summary.json',{'planned_unique_replays':len(groups),'completed':len(results),'episodes':results})
        resources.cleanup_case();print(json.dumps({'event':'completed',**result}),flush=True)
    resources.cleanup_case();resources.case_cache.rmdir()

if __name__=='__main__':main()
