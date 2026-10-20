"""Shared execution adapter with explicit contact-phase records."""
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import numpy as np
from .profiled_sim_client import ProfiledClient, execute_batch
from .coupled_storage import read_json, write_json
from .shared_feedback import FeedbackConfig, bounded_vector


class FeedbackClient(ProfiledClient):
    def __init__(self,log_path,timeout=240,*,variant='reactive',timeline=None,fast=True):
        root=Path(__file__).resolve().parents[2];self.folder=Path(log_path).parent
        self.python='/home/duo/miniforge3/envs/vlm-tamp/bin/python';self.timeout=float(timeout)
        self.log_handle=open(log_path,'a');self.timeline=timeline;self.rpc_events=[];self.fast=True
        self.variant=variant;self.context=None;self.execution_calls=0;self.segment_count=0
        self.visual_offset=np.zeros(3);self.depth_records=[];self.depth_reference=None;self.depth_hand=None
        env=os.environ.copy()
        for key in list(env):
            if key.endswith(('_API_KEY','_TOKEN','_SECRET')):env.pop(key,None)
        env['PYTHONPATH']=str(root/'src');env['VLM_TAMP_BATCH_IO']='1'
        self.process=subprocess.Popen([self.python,'-m','vlm_tamp.shared_feedback_worker'],cwd=root,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log_handle,text=True,bufsize=1)
        self.messages=queue.Queue();self.request_id=0;self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()

    def contact_context(self, selected, cloud):
        self.context=dict(selected);self.depth_reference=np.asarray(cloud).copy()
        write_json(self.folder/'execution-contact.json',self.context)

    def observe(self,*args,**kwargs):
        result=super().observe(*args,**kwargs)
        from .mcvamp_mechanism import clear_native_cache
        from .mcvamp_contact_numerical import clear_native_cache as clear_arc
        clear_native_cache();clear_arc()
        if self.context and self.execution_calls>=2:
            from .cohort_execution import read_cohort_observation
            from .handle_contact import translated_surface
            cameras,_=read_cohort_observation(Path(result['observation_dir'])/'obs.npz',result['metadata'])
            hand=np.asarray(result['metadata']['eef_pose_world_xyzw'][:3])
            expected=np.zeros(3) if self.depth_hand is None else hand-self.depth_hand
            tracking=translated_surface(self.depth_reference,cameras,result['metadata']['robot_collision_volumes_world'],expected)
            if tracking['accepted']:
                measured=np.asarray(tracking['translation_world_m'])
                self.depth_reference+=measured
                self.visual_offset=(bounded_vector(measured-expected,.008) if self.depth_hand is not None else np.zeros(3))
            else:self.visual_offset=np.zeros(3)
            # The template stays at its last accepted observation. Predict
            # from that same hand observation if a frame was occluded.
            if tracking['accepted'] or self.depth_hand is None:self.depth_hand=hand
            self.depth_records.append({'control_step':result['control_steps'],'tracking':tracking,
                'next_reference_offset_world_m':self.visual_offset.tolist(),'execution_veto':False})
            write_json(self.folder/'feedback-depth-tracking.json',self.depth_records)
        return result


def _issue(sim,state,refs,gripper,phase,event_sink):
    target=sim.folder/f'execution-segments/{sim.segment_count:03d}-{phase}'
    target.mkdir(parents=True);sim.segment_count+=1
    refs=np.asarray(refs,float);np.save(target/'nominal.npy',refs)
    if len(refs)>600-int(state['control_steps']):raise ValueError('Whole issued segment exceeds episode budget')
    before=time.perf_counter()
    if sim.context is None:
        state,done=execute_batch(sim,state,refs,gripper=gripper)
        for e in done['events']:e['nominal_reference_joints']=e['reference_joints']
    else:
        context={'variant':sim.variant,'phase':phase,'normal_world':sim.context['contact_normal_world'],
                 'motion_tangent_world':sim.context.get('motion_tangent_world'),'mode':sim.context['mode'],
                 'visual_offset_world_m':sim.visual_offset.tolist(),'config':FeedbackConfig().record()}
        state=sim.request('feedback_trajectory',references=refs.tolist(),gripper=int(gripper),context=context)
        done=state.pop('trajectory')
    done['wall_seconds']=time.perf_counter()-before
    np.save(target/'issued.npy',[e['reference_joints'] for e in done['events']])
    write_json(target/'tracking.json',done['events'])
    write_json(target/'execution.json',{k:v for k,v in done.items() if k!='events'})
    if event_sink:
        for event in done['events']:event_sink(event)
    return state,done


def execute_shared(sim,state,references,*,gripper=1,maximum_control_steps=600,event_sink=None):
    call=sim.execution_calls;sim.execution_calls+=1
    phase='approach' if call==0 else 'contact' if call==1 else 'manipulate'
    if sim.context is not None:
        if call==0 and sim.context['mode']!='pinch':
            current=float(np.mean(np.abs(state['metadata']['gripper_qpos'])))
            count=max(0,int(np.rint((current-sim.context['half_aperture_m'])/.004)))
            if count:
                q=np.asarray(state['metadata']['joint_positions'])
                state,_=_issue(sim,state,np.repeat(q[None],count,axis=0),1,'free',event_sink)
        if sim.context['mode']!='pinch':gripper=0
    return _issue(sim,state,references,gripper,phase,event_sink)
