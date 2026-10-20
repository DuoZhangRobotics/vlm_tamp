"""Profiled simulator I/O; the original controller executes every reference."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from PIL import Image
from .coupled_worker import CoupledWorker,checked_joint_target
from .rekep_libero.sim_worker import MAX_CONTROL_STEPS


class ProfiledWorker(CoupledWorker):
    def __init__(self,fast=True):
        super().__init__();self.fast=bool(fast);self.in_batch=False;self.stats={};self.last_frame=-1
    @contextlib.contextmanager
    def timed(self,name):
        before=time.perf_counter()
        try:yield
        finally:
            seconds=time.perf_counter()-before
            row=self.stats.setdefault(name,{'seconds':0.,'calls':0});row['seconds']+=seconds;row['calls']+=1
    def reset(self,*args,stop_on_success=False,**kwargs):
        self.last_frame=-1
        self.stop_on_success=bool(stop_on_success)
        result=super().reset(*args,**kwargs)
        original_step=self.env.step
        def step(action):
            with self.timed('physics_and_environment_step'):return original_step(action)
        self.env.step=step
        disabled=[]
        if self.fast:
            inner=self.env.env
            for name in list(inner._observables):
                if name.startswith(('agentview_','robot0_eye_in_hand_')):
                    inner.modify_observable(name,'enabled',False);disabled.append(name)
        result['io_policy']={'mode':'batched' if self.fast else 'baseline',
            'automatic_camera_observables_disabled':disabled,'explicit_planning_views_unchanged':True,
            'physics_and_controller_unchanged':True,'video_every_joint_command':True}
        return result
    def _sensor_arrays(self):
        with self.timed('capture_all_calibrated_rgbd_and_robot_priors'):return super()._sensor_arrays()
    def _state_metadata(self):
        with self.timed('full_robot_metadata'):return super()._state_metadata()
    def _response_state(self):
        if not self.in_batch:return super()._response_state()
        return {'metadata':{'joint_positions':self.env.sim.data.qpos[self.robot._ref_joint_pos_indexes].tolist(),
            'eef_pose_world_xyzw':self._eef_pose().tolist(),
            'gripper_qpos':self.env.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes].tolist()},
            'control_steps':self.control_steps,'step_limit_reached':self.control_steps>=MAX_CONTROL_STEPS}
    def _save_frame(self,force=False):
        if self.fast and self.last_frame==self.control_steps:return
        if not force and self.control_steps%5:return
        with self.timed('video_frame_render_and_write'):super()._save_frame(force=force)
        self.last_frame=self.control_steps
    def _write_action_log(self):
        if self.in_batch:return
        with self.timed('write_action_and_contact_logs'):return super()._write_action_log()
    def observe(self,output_dir):
        if not self.fast:
            with self.timed('observe_total'):return super().observe(output_dir)
        with self.timed('observe_total'):
            self._require_env();output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
            if (output/'obs.npz').exists():raise FileExistsError('Observation already exists')
            arrays,cameras=self._sensor_arrays();metadata=self._state_metadata()
            metadata.update(cameras=cameras,primary_camera='frontview' if self.task_index==0 else 'agentview',
                instruction=self.instruction,no_simulator_inventory=True,schema_version='rekep-libero-observation-1.0')
            with self.timed('write_lossless_observation'):
                # The same arrays, without compression on the control path.
                np.savez(output/'obs.npz',**arrays)
                # Intermediate planning reads the unchanged RGB arrays from
                # obs.npz. Separate PNG copies are needed only for initial
                # grounding/cache validation and the final retained view.
                if output.name in {'initial','opened','final'}:
                    for name in ['agentview','wrist','frontview']:
                        Image.fromarray(arrays[name+'_rgb']).save(output/(name+'.png'))
                    Image.fromarray(arrays[metadata['primary_camera']+'_rgb']).save(output/'rgb.png')
                (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            return {'observation_dir':str(output),'metadata_file':str(output/'metadata.json'),
                'evaluation':{'task_success':bool(self.env.check_success())},'control_steps':self.control_steps,
                'step_limit_reached':self.control_steps>=MAX_CONTROL_STEPS}
    def joint_trajectory(self,references,gripper=1,stop_on_success=False):
        self._require_env();refs=np.asarray(references,float)
        if refs.ndim!=2 or refs.shape[1]!=7 or not np.isfinite(refs).all():raise ValueError('Finite seven-joint trajectory required')
        if len(refs)>MAX_CONTROL_STEPS-self.control_steps:raise ValueError('Complete trajectory exceeds remaining budget')
        for q in refs:checked_joint_target(q,self.ik_solver.limits)
        if gripper not in (-1,0,1):raise ValueError('Invalid gripper command')
        stop_on_success=bool(stop_on_success or getattr(self,'stop_on_success',False))
        terminal=bool(stop_on_success and self.env.check_success())
        before=time.perf_counter();events=[];self.in_batch=True
        try:
            for i,q in enumerate(refs):
                if terminal:break
                # Reuse the unchanged target normalization, official controller,
                # physics stepping and per-command diagnostics from the baseline.
                state=super().joint_action(q,gripper=gripper,max_steps=1)
                measured=np.asarray(state['metadata']['joint_positions'])
                events.append({'control_step':state['control_steps'],'reference_index':i,
                    'reference_joints':q.tolist(),'measured_joints':measured.tolist(),
                    'joint_reference_error_rad':float(np.max(np.abs(measured-q))),
                    'command_issued':True,'execution_veto_checks':0})
                terminal=bool(stop_on_success and self.action_log[-1]['evaluation_task_success'])
        finally:self.in_batch=False;self._write_action_log()
        state=self._response_state()
        return {**state,'trajectory':{'all_commands_issued':len(events)==len(refs),'planned_commands':len(refs),'issued_commands':len(events),
            'terminated_on_simulator_success':terminal,
            'termination_reason':'simulator_success' if terminal else 'trajectory_complete',
            'worker_seconds':time.perf_counter()-before,'execution_veto_checks':0,
            'maximum_joint_reference_error_rad':max((r['joint_reference_error_rad'] for r in events),default=0.),'events':events}}
    def dispatch(self,request):
        if request['op']=='joint_trajectory':
            return self.joint_trajectory(**{k:v for k,v in request.items() if k not in {'id','op'}})
        return super().dispatch(request)


def main():
    protocol=sys.stdout;worker=ProfiledWorker(os.environ.get('VLM_TAMP_BATCH_IO','1')=='1')
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                request={};worker.stats={};before=time.perf_counter()
                try:
                    request=json.loads(line);response={'ok':True,'id':request.get('id'),**worker.dispatch(request)}
                except Exception as error:
                    traceback.print_exc(file=sys.stderr)
                    response={'ok':False,'id':request.get('id'),'error':str(error),'error_type':type(error).__name__}
                response['worker_profile']={'server_seconds':time.perf_counter()-before,'stages':worker.stats}
                protocol.write(json.dumps(response,allow_nan=False)+'\n');protocol.flush()
                if request.get('op')=='close':break
        finally:worker.close()

if __name__=='__main__':main()
