"""Profiled simulator I/O; the original controller executes every reference."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from .coupled_worker import CoupledWorker,checked_joint_target
from .execution_limits import DEFAULT_CONTROL_STEPS, checked_control_steps


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
    def reset(self,*args,stop_on_success=False,maximum_control_steps=DEFAULT_CONTROL_STEPS,**kwargs):
        self.maximum_control_steps=checked_control_steps(maximum_control_steps)
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
            'control_steps':self.control_steps,'maximum_control_steps':self.maximum_control_steps,'step_limit_reached':self.control_steps>=self.maximum_control_steps}
    def _save_frame(self,force=False):
        if self.fast and self.last_frame==self.control_steps:return
        if not force and self.control_steps%5:return
        with self.timed('video_frame_render_and_write'):super()._save_frame(force=force)
        self.last_frame=self.control_steps
    def _write_action_log(self):
        if self.in_batch:return
        with self.timed('write_action_and_contact_logs'):return super()._write_action_log()
    def observe(self,output_dir,camera_names=None):
        if camera_names is not None:
            from .rekep_libero.sim_worker_v3 import ARM_VIEW_POLICIES
            if (not isinstance(camera_names,list) or not camera_names
                    or len(set(camera_names))!=len(camera_names)
                    or any(n not in {'agentview','wrist','frontview',*ARM_VIEW_POLICIES} for n in camera_names)):
                raise ValueError('Choose distinct calibrated cameras')
        self.selected_capture_cameras=camera_names
        try:return self._observe_selected(output_dir)
        finally:self.selected_capture_cameras=None
    def _observe_selected(self,output_dir):
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
                'maximum_control_steps':self.maximum_control_steps,'step_limit_reached':self.control_steps>=self.maximum_control_steps}
    def joint_trajectory(self, references: ArrayLike, gripper: int = 1,
                         stop_on_success: bool = False, push_feedback: dict | None = None,
                         arrival: dict | None = None) -> dict[str, Any]:
        """Execute references with measured completion for the declared contact model."""
        self._require_env();refs=np.asarray(references,float)
        if refs.ndim!=2 or refs.shape[1]!=7 or not np.isfinite(refs).all():raise ValueError('Finite seven-joint trajectory required')
        if len(refs)>self.maximum_control_steps-self.control_steps:raise ValueError('Complete trajectory exceeds remaining budget')
        for q in refs:checked_joint_target(q,self.ik_solver.limits)
        if gripper not in (-1,0,1):raise ValueError('Invalid gripper command')
        stop_on_success=bool(stop_on_success or getattr(self,'stop_on_success',False))
        terminal=bool(stop_on_success and self.env.check_success())
        feedback=None;tracking=None;tracking_failure=False
        if arrival is not None:
            from .timed_execution import pose_arrival
            if not len(refs):raise ValueError('Arrival tracking requires a nonempty trajectory')
            wait_steps=arrival.get('maximum_wait_steps',20)
            if type(wait_steps) is not int or wait_steps<0:raise ValueError('Use nonnegative maximum_wait_steps')
            tracked_offset=arrival.get('contact_local_m')
            tracked_normal=arrival.get('contact_normal_local')
            if tracked_offset is not None:
                tracked_offset=np.asarray(tracked_offset,float)
                if tracked_offset.shape!=(3,) or not np.isfinite(tracked_offset).all():raise ValueError('Finite contact offset required')
            if tracked_normal is not None:
                from .contact_orientation import normal_vector
                if tracked_offset is None:raise ValueError('A normal-constrained arrival requires its contact offset')
                tracked_normal=normal_vector(tracked_normal,'robot contact normal')
            tracking={'mode':'bounded_reference_lead_then_measured_arrival','maximum_wait_steps_per_reference':wait_steps,
                'maximum_joint_lead_rad':1./20.,'lead_bound_source':'existing 1 rad/s reference limit at 20 Hz',
                'extra_controls':0,'references_advanced':0,'native_collision_rechecks':0}
            goal_pose=self.ik_solver.fk(refs[-1])
        if push_feedback is not None:
            from .push_execution_feedback import PushProgress
            feedback=PushProgress(push_feedback,self._eef_pose())
            if feedback.references.shape!=(len(refs),3):raise ValueError('Push reference points must match the joint references')
        before=time.perf_counter();events=[];self.in_batch=True
        try:
            for i,q in enumerate(refs):
                if terminal or self.control_steps>=self.maximum_control_steps:break
                expected=self.ik_solver.fk(q) if tracking is not None else None
                repeats=0
                while True:
                    # Hold the same reference while the physical arm catches up.
                    # No IK correction, path replacement or simulator state edit.
                    state=super().joint_action(q,gripper=gripper,max_steps=1)
                    measured=np.asarray(state['metadata']['joint_positions'])
                    events.append({'control_step':state['control_steps'],'reference_index':i,
                        'reference_joints':q.tolist(),'measured_joints':measured.tolist(),
                        'joint_reference_error_rad':float(np.max(np.abs(measured-q))),
                        'command_issued':True,'execution_veto_checks':0,
                        **({'reference_repeat':repeats>0} if tracking is not None else {})})
                    terminal=bool(stop_on_success and self.action_log[-1]['evaluation_task_success'])
                    if feedback is not None:feedback.update(state['metadata']['eef_pose_world_xyzw'],i)
                    if tracking is None or terminal:break
                    measured_arrival=pose_arrival(expected,state['metadata']['eef_pose_world_xyzw'],tracked_offset,tracked_normal)
                    joint_lead=float(np.max(np.abs(measured-q)))
                    ready=(measured_arrival['reached'] if i==len(refs)-1 else joint_lead<=tracking['maximum_joint_lead_rad'])
                    tracking['last_reference']={'index':i,**measured_arrival,'joint_lead_rad':joint_lead,
                        'ready_for_next_reference':ready,'criterion':'final_pose_arrival' if i==len(refs)-1 else 'one_control_joint_lead'}
                    if ready:
                        tracking['references_advanced']+=1;break
                    if repeats>=wait_steps or self.control_steps>=self.maximum_control_steps:
                        tracking_failure=True;break
                    repeats+=1;tracking['extra_controls']+=1
                if tracking_failure:break
            if tracking is not None:
                complete=tracking['references_advanced']==len(refs)
                tracking['goal']={**pose_arrival(goal_pose,self._eef_pose(),tracked_offset,tracked_normal),'reference_path_completed':complete}
                tracking['goal_reached']=bool(complete and tracking['goal']['reached'])
                tracking['physical_infeasibility_proved']=False
        finally:self.in_batch=False;self._write_action_log()
        state=self._response_state()
        issued_path=0 if not events else events[-1]['reference_index']+1
        complete=(issued_path==len(refs) and (tracking is None or tracking['goal_reached']))
        budget_stop=bool(not terminal and not complete and self.control_steps>=self.maximum_control_steps)
        return {**state,'trajectory':{'all_commands_issued':complete,'nominal_path_commands':len(refs),
            'planned_commands':len(refs)+(tracking['extra_controls'] if tracking is not None else 0),'issued_commands':len(events),
            'terminated_on_simulator_success':terminal,
            'termination_reason':'simulator_success' if terminal else
                'control_budget_exhausted' if budget_stop else
                'reference_not_reached' if tracking_failure else 'trajectory_complete',
            'terminated_on_tracking_error':bool(tracking_failure and not terminal and not budget_stop),
            'terminated_on_control_budget':budget_stop,
            'physical_stall_is_execution_gate':False,
            **({'arrival':tracking} if tracking is not None else {}),
            **({'push_feedback':feedback.result()} if feedback is not None else {}),
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
