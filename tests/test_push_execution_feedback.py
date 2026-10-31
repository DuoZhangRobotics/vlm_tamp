from types import SimpleNamespace
import numpy as np
import pytest
from vlm_tamp.push_execution_feedback import PushProgress


def pose(x):return [x,0.,0.,0.,0.,0.,1.]


def test_physical_stall_requires_a_second_of_attempted_motion():
    refs=np.c_[np.arange(1,41)*.0015,np.zeros((40,2))]
    feedback=PushProgress({'contact_local_m':[0.,0.,0.],'reference_points_world_m':refs},pose(0))
    for index in range(19):assert not feedback.update(pose(0),index)
    assert feedback.update(pose(0),19)
    result=feedback.result()
    assert result['samples']==20 and result['stall_window']['commanded_travel_m']==pytest.approx(.03)
    assert result['measured_contact_displacement_m']==0 and not result['scene_prediction_veto']


def test_slow_but_real_hand_progress_is_not_a_stall():
    refs=np.c_[np.arange(1,41)*.0015,np.zeros((40,2))]
    feedback=PushProgress({'contact_local_m':[0.,0.,0.],'reference_points_world_m':refs},pose(0))
    for index in range(40):assert not feedback.update(pose((index+1)*.0003),index)


def test_native_worker_attempts_motion_before_returning_physical_stall(monkeypatch):
    from vlm_tamp.profiled_sim_worker import ProfiledWorker
    from vlm_tamp.coupled_worker import CoupledWorker
    worker=ProfiledWorker();worker.maximum_control_steps=600;worker.control_steps=60;worker.action_log=[]
    worker.ik_solver=SimpleNamespace(limits=np.tile([-3.,3.],(7,1)))
    worker.env=SimpleNamespace(check_success=lambda:False)
    worker._eef_pose=lambda:np.array(pose(0));worker._write_action_log=lambda:None
    worker._response_state=lambda:{'control_steps':worker.control_steps}
    def step(self,q,gripper,max_steps):
        self.control_steps+=1;self.action_log.append({'evaluation_task_success':False})
        return {'control_steps':self.control_steps,'metadata':{'joint_positions':[0.]*7,'eef_pose_world_xyzw':pose(0)}}
    monkeypatch.setattr(CoupledWorker,'joint_action',step)
    refs=np.zeros((60,7));points=np.c_[np.arange(1,61)*.0015,np.zeros((60,2))]
    result=worker.joint_trajectory(refs,gripper=0,push_feedback={'contact_local_m':[0.,0.,0.],
        'reference_points_world_m':points})['trajectory']
    assert result['issued_commands']==20 and result['termination_reason']=='measured_push_stall'
    assert not result['terminated_on_simulator_success'] and not result['all_commands_issued']
    assert result['push_feedback']['stalled'] and result['execution_veto_checks']==0
