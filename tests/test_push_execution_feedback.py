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


def test_short_strokes_share_the_existing_stall_window():
    # Two separately scheduled 0.7 s strokes must not reset a 1 s detector.
    refs=np.c_[np.arange(1,15)*.0015,np.zeros((14,2))]
    first=PushProgress({'contact_local_m':[0.,0.,0.],'reference_points_world_m':refs},pose(0))
    for index in range(14):assert not first.update(pose(0),index)
    second=PushProgress({'contact_local_m':[0.,0.,0.],'reference_points_world_m':refs+np.array([.021,0,0]),
        'previous_history':first.result()['_continuation_history']},pose(0))
    for index in range(5):assert not second.update(pose(0),index)
    assert second.update(pose(0),5)
    assert second.result()['carried_history_samples']==15


def test_short_stroke_history_preserves_real_progress():
    history=[]
    for batch in range(4):
        start=batch*.014
        refs=np.c_[start+np.arange(1,15)*.001,np.zeros((14,2))]
        feedback=PushProgress({'contact_local_m':[0.,0.,0.],'reference_points_world_m':refs,
            'previous_history':history},pose(start))
        for index in range(14):assert not feedback.update(pose(start+(index+1)*.001),index)
        history=feedback.result()['_continuation_history']
    assert feedback.result()['stall_window'] is not None


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
