import numpy as np
import pytest

from vlm_tamp.timed_execution import schedule,execute


class BlockedSimulator:
    def __init__(self):self.commands=[]
    def joint_action(self,target_joints,gripper,max_steps):
        self.commands.append(np.asarray(target_joints).copy())
        # Neither tracking progress nor an already-true success predicate may
        # stop the requested sequence. Physics is deliberately blocked here.
        return {'control_steps':len(self.commands),'task_success':True,
            'metadata':{'joint_positions':np.zeros(7)}}


def test_blocked_arm_still_receives_every_command():
    sim=BlockedSimulator();refs=np.linspace(np.zeros(7),np.ones(7)*.5,50)
    _,result=execute(sim,{'control_steps':0},refs)
    assert result['all_commands_issued']
    assert result['issued_commands']==50
    assert result['execution_veto_checks']==0
    np.testing.assert_array_equal(sim.commands[-1],refs[-1])
    assert result['maximum_joint_reference_error_rad']==.5


def test_episode_limit_is_checked_before_any_command():
    sim=BlockedSimulator()
    with pytest.raises(ValueError,match='remaining episode'):
        execute(sim,{'control_steps':590},np.zeros((11,7)))
    assert not sim.commands


def test_timing_is_independent_of_dense_planner_sample_count():
    sparse=np.linspace(np.zeros(7),np.ones(7)*.5,2)
    dense=np.linspace(np.zeros(7),np.ones(7)*.5,1001)
    a=schedule(sparse,maximum_joint_speed=.5,endpoint_hold_steps=5)
    b=schedule(dense,maximum_joint_speed=.5,endpoint_hold_steps=5)
    assert len(a)==len(b)==25
    np.testing.assert_allclose(a,b,atol=1e-14)
    assert np.max(np.abs(np.diff(a,axis=0)))<=.025+1e-12
    np.testing.assert_array_equal(a[-1],sparse[-1])


def test_stationary_path_has_finite_commands():
    result=schedule(np.zeros((3,7)),endpoint_hold_steps=4)
    assert result.shape==(5,7)
    assert np.isfinite(result).all()
