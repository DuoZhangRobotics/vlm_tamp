"""Closing must finish mechanically before transport, without silently reopening an attempt."""
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.completion_gripper import closure_sample
from vlm_tamp.coupled_worker import CoupledWorker
from vlm_tamp.execution_grasp import require_new_grasp
from vlm_tamp.profiled_sim_worker import ProfiledWorker


def test_completion_needs_both_finished_reference_and_measured_finger_rest() -> None:
    """Neither a stalled command ramp nor a transient low displacement proves completion."""
    reference=[-1.,1.]
    assert not closure_sample([.02,-.02],[.02,-.02],[0,0],[0,0],reference)['settled']
    assert not closure_sample([.02,-.02],[.01999,-.01999],[.01,-.01],reference,reference)['settled']
    assert not closure_sample([.02,-.02],[.019,-.019],[0,0],reference,reference)['settled']
    result=closure_sample([.002,-.002],[.002001,-.002001],[.0001,-.0001],reference,reference)
    assert result['settled'] and result['retention']=='not_tested'
    assert not result['simulator_contacts_used']
    # Fully closed without an object is still mechanical completion, not retention.
    assert closure_sample([0,0],[0,0],[0,0],reference,reference)['retention']=='not_tested'


def worker_with_script(monkeypatch: pytest.MonkeyPatch, maximum_controls: int = 600) -> tuple:
    """Use the actual batch controller with deterministic proprioceptive samples."""
    worker=ProfiledWorker();worker.control_steps=60;worker.maximum_control_steps=maximum_controls
    worker.action_log=[];worker._require_env=lambda:None;worker._write_action_log=lambda:None
    data=SimpleNamespace(qpos=np.array([.04,-.04]),qvel=np.zeros(2))
    worker.env=SimpleNamespace(sim=SimpleNamespace(data=data),check_success=lambda:False)
    worker.robot=SimpleNamespace(_ref_gripper_joint_pos_indexes=[0,1],_ref_gripper_joint_vel_indexes=[0,1],
                                 gripper=SimpleNamespace(current_action=np.zeros(2)))
    worker.ik_solver=SimpleNamespace(limits=np.tile([-3.,3.],(7,1)))
    worker._response_state=lambda:{'control_steps':worker.control_steps}
    sent=[]
    values=[([.03,-.03],[-.2,.2],[-.5,.5]),([.02,-.02],[-.1,.1],[-1.,1.]),
            ([.02,-.02],[0.,0.],[-1.,1.])]
    def step(self: ProfiledWorker, q: np.ndarray, gripper: int, max_steps: int) -> dict:
        """Advance physics once and expose current values to the real completion logic."""
        position,velocity,reference=values[min(len(sent),2)]
        data.qpos[:]=position;data.qvel[:]=velocity;worker.robot.gripper.current_action[:]=reference
        sent.append((q.copy(),gripper));worker.control_steps+=1
        worker.action_log.append({'evaluation_task_success':False})
        return {'control_steps':worker.control_steps,'metadata':{'joint_positions':q.tolist()}}
    monkeypatch.setattr(CoupledWorker,'joint_action',step)
    return worker,sent


def test_worker_holds_arm_until_measurement_finishes_without_waiting_to_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forty-control ceiling is not a forty-control waiting buffer."""
    worker,sent=worker_with_script(monkeypatch)
    result=worker.joint_trajectory(np.zeros((40,7)),gripper=1,
        gripper_completion={'minimum_controls':1})['trajectory']
    assert result['issued_commands']==3
    assert result['termination_reason']=='gripper_settled'
    assert result['gripper_completion']['settled']
    assert all(np.array_equal(q,np.zeros(7)) and grip==1 for q,grip in sent)
    assert result['events'][-1]['gripper_motion']['settled']


def test_budget_cutoff_cannot_be_reported_as_finished_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two commands execute, but the measured moving fingers are explicitly incomplete."""
    worker,sent=worker_with_script(monkeypatch,maximum_controls=62)
    result=worker.joint_trajectory(np.zeros((2,7)),gripper=1,
        gripper_completion={'minimum_controls':1})['trajectory']
    assert len(sent)==2 and not result['gripper_completion']['settled']
    assert result['termination_reason']=='control_budget_exhausted'
    assert not result['all_commands_issued']


def test_moving_arm_cannot_be_labeled_a_stationary_closing_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a malformed completion request before any physical commands are issued."""
    worker,sent=worker_with_script(monkeypatch);refs=np.zeros((2,7));refs[1,0]=.1
    with pytest.raises(ValueError,match='stationary arm'):
        worker.joint_trajectory(refs,gripper=1,gripper_completion={'minimum_controls':1})
    assert not sent


def test_retry_preserves_attempt_until_caller_explicitly_chooses_regrasp() -> None:
    """Unknown retention is not an automatic release signal."""
    from serve_subscription_robot import RobotService
    service=SimpleNamespace(gripper_closure={'settled':True,'retention':'not_tested'})
    with pytest.raises(ValueError,match='previous closing attempt is preserved') as caught:
        RobotService._select_contact(service,3.,8,0)
    assert caught.value.planning_diagnostic['physical_motion_issued'] is False
    assert service.gripper_closure['settled']
    require_new_grasp(service,regrasp=True)
    service.gripper_closure=None
    require_new_grasp(service)


def test_callable_schema_and_compact_feedback_keep_mechanical_state() -> None:
    """The caller can explicitly choose a retry and see why the hand was preserved."""
    from vlm_tamp.robot_contracts import validate_operation
    from vlm_tamp.compact_robot_interface import compact_response
    validate_operation('grasp_object',{'object_id':'object1','regrasp':True})
    validate_operation('plan_grasp',{'regrasp':True})
    measured={'settled':True,'retention':'not_tested'}
    compact=compact_response({'ok':True,'result':{'execution':{'gripper_completion':measured}}})
    assert compact['result']['execution']['gripper_completion']==measured
