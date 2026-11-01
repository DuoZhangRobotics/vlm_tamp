"""Approach execution follows physical progress before allowing a push."""
from types import SimpleNamespace
import numpy as np
import pytest
from vlm_tamp.profiled_sim_worker import ProfiledWorker
from vlm_tamp.coupled_worker import CoupledWorker


def worker(monkeypatch, *, response=.5, controls=600):
    robot=ProfiledWorker();robot.control_steps=60;robot.maximum_control_steps=controls
    robot.action_log=[];actual=np.zeros(7);calls=[]
    pose=lambda q:np.r_[q[:3],[0.,0.,0.,1.]]
    robot.ik_solver=SimpleNamespace(limits=np.tile([-3.,3.],(7,1)),fk=pose)
    robot.env=SimpleNamespace(check_success=lambda:False)
    robot._eef_pose=lambda:pose(actual)
    robot._write_action_log=lambda:None
    robot._response_state=lambda:{'control_steps':robot.control_steps,'metadata':{'joint_positions':actual.tolist(),'eef_pose_world_xyzw':pose(actual).tolist()}}
    def step(self,goal,gripper,max_steps):
        calls.append(np.asarray(goal).copy());actual[:]+=response*(goal-actual)
        self.control_steps+=1;self.action_log.append({'evaluation_task_success':False})
        return self._response_state()
    monkeypatch.setattr(CoupledWorker,'joint_action',step)
    return robot,calls,actual


def test_waits_for_measured_reference_before_advancing_without_changing_targets(monkeypatch):
    robot,calls,actual=worker(monkeypatch)
    refs=np.zeros((2,7));refs[:,0]=[.2,.4]
    result=robot.joint_trajectory(refs,gripper=0,arrival={'maximum_wait_steps':20})['trajectory']
    assert result['arrival']['goal_reached'] and result['all_commands_issued']
    assert len(calls)>len(refs)
    assert all(np.array_equal(q,refs[0]) or np.array_equal(q,refs[1]) for q in calls)
    split=next(i for i,q in enumerate(calls) if np.array_equal(q,refs[1]))
    assert .2*(.5**split)<=.05
    assert abs(actual[0]-.4)<=.004
    assert result['issued_commands']==result['planned_commands']==len(calls)
    assert result['nominal_path_commands']==2
    assert result['arrival']['extra_controls']==len(calls)-2


def test_blocked_reference_does_not_advance_to_later_targets(monkeypatch):
    robot,calls,_=worker(monkeypatch,response=0.)
    refs=np.zeros((2,7));refs[:,0]=[.1,.2]
    result=robot.joint_trajectory(refs,gripper=0,arrival={'maximum_wait_steps':3})['trajectory']
    assert len(calls)==4 and all(np.array_equal(q,refs[0]) for q in calls)
    assert not result['all_commands_issued'] and not result['arrival']['goal_reached']
    assert result['termination_reason']=='reference_not_reached'
    assert result['terminated_on_tracking_error']
    assert result['arrival']['last_reference']['position_error_m']==pytest.approx(.1)
    assert result['arrival']['goal']['position_error_m']==pytest.approx(.2)
    assert not result['arrival']['physical_infeasibility_proved']


def test_waiting_consumes_real_episode_controls(monkeypatch):
    robot,calls,_=worker(monkeypatch,response=0.,controls=62)
    refs=np.zeros((2,7));refs[:,0]=[.04,.08]
    result=robot.joint_trajectory(refs,arrival={'maximum_wait_steps':20})['trajectory']
    assert robot.control_steps==62 and len(calls)==2
    assert not result['arrival']['goal_reached']


def test_no_waits_are_added_when_robot_already_tracks_each_target(monkeypatch):
    robot,calls,_=worker(monkeypatch,response=1.)
    refs=np.zeros((2,7));refs[:,0]=[.04,.08]
    result=robot.joint_trajectory(refs,arrival={'maximum_wait_steps':20})['trajectory']
    np.testing.assert_array_equal(calls,refs)
    assert result['arrival']['extra_controls']==0
    assert result['arrival']['goal_reached']


from test_retained_push_contact import contact_service


def test_targeted_push_cannot_follow_an_unreached_approach(contact_service,monkeypatch):
    from test_retained_push_contact import call
    service,counts,_=contact_service;original=service.issue;labels=[]
    import vlm_tamp.contact_skill_service as module
    original_engage=module.plan_engagement
    def fresh(*a,**k):
        result=original_engage(*a,**k);refs=np.zeros((1,7))
        result.update(reuse_existing_contact=False,references=refs,free_references=refs)
        return result
    monkeypatch.setattr(module,'plan_engagement',fresh)
    def blocked(refs,gripper,label,**kwargs):
        labels.append(label);result=original(refs,gripper,label,**kwargs)
        if label=='free-approach':
            result['execution']['arrival']={'goal_reached':False,'goal':{'position_error_m':.074},
                'last_reference':{'position_error_m':.074}}
        return result
    monkeypatch.setattr(service,'issue',blocked)
    result=call(service)
    assert labels==['free-approach']
    assert result['stop_reason']=='approach_not_reached'
    assert result['robot_motion_executed'] and not result['continue_push_available']
    assert result['diagnostic']['push_stroke_issued'] is False
    assert result['approach_arrival']['goal']['position_error_m']==.074


def test_point_goal_completion_does_not_add_a_wrist_constraint():
    from scipy.spatial.transform import Rotation
    from vlm_tamp.timed_execution import pose_arrival
    target=[0,0,0,0,0,0,1]
    actual=[0,0,0,*Rotation.from_euler('z',60,degrees=True).as_quat()]
    assert not pose_arrival(target,actual)['reached']
    assert pose_arrival(target,actual,[0,0,0])['reached']


def test_dashboard_explains_measured_failure_and_whether_push_started():
    from live_study_dashboard import issue_explanation
    arrival={'goal_reached':False,'last_reference':{'index':64,'position_error_m':.0312},
        'goal':{'position_error_m':.0809},'extra_controls':32}
    report=issue_explanation({'arrival':arrival,'push_stroke_issued':False})
    assert '31.2 mm' in report and '80.9 mm' in report
    assert 'push was not started' in report
    active=issue_explanation({'arrival':arrival,'push_stroke_issued':True})
    assert 'Remaining motion references' in active


def test_budget_ends_between_references_without_claiming_completion(monkeypatch):
    robot,calls,_=worker(monkeypatch,response=.5,controls=62)
    refs=np.zeros((2,7));refs[:,0]=[.2,.4]
    result=robot.joint_trajectory(refs,arrival={'maximum_wait_steps':20})['trajectory']
    assert robot.control_steps==62
    assert result['termination_reason']=='control_budget_exhausted'
    assert result['terminated_on_control_budget']
    assert not result['all_commands_issued'] and not result['arrival']['goal_reached']
    assert not result['terminated_on_tracking_error']
    assert result['arrival']['last_reference']['ready_for_next_reference']
