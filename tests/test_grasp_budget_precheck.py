"""Reject impossible grasp reservations before generation, IK, or scene work."""
from types import SimpleNamespace
import pytest
from vlm_tamp.hinged_contact_tools import grasp_budget_check, require_grasp_budget, select_grasp


@pytest.mark.parametrize('step',[555,560,594,600])
def test_late_grasp_rejected_without_candidate_or_geometry_access(step,tmp_path):
    state={'control_steps':step}
    with pytest.raises(ValueError,match='Insufficient grasp budget') as exc:
        select_grasp(None,None,state,None,None,None,None,None,None,tmp_path/'selection',{})
    assert not (tmp_path/'selection').exists()
    assert exc.value.planning_diagnostic['status']=='insufficient_grasp_budget'
    assert exc.value.planning_diagnostic['physical_motion_issued'] is False


def test_service_checks_budget_before_calling_grasp_resources():
    from serve_subscription_robot import RobotService
    service=RobotService.__new__(RobotService)
    service.scene=object();service.state={'control_steps':560};service.config={}
    # No resources, simulator, IK, or output directory: precheck must not use them.
    with pytest.raises(ValueError,match='Insufficient grasp budget'):
        service._select_contact(3,24,0)


def test_actual_path_is_priced_without_implicit_200_step_reservation():
    assert require_grasp_budget({'control_steps':60},{})['available_approach_steps']==500
    check=grasp_budget_check({'control_steps':380},{})
    assert check['accepted'] and check['available_approach_steps']==180
    assert check['minimum_approach_steps']==7
    assert require_grasp_budget({'control_steps':400},{'minimum_manipulation_steps_reserved':80})['available_approach_steps']==80


def test_requested_execution_timing_changes_the_budget_without_changing_episode_cap():
    from vlm_tamp.hinged_contact_tools import grasp_timing_options
    old=grasp_budget_check({'control_steps':529},{})
    requested=grasp_budget_check({'control_steps':529},grasp_timing_options(.5,0,12))
    assert old['available_approach_steps']==31
    assert requested['available_approach_steps']==59
    assert requested['remaining_control_steps']==old['remaining_control_steps']==71
    assert requested['minimum_approach_steps']==1
    assert requested['maximum_joint_speed']==.5
    with pytest.raises(ValueError):grasp_timing_options(2.,0,12)
