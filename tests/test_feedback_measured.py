"""Actionable feedback must retain real failures without inventing success."""
from __future__ import annotations

import copy

from vlm_tamp.actor_structured import feedback_record, INSTRUCTIONS
from vlm_tamp.compact_robot_interface import compact_response
from vlm_tamp.feedback_measured import measured_feedback, AFTER_CLOSURE
from vlm_tamp.robot_contracts import operation_schemas


def closing_response(settled: bool = True) -> dict:
    """Represent a measured close with legacy boilerplate still present in archived records."""
    completion={'settled':settled,'executed_controls':14,'retention':'not_tested',
                'maximum_finger_speed_m_s':.0008 if settled else .02,
                'controller_reference_stopped_changing':True}
    return {'ok':True,'result':{
        'task_success':False,'episode_done':False,
        'grasp_execution':{'approach_reached':True,'closure_completed':settled,
            'executed_close_controls':14,'mechanical_completion':completion,'object_retention':'unverified'},
        'execution':{'issued_commands':14,'gripper_completion':completion,
                     'termination_reason':'gripper_settled' if settled else 'gripper_unsettled'},
        'grasp_evidence':{'pad_gap_mm':3.1,'status':'grasp_state_unknown',
            'contact_sensing_available':False,'physical_contacts_observed':None,
            'interpretation':'Contact sensing is unavailable; retention unknown.'},
        'object_model':{'object_id':'object1','pose_status':'observed','pose_current':True,
            'reference_point_world_m':[.1,.2,.9],'attachment_verified':False,'semantic_identity_verified':False}}}


def test_settled_close_has_an_actionable_result_without_certification_boilerplate() -> None:
    """Closing supports the next motion; it does not manufacture successful object transport."""
    raw=closing_response();original=copy.deepcopy(raw)
    shown=compact_response(raw)
    assert shown['feedback']['message']=='Fingers settled.'
    assert shown['feedback']['next_action']==AFTER_CLOSURE
    assert shown['feedback']['pad_gap_mm']==3.1
    assert shown['result']['task_success'] is False
    assert shown['result']['grasp_execution']['closure_completed'] is True
    assert 'object_retention' not in shown['result']['grasp_execution']
    assert shown['result']['grasp_evidence']=={'pad_gap_mm':3.1,'status':'gap_measured'}
    assert 'attachment_verified' not in shown['result']['object_model']
    assert shown['result']['object_model']['reference_point_world_m']==[.1,.2,.9]
    assert raw==original


def test_moving_fingers_do_not_receive_a_lift_instruction() -> None:
    """The measured unfinished actuator state remains an explicit action failure."""
    shown=compact_response(closing_response(False))
    assert shown['feedback']['action_status']=='closing_incomplete'
    assert shown['feedback']['maximum_finger_speed_m_s']==.02
    assert shown['feedback']['stop_reason']=='gripper_unsettled'
    assert shown['feedback']['next_action']=='Continue closing before moving the arm.'
    assert not shown['result']['grasp_execution']['closure_completed']


def test_legacy_fixed_interval_is_not_relabelled_as_measured_settling() -> None:
    """Old logs certified command counts only; presentation must not invent a measurement."""
    raw={'ok':True,'result':{'grasp_execution':{'closure_completed':True,
        'executed_close_controls':8,'object_retention':'unverified'},
        'stop_reason':'closure_completed','task_success':False}}
    original=copy.deepcopy(raw)
    shown=compact_response(raw)
    assert shown['feedback']['action_status']=='closing_commands_sent'
    assert 'closure_completed' not in shown['result']['grasp_execution']
    assert shown['result']['grasp_execution']['closing_interval_completed'] is True
    assert shown['result']['stop_reason']=='closing_commands_sent'
    assert raw==original


def test_native_terminal_success_does_not_request_further_motion() -> None:
    """Task termination remains independent of the positive closing message."""
    response=closing_response();response['result'].update(task_success=True,episode_done=True)
    shown=compact_response(response)
    assert shown['result']['task_success'] is True
    assert 'next_action' not in shown['feedback']


def test_blocked_approach_reports_the_actual_errors_and_does_not_tell_the_agent_to_lift() -> None:
    """Useful failure feedback includes position and orientation, not a generic warning."""
    goal={'position_error_m':.0054,'orientation_error_rad':.01,'position_tolerance_m':.004,
          'orientation_tolerance_rad':.05,'orientation_constrained':True,'reference_path_completed':False}
    raw={'ok':True,'result':{'grasp_execution':{'approach_reached':False,'closure_issued':False},
        'execution':{'arrival':{'goal':goal}},'stop_reason':'approach_not_reached','task_success':False}}
    shown=compact_response(raw)
    assert shown['feedback']['goal']==goal
    assert shown['feedback']['stop_reason']=='approach_not_reached'
    assert 'next_action' not in shown['feedback']


def test_actual_negative_evidence_zero_motion_and_failure_reasons_are_preserved() -> None:
    """Never remove real bad news to make a result sound positive."""
    observed={'accepted':False,'task_success':False,'goal_reached':False,
        'object_retention':'lost','physical_contacts_observed':False,
        'object_motion':{'accepted':False,'reason':'insufficient_visible_matches','inliers':2},
        'motion_effect':{'status':'selected_region_stationary_despite_hand_motion',
                        'hand_rotation_deg':6.,'observed_region_rotation_deg':0.},
        'payload_collision_checked':False,'attachment_model':'rigid'}
    assert measured_feedback(observed)==observed


def test_budget_rejection_keeps_the_numbers_that_explain_it() -> None:
    """A finite search failure exposes available controls, required phases and solver work."""
    diagnostic={'status':'object_transfer_candidates_exhausted','available_control_steps':14,
        'hold_steps':6,'release_steps':20,'budget_seconds':60,'elapsed_seconds':4.,
        'budget_rejected_prefixes':8,'ik_queries':1,'path_queries':8,
        'search_deadline_reached':False,'physical_motion_issued':False,'physical_infeasibility_proved':False}
    shown=compact_response({'ok':False,'error':'No complete transfer fits','planning_diagnostic':diagnostic})
    assert shown['planning_diagnostic']==diagnostic
    assert shown['ok'] is False and shown['error']=='No complete transfer fits'


def test_visual_tools_and_memory_notes_use_the_same_policy_presentation() -> None:
    """The alternate visual-tool path and retained records cannot reintroduce the old labels."""
    raw={'ok':True,'result':{'semantic_identity_verified':False,'point_world_m':[1,2,3]}}
    shown=feedback_record(raw,'scene_views')
    assert shown['result']=={'point_world_m':[1,2,3]}
    note={'learner_note_unverified':'My prior plan was to lift.'}
    assert measured_feedback(note)=={'learner_note':'My prior plan was to lift.'}
    assert note=={'learner_note_unverified':'My prior plan was to lift.'}


def test_prompt_and_help_match_measured_closure_and_supported_parameters() -> None:
    """Instructions teach the real tool sequence without adding a task-specific recipe."""
    assert 'holds the arm until the fingers settle' in INSTRUCTIONS
    assert 'using\nthe planned close_steps' not in INSTRUCTIONS
    schemas={name: item['function'] for name,item in operation_schemas().items()}
    assert 'lift or continue the transfer' in schemas['grasp_object']['description']
    assert 'max_candidates limits selection within this pool' in schemas['plan_grasp']['description']
    assert 'episode_done=false' in schemas['finish']['description']
