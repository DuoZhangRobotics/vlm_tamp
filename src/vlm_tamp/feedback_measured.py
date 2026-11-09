"""Policy-facing action facts, without repeated unmeasured certification flags."""

from __future__ import annotations

from typing import Any


# These fields are fixed negative assertions in the current tools, not results
# of a sensor test. Actual outcomes (accepted, reached, task_success, etc.) stay.
UNMEASURED_ASSERTIONS = frozenset({
    'attachment_verified', 'physical_attachment_verified', 'physical_grasp_verified',
    'semantic_identity_verified', 'physical_contact_verified', 'task_success_predicted',
    'task_direction_verified', 'task_direction_verified_by_motion_fit',
    'collision_free_certificate', 'environment_collision_free_certificate',
    'continuous_collision_certificate', 'contact_pair_penetration_certificate',
    'calibrated_accuracy_guarantee',
})
AFTER_CLOSURE = ('Proceed with the planned motion. For picking, lift or transfer and observe '
                 'whether the object follows; choose another grasp if it stays behind or slips.')


def measured_feedback(value: Any) -> Any:
    """Copy JSON feedback, keeping measurements, failures, assumptions and provenance.

    Only known default assertions are removed. A measured zero displacement,
    rejected plan, failed tracking fit, lost engagement or false native success
    must survive unchanged. Raw service records are never mutated.
    """
    if isinstance(value, list):
        return [measured_feedback(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in UNMEASURED_ASSERTIONS and (item is False or item is None):
            continue
        if key in {'retention', 'object_retention'} and isinstance(item, str) and item in {
            'unverified', 'not_tested', 'unknown',
        }:
            continue
        if key == 'physical_contacts_observed' and item is None:
            continue
        if key == 'contact_sensing_available' and item is False:
            continue
        if key == 'task_direction_status' and item == 'unverified':
            continue
        if key == 'learner_note_unverified':
            if 'learner_note' not in value:
                result['learner_note'] = measured_feedback(item)
            continue
        if key in {'gripper_completion', 'mechanical_completion'} and isinstance(item, dict) and item.get('settled') is True:
            result[key] = {field: item[field] for field in (
                'settled', 'executed_controls', 'additional_controls', 'minimum_interval_completed',
                'control_step') if field in item}
            continue
        result[key] = measured_feedback(item)
    if 'pad_gap_mm' in result:
        if result.get('status') == 'grasp_state_unknown':
            result['status'] = 'gap_measured'
        if result.get('status') == 'near_closed_without_bar_width_support':
            result['status'] = 'gap_smaller_than_selected_handle'
        if result.get('status') == 'gap_measured':
            result.pop('interpretation', None)
        for key in ('previous_pad_gap_mm', 'selected_bar_diameter_mm', 'gap_to_selected_bar_ratio'):
            if result.get(key) is None:
                result.pop(key, None)
    grasp = result.get('grasp_execution')
    if isinstance(grasp, dict) and 'mechanical_completion' not in grasp and grasp.get('closure_completed') is True:
        # In archived fixed-duration replies this flag meant only "commands sent".
        grasp['closing_interval_completed'] = grasp.pop('closure_completed')
        for record in (grasp, result, result.get('execution', {})):
            for key in ('stop_reason', 'termination_reason'):
                if record.get(key) == 'closure_completed':
                    record[key] = 'closing_commands_sent'
    return result


def grasp_action_feedback(value: dict) -> dict:
    """Present closure and approach results as actions, with an executable next step."""
    grasp = value.get('grasp_execution', {})
    execution = value.get('execution', {})
    closing = grasp.get('mechanical_completion') or execution.get('gripper_completion')
    if not grasp and not closing:
        return {}
    if grasp.get('approach_reached') is False:
        goal = execution.get('arrival', {}).get('goal', {})
        return {'action_status': 'approach_not_reached',
                'message': 'Approach stopped before arrival. Closing was not issued.',
                'goal': {key: goal[key] for key in (
                    'position_error_m', 'orientation_error_rad', 'contact_normal_error_rad',
                    'position_tolerance_m', 'orientation_tolerance_rad',
                    'orientation_constrained', 'reference_path_completed') if key in goal},
                'stop_reason': value.get('stop_reason', execution.get('termination_reason'))}
    if not closing:
        # Historical fixed-duration records did not measure mechanical completion.
        return {'action_status': 'closing_commands_sent',
                'executed_close_controls': grasp.get('executed_close_controls', 0)}
    settled = bool(closing['settled'])
    result = {'action_status': 'gripper_settled' if settled else 'closing_incomplete',
              'message': 'Fingers settled.' if settled else 'Closing is still in progress.',
              'executed_close_controls': grasp.get('executed_close_controls', closing.get('executed_controls'))}
    gap = value.get('grasp_evidence', {}).get('pad_gap_mm')
    if gap is not None:
        result['pad_gap_mm'] = gap
    if not settled:
        result.update({key: closing[key] for key in (
            'maximum_finger_travel_m', 'maximum_finger_speed_m_s',
            'controller_reference_stopped_changing') if key in closing})
        result['stop_reason'] = value.get('stop_reason', execution.get('termination_reason'))
    if not value.get('episode_done') and not value.get('task_success'):
        result['next_action'] = AFTER_CLOSURE if settled else 'Continue closing before moving the arm.'
    return result
