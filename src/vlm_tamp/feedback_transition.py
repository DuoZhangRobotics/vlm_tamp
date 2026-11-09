"""Presentation-only motion receipts assembled from already public tool evidence."""

from __future__ import annotations

import copy
import json
from typing import Any
from .feedback_measured import measured_feedback


def selected(value: dict, names: tuple[str, ...]) -> dict:
    """Copy available fields without filling missing measurements with zero."""
    return {name: copy.deepcopy(value[name]) for name in names if name in value}


def arrival_record(arrival: dict) -> dict:
    """Keep final-goal arrival separate from tracking the last issued reference."""
    return {
        'goal': selected(arrival.get('goal', {}), (
            'reached', 'position_error_m', 'orientation_error_rad', 'contact_normal_error_rad',
            'tracked_point', 'orientation_constrained', 'orientation_constraint',
            'reference_path_completed')),
        'last_issued_reference': selected(arrival.get('last_reference', {}), (
            'reached', 'position_error_m', 'orientation_error_rad', 'tracked_point', 'criterion')),
    }


def transition_record(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize one executed interaction, preserving unknown contact and object outcomes."""
    receipt = record['receipt']
    value = receipt.get('result', {})
    feedback = receipt.get('feedback', {})
    execution = value.get('execution', {})
    comparison = value.get('visual_comparison', {})
    phases = []
    comparisons = value.get('phase_visual_comparisons', [])
    for stage in value.get('stages', []):
        matching = [item for item in comparisons
                    if item.get('stage') == stage.get('stage')
                    and item.get('end_control_step') == stage.get('control_steps')]
        phase = {
            'phase': stage['stage'],
            'end_control': stage.get('control_steps'),
            'arrival': arrival_record(stage.get('arrival', stage.get('execution', {}).get('arrival', {}))),
            'hand_motion': selected(stage, (
                'requested_motion_m', 'requested_contact_speed_m_s', 'measured_contact_speed',
                'physical_motion', 'robot_reference_tracking', 'observed_contact_progress_m')),
            'object_motion': copy.deepcopy(stage.get('object_motion')),
            'engagement_evidence': copy.deepcopy(stage.get('engagement')),
            **selected(stage, ('scene_update', 'material_point_offset_m', 'control_budget_truncated', 'grasp_evidence')),
        }
        if matching:
            phase['start_control'] = matching[0]['start_control_step']
            phase['views'] = selected(matching[0], (
                'before_observation_id', 'after_observation_id', 'cameras',
                'moving_cameras', 'comparison_image'))
        phases.append(phase)
    result = {
        **selected(record, ('type', 'decision', 'operation', 'expected_effect',
                            'learner_note', 'learner_note_unverified', 'before_control', 'after_control',
                            'before_observation', 'after_observation', 'before_images', 'after_images')),
        'command': json.loads(record['arguments_json']),
        'receipt': {
            'command_accepted': receipt.get('ok'),
            'execution': {
                'scope': 'whole_tool_call; requested stroke and incidental approach effects differ',
                'issued_controls': record['after_control'] - record['before_control'],
                **selected(execution, ('termination_reason', 'all_commands_issued',
                                       'planned_commands', 'issued_commands', 'nominal_path_commands', 'arrival_scope')),
                **selected(value, ('stop_reason', 'status', 'robot_motion_executed')),
                'arrival': arrival_record(execution.get('arrival', {})),
            },
            'requested_and_measured_hand_motion': selected(feedback, (
                'requested_direction_world', 'requested_hand_distance_m',
                'requested_rotation_direction', 'rotation_view_convention',
                'hand_reference_reached', 'measured_hand_travel_along_request_m')),
            'contact_evidence': {
                **selected(value, ('engagement_status', 'physical_contact_verified', 'grasp_evidence')),
            },
            'selected_contact': selected(value.get('contact', {}), (
                'camera', 'region_xyxy', 'contact_pixel_xy', 'point_world_m', 'contact_name',
                'contact_local_m', 'contact_constraint', 'surface_normal_world',
                'normal_toward_camera_world', 'geometry_observed_at_control_step', 'selected_parts')),
            'contact_effect': copy.deepcopy(feedback.get('contact_effect')),
            'object_motion': {
                **selected(value, ('object_motion_status', 'part_motion_evidence',
                                   'observed_progress', 'progress_units', 'motion_measurement_scope',
                                   'object_motion_tracker_class', 'task_direction_verified',
                                   'scene_pose_observation', 'scene_pose_update', 'motion_effect')),
            },
            'task_success': value.get('task_success'),
            'observation_after': selected(value.get('observation', {}), (
                'observation_id', 'control_steps', 'remaining_control_steps', 'eef_pose_world_xyzw',
                'gripper_qpos', 'images', 'camera_calibration', 'image_coordinate_system', 'image_sizes')),
            'views': selected(comparison, ('before_observation_id', 'after_observation_id',
                                            'cameras', 'moving_cameras', 'comparison_image', 'video_lookup')),
            'phases': phases,
            'earlier_phases_omitted_by_original_reply': max(0, value.get('stage_count', len(phases)) - len(phases)),
            'continuation': copy.deepcopy(value.get('continuation')),
            **selected(value, ('grasp_execution', 'object_point_goal', 'object_model', 'object_pose_update',
                               'planning_object_models', 'release_issued', 'release_complete', 'prepared_transfer',
                               'diagnostic', 'planning_diagnostic', 'remaining_skill_seconds',
                               'next_geometry_interfaces', 'scene_refresh_required', 'contact_handoff')),
            **selected(receipt, ('error', 'error_type', 'execution_status')),
        },
    }
    return measured_feedback(result)


def compact_transition_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace executed receipts, keeping image order, non-motion feedback and tools identical."""
    result = copy.deepcopy(payload)
    events = result['memory']['events']
    for index, original in enumerate(payload['memory']['events']):
        if original.get('type') != 'executed_transition':
            continue
        events[index] = transition_record(original)
        if original['receipt'] == payload['latest_feedback']:
            result['latest_feedback'] = {'executed_transition_decision': original['decision'],
                                        'location': 'memory.events',
                                        'meaning': 'Latest executed receipt is included once in that transition.'}
    result['feedback_format'] = 'compact_transition_v1'
    return result
