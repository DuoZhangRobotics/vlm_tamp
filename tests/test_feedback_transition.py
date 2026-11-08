"""Compact receipts preserve measured outcomes without manufacturing task evidence."""

import copy
import json

from vlm_tamp.feedback_transition import compact_transition_context, transition_record


def motion_record() -> dict:
    """Describe a reached hand target with unobservable object motion."""
    return {
        'type': 'executed_transition', 'decision': 4, 'operation': 'push',
        'arguments_json': json.dumps({'direction_world': [-1, 0, 0], 'distance_m': .08}),
        'expected_effect': 'move the selected part toward its goal',
        'learner_note_unverified': 'direction is a hypothesis',
        'before_control': 60, 'after_control': 130,
        'before_images': {'agentview': 'before.png', 'wrist': 'before-wrist.png'},
        'after_images': {'agentview': 'after.png', 'wrist': 'after-wrist.png'},
        'receipt': {'ok': True, 'feedback': {
            'requested_direction_world': [-1, 0, 0], 'requested_hand_distance_m': .08,
            'hand_reference_reached': True, 'measured_hand_travel_along_request_m': .0772},
            'result': {'task_success': False, 'engagement_status': 'object_motion_unresolved',
                       'physical_contact_verified': False, 'object_motion_status': 'unresolved',
                       'part_motion_evidence': {'latest': {'accepted': False, 'pairs': 0,
                                                          'reason': 'insufficient_visible_matches'}},
                       'stage_count': 1,
                       'stages': [{'stage': 'push-stroke-1', 'control_steps': 130,
                                   'arrival': {'goal': {'reached': True, 'position_error_m': .0028}},
                                   'object_motion': {'accepted': False, 'pairs': 0}}],
                       'phase_visual_comparisons': [{'stage': 'push-stroke-1',
                           'start_control_step': 90, 'end_control_step': 130,
                           'before_observation_id': 3, 'after_observation_id': 4,
                           'moving_cameras': ['wrist'], 'comparison_image': 'stroke.png'}]}}
    }


def test_hand_success_does_not_become_contact_object_or_task_success() -> None:
    """A truthful unknown must survive even when the robot reached its reference."""
    result = transition_record(motion_record())['receipt']
    assert result['command_accepted'] is True
    assert result['requested_and_measured_hand_motion']['hand_reference_reached'] is True
    assert result['contact_evidence']['physical_contact_verified'] is False
    assert result['object_motion']['object_motion_status'] == 'unresolved'
    assert result['object_motion']['part_motion_evidence']['latest']['pairs'] == 0
    assert 'observed_progress' not in result['object_motion']
    assert result['task_success'] is False
    assert result['phases'][0]['start_control'] == 90  # Not the whole-call start at 60.
    assert result['phases'][0]['arrival']['goal']['position_error_m'] == .0028


def test_goal_error_is_distinct_from_tracking_last_issued_reference() -> None:
    """Stopping well short of a goal cannot be hidden by a smaller intermediate error."""
    record = motion_record()
    record['receipt']['result']['execution'] = {'termination_reason': 'reference_not_reached',
        'arrival': {'goal': {'reached': False, 'position_error_m': .15},
                    'last_reference': {'reached': False, 'position_error_m': .019}}}
    result = transition_record(record)['receipt']['execution']
    assert result['arrival']['goal']['position_error_m'] == .15
    assert result['arrival']['last_issued_reference']['position_error_m'] == .019
    assert result['termination_reason'] == 'reference_not_reached'


def test_measured_zero_and_measured_transform_remain_observations() -> None:
    """Measured zero is retained, while a valid transform keeps its frame and provenance."""
    record = motion_record()
    value = record['receipt']['result']
    value.update(observed_progress=0.0, progress_units='metres', object_motion_status='measured')
    value['part_motion_evidence'] = {'latest': {'accepted': True, 'reference_control_step': 90,
        'control_step': 130, 'transform_world': [[1, 0, 0, 0], [0, 1, 0, .02],
                                               [0, 0, 1, 0], [0, 0, 0, 1]]}}
    result = transition_record(record)['receipt']['object_motion']
    assert result['observed_progress'] == 0.0
    assert result['part_motion_evidence'] == value['part_motion_evidence']
    assert result['object_motion_status'] == 'measured'


def test_generic_execution_scene_motion_keeps_its_measurement_provenance() -> None:
    """Non-push tools can report object motion through the shared scene-pose fields."""
    record = motion_record()
    observation = {'source': 'measured_rgbd_motion', 'fit': {'accepted': True, 'rotation_rad': .1},
                   'reference_control_step': 130, 'assumes_contact_retention': False}
    record['receipt']['result']['scene_pose_observation'] = observation
    result = transition_record(record)['receipt']['object_motion']
    assert result['scene_pose_observation'] == observation


def test_context_keeps_images_memory_selection_and_nonmotion_feedback_unchanged() -> None:
    """Only executed receipts change presentation; no evidence is synthesized or reordered."""
    record = motion_record()
    payload = {'task': 'fixture', 'tools': {'push': 'contract'}, 'current_observation': {'control_steps': 130},
               'images': [{'path': 'before.png', 'sha256': 'source'}, {'path': 'after.png', 'sha256': 'end'}],
               'memory': {'events': [{'type': 'TRAJECTORY_GAP'}, record],
                          'recent_nonmotion_feedback': [], 'slot_limit': 8},
               'latest_feedback': copy.deepcopy(record['receipt'])}
    saved = copy.deepcopy(payload)
    result = compact_transition_context(payload)
    assert payload == saved
    for key in ('task', 'tools', 'current_observation', 'images'):
        assert result[key] == saved[key]
    assert result['memory']['events'][0] == saved['memory']['events'][0]
    assert result['latest_feedback']['executed_transition_decision'] == 4
    help_reply = {'ok': True, 'result': {'name': 'push', 'parameters': {}}}
    payload['latest_feedback'] = help_reply
    payload['memory']['recent_nonmotion_feedback'] = [{'operation': 'help', 'receipt': help_reply}]
    result = compact_transition_context(payload)
    assert result['latest_feedback'] == help_reply
    assert result['memory']['recent_nonmotion_feedback'] == payload['memory']['recent_nonmotion_feedback']


def test_omitted_phases_and_unavailable_measurements_remain_explicit() -> None:
    """A shortened source receipt cannot become a claimed complete history."""
    record = motion_record()
    record['receipt']['result']['stage_count'] = 5
    record['receipt']['result'].pop('phase_visual_comparisons')
    record['receipt']['result'].pop('task_success')
    result = transition_record(record)['receipt']
    assert result['earlier_phases_omitted_by_original_reply'] == 4
    assert 'start_control' not in result['phases'][0]
    assert result['task_success'] is None
