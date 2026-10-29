"""Compact display must preserve decisions, provenance, and exact raw access."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vlm_tamp.compact_robot_interface import compact_response, install, main, present

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_replies_remain_exact_and_success_and_prediction_are_not_hidden(tmp_path):
    source = ROOT / 'results/reflexion-checkpoint-study-20261021/study-v2/episodes/trial01-reflection/tool-events.jsonl'
    if not source.exists():
        pytest.skip('Archived physical run unavailable')
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    for row in rows:
        raw = copy.deepcopy(row['response'])
        display = present(tmp_path, row['request']['operation'], row['request']['arguments'], raw)
        assert raw == row['response']
        assert json.loads((tmp_path / display['full_response']).read_text()) == raw
        assert display['ok'] == raw['ok']
        if not raw['ok']:
            assert display['error'] == raw['error']
        value = raw.get('result', {})
        for key in ('task_success', 'episode_done', 'termination_reason', 'plan_id'):
            if key in value:
                assert display['result'][key] == value[key]
        if 'scene_pose_update' in value:
            before, after = value['scene_pose_update'], display['result']['scene_pose_update']
            assert after['source'] == before['source']
            assert after['assumes_contact_retention'] == before['assumes_contact_retention']
            assert after['fit']['pose_accepted'] == before['fit']['pose_accepted']
    assert compact_response(rows[61]['response'])['result']['execution']['issued_commands'] == 6
    assert compact_response(rows[47]['response'])['result']['scene_pose_update']['fit']['reason'] == 'insufficient_moving_landmarks'


def test_camera_selection_preserves_arrays_and_access_to_all_views(tmp_path):
    value = {'ok': True, 'result': {'images': {name: name + '.png' for name in ['wrist', 'side', 'frontview', 'agentview']},
        'sensor_arrays': 'sensors.npz', 'joint_positions': [0.] * 7, 'control_steps': 17,
        'remaining_control_steps': 583, 'camera_calibration': 'calibration.json'}}
    display = present(tmp_path, 'observe', {}, value)
    assert list(display['result']['images']) == ['frontview', 'agentview']
    assert display['result']['available_camera_count'] == 4
    assert display['result']['sensor_arrays'] == 'sensors.npz'
    assert json.loads((tmp_path / display['full_response']).read_text()) == value
    assert compact_response({'ok': True, 'result': {'images': ['overlay.png']}})['result']['images'] == ['overlay.png']


def test_help_never_calls_service_and_full_mode_never_reissues_motion(tmp_path, capsys):
    document = '- execute(plan_id): one physical execution.\n- observe(): no motion.\n'
    install(tmp_path, document, attempts=5, initial_attempts=5, practice_mode=True)
    calls = []
    response = {'ok': True, 'result': {'task_success': True, 'execution': {'issued_commands': 6}, 'large': list(range(300))}}
    def fake(operation, **arguments):
        calls.append((operation, arguments))
        return response
    main(fake, tmp_path, ['help', 'execute'])
    assert 'one physical execution' in capsys.readouterr().out and not calls
    main(fake, tmp_path, ['--full', 'execute', '{"plan_id":"p01"}'])
    assert json.loads(capsys.readouterr().out) == response
    assert calls == [('execute', {'plan_id': 'p01'})]
    main(fake, tmp_path, ['execute', '{"plan_id":"p02"}'])
    display = json.loads(capsys.readouterr().out)
    assert len(calls) == 2 and display['result']['task_success'] is True
    assert json.loads((tmp_path / display['full_response']).read_text()) == response
    assert len(calls) == 2
    with pytest.raises(ValueError):
        main(fake, tmp_path, ['help', '../../private'])


def test_original_python_call_contract_and_single_request_brief(tmp_path, monkeypatch):
    prep = module_at('compact_prep_for_test', ROOT / 'scripts/prepare_subscription_pilot.py')
    install(tmp_path, '- observe(): current sensors.\n', attempts=5, initial_attempts=5, practice_mode=True)
    (tmp_path / 'robot.py').write_text(prep.CLIENT)
    (tmp_path / 'robot_data').mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    client = module_at('compact_client_for_test', tmp_path / 'robot.py')
    response = {'ok': True, 'result': {'joint_positions': [1] * 7, 'task_success': False, 'arbitrary': [1, 2, 3]}}
    calls = []
    def fake(operation, **arguments):
        calls.append(operation)
        return response
    monkeypatch.setattr(client, '_call', fake)
    assert client.call('observe') is response
    display = client.brief('observe')
    assert calls == ['observe', 'observe']
    assert json.loads((tmp_path / display['full_response']).read_text()) == response


def test_phase_instructions_and_real_budget_are_preserved():
    value = {'ok': True, 'result': {'granted_attempts': 5, 'maximum_attempts': 5,
        'attempts_started': 3, 'remaining_granted_attempts': 2,
        'completed_attempts': [{'attempt': 1, 'task_success': False}],
        'learning_phase': {'phase': 'efficiency', 'verified_baseline_attempts': [2, 3],
                           'instructions': 'Preserve the working method and test efficiency.'}}}
    assert compact_response(value)['result'] == value['result']


def test_arc_description_matches_the_unchanged_numeric_query(tmp_path, monkeypatch):
    service = module_at('compact_service_for_test', ROOT / 'scripts/serve_subscription_robot.py')
    robot = service.RobotService.__new__(service.RobotService)
    robot.contact_scene_kind = 'articulated'
    robot.config = {'grasp_constraint': 'point'}
    robot.spec = {'joint_origin_world_m': [.1, .2, 1.], 'joint_axis_world': [0., 0., -1.]}
    robot.state = {'metadata': {'joint_positions': [0.] * 7}}
    robot.epoch = 4
    robot.ik, robot.scene, robot.record = object(), object(), object()
    robot.folder = lambda label: tmp_path / label
    robot.keep_plan = lambda path, result, kind: {'plan_id': 'p1', 'result': result, 'kind': kind}
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return np.zeros((2, 7)), {'status': 'solved'}
    monkeypatch.setattr(service, 'plan_signed_arc', fake)
    monkeypatch.setattr(service, 'measured_robot_scene', lambda s:s.scene)
    response = robot.plan_arc(15, refined=True, grasp_constraint='jaw_axis')
    assert len(calls) == 1
    assert calls[0][1]['config']['grasp_constraint'] == 'jaw_axis'
    assert calls[0][0][1] is robot.scene
    assert calls[0][0][5] == pytest.approx(15)
    assert response['constraints']['hinge_world_m'] == robot.spec['joint_origin_world_m']
    assert response['constraints']['grasp_constraint'] == 'jaw_axis'
    assert response['constraints']['contact_retention_verified'] is False
    response['constraints']['hinge_world_m'][0] = 999
    assert robot.spec['joint_origin_world_m'][0] == .1


def test_new_practice_prompt_is_one_context_without_reviewer():
    runner = module_at('compact_practice_for_test', ROOT / 'scripts/run_compact_luna_practice.py')
    text = runner.prompt(5, 1800, 5400)
    assert 'one continuing session' in text and '5 attempts' in text
    assert 'no separate review agent' in text
    assert 'working-state.txt' in text and 'full_response' in text
    assert len(text.split()) < 450


def test_continuation_reminder_uses_the_registered_cap():
    from vlm_tamp.episode_persistence import practice_continuation_prompt
    text = practice_continuation_prompt({'initial_attempts': 2, 'maximum_attempts': 5})
    assert 'Initially 2' in text and 'hard cap is 5' in text
    assert 'twenty' not in text and '20' not in text
def test_contact_skill_feedback_is_not_overwritten_by_sequence_formatter():
    from vlm_tamp.compact_robot_interface import compact_response
    stage={'stage':'push-stroke-1','control_steps':125,
        'object_motion':{'accepted':True,'inliers':15,'rms_m':.001},
        'observed_contact_progress_m':.008,
        'engagement':{'status':'object_followed_contact'},
        'requested_contact_speed_m_s':.03,
        'measured_contact_speed':{'mean_m_s':.026,'peak_m_s':.029}}
    result=compact_response({'ok':True,'result':{'skill':'push','stages':[stage]}})['result']
    assert result['stages'][0]['observed_contact_progress_m']==.008
    assert result['stages'][0]['object_motion']['inliers']==15
    assert result['stages'][0]['engagement']['status']=='object_followed_contact'
    assert result['feedback_count']==1


def test_surface_measurement_values_survive_compaction():
    from vlm_tamp.compact_robot_interface import compact_response
    value={'measurement_id':'m1','kind':'surface_geometry','supported':True,
        'values':{'plane_angle_deg':45.},'control_steps':90}
    result=compact_response({'ok':True,'result':value})['result']
    assert result['measurement_id']=='m1' and result['values']['plane_angle_deg']==45.
def test_compact_plan_keeps_executor_stage_contract_and_search_timing():
    from vlm_tamp.compact_robot_interface import compact_response
    value={'plan_id':'p1','kind':'approach','expected_executor':'execute',
        'grasp_stage_contract':{'approach_gripper_command':-1,'close_after_arrival':True},
        'search_timing_seconds':{'ik_generation_and_queue':1.,'approach_queries':2.}}
    result=compact_response({'ok':True,'result':value})['result']
    for key in value:assert result[key]==value[key]
