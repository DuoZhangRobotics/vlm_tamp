import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from vlm_tamp import coupled_storage as storage


def snapshot(root, name, step):
    folder = root/name
    folder.mkdir()
    (folder/'metadata.json').write_text(json.dumps({'control_steps': step}))
    np.savez_compressed(folder/'obs.npz', depth=np.ones((4, 4)))
    np.savez_compressed(folder/'robot_visual_meshes.npz', vertices=np.zeros((3, 3)))
    Image.new('RGB', (16, 16)).save(folder/'arm_side.png')
    return folder


def episode(root):
    folder = root/'episode'
    folder.mkdir()
    (folder/'summary.json').write_text('{"official_task_success": false}')
    return folder


def test_cleanup_keeps_current_inputs_and_every_outcome_but_prunes_old_sensor_copies(tmp_path):
    folder = episode(tmp_path)
    snapshot(folder, 'initial', 0)
    snapshot(folder, 'opened', 60)
    snapshot(folder, 'observation-00', 90)
    snapshot(folder, 'final', 100)
    (folder/'trajectory.npy').write_bytes(b'necessary trajectory')
    (folder/'trusted-local-input.pkl').write_bytes(b'temporary worker input')
    (folder/'initial/actions.json').write_text('[{"evaluation_task_success": false}]')
    out = storage.compact_episode(folder, allowed_root=tmp_path)
    assert out['raw_observations_retained'] == ['final', 'opened']
    assert not (folder/'initial/obs.npz').exists()
    assert not (folder/'observation-00/obs.npz').exists()
    assert not (folder/'trusted-local-input.pkl').exists()
    assert (folder/'trajectory.npy').read_bytes() == b'necessary trajectory'
    assert storage.read_json(folder/'initial/actions.json') == [{'evaluation_task_success': False}]
    assert storage.compact_episode(folder, allowed_root=tmp_path)['bytes_reclaimed_this_pass'] == 0


def test_lossless_large_diagnostic_archive_can_be_read_by_reporter(tmp_path):
    folder = episode(tmp_path)
    plan = folder/'plan-00'
    plan.mkdir()
    path = plan/'request.json'
    value = {'numeric_evidence': list(range(200000))}
    path.write_text(json.dumps(value))
    storage.compact_episode(folder, allowed_root=tmp_path)
    assert not path.exists()
    assert storage.read_json(path) == value
    evidence = storage.read_json(folder/'retention.json')['archived'][0]
    assert evidence['sha256'] and evidence['after_bytes'] < evidence['before_bytes']


def test_unfinished_or_outside_root_is_rejected_before_any_deletion(tmp_path):
    folder = tmp_path/'unfinished'
    folder.mkdir()
    original = snapshot(folder, 'initial', 0)/'obs.npz'
    with pytest.raises(ValueError, match='unfinished'):
        storage.compact_episode(folder, allowed_root=tmp_path)
    assert original.exists()
    with pytest.raises(ValueError, match='beneath'):
        storage.compact_episode(folder, allowed_root=folder)
    assert original.exists()


def test_encoder_failure_keeps_all_raw_evidence(tmp_path):
    folder = episode(tmp_path)
    source = snapshot(folder, 'initial', 0)/'obs.npz'
    with patch.object(storage, '_movie_index', side_effect=RuntimeError('encoder unavailable')):
        with pytest.raises(RuntimeError, match='encoder unavailable'):
            storage.compact_episode(folder, allowed_root=tmp_path)
    assert source.exists()


def test_video_survives_cleanup_with_exact_control_step_index(tmp_path):
    folder = episode(tmp_path)
    initial = snapshot(folder, 'initial', 0)
    frames = initial/'frames'
    frames.mkdir()
    for step, color in [(0, 'red'), (5, 'green'), (10, 'blue')]:
        Image.new('RGB', (16, 16), color).save(frames/f'step{step:04d}.jpg')
    storage.compact_episode(folder, allowed_root=tmp_path)
    assert not list(frames.glob('*.jpg'))
    recovered = list(storage.video_frames(folder))
    assert [step for step, _ in recovered] == [0, 5, 10]
    assert recovered[0][1].getpixel((8, 8))[0] > 200
    assert recovered[-1][1].getpixel((8, 8))[2] > 200


def test_cleanup_never_follows_a_link_to_another_run(tmp_path):
    folder = episode(tmp_path)
    outside = tmp_path/'separate-run'
    outside.mkdir()
    original = snapshot(outside, 'initial', 0)/'obs.npz'
    (folder/'linked-run').symlink_to(outside, target_is_directory=True)
    storage.compact_episode(folder, allowed_root=tmp_path)
    assert original.exists()


def planning_scene():
    from vlm_tamp.coupled_scene import Scene
    spec = {'joint_origin_world_m': [0., 0., 0.], 'joint_axis_world': [0., 0., 1.],
            'front_plane': {'point_world_m': [1., 0., 1.]},
            'panel_normal_world': [1., 0., 0.], 'motion': 'open'}
    return Scene.from_points([[.1, .2, .3]], [[1., 0., 1.]], [[1., 0., 1.]], spec)


def test_deduplicated_collision_arrays_are_exact_and_readable_without_pickle(tmp_path):
    folder = episode(tmp_path)
    (folder/'robot-chain.json').write_text('{}')
    scene = planning_scene()
    for name in ['first', 'second']:
        output = folder/name
        output.mkdir()
        storage.preserve_scene(scene, output)
    first = storage.read_json(folder/'first/scene-reference.json')
    second = storage.read_json(folder/'second/scene-reference.json')
    assert first['arrays_path'] == second['arrays_path']
    assert len(list((folder/'collision-scenes').glob('*.npz'))) == 1
    with np.load(first['arrays_path'], allow_pickle=False) as data:
        assert np.array_equal(data['static_points'], scene.static_points)
        assert np.array_equal(data['moving_reference'], scene.moving_reference)


def test_timed_out_planner_deletes_pickle_but_keeps_collision_evidence(tmp_path):
    from vlm_tamp.coupled_pipeline import isolated_plan
    folder = tmp_path/'attempt'
    with patch('vlm_tamp.coupled_pipeline.subprocess.run', side_effect=subprocess.TimeoutExpired('worker', 1)):
        with pytest.raises(subprocess.TimeoutExpired):
            isolated_plan(SimpleNamespace(), np.zeros(8), np.ones(8), planning_scene(), folder, lambda *a: None)
    assert not (folder/'trusted-local-input.pkl').exists()
    assert (folder/'scene-reference.json').exists()
