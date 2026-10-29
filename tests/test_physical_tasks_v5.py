"""Regression tests for mask ambiguity, complete-task admission and feedback."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.rekep_libero.drawer_feedback_v5 import measure_progress, retained_progress
from vlm_tamp.rekep_libero.physical_tasks_v4 import cached_scene
from vlm_tamp.rekep_libero.physical_tasks_v5 import choose_grasp
from vlm_tamp.rekep_libero.task_geometry_v4 import bowl_spec as legacy_bowl_spec
from vlm_tamp.rekep_libero.task_geometry_v5 import _bar_clusters, bowl_spec, compile_scene
from vlm_tamp.rekep_libero.task_ik_v5 import TaskIK

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/physical_tasks_v5.json').read_text())


def test_broad_overlapping_mask_does_not_erase_plate_roles():
    source_mask = np.zeros((64, 64), bool)
    source_mask[5:15, 5:15] = True
    plate = np.zeros_like(source_mask)
    plate[25:35, 25:35] = True
    broad = np.zeros_like(source_mask)
    broad[5:35, 5:35] = True
    yy, xx = np.indices(source_mask.shape)
    world = np.dstack([xx * .002, yy * .002, np.full_like(xx, .91, float)])
    definition = ast.parse('def f(ee, keypoints):\n return keypoints[0] - keypoints[1] + keypoints[2]').body[0]
    program = SimpleNamespace(grasp_keypoints=[0], stages=[None, SimpleNamespace(
        subgoal_constraints=[SimpleNamespace(definition=definition)])])
    bundle = {'main_camera': 'front', 'cameras': {'front': {'depth': np.ones(source_mask.shape),
               'world_points': world}}, 'masks': [source_mask, plate, broad],
              'pixels_yx': np.array([[8, 8], [28, 28], [10, 10]]),
              'points_world': world[[8, 28, 10], [8, 28, 10]]}
    cloud = world[source_mask]
    with pytest.raises(ValueError, match='separate target'):
        legacy_bowl_spec(bundle, program, cloud)
    spec, _, _, selected, _ = bowl_spec(bundle, program, cloud, source_mask, CONFIG)
    assert spec['target_keypoints'] == [1]
    np.testing.assert_array_equal(selected, plate)


def test_middle_relation_requires_three_separated_supported_bars():
    generator = np.random.default_rng(0)
    bars = [np.column_stack([generator.uniform(-.04, .04, 150), generator.uniform(-.002, .002, 150),
                             generator.uniform(z - .005, z + .005, 150)]) for z in [.95, 1.02, 1.095]]
    cloud = np.vstack(bars)
    groups = _bar_clusters(cloud, np.arange(len(cloud)))
    assert abs(groups[1]['center'][2] - 1.02) < .001
    with pytest.raises(ValueError, match='three supported bars'):
        _bar_clusters(np.vstack(bars[:2]), np.arange(300))


@pytest.mark.parametrize('task,kind', [(8, 'bowl'), (0, 'drawer')])
@pytest.mark.parametrize('state', [1, 11, 21, 31, 41])
def test_all_cached_regions_compile_without_scene_identity_inputs(task, kind, state):
    source = ROOT / f'results/rekep-libero-v2/libero_goal-{task:02d}-state{state:03d}-seed0'
    observation = ROOT / f'results/physical-tasks-v4-eval/{kind}/goal{task:02d}-state{state:03d}/opened'
    if not observation.is_dir():
        pytest.skip('Cached project scenes are required for this integration regression')
    bundle, program, cloud, mask = cached_scene(source, observation)
    spec, selected, _, _, _ = compile_scene(bundle, program, cloud, mask, CONFIG)
    assert not spec['simulator_object_state_used']
    assert len(selected) >= 75
    if task == 0:
        assert abs(spec['source_anchor_world_m'][2] - spec['observed_handle_centers_world_m'][1][2]) < 1e-8
        assert spec['automatic_ordinal_role_correction'] == (state in [11, 21, 41])


def test_same_calibrated_pose_is_reachable_and_self_collision_is_respected():
    record = json.loads((ROOT / 'tests/fixtures/calibrated_panda.json').read_text())
    ik = TaskIK(record, CONFIG)
    solution = ik.solve(ik.chain.forward(ik.chain.reference_joints), ik.chain.reference_joints)
    assert solution is not None
    assert solution['position_error_m'] < 1e-8
    blocked = TaskIK(record, CONFIG, self_valid=lambda _q: False)
    assert blocked.solve(ik.chain.forward(ik.chain.reference_joints), ik.chain.reference_joints, seed_count=1) is None


def test_grasp_reachability_alone_does_not_admit_an_unreachable_task(tmp_path):
    class FakeIK:
        config = CONFIG

        def solve(self, pose, seed):
            # First grasp reaches the object, but cannot reach its job endpoint.
            if pose[2] > .5 and pose[0] < .01:
                return None
            return {'joints': [0.] * 7, 'pose_world_xyzw': list(pose),
                    'position_error_m': 0., 'orientation_error_rad': 0.}

        def path(self, poses, seed):
            return {'accepted': True, 'solutions': []}

        def approach(self, initial, grasp):
            return {'accepted': True, 'poses_world_xyzw': [grasp['pose_world_xyzw']], 'joint_postures': [grasp['joints']]}

    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = .015
    generated = {'poses_world_base': poses, 'scores': np.array([.9, .8]), 'retained_indices': [0, 1]}
    calibration = {'physical_finger_attach_z_m': .0584, 'graspgen_base_to_tcp': np.eye(4).tolist()}
    spec = {'task_type': 'pick_place', 'source_anchor_world_m': [0., 0., 0.],
            'goal': {'target_world_m': [0., 0., 1.]}, 'object_support_point_initial_world_m': [0., 0., 0.]}
    result = choose_grasp(generated, np.array([[0., 0., 0.], [.015, 0., 0.]]), spec,
                          calibration, FakeIK(), np.zeros(7), tmp_path)
    assert result['index'] == 1
    assert any(row['reason'] == 'complete_task_endpoint_unreachable' for row in result['rejected_candidates'])


def bar_observation(progress):
    generator = np.random.default_rng(2)
    reference = generator.uniform([-.04, -.001, 1.014], [.04, .001, 1.026], (900, 3))
    observed = reference + [0., progress, 0.]
    return reference, {'view': {'world_points': observed.reshape(30, 30, 3)}}


def test_feedback_measures_observed_travel_instead_of_commanded_travel():
    reference, cameras = bar_observation(.06)
    result = measure_progress(reference, cameras, [], [0., 1., 0.], .068, previous_progress=.035)
    assert result['tracking_valid']
    assert abs(result['observed_progress_m'] - .06) < .0005
    assert retained_progress(result, [.01, -.01])


def test_feedback_rejects_stationary_handle_despite_hand_movement():
    reference, cameras = bar_observation(.035)
    result = measure_progress(reference, cameras, [], [0., 1., 0.], .07, previous_progress=.035)
    assert result['tracking_valid']
    assert not retained_progress(result, [.01, -.01])
