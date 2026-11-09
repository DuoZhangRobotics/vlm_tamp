"""Native carry constraints must preserve the requested axis and support half-space."""

from pathlib import Path

import numpy as np
import pytest

from vlm_tamp.coupled_scene import Scene
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.mcvamp_contact_numerical import NativeScene, clear_native_cache
from vlm_tamp.rekep_libero.geometry import pose_matrix


@pytest.fixture
def carry_data() -> tuple:
    """Read a recorded measured grasp and the subsequent unconstrained trajectory."""
    root = Path(__file__).resolve().parents[1]/'results/object-state-integration-20261108/physical-measured-fixed-grasp'
    if not (root/'summary.json').exists():
        pytest.skip('Recorded carry fixture unavailable')
    summary = read_json(root/'summary.json')
    episode = root/'trial/attempt01'
    record = read_json(episode/'robot-chain.json')
    metadata = read_json(episode/'006-observation/metadata.json.gz')
    point_plan = next(row for row in summary['requests'] if row['operation']=='plan_object_point')
    binding = point_plan['response']['result']['object_point_goal']
    axis = np.asarray(summary['observed_object_motion']['transform'])[:3, :3] @ [0, 0, 1]
    pose = pose_matrix(metadata['eef_pose_world_xyzw'])
    spec = {'joint_origin_world_m':[0,0,0],'joint_axis_world':[0,0,1],
            'front_plane':{'point_world_m':[0,0,0]},'panel_normal_world':[0,1,0]}
    scene = Scene.from_points([], [], [binding['target_point_world_m']], spec)
    actions = read_json(episode/'initial/actions.json')
    return scene, record, metadata, binding, pose[:3,:3].T @ axis, axis, actions


def test_native_constraint_rejects_observed_transport_tilt(carry_data: tuple) -> None:
    """The old endpoint-only path violates the requested carrying relation."""
    scene, record, metadata, binding, local_axis, axis, actions = carry_data
    q = metadata['joint_positions']
    bad = next(row['joint_positions'] for row in actions if row['step']==200)
    with NativeScene(scene, record, metadata, q, 0., articulated=False) as context:
        context.set_object_carry(binding['contact_local_m'], local_axis, axis,
                                 binding['target_point_world_m'], [0,0,1])
        assert context.constraint_errors(q)['accepted']
        failure = context.constraint_errors(bad)
        assert not failure['accepted']
        assert failure['axis_error_rad'] > np.radians(15)
    clear_native_cache()


def test_native_constraint_measures_plane_intrusion_without_a_buffer(carry_data: tuple) -> None:
    """A 10 mm support violation is reported as a geometric distance, not an arbitrary margin."""
    scene, record, metadata, binding, local_axis, axis, _ = carry_data
    pose = pose_matrix(metadata['eef_pose_world_xyzw'])
    point = pose[:3,3] + pose[:3,:3] @ binding['contact_local_m']
    with NativeScene(scene, record, metadata, metadata['joint_positions'], 0., articulated=False) as context:
        context.set_object_carry(binding['contact_local_m'], local_axis, axis,
                                 point+[0,0,.01], [0,0,1])
        failure = context.constraint_errors(metadata['joint_positions'])
        assert not failure['accepted']
        assert failure['support_plane_violation_m'] == pytest.approx(.01, abs=1e-6)
        assert context.record['object_carry_constraint']['additional_clearance_m'] == 0
    clear_native_cache()


def test_native_carry_path_uses_constrained_planner_without_postcheck(carry_data: tuple) -> None:
    """Check planner output in the test, without adding execution or dense-validation gates."""
    scene, record, metadata, binding, local_axis, axis, _ = carry_data
    q = np.asarray(metadata['joint_positions'])
    with NativeScene(scene, record, metadata, q, 0., articulated=False) as context:
        context.set_object_carry(binding['contact_local_m'], local_axis, axis,
                                 binding['target_point_world_m'], [0,0,1])
        target = q.copy();target[0] += .08
        projected, evidence = context.project(target)
        assert evidence['accepted']
        path, result = context.plan(q, projected, constrained=True, seconds=2.)
        assert result['exact_solution']
        assert not result['additional_dense_validation']
        assert result['planner'] == 'MC-VAMP CRRTC'
        assert all(context.constraint_errors(state)['accepted'] for state in path)
    clear_native_cache()
