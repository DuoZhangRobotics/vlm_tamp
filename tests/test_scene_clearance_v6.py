"""Physical-geometry regression checks for occupied surfaces and point goals."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.scene_clearance_v6 import ObservedClearance
from vlm_tamp.rekep_libero.task_ik_v6 import TaskIK

ROOT = Path(__file__).resolve().parents[1]


def volume(low, high, name):
    corners = np.array([[x, y, z] for x in [low[0], high[0]] for y in [low[1], high[1]] for z in [low[2], high[2]]])
    return {'type': 'convex', 'fixed': False, 'robot_geom_name': name,
            'vertices_world': corners.tolist(), 'planes_world': ConvexHull(corners).equations.tolist()}


def checker(points):
    camera = {'world_points': np.asarray(points).reshape(2, 2, 3), 'depth': np.ones((2, 2)),
              'intrinsic': np.eye(3), 'camera_to_world': np.eye(4)}
    return ObservedClearance({'view': camera}, {}, np.asarray(points))


def test_selected_object_cannot_be_exempted_from_palm_collision():
    points = [[.001, .001, 1.001], [.003, .001, 1.001], [.001, .003, 1.001], [.003, .003, 1.001]]
    scene = checker(points)
    palm = volume([-.01, -.01, .99], [.01, .01, 1.01], 'gripper0_hand_collision')
    result = scene.check([palm], [0, 0, 1, 0, 0, 0, 1])
    assert not result['accepted']
    assert result['intended_inward_pad_contact_exemptions'] == 0
    assert not result['unknown_space_validated']


def test_only_inward_pad_face_can_touch_selected_surface():
    points = [[.018, -.003, .998], [.018, .003, .998], [.018, -.003, 1.002], [.018, .003, 1.002]]
    pad = volume([.018, -.01, .99], [.025, .01, 1.01], 'gripper0_finger1_pad_collision')
    result = checker(points).check([pad], [0, 0, 1, 0, 0, 0, 1])
    assert result['accepted']
    assert result['intended_inward_pad_contact_exemptions'] == 4
    moved = np.asarray(points) + [.005, 0, 0]
    assert not checker(moved).check([pad], [0, 0, 1, 0, 0, 0, 1])['accepted']


def test_point_goal_is_checked_in_object_point_frame():
    record = json.loads((ROOT / 'tests/data/panda_chain_v4.json').read_text())
    config = json.loads((ROOT / 'configs/physical_tasks_v6.json').read_text())
    ik = TaskIK(record, config)
    seed = ik.chain.reference_joints
    local = np.array([.02, -.01, -.03])
    initial = pose_matrix(ik.chain.forward(seed))
    target = initial[:3, 3] + initial[:3, :3] @ local + [.02, -.01, -.03]
    solved = ik.solve_point(target, local, seed)
    assert solved is not None
    actual = pose_matrix(solved['pose_world_xyzw'])
    assert np.linalg.norm(actual[:3, 3] + actual[:3, :3] @ local - target) <= .001
    assert solved['orientation_is_soft_preference']
