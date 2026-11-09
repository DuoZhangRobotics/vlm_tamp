"""Selected-object contact roles stay identical through native and GPU scene setup."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vlm_tamp.collision_scene_points import planner_points
from vlm_tamp.coupled_scene import Scene
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def test_whole_selected_mask_is_contact_without_erasing_neighboring_obstacles() -> None:
    """A bowl point outside the old patch changes role; a nearby table point does not."""
    patch = np.array([[0., 0., 1.]])
    rim = np.array([[.05, 0., 1.]])
    table = np.array([[.05, 0., .98]])
    spec = {'joint_origin_world_m': [0, 0, 1], 'joint_axis_world': [0, 0, 1],
            'front_plane': {'point_world_m': [0, 0, 1]}, 'panel_normal_world': [1, 0, 0]}
    scene = Scene.from_points(np.vstack([rim, table]), patch, patch, spec)
    original = scene.static_points.copy()
    scene.grasp_contact_reference = np.vstack([patch, rim])
    scene.skill_contact_surface = 'palm_fingertips'
    clouds, counts = planner_points(scene, np.eye(4))
    np.testing.assert_allclose(clouds['stationary'], table)
    np.testing.assert_allclose(clouds['contact'], np.vstack([patch, rim]))
    assert counts == {'stationary': 1, 'moving': 0, 'contact': 2}
    np.testing.assert_array_equal(scene.static_points, original)
    scene.skill_contact_surface = 'fingertips'
    ordinary, _ = planner_points(scene, np.eye(4))
    assert len(ordinary['stationary']) == 2 and len(ordinary['contact']) == 1


def test_gpu_rpc_receives_native_contact_roles_and_permissions(tmp_path: Path) -> None:
    """The native target exemption must not disappear at the IK RPC boundary."""
    calls = []
    ik = TaskIK.__new__(TaskIK)
    ik.output = tmp_path
    ik.client = SimpleNamespace(request=lambda operation, **kwargs: calls.append((operation, kwargs)))
    clouds = {name: np.array([[i, 0., 1.]], np.float32)
              for i, name in enumerate(('stationary', 'moving', 'contact'))}
    model = {'coefficients': [[0, 0, 0, .012]]*4}
    context = SimpleNamespace(ik_clouds_world=clouds, record={
        'palm_contact_allowed': True, 'fingertip_contact_allowed': True, 'finger_model': model})
    ik.configure_scene(context)
    operation, request = calls[0]
    assert operation == 'configure_scene'
    assert request['palm_contact'] and request['fingertip_contact']
    assert request['finger_model'] == model
    with np.load(request['arrays_path']) as saved:
        for name, values in clouds.items():
            np.testing.assert_array_equal(saved[name], values)
