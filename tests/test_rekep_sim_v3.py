"""Deterministic restart and calibrated arm-view checks for the v3 adapter."""

import os
import json

import numpy as np
import pytest

from vlm_tamp.rekep_libero.ik_v3 import MultiStartPandaIK
from vlm_tamp.rekep_libero.sim_worker_v3 import ARM_VIEW_POLICIES, camera_rotation
from vlm_tamp.rekep_libero.sim_worker_v3 import scaled_osc_action
from vlm_tamp.rekep_libero.sim_worker_v3 import controller_target_reached
from vlm_tamp.rekep_libero.sim_worker_v3 import completion_tolerances
from vlm_tamp.rekep_libero.sim_worker_v3 import checked_nullspace_kp
from vlm_tamp.rekep_libero.kinematics import osc_action


def test_arm_camera_basis_and_fixed_policy():
    assert set(ARM_VIEW_POLICIES) == {"arm_side", "arm_back", "arm_top", "arm_base",
                                      "arm_lower_left", "arm_lower_right", "arm_wide_front",
                                      "arm_wide_back", "arm_wide_side"}
    for policy in ARM_VIEW_POLICIES.values():
        rotation = camera_rotation(policy["position_world_m"], policy["look_at_world_m"])
        assert np.allclose(rotation.T @ rotation, np.eye(3))
        assert np.isclose(np.linalg.det(rotation), 1)
        optical = -rotation[:, 2]
        target = np.subtract(policy["look_at_world_m"], policy["position_world_m"])
        assert np.allclose(optical, target / np.linalg.norm(target))


def test_restart_seeds_all_seven_bounded_deterministic():
    solver = object.__new__(MultiStartPandaIK)
    solver.limits = np.array([[-2.9, 2.9], [-1.7, 1.7], [-2.9, 2.9], [-3.07, -0.07],
                              [-2.9, 2.9], [-0.01, 3.75], [-2.9, 2.9]])
    start = [0, 0, 0, -2, 0, 2, 0]
    seeds = solver.seed_configurations(start)
    assert len(seeds) == 12
    assert all(len(seed) == 7 for seed in seeds)
    assert np.array_equal(seeds[0], start)
    assert all(np.all(seed > solver.limits[:, 0]) and np.all(seed < solver.limits[:, 1])
               for seed in seeds)
    assert np.array_equal(seeds, solver.seed_configurations(start))
    with pytest.raises(ValueError):
        solver.seed_configurations([0] * 6)
    with pytest.raises(ValueError):
        solver.seed_configurations(start, max_starts=0)


def test_optional_servo_gain_preserves_default_and_gripper():
    current = [0, 0, 1, 0, 0, 0, 1]
    target = [.02, -.01, 1.1, .1, 0, 0, .99]
    default = osc_action(current, target, gripper=1)
    assert np.array_equal(scaled_osc_action(current, target, 1), default)
    slowed = scaled_osc_action(current, target, 1, .5)
    assert np.array_equal(slowed[:6], default[:6] * .5)
    assert slowed[6] == default[6] == 1
    for invalid in (0, -1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            scaled_osc_action(current, target, 1, invalid)


def test_posture_target_must_settle_despite_reached_tcp():
    assert controller_target_reached(.002, .01)
    assert not controller_target_reached(.002, .01, .031)
    assert not controller_target_reached(.002, .01, .03)
    assert controller_target_reached(.002, .01, .029)
    assert not controller_target_reached(.004, .01, .029)
    assert not controller_target_reached(.002, .05, .029)


def test_optional_tolerances_can_only_tighten_completion():
    tightened = completion_tolerances(.001, .01, .01)
    assert controller_target_reached(.003, .02, .02)
    assert not controller_target_reached(.003, .02, .02, **tightened)
    assert controller_target_reached(.0009, .009, .009, **tightened)
    for values in ((.005, .05, .03), (.004, .051, .03), (.004, .05, .031),
                   (0, .01, .01), (.001, np.nan, .01), (.001, .01, -1)):
        with pytest.raises(ValueError):
            completion_tolerances(*values)


def test_nullspace_gain_limits_and_default():
    assert checked_nullspace_kp() == 10
    assert checked_nullspace_kp(80) == 80
    for invalid in (0, -1, 100.01, np.nan, np.inf):
        with pytest.raises(ValueError):
            checked_nullspace_kp(invalid)


@pytest.mark.skipif(os.environ.get("RUN_LIBERO_PHYSICAL_TESTS") != "1",
                    reason="Requires the protected physical LIBERO environment")
def test_v3_calibration_and_isolated_multiik(tmp_path):
    from vlm_tamp.rekep_libero.sim_worker import SimulatorWorker
    from vlm_tamp.rekep_libero.sim_worker_v3 import SimulatorWorkerV3

    worker = SimulatorWorkerV3()
    try:
        response = worker.reset(8, 1, tmp_path / "physical-v3", seed=0)
        initial_metadata = response["metadata"]
        assert len(initial_metadata["robot_collision_point_geom_names"]) == len(
            initial_metadata["robot_collision_points_world"])
        assert len(initial_metadata["robot_collision_contact_face_flags"]) == len(
            initial_metadata["robot_collision_points_world"])
        assert initial_metadata["robot_visual_volumes_world"]
        for volume in initial_metadata["robot_visual_volumes_world"]:
            assert volume["robot_geom_name"].startswith(("robot0_", "gripper0_"))
            vertices, planes = np.array(volume["vertices_world"]), np.array(volume["planes_world"])
            assert np.max(vertices @ planes[:, :3].T + planes[:, 3]) < 1e-6
        assert np.isclose(initial_metadata["robot_calibration"]["physical_finger_attach_z_m"], .0524)
        artifact = initial_metadata["robot_visual_meshes_artifact"]
        assert artifact["current_pose_only"] and artifact["no_scene_geometry"]
        with open(artifact["manifest_path"]) as handle:
            mesh_manifest = json.load(handle)
        mesh_arrays = np.load(artifact["npz_path"])
        assert len(mesh_manifest["meshes"]) == artifact["mesh_count"]
        for mesh in mesh_manifest["meshes"]:
            assert mesh["robot_geom_name"].startswith(("robot0_", "gripper0_"))
            vertices = mesh_arrays[mesh["vertices_key"]]
            triangles = mesh_arrays[mesh["faces_key"]]
            assert triangles.shape == (mesh["triangle_count"], 3)
            assert np.min(triangles) >= 0 and np.max(triangles) < len(vertices)
            assert mesh_arrays[mesh["rotation_key"]].shape == (3, 3)
            assert mesh_arrays[mesh["position_key"]].shape == (3,)
        original_state = worker.env.get_sim_state().copy()
        original_geom_groups = worker.env.sim.model._model.geom_group.copy()
        original_render_groups = worker.env.sim._render_context_offscreen.vopt.geomgroup.copy()
        original_arrays, _ = SimulatorWorker._sensor_arrays(worker)
        arrays, cameras = worker._sensor_arrays()
        for name, array in original_arrays.items():
            assert np.array_equal(array, arrays[name]), name
        assert np.array_equal(original_state, worker.env.get_sim_state())
        assert np.array_equal(original_geom_groups, worker.env.sim.model._model.geom_group)
        assert np.array_equal(original_render_groups, worker.env.sim._render_context_offscreen.vopt.geomgroup)
        assert len(cameras) == 12
        self_matches, preserved_occluders = 0, 0
        for camera, calibration in cameras.items():
            cloud = arrays[camera + "_world_points"].astype(float)
            transform = np.asarray(calibration["camera_to_world"])
            intrinsic = np.asarray(calibration["intrinsic"])
            optical = (cloud - transform[:3, 3]) @ transform[:3, :3]
            projected = optical @ intrinsic.T
            pixels = projected[..., :2] / projected[..., 2:]
            yy, xx = np.mgrid[:512, :512]
            assert np.max(np.linalg.norm(pixels - np.stack([xx, yy], -1), axis=-1)) < 0.01
            assert np.allclose(optical[..., 2], arrays[camera + "_depth"], atol=1e-5)
            prior = arrays[camera + "_robot_depth_prior"]
            actual = arrays[camera + "_depth"]
            assert prior.shape == (512, 512)
            assert np.isnan(prior).any()
            matched = np.isfinite(prior) & (np.abs(actual - prior) <= 1e-5)
            occluded = np.isfinite(prior) & (actual < prior - .001)
            self_matches += int(matched.sum())
            preserved_occluders += int(occluded.sum())
            assert not np.any(matched & occluded)
            assert calibration["robot_depth_prior"]["scene_geometry_used"] is False
        assert self_matches > 1000
        assert preserved_occluders > 0
        # These are audited robot shoulder samples, not environment object poses.
        shoulder = np.array([[-.75, .04, 1.22], [-.75, .04, 1.26],
                             [-.75, .067, 1.22], [-.75, .067, 1.26]])
        base_calibration = cameras["arm_base"]
        transform = np.array(base_calibration["camera_to_world"])
        optical = (shoulder - transform[:3, 3]) @ transform[:3, :3]
        projected = optical @ np.array(base_calibration["intrinsic"]).T
        pixels = projected[:, :2] / projected[:, 2:]
        assert np.all(optical[:, 2] > base_calibration["near_m"])
        assert np.all(optical[:, 2] < base_calibration["far_m"])
        assert np.all((pixels >= 0) & (pixels < 512))
        for name in ("arm_lower_left", "arm_lower_right"):
            calibration = cameras[name]
            transform = np.array(calibration["camera_to_world"])
            known_underside = np.array([[-.75, .05, 1.24], [-.6, 0, 1.05]])
            optical = (known_underside - transform[:3, 3]) @ transform[:3, :3]
            projected = optical @ np.array(calibration["intrinsic"]).T
            pixels = projected[:, :2] / projected[:, 2:]
            assert np.all((pixels >= 0) & (pixels < 512))
            assert np.all(optical[:, 2] > calibration["near_m"])
        for name in ("arm_wide_front", "arm_wide_back", "arm_wide_side"):
            calibration = cameras[name]
            transform = np.array(calibration["camera_to_world"])
            workspace = np.array([[-.7, 0, 1.1], [-.44, .04, 1.59], [-.43, .05, 1.6],
                                  [-.6, 0, 1.05], [-.75, .05, 1.24]])
            optical = (workspace - transform[:3, 3]) @ transform[:3, :3]
            projected = optical @ np.array(calibration["intrinsic"]).T
            pixels = projected[:, :2] / projected[:, 2:]
            assert np.all((pixels >= 0) & (pixels < 512))
            assert np.all(optical[:, 2] > calibration["near_m"])
        with open(initial_metadata["robot_workspace_coverage_artifact"]) as handle:
            coverage = json.load(handle)
        assert len(coverage["cameras"]) == 12
        assert "per_point_observed_free_views" in coverage
        joints = response["metadata"]["joint_positions"]
        live_pose = worker._eef_pose()
        fk = worker.fk(joints)
        assert np.allclose(fk["eef_pose_world_xyzw"], live_pose)
        result = worker.multiik(live_pose, seed_joints=joints, max_iterations=200)
        assert result["success"] and result["num_starts"] == 12
        assert result["num_feasible_solutions"] >= 1
        assert all(len(solution["cspace_position"]) == 7 for solution in result["solutions"])
        assert not result["robot_collision_points_include_fixed_base"]
        assert len(result["robot_collision_point_geom_names"]) == len(result["robot_collision_points_world"])
        assert len(result["robot_collision_contact_face_flags"]) == len(result["robot_collision_points_world"])
        assert np.array_equal(original_state, worker.env.get_sim_state())
        assert "segmentation" not in " ".join(arrays)
        # Independent full-forward state verifies the FK-only IK pipeline.
        import mujoco
        reference = mujoco.MjData(worker.ik_solver.model)
        rng = np.random.default_rng(0)
        for _ in range(10):
            configuration = np.clip(np.array(joints) + rng.uniform(-.35, .35, 7),
                                    worker.ik_solver.limits[:, 0] + 1e-5,
                                    worker.ik_solver.limits[:, 1] - 1e-5)
            worker.ik_solver.fk(configuration)
            for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
                getattr(reference, field)[:] = getattr(worker.ik_solver.data, field)
            mujoco.mj_forward(worker.ik_solver.model, reference)
            assert np.allclose(reference.site_xpos, worker.ik_solver.data.site_xpos, atol=1e-12)
            assert np.allclose(reference.site_xmat, worker.ik_solver.data.site_xmat, atol=1e-12)
            assert np.allclose(reference.geom_xpos, worker.ik_solver.data.geom_xpos, atol=1e-12)
            assert np.allclose(reference.geom_xmat, worker.ik_solver.data.geom_xmat, atol=1e-12)
            slow_position, slow_rotation = np.zeros((3, worker.ik_solver.model.nv)), np.zeros((3, worker.ik_solver.model.nv))
            fast_position, fast_rotation = slow_position.copy(), slow_rotation.copy()
            mujoco.mj_jacSite(worker.ik_solver.model, reference, slow_position, slow_rotation, worker.ik_solver.site_id)
            worker.ik_solver.mujoco.mj_jacSite(worker.ik_solver.model, worker.ik_solver.data,
                                            fast_position, fast_rotation, worker.ik_solver.site_id)
            assert np.allclose(slow_position, fast_position, atol=1e-12)
            assert np.allclose(slow_rotation, fast_rotation, atol=1e-12)
            slow_points, slow_flags = worker.ik_solver.collision_points(data=reference, moving_only=True)
            fast_points, fast_flags = worker.ik_solver.collision_points(moving_only=True)
            assert np.allclose(slow_points, fast_points, atol=1e-12)
            assert np.array_equal(slow_flags, fast_flags)
        assert np.array_equal(original_state, worker.env.get_sim_state())
        nullspace_before = worker.robot.controller.initial_joint.copy()
        assert worker._set_nullspace_joints(None) is None
        assert np.array_equal(nullspace_before, worker.robot.controller.initial_joint)
        nullspace = np.array(joints)
        nullspace[2] += .1
        worker._set_nullspace_joints(nullspace)
        assert np.array_equal(worker.robot.controller.initial_joint, nullspace)
        assert np.array_equal(original_state, worker.env.get_sim_state())
        for invalid in ([0] * 6, [np.nan] * 7, [100] * 7):
            with pytest.raises(ValueError):
                worker._set_nullspace_joints(invalid)
        assert np.array_equal(original_state, worker.env.get_sim_state())
        moved = live_pose.copy()
        moved[0] += .01
        response = worker.action(moved, gripper=0, max_steps=3, servo_scale=.5,
                                 nullspace_joint_target=nullspace)
        assert response["executed_steps"] >= 1
        for row in worker.action_log:
            assert len(row["joint_positions"]) == len(row["joint_velocities"]) == 7
            assert row["servo_scale"] == .5
            assert row["nullspace_target_joints"] == nullspace.tolist()
            assert row["controller_nullspace_reference_joints"] == nullspace.tolist()
        assert "contacts" not in json.dumps(response).lower()
        with open(worker.episode_dir / "evaluation_contact_diagnostics.json") as handle:
            diagnostics = json.load(handle)
        assert diagnostics["evaluation_only"] and diagnostics["excluded_from_policy_input"]
        assert len(diagnostics["steps"]) == response["executed_steps"]
        pose_now = worker._eef_pose()
        posture_now = worker.env.sim.data.qpos[worker.robot._ref_joint_pos_indexes].copy()
        posture_now[2] += .15
        missed = worker.action(pose_now, gripper=0, max_steps=1, servo_scale=.5,
                               nullspace_joint_target=posture_now)
        assert missed["tcp_pose_reached"]
        assert not missed["reached_pose"]
        assert missed["posture_error_rad"] >= .03
        assert "max_joint_error_lt_0.03rad" in missed["reached_criterion"]
        assert missed["completion_tolerances"] == completion_tolerances()
        before_invalid = worker.env.get_sim_state().copy()
        with pytest.raises(ValueError):
            worker.action(pose_now, max_steps=1, position_tolerance=.005)
        assert np.array_equal(before_invalid, worker.env.get_sim_state())
        controller = worker.robot.controller
        original_method = controller.run_controller
        original_helper = original_method.__func__.__globals__["nullspace_torques"]
        original_cartesian_gains = controller.kp.copy()
        controller.update(force=True)
        baseline_torques = original_method().copy()
        worker._set_nullspace_kp(10)
        controller.update(force=True)
        assert np.array_equal(controller.run_controller().copy(), baseline_torques)
        worker._set_nullspace_kp(80)
        assert controller.run_controller.__func__.__closure__ == original_method.__func__.__closure__
        assert original_method.__func__.__globals__["nullspace_torques"] is original_helper
        assert np.array_equal(controller.kp, original_cartesian_gains)
        controller.update(force=True)
        assert np.isfinite(controller.run_controller()).all()
        gain_state = worker.env.get_sim_state().copy()
        worker._set_nullspace_kp(10)
        assert controller.run_controller.__func__ is original_method.__func__
        assert np.array_equal(gain_state, worker.env.get_sim_state())
        gain_action = worker.action(worker._eef_pose(), max_steps=1, nullspace_kp=80,
                                    nullspace_joint_target=worker.env.sim.data.qpos[worker.robot._ref_joint_pos_indexes])
        assert gain_action["nullspace_kp"] == 80
        assert worker.action_log[-1]["nullspace_kp"] == 80
        assert np.isfinite(worker.action_log[-1]["joint_positions"]).all()
    finally:
        worker.close()
