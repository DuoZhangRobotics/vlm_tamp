"""Frame, isolated IK and physical controller checks for the LIBERO adapter."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.rekep_libero.kinematics import matrix_pose, osc_action, pose_matrix, tcp_to_rekep


def test_pose_conventions_and_rekep_approach():
    pose = np.r_[[0.1, 0.2, 0.3], Rotation.from_euler("xyz", [0.3, -0.2, 0.7]).as_quat()]
    assert np.allclose(pose_matrix(matrix_pose(pose_matrix(pose))), pose_matrix(pose))
    assert np.allclose(tcp_to_rekep()[:3, 0], [0, 0, 1])
    assert np.isclose(np.linalg.det(tcp_to_rekep()[:3, :3]), 1)
    with pytest.raises(ValueError):
        pose_matrix([0, 0, 0, 0, 0, 0, 0])


def test_osc_left_rotation_and_scaling():
    current = pose_matrix([0, 0, 0, 0, 0, 0, 1])
    current[:3, :3] = Rotation.from_euler("y", 0.4).as_matrix()
    target = current.copy()
    target[:3, 3] += [0.01, -0.005, 0]
    target[:3, :3] = Rotation.from_rotvec([0, 0, 0.05]).as_matrix() @ current[:3, :3]
    action = osc_action(matrix_pose(current), matrix_pose(target), 1)
    assert np.allclose(action, [0.2, -0.1, 0, 0, 0, 0.1, 1], atol=1e-9)


@pytest.mark.skipif(os.environ.get("RUN_LIBERO_PHYSICAL_TESTS") != "1",
                    reason="Physical LIBERO checks require the protected simulator environment")
def test_physical_calibration(tmp_path):
    from vlm_tamp.rekep_libero.sim_worker import SimulatorWorker

    output = Path(os.environ.get("REKEP_CALIBRATION_OUTPUT", str(tmp_path / "physical")))
    if output.exists():
        raise RuntimeError("Calibration output already exists; choose a new directory")
    worker = SimulatorWorker()
    checks = {}
    try:
        response = worker.reset(task_index=8, state_index=0, output_dir=output, seed=0)
        assert response["metadata"]["settling_steps"] == 5
        arrays = np.load(output / "obs.npz")
        assert set(arrays.files) == {view + "_" + kind for view in ("agentview", "wrist", "frontview")
                                    for kind in ("rgb", "depth", "world_points")}
        for camera, calibration in response["metadata"]["cameras"].items():
            cloud = arrays[camera + "_world_points"].astype(float)
            transform = np.asarray(calibration["camera_to_world"])
            intrinsic = np.asarray(calibration["intrinsic"])
            optical = (cloud - transform[:3, 3]) @ transform[:3, :3]
            pixels = optical @ intrinsic.T
            pixels = pixels[..., :2] / pixels[..., 2:]
            yy, xx = np.mgrid[:512, :512]
            error = np.max(np.linalg.norm(pixels - np.stack([xx, yy], axis=-1), axis=-1))
            checks[camera + "_maximum_roundtrip_pixel_error"] = float(error)
            assert error < 0.01
            assert np.allclose(optical[..., 2], arrays[camera + "_depth"], atol=1e-5)
        original_state = worker.env.get_sim_state().copy()
        live_pose = worker._eef_pose()
        original_joints = worker.env.sim.data.qpos[worker.robot._ref_joint_pos_indexes].copy()
        result = worker.ik(live_pose, seed_joints=original_joints)
        assert result["success"]
        assert result["position_error"] < 1e-9
        assert result["orientation_error"] < 1e-9
        assert len(result["cspace_position"]) == 7
        assert not result["robot_collision_points_include_fixed_base"]
        assert len(result["robot_collision_points_world"]) < len(response["metadata"]["robot_collision_points_world"])
        assert np.array_equal(original_state, worker.env.get_sim_state())
        volumes = response["metadata"]["robot_collision_volumes_world"]
        assert volumes and any(volume["fixed"] for volume in volumes)
        assert any(not volume["fixed"] for volume in volumes)
        for volume in volumes:
            assert volume["robot_geom_name"].startswith(("robot0_", "gripper0_"))
            if volume["type"] == "convex":
                vertices = np.asarray(volume["vertices_world"])
                planes = np.asarray(volume["planes_world"])
                assert np.max(vertices @ planes[:, :3].T + planes[:, 3]) < 1e-7
                assert np.max(vertices.mean(axis=0) @ planes[:, :3].T + planes[:, 3]) < 0
                outside = vertices.mean(axis=0) + [2, 2, 2]
                assert np.max(outside @ planes[:, :3].T + planes[:, 3]) > 0.1
        checks["exact_robot_volumes_count"] = len(volumes)
        checks["fixed_robot_volumes_count"] = sum(int(volume["fixed"]) for volume in volumes)
        checks["candidate_collision_points_omit_fixed_base"] = True
        checks["isolated_ik_live_tcp"] = result
        shifted = pose_matrix(live_pose)
        shifted[:3, 3] += [0.01, 0, 0]
        shifted[:3, :3] = Rotation.from_rotvec([0, 0, np.deg2rad(5)]).as_matrix() @ shifted[:3, :3]
        shifted_pose = matrix_pose(shifted)
        shifted_ik = worker.ik(shifted_pose)
        assert shifted_ik["success"]
        assert np.array_equal(original_state, worker.env.get_sim_state())
        checks["isolated_ik_shifted_tcp"] = shifted_ik
        result = worker.action(shifted_pose, gripper=0, max_steps=40)
        moved = pose_matrix(result["metadata"]["eef_pose_world_xyzw"])
        checks["physical_shift_position_error_m"] = float(np.linalg.norm(moved[:3, 3] - shifted[:3, 3]))
        checks["physical_shift_orientation_error_rad"] = float(np.linalg.norm(
            Rotation.from_matrix(shifted[:3, :3] @ moved[:3, :3].T).as_rotvec()))
        assert checks["physical_shift_position_error_m"] < 0.004
        assert checks["physical_shift_orientation_error_rad"] < 0.05
        hold = worker._eef_pose()
        opened = worker.action(hold, gripper=-1, max_steps=60)
        opened_q = np.asarray(opened["metadata"]["gripper_qpos"])
        closed = worker.action(hold, gripper=1, max_steps=120)
        closed_q = np.asarray(closed["metadata"]["gripper_qpos"])
        checks["opened_gripper_qpos"] = opened_q.tolist()
        checks["closed_gripper_qpos"] = closed_q.tolist()
        checks["gripper_physical_gap_change_m"] = float(np.abs(opened_q).sum() - np.abs(closed_q).sum())
        assert checks["gripper_physical_gap_change_m"] > 0.02
        robot_calibration = response["metadata"]["robot_calibration"]
        assert np.allclose(np.asarray(robot_calibration["graspgen_base_to_tcp"])[:3, 3], [0, 0, 0.097])
        assert len(robot_calibration["tool_collision_points_tcp"]) == len(robot_calibration["tool_collision_finger_flags"])
        assert any(robot_calibration["tool_collision_finger_flags"])
        assert "segmentation" not in " ".join(arrays.files)
        checks["robot_calibration"] = robot_calibration
        checks["passed"] = True
    finally:
        close = worker.close()
        checks["video_path"] = close.get("video_path")
        output.mkdir(parents=True, exist_ok=True)
        (output / "calibration_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
