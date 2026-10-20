"""LIBERO RGB-D and physical-action worker, isolated from GPU dependencies.

stdin/stdout carry JSON lines; library logs go to stderr.  Observation artifacts
contain calibrated sensor data and robot proprioception only.  The official
success predicate is returned in a separate evaluation envelope.
"""

import contextlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from vlm_tamp.libero import configure_libero
from vlm_tamp.rgbd_numericals import metric_depth, world_points
from .kinematics import PandaIK, matrix_pose, osc_action, rotation_error, pose_matrix, tcp_to_rekep


IMAGE_SIZE = 512
MAX_CONTROL_STEPS = 600
FRONTVIEW_POLICY = {
    "position_world_m": [0.04, 0.55, 1.55],
    "look_at_world_m": [0.04, -0.23, 0.96], "up_world": [0, 0, 1],
    "source": "fixed existing RGB-selected development camera policy",
}


def frontview_rotation(policy=None):
    policy = FRONTVIEW_POLICY if policy is None else policy
    position = np.asarray(policy["position_world_m"], dtype=float)
    target = np.asarray(policy["look_at_world_m"], dtype=float)
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross(policy["up_world"], backward)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(backward, right), backward])


class SimulatorWorker:
    def __init__(self):
        self.env = None
        self.control_steps = 0
        self.settling_steps = 0
        self.episode_dir = None
        self.action_log = []
        self.frame_paths = []
        self.video_path = None

    def _require_env(self):
        if self.env is None:
            raise RuntimeError("Reset a simulator episode before using this operation")

    def reset(self, task_index, state_index, output_dir, seed=0):
        if task_index not in (0, 8):
            raise ValueError("This experiment supports LIBERO Goal tasks 0 and 8")
        if (Path(output_dir).resolve() / "obs.npz").exists():
            raise FileExistsError("Episode output already exists; choose a new directory")
        if self.env is not None:
            self._finish_episode()
            self.env.close()
            self.env = None
        configure_libero(gl="egl")
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        random.seed(seed)
        np.random.seed(seed)
        suite = benchmark.get_benchmark_dict()["libero_goal"](task_order_index=0)
        states = suite.get_task_init_states(task_index)
        if not 0 <= int(state_index) < len(states):
            raise ValueError("Invalid official initial state index")
        self.task_index, self.state_index = int(task_index), int(state_index)
        self.instruction = suite.get_task(task_index).language
        self.env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(task_index),
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=IMAGE_SIZE, camera_widths=IMAGE_SIZE,
            camera_depths=True, camera_segmentations=None,
            control_freq=20, horizon=1000, ignore_done=True,
        )
        self.env.seed(seed)
        self.env.reset()
        self.latest_obs = self.env.set_init_state(np.asarray(states[state_index]))
        self.control_steps, self.settling_steps = 0, 0
        self.episode_dir = Path(output_dir).resolve()
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.action_log, self.frame_paths, self.video_path = [], [], None
        self.robot = self.env.robots[0]
        self.ik_solver = PandaIK(self.env.sim, self.robot)
        for _ in range(5):
            self.latest_obs, _, _, _ = self.env.step(np.zeros(7))
            self.settling_steps += 1
        self.calibration = self._robot_calibration()
        self._save_frame(force=True)
        return self.observe(self.episode_dir)

    def _eef_pose(self):
        sim = self.env.sim
        transform = np.eye(4)
        transform[:3, :3] = sim.data.site_xmat[self.robot.eef_site_id].reshape(3, 3)
        transform[:3, 3] = sim.data.site_xpos[self.robot.eef_site_id]
        return matrix_pose(transform)

    def _robot_calibration(self):
        sim = self.env.sim
        base_id = sim.model.body_name2id(self.robot.robot_model.root_body)
        gripper_id = sim.model.body_name2id(self.robot.gripper.root_body)
        world_base, world_gripper = np.eye(4), np.eye(4)
        for transform, body_id in ((world_base, base_id), (world_gripper, gripper_id)):
            transform[:3, :3] = sim.data.body_xmat[body_id].reshape(3, 3)
            transform[:3, 3] = sim.data.body_xpos[body_id]
        gripper_tcp = np.linalg.inv(world_gripper) @ pose_matrix(self._eef_pose())
        if not np.allclose(gripper_tcp[:3, 3], [0, 0, 0.097], atol=1e-7):
            raise ValueError("Unexpected Panda gripper-to-TCP calibration")
        tool_world, finger_flags = self.ik_solver.collision_points(data=sim.data._data, tool_only=True)
        tcp = pose_matrix(self._eef_pose())
        tool_tcp = (tool_world - tcp[:3, 3]) @ tcp[:3, :3]
        return {
            "pose_convention": "world xyz quaternion xyzw",
            "tcp": "robot controller grip_site",
            "graspgen_base_to_tcp": gripper_tcp.tolist(),
            "tcp_to_rekep": tcp_to_rekep().tolist(),
            "world_to_robot": np.linalg.inv(world_base).tolist(),
            "joint_limits": self.ik_solver.limits.tolist(),
            "tcp_approach_axis": [0, 0, 1], "tcp_closing_axis": [1, 0, 0],
            "graspgen_convention_source": "pinned franka_panda identity asset offset; +z approach, +x closing",
            "gripper_command": {"open": -1, "hold": 0, "close": 1},
            "osc_position_scale_m": 0.05, "osc_rotation_scale_rad": 0.5,
            "geometry_source": "robot model and controller configuration only",
            "tool_collision_points_tcp": tool_tcp.tolist(),
            "tool_collision_finger_flags": finger_flags.tolist(),
            "tool_geometry_configuration": "physical gripper after five settling actions",
        }

    def _camera_calibration(self, camera_name):
        sim = self.env.sim
        cam = sim.model.camera_name2id(camera_name)
        near = float(sim.model.vis.map.znear * sim.model.stat.extent)
        far = float(sim.model.vis.map.zfar * sim.model.stat.extent)
        focal = IMAGE_SIZE / 2 / np.tan(np.deg2rad(sim.model.cam_fovy[cam]) / 2)
        intrinsic = np.array([[focal, 0, IMAGE_SIZE / 2], [0, focal, IMAGE_SIZE / 2], [0, 0, 1]])
        transform = np.eye(4)
        transform[:3, :3] = sim.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = sim.data.cam_xpos[cam]
        return {
            "intrinsic": intrinsic.tolist(), "camera_to_world": transform.tolist(),
            "near_m": near, "far_m": far, "height": IMAGE_SIZE, "width": IMAGE_SIZE,
            "frame": "world", "units": "meters", "image_origin": "top_left",
            "depth_convention": "optical-axis z", "source": "sensor configuration/calibration only",
        }

    def _frontview(self):
        sim = self.env.sim
        policy = getattr(self, 'frontview_policy', FRONTVIEW_POLICY)
        cam = sim.model.camera_name2id("agentview")
        if int(sim.model.cam_bodyid[cam]) != 0:
            raise ValueError("Fixed frontal camera requires world parent")
        previous_position = sim.model.cam_pos[cam].copy()
        previous_quaternion = sim.model.cam_quat[cam].copy()
        try:
            sim.model.cam_pos[cam] = policy["position_world_m"]
            quaternion = Rotation.from_matrix(frontview_rotation(policy)).as_quat()
            sim.model.cam_quat[cam] = quaternion[[3, 0, 1, 2]]
            sim.forward()
            rgb, raw_depth = sim.render(camera_name="agentview", width=IMAGE_SIZE,
                                        height=IMAGE_SIZE, depth=True)
            # Direct MuJoCo framebuffer renders always have bottom-left origin.
            rgb, raw_depth = rgb[::-1], raw_depth[::-1]
            calibration = self._camera_calibration("agentview")
            calibration["view_policy"] = policy
            return rgb, raw_depth, calibration
        finally:
            sim.model.cam_pos[cam] = previous_position
            sim.model.cam_quat[cam] = previous_quaternion
            sim.forward()

    def _sensor_arrays(self):
        import robosuite.macros as macros

        # Re-read sensors after steps so no object-relative observation can leak.
        self.env._update_observables(force=True)
        observation = self.env.env._get_observations()
        arrays, cameras = {}, {}
        for logical, actual in (("agentview", "agentview"), ("wrist", "robot0_eye_in_hand")):
            rgb = np.asarray(observation[actual + "_image"])
            raw_depth = np.asarray(observation[actual + "_depth"])
            if macros.IMAGE_CONVENTION == "opengl":
                rgb, raw_depth = rgb[::-1], raw_depth[::-1]
            calibration = self._camera_calibration(actual)
            depth = metric_depth(raw_depth, calibration["near_m"], calibration["far_m"])
            arrays[logical + "_rgb"] = rgb
            arrays[logical + "_depth"] = depth.astype(np.float32)
            arrays[logical + "_world_points"] = world_points(depth, calibration["intrinsic"], calibration["camera_to_world"]).astype(np.float32)
            cameras[logical] = calibration
        rgb, raw_depth, calibration = self._frontview()
        depth = metric_depth(raw_depth, calibration["near_m"], calibration["far_m"])
        arrays["frontview_rgb"] = rgb
        arrays["frontview_depth"] = depth.astype(np.float32)
        arrays["frontview_world_points"] = world_points(depth, calibration["intrinsic"], calibration["camera_to_world"]).astype(np.float32)
        cameras["frontview"] = calibration
        return arrays, cameras

    def _state_metadata(self):
        # Finger collision geometry changes during real closure/opening.
        self.calibration = self._robot_calibration()
        robot_points, robot_fingers = self.ik_solver.collision_points(data=self.env.sim.data._data)
        return {
            "eef_pose_world_xyzw": self._eef_pose().tolist(),
            "joint_positions": self.env.sim.data.qpos[self.robot._ref_joint_pos_indexes].tolist(),
            "gripper_qpos": self.env.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes].tolist(),
            "robot_calibration": self.calibration, "control_steps": self.control_steps,
            "settling_steps": self.settling_steps,
            "robot_collision_points_world": robot_points.tolist(),
            "robot_collision_finger_flags": robot_fingers.tolist(),
            "robot_collision_volumes_world": self.ik_solver.collision_volumes(data=self.env.sim.data._data),
        }

    def _response_state(self):
        return {"metadata": self._state_metadata(), "control_steps": self.control_steps,
                "evaluation": {"task_success": bool(self.env.check_success())},
                "step_limit_reached": self.control_steps >= MAX_CONTROL_STEPS}

    def observe(self, output_dir):
        self._require_env()
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        if (output / "obs.npz").exists():
            raise FileExistsError(f"Observation already exists: {output}; choose a new directory")
        arrays, cameras = self._sensor_arrays()
        metadata = self._state_metadata()
        metadata.update(cameras=cameras, primary_camera="frontview" if self.task_index == 0 else "agentview",
                        instruction=self.instruction, no_simulator_inventory=True,
                        schema_version="rekep-libero-observation-1.0")
        np.savez_compressed(output / "obs.npz", **arrays)
        for camera in cameras:
            Image.fromarray(arrays[camera + "_rgb"]).save(output / (camera + ".png"))
        Image.fromarray(arrays[metadata["primary_camera"] + "_rgb"]).save(output / "rgb.png")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return {"observation_dir": str(output), "metadata": metadata,
                "evaluation": {"task_success": bool(self.env.check_success())},
                "control_steps": self.control_steps, "step_limit_reached": self.control_steps >= MAX_CONTROL_STEPS}

    def ik(self, target_pose_world_xyzw, seed_joints=None, max_iterations=120):
        self._require_env()
        result = self.ik_solver.solve(target_pose_world_xyzw, initial_joint_pos=seed_joints,
                                     max_iterations=max_iterations)
        points, finger_flags = self.ik_solver.collision_points_at_joints(result.cspace_position)
        return {**result.as_dict(), "robot_collision_points_world": points.tolist(),
                "robot_collision_finger_flags": finger_flags.tolist(),
                "robot_collision_points_include_fixed_base": False}

    def _save_frame(self, force=False):
        if not force and self.control_steps % 5:
            return
        frames = self.episode_dir / "frames"
        frames.mkdir(exist_ok=True)
        rgb = self.env.sim.render(camera_name="agentview", width=IMAGE_SIZE, height=IMAGE_SIZE)[::-1]
        path = frames / f"step{self.control_steps:04d}.jpg"
        if not path.exists():
            Image.fromarray(rgb).save(path, quality=85)
            self.frame_paths.append(str(path))

    def action(self, target_pose_world_xyzw, gripper=0, max_steps=20, output_dir=None):
        self._require_env()
        target = pose_matrix(target_pose_world_xyzw)
        if gripper not in (-1, 0, 1):
            raise ValueError("Gripper must be -1 open, 0 hold, or +1 close")
        if not 1 <= int(max_steps) <= MAX_CONTROL_STEPS:
            raise ValueError("Action max_steps must be in [1,600]")
        executed, reached, task_success = 0, False, False
        for _ in range(min(int(max_steps), MAX_CONTROL_STEPS - self.control_steps)):
            current = self._eef_pose()
            control = osc_action(current, target_pose_world_xyzw, gripper)
            self.latest_obs, _, _, _ = self.env.step(control)
            self.control_steps += 1
            executed += 1
            actual = pose_matrix(self._eef_pose())
            position_error = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
            orientation_error = float(np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3])))
            task_success = bool(self.env.check_success())
            reached = position_error < 0.004 and orientation_error < 0.05
            self.action_log.append({"step": self.control_steps, "target_pose_world_xyzw": list(target_pose_world_xyzw),
                                    "control": control.tolist(), "eef_pose_world_xyzw": self._eef_pose().tolist(),
                                    "gripper_qpos": self.env.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes].tolist(),
                                    "position_error_m": position_error, "orientation_error_rad": orientation_error,
                                    "evaluation_task_success": task_success})
            self._save_frame()
            if getattr(self,'stop_on_success',False) and task_success:
                break
            # A gripper transition must receive all requested control steps;
            # reaching the arm pose alone does not finish a physical closure.
            if reached and gripper == 0:
                break
        self._save_frame(force=True)
        self._write_action_log()
        response = self._response_state()
        response.update(executed_steps=executed, reached_pose=reached)
        if output_dir is not None:
            response.update(self.observe(output_dir))
        return response

    def _write_action_log(self):
        if self.episode_dir:
            (self.episode_dir / "actions.json").write_text(json.dumps(self.action_log, indent=2) + "\n")

    def _finish_episode(self):
        self._write_action_log()
        if not self.frame_paths:
            return
        self.video_path = self.episode_dir / "rollout.mp4"
        encoder = shutil.which("ffmpeg")
        if not encoder:
            (self.episode_dir / "video_error.txt").write_text("Existing ffmpeg encoder unavailable\n")
            self.video_path = None
            return
        try:
            with (self.episode_dir / "video_encoder.log").open("w") as log:
                video = subprocess.Popen([
                    encoder, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", f"{IMAGE_SIZE}x{IMAGE_SIZE}", "-framerate", "4", "-i", "pipe:0",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.video_path),
                ], stdin=subprocess.PIPE, stderr=log, stdout=log)
                try:
                    for frame in self.frame_paths:
                        with Image.open(frame) as image:
                            video.stdin.write(image.convert("RGB").tobytes())
                    video.stdin.close()
                    if video.wait(timeout=60) != 0:
                        raise RuntimeError("Video encoder failed; see video_encoder.log")
                finally:
                    if video.poll() is None:
                        video.kill()
                        video.wait(timeout=10)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            (self.episode_dir / "video_error.txt").write_text(str(exc) + "\n")
            self.video_path = None

    def close(self):
        response = {}
        if self.env is not None:
            response = self._response_state()
            self._finish_episode()
            response["video_path"] = str(self.video_path) if self.video_path else None
            self.env.close()
            self.env = None
        return response

    def dispatch(self, request):
        operation = request["op"]
        if operation not in {"reset", "observe", "action", "ik", "close"}:
            raise ValueError("Unknown simulator operation")
        arguments = {key: value for key, value in request.items() if key not in {"op", "id"}}
        return getattr(self, operation)(**arguments)


def main():
    protocol_output = sys.stdout
    worker = SimulatorWorker()
    # Native LIBERO prints task-order information during construction.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                request = {}
                try:
                    request = json.loads(line)
                    response = {"ok": True, "id": request.get("id"), **worker.dispatch(request)}
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    response = {"ok": False, "id": request.get("id"), "error": str(exc), "error_type": type(exc).__name__}
                protocol_output.write(json.dumps(response, allow_nan=False) + "\n")
                protocol_output.flush()
                if request.get("op") == "close":
                    break
        finally:
            worker.close()


if __name__ == "__main__":
    main()
