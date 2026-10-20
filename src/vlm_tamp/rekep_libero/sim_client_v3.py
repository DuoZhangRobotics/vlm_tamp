"""Versioned JSON-lines client preserving the protected simulator runtime."""

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from .sim_client import SimClient


class SimClientV3(SimClient):
    def __init__(self, python=None, timeout=180, log_path=None):
        self.python = str(python or "/home/duo/miniforge3/envs/vlm-tamp/bin/python")
        self.timeout = float(timeout)
        self.log_handle = open(log_path, "a") if log_path else None
        root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        for name in list(environment):
            if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")) or name in {"HF_TOKEN", "OPENAI_ADMIN_KEY"}:
                environment.pop(name, None)
        environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [self.python, "-m", "vlm_tamp.rekep_libero.sim_worker_v3"], cwd=root,
            env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log_handle or sys.stderr, text=True, bufsize=1)
        self.messages = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.request_id = 0

    def multiik(self, target_pose_world_xyzw, seed_joints=None, max_iterations=200,
                max_starts=12, include_volumes=False):
        kwargs = {"target_pose_world_xyzw": list(target_pose_world_xyzw),
                  "max_iterations": int(max_iterations), "max_starts": int(max_starts),
                  "include_volumes": bool(include_volumes)}
        if seed_joints is not None:
            kwargs["seed_joints"] = list(seed_joints)
        return self.request("multiik", **kwargs)

    def fk(self, joints=None, include_volumes=False):
        kwargs = {"include_volumes": bool(include_volumes)}
        if joints is not None:
            kwargs["joints"] = list(joints)
        return self.request("fk", **kwargs)

    multi_ik = multiik

    def robot_visual_meshes(self, output_dir):
        return self.request("robot_visual_meshes", output_dir=str(output_dir))

    def action(self, target_pose_world_xyzw, gripper=0, max_steps=20, output_dir=None, servo_scale=1,
               nullspace_joint_target=None, position_tolerance=.004, orientation_tolerance=.05,
               posture_tolerance=.03, nullspace_kp=10):
        kwargs = dict(target_pose_world_xyzw=list(target_pose_world_xyzw), gripper=int(gripper),
                      max_steps=int(max_steps), servo_scale=float(servo_scale),
                      position_tolerance=float(position_tolerance), orientation_tolerance=float(orientation_tolerance),
                      posture_tolerance=float(posture_tolerance), nullspace_kp=float(nullspace_kp))
        if output_dir is not None:
            kwargs["output_dir"] = str(output_dir)
        if nullspace_joint_target is not None:
            kwargs["nullspace_joint_target"] = list(nullspace_joint_target)
        return self.request("action", **kwargs)
