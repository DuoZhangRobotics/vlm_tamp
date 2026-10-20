"""JSON-lines client for the protected legacy LIBERO simulator process."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import queue


class SimClient:
    def __init__(self, python=None, timeout=180, log_path=None):
        self.python = str(python or "/home/duo/miniforge3/envs/vlm-tamp/bin/python")
        self.timeout = float(timeout)
        self.log_handle = open(log_path, "a") if log_path else None
        root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        # Simulator process has no API role and must not inherit service keys.
        for name in list(environment):
            if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")) or name in {"HF_TOKEN", "OPENAI_ADMIN_KEY"}:
                environment.pop(name, None)
        environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [self.python, "-m", "vlm_tamp.rekep_libero.sim_worker"], cwd=root,
            env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log_handle or sys.stderr, text=True, bufsize=1,
        )
        self.messages = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.request_id = 0

    def _read(self):
        for line in self.process.stdout:
            try:
                self.messages.put(json.loads(line))
            except json.JSONDecodeError:
                self.messages.put({"ok": False, "error": "Non-JSON simulator protocol output", "line": line[:200]})
        self.messages.put({"ok": False, "error": "Simulator worker exited"})

    def request(self, operation, **kwargs):
        if self.process.poll() is not None:
            raise RuntimeError("Simulator process has exited")
        self.request_id += 1
        payload = {"op": operation, "id": self.request_id, **kwargs}
        self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
        self.process.stdin.flush()
        try:
            response = self.messages.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"Simulator {operation} exceeded {self.timeout}s") from exc
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Simulator request failed"))
        if response.get("id") != self.request_id:
            raise RuntimeError("Simulator response ID mismatch")
        return response

    def reset(self, task_index, state_index, output_dir, seed=0):
        return self.request("reset", task_index=int(task_index), state_index=int(state_index),
                            output_dir=str(output_dir), seed=int(seed))

    def observe(self, output_dir):
        return self.request("observe", output_dir=str(output_dir))

    def action(self, target_pose_world_xyzw, gripper=0, max_steps=20, output_dir=None):
        kwargs = dict(target_pose_world_xyzw=list(target_pose_world_xyzw), gripper=int(gripper), max_steps=int(max_steps))
        if output_dir is not None:
            kwargs["output_dir"] = str(output_dir)
        return self.request("action", **kwargs)

    def ik(self, target_pose_world_xyzw, seed_joints=None, max_iterations=120):
        kwargs = dict(target_pose_world_xyzw=list(target_pose_world_xyzw), max_iterations=int(max_iterations))
        if seed_joints is not None:
            kwargs["seed_joints"] = list(seed_joints)
        return self.request("ik", **kwargs)

    def close(self):
        response = {}
        if self.process.poll() is None:
            try:
                response = self.request("close")
                self.process.wait(timeout=10)
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired, BrokenPipeError):
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        if self.log_handle:
            self.log_handle.close()
        return response

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
