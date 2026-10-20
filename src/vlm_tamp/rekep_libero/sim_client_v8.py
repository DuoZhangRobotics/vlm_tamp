"""Protected LIBERO process exposing hypothetical gripper geometry."""

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from .sim_client_v4 import SimClientV4


class SimClientV8(SimClientV4):
    def __init__(self, python=None, timeout=180, log_path=None):
        self.python = str(python or '/home/duo/miniforge3/envs/vlm-tamp/bin/python')
        self.timeout = float(timeout)
        self.log_handle = open(log_path, 'a') if log_path else None
        root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        for name in list(environment):
            if name.endswith(('_API_KEY', '_TOKEN', '_SECRET')) or name in {'HF_TOKEN', 'OPENAI_ADMIN_KEY'}:
                environment.pop(name, None)
        environment['PYTHONPATH'] = str(root / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
        self.process = subprocess.Popen(
            [self.python, '-m', 'vlm_tamp.rekep_libero.sim_worker_v8'], cwd=root, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_handle or sys.stderr,
            text=True, bufsize=1)
        self.messages = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.request_id = 0

    def fk(self, joints=None, include_volumes=False, gripper_qpos=None):
        kwargs = {'include_volumes': bool(include_volumes)}
        if joints is not None:
            kwargs['joints'] = list(joints)
        if gripper_qpos is not None:
            kwargs['gripper_qpos'] = list(gripper_qpos)
        return self.request('fk', **kwargs)
