"""Separate process for official articulated LIBERO suite tasks."""

import os
from pathlib import Path
import queue
import subprocess
import threading

from .sim_client_v8 import SimClientV8


class ArticulatedClient(SimClientV8):
    def __init__(self, log_path=None):
        root = Path(__file__).resolve().parents[3]
        self.python = '/home/duo/miniforge3/envs/vlm-tamp/bin/python'
        self.timeout = 240
        self.log_handle = open(log_path, 'a') if log_path else None
        environment = os.environ.copy()
        for name in list(environment):
            if name.endswith(('_API_KEY', '_TOKEN', '_SECRET')) or name in {'HF_TOKEN', 'OPENAI_ADMIN_KEY'}:
                environment.pop(name, None)
        environment['PYTHONPATH'] = str(root/'src')
        self.process = subprocess.Popen([self.python, '-m', 'vlm_tamp.rekep_libero.sim_worker_articulated'],
            cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log_handle, text=True, bufsize=1)
        self.messages, self.request_id = queue.Queue(), 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def reset(self, task_name, state_index, output_dir, seed=0, suite_name='libero_90'):
        return self.request('reset', task_name=task_name, state_index=state_index,
            output_dir=str(output_dir), seed=seed, suite_name=suite_name)
