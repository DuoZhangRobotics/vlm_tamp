"""Process boundary keeps cuRobo's Torch/CUDA separate from LIBERO and GraspGen."""

import os
from pathlib import Path
import queue
import subprocess
import threading

from .sim_client import SimClient


class CuroboClient(SimClient):
    def __init__(self, model, config, log_path):
        root = Path(__file__).resolve().parents[3]
        self.backend = config.get('curobo_backend', 'v1')
        if self.backend not in {'v1', 'v2'}:
            raise ValueError('curobo_backend must be v1 or v2')
        prefix = 'curobo-v2' if self.backend == 'v2' else 'curobo-libero'
        self.python = str(root/f'.runtime/envs/{prefix}/bin/python')
        self.timeout = 600 if self.backend == 'v2' else 240
        self.log_handle = open(log_path, 'a')
        environment = os.environ.copy()
        for name in list(environment):
            if name.endswith(('_API_KEY', '_TOKEN', '_SECRET')) or name in {'HF_TOKEN', 'OPENAI_ADMIN_KEY'}:
                environment.pop(name, None)
        environment['PYTHONPATH'] = str(root/'src')
        environment.setdefault('CUDA_HOME', '/usr/local/cuda')
        environment['TORCH_EXTENSIONS_DIR'] = str(root/'.runtime/curobo-torch-extensions')
        environment['XDG_CACHE_HOME'] = str(root/('.runtime/curobo-v2-cache' if self.backend == 'v2' else '.runtime/curobo-cache'))
        worker = 'curobo_v2_worker' if self.backend == 'v2' else 'curobo_worker'
        self.process = subprocess.Popen([self.python, '-m', f'vlm_tamp.rekep_libero.{worker}'],
            cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log_handle, text=True, bufsize=1)
        self.messages, self.request_id = queue.Queue(), 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.metadata = self.request('initialize', robot_config=model, config=config)
        except BaseException:
            self.close()
            raise

    def solve(self, pose, seed):
        return self.request('solve', pose=list(pose), seed=list(seed))

    def forward(self, joints):
        return self.request('fk', joints=joints)['poses_world_xyzw']
