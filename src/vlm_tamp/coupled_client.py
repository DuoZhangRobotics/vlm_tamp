"""Protected LIBERO process exposing physical joint reference tracking."""

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from .rekep_libero.sim_client_articulated import ArticulatedClient


class CoupledClient(ArticulatedClient):
    def __init__(self, log_path=None, timeout=240):
        root = Path(__file__).resolve().parents[2]
        self.python = '/home/duo/miniforge3/envs/vlm-tamp/bin/python'
        self.timeout = float(timeout)
        self.log_handle = open(log_path, 'a') if log_path else None
        environment = os.environ.copy()
        for name in list(environment):
            if name.endswith(('_API_KEY', '_TOKEN', '_SECRET')) or name in {'HF_TOKEN', 'OPENAI_ADMIN_KEY'}:
                environment.pop(name, None)
        environment['PYTHONPATH'] = str(root / 'src')
        self.process = subprocess.Popen([self.python, '-m', 'vlm_tamp.coupled_worker'],
            cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log_handle or sys.stderr, text=True, bufsize=1)
        self.messages, self.request_id = queue.Queue(), 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def enable_joint_tracking(self,kp=150.,reference_mode='measured_delta'):
        return self.request('enable_joint_tracking',kp=float(kp),reference_mode=reference_mode)

    def set_joint_reference_mode(self,mode):
        return self.request('set_joint_reference_mode',mode=mode)

    def joint_action(self, target_joints, gripper=1, max_steps=1, output_dir=None):
        arguments = {'target_joints': list(target_joints), 'gripper': gripper,
                     'max_steps': max_steps}
        if output_dir is not None:
            arguments['output_dir'] = str(output_dir)
        return self.request('joint_action', **arguments)
