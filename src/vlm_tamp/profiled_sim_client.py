"""Timed simulator RPCs, with optional whole-trajectory execution."""
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import numpy as np
from .coupled_client import CoupledClient
from .coupled_storage import read_json
from .runtime_paths import simulator_python
from .execution_limits import control_limit


class ProfiledClient(CoupledClient):
    def __init__(self,log_path=None,timeout=240,*,fast=True,timeline=None,python=None):
        root=Path(__file__).resolve().parents[2]
        self.python=simulator_python(python);self.timeout=float(timeout)
        self.log_handle=open(log_path,'a') if log_path else None;self.timeline=timeline;self.rpc_events=[];self.fast=fast
        env=os.environ.copy()
        for k in list(env):
            if k.endswith(('_API_KEY','_TOKEN','_SECRET')) or k in {'HF_TOKEN','OPENAI_ADMIN_KEY'}:env.pop(k,None)
        env['PYTHONPATH']=str(root/'src');env['VLM_TAMP_BATCH_IO']='1' if fast else '0'
        self.process=subprocess.Popen([self.python,'-m','vlm_tamp.profiled_sim_worker'],cwd=root,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log_handle,text=True,bufsize=1)
        self.messages=queue.Queue();self.request_id=0;self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
    def request(self,operation,**kwargs):
        before=time.perf_counter();start=self.timeline.now() if self.timeline else None
        result=super().request(operation,**kwargs)
        if 'metadata_file' in result:result['metadata']=read_json(result['metadata_file'])
        row={'operation':operation,'seconds':time.perf_counter()-before,'start_s':start,
            'control_steps':result.get('control_steps'),'worker':result.get('worker_profile',{})}
        self.rpc_events.append(row)
        if self.timeline:self.timeline.marks.append({'name':'rpc_complete','at_s':self.timeline.now(),**row})
        return result
    def joint_trajectory(self,references,gripper=1,stop_on_success=False):
        return self.request('joint_trajectory',references=np.asarray(references,float).tolist(),gripper=int(gripper),stop_on_success=bool(stop_on_success))
    def observe(self,output_dir,camera_names=None):
        return self.request('observe',output_dir=str(output_dir),**({'camera_names':camera_names} if camera_names is not None else {}))


def execute_batch(sim,state,references,*,gripper=1,maximum_control_steps=None,event_sink=None,stop_on_success=False):
    refs=np.asarray(references,float)
    if refs.ndim!=2 or refs.shape[1]!=7 or not np.isfinite(refs).all():raise ValueError('Finite seven-joint references required')
    limit=control_limit(state) if maximum_control_steps is None else min(control_limit(state),maximum_control_steps)
    if len(refs)>limit-int(state['control_steps']):raise ValueError('Complete trajectory exceeds remaining budget')
    before=time.perf_counter();state=sim.joint_trajectory(refs,gripper=gripper,**({'stop_on_success':True} if stop_on_success else {}));done=state.pop('trajectory')
    done['wall_seconds']=time.perf_counter()-before
    if event_sink:
        for event in done['events']:event_sink(event)
    return state,done
