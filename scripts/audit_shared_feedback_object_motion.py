"""Evaluation-only replay of recorded actions to measure free-object motion.

Object measurements are accepted only if robot joint traces and native success
reproduce the recorded episode. Nothing from this audit is returned to a policy.
Temporary replay observations are removed after every run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_sim_worker import ProfiledWorker


def audit(source,out,task):
    import mujoco
    actions=read_json(source/'initial/actions.json');summary=read_json(source/'summary.json')
    out.mkdir(exist_ok=True,parents=True)
    scratch=Path(tempfile.mkdtemp(prefix='object-audit-',dir=out));worker=ProfiledWorker(fast=True)
    worker._save_frame=lambda *a,**k:None
    worker._write_action_log=lambda *a,**k:None
    try:
        worker.reset(task['name'],summary['initial_state'],scratch/'initial',seed=0,suite_name=task['suite'])
        model=worker.env.sim.model._model;data=worker.env.sim.data._data;free=[]
        for i in range(model.njnt):
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) or f'joint_{i}'
            if int(model.jnt_type[i])==int(mujoco.mjtJoint.mjJNT_FREE) and not name.startswith(('robot','gripper')):
                address=int(model.jnt_qposadr[i]);initial=data.qpos[address:address+7].copy()
                free.append({'name_evaluation_only':name,'address':address,'initial_pose':initial,'peak_translation_m':0.})
        switched=False;error=0.;success=False;per_step_success_matches=True
        for record in actions:
            if record.get('execution_controller')=='JOINT_POSITION' and not switched:
                worker.enable_joint_tracking(kp=300.);switched=True
            worker.env.step(np.asarray(record['control'],float));worker.control_steps+=1
            joints=np.asarray(data.qpos[worker.robot._ref_joint_pos_indexes])
            error=max(error,float(np.max(abs(joints-record['joint_positions']))))
            native=bool(worker.env.check_success());success|=native
            per_step_success_matches &= native==record['evaluation_task_success']
            for obj in free:
                current=data.qpos[obj['address']:obj['address']+3]
                obj['peak_translation_m']=max(obj['peak_translation_m'],float(np.linalg.norm(current-obj['initial_pose'][:3])))
        objects=[]
        for obj in free:
            current=data.qpos[obj['address']:obj['address']+7].copy()
            objects.append({'name_evaluation_only':obj['name_evaluation_only'],
                'initial_pose_world':obj['initial_pose'].tolist(),'final_pose_world':current.tolist(),
                'peak_translation_m':obj['peak_translation_m'],
                'final_translation_m':float(np.linalg.norm(current[:3]-obj['initial_pose'][:3]))})
        result={'source':str(source),'evaluation_only':True,'policy_inputs_modified':False,
            'maximum_robot_joint_replay_error_rad':error,'per_step_native_success_matches':bool(per_step_success_matches),
            'native_success_matches':success==summary['official_task_success'],
            'accepted_as_original_episode_measurement':bool(error<1e-5 and per_step_success_matches),
            'objects':objects,'replayed_steps':len(actions),'video_regenerated':False}
        write_json(out/'object-motion.json',result)
        return result
    finally:
        worker.close();removed=sum(p.stat().st_size for p in scratch.rglob('*') if p.is_file())
        shutil.rmtree(scratch)
        write_json(out/'cleanup.json',{'temporary_bytes_removed':removed,'temporary_replay_observations_retained':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--output',required=True)
    p.add_argument('--task-name',required=True);p.add_argument('--suite',required=True);a=p.parse_args()
    source=(ROOT/a.episode).resolve();out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Audit output must be inside results')
    result=audit(source,out,{'name':a.task_name,'suite':a.suite})
    print(json.dumps({k:v for k,v in result.items() if k!='objects'}))

if __name__=='__main__':main()
