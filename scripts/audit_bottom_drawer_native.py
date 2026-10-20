"""Evaluation-only native contact audit of rejected bottom-drawer poses.

No simulator geometry or labels are passed to the planner. Hypothetical robot
joints are assigned only in a private clone/render environment; no task success
or grasp retention is inferred from these static configurations.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json,compact_episode
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.profiled_sim_worker import ProfiledWorker
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.geometry import pose_matrix


def compact_static_audit(folder):
    """Retain the source pointer, native poses and renders, not a duplicate depth reset."""
    folder = Path(folder).resolve()
    if not folder.is_relative_to(ROOT / 'results') or folder == ROOT / 'results':
        raise ValueError('Static-audit cleanup is restricted to experiment folders')
    summary = read_json(folder / 'summary.json')
    if not summary.get('evaluation_render_only'):
        raise ValueError('Static-audit cleanup must not apply to physical episodes')
    if 'storage' not in summary:
        if summary.get('status') == 'setup_failed' and summary.get('control_steps') == 0:
            # A setup-only run has one still frame, not a physical rollout.
            # Preserve that frame and the error rather than requiring a movie.
            size = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
            summary['storage'] = {'bytes_before_cleanup': size, 'bytes_reclaimed_this_pass': 0}
            write_json(folder / 'retention.json', {'setup_failure_only': True, 'single_reset_frame_retained': True})
        else:
            summary['storage'] = compact_episode(folder)
    removed = []
    path = folder / 'initial/obs.npz'
    if path.exists():
        removed.append({'path': 'initial/obs.npz', 'bytes': path.stat().st_size,
                        'reason': 'Static renderer reset; original policy RGB-D retained in source episode'})
        path.unlink()
    retention = read_json(folder / 'retention.json')
    retention['static_audit_cleanup'] = retention.get('static_audit_cleanup', []) + removed
    retention['raw_observations_retained'] = []
    metadata = folder / 'initial/metadata.json'
    if metadata.exists() and metadata.stat().st_size > 1024 * 1024:
        raw = metadata.read_bytes()
        compressed = metadata.with_suffix('.json.gz')
        compressed.write_bytes(gzip.compress(raw, mtime=0))
        if gzip.decompress(compressed.read_bytes()) != raw:
            raise ValueError('Static metadata recompression must be byte-identical')
        metadata.unlink()
        saved = len(raw) - compressed.stat().st_size
        retention['static_audit_metadata_compression'] = {
            'path': 'initial/metadata.json.gz', 'original_sha256': hashlib.sha256(raw).hexdigest(),
            'bytes_reclaimed': saved, 'byte_identical': True}
        summary['storage']['bytes_reclaimed_this_pass'] += saved
    write_json(folder / 'retention.json', retention)
    summary['storage']['bytes_reclaimed_this_pass'] += sum(r['bytes'] for r in removed)
    summary['storage']['raw_observations_retained'] = []
    summary['storage']['bytes_after_cleanup'] = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    write_json(folder / 'summary.json', summary)
    return removed


def contacts(model,data,cloud):
    import mujoco
    tree=cKDTree(cloud);rows=[]
    for contact in data.contact[:data.ncon]:
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(i)) or '' for i in [contact.geom1,contact.geom2]]
        robot=[n.startswith(('robot0_','gripper0_')) for n in names]
        if sum(robot)!=1:continue
        point=np.asarray(contact.pos).copy();robot_name=names[robot.index(True)]
        distance=float(tree.query(point)[0])
        rows.append({'names_evaluation_only':names,'robot_part':robot_name,'scene_part':names[robot.index(False)],
            'signed_distance_m':float(contact.dist),'point_world_m':point.tolist(),
            'distance_to_predicted_handle_m':distance,
            'finger_near_selected_handle':bool('finger' in robot_name and distance<.008)})
    return rows


def ray_target(model,data,metadata,choices):
    import mujoco
    c=metadata['cameras'][choices['camera']];t=np.asarray(c['camera_to_world']);k=np.asarray(c['intrinsic'])
    ray=t[:3,:3]@np.linalg.solve(k,np.r_[choices['grasp_pixel_xy'],1.]);ray/=np.linalg.norm(ray)
    geom=np.array([-1],np.int32)
    distance=mujoco.mj_ray(model,data,t[:3,3],ray,None,1,-1,geom)
    return {'distance_m':float(distance),'geom_name_evaluation_only':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(geom[0])) if geom[0]>=0 else None,
        'point_world_m':(t[:3,3]+distance*ray).tolist() if distance>=0 else None,'policy_input':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='results/family-transfer-frozen-20261016')
    parser.add_argument('--output',required=True);parser.add_argument('--states',nargs='+',type=int,default=[5,15,25,45])
    parser.add_argument('--tasks',nargs='+',choices=['open_bottom_drawer','close_bottom_drawer'],default=['open_bottom_drawer','close_bottom_drawer'])
    parser.add_argument('--reason')
    parser.add_argument('--maximum',type=int,default=256);args=parser.parse_args()
    root=ROOT/args.run;out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    for name in list(os.environ):
        if name.endswith(('_API_KEY','_TOKEN','_SECRET')):os.environ.pop(name,None)
    import mujoco
    all_rows=[];worker=ProfiledWorker(fast=True)
    try:
        for label in args.tasks:
            for index in args.states:
                source=root/f'{label}-state{index:03d}';folder=out/source.name;folder.mkdir()
                summary=read_json(source/'summary.json');spec=read_json(source/'task-spec.json')
                saved=read_json(source/'opened/metadata.json');choices=read_json(source/'grounding/grounding.json')['choices']
                cloud=np.load(source/'grasp-cloud.npy');chain=RobotChain(read_json(source/'robot-chain.json'))
                state=worker.reset(summary['official_task_name'],index,folder/'initial',seed=0,suite_name=summary['suite'])
                state=worker.action(worker._eef_pose(),gripper=-1,max_steps=60)
                sim=worker.env.sim;model,data=sim.model._model,sim.data._data
                arm=np.asarray(worker.robot._ref_joint_pos_indexes);fingers=np.asarray(worker.robot._ref_gripper_joint_pos_indexes)
                baseline={n:np.asarray(getattr(data,n)).copy() for n in ['qpos','qvel','act','ctrl','mocap_pos','mocap_quat']}
                clone=mujoco.MjData(model)
                for n,a in baseline.items():getattr(clone,n)[:]=a
                clone.time=data.time;mujoco.mj_forward(model,clone)
                input_audit=ray_target(model,clone,saved,choices)
                # Capture task-object joints and geom names for interpretation
                # only, kept in this evaluation directory.
                articulation=worker._evaluation_contacts()['evaluation_only_articulation_state']
                target_joints=[j for j in articulation if 'cabinet' in j['joint_name'] and 'bottom' in j['joint_name']]
                predicted=np.asarray(spec['source_anchor_world_m']);normal=np.asarray(spec['panel_normal_world'])
                admission=read_json(source/'grasp-admission.json');rejected=[r for r in admission['rejected_candidates'] if r.get('joints') and (not args.reason or r['reason']==args.reason)][:args.maximum]
                rows=[];selected_render_indices={}
                for ordinal,row in enumerate(rejected):
                    clone.qpos[:]=baseline['qpos'];clone.qvel[:]=0
                    aperture=saved['gripper_qpos']
                    if row['reason']=='closure_scene_collision':
                        from vlm_tamp.rekep_libero.collision_paths_v7 import estimated_contact_aperture
                        aperture=estimated_contact_aperture(worker,row['joints'],cloud,saved['gripper_qpos'])
                        if aperture is None:raise ValueError('Cannot reconstruct the original closure-check aperture')
                    clone.qpos[arm]=row['joints'];clone.qpos[fingers]=aperture
                    mujoco.mj_forward(model,clone)
                    expected=pose_matrix(chain.forward(row['joints']));position=clone.site_xpos[int(worker.robot.eef_site_id)]
                    if np.linalg.norm(position-expected[:3,3])>2e-5:raise ValueError('Native TCP differs from frozen calibration')
                    pairs=contacts(model,clone,cloud);penetrations=[c for c in pairs if c['signed_distance_m']<-.0001]
                    other=[c for c in penetrations if not c['finger_near_selected_handle']]
                    record={'ordinal':ordinal,'detector_index':row['index'],'ik_rank':row.get('ik_rank'),
                        'joint_positions':row['joints'],'proposed_tcp':row['pose'],'gripper_qpos':clone.qpos[fingers].tolist(),
                        'planner_reason':row['reason'],'planner_diagnostic':row.get('diagnostic'),
                        'aperture_matches_original_check':True,
                        'native_contacts':pairs,'native_penetrations_over_0_1mm':len(penetrations),
                        'native_penetrations_outside_local_finger_contact':len(other),
                        'deepest_native_contact_m':min((c['signed_distance_m'] for c in pairs),default=0.)}
                    rows.append(record)
                    kind='native_clear' if not penetrations else 'local_finger_contact_only' if not other else 'other_penetration'
                    if ordinal==0:selected_render_indices['first_proposed']=ordinal
                    selected_render_indices.setdefault(kind,ordinal)
                    part=row.get('diagnostic',{}).get('last_rejected_part',-1)
                    if part>=1000:selected_render_indices.setdefault('gripper_model_rejection',ordinal)
                renders=[]
                for key,ordinal in selected_render_indices.items():
                    row=rows[ordinal]
                    data.qpos[:]=baseline['qpos'];data.qvel[:]=0;data.qpos[arm]=row['joint_positions'];data.qpos[fingers]=row['gripper_qpos'];sim.forward()
                    front,_,_=worker._frontview();Image.fromarray(front).save(folder/(key+'-front.png'))
                    rgb=sim.render(camera_name='agentview',width=512,height=512);Image.fromarray(rgb[::-1]).save(folder/(key+'-agent.png'))
                    volumes=worker.ik_solver.collision_volumes(data=data)
                    write_json(folder/(key+'-robot-volumes.json'),[v for v in volumes if 'gripper' in v['robot_geom_name']])
                    renders.append({'kind':key,'ordinal':ordinal,'images':[key+'-front.png',key+'-agent.png']})
                for name,value in baseline.items():getattr(data,name)[:]=value
                sim.forward()
                if not np.array_equal(data.qpos,baseline['qpos']):raise ValueError('Evaluation renderer did not restore its private scene')
                record={'case':source.name,'source':str(source),'evaluation_only':True,'hypothetical_poses_executed':False,
                    'simulator_object_information_used_for_policy':False,'measured_grasp_cloud_bounds_m':np.quantile(cloud,[.01,.99],axis=0).tolist(),
                    'fitted_plane_tilt_from_vertical_deg':float(np.degrees(np.arcsin(abs(normal[2])))),
                    'predicted_anchor_world_m':predicted.tolist(),'contact_pixel_ray_evaluation':input_audit,
                    'target_joints_evaluation_only':target_joints,'audited_poses':len(rows),
                    'poses_without_native_penetration':sum(r['native_penetrations_over_0_1mm']==0 for r in rows),
                    'poses_with_only_local_finger_penetration':sum(r['native_penetrations_over_0_1mm']>0 and r['native_penetrations_outside_local_finger_contact']==0 for r in rows),
                    'poses_with_other_native_penetration':sum(r['native_penetrations_outside_local_finger_contact']>0 for r in rows),
                    'renders':renders,'poses':rows}
                write_json(folder/'native-audit.json',record);all_rows.append({k:v for k,v in record.items() if k not in {'poses','renders'}})
                worker.close();write_json(folder/'summary.json',{'task':label,'initial_state':index,'status':'static_evaluation_only',
                    'control_steps':60,'official_task_success':False,'evaluation_render_only':True,'no_grasp_execution':True})
                compact_static_audit(folder)
                print(json.dumps(all_rows[-1]),flush=True)
    finally:worker.close()
    write_json(out/'summary.json',{'cases':all_rows,'new_api_calls':0,'scope':'evaluation-only native geometry checks; no dynamics or task success inferred'})


if __name__=='__main__':main()
