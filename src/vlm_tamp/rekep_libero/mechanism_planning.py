"""Grasp and coupled-manifold admission across multiple cuRobo IK postures."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .collision_paths_v7 import check_joint_path, estimated_contact_aperture
from .geometry import interpolate_path, pose_matrix
from .physical_tasks_v4 import candidate_variants


def panel_probe_admission(scene,metadata,mechanism):
    """This rigid panel path is independent of which robot grasp is chosen."""
    horizon=min(mechanism.goal,.03 if mechanism.kind=='prismatic' else .20)
    samples=[]
    for progress in np.linspace(0.,horizon,4):
        result=scene.check(metadata['robot_collision_volumes_world'],metadata['eef_pose_world_xyzw'],
            stage='pull',moving_transform=mechanism.transform(float(progress)),check_payload=True)
        payload=[c for c in result['collisions'] if c['robot_geom_name']=='predicted_moving_source_envelope']
        samples.append({'progress':float(progress),'predicted_panel_admission_accepted':not payload,
                        'collisions':payload})
    predicted=all(row['predicted_panel_admission_accepted'] for row in samples)
    enforce=scene.config.get('scene_mode','strict')=='strict'
    return {'accepted':predicted or not enforce,'predicted_panel_admission_accepted':predicted,
        'scene_admission_enforced':enforce,'samples':samples,'grasp_independent':True,
        'scope':'predicted observed-panel envelope, not proof of physical task infeasibility'}


def mechanism_path_check(scene,sim,ik,qs,mechanism,grasp,aperture,closure=None):
    closure=np.eye(4) if closure is None else closure
    transform=lambda pose: mechanism.transform(mechanism.progress_of_pose(grasp,pose))@closure
    result=check_joint_path(scene,sim,ik,qs,gripper_qpos=aperture,
        transform_at_pose=transform,stage_at_pose=lambda pose:'pull',check_payload=True,config=ik.config)
    maximum_position,maximum_angle=0.,0.
    for a,b in zip(qs[:-1],qs[1:]):
        count=max(1,int(np.ceil(np.max(np.abs(np.asarray(b)-a))/.025)))
        for w in np.linspace(0,1,count+1):
            residual=mechanism.residual(grasp,ik.chain.forward(np.asarray(a)+w*(np.asarray(b)-a)))
            maximum_position=max(maximum_position,residual['position_residual_m'])
            maximum_angle=max(maximum_angle,residual['orientation_residual_rad'])
    manifold_ok=maximum_position<=.015 and maximum_angle<=.10
    return {**result,'accepted':result['accepted'] and manifold_ok,
        'manifold_residual_passed':manifold_ok,'maximum_manifold_position_residual_m':maximum_position,
        'maximum_manifold_orientation_residual_rad':maximum_angle}


def ranked_postures(batches):
    """Cover different grasp poses before spending the budget on more IK roots."""
    for alternative in range(max((len(solutions) for _,_,solutions in batches),default=0)):
        for ordinal,row,solutions in batches:
            if alternative<len(solutions):
                yield ordinal,row,alternative,solutions[alternative]


def choose_grasp(generated,cloud,calibration,ik,initial,sim,scene,mechanism,open_qpos,output,save):
    anchor=np.asarray(mechanism.spec['source_anchor_world_m'])
    tree=cKDTree(cloud)
    normal=np.asarray(mechanism.spec['panel_normal_world'])
    _,singular,vt=np.linalg.svd(cloud-cloud.mean(0),full_matrices=False)
    elongated=singular[0]>2*singular[1]
    eligible,rejected,chosen=[],[],None
    for index,variant,pose in candidate_variants(generated,calibration):
        row={'index':index,'variant':variant,'score':float(generated['scores'][index]),
             'tcp_pose_world_xyzw':pose.tolist(),'anchor_distance_m':float(np.linalg.norm(pose[:3]-anchor))}
        rotation=pose_matrix(pose)[:3,:3]
        if row['anchor_distance_m']>.12 or tree.query(pose[:3])[0]>.025:
            rejected.append({**row,'reason':'target_region_or_proximity'});continue
        if -float(rotation[:,2]@normal)<.65 or (elongated and abs(float(rotation[:,0]@vt[0]))>.65):
            rejected.append({**row,'reason':'contact_approach_or_closing_direction'});continue
        eligible.append(row)
    eligible.sort(key=lambda row:(row['anchor_distance_m'],-row['score'],row['index'],row['variant']))
    batches=[]
    for ordinal,row in enumerate(eligible):
        solutions=ik.solve_many(row['tcp_pose_world_xyzw'],initial,maximum=16)
        if not solutions:
            rejected.append({**row,'reason':'no_robot_valid_ik'})
        batches.append((ordinal,row,solutions))
    tested=0
    for ordinal,row,alternative,grasp in ranked_postures(batches):
        if tested>=ik.config['maximum_total_ik_postures']:
            break
        save(Path(output)/'grasp-admission-progress.json',{'eligible_ordinal':ordinal,'eligible_poses':len(eligible),
            'ik_alternative':alternative,'tested_ik_postures':tested,'selected':False})
        tested+=1
        evidence={**row,'ik_alternative':alternative,'joints':grasp['joints']}
        checked=scene.joints(sim,grasp['joints'],gripper_qpos=open_qpos,stage='grasp')
        if not checked['accepted']:
            rejected.append({**evidence,'reason':'grasp_scene_collision','clearance':checked});continue
        aperture=estimated_contact_aperture(sim,grasp['joints'],scene.region.contact_reference,open_qpos)
        if aperture is None:
            rejected.append({**evidence,'reason':'no_observed_contact_aperture'});continue
        checked=scene.joints(sim,grasp['joints'],gripper_qpos=aperture,stage='grasp')
        if not checked['accepted']:
            rejected.append({**evidence,'reason':'closure_scene_collision','clearance':checked});continue
        approach=ik.approach(np.asarray(initial),grasp)
        if not approach['accepted']:
            rejected.append({**evidence,'reason':approach['reason']});continue
        target=np.asarray(grasp['pose_world_xyzw'])
        checked=check_joint_path(scene,sim,ik,approach['joint_postures'],gripper_qpos=open_qpos,
            transform_at_pose=lambda pose:np.eye(4),
            stage_at_pose=lambda pose:'grasp' if np.linalg.norm(np.asarray(pose)[:3]-target[:3])<=.012 else 'approach',
            check_payload=False,config=ik.config)
        if not checked['accepted']:
            rejected.append({**evidence,'reason':'approach_scene_collision','path':checked});continue
        horizon=min(mechanism.goal,.03 if mechanism.kind=='prismatic' else .20)
        future=mechanism.path(target,end=horizon)
        path=ik.path(future[1:],grasp['joints'])
        if not path['accepted']:
            rejected.append({**evidence,'reason':path['reason']});continue
        qs=[grasp['joints']]+[item['joints'] for item in path['solutions']]
        task_check=mechanism_path_check(scene,sim,ik,qs,mechanism,target,aperture)
        if not task_check['accepted']:
            rejected.append({**evidence,'reason':'mechanism_path_collision_or_constraint','path':task_check});continue
        straight=interpolate_path([target,mechanism.pose(target,mechanism.goal)],position_step=.01,rotation_step=.05)
        residuals=[mechanism.residual(target,p) for p in straight]
        chosen={**evidence,'approach':approach,'predicted_gripper_qpos':aperture,
            'solution_joints':grasp['joints'],'fk_pose_world_xyzw':grasp['pose_world_xyzw'],
            'joint_reserve_rad':grasp['joint_reserve_rad'],'selection_ordinal':ordinal,
            'admission_horizon_progress':horizon,'full_goal_path_prevalidated':False,
            'task_path':path,'task_path_check':task_check,'tested_ik_postures':tested,
            'endpoint_only_geometric_comparison':{'executed':False,
                'maximum_estimated_manifold_position_residual_m':max(r['position_residual_m'] for r in residuals),
                'maximum_estimated_manifold_orientation_residual_rad':max(r['orientation_residual_rad'] for r in residuals)}}
        break
    save(Path(output)/'grasp-admission.json',{'selected':chosen,'rejected_candidates':rejected,
        'eligible_poses':len(eligible),'tested_ik_postures':tested,
        'posture_budget':ik.config['maximum_total_ik_postures'],'simulator_scene_state_used':False})
    if chosen is None:
        raise ValueError('IK posture search budget exhausted' if tested>=ik.config['maximum_total_ik_postures']
                         else 'No candidate passes contact, approach and coupled-probe admission')
    return chosen
