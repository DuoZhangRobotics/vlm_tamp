"""Admit part-relative contact modes with the existing MC-VAMP checker."""
from pathlib import Path
import numpy as np
from .coupled_storage import write_json
from .shared_contact import contact_menu
from .shared_drag_placement import rear_surface_contacts
from .mcvamp_mechanism import NativeScene
from .reliable_mechanisms import select_grasp as select_pinch, metadata_at, segment
from .rekep_libero.geometry import pose_matrix


def select_contact(generated,cloud,spec,scene,record,ik,state,sim,folder,config,*,strategy=None):
    if strategy is not None and strategy['contact_mode'] not in {'pinch','press_drag','push'}:
        raise ValueError('Unknown caller-selected contact mode')
    menu,evidence=contact_menu(cloud,spec,state['metadata'],np.vstack([scene.static_points,scene.moving_reference]),
                               depth_resolution=config['scene_voxel_spacing_m'],
                               mode_order=[strategy['contact_mode']] if strategy is not None else None)
    menu=rear_surface_contacts(menu,cloud,spec,state['metadata'])
    if strategy is not None and strategy.get('approach_tilt_deg') is not None:
        menu=[row for row in menu if row['approach_tilt_deg']==strategy['approach_tilt_deg']]
    if not menu:raise ValueError('Requested strategy has no candidate in the declared tool library')
    write_json(folder/'contact-menu.json',{'candidates':menu,'selection_evidence':evidence})
    attempts=[];tested_pinch=False;q0=np.asarray(state['metadata']['joint_positions'])
    for ordinal,row in enumerate(menu):
        if row['mode']=='pinch':
            if tested_pinch:continue
            tested_pinch=True
            try:
                chosen,path=select_pinch(generated,cloud,spec,scene,record,ik,state,sim,folder,config,
                                         **({'strategy':strategy} if strategy is not None and strategy.get('version')=='strategy-v2' else {}))
            except ValueError as error:
                attempts.append({'mode':'pinch','reason':str(error)});continue
            chosen.update(mode='pinch',contact_normal_world=row['contact_normal_world'],half_aperture_m=.04,
                          motion_tangent_world=motion_tangent(spec,chosen['tcp_pose_world_xyzw']))
            sim.contact_context(chosen,cloud)
            write_json(folder/'contact-mode-admission.json',{'selected':chosen,'attempts':attempts})
            return chosen,path
        aperture=[row['half_aperture_m'],-row['half_aperture_m']]
        metadata=metadata_at(sim,state['metadata'],q0,aperture)
        solutions=ik.solve_many(row['pose_world_xyzw'],q0,maximum=6)
        requested_rank=strategy.get('posture_rank') if strategy else None
        if requested_rank is not None:
            solutions=solutions[requested_rank:requested_rank+1]
            if not solutions:
                attempts.append({'candidate':ordinal,'mode':row['mode'],'reason':'requested_posture_unavailable',
                                 'requested_posture_rank':requested_rank})
        if not solutions:
            attempts.append({'candidate':ordinal,'mode':row['mode'],'reason':'no_robot_valid_ik'})
        with NativeScene(scene,record,metadata,q0,.03,articulated=False) as free:
            for rank,solution in enumerate(solutions):
                q=np.asarray(solution['joints']);attempt={'candidate':ordinal,'mode':row['mode'],
                    'ik_rank':solution.get('posture_rank',rank)}
                valid=free.validate(q)
                if not valid['accepted']:
                    attempts.append({**attempt,'reason':'contact_pose_collision','diagnostic':valid});continue
                trial=folder/f'contact-candidate-{ordinal:02d}-{rank:02d}';trial.mkdir()
                at_contact=metadata_at(sim,state['metadata'],q,aperture)
                progress=.025 if spec['joint_family']=='prismatic' else np.deg2rad(6.)
                if strategy and strategy.get('step_mm') is not None and spec['joint_family']=='prismatic':
                    progress=strategy['step_mm']/1000
                probe,probe_record=segment(scene,record,at_contact,ik,q,progress,trial/'probe',seconds=4.)
                if probe is None:
                    attempts.append({**attempt,'reason':'contact_probe_unplanned','planning':probe_record});continue
                pose=np.asarray(solution['pose_world_xyzw']);pre=pose.copy();pre[:3]-=.07*pose_matrix(pose)[:3,2]
                for pre_solution in ik.solve_many(pre,q,maximum=4):
                    pre_q=np.asarray(pre_solution['joints'])
                    first,one=free.plan(q0,pre_q,constrained=False,seconds=5.,iterations=4000)
                    if not one['exact_solution']:continue
                    last,two=free.plan(pre_q,q,constrained=False,seconds=4.,iterations=3000)
                    if not two['exact_solution']:continue
                    path=np.vstack([first,last[1:]])
                    selected={**row,'solution_joints':q.tolist(),'joints':q.tolist(),
                        'tcp_pose_world_xyzw':pose.tolist(),'predicted_gripper_qpos':aperture,
                        'ik_rank':solution.get('posture_rank',rank),'probe_distance':progress,
                        'probe':probe_record,'approach_planner':'VAMP RRTC','manipulation_planner':'MC-VAMP CRRTC',
                        'scene_admission_enabled':True,'physical_attachment_only':True}
                    selected['motion_tangent_world']=motion_tangent(spec,pose)
                    np.save(folder/'approach-path.npy',path)
                    write_json(folder/'contact-mode-admission.json',{'selected':selected,'attempts':attempts,
                        'palm_arm_contact_exempt':False,'allowed_contact':'selected fingers with observed target region'})
                    sim.contact_context(selected,cloud)
                    return selected,path
                attempts.append({**attempt,'reason':'free_approach_unplanned'})
    write_json(folder/'contact-mode-admission.json',{'selected':None,'attempts':attempts})
    raise ValueError('No contact mode has an admitted approach and manipulation probe')


def motion_tangent(spec,pose):
    axis=np.asarray(spec['joint_axis_world'],float)
    if spec['joint_family']=='prismatic':return (axis/np.linalg.norm(axis)).tolist()
    tangent=np.cross(axis,np.asarray(pose[:3])-spec['joint_origin_world_m'])
    if np.linalg.norm(tangent)<1e-8:
        tangent=np.cross(axis,[1,0,0] if abs(axis[0])<.8 else [0,1,0])
    return (tangent/np.linalg.norm(tangent)).tolist()
