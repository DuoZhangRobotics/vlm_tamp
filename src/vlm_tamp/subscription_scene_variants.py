"""Trusted simulator-only scene perturbations; never supplied as policy geometry."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation

VARIANTS={
    'nominal':{},
    'translated':{'microwave_translation_m':[.04,.02,0.]},
    'yaw_positive':{'microwave_yaw_degrees':12.},
    'yaw_negative_shifted':{'microwave_yaw_degrees':-12.,'microwave_translation_m':[-.03,.02,0.]},
    'frontview_shifted':{'frontview_position_delta_m':[.12,0.,.03]},
    'clutter_shifted':{'plate_translation_m':[-.10,.05,0.]},
}

def transform_pose(position,quaternion_wxyz,origin,rotation,translation):
    position=np.asarray(position,float);q=np.asarray(quaternion_wxyz,float)
    p=np.asarray(origin)+rotation.apply(position-origin)+np.asarray(translation)
    out=(rotation*Rotation.from_quat(q[[1,2,3,0]])).as_quat()
    return p,out[[3,0,1,2]]

def apply_scene_variant(env,name,frontview_policy):
    if name not in VARIANTS:raise ValueError('Unknown predeclared scene variant')
    settings=VARIANTS[name];record={'name':name,'settings':copy.deepcopy(settings),
        'scope':'simulation initialization only; native success predicate unchanged; not exposed to policy',
        'changed_bodies':[]}
    policy=copy.deepcopy(frontview_policy)
    if name=='nominal':return record,policy
    sim=env.sim;model=sim.model;data=sim.data;inner=env.env
    microwave=inner.fixtures_dict['microwave_1'];root=model.body_name2id(microwave.root_body)
    if model.body_parentid[root]!=0 or model.body_jntnum[root]!=0:
        raise ValueError('Expected a fixed world-parent microwave fixture')
    origin=np.asarray(model.body_pos[root]).copy()
    rotation=Rotation.from_euler('z',settings.get('microwave_yaw_degrees',0.),degrees=True)
    translation=np.asarray(settings.get('microwave_translation_m',[0.,0.,0.]))
    rigid_change=bool(np.any(translation) or settings.get('microwave_yaw_degrees',0.))
    if rigid_change:
        p,q=transform_pose(model.body_pos[root],model.body_quat[root],origin,rotation,translation)
        model.body_pos[root]=p;model.body_quat[root]=q
        record['changed_bodies'].append({'name':microwave.root_body,'position':p.tolist(),'quaternion_wxyz':q.tolist()})
    for object_name in ['white_bowl_1','plate_1']:
        obj=inner.objects_dict[object_name]
        joint=model.joint_name2id(obj.joints[0]);address=int(model.jnt_qposadr[joint])
        if model.jnt_type[joint]!=0:raise ValueError('Expected free object joint')
        p=data.qpos[address:address+3].copy();q=data.qpos[address+3:address+7].copy();changed=False
        if object_name=='white_bowl_1' and rigid_change:
            p,q=transform_pose(p,q,origin,rotation,translation);changed=True
            dof=int(model.jnt_dofadr[joint]);data.qvel[dof:dof+3]=rotation.apply(data.qvel[dof:dof+3]);data.qvel[dof+3:dof+6]=rotation.apply(data.qvel[dof+3:dof+6])
        key='bowl_local_translation_m' if object_name=='white_bowl_1' else 'plate_translation_m'
        if key in settings:
            offset=np.asarray(settings[key])
            if object_name=='white_bowl_1':offset=Rotation.from_quat(model.body_quat[root][[1,2,3,0]]).apply(offset)
            p+=offset;changed=True
        if changed:
            data.qpos[address:address+3]=p;data.qpos[address+3:address+7]=q
            record['changed_bodies'].append({'name':obj.root_body,'position':p.tolist(),'quaternion_wxyz':q.tolist()})
    if 'frontview_position_delta_m' in settings:
        policy['position_world_m']=(np.asarray(policy['position_world_m'])+settings['frontview_position_delta_m']).tolist()
        policy['source']='predeclared calibrated camera placement'
    sim.forward()
    record['frontview_policy']=policy
    record['initial_native_success']=bool(env.check_success())
    return record,policy
