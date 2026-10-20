"""Full-arc backend and explicit frozen-grasp diagnostic inputs."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .coupled_storage import read_json,write_json,preserve_scene
from .mcvamp_microwave import NativeScene
from .partial_pose_arc import full_arc_plan
from .rekep_libero.geometry import pose_matrix


def contact_arc(scene,record,metadata,start,goal,folder,config):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    preserve_scene(scene,folder)
    compact={key:metadata[key] for key in ['joint_positions','eef_pose_world_xyzw']}
    compact['robot_collision_volumes_world']=[v for v in metadata['robot_collision_volumes_world'] if 'gripper' in v['robot_geom_name']]
    write_json(folder/'planning-input.json',{'task_spec':scene.spec,'metadata':compact,
        'reference_joints':start,'goal_angle_rad':goal,'planning_only':True,
        'backend':'continuous_partial_pose_arc','end_effector_quaternion_prescribed':False,
        'search_options':{'motion_policy':config.get('motion_policy'),
            'hinge_refinement_accepted':bool(config.get('hinge_refinement_accepted',False)),
            'arc_search_seconds':float(config.get('arc_search_seconds',60.)),
            'arc_goal_proposals':int(config.get('arc_goal_proposals',128)),
            'arc_seed':int(config.get('arc_seed',0))},
        'refined_flag_semantics':'Selects wrist-excursion allowance; it does not perform a new hinge estimate.',
        'reference_semantics':'Goal angle and moving-cloud rotation are relative to this query start; cloud pose must already correspond to that start.'})
    with NativeScene(scene,record,metadata,start,goal,aligned=False) as context:
        context.motion_options=config
        path,result=full_arc_plan(context,goal,seconds=float(config.get('arc_search_seconds',60.)),
            proposals=int(config.get('arc_goal_proposals',128)),seed=int(config.get('arc_seed',0)))
        result['exact_solution']=bool(len(path)>1 and result['status']=='solved')
        write_json(folder/'result.json',result);write_json(folder/'native-scene.json',context.record)
        if not result['exact_solution']:
            raise ValueError('Continuous partial-pose full-arc search: '+result['status'])
        np.save(folder/'path.npy',path);np.save(folder/'door-progress.npy',[context.progress(q) for q in path])
        return path,result


def frozen_grasp(source,state,ik,record,open_scene,closed_scene,folder):
    """Replay only a sensor-validated original approach/grasp, never its pull."""
    source=Path(source);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    previous=read_json(source/'summary.json');metadata=read_json(source/'opened/metadata.json')
    initial=np.asarray(state['metadata']['joint_positions'])
    difference=float(np.max(np.abs(initial-metadata['joint_positions'])))
    if difference>1e-7:raise ValueError('Frozen-grasp initial proprioception mismatch')
    from try_rekep_bowl_v3 import read_observation_v3
    cameras,_=read_observation_v3(folder.parent/'opened',state['metadata']);rgb={}
    for name in ['agentview','frontview']:
        expected=np.asarray(Image.open(source/'opened'/f'{name}.png'))
        rgb[name]=float(np.max(np.abs(cameras[name]['rgb'].astype(float)-expected)))
    if any(rgb.values()):raise ValueError('Frozen-grasp initial RGB mismatch')
    if not np.array_equal(np.load(source/'handle-cloud.npy'),np.load(folder.parent/'handle-cloud.npy')):
        raise ValueError('Frozen-grasp predicted handle cloud mismatch')
    selected=read_json(source/'chosen-grasp.json')
    approach=np.load(source/'selection-bounded-segment/approach.npy')
    contact=np.load(source/'pregrasp-complete-path.npy');q=np.asarray(selected['joints'])
    goal=float(previous['selected_initial_horizon_rad'])
    with NativeScene(open_scene,record,state['metadata'],initial,goal,articulated=False) as context:
        approach_valid=all(context.validate(row)['accepted'] for row in approach)
    predicted={**state['metadata'],'joint_positions':q.tolist(),'eef_pose_world_xyzw':ik.chain.forward(q).tolist(),
               'robot_collision_volumes_world':closed_scene.robot.forward(q)}
    with NativeScene(closed_scene,record,predicted,q,goal,aligned=False) as context:
        contact_valid=all(context.validate(row)['accepted'] for row in contact)
    write_json(folder/'replay-evidence.json',{'source':str(source),'same_rgb':rgb,
        'maximum_initial_joint_difference_rad':difference,'same_predicted_handle_cloud':True,
        'approach_admitted_before_execution':approach_valid,'old_contact_path_admitted':contact_valid,
        'old_contact_path_executed':False,'postclosure_full_arc_planned_afresh':True,
        'source_approach_sha256':hashlib.sha256((source/'selection-bounded-segment/approach.npy').read_bytes()).hexdigest(),
        'physical_grasp_only':True,'not_independent_benchmark':True})
    if not approach_valid or not contact_valid:raise ValueError('Frozen input path fails current planning model')
    selected.update(approach=approach,planned_contact=contact,search_seconds=0.)
    np.save(folder/'approach.npy',approach)
    return selected,goal
