"""Full-arc backend and explicit frozen-grasp diagnostic inputs."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .coupled_storage import read_json,write_json,preserve_scene
from .mcvamp_contact_numerical import NativeScene
from .arc_execution import contact_arc
from .rekep_libero.geometry import pose_matrix




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
