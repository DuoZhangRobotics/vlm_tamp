"""Configured arm geometry must reject real occupancy and avoid sphere overhang."""
from pathlib import Path
import numpy as np
import pytest
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.mcvamp_contact_refined import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/all-prior-tasks-frozen-v3-20261015'


@pytest.mark.parametrize('offset',[0.,.2])
@pytest.mark.parametrize('contact',[False,True])
def test_configured_arm_collision_cannot_use_finger_contact_permission(offset,contact):
    folder=BASE/'open_middle_drawer-state001';record=read_json(folder/'robot-chain.json');meta=read_json(folder/'opened/metadata.json')
    chain=RobotChain(record);q=np.asarray(meta['joint_positions']);q[0]+=offset
    volumes=RobotVolumes(chain,meta['robot_collision_volumes_world'],meta['joint_positions']).forward(q)
    current={**meta,'joint_positions':q.tolist(),'eef_pose_world_xyzw':chain.forward(q).tolist(),'robot_collision_volumes_world':volumes}
    spec={'joint_origin_world_m':[0,0,1.],'joint_axis_world':[0,0,1.],
        'front_plane':{'point_world_m':[0,0,1.]},'panel_normal_world':[0,1,0]}
    point=np.asarray(next(v for v in volumes if v['robot_geom_name']=='robot0_link4_collision')['vertices_world']).mean(0)
    far=np.array([[3.,3.,3.]])
    clean=Scene.from_points(np.empty((0,3)),far,far,spec)
    with NativeScene(clean,record,current,q,.1,articulated=False) as ctx:assert ctx.validate(q)['accepted']
    scene=Scene.from_points(np.empty((0,3)) if contact else point[None],point[None] if contact else far,
        point[None] if contact else far,spec)
    with NativeScene(scene,record,current,q,.1,articulated=False) as ctx:
        result=ctx.validate(q);assert not result['accepted'] and result['last_rejected_part']>=2001
    clear_native_cache()


@pytest.mark.parametrize('state,plan',[(21,'plan-08'),(31,'plan-10')])
def test_saved_false_collision_rejections_are_resolved_by_configured_geometry(state,plan):
    folder=BASE/f'close_microwave-state{state:03d}';p=folder/plan/'horizon-0';data=read_json(p/'planning-input.json');ref=read_json(p/'scene-reference.json')
    record=read_json(folder/'robot-chain.json');opened=read_json(folder/'opened/metadata.json');chain=RobotChain(record)
    volumes=RobotVolumes(chain,opened['robot_collision_volumes_world'],opened['joint_positions']).forward(data['reference_joints'])
    meta={**data['metadata'],'robot_collision_volumes_world':[v for v in volumes if 'gripper' not in v['robot_geom_name']]+data['metadata']['robot_collision_volumes_world']}
    with np.load(ref['arrays_path']) as a:
        scene=Scene.from_points(a['static_points'],a['moving_reference'],a['contact_reference'],data['task_spec'],config=ref['configuration'])
    with NativeScene(scene,record,meta,data['reference_joints'],data['goal_angle_rad']) as ctx:assert ctx.validate(data['reference_joints'])['accepted']
    clear_native_cache()
