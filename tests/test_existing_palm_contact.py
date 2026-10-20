"""Submillimetre existing contact is local; it is never an arm/world bypass."""
from pathlib import Path
import numpy as np
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/reliability-microwave-final-open-20261016/open_microwave-state001'


def inputs():
    path=FOLDER/'plan-00/horizon-0';data=read_json(path/'planning-input.json');ref=read_json(path/'scene-reference.json')
    record=read_json(FOLDER/'robot-chain.json');opened=read_json(FOLDER/'opened/metadata.json');chain=RobotChain(record)
    arm=RobotVolumes(chain,opened['robot_collision_volumes_world'],opened['joint_positions']).forward(data['reference_joints'])
    meta={**data['metadata'],'robot_collision_volumes_world':[v for v in arm if 'gripper' not in v['robot_geom_name']]+data['metadata']['robot_collision_volumes_world']}
    with np.load(ref['arrays_path']) as a:scene=Scene.from_points(a['static_points'],a['moving_reference'],a['contact_reference'],data['task_spec'],config=ref['configuration'])
    return scene,record,meta,data


def test_measured_73_micron_palm_contact_can_start_a_replan():
    scene,record,meta,data=inputs()
    with NativeScene(scene,record,meta,data['reference_joints'],data['goal_angle_rad']) as native:
        evidence=native.record['existing_palm_contact']
        assert evidence['enabled'] and evidence['initial_depth_m']<.0001
        assert evidence['tolerance_m']<.0002 and native.validate(data['reference_joints'])['accepted']
    clear_native_cache()


def test_same_shallow_overlap_on_unrelated_static_surface_remains_collision():
    scene,record,meta,data=inputs();point=np.array([[-.05652598664,-.07859881967,1.0658248663]])
    test=Scene.from_points(point,np.array([[3,3,3.]]),np.array([[3,3,3.]]),data['task_spec'])
    with NativeScene(test,record,meta,data['reference_joints'],.1) as native:
        assert not native.validate(data['reference_joints'])['accepted']
        assert not native.record['existing_palm_contact']['enabled']
    clear_native_cache()


def test_deeper_palm_overlap_is_not_accepted_as_existing_contact():
    scene,record,meta,data=inputs();point=np.array([-.05652598664,-.07859881967,1.0658248663])
    palm=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
    planes=np.array(palm['planes_world']);face=int(np.argmax(planes[:,:3]@point+planes[:,3]));point-=.001*planes[face,:3]
    test=Scene.from_points(np.empty((0,3)),point[None],point[None],data['task_spec'])
    with NativeScene(test,record,meta,data['reference_joints'],.1) as native:
        assert not native.validate(data['reference_joints'])['accepted']
        assert not native.record['existing_palm_contact']['enabled']
    clear_native_cache()


def test_releasing_obsolete_index_cache_keeps_live_context_and_results_valid():
    scene,record,meta,data=inputs();q=data['reference_joints'];goal=data['goal_angle_rad']
    with NativeScene(scene,record,meta,q,goal) as current:
        before=current.validate(q)
        clear_native_cache()
        after=current.validate(q)
        assert before['accepted']==after['accepted']==True
    with NativeScene(scene,record,meta,q,goal) as rebuilt:assert rebuilt.validate(q)['accepted']==before['accepted']
    clear_native_cache()
