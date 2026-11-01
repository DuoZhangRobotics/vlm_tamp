"""Exercise the compiled planner's sphere/contact policy without a simulator."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vlm_tamp.coupled_scene import Scene
from vlm_tamp.mcvamp_contact_numerical import NativeScene, clear_native_cache
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.vamp_sphere_model import native_spheres, ARM_SPHERE_RANGES, PALM_SPHERE_RANGE

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def robot():
    vamp=pytest.importorskip('vamp')
    record=json.loads((ROOT/'tests/fixtures/calibrated_panda.json').read_text())
    # Robot-base transform from the retained calibration; no object geometry.
    world=np.eye(4);world[:3,3]=[-.66,0.,.912]
    record['calibration']={'world_to_robot':np.linalg.inv(world).tolist()}
    chain=RobotChain(record);q=chain.reference_joints;tcp=pose_matrix(chain.forward(q))
    metadata={'joint_positions':q.tolist(),'eef_pose_world_xyzw':chain.forward(q).tolist(),
        'robot_collision_volumes_world':[
            {'robot_geom_name':f'gripper0_finger{i}_pad_collision',
             'vertices_world':[(tcp[:3,3]+tcp[:3,:3]@np.array([offset,0,0])).tolist()]}
            for i,offset in [(1,-.02),(2,.02)]]}
    values=native_spheres(q);values[:,:3]=values[:,:3]@world[:3,:3].T+world[:3,3]
    yield record,metadata,values,chain,world
    clear_native_cache()


def scene_for(point,role):
    far=np.array([[5.,5.,5.]]);point=np.asarray(point)[None]
    spec={'joint_origin_world_m':[0,0,0],'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':[0,0,0]},'panel_normal_world':[1,0,0]}
    return Scene.from_points(point if role=='stationary' else far,
        point if role!='stationary' else far,point if role=='contact' else far,spec)


def test_sphere_partition_matches_pinned_upstream_robot():
    links=ET.parse(ROOT/'third_party/mc-vamp/resources/panda/panda_spherized.urdf').getroot().findall('link')
    index=0;parts={}
    for link in links:
        n=len(link.findall('collision'))
        if n:parts[link.attrib['name']]=(index,index+n);index+=n
    assert tuple(parts[f'panda_link{i}'] for i in range(1,8))==ARM_SPHERE_RANGES
    assert parts['panda_hand']==PALM_SPHERE_RANGE
    assert parts['panda_leftfinger']==(55,57) and parts['panda_rightfinger']==(57,59)


@pytest.mark.parametrize('role',['stationary','moving','contact'])
@pytest.mark.parametrize('sphere,allowed',[(56,True),(40,False),(14,False)])
def test_native_uses_arm_palm_spheres_and_exempts_fingers_on_every_surface(robot,role,sphere,allowed):
    record,metadata,values,chain,world=robot
    point=values[sphere,:3]
    if allowed:
        assert np.min(np.linalg.norm(values[1:55,:3]-point,axis=1)-values[1:55,3])>0
    with NativeScene(scene_for(point,role),record,metadata,metadata['joint_positions'],.1,articulated=False) as context:
        result=context.validate(metadata['joint_positions'])
        assert result['accepted']==allowed,result
        assert context.record['configured_convex_arm_links']==0
        assert not context.record['finger_environment_collision_checked']
        assert context.record['point_radius_m']==0
        if not allowed:assert result['blocking_sphere_index'] in range(1,55)


def test_native_self_collision_matches_upstream_without_custom_geometry(robot):
    import vamp
    record,metadata,values,chain,world=robot;empty=vamp.Environment()
    with NativeScene(scene_for([5,5,5],'stationary'),record,metadata,metadata['joint_positions'],.1,articulated=False) as context:
        for q in np.random.default_rng(73).uniform(chain.limits[:,0],chain.limits[:,1],(24,7)):
            assert context.validate(q)['accepted']==bool(vamp.panda.validate(q.tolist(),empty,True))


def test_hand_pose_filter_uses_stock_palm_and_never_fingers(robot):
    record,metadata,values,chain,world=robot
    for sphere,allowed in [(56,True),(40,False)]:
        with NativeScene(scene_for(values[sphere,:3],'stationary'),record,metadata,metadata['joint_positions'],.1,articulated=False) as context:
            assert context.validate_hand_pose(metadata['eef_pose_world_xyzw'])['accepted']==allowed


def test_native_planner_owns_constraint_projection(robot,monkeypatch):
    record,metadata,values,chain,world=robot
    with NativeScene(scene_for([5,5,5],'stationary'),record,metadata,metadata['joint_positions'],.1,articulated=False) as context:
        monkeypatch.setattr(context,'project',lambda *a:pytest.fail('extra endpoint projection'))
        q=np.array(metadata['joint_positions']);goal=q.copy();goal[0]+=.01
        path,result=context.plan(q,goal,constrained=False,seconds=.2)
        assert result['exact_solution'],result
        assert not result['additional_dense_validation']
        np.testing.assert_allclose(path[[0,-1]],[q,goal],atol=1e-6)


@pytest.mark.parametrize('aligned',[False,True])
def test_native_point_mode_allows_jaw_parallel_to_hinge(robot,aligned):
    record,metadata,values,chain,world=robot
    q=np.asarray(metadata['joint_positions']);tcp=pose_matrix(chain.forward(q))
    # This orientation violated the former jaw-tilt rule. The requested point
    # is exactly on the circle, so point mode must accept it without rotation.
    axis=tcp[:3,0];origin=tcp[:3,3]-.2*tcp[:3,1]
    spec={'joint_origin_world_m':origin.tolist(),'joint_axis_world':axis.tolist(),
        'front_plane':{'point_world_m':origin.tolist()},'panel_normal_world':tcp[:3,2].tolist()}
    far=np.array([[5.,5.,5.]])
    scene=Scene.from_points(far,far,far,spec)
    with NativeScene(scene,record,metadata,q,.2,aligned=aligned,grasp_constraint='point') as context:
        assert context.constraint_errors(q)['accepted']
        assert not context.record['jaw_tilt_limit_applied']
        assert not context.record['orientation_constraint_applied']


@pytest.mark.parametrize('offset',[0.,.06])
def test_common_rigid_hand_door_motion_preserves_collision_even_with_wrong_pivot(robot,offset):
    from scipy.spatial.transform import Rotation
    from vlm_tamp.rekep_libero.geometry import matrix_pose,transform_points
    record,metadata,values,chain,world=robot;pose=pose_matrix(metadata['eef_pose_world_xyzw'])
    pivot=pose[:3,3]+np.array([.2+offset,.1,0.])
    rotation=Rotation.from_rotvec([0,0,.25]).as_matrix()
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=pivot-rotation@pivot
    for points in [values[40,:3][None],values[56,:3][None]]:
        outcomes=[]
        for cloud,hand in [(points,pose),(transform_points(points,transform),transform@pose)]:
            scene=scene_for(cloud[0],'moving')
            with NativeScene(scene,record,metadata,metadata['joint_positions'],.3,articulated=False) as context:
                outcomes.append(context.validate_hand_pose(matrix_pose(hand))['accepted'])
        assert outcomes[0]==outcomes[1]
