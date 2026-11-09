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
from vlm_tamp.vamp_sphere_model import native_spheres, ARM_SPHERE_RANGES, PALM_SPHERE_RANGE, finger_environment_model

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def robot():
    pytest.importorskip('vamp')
    record=json.loads((ROOT/'tests/fixtures/calibrated_panda.json').read_text())
    # Robot-base transform from the retained calibration; no object geometry.
    world=np.eye(4);world[:3,3]=[-.66,0.,.912]
    record['calibration']={'world_to_robot':np.linalg.inv(world).tolist()}
    chain=RobotChain(record);q=chain.reference_joints;tcp=pose_matrix(chain.forward(q))
    metadata={'joint_positions':q.tolist(),'eef_pose_world_xyzw':chain.forward(q).tolist(),
        'gripper_qpos':[.04,-.04],
        'robot_collision_volumes_world':[
            {'robot_geom_name':f'gripper0_finger{i}_pad_collision',
             'vertices_world':[(tcp[:3,3]+tcp[:3,:3]@np.array([offset,0,0])).tolist()]}
            for i,offset in [(1,-.02),(2,.02)]]}
    from scipy.spatial.transform import Rotation
    from vlm_tamp.rekep_libero.geometry import matrix_pose
    for index,side in [(1,-1),(2,1)]:
        local=np.eye(4);local[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
        local[:3,3]=[side*.04,0.,-.0446]
        metadata['robot_collision_volumes_world'].append({
            'robot_geom_name':f'gripper0_finger{index}_collision',
            'body_pose_world_xyzw':matrix_pose(tcp@local).tolist()})
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


@pytest.mark.parametrize('role',['stationary','moving','contact'])
def test_free_approach_checks_measured_fingers_and_contact_phase_releases_them(robot,role):
    """The same physical finger obstacle changes policy only at the phase boundary."""
    record,metadata,values,chain,world=robot
    model=finger_environment_model(metadata,world)
    tcp=pose_matrix(metadata['eef_pose_world_xyzw'])
    finger=np.asarray(model['hand_spheres_tcp'])[-3]
    point=tcp[:3,:3]@finger[:3]+tcp[:3,3]
    assert np.min(np.linalg.norm(values[1:55,:3]-point,axis=1)-values[1:55,3])>0
    for checked in (True,False,True):
        with NativeScene(scene_for(point,role),record,metadata,metadata['joint_positions'],.1,
                         articulated=False,check_fingers=checked,allow_contact=not checked) as context:
            result=context.validate(metadata['joint_positions'])
            assert result['accepted']==(not checked),result
            assert context.validate_hand_pose(metadata['eef_pose_world_xyzw'])['accepted']==(not checked)
            if checked:
                assert result['blocking_robot_part'].startswith('gripper0_finger')
                np.testing.assert_allclose(result['blocking_sphere_world_m'][:3],point,atol=2e-6)
            assert context.record['finger_environment_collision_checked']==checked


def test_measured_aperture_changes_finger_centers_without_inflating_spheres(robot):
    """Robot link transforms, including aperture, determine both planners' geometry."""
    import copy
    from vlm_tamp.rekep_libero.geometry import matrix_pose
    record,metadata,values,chain,world=robot
    first=finger_environment_model(metadata,world)
    moved=copy.deepcopy(metadata);tcp=pose_matrix(metadata['eef_pose_world_xyzw'])
    for volume in moved['robot_collision_volumes_world']:
        if volume['robot_geom_name']=='gripper0_finger1_collision':
            transform=pose_matrix(volume['body_pose_world_xyzw'])
            transform[:3,3]+=tcp[:3,0]*.02
            volume['body_pose_world_xyzw']=matrix_pose(transform).tolist()
    second=finger_environment_model(moved,world)
    a,b=[np.asarray(m['hand_spheres_tcp']) for m in (first,second)]
    np.testing.assert_allclose(b[-4:-2,:3]-a[-4:-2,:3],[[.02,0,0]]*2,atol=1e-8)
    np.testing.assert_array_equal(a[:,3],b[:,3])
    np.testing.assert_allclose(a[-2:],b[-2:],atol=1e-8)


@pytest.mark.parametrize('role', ['stationary', 'moving', 'contact'])
@pytest.mark.parametrize('sphere', [14, 52, 55, 56, 57, 58])
def test_selected_object_contact_matches_hand_filter_and_native_fk(robot, role, sphere):
    """Only the target can touch palm/tips; arm and proximal fingers still collide."""
    record, metadata, values, chain, world = robot
    if sphere >= 55:
        model = finger_environment_model(metadata, world)
        tcp = pose_matrix(metadata['eef_pose_world_xyzw'])
        local = np.asarray(model['hand_spheres_tcp'])[sphere-37, :3]
        point = tcp[:3, :3]@local+tcp[:3, 3]
    else:
        point = values[sphere, :3]
    scene = scene_for(point, role)
    scene.skill_contact_surface = 'palm_fingertips'
    allowed = role == 'contact' and sphere in (52, 56, 58)
    with NativeScene(scene, record, metadata, metadata['joint_positions'], 0.,
                     articulated=False, check_fingers=True) as context:
        assert context.validate(metadata['joint_positions'])['accepted'] == allowed
        hand = context.validate_hand_pose(metadata['eef_pose_world_xyzw'])
        assert hand['accepted'] == (allowed or sphere == 14)
        assert context.record['fingertip_contact_allowed']
        if allowed:
            goal = np.asarray(metadata['joint_positions']) + .001
            path, result = context.plan(metadata['joint_positions'], goal,
                                        constrained=False, seconds=.2)
            assert result['exact_solution'], result
            assert len(path) >= 2


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
