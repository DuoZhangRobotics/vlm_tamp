"""Actual MC-VAMP contact-pair checks using retained robot-only calibration."""
from pathlib import Path

import numpy as np
import pytest

from vlm_tamp.coupled_storage import read_json
from vlm_tamp.robot_contracts import validate_operation,RequestValidationError


def test_runtime_defaults_cannot_silently_clear_the_observed_world():
    from vlm_tamp.hinged_contact_tools import tool_configuration
    root=Path(__file__).resolve().parents[1]
    config=tool_configuration(read_json(root/'configs/articulated_libero.json'))
    assert config['environment_collision_policy']=='veto'
    for operation,args in [('plan_grasp',{}),('plan_arc',{'degrees':10}),
                           ('push',{'observation_id':1,'camera':'frontview','region_xyxy':[1,1,20,20]})]:
        validate_operation(operation,{**args,'collision_policy':'veto'})
        with pytest.raises(RequestValidationError):
            validate_operation(operation,{**args,'collision_policy':'advisory'})


def test_open_grasp_search_explores_after_a_blocked_direct_connection():
    pytest.importorskip('vamp')
    from vlm_tamp.coupled_scene import Scene,RobotVolumes
    from vlm_tamp.hinged_contact_tools import tool_configuration
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
    root=Path(__file__).resolve().parents[1]
    folder=root/'results/open-hand-grasp-admission-20261030/open_microwave-state037'
    if not folder.exists():pytest.skip('Recorded non-direct open-hand query unavailable')
    meta=read_json(folder/'attempt01/001-observation/metadata.json')
    record=read_json(folder/'attempt01/robot-chain.json');chain=RobotChain(record)
    spec=read_json(folder/'attempt01/002-grounding/model.json')
    goal=read_json(folder/'attempt01/003-grasp/selection/grasp-attempts.json')[0]['joints']
    robot=RobotVolumes(chain,meta['robot_collision_volumes_world'],meta['joint_positions'])
    with np.load(folder/'workspace/robot_data/attempt01/geometry002/points.npz') as a:
        scene=Scene.from_points(a['static'],a['moving'],a['contact'],spec,
            config=tool_configuration(read_json(root/'configs/articulated_libero.json')),robot=robot)
    scene.skill_contact_surface='fingertips'
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        path,result=context.plan(meta['joint_positions'],goal,constrained=False,seconds=2.,iterations=3000)
    clear_native_cache()
    assert result['exact_solution'],result
    assert result['iterations']>0  # Requires exploration after a blocked straight connection.
    assert not result['dynamic_domain_pruning']
    assert result['resampling_seconds']==0
    np.testing.assert_allclose(path[0],meta['joint_positions'],atol=1e-6)
    np.testing.assert_allclose(path[-1],goal,atol=1e-6)


@pytest.fixture
def calibrated():
    pytest.importorskip('vamp')
    folder=Path(__file__).resolve().parents[1]/'results/small-graspgen-20261029-planner-path/open_microwave-state037/attempt01'
    if not folder.exists():pytest.skip('Retained robot calibration for native integration test unavailable')
    return read_json(folder/'opened/metadata.json'),read_json(folder/'robot-chain.json')


@pytest.mark.parametrize('part,allowed',[
    ('gripper0_finger1_pad_collision',True),
    ('gripper0_hand_collision',False),
    ('robot0_link4_collision',False),
])
@pytest.mark.parametrize('surface',['target','unrelated'])
@pytest.mark.parametrize('mode',['fingertips','grasped_fingers'])
def test_native_arm_palm_and_unrelated_surfaces_block_but_target_pad_contact_is_allowed(calibrated,part,allowed,surface,mode):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    meta,record=calibrated
    volume=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']==part)
    point=np.mean(volume['vertices_world'],axis=0)[None]
    far=point+3
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=(Scene.from_points(np.empty((0,3)),point,point,spec) if surface=='target'
        else Scene.from_points(point,far,far,spec))
    scene.skill_contact_surface=mode
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        result=context.validate(meta['joint_positions'])
        assert result['accepted']==(allowed and surface=='target'),result
        hand=context.validate_hand_pose(meta['eef_pose_world_xyzw'])
        assert hand['accepted']==(part=='robot0_link4_collision' or (allowed and surface=='target')),hand
        assert hand['arm_posture_checked'] is False
        assert context.record['environment_collision_policy']=='veto'
        assert not context.record['existing_palm_contact']['enabled']
        assert context.record['existing_palm_contact']['maximum_allowed_depth_m']==0.
        assert context.record['proximal_fingers_checked']==(mode=='fingertips')
        assert context.record['proximal_fingers_checked_against_other_surfaces']
        assert context.record['static_points']+context.record['selected_contact_points']>0
    clear_native_cache()


def test_hand_filter_uses_requested_pose_without_solving_an_arm_posture(calibrated):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    meta,record=calibrated
    palm=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
    shift=np.array([-.3,.3,.2]);point=np.mean(palm['vertices_world'],axis=0)[None]+shift;far=point+3
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=Scene.from_points(point,far,far,spec)
    goal=np.asarray(meta['eef_pose_world_xyzw']).copy();goal[:3]+=shift
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        assert context.validate_hand_pose(meta['eef_pose_world_xyzw'])['accepted']
        result=context.validate_hand_pose(goal)
        assert not result['accepted'] and result['blocking_robot_part']=='gripper0_hand_collision'
    clear_native_cache()


@pytest.mark.parametrize('mode,allowed',[('fingertips',False),('grasped_fingers',True)])
@pytest.mark.parametrize('surface',['target','unrelated'])
def test_grasped_finger_body_contact_is_target_specific(calibrated,mode,allowed,surface):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points
    meta,record=calibrated
    # Known Panda local point inside the proximal finger, outside the palm and
    # pad volumes. A finger/palm overlap would correctly remain rejected.
    point=transform_points([[-.05,0,-.023]],pose_matrix(meta['eef_pose_world_xyzw']));far=point+3
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=(Scene.from_points(np.empty((0,3)),point,point,spec) if surface=='target'
        else Scene.from_points(point,far,far,spec))
    scene.skill_contact_surface=mode
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        assert context.validate(meta['joint_positions'])['accepted']==(allowed and surface=='target')
        assert not context.record['palm_contact_allowed']
        assert context.record['finger_body_target_contact_allowed']==allowed
    clear_native_cache()
