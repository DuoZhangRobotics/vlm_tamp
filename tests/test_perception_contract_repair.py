from pathlib import Path
import json
import numpy as np
import pytest
from vlm_tamp.rigid_hinge_tracking import stationary_pose_evidence,measured_pose
from vlm_tamp.visual_inspection import overview
from vlm_tamp.compact_robot_interface import compact_response


def test_stationary_target_evidence_is_not_discarded_for_lack_of_hinge_rotation():
    rng=np.random.default_rng(2);points=rng.uniform([-.2,-.2,.9],[.2,.2,1.1],(53,3))
    after=points.copy();after[:4,0]+=.02
    result=stationary_pose_evidence(points,after)
    assert result['pose_accepted'] and not result['accepted']
    assert result['observed_motion_status']=='stationary'
    np.testing.assert_allclose(measured_pose(result),np.eye(4),atol=1e-12)
    assert stationary_pose_evidence(points[:5],after[:5]) is None
    assert stationary_pose_evidence(points,points+[.01,0,0]) is None


def test_overview_exposes_occluded_camera_alternatives_without_new_sensor_capture(tmp_path):
    from PIL import Image
    obs=tmp_path/'robot_data/attempt01/obs001';obs.mkdir(parents=True)
    names=['frontview','agentview','side','back']
    (obs/'robot.json').write_text(json.dumps({'control_steps':60}))
    (obs/'calibration.json').write_text(json.dumps({n:{} for n in names}))
    for i,n in enumerate(names):Image.new('RGB',(512,512),(i*30,20,30)).save(obs/(n+'.png'))
    result=overview(tmp_path,{'observation_id':1})
    assert set(result['cameras'])==set(names) and not result['new_sensor_capture']
    assert (tmp_path/result['comparison_image']).exists()
    response=compact_response({'ok':True,'result':{'images':{n:n+'.png' for n in names},'camera_overview':result}})
    assert response['result']['camera_overview']==result
    assert len(response['result']['images'])==2


def test_stationary_motion_conflict_survives_compact_display():
    effect={'status':'selected_region_stationary_despite_hand_motion','hand_rotation_deg':6,
        'observed_region_rotation_deg':0,'physical_grasp_verified':False}
    assert compact_response({'ok':True,'result':{'motion_effect':effect}})['result']['motion_effect']==effect


@pytest.mark.parametrize('sign',[-1,1])
def test_neither_direction_source_silently_overrides_a_conflict(sign):
    from vlm_tamp.handle_contact import require_direction_agreement
    direction={'direction_conflict':True,'vlm_image_axis':[0,0,sign],'depth_prior_axis':[0,0,-sign]}
    with pytest.raises(ValueError) as exc:require_direction_agreement({'motion_direction_check':direction})
    assert exc.value.planning_diagnostic['model_changed'] is False
    assert exc.value.planning_diagnostic['caller_hint_axis']==[0,0,sign]


def test_gripper_gap_collapse_is_sensor_evidence_not_a_contact_certificate():
    from vlm_tamp.grasp_observation import grasp_observation
    def metadata(gap):
        return {'eef_pose_world_xyzw':[0,0,0,0,0,0,1],
            'robot_collision_volumes_world':[
                {'robot_geom_name':'left_pad_collision','vertices_world':[[-gap/2-.002,0,0],[-gap/2,0,0]]},
                {'robot_geom_name':'right_pad_collision','vertices_world':[[gap/2,0,0],[gap/2+.002,0,0]]}]}
    result=grasp_observation(metadata(.001),{'observed_handle_radius_m':.01},before=metadata(.025),manipulation=True)
    assert result['status']=='gripper_gap_collapsed_during_manipulation'
    assert result['pad_gap_mm']==pytest.approx(1) and result['previous_pad_gap_mm']==pytest.approx(25)
    assert not result['physical_contacts_observed']
    assert grasp_observation(metadata(.022),before=metadata(.025),manipulation=True)['status']=='grasp_not_verified'
    assert grasp_observation({})['available'] is False
    assert grasp_observation(metadata(.001),{'observed_handle_radius_m':.01})['status']=='near_closed_without_bar_width_support'


def test_sequence_compaction_retains_contact_failure_and_pose_timestamps():
    stage={'stage':'closure','grasp_evidence':{'pad_gap_mm':.16,'selected_bar_diameter_mm':19.5},
        'motion_effect':{'status':'object_motion_unresolved'},'scene_reference_control_step':100,
        'observation':{'observation_id':16,'control_steps':110}}
    result=compact_response({'ok':True,'result':{'stages':[stage]}})['result']['stages'][0]
    assert result['grasp_evidence']==stage['grasp_evidence'] and result['motion_effect']==stage['motion_effect']
    assert result['scene_reference_control_step']==100 and result['observation']['observation_id']==16


def test_refresh_contract_has_no_required_hinge_reannotation():
    from vlm_tamp.robot_contracts import validate_operation
    arguments={'camera':'view','contact_box_xyxy':[10,20,30,40],'moving_panel_box_xyxy':[5,10,50,60]}
    validate_operation('refresh_contact',arguments,{'view':{'width':100,'height':100}})
    with pytest.raises(ValueError):validate_operation('refresh_contact',{**arguments,'camera':'missing'},{'view':{'width':100,'height':100}})
    with pytest.raises(ValueError):validate_operation('refresh_contact',{**arguments,'contact_box_xyxy':[30,20,10,40]})


def test_joint_parameter_update_does_not_resegment_new_images_or_move_geometry():
    from serve_subscription_robot import RobotService
    from vlm_tamp.coupled_scene import Scene
    import copy
    scene=Scene.from_points([[2,2,1]],[[.1,0,1],[.1,.1,1]],[[.12,0,1]],
        {'joint_origin_world_m':[0,0,1],'joint_axis_world':[0,0,1],
         'front_plane':{'point_world_m':[.1,0,1]},'panel_normal_world':[1,0,0]})
    service=RobotService.__new__(RobotService);service.scene=scene;service.spec=copy.deepcopy(scene.spec)
    service.spec['joint_origin_world_m']=[.005,0,1]
    service._update_joint_parameters()
    assert service.scene is not scene
    for key in ['static_points','moving_reference','contact_reference']:
        np.testing.assert_array_equal(getattr(service.scene,key),getattr(scene,key))
    np.testing.assert_array_equal(service.scene.origin,[.005,0,1])


@pytest.mark.parametrize('retained',[False,True])
def test_measured_part_pose_preserves_hinge_and_prevents_double_application(monkeypatch,retained):
    import serve_subscription_robot as module
    from serve_subscription_robot import RobotService
    from vlm_tamp.coupled_scene import Scene
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    points=np.random.default_rng(3).uniform([.1,-.15,.9],[.3,.15,1.1],(80,3))
    rotation=Rotation.from_euler('z',15,degrees=True).as_matrix()
    world=np.stack([points,points@rotation.T]);valid=np.ones(world.shape[:2],bool)
    scene=Scene.from_points([[2,2,1]],points,points[:20],
        {'joint_origin_world_m':[0,0,1],'joint_axis_world':[0,0,1],
         'front_plane':{'point_world_m':[.2,0,1]},'panel_normal_world':[1,0,0]})
    service=RobotService.__new__(RobotService);service.scene=scene;service.spec=scene.spec
    service.contact_scene_kind='articulated';service.scene_reference_control_step=10
    service.state={'control_steps':20};service.epoch=0;service.anchors=[[.2,0,1]]
    service.config={};service.cameras={}
    monkeypatch.setattr(module,'depth_support',lambda *a,**k:({'multi_view_free_space_conflict_fraction':0.,'supported_fraction':1.},None,None))
    service._perception_tools=SimpleNamespace(tracks={'t':{'observation_ids':[1,2],'world':world,'valid':valid}},
        _record=lambda i:{'control_steps':i*10})
    service.export_geometry=lambda x:{'model':service.spec}
    if retained:
        from vlm_tamp.observed_joint_scene import retain_observed_articulation
        # Separate masks of the same observed object need not have the same ID.
        # Timestamp, measured motion and geometry consistency govern this update.
        service.spec.update(generic_part_joint_model=True,part_id='current-mask')
        service._perception_tools.tracks['t']['part_id']='earlier-mask'
        retain_observed_articulation(service)
        service.scene=service.spec=None;service.contact_scene_kind='surface';service.scene_reference_control_step=None
        with pytest.raises(ValueError,match='reference capture'):
            service._perception_tools._record=lambda i:{'control_steps':i*10+1}
            service.apply_part_motion('t',1,2)
        assert service.scene is None and service.spec is None and service.epoch==0
        service._perception_tools._record=lambda i:{'control_steps':i*10}
    result=service.apply_part_motion('t',1,2)
    np.testing.assert_allclose(service.scene.moving_reference,world[1],atol=1e-10)
    np.testing.assert_array_equal(service.scene.static_points,scene.static_points)
    assert service.spec['joint_origin_world_m']==[0,0,1]
    assert service.scene_reference_control_step==20 and result['hinge_preserved']
    assert result['retained_observed_scene_reused']==retained
    assert service.contact_scene_kind=='articulated'
    with pytest.raises(ValueError,match='twice'):service.apply_part_motion('t',1,2)
