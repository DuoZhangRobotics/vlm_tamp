import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_sensor_geometry_check import camera,pixels
from vlm_tamp.multiview_grounding import fit_multiview,compact_review
from vlm_tamp.robot_contracts import validate_operation,RequestValidationError,observed_image_sizes


def inputs():
    views={'a':camera([1,0,1.7]),'b':camera([0,-1,1.5]),'check':camera([-1,.3,1.6])}
    for view in views.values():view['rgb']=np.zeros((512,512,3),np.uint8)
    selections={name:{'camera':name,'contact_box_xyxy':[240,220,270,310],
        'moving_panel_box_xyxy':[180,180,300,330],
        'hinge_pixels_xy':pixels(view,[.2,-.1]).tolist(),'initial_image_direction_xy':None}
        for name,view in views.items()}
    arguments={'observation_id':1,'regions':selections['a'],'comparison_regions':selections['b'],
        'check_view':{k:selections['check'][k] for k in ['camera','hinge_pixels_xy']}}
    return views,arguments


@pytest.fixture
def fitted(monkeypatch):
    from vlm_tamp import hinged_contact_tools
    def fit(cameras,choices,template,metadata,**kwargs):
        spec={**template,'source_anchor_world_m':[.4,0.,1.],
              'joint_origin_world_m':[.1,-.1,1.],'joint_axis_world':[0,0,1],
              'panel_normal_world':[1.,0.,0.]}
        cloud=np.array([[.4,0.,z] for z in np.linspace(.9,1.1,20)])
        return SimpleNamespace(spec=spec),cloud,np.zeros((512,512),bool),[[.4,0.,1.]],{
            'cylinder_radius_m':.01,'motion_direction_check':{'observed_bulk_world_m':[.4,.5,1.]}}
    monkeypatch.setattr(hinged_contact_tools,'ground_handle',fit)


def test_consistent_caller_edges_recover_hinge_without_altering_inputs(fitted):
    views,args=inputs();before=copy.deepcopy(args)
    candidate,review=fit_multiview(views,{}, {'motion':'close','instruction':'close the door'},args)
    assert review['accepted'] and not review['semantic_identity_verified']
    np.testing.assert_allclose(candidate['spec']['joint_origin_world_m'],[.2,-.1,1.],atol=1e-12)
    assert candidate['spec']['joint_axis_world']==[0,0,1]
    assert not review['triangulation']['views']['check']['used_for_fit']
    assert args==before


def test_wrong_withheld_edge_is_rejected_with_specific_feedback(fitted):
    views,args=inputs();args['check_view']['hinge_pixels_xy']=pixels(views['check'],[.2,.15]).tolist()
    _,review=fit_multiview(views,{}, {'motion':'close','instruction':'close the door'},args)
    assert not review['accepted']
    assert 'hinge_projections_within_4px' in review['failed_checks']
    assert review['measurements']['hinge_projection_error_px']['check']>20


@pytest.mark.parametrize('problem',['duplicate_camera','check_pixel_outside','comparison_box_inverted'])
def test_invalid_multiview_request_is_rejected_before_geometry(problem):
    views,args=inputs()
    if problem=='duplicate_camera':args['check_view']['camera']='a'
    elif problem=='check_pixel_outside':args['check_view']['hinge_pixels_xy'][0][0]=512
    else:args['comparison_regions']['contact_box_xyxy']=[100,100,90,140]
    with pytest.raises(RequestValidationError):validate_operation('ground_multiview',args,observed_image_sizes(views))


def service(tmp_path):
    import serve_subscription_robot as module
    robot=module.RobotService.__new__(module.RobotService)
    robot.workspace=tmp_path/'public';robot.workspace.mkdir();robot.public=robot.workspace/'robot_data';robot.public.mkdir()
    robot.episode=tmp_path/'attempt01';robot.episode.mkdir()
    robot.calls=2;robot.attempt=1;robot.multiview_grounding=True;robot.sim=object();robot.epoch=7
    robot.cameras,args=inputs();robot.latest_observation={'observation_id':1}
    robot.spec={'previous':True};robot.scene=object();robot.tracker=object()
    robot.state={'control_steps':60,'metadata':{'robot_collision_volumes_world':[],'joint_positions':[0.]*7}}
    robot.ik=SimpleNamespace(chain=object());robot.config={}
    robot.task_spec={'motion':'close','instruction':'close the door'}
    return module,robot,args


def test_failed_consistency_leaves_previous_model_scene_and_epoch_untouched(tmp_path,fitted):
    _,robot,args=service(tmp_path);old=(robot.spec,robot.scene,robot.tracker)
    args['check_view']['hinge_pixels_xy']=pixels(robot.cameras['check'],[.2,.15]).tolist()
    result=robot.ground_multiview(**args)
    assert not result['accepted'] and not result['planner_model_updated']
    assert (robot.spec,robot.scene,robot.tracker)==old and robot.epoch==7
    assert robot.state['control_steps']==60 and len(result['images'])==3


def test_passing_fit_installs_fresh_scene_and_invalidates_old_plan_epoch(tmp_path,fitted,monkeypatch):
    module,robot,args=service(tmp_path);old=robot.scene
    monkeypatch.setattr(module,'RobotVolumes',lambda *a,**k:object())
    scene=SimpleNamespace(moving_reference=np.zeros((1,3)),static_points=np.zeros((1,3)))
    monkeypatch.setattr(module,'ArcContactScene',lambda *a,**k:scene)
    result=robot.ground_multiview(**args)
    assert result['accepted'] and result['planner_model_updated']
    assert robot.epoch==8 and robot.scene is scene and robot.scene is not old and robot.tracker is None
    assert robot.state['control_steps']==60


def test_stale_capture_cannot_replace_geometry(tmp_path,monkeypatch):
    _,robot,args=service(tmp_path);args['observation_id']=0
    from vlm_tamp import multiview_grounding
    monkeypatch.setattr(multiview_grounding,'fit_multiview',lambda *a:pytest.fail('Stale images must fail before fitting'))
    with pytest.raises(RequestValidationError,match='current observation'):robot.ground_multiview(**args)
    assert robot.epoch==7 and robot.spec=={'previous':True}


def test_compact_reply_keeps_failure_measurements_and_three_review_images(fitted):
    from vlm_tamp.compact_robot_interface import compact_response
    views,args=inputs();args['check_view']['hinge_pixels_xy']=pixels(views['check'],[.2,.15]).tolist()
    _,review=fit_multiview(views,{}, {'motion':'close','instruction':'close the door'},args)
    value={'grounding_review':compact_review(review),'accepted':False,'planner_model_updated':False,
           'images':['a.png','b.png','check.png'],'review_file':'review.json'}
    compact=compact_response({'ok':True,'result':value})['result']
    assert compact['grounding_review']['measurements']==review['measurements']
    assert compact['images']==value['images'] and not compact['planner_model_updated']
    assert len(json.dumps(compact))<4000 and 'depth_fits' not in compact['grounding_review']
