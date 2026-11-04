"""Caller geometry and native solver results must survive the wrappers."""
from types import SimpleNamespace
import numpy as np
import pytest

from vlm_tamp.explicit_grounding import ground_pixels
from vlm_tamp.surface_contact_tools import selected_pixel_point
from vlm_tamp.robot_contracts import validate_operation


def capture():
    y,x=np.indices((12,12))
    points=np.stack([x*.01,y*.01,1+x*.002+y*.003],axis=-1)
    view={'world_points':points,'depth':points[:,:,2],
        'camera_to_world':np.eye(4),'intrinsic':np.diag([100.,100.,1.])}
    choices={'camera':'test','contact_box_xyxy':[5,3,10,9],
        'moving_panel_box_xyxy':[0,0,11,11],'contact_pixel_xy':[8,4],
        'hinge_pixels_xy':[[2,2],[3,9]],'initial_image_direction_xy':None}
    return view,choices


def test_missing_depth_does_not_snap_to_nearby_valid_pixel():
    view,_=capture();view['world_points'][4,8]=np.nan
    with pytest.raises(ValueError,match='selected contact pixel has no measured'):
        selected_pixel_point(view,[8,4])


def test_raw_grounding_keeps_exact_contact_nonvertical_hinge_and_selected_cloud():
    view,choices=capture();candidate=ground_pixels({'test':view},{},choices)
    np.testing.assert_array_equal(candidate['anchors'][0],view['world_points'][4,8])
    expected=view['world_points'][9,3]-view['world_points'][2,2]
    expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(candidate['spec']['joint_axis_world'],expected)
    assert candidate['cloud'].shape==(7*6,3)
    assert 'observed_handle_radius_m' not in candidate['spec']
    assert candidate['evidence']['axis_prior'] is None
    assert candidate['spec']['direction_source']['task_direction_status']=='unknown'


def test_image_arrows_do_not_redefine_canonical_hinge_coordinates():
    view,choices=capture();choices['initial_image_direction_xy']=[1,0]
    first=ground_pixels({'test':view},{},choices)
    choices['initial_image_direction_xy']=[-1,0]
    second=ground_pixels({'test':view},{},choices)
    np.testing.assert_allclose(first['spec']['joint_axis_world'],second['spec']['joint_axis_world'])
    assert first['spec']['direction_source']['legacy_image_hint_recorded_only']==[1,0]
    np.testing.assert_array_equal(first['spec']['joint_origin_world_m'],second['spec']['joint_origin_world_m'])
    np.testing.assert_array_equal(first['cloud'],second['cloud'])


@pytest.mark.parametrize('operation',['plan_arc','plan_sequence'])
def test_attachment_model_is_an_explicit_request(operation):
    with pytest.raises(ValueError,match='grasp_constraint'):
        validate_operation(operation,{'degrees':15})
    validate_operation(operation,{'degrees':15,'grasp_constraint':'point'})


@pytest.mark.parametrize('operation',['push','hook_pull'])
def test_targeted_contact_cannot_silently_choose_a_pixel_or_robot_landmark(operation):
    args={'observation_id':1,'camera':'test','region_xyxy':[1,1,9,9],'direction_world':[1,0,0]}
    with pytest.raises(ValueError,match='contact_'):
        validate_operation(operation,args)
    validate_operation(operation,{**args,'contact_pixel_xy':[5,5],'contact_name':'gripper0_finger1_pad_collision'})


def test_contact_approach_offers_unmodified_endpoints_to_native_planner(monkeypatch):
    import vlm_tamp.contact_skills as skill
    import vlm_tamp.surface_contact_tools as surface
    q=np.zeros(7);point=np.array([.2,.1,1.]);offset=np.array([.02,0,.04])
    solutions=[{'joints':(q+i*.001).tolist()} for i in range(9)]
    calls=[]
    candidate={'contact_local_m':offset.tolist(),'contact_normal_local':[0,0,1],
        'robot_collision_geom':'selected_pad','physical_collision_masks':{'contype':1,'conaffinity':1},
        'initiates_simulator_contacts':True}
    monkeypatch.setattr(surface,'contact_pose_candidates',lambda *a,**kw:[candidate])
    def ik_query(ik,metadata,selected,name):
        np.testing.assert_array_equal(selected,point)
        assert name=='selected_pad'
        return candidate,solutions
    monkeypatch.setattr(surface,'contact_solutions',ik_query)
    class Context:
        record={}
        finger_model={'hand_spheres_tcp':[[0,0,0,.01]]}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def plan(self,start,goal,**kw):
            calls.append(goal)
            return np.vstack([start,goal]),{'exact_solution':len(calls)==9}
        def validate(self,*a):pytest.fail('No independent endpoint validation')
    monkeypatch.setattr(skill,'skill_native_scene',lambda *a,**kw:Context())
    monkeypatch.setattr(skill,'cartesian_path',lambda *a,**kw:np.vstack([a[4],a[4]]))
    ik=SimpleNamespace(configure_scene=lambda context: None,configure_observed_scene=lambda scene:None,
        solve_contact=lambda p,c,seed,r,**kw:{'joints':seed},
        chain=SimpleNamespace(forward=lambda q:np.r_[point-offset,[0.,0.,0.,1.]]))
    evidence={'point_world_m':point.tolist(),'contact_name':'selected_pad',
        'contact_surface':'fingertips','direction_world':[1,0,0]}
    result=skill.plan_engagement(ik,SimpleNamespace(skill_approach_speed_rad_s=.6),{},
        {'joint_positions':q},evidence,None,'push',skill.Budget(5))
    assert calls==[s['joints'] for s in solutions]
    assert not result['direct_contact_approach']
    np.testing.assert_array_equal(result['contact_world'],point)
    assert len(result['insertion_references'])>0


def test_gripper_holds_existing_reference_despite_measured_joint_overshoot():
    from serve_subscription_robot import RobotService
    service=RobotService.__new__(RobotService)
    reference=np.linspace(-.1,.1,7)
    service.state={'metadata':{'joint_positions':[4.]*7,
        'joint_servo':{'reference_joint_positions':reference.tolist()}}}
    calls=[]
    service.issue=lambda refs,*a:(calls.append(refs) or {})
    reply=service.gripper(close=True,steps=3)
    np.testing.assert_array_equal(calls[0],np.repeat(reference[None],3,axis=0))
    assert not reply['arm_reference']['automatic_arm_recovery']


def test_float32_boundary_is_preserved_without_clipping_or_extra_margin():
    from vlm_tamp.coupled_worker import checked_joint_target
    limits=np.tile([-.123456789,.123456789],(7,1))
    target=limits[:,1].astype(np.float32).astype(float)
    np.testing.assert_array_equal(checked_joint_target(target,limits),target)
    with pytest.raises(ValueError,match='limits'):
        checked_joint_target(target+.0001,limits)
