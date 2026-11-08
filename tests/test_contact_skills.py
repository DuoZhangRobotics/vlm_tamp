from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from vlm_tamp.contact_skills import (SkillStop,Budget,contact_landmarks,
    hook_feature,fit_object_motion,measured_contact_speed)
from vlm_tamp.contact_skill_service import run_skill,momentum_status
from vlm_tamp.timed_execution import schedule_contact
from vlm_tamp.robot_contracts import validate_operation


class LinearChain:
    def forward(self,q):return np.r_[np.asarray(q)[:3],[0,0,0,1.]]


def test_rigid_feedback_accepts_a_spatially_supported_majority_with_background_outliers():
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(2026);first=rng.uniform(-.15,.15,(30,3))
    rotation=Rotation.from_euler('z',12,degrees=True).as_matrix();translation=np.array([.02,-.01,0.])
    second=first@rotation.T+translation
    second[17:]=rng.uniform(.5,.9,(13,3))
    fit=fit_object_motion(first,second)
    assert fit['accepted'] and fit['inliers']==17
    assert np.allclose(np.array(fit['transform'])[:3,:3],rotation)
    assert fit['rms_m']<1e-6


def test_rigid_feedback_rejects_tied_motion_groups():
    rng=np.random.default_rng(23);first=rng.uniform(-.15,.15,(20,3));second=first.copy()
    second[:10]+=[.03,0,0];second[10:]+=[0,.04,0]
    assert not fit_object_motion(first,second)['accepted']


def test_push_can_slide_on_selected_part_but_hook_keeps_feature_contact_scope():
    from vlm_tamp.contact_collision_evidence import selected_contact_points
    scene=SimpleNamespace(moving_reference=np.array([[0.,0,0],[.05,0,0]]),
        contact_reference=np.array([[0.,0,0]]),skill_kind='push')
    assert selected_contact_points(scene).tolist()==[True,True]
    scene.skill_kind='hook_pull'
    assert selected_contact_points(scene).tolist()==[True,False]


def test_contact_tracking_reuses_absolute_reference_and_returns_incremental_motion(monkeypatch):
    import vlm_tamp.rigid_hinge_tracking as module
    from vlm_tamp.contact_skills import ContactTracker
    rng=np.random.default_rng(22);points=rng.uniform(-.1,.1,(30,3));instances=[]
    class Tracker:
        def __init__(self,*args,**kwargs):
            self.reference_step=kwargs['reference_step']
            self.references={name:{'points':points+np.array([self.reference_step*.001,0,0])} for name in args[0]}
            instances.append(self)
        def observe(self,cameras,*,control_step,estimate_joint):
            assert not estimate_joint
            self.last_pairs=(points,points+np.array([control_step*.001,0,0]))
            return {'camera_counts':[{'camera':'side','tracked_pairs':30}]}
    monkeypatch.setattr(module,'RGBDHingeTracker',Tracker)
    tracker=ContactTracker({'front':{},'side':{}},points,0)
    first=tracker.observe({'front':{}},10);second=tracker.observe({'front':{},'side':{}},25)
    assert first['accepted'] and second['accepted'] and len(instances)==3
    assert first['tracking_images_advanced']==['front']
    assert np.allclose(tracker.tracker.references['front']['points'],points)
    assert np.array(first['transform'])[0,3]==pytest.approx(.010)
    assert np.array(second['transform'])[0,3]==pytest.approx(.015)
    assert second['reference_camera_count']==2 and tracker.last_view_counts['side']==30


@pytest.fixture
def ik_point_world(monkeypatch):
    from contextlib import nullcontext
    import vlm_tamp.contact_skills as module
    import vlm_tamp.current_robot_scene as measured
    context=SimpleNamespace()  # No path planner or independent collision validator.
    monkeypatch.setattr(module,'skill_native_scene',lambda *args,**kw:nullcontext(context))
    monkeypatch.setattr(measured,'clear_current_robot_scene',lambda scene,*args:scene)
    updates=[]
    return {'configure_observed_scene':lambda value:updates.append(('scene',value)),
            'move_scene_target':lambda value:updates.append(('target',value.copy()))},updates


def test_physical_push_uses_collision_aware_ik_without_a_path_planner(ik_point_world):
    import vlm_tamp.contact_skills as module
    methods,updates=ik_point_world
    scene=SimpleNamespace(skill_collision_policy='veto')
    def solve(point,local,seed,rotation,**kwargs):
        # The target geometry must move with each commanded point, not remain
        # an obstacle at its pre-push position.
        np.testing.assert_allclose(updates[-1][1][:3,3],point)
        return {'joints':np.r_[point,np.zeros(4)]}
    ik=SimpleNamespace(chain=LinearChain(),solve_contact=solve,**methods)
    meta={'joint_positions':[0.]*7}
    path,report=module.plan_contact_stroke(ik,scene,{},meta,np.zeros(3),np.array([.1,0,0]),Budget(5),physical_feedback=True)
    assert path[-1,0]==pytest.approx(.1) and report['physical_feedback']
    assert updates[0][0]=='scene'
    assert report['ik_environment_collision_checked'] and not report['environment_planner_used']


def test_push_stroke_neither_prepares_nor_queries_an_environment_world(monkeypatch):
    import vlm_tamp.contact_skills as module
    import vlm_tamp.current_robot_scene as measured
    def forbidden(*args,**kwargs):raise AssertionError('Push continuation must not use a collision world')
    monkeypatch.setattr(module,'skill_native_scene',forbidden)
    monkeypatch.setattr(measured,'clear_current_robot_scene',forbidden)
    calls=[]
    def solve(point,local,seed,rotation,*,environment_collision):
        assert environment_collision is False
        calls.append(seed.copy())
        return {'joints':np.r_[point,np.zeros(4)]}
    ik=SimpleNamespace(chain=LinearChain(),solve_contact=solve,
        configure_observed_scene=forbidden,move_scene_target=forbidden)
    scene=SimpleNamespace()
    path,report=module.plan_contact_stroke(ik,scene,{}, {'joint_positions':[0.]*7},
        np.zeros(3),np.array([.02,0,0]),Budget(5),physical_feedback=True,environment_collision=False)
    np.testing.assert_allclose(path[-1,:3],[.02,0,0])
    assert not report['ik_environment_collision_checked']
    for seed,previous in zip(calls,path[:-1]):np.testing.assert_array_equal(seed,previous)


def test_physical_push_keeps_a_reachable_prefix_when_the_next_ik_fails(ik_point_world):
    import vlm_tamp.contact_skills as module
    methods,_=ik_point_world
    scene=SimpleNamespace(skill_collision_policy='veto')
    ik=SimpleNamespace(chain=LinearChain(),**methods,
        solve_contact=lambda point,local,seed,rotation,**kwargs:None if point[0]>.02 else {'joints':np.r_[point,np.zeros(4)]})
    path,report=module.plan_contact_stroke(ik,scene,{}, {'joint_positions':[0.]*7},np.zeros(3),np.array([.1,0,0]),Budget(5),physical_feedback=True)
    assert .015<path[-1,0]<=.02 and report['reachable_prefix']['reason']=='contact_ik_failed'


def test_push_continuation_tracks_the_contact_offset_from_each_previous_posture(ik_point_world):
    from scipy.spatial.transform import Rotation
    import vlm_tamp.contact_skills as module
    methods,_=ik_point_world
    rotation=Rotation.from_euler('z',90).as_matrix();quat=Rotation.from_matrix(rotation).as_quat()
    chain=SimpleNamespace(forward=lambda q:np.r_[np.asarray(q)[:3],quat])
    contact=np.array([.04,0,0]);calls=[]
    def solve(point,local,seed,preferred,**kwargs):
        calls.append((point.copy(),seed.copy()))
        assert np.allclose(local,contact) and np.allclose(preferred,rotation)
        return {'joints':np.r_[point-rotation@local,np.zeros(4)]}
    ik=SimpleNamespace(chain=chain,solve_contact=solve,**methods)
    path,report=module.plan_contact_stroke(ik,SimpleNamespace(skill_collision_policy='veto'),{},
        {'joint_positions':[0.]*7},contact,np.array([.023,0,0]),Budget(5),physical_feedback=True)
    np.testing.assert_allclose(path[-1,:3],[.023,0,0],atol=1e-12)
    for i,(_,seed) in enumerate(calls):np.testing.assert_array_equal(seed,path[i])
    assert not report['orientation_is_soft_preference']
    assert report['motion_backend']=='cartesian_ik' and not report['environment_planner_used']


def test_projected_contact_roles_include_other_views_without_widening_the_region():
    from vlm_tamp.contact_skills import projected_surface_membership
    view={'camera_to_world':np.eye(4),'intrinsic':np.array([[100.,0,50],[0,100,50],[0,0,1]])}
    # The second observed surface sample is 5 mm behind the first, beyond the
    # former nearest-neighbor cutoff but in the same selected thin surface.
    points=np.array([[0.,0,1.],[0.,0,1.005],[.2,0,1.],[0.,0,1.1],[0,0,-1.]])
    membership=projected_surface_membership(points,view,[45,45,55,55],np.array([0.,0,1.]),np.array([0.,0,1.]))
    assert membership.tolist()==[True,True,False,False,False]






def test_contact_timing_respects_cartesian_and_joint_limits():
    path=np.array([[0.]*7,[.03,0,0,0,0,0,0]])
    refs=schedule_contact(path,LinearChain(),[0,0,0],contact_speed_m_s=.03)
    full=np.vstack([path[0],refs]);assert len(refs)==20
    assert np.max(np.linalg.norm(np.diff(full[:,:3],axis=0),axis=1)*20)<=.03+1e-10
    path[1,6]=1.5
    refs=schedule_contact(path,LinearChain(),[0,0,0],contact_speed_m_s=.12)
    assert len(refs)>=30
    assert np.max(np.abs(np.diff(np.vstack([path[0],refs]),axis=0)))*20<=1.+1e-10


@pytest.mark.parametrize('speed',[float('nan'),float('inf'),0,-1])
def test_speed_contract_rejects_invalid_requests(speed):
    with pytest.raises(ValueError):schedule_contact(np.zeros((2,7)),LinearChain(),[0,0,0],contact_speed_m_s=speed)


def test_measured_speed_uses_robot_feedback_not_requested_motion():
    events=[{'measured_joints':[.001,0,0,0,0,0,0]},{'measured_joints':[.002,0,0,0,0,0,0]}]
    measured=measured_contact_speed(events,LinearChain(),np.zeros(3),np.zeros(7))
    assert measured['mean_m_s']==pytest.approx(.02)


def test_flat_surface_cannot_silently_become_hook_or_grasp():
    y,z=np.meshgrid(np.linspace(-.08,.08,20),np.linspace(-.08,.08,20))
    flat=np.c_[np.zeros(y.size),y.ravel(),z.ravel()]
    with pytest.raises(SkillStop,match='no_hookable_feature'):hook_feature(flat,[1,0,0])


def test_hook_uses_rear_measured_surface_on_retaining_bar():
    a,z=np.meshgrid(np.linspace(0,2*np.pi,40),np.linspace(-.08,.08,30))
    points=np.c_[.01*np.cos(a.ravel()),.01*np.sin(a.ravel()),z.ravel()]
    point,evidence=hook_feature(points,[1,0,0])
    assert point[0]<-.008 and not evidence['catch_verified']
    assert np.linalg.norm(points-point,axis=1).min()<1e-9


def test_motion_fit_recovers_observed_translation_and_rejects_disagreement():
    rng=np.random.default_rng(7);a=rng.uniform(-.1,.1,(40,3));b=a+[.008,0,0]
    fit=fit_object_motion(a,b)
    assert fit['accepted'] and np.asarray(fit['transform'])[:3,3]==pytest.approx([.008,0,0])
    broken=fit_object_motion(a,rng.uniform(-.1,.1,(40,3)))
    assert not broken['accepted']
    assert fit_object_motion(a,a)['accepted']




def test_collision_witness_excludes_permitted_contact_points():
    from itertools import product
    from vlm_tamp.contact_collision_evidence import current_collision_witness
    name='gripper0_finger1_collision'
    volume={'robot_geom_name':name,'type':'convex','vertices_world':list(product([-.02,.02],repeat=3)),
        'planes_world':np.c_[np.vstack([np.eye(3),-np.eye(3)]),np.full(6,-.02)]}
    scene=SimpleNamespace(static_points=np.empty((0,3)),moving_reference=np.array([[0.,0,0],[.015,0,0]]),
        contact_reference=np.array([[0.,0,0]]))
    result=current_collision_witness(scene,{'robot_collision_volumes_world':[volume]},
        {'blocking_robot_part':name,'last_rejected_surface':1,'blocking_surface':'moving target outside contact patch'})
    assert result['observed_points_inside_volume']==1
    assert result['overlap_depth_mm_range']==pytest.approx([5.,5.])
    assert result['distance_to_selected_contact_mm_range']==pytest.approx([15.,15.])


def test_expired_budget_and_stale_observation_do_not_issue_motion():
    with pytest.raises(SkillStop,match='budget_exhausted'):Budget(0).check()
    service=SimpleNamespace(sim=object(),latest_observation={'observation_id':2,'control_steps':20},state={'control_steps':20})
    with pytest.raises(ValueError,match='latest observation_id'):
        run_skill(service,'push',observation_id=1,camera='agentview',region_xyxy=[1,1,10,10],direction_world=[1.,0,0])


@pytest.mark.parametrize('total',[20.,30.,45.,120.])
def test_short_skill_does_not_shrink_fixed_local_planning_allowance(total):
    from collections import Counter
    from vlm_tamp.contact_skill_service import planning_allowances,planning_stage
    search,stroke=planning_allowances();parent=Budget(total);result={};timing=Counter()
    assert search==30. and stroke==5.
    with planning_stage(parent,stroke,'contact_insertion',result,timing,'planning') as (child,nested):
        assert child.seconds==5.
    assert result['planning_stages'][0]['status']=='completed'
    assert planning_allowances(2.,8.)==(2.,8.)


@pytest.mark.parametrize('stage,reason',[
    ('initial_contact_search','contact_search_budget_exhausted'),
    ('contact_insertion','contact_insertion_budget_exhausted'),
    ('contact_stroke','stroke_planning_budget_exhausted'),
    ('withdrawal','withdrawal_planning_budget_exhausted')])
def test_failed_planning_keeps_its_stage_timing_and_distinguishes_parent_deadline(stage,reason,monkeypatch):
    from collections import Counter
    from vlm_tamp.contact_skill_service import planning_stage
    import vlm_tamp.contact_skills as geometry
    now=[100.];monkeypatch.setattr(geometry.time,'monotonic',lambda:now[0])
    parent=Budget(20.,started=100.);result={};timing=Counter()
    with pytest.raises(SkillStop,match=reason) as caught:
        with planning_stage(parent,5.,stage,result,timing,'planning') as (child,nested):
            child.started=100.;nested['ik_seconds']=2.;now[0]+=6.;child.check()
    row=caught.value.evidence['planning_stage']
    assert row['effective_seconds']==5. and row['remaining_skill_seconds']==14.
    assert row['expired_budget_scope']=='local_planning_stage'
    assert row['timing_seconds']['ik_seconds']==2. and timing['planning']>0
    assert not caught.value.evidence['physical_infeasibility_proved']
    now[0]=118.
    with pytest.raises(SkillStop,match=reason) as caught:
        with planning_stage(parent,5.,stage,result,timing,'planning') as (child,nested):
            assert child.seconds==2.;child.started=118.;now[0]=121.;child.check()
    assert caught.value.evidence['planning_stage']['expired_budget_scope']=='skill_or_episode'


def test_public_schemas_accept_contact_skills_and_reject_poses():
    args={'observation_id':1,'camera':'agentview','contact_pixel_xy':[5,5],'contact_name':'gripper0_finger1_pad_collision','region_xyxy':[1,1,10,10],'direction_world':[1.,0,0],'preview':True}
    validate_operation('push',args)
    validate_operation('hook_pull',{**args,'direction_world':[1,0,0]})
    with pytest.raises(ValueError):validate_operation('push',{**args,'pose_world_xyzw':[0]*7})


def test_simulation_availability_is_distinct_from_stale_or_missing_validation(tmp_path,monkeypatch):
    import json
    path=tmp_path/'qualification.json';monkeypatch.setenv('VLM_TAMP_MOMENTUM_QUALIFICATION',str(path))
    assert momentum_status()['available'] and not momentum_status()['validated']
    assert momentum_status()['supported_speeds_m_s']==[.03,.06,.12]
    path.write_text(json.dumps({'execution_qualified':True,'sources_sha256':{}}))
    assert momentum_status()['available'] and not momentum_status()['validated']


@pytest.fixture
def calibrated_robot():
    from vlm_tamp.coupled_storage import read_json
    root=Path(__file__).resolve().parents[1]
    folder=root/'results/luna-compact-transition-20261108/trials/03-place_bowl-s001-existing/attempt01'
    if not folder.exists():pytest.skip('Recorded calibrated robot fixture unavailable')
    meta=read_json(folder/'010-observation/metadata.json.gz');record=read_json(folder/'robot-chain.json')
    return meta,record


@pytest.mark.parametrize('surface',['fingertips','palm'])
@pytest.mark.parametrize('sphere',[56,54])
def test_native_finger_exemption_and_explicit_palm_contact(calibrated_robot,surface,sphere):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.rekep_libero.geometry import transform_points
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    from vlm_tamp.vamp_sphere_model import native_spheres
    meta,record=calibrated_robot
    point=transform_points(native_spheres(meta['joint_positions'])[sphere,:3][None],
        np.linalg.inv(np.asarray(record['calibration']['world_to_robot'])))
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=Scene.from_points(np.empty((0,3)),point,point,spec);scene.skill_contact_surface=surface
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        assert context.validate(meta['joint_positions'])['accepted']==(sphere==56 or surface=='palm')
        assert not context.record['proximal_fingers_checked']
    far=point+3
    static=Scene.from_points(point,far,far,spec);static.skill_contact_surface=surface
    with NativeScene(static,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        assert context.validate(meta['joint_positions'])['accepted']==(sphere==56)
    clear_native_cache()


def test_calibrated_contact_offsets_use_actual_aperture(calibrated_robot):
    meta,_=calibrated_robot
    pads=contact_landmarks(meta,'fingertips',hook=True)
    assert len(pads)==2
    assert all('pad_collision' in p['name'] for p in pads)
    assert pads[0]['normal']@pads[1]['normal']==pytest.approx(-1.)


@pytest.mark.parametrize('part,stationary,articulated',[
    ('gripper0_finger1_collision',False,True),
    ('gripper0_finger1_collision',False,False),
    ('gripper0_finger1_collision',True,True),
    ('gripper0_hand_collision',False,True),
])
def test_known_current_robot_overlap_is_cleared_for_all_planning_modes(calibrated_robot,part,stationary,articulated):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    from vlm_tamp.current_robot_scene import clear_current_robot_scene
    meta,record=calibrated_robot
    volume=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']==part)
    point=np.asarray(volume['vertices_world']).mean(0,keepdims=True);far=point+3
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=Scene.from_points(point if stationary else np.empty((0,3)),far if stationary else point,far,spec)
    scene.skill_contact_surface='fingertips'
    scene=clear_current_robot_scene(scene,meta,10)
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=articulated) as context:
        assert context.validate(meta['joint_positions'])['accepted']
        assert context.record['current_robot_point_filter']['padding_m']==0.
    clear_native_cache()


def test_planner_sphere_overlap_outside_physical_mesh_passes_current_state(calibrated_robot):
    """Native finger admission must agree with measured-state point-cloud cleanup."""
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.current_robot_scene import clear_current_robot_scene, points_outside_current_robot
    from vlm_tamp.mcvamp_contact_numerical import NativeScene, clear_native_cache
    from vlm_tamp.vamp_sphere_model import current_environment_spheres

    meta,record=calibrated_robot
    spheres=current_environment_spheres(meta)
    offsets=np.asarray(list(product(range(-2,3),repeat=3)),float)
    offsets=offsets[np.linalg.norm(offsets,axis=1)>0]
    offsets=.999*offsets/np.linalg.norm(offsets,axis=1)[:,None]
    samples=np.vstack([sphere[:3]+offsets*sphere[3] for sphere in spheres[-4:]])
    outside,_=points_outside_current_robot(samples,meta)
    assert len(outside)>0
    point=outside[:1];far=point+3
    spec={'joint_origin_world_m':far[0].tolist(),'joint_axis_world':[0,0,1],
          'front_plane':{'point_world_m':far[0].tolist()},'panel_normal_world':[1,0,0]}
    raw=Scene.from_points(point,far,far,spec)
    with NativeScene(raw,record,meta,meta['joint_positions'],0.,articulated=False,
                     allow_contact=False,check_fingers=True) as context:
        assert not context.validate(meta['joint_positions'])['accepted']
    clean=clear_current_robot_scene(raw,meta,10)
    with NativeScene(clean,record,meta,meta['joint_positions'],0.,articulated=False,
                     allow_contact=False,check_fingers=True) as context:
        assert context.validate(meta['joint_positions'])['accepted']
    assert len(raw.static_points)==1 and len(clean.static_points)==0
    clear_native_cache()


def test_continuation_permits_finger_environment_contact(calibrated_robot):
    from vlm_tamp.coupled_scene import Scene,RobotVolumes
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
    from vlm_tamp.current_robot_scene import clear_current_robot_scene
    meta,record=calibrated_robot;chain=RobotChain(record);q=np.asarray(meta['joint_positions']);later=q.copy();later[0]+=.25
    robot=RobotVolumes(chain,meta['robot_collision_volumes_world'],q)
    volume=next(v for v in robot.forward(later) if v['robot_geom_name']=='gripper0_finger1_collision')
    point=np.asarray(volume['vertices_world']).mean(0,keepdims=True);far=point+3
    for initial in meta['robot_collision_volumes_world']:
        if 'gripper' not in initial['robot_geom_name']:continue
        planes=np.asarray(initial['planes_world'])
        assert np.max(point@planes[:,:3].T+planes[:,3])>0
    # This target point lies on its rotation axis, so it stays fixed as the
    # hand moves. It was not touching the finger at the actual reference state.
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=Scene.from_points(np.empty((0,3)),point,far,spec);scene.skill_contact_surface='fingertips'
    scene=clear_current_robot_scene(scene,meta,10)
    with NativeScene(scene,record,meta,q,.05,articulated=True) as context:
        assert context.validate(q)['accepted']
        failure=context.validate(later)
        assert failure['accepted']
    clear_native_cache()


def test_stationary_index_reuse_keeps_collisions_and_live_queries_across_eviction(calibrated_robot):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache,_TREE_CACHE
    meta,record=calibrated_robot;q=meta['joint_positions'];clear_native_cache()
    palm=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
    point=np.asarray(palm['vertices_world']).mean(0,keepdims=True);far=point+3
    spec={'joint_origin_world_m':far[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':far[0].tolist()},'panel_normal_world':[1,0,0]}
    first=Scene.from_points(point,far,far,spec)
    with NativeScene(first,record,meta,q,.05,articulated=False) as live:
        assert not live.record['stationary_index_reused']
        assert not live.validate(q)['accepted']
        # Make the matching static template oldest. Creating the next moving
        # scene evicts that template while the shared index is pinned.
        other=Scene.from_points(far,far+.1,far+.1,spec)
        with NativeScene(other,record,meta,q,.05,articulated=False) as query:
            assert not query.record['stationary_index_reused']
            assert query.validate(q)['accepted']
        for n in range(1,6):
            moved=far+[n*.01,0,0];scene=Scene.from_points(point,moved,moved,spec)
            with NativeScene(scene,record,meta,q,.05,articulated=False) as query:
                assert query.record['stationary_index_reused']
                assert not query.validate(q)['accepted']
            assert len(_TREE_CACHE)<=2
        clear_native_cache()
        assert not live.validate(q)['accepted']
    clear=Scene.from_points(far,far+.1,far+.1,spec)
    with NativeScene(clear,record,meta,q,.05,articulated=False) as query:
        assert not query.record['stationary_index_reused']
        assert query.validate(q)['accepted']
    clear_native_cache()


def test_occupancy_refresh_clears_measured_free_space_but_keeps_occlusion(monkeypatch):
    import vlm_tamp.contact_skills as module
    old=np.array([[0.,0.,1.],[.1,0,1.],[.2,0,1.]])
    scene=SimpleNamespace(static_points=old,moving_reference=np.array([[10.,10.,10.]]),
        contact_reference=np.array([[10.,10.,10.]]))
    depth=np.full((5,5),np.nan);depth[2,2]=2.;depth[2,3]=.5
    view={'camera_to_world':np.eye(4),'intrinsic':np.array([[10.,0,2],[0,10,2],[0,0,1]]),'depth':depth}
    monkeypatch.setattr(module,'observed_points',lambda *a,**k:(np.array([[0.,0.,2.]]),{}))
    refreshed,evidence=module.refresh_collision_observations(scene,{'test':view},{})
    assert evidence['old_points_cleared_by_measured_free_space']==1
    for point in old[1:]:assert np.linalg.norm(refreshed.static_points-point,axis=1).min()<1e-9
    assert np.linalg.norm(refreshed.static_points-old[0],axis=1).min()>.01


def test_hook_back_surface_must_be_measured_from_a_supporting_view():
    a,z=np.meshgrid(np.linspace(0,2*np.pi,40),np.linspace(-.08,.08,30))
    points=np.c_[.01*np.cos(a.ravel()),.01*np.sin(a.ravel()),z.ravel()]
    with pytest.raises(SkillStop,match='hook_back_surface_unobserved'):
        hook_feature(points,[1,0,0],cameras={})


def test_skills_use_zero_added_padding_without_buffer_fallback(monkeypatch):
    import vlm_tamp.contact_skills as module
    calls=[]
    def native(*args,**kwargs):
        calls.append(kwargs);return SimpleNamespace(record={})
    monkeypatch.setattr(module,'NativeScene',native)
    context=module.skill_native_scene(None,None,None,None,allow_contact=False)
    assert len(calls)==1 and calls[0]['gripper_padding']==0.
    assert not calls[0]['allow_contact']
    assert context.record['contact_skill_clearance']['used_m']==0.




def test_dense_validation_timeout_is_not_reported_as_collision():
    from vlm_tamp.mcvamp_contact_numerical import native_plan_status
    assert native_plan_status(-4,{'timed_out':1})=='dense_validation_timeout'
    assert native_plan_status(-4,{'timed_out':0})=='dense_scene_collision'
    assert native_plan_status(0,{'timed_out':1})=='search_timeout'
    assert native_plan_status(2,{'timed_out':0})=='solved'
