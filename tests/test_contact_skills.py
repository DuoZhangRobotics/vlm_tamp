from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from vlm_tamp.contact_skills import (SkillStop,Budget,contact_landmarks,orientation_menu,
    hook_feature,fit_object_motion,measured_contact_speed)
from vlm_tamp.contact_skill_service import run_skill,momentum_status
from vlm_tamp.timed_execution import schedule_contact
from vlm_tamp.robot_contracts import validate_operation


class LinearChain:
    def forward(self,q):return np.r_[np.asarray(q)[:3],[0,0,0,1.]]


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


def test_contact_orientation_maps_robot_face_to_requested_force():
    for rotation in orientation_menu([0,0,1],[1,0,0],np.eye(3)):
        assert rotation@np.array([0,0,1])==pytest.approx([1,0,0])
        assert np.linalg.det(rotation)==pytest.approx(1.)


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
        run_skill(service,'push',observation_id=1,camera='agentview',region_xyxy=[1,1,10,10])


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
    args={'observation_id':1,'camera':'agentview','region_xyxy':[1,1,10,10],'preview':True}
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
    folder=root/'results/perception-contract-repair-20261028/trials/close_microwave-state017-perception-repaired/attempt01'
    if not folder.exists():pytest.skip('Recorded calibrated robot fixture unavailable')
    from vlm_tamp.coupled_scene import RobotVolumes
    from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
    meta=read_json(folder/'051-arc/planning-input.json')['metadata'];record=read_json(folder/'robot-chain.json')
    opened=read_json(folder/'opened/metadata.json')
    arm=RobotVolumes(RobotChain(record),opened['robot_collision_volumes_world'],opened['joint_positions']).forward(meta['joint_positions'])
    meta['robot_collision_volumes_world']=[v for v in arm if 'gripper' not in v['robot_geom_name']]+meta['robot_collision_volumes_world']
    return meta,record


@pytest.mark.parametrize('surface,local,permitted',[
    ('fingertips',[-.004,0,0],True),
    ('fingertips',[-.018,0,-.032],False),
    ('fingertips',[0,0,-.065],False),
    ('palm',[0,0,-.065],True),
    ('palm',[-.018,0,-.032],False),
])
def test_native_contact_permissions_are_geometric_and_target_specific(calibrated_robot,surface,local,permitted):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    meta,record=calibrated_robot;t=pose_matrix(meta['eef_pose_world_xyzw']);point=transform_points([local],t)
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=Scene.from_points(np.empty((0,3)),point,point,spec)
    scene.skill_contact_surface=surface
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        assert context.validate(meta['joint_positions'])['accepted']==permitted
        assert context.record['proximal_fingers_checked']
    far=point+3
    static=Scene.from_points(point,far,far,spec);static.skill_contact_surface=surface
    with NativeScene(static,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        assert not context.validate(meta['joint_positions'])['accepted']
    clear_native_cache()


def test_calibrated_contact_offsets_use_actual_aperture(calibrated_robot):
    meta,_=calibrated_robot
    pads=contact_landmarks(meta,'fingertips',hook=True)
    assert len(pads)==2
    assert all('pad_collision' in p['name'] for p in pads)
    assert pads[0]['normal']@pads[1]['normal']==pytest.approx(-1.)


@pytest.mark.parametrize('part,stationary,articulated,allowed',[
    ('gripper0_finger1_collision',False,True,True),
    ('gripper0_finger1_collision',False,False,False),
    ('gripper0_finger1_collision',True,True,False),
    ('gripper0_hand_collision',False,True,False),
])
def test_existing_target_finger_contact_can_continue_but_is_not_a_world_exemption(calibrated_robot,part,stationary,articulated,allowed):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    meta,record=calibrated_robot
    volume=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']==part)
    point=np.asarray(volume['vertices_world']).mean(0,keepdims=True);far=point+3
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=Scene.from_points(point if stationary else np.empty((0,3)),far if stationary else point,far,spec)
    scene.skill_contact_surface='fingertips'
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=articulated) as context:
        assert context.validate(meta['joint_positions'])['accepted']==allowed
        assert context.record['existing_finger_target_contact']['additional_contact_padding_m']==0.
    clear_native_cache()


def test_continuation_does_not_permit_a_new_finger_collision(calibrated_robot):
    from vlm_tamp.coupled_scene import Scene,RobotVolumes
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
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
    with NativeScene(scene,record,meta,q,.05,articulated=True) as context:
        assert context.validate(q)['accepted']
        failure=context.validate(later)
        assert not failure['accepted'] and failure['moving_rejections']>0
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


def test_region_candidates_are_measured_points_and_cover_surface_height():
    from vlm_tamp.contact_skills import region_contact_candidates
    y,z=np.meshgrid(np.linspace(-.06,.06,20),np.linspace(.9,1.15,30))
    points=np.c_[np.zeros(y.size),y.ravel(),z.ravel()]
    candidates=region_contact_candidates(points,points[0])
    assert len(candidates)==5
    assert max(p[2] for p in candidates)>1.09
    assert all(np.linalg.norm(points-p,axis=1).min()<1e-12 for p in candidates)


def test_dense_validation_timeout_is_not_reported_as_collision():
    from vlm_tamp.mcvamp_contact_numerical import native_plan_status
    assert native_plan_status(-4,{'timed_out':1})=='dense_validation_timeout'
    assert native_plan_status(-4,{'timed_out':0})=='dense_scene_collision'
    assert native_plan_status(0,{'timed_out':1})=='search_timeout'
    assert native_plan_status(2,{'timed_out':0})=='solved'


def test_advisory_approach_context_preserves_world_data_but_does_not_veto_it(calibrated_robot):
    from vlm_tamp.contact_skills import robot_only_planning_scene,skill_native_scene
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points
    meta,record=calibrated_robot
    point=transform_points([[0,0,-.065]],pose_matrix(meta['eef_pose_world_xyzw']))
    far=point+3
    spec={'joint_origin_world_m':far[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':far[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=Scene.from_points(point,far,far,spec);scene.skill_contact_surface='fingertips'
    with skill_native_scene(scene,record,meta,meta['joint_positions']) as checker:
        assert not checker.validate(meta['joint_positions'])['accepted']
    planning=robot_only_planning_scene(scene)
    with skill_native_scene(planning,record,meta,meta['joint_positions']) as checker:
        assert checker.validate(meta['joint_positions'])['accepted']
    assert np.array_equal(scene.static_points,point)
    assert len(planning.static_points)==0 and len(planning.moving_reference)==0


@pytest.mark.parametrize('policy,robot_valid,allowed',[('advisory',True,True),('veto',True,False),('advisory',False,False)])
def test_advisory_environment_predictions_do_not_veto_robot_valid_strokes(monkeypatch,policy,robot_valid,allowed):
    import vlm_tamp.contact_skills as module
    class Check:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):return {'accepted':False,'joint_bound_violations':[],
            'timed_out':False,'blocking_robot_part':'gripper0_finger2_collision',
            'blocking_surface':'observed stationary geometry'}
    monkeypatch.setattr(module,'skill_native_scene',lambda *a,**k:Check())
    scene=SimpleNamespace(moving_reference=np.array([[1.,1.,1.]]),contact_reference=np.array([[1.,1.,1.]]),
        skill_collision_policy=policy,skill_contact_collision_log={'count':0,'examples':[]})
    ik=SimpleNamespace(chain=LinearChain(),self_valid=lambda q:robot_valid,
        solve_many=lambda pose,seed,**kwargs:[{'joints':np.r_[pose[:3],np.zeros(4)]}])
    call=lambda:module.cartesian_path(ik,scene,{}, {},np.zeros(7),[.03,0,0,0,0,0,1],Budget(5),contact_local=np.zeros(3))
    if allowed:
        path=call()
        assert path[-1,0]==pytest.approx(.03)
        # Scene diagnostics are outside this bounded IK/path query, so they
        # cannot exhaust its budget and indirectly veto an advisory stroke.
        assert scene.skill_contact_collision_log['count']==0
    else:
        with pytest.raises(SkillStop,match='contact_path_blocked'):call()


@pytest.mark.parametrize('observed_solved',[False,True])
def test_approach_prefers_observed_route_but_failed_prediction_cannot_veto(observed_solved):
    from collections import Counter
    from vlm_tamp.contact_skills import approach_connection
    class Context:
        def __init__(self,solved):self.solved=solved;self.calls=0
        def validate(self,q):return {'accepted':True}
        def plan(self,*a,**k):self.calls+=1;return np.zeros((2,7)),{'exact_solution':self.solved}
    robot=Context(True);observed=Context(observed_solved)
    path,result,choice=approach_connection(robot,observed,np.zeros(7),np.ones(7),Budget(10),Counter())
    assert result['exact_solution'] and not choice['environment_prediction_is_veto']
    assert robot.calls==int(not observed_solved)


@pytest.mark.parametrize('policy,allowed',[('advisory',True),('veto',False)])
def test_curved_ik_interpolation_is_advisory_in_simulation(monkeypatch,policy,allowed):
    import vlm_tamp.contact_skills as module
    class CurvedChain:
        def forward(self,q):return np.r_[[q[0],.008*np.sin(np.pi*q[6]),q[2]],[0,0,0,1.]]
    class Check:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):return {'accepted':True}
    monkeypatch.setattr(module,'skill_native_scene',lambda *a,**k:Check())
    log={'maximum_m':0.,'samples_over_threshold':0}
    scene=SimpleNamespace(moving_reference=np.array([[1.,1.,1.]]),contact_reference=np.array([[1.,1.,1.]]),
        skill_collision_policy=policy,skill_path_deviation_log=log)
    ik=SimpleNamespace(chain=CurvedChain(),self_valid=lambda q:True,
        solve_many=lambda pose,seed,**kwargs:[{'joints':np.r_[pose[:3],[0.,0.,0.,1.]]}])
    call=lambda:module.cartesian_path(ik,scene,{}, {},np.zeros(7),[.03,0,0,0,0,0,1],Budget(5),contact_local=np.zeros(3))
    if allowed:
        assert call()[-1,0]==pytest.approx(.03)
        assert log['maximum_m']>.006 and log['samples_over_threshold']>0
    else:
        with pytest.raises(SkillStop,match='contact_path_blocked'):call()


@pytest.mark.parametrize('preferred_succeeds',[True,False])
def test_hook_prefers_observed_posture_and_preserves_advisory_fallback(monkeypatch,preferred_succeeds):
    import vlm_tamp.contact_skills as module
    calls=[]
    def search(ik,scene,*args):
        calls.append(scene.skill_collision_policy)
        if scene.skill_collision_policy=='veto' and not preferred_succeeds:
            raise SkillStop('contact_search_exhausted')
        return {'total_seconds':0.}
    monkeypatch.setattr(module,'_plan_contact_at_point',search)
    scene=SimpleNamespace(skill_collision_policy='advisory',contact_reference=np.zeros((1,3)))
    result=module.plan_engagement(None,scene,{}, {},{},None,'hook_pull',Budget(10))
    assert calls==(['veto'] if preferred_succeeds else ['veto','advisory'])
    assert scene.skill_collision_policy=='advisory'
    assert result['observed_hook_posture_preference']['selected']==preferred_succeeds
