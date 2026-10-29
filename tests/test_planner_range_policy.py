"""Regression cases for feasible proposals previously vetoed by heuristics."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vlm_tamp.partial_pose_arc import short_wrist_limit
from vlm_tamp.curobo_v2_arc import CuroboV2ArcProposer
from vlm_tamp.robot_contracts import validate_operation


def test_horizon_and_refinement_flag_do_not_impose_a_hidden_wrist_cap():
    context=SimpleNamespace(scene=SimpleNamespace(config={}),grasp_constraint='point',
        motion_options={'motion_policy':'short','hinge_refinement_accepted':False})
    assert short_wrist_limit(context,np.radians(3))==np.pi
    context.motion_options['max_wrist_reorientation_deg']=8.
    assert short_wrist_limit(context,np.radians(3))==pytest.approx(np.radians(8))
    context.motion_options['max_wrist_reorientation_deg']=float('nan')
    with pytest.raises(ValueError):short_wrist_limit(context,.1)


def test_failed_gpu_seed_outside_final_tolerance_gets_refined(tmp_path,monkeypatch):
    import vlm_tamp.curobo_v2_arc as module
    q=np.zeros(7);projected=[]
    def project(seed,angle,**kwargs):
        projected.append(seed.copy())
        return q,{'accepted':True,'position_error_m':.0001,'position_tolerance_m':.003}
    context=SimpleNamespace(axis=np.array([0.,0.,1.]),scene=SimpleNamespace(config={}),
        project=lambda q:(q,{'accepted':True}),validate=lambda q:{'accepted':True},progress=lambda q:.1)
    proposer=object.__new__(CuroboV2ArcProposer)
    proposer.closed=False;proposer.context=context;proposer.goal=.1;proposer.arc_id=1
    proposer.folder=tmp_path;proposer.queries=[]
    proposer.model=SimpleNamespace(reference_rotation=np.eye(3),grasp_constraint='point',position_tolerance_m=.003,
        point=lambda a:np.zeros(3),residual=lambda q,a:np.array([.0038,0.,0.,0.]),project=project)
    proposer.ik=SimpleNamespace(client=SimpleNamespace(request=lambda *a,**k:{'solutions':[q.tolist()],
        'success':[False],'elapsed_seconds':.1,'solver_seconds':.1}))
    monkeypatch.setattr(module,'wrist_excursion',lambda *a:np.radians(29.6))
    goals,checks=proposer.candidates(q,.1)
    assert len(projected)==1 and len(goals)==1
    assert checks[0]['gpu_input_position_error_m']>.003
    assert checks[0]['accepted'] and checks[0]['wrist_limit_deg']==180
    assert proposer.queries[0]['unexamined_seeds']==0


def test_pose_budget_includes_learned_and_oblique_grasps(monkeypatch):
    import vlm_tamp.coupled_pipeline as module
    # This neural orientation violates the old frontality threshold and its
    # original translation is remote. Translation is subsequently re-grounded.
    pose=np.r_[[2.,2.,2.],Rotation.from_rotvec([.5,.4,.3]).as_quat()]
    monkeypatch.setattr(module,'candidate_variants',lambda *a:[(0,'test',pose)])
    metadata={'eef_pose_world_xyzw':[0.,0.,0.,0.,0.,0.,1.],'robot_calibration':{},
        'robot_collision_volumes_world':[{'robot_geom_name':'left_pad_collision','vertices_world':[[0.,0.,0.]]},
                                        {'robot_geom_name':'right_pad_collision','vertices_world':[[0.,0.,0.]]}]}
    anchors=[np.array([.1,.2,z]) for z in np.linspace(.8,1.1,8)]
    rows,pad=module.grasp_poses({},anchors,np.array([0.,-1.,0.]),metadata)
    first=rows[:24]
    assert {r['orientation_family'] for r in first}=={'frontal','oblique','learned'}
    assert {r['anchor_index'] for r in first}.issuperset({0,4,7})
    assert all(min(np.linalg.norm(r['pose'][:3]-a) for a in anchors)<1e-10 for r in rows)


def test_public_schema_accepts_explicit_tolerances_and_wrist_policy():
    validate_operation('plan_arc',{'degrees':15,'position_tolerance_m':.006,
        'orientation_tolerance_deg':3,'max_wrist_reorientation_deg':None})
    with pytest.raises(ValueError):
        validate_operation('plan_arc',{'degrees':15,'position_tolerance_m':-.006})


def test_plan_sequence_prices_and_reports_the_requested_timing(tmp_path):
    from serve_subscription_robot import RobotService
    service=RobotService.__new__(RobotService)
    service.config={}
    service.episode_status=lambda:{'remaining_wall_seconds':120.}
    service.workspace=tmp_path;service.public=tmp_path/'robot_data';service.attempt=1
    (service.public/'attempt01').mkdir(parents=True)
    received=[];timing={'speed':.5,'hold':0,'close_steps':12,'complete_sequence_controls':62}
    def select(degrees,candidates,seed,options):
        received.append(options)
        return {'planned_contact':np.zeros((2,7)),'approach':np.zeros((2,7)),
            'source':'test','search_seconds':0.,'joints':[0.]*7,'budget_timing_assumptions':timing}
    service._select_contact=select
    service.keep_plan=lambda *a,**k:{'plan_id':'p1'}
    args={'degrees':15,'speed':.35,'approach_speed':.5,'hold':0,'close_steps':12}
    validate_operation('plan_sequence',args)
    result=service.plan_sequence(**args)
    assert received[0]['grasp_budget_speed_rad_s']==.5
    assert received[0]['grasp_budget_hold_steps']==0
    assert received[0]['grasp_closure_steps_reserved']==12
    assert result['budget_timing_assumptions']==timing


def test_free_pose_search_can_reach_a_valid_ninth_ik_branch(tmp_path,monkeypatch):
    import vlm_tamp.surface_contact_tools as module
    monkeypatch.setattr(module,'preserve_scene',lambda *a:None)
    class Context:
        record={}
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def plan(self,start,goal,**kwargs):
            good=goal[0]==8
            return np.vstack([start,goal]),{'exact_solution':good,'status':'solved' if good else 'invalid_endpoint'}
    monkeypatch.setattr(module,'NativeScene',Context)
    rows=[{'joints':[float(i)]+[0.]*6} for i in range(9)]
    ik=SimpleNamespace(solve_many=lambda pose,start,maximum=None:rows if maximum is None else rows[:maximum])
    metadata={'joint_positions':[0.]*7,'eef_pose_world_xyzw':[0.,0.,0.,0.,0.,0.,1.],'robot_collision_volumes_world':[]}
    path,result=module.plan_pose(ik,None,None,metadata,metadata['eef_pose_world_xyzw'],tmp_path/'pose')
    assert result['exact_solution'] and result['ik_index']==8 and path[-1,0]==8
