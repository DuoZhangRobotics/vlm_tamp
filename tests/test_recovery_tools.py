import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from vlm_tamp.recovery_decision import parse_decision
from vlm_tamp.recovery_feedback import failure_loci
from vlm_tamp.coupled_storage import write_json

def choice(**changes):
    return dict(version='strategy-v2',operation='try_contact',contact_mode='press_drag',
        approach_tilt_deg=45,motion_extent='probe',step_mm=5,posture_rank=None,retreat_mm=None,
        evidence='The last plan failed at its goal.',expected_observation='A shorter feasible motion.',**changes)

@pytest.mark.parametrize('key,value', [('step_mm',0),('step_mm',100),('step_mm',True),
    ('posture_rank',6),('posture_rank',False),('retreat_mm',20),('operation','disable_collision')])
def test_invalid_motion_choices_rejected(key,value):
    data=choice();data[key]=value
    with pytest.raises(ValueError):parse_decision(json.dumps(data))

def test_release_schema_has_no_hidden_contact_selection():
    data=choice();data.update(operation='release_reposition',contact_mode=None,approach_tilt_deg=None,
        motion_extent=None,step_mm=None,posture_rank=2,retreat_mm=40)
    assert parse_decision(json.dumps(data))==data

def test_feedback_distinguishes_start_from_goal_collision_and_search(tmp_path):
    write_json(tmp_path/'planning.json',{'attempts':[
        {'status':'invalid_endpoint','exact_solution':False,'endpoints':{'start':{'accepted':False},'goal':{'accepted':True}}},
        {'reason':'terminal_projection'}, {'status':'native_worker_timeout','exact_solution':False},
        {'status':'invalid_endpoint','exact_solution':False,'endpoints':{'start':{'accepted':True},'goal':{'accepted':False}}}]})
    assert failure_loci(tmp_path)=={'start_collision':1,'goal_constraint':1,'path_search':1,'goal_collision':1}

def test_step_choice_changes_actual_goal_distance_and_does_not_finish_early(tmp_path,monkeypatch):
    from vlm_tamp import recovery_episode as module
    session=object.__new__(module.RecoveryTools)
    session.current_choice={};session.progress=SimpleNamespace(observed_progress=0.,progress=0.)
    session.spec={'target_progress':.015,'motion':'close'};session.config={'scene_voxel_spacing_m':.003}
    session.state={'control_steps':100,'metadata':{'joint_positions':[0]*7,'eef_pose_world_xyzw':[0,0,0,0,0,0,1]}}
    session.folder=tmp_path;session.plan_number=0;session.step_m=.005;session.posture_rank=2;session.goal_reached=False
    session.scene=session.record=session.ik=None
    planner=Mock(return_value=(np.zeros((2,7)),{'accepted':True}));monkeypatch.setattr(module,'segment',planner)
    session.issue=Mock()
    def observe(**kwargs):
        session.progress.progress+=.005;session.progress.observed_progress+=.005
        return {'tracking_accepted':True}
    session.observe=observe
    result=session.motion('probe',tmp_path)
    assert planner.call_args.args[5]==.005 and planner.call_args.kwargs['posture_rank']==2
    assert result['executed_plan_distances_m']==[.005] and not session.goal_reached
    assert session.issue.call_count==1

def test_requested_ik_rank_never_silently_falls_back(tmp_path,monkeypatch):
    from vlm_tamp import reliable_mechanisms as module
    class Context:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):return {'accepted':True}
        def project(self,q):raise AssertionError('Unavailable rank must not plan a different posture')
    monkeypatch.setattr(module,'NativeScene',Context)
    monkeypatch.setattr(module,'Mechanism',lambda spec:SimpleNamespace(pose=lambda pose,p:pose))
    ik=SimpleNamespace(chain=SimpleNamespace(forward=lambda q:[0,0,0,0,0,0,1]),solve_many=lambda *a,**kw:[{'joints':[0]*7}])
    path,record=module.segment(SimpleNamespace(spec={}),{}, {},ik,np.zeros(7),.005,tmp_path,posture_rank=2)
    assert path is None and record['reason']=='requested_posture_unavailable'

def test_release_executes_opening_but_does_not_bypass_invalid_start(tmp_path,monkeypatch):
    from vlm_tamp import recovery_episode as module
    session=object.__new__(module.RecoveryTools)
    session.state={'metadata':{'eef_pose_world_xyzw':[0,0,0,0,0,0,1],'joint_positions':[0]*7}}
    session.current_choice={'contact_mode':'pinch'};session.sim=SimpleNamespace(context={'mode':'pinch'})
    session.resources=SimpleNamespace(grasps=Mock(return_value={}))
    session.cloud=np.zeros((3,3));session.scene=session.record=None
    session.ik=SimpleNamespace(solve_many=Mock(side_effect=AssertionError('Invalid start cannot be waived')))
    def issue(path,label,gripper,**kwargs):
        assert gripper==-1 and session.sim.context is None
    session.issue=Mock(side_effect=issue);session.observe=Mock(return_value={'tracking_accepted':False})
    class Context:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):return {'accepted':False,'static_rejections':1}
    monkeypatch.setattr(module,'NativeScene',Context)
    result=session.release_reposition({'retreat_mm':20,'posture_rank':None},tmp_path)
    assert result['release_executed'] and not result['retreat_executed']
    assert result['failure_loci']=={'start_collision':1}
    assert session.current_choice is None and session.sim.context is None
    assert session.issue.call_count==1 and session.observe.call_count==1

def test_pinch_reports_the_actual_requested_posture_rank(tmp_path,monkeypatch):
    from vlm_tamp import reliable_mechanisms as module
    pose=np.array([0.,0.,0.,0.,0.,0.,1.])
    monkeypatch.setattr(module,'candidate_variants',lambda *a:[(0,'test',pose)])
    monkeypatch.setattr(module,'estimated_contact_aperture',lambda *a:[.02,-.02])
    monkeypatch.setattr(module,'metadata_at',lambda sim,meta,q,ap:meta)
    monkeypatch.setattr(module,'segment',lambda *a,**kw:(np.zeros((2,7)),{'progress':a[5]}))
    solutions=[{'joints':[float(i)]*7,'pose_world_xyzw':pose.tolist(),'posture_rank':i} for i in range(3)]
    ik=SimpleNamespace(solve_many=lambda *a,**kw:solutions)
    class Context:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):return {'accepted':True}
        def plan(self,a,b,**kw):return np.array([a,b]),{'exact_solution':True}
    monkeypatch.setattr(module,'NativeScene',Context)
    cloud=np.array([[x,y,z] for x in (-.001,.001) for y in (-.001,.001) for z in (-.001,.001)])
    state={'metadata':{'joint_positions':[0]*7,'gripper_qpos':[.04,-.04],'robot_calibration':{}}}
    spec={'source_anchor_world_m':[0,0,0],'panel_normal_world':[0,0,-1],'joint_family':'prismatic'}
    config={'maximum_candidate_grasps':100,'maximum_total_ik_postures':256}
    selected,path=module.select_grasp({'scores':[1.]},cloud,spec,None,{},ik,state,None,tmp_path,config,
                                     strategy={'posture_rank':2,'step_mm':5})
    assert selected['solution_joints']==[2.]*7 and selected['ik_rank']==2
    assert selected['probe']['progress']==.005
