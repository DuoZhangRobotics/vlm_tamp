"""Microwave grasp selection must fit physical execution into the episode."""
from types import SimpleNamespace
import numpy as np
import pytest
import reliable_microwave_final as engine
from vlm_tamp import hinged_contact_tools as selector


def setup_selector(monkeypatch,steps,far_q=2.,observed_blocked=False):
    row={'pose':np.r_[np.zeros(3),[0,0,0,1]],'source':'test depth-supported handle'}
    monkeypatch.setattr(selector,'grasp_poses',lambda *args:([row],np.zeros(3)))
    near={'joints':[.1]*7,'pose_world_xyzw':row['pose'].tolist(),'joint_reserve_rad':.2,'posture_cost':.3}
    far={**near,'joints':[far_q]*7,'joint_reserve_rad':1.,'posture_cost':5.}
    ik=SimpleNamespace(chain=None,solve_many=lambda *args,**kw:[far.copy(),near.copy()])
    class Context:
        record={}
        def __init__(self,*args,**kwargs):self.policy=kwargs['collision_policy']
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):return {'accepted':not (observed_blocked and self.policy=='veto' and np.max(q)>.05)}
        def plan(self,a,b,**kw):return np.array([a,b]),{'exact_solution':True}
    monkeypatch.setattr(selector,'NativeScene',Context)
    monkeypatch.setattr(selector,'virtual_metadata',lambda *args:{})
    monkeypatch.setattr(selector,'contact_plan',lambda *args,**kw:pytest.fail('Manipulation must wait for physical closure'))
    state={'metadata':{'joint_positions':[0]*7,'eef_pose_world_xyzw':[0,0,0,0,0,0,1],
        'robot_collision_volumes_world':[{'robot_geom_name':'finger_pad_collision','vertices_world':[[0,0,-.01],[0,0,.01]]}]},'control_steps':steps}
    config={'maximum_grasp_poses':24,'maximum_grasp_search_seconds':10.,'maximum_contact_plan_seconds':1.,
        'grasp_standoff_distances_m':[0.],'grasp_pad_depth_fractions':[.5]}
    return state,ik,config


def test_selects_lower_control_cost_instead_of_larger_extra_joint_reserve(tmp_path,monkeypatch):
    state,ik,config=setup_selector(monkeypatch,60);scene=SimpleNamespace(robot=None)
    chosen=engine.select_grasp({},[],state,ik,{},scene,scene,{'panel_normal_world':[0,1,0]},.05,tmp_path/'search',config)
    assert np.allclose(chosen['joints'],.1)
    assert chosen['planned_contact'] is None
    assert chosen['manipulation_plan_requires_measured_closure']


def test_rejects_approach_that_cannot_fit_with_explicit_closure(tmp_path,monkeypatch):
    state,ik,config=setup_selector(monkeypatch,551);scene=SimpleNamespace(robot=None)
    with pytest.raises(ValueError,match='No executable approach'):
        engine.select_grasp({},[],state,ik,{},scene,scene,{'panel_normal_world':[0,1,0]},.05,tmp_path/'search',config)


def test_skips_longer_approach_when_it_exceeds_the_budget(tmp_path,monkeypatch):
    state,ik,config=setup_selector(monkeypatch,60,far_q=4.);scene=SimpleNamespace(robot=None)
    chosen=engine.select_grasp({},[],state,ik,{},scene,scene,{'panel_normal_world':[0,1,0]},.05,tmp_path/'search',config)
    assert np.allclose(chosen['joints'],.1)


def test_advisory_policy_keeps_robot_valid_fallback_when_scene_prediction_blocks(tmp_path,monkeypatch):
    state,ik,config=setup_selector(monkeypatch,400,observed_blocked=True)
    scene=SimpleNamespace(robot=None)
    chosen=engine.select_grasp({},[],state,ik,{},scene,scene,{'panel_normal_world':[0,1,0]},.05,tmp_path/'search',config)
    assert np.allclose(chosen['joints'],.1)
    assert chosen['search_summary']['advisory_fallback_used']
    assert chosen['approach_selection']['route']=='advisory'


def test_keeps_feasible_result_when_search_deadline_expires(tmp_path,monkeypatch):
    state,ik,config=setup_selector(monkeypatch,60);scene=SimpleNamespace(robot=None)
    clock=[0.];monkeypatch.setattr(selector.time,'perf_counter',lambda:clock[0])
    context=selector.NativeScene
    def plan(self,a,b,**kwargs):
        clock[0]=11.
        return np.array([a,b]),{'exact_solution':True}
    monkeypatch.setattr(context,'plan',plan)
    chosen=engine.select_grasp({},[],state,ik,{},scene,scene,{'panel_normal_world':[0,1,0]},.05,tmp_path/'search',config)
    assert np.allclose(chosen['joints'],.1)
    assert chosen['search_summary']['deadline_reached']


def test_closing_completion_adds_one_bounded_seating_motion_only_near_goal():
    config={'closing_seating_trigger_deg':3.,'closing_seating_allowance_deg':2.}
    requested,seating=engine.completion_motion({'motion':'close'},np.radians(2.),config)
    assert seating and abs(requested-np.radians(4.))<1e-12
    requested,seating=engine.completion_motion({'motion':'close'},np.radians(10.),config)
    assert not seating and abs(requested-np.radians(10.))<1e-12
    requested,seating=engine.completion_motion({'motion':'open'},np.radians(2.),config)
    assert not seating and abs(requested-np.radians(2.))<1e-12
