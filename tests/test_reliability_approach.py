"""Microwave grasp selection must fit physical execution into the episode."""
from types import SimpleNamespace
import numpy as np
import pytest
import reliable_microwave_final as engine
from vlm_tamp import hinged_contact_tools as selector


def setup_selector(monkeypatch,steps,far_q=2.):
    row={'pose':np.r_[np.zeros(3),[0,0,0,1]],'source':'test depth-supported handle'}
    monkeypatch.setattr(selector,'grasp_poses',lambda *args:([row],np.zeros(3)))
    near={'joints':[.1]*7,'pose_world_xyzw':row['pose'].tolist(),'joint_reserve_rad':.2,'posture_cost':.3}
    far={**near,'joints':[far_q]*7,'joint_reserve_rad':1.,'posture_cost':5.}
    ik=SimpleNamespace(chain=None,solve_many=lambda *args,**kw:[far.copy(),near.copy()])
    class Context:
        record={}
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):return {'accepted':True}
        def plan(self,a,b,**kw):return np.array([a,b]),{'exact_solution':True}
    monkeypatch.setattr(selector,'NativeScene',Context)
    monkeypatch.setattr(selector,'virtual_metadata',lambda *args:{})
    monkeypatch.setattr(selector,'contact_plan',lambda *args,**kw:(np.array([[0]*7,[.1]*7]),{'exact_solution':True}))
    state={'metadata':{'joint_positions':[0]*7},'control_steps':steps}
    config={'maximum_grasp_poses':24,'maximum_grasp_search_seconds':10.,'maximum_contact_plan_seconds':1.}
    return state,ik,config


def test_preserves_joint_reserve_when_complete_approaches_fit(tmp_path,monkeypatch):
    state,ik,config=setup_selector(monkeypatch,60);scene=SimpleNamespace(robot=None)
    chosen=engine.select_grasp({},[],state,ik,{},scene,scene,{'panel_normal_world':[0,1,0]},.05,tmp_path/'search',config)
    assert np.allclose(chosen['joints'],2.)


def test_rejects_complete_approach_that_leaves_no_manipulation_budget(tmp_path,monkeypatch):
    state,ik,config=setup_selector(monkeypatch,400);scene=SimpleNamespace(robot=None)
    with pytest.raises(ValueError,match='No candidate'):
        engine.select_grasp({},[],state,ik,{},scene,scene,{'panel_normal_world':[0,1,0]},.05,tmp_path/'search',config)


def test_skips_higher_reserve_approach_when_it_exceeds_the_budget(tmp_path,monkeypatch):
    state,ik,config=setup_selector(monkeypatch,60,far_q=4.);scene=SimpleNamespace(robot=None)
    chosen=engine.select_grasp({},[],state,ik,{},scene,scene,{'panel_normal_world':[0,1,0]},.05,tmp_path/'search',config)
    assert np.allclose(chosen['joints'],.1)


def test_closing_completion_adds_one_bounded_seating_motion_only_near_goal():
    config={'closing_seating_trigger_deg':3.,'closing_seating_allowance_deg':2.}
    requested,seating=engine.completion_motion({'motion':'close'},np.radians(2.),config)
    assert seating and abs(requested-np.radians(4.))<1e-12
    requested,seating=engine.completion_motion({'motion':'close'},np.radians(10.),config)
    assert not seating and abs(requested-np.radians(10.))<1e-12
    requested,seating=engine.completion_motion({'motion':'open'},np.radians(2.),config)
    assert not seating and abs(requested-np.radians(2.))<1e-12
