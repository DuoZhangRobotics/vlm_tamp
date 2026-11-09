"""Whole-sequence costs must preserve feasible branches without masking numerical failures."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vlm_tamp.planner_object_transfer as planner


def setup_search(monkeypatch: pytest.MonkeyPatch, *, fail_runtime: bool = False) -> tuple:
    """Provide a graph whose first valid lift leads to an over-budget later path."""
    calls=[]

    def posture(value: float) -> np.ndarray:
        """Encode a candidate in one joint while leaving the other joints fixed."""
        result=np.zeros(7);result[0]=value;result[3]=-1.5
        return result

    class Context:
        def __init__(self,*args: object,**kwargs: object) -> None:
            """Use a deterministic stand-in for the existing native scene."""
        def __enter__(self) -> 'Context':
            """Keep the context live for its native queries."""
            return self
        def __exit__(self,*args: object) -> None:
            """Do not swallow exceptions."""
        def set_object_carry(self,*args: object) -> None:
            """Record that every branch uses the same declared constraint family."""
            calls.append('constraint')
        def plan(self,start: np.ndarray,goal: np.ndarray,**kwargs: object) -> tuple:
            """Return original endpoint states, including explicit worker failure."""
            calls.append(('path',float(start[0]),float(goal[0]),kwargs['constrained']))
            if fail_runtime and start[0]!=0:
                return np.empty((0,7)),{'exact_solution':False,'status':'native_worker_failed'}
            return np.array([start,goal]),{'exact_solution':True,'status':'solved'}

    def solve(target: np.ndarray,local: np.ndarray,seed: np.ndarray,rotation: np.ndarray,**kwargs: object) -> list:
        """Return only native candidates from the declared graph."""
        phase=int(target[0])
        values=([1.,-1.] if phase==0 else [-2.] if phase==1 and seed[0]>0
                else [-1.1] if phase==1 else [2.] if seed[0]<-1.5 else [-1.2])
        calls.append(('ik',phase,float(seed[0])))
        return [{'joints':posture(value).tolist()} for value in values]

    monkeypatch.setattr(planner,'NativeScene',Context)
    meta={'joint_positions':posture(0).tolist(),'eef_pose_world_xyzw':[0,0,0,0,0,0,1]}
    ik=SimpleNamespace(events=[{'backend':'test_native'}],configure_scene=lambda ctx:None,
                       solve_contact_many=solve,chain=SimpleNamespace(forward=lambda q:np.r_[q[:3],[0,0,0,1]]))
    metadata_at=lambda q:{**meta,'joint_positions':q.tolist(),'eef_pose_world_xyzw':ik.chain.forward(q).tolist()}
    targets=[(name,np.array([index,0.,0.]),np.zeros(3)) for index,name in enumerate(['lift','transfer','lower'])]
    return ik,meta,metadata_at,targets,calls


def test_complete_sequence_keeps_an_alternative_when_greedy_branch_exceeds_budget(
    tmp_path: Path,monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retaining two prefixes finds a valid complete sequence that first-valid selection loses."""
    ik,meta,metadata_at,targets,calls=setup_search(monkeypatch)
    path,stages,report=planner.plan_object_transfer(ik,object(),{},meta,metadata_at,
        np.zeros(3),np.array([0,0,1]),np.array([0,0,1]),np.array([0,0,1]),targets,tmp_path,
        speed=1.,hold=0,release_steps=0,available_controls=30,budget_seconds=10,width=2)
    assert path[0,0]==0 and path[-1,0]==pytest.approx(-1.2)
    assert [row['stage'] for row in stages]==['lift','transfer','lower']
    assert report['complete_controls_including_release']<=30
    assert report['selected_branches']==[1,0,0]
    assert report['budget_rejected_prefixes']>=1
    assert report['global_optimality_claimed'] is False
    assert all(row[3] for row in calls if isinstance(row,tuple) and row[0]=='path')


def test_narrow_search_failure_does_not_claim_geometric_infeasibility(
    tmp_path: Path,monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite proposal limit can miss a feasible sequence without proving the task impossible."""
    ik,meta,metadata_at,targets,_=setup_search(monkeypatch)
    with pytest.raises(ValueError) as error:
        planner.plan_object_transfer(ik,object(),{},meta,metadata_at,np.zeros(3),
            np.array([0,0,1]),np.array([0,0,1]),np.array([0,0,1]),targets,tmp_path,
            speed=1.,hold=0,release_steps=0,available_controls=30,budget_seconds=10,width=1)
    diagnostic=error.value.planning_diagnostic
    assert diagnostic['status']=='object_transfer_candidates_exhausted'
    assert not diagnostic['physical_infeasibility_proved']
    assert not diagnostic['physical_motion_issued']


def test_worker_failure_is_not_hidden_by_a_previously_valid_prefix(
    tmp_path: Path,monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native runtime failure remains an error instead of becoming another branch or old solution."""
    ik,meta,metadata_at,targets,calls=setup_search(monkeypatch,fail_runtime=True)
    with pytest.raises(RuntimeError,match='native_worker_failed'):
        planner.plan_object_transfer(ik,object(),{},meta,metadata_at,np.zeros(3),
            np.array([0,0,1]),np.array([0,0,1]),np.array([0,0,1]),targets,tmp_path,
            speed=1.,hold=0,release_steps=0,available_controls=200,budget_seconds=10,width=2)
    assert sum(row[0]=='ik' for row in calls if isinstance(row,tuple))==2


def test_release_and_hold_are_included_in_the_single_joined_schedule() -> None:
    """Per-phase rounding and repeated holds cannot inflate or hide the actual execution cost."""
    start=np.zeros(7);middle=start.copy();middle[0]=.011;end=start.copy();end[0]=.022
    paths=[np.array([start,middle]),np.array([middle,end])]
    assert planner.scheduled_controls(paths,1.,6,12)==19
