from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_preplanned_sequence_regression as sequence


def test_success_in_prepared_motion_prevents_release_or_retreat(tmp_path,monkeypatch):
    folder=tmp_path/'prepared-sequence';folder.mkdir()
    sim=SimpleNamespace(folder=tmp_path,segments=1,prepared_sequence={
        'path':np.array([[0.]*7,[.2]*7]),'release':True,'source':'test'})
    calls=[]
    def execute(sim,state,refs,**kwargs):
        sim.segments+=1;calls.append(kwargs['gripper'])
        if len(calls)==2:raise sequence.experiment.NativeSuccess('simulator_success')
        return {'control_steps':120,'metadata':{'joint_positions':[.01]*7}},{}
    monkeypatch.setattr(sequence,'original_execute',execute)
    with pytest.raises(sequence.experiment.NativeSuccess):
        sequence.execute_prepared(sim,{},np.zeros((5,7)),gripper=1)
    assert calls==[1,1]
    assert (folder/'outcome.json').exists()


def test_bowl_preflight_retreat_is_excluded_before_execution(tmp_path,monkeypatch):
    qs=[[0.]*7,[1.,0.,0.,0.,0.,0.,0.],[1.,0.,.05,0.,0.,0.,0.]]
    chosen={'solution_joints':qs[0],'fk_pose_world_xyzw':[0,0,0,0,0,0,1],
        'task_preflight':{'solutions':[{'joints':q} for q in qs]}}
    monkeypatch.setattr(sequence,'original_bowl_select',lambda *a,**k:chosen)
    chain=SimpleNamespace(forward=lambda q:[*q[:3],0,0,0,1])
    sim=SimpleNamespace(folder=tmp_path)
    spec={'task_type':'pick_place','object_support_point_initial_world_m':[0,0,0],
          'goal':{'target_world_m':[1,0,0]}}
    sequence.bowl_select(None,None,spec,None,SimpleNamespace(chain=chain),None,sim,None,tmp_path,{})
    np.testing.assert_array_equal(sim.prepared_sequence['path'][-1],qs[1])
    assert not any(q[2]>.01 for q in sim.prepared_sequence['path'])


def test_unknown_rotary_direction_retains_the_existing_probe(tmp_path,monkeypatch):
    chosen={'horizon_support':'full_task','tested_ik_postures':1}
    monkeypatch.setattr(sequence,'original_mechanism_select',lambda *a,**k:(chosen,np.zeros((2,7))))
    sim=SimpleNamespace(folder=tmp_path)
    sequence.mechanism_select(None,None,{'joint_family':'revolute'},None,None,None,None,sim,tmp_path,{})
    assert not hasattr(sim,'prepared_sequence')
    assert (tmp_path/'prepared-sequence-skipped.json').exists()
