import json
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.graspgen_selection import learned_poses, select_direct_grasp


def proposals():
    poses=np.repeat(np.eye(4)[None],3,axis=0)
    poses[:,:3,3]=[[.1,.2,.3],[.2,.3,.4],[.3,.4,.5]]
    return {'poses_world_base':poses,'scores':np.array([.2,.9,.5]),'retained_indices':np.array([1,2,0])}


def calibration():
    tcp=np.eye(4);tcp[2,3]=.097
    return {'physical_finger_attach_z_m':.0524,'graspgen_base_to_tcp':tcp}


def test_learned_positions_are_preserved_with_one_fixed_asset_conversion():
    generated=proposals();saved=generated['poses_world_base'].copy()
    rows=list(learned_poses(generated,calibration(),2))
    assert [r['graspgen_index'] for r in rows]==[1,2]
    for row,index in zip(rows,[1,2]):
        np.testing.assert_allclose(row['pose'][:3],saved[index,:3,3]+[0,0,.103])
        assert row['contact_depth_offset_m']==0
    np.testing.assert_array_equal(generated['poses_world_base'],saved)


def test_first_executable_pose_stops_search_without_variants(monkeypatch,tmp_path):
    import vlm_tamp.graspgen_selection as module
    import vlm_tamp.hinged_contact_tools as shared
    calls=[]
    class Context:
        def __init__(self,*args,**kw):self.record={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):return {'accepted':True}
        def plan(self,a,b,**kw):
            calls.append('path');return np.array([a,b]),{'exact_solution':True,'status':'solved'}
    class IK:
        chain=None
        def solve(self,pose,initial):
            calls.append(pose.copy())
            return {'joints':np.full(7,.1),'pose_world_xyzw':pose}
    monkeypatch.setattr(module,'NativeScene',Context)
    monkeypatch.setattr(shared,'virtual_metadata',lambda *args:{})
    state={'metadata':{'joint_positions':np.zeros(7),'robot_calibration':calibration()}}
    scene=SimpleNamespace(robot=None)
    budget={'maximum_joint_speed':.6,'endpoint_hold_steps':6,'available_approach_steps':500,
        'grasp_closure_steps_reserved':40,'remaining_control_steps':540}
    result=select_direct_grasp(proposals(),state,IK(),{},scene,scene,.2,tmp_path/'selection',{},budget)
    assert len(calls)==2 and calls[1]=='path'
    assert result['graspgen_index']==1
    assert result['search_summary']['generated_ik_candidates']==1
    assert not result['search_summary']['geometric_fallback_used']


def test_sampling_counts_are_part_of_cache_identity(tmp_path):
    from vlm_tamp.coupled_resources import CoupledResources
    resources=CoupledResources(tmp_path/'cache');resources.pins={'checkpoint':'fixed'}
    generated=[]
    class Model:
        def generate(self,cloud,output,**kwargs):
            generated.append(kwargs);output.mkdir()
            result=proposals()
            np.savez(output/'grasp_candidates.npz',**result)
            return result
    resources.graspgen=Model();cloud=np.ones((30,3),np.float32)
    resources.grasps(cloud,tmp_path/'one',num_grasps=8,topk=8)
    resources.grasps(cloud,tmp_path/'two',num_grasps=8,topk=8)
    resources.grasps(cloud,tmp_path/'three',num_grasps=4,topk=4)
    assert len(generated)==2
    assert json.loads((tmp_path/'two/generation-timing.json').read_text())['cache_reused']
    with pytest.raises(ValueError):resources.grasps(cloud,tmp_path/'bad',num_grasps=4,topk=8)
