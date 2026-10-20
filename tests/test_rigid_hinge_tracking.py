import numpy as np
from vlm_tamp.rigid_hinge_tracking import estimate_rigid_hinge,measured_pose
from vlm_tamp.arc_contact_scene import observed_motion_roles


def test_rigid_motion_recovers_hinge_with_stationary_and_bad_matches():
    rng=np.random.default_rng(3);hinge=np.array([.18,-.16]);theta=-.09
    a=np.c_[rng.uniform([-.08,-.13],[.15,-.09],(50,2)),rng.uniform(.95,1.15,50)]
    R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    b=a.copy();b[:,:2]=(a[:,:2]-hinge)@R.T+hinge
    b+=rng.normal(0,.00012,b.shape)
    b[35:45]=a[35:45];b[45:]+=rng.normal(0,.03,(5,3))
    result=estimate_rigid_hinge(a,b,prior=[.14,-.14,1.])
    assert result['accepted'],result
    assert np.linalg.norm(np.asarray(result['hinge_world_m'])[:2]-hinge)<.004


def test_single_vertical_bar_does_not_observe_rotation():
    a=np.c_[np.zeros(20),np.zeros(20),np.linspace(0,.15,20)]
    b=a+[.01,.002,0]
    result=estimate_rigid_hinge(a,b)
    assert not result['accepted']
    assert measured_pose(result) is None


def test_short_measured_motion_transports_handle_without_claiming_stable_pivot():
    rng=np.random.default_rng(9);hinge=np.array([.18,-.16]);theta=-np.radians(1.)
    a=np.c_[rng.uniform([-.08,-.13],[.10,-.09],(40,2)),rng.uniform(.95,1.15,40)]
    R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    b=a.copy();b[:,:2]=(a[:,:2]-hinge)@R.T+hinge
    result=estimate_rigid_hinge(a,b)
    assert result['pose_accepted'] and not result['accepted']
    transform=measured_pose(result)
    assert np.max(np.linalg.norm(a@transform[:3,:3].T+transform[:3,3]-b,axis=1))<1e-8


def test_observed_motion_removes_ghost_but_preserves_stationary_evidence():
    transform=np.eye(4);transform[0,3]=.02
    old_static=np.array([[0,0,1],[0,.004,1],[.10,0,1]])
    moving=np.array([[.02,0,1],[.02,.004,1]])
    current_static=np.array([[0,.004,1],[.10,0,1]])
    prior,current,updated,record=observed_motion_roles(old_static,[[0,0,1]],[[0,0,1]],
        current_static,moving,[[.02,0,1]],transform)
    assert record['historical_moving_ghosts_removed']==1
    assert any(np.array_equal(p,[0,.004,1]) for p in prior)
    assert any(np.array_equal(p,[.10,0,1]) for p in current)


def test_failed_two_degree_wrist_sweep_is_rejected_during_planning():
    from pathlib import Path
    import pytest
    from vlm_tamp.coupled_storage import read_json
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
    from vlm_tamp.mcvamp_microwave import NativeScene,clear_native_cache
    from vlm_tamp.partial_pose_arc import path_wrist_admitted
    root=Path(__file__).resolve().parents[1]
    case=root/'results/short-planned-hinge-state11-20261015/open_microwave-state011'
    if not case.exists():pytest.skip('Saved failure fixture required')
    folder=case/'plan-02/horizon-0';inputs=read_json(folder/'planning-input.json')
    arrays=np.load(read_json(folder/'scene-reference.json')['arrays_path'])
    scene=Scene.from_points(arrays['static_points'],arrays['moving_reference'],arrays['contact_reference'],
        inputs['task_spec'],config={'motion_policy':'short'})
    path=np.load(case/'plan-02/path.npy')
    with NativeScene(scene,read_json(case/'robot-chain.json'),inputs['metadata'],path[0],np.radians(2),aligned=False) as context:
        admitted,info=path_wrist_admitted(context,path,np.radians(2))
        assert not admitted and info['maximum_wrist_excursion_deg']>30
        assert info['planning_wrist_limit_deg']==8.
        assert info['execution_veto'] is False
    clear_native_cache()
