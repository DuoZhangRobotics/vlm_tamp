"""Exercise the actual cuRobo LM continuation with sampled optimization forbidden."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('curobo')
pytest.importorskip('vamp')
if not torch.cuda.is_available():
    pytest.skip('Native cuRobo continuation requires CUDA', allow_module_level=True)

from vlm_tamp.rekep_libero.curobo_collision_model import calibrated_spheres, native_self_collision_pairs
from vlm_tamp.rekep_libero.curobo_model import export_model
from vlm_tamp.rekep_libero.curobo_v2_worker import Worker
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from test_vamp_sphere_policy import robot as calibrated_robot_fixture, scene_for

calibrated_robot = calibrated_robot_fixture


@pytest.fixture
def local_worker(tmp_path, monkeypatch):
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'tests/fixtures/calibrated_panda.json').read_text())
    chain=RobotChain(record)
    world=np.eye(4);world[:3,3]=[-.66,0.,.912]
    model=export_model(record,tmp_path/'robot')
    spheres,provenance=calibrated_spheres(chain,None,world,include_self=True)
    model['kinematics'].update(collision_link_names=list(spheres),collision_spheres=spheres,
        collision_sphere_buffer=0.,self_collision_ignore={},self_collision_buffer={})
    model['native_sphere_indices']=[i for group in provenance['groups'] for i in group['native_indices']]
    model['native_self_collision_pairs']=native_self_collision_pairs()
    worker=Worker()
    worker.initialize(model,{'seed':0,'curobo_num_seeds':1,'curobo_return_seeds':1,
        'ik_position_tolerance_m':.003,'ik_orientation_tolerance_rad':.01})
    create=worker.create_solver
    def create_without_sampled_optimizer(*args,**kwargs):
        solver=create(*args,**kwargs)
        def forbidden(*args,**kwargs):
            pytest.fail('Continuation invoked the main sampled optimizer')
        monkeypatch.setattr(solver.optimizer,'optimize',forbidden)
        return solver
    monkeypatch.setattr(worker,'create_solver',create_without_sampled_optimizer)
    yield worker,chain
    for value in getattr(worker,'contact_solvers',{}).values():value['solver'].destroy()
    worker.solver.destroy()


@pytest.mark.parametrize('oriented',[False,True])
def test_native_small_cartesian_steps_stay_near_previous_posture(local_worker,oriented):
    worker,chain=local_worker
    previous=chain.reference_joints.copy();pose=chain.forward(previous);matrix=pose_matrix(pose)
    offset=np.array([.0,.02,.055]);start=matrix[:3,3]+matrix[:3,:3]@offset
    normal=np.array([0.,0.,1.]);surface=-matrix[:3,:3]@normal
    optional={'contact_normal_local':normal,'surface_normal_world':surface} if oriented else {}
    for index in range(1,8):
        target=start+matrix[:3,0]*(.003*index)
        result=worker.solve_contact(target,offset,pose[3:],previous,local=True,
            environment_collision=False,**optional)
        assert result['success']==[True],result
        assert result['local_solver_success']==[True]
        assert not result['sampled_optimizer_used'] and not result['continuation_fallback_used']
        assert not result['scene_collision_checked'] and result['self_collision_checked']
        actual=np.asarray(result['solutions'][0]);actual_pose=pose_matrix(chain.forward(actual))
        assert np.linalg.norm(actual_pose[:3,3]+actual_pose[:3,:3]@offset-target)<.003
        # A test expectation for this nonsingular fixture, not a runtime veto.
        assert np.max(np.abs(actual-previous))<.1
        if oriented:
            angle=np.arccos(np.clip((actual_pose[:3,:3]@normal)@(-surface),-1.,1.))
            assert angle<.01
        if index>1:assert not result['solver_created']
        previous=actual


def test_unreachable_native_local_goal_returns_failure_without_a_retry(local_worker):
    worker,chain=local_worker
    previous=chain.reference_joints;pose=chain.forward(previous)
    result=worker.solve_contact([20.,20.,20.],[0.,0.,0.],pose[3:],previous,
        local=True,environment_collision=False)
    assert result['success']==[False]
    assert not result['sampled_optimizer_used'] and not result['continuation_fallback_used']


def test_measured_finger_model_matches_native_and_gpu_across_arm_poses(local_worker,calibrated_robot,tmp_path):
    """Both backends place the same finger spheres as the arm configuration changes."""
    import hashlib
    from vlm_tamp.vamp_sphere_model import finger_environment_model,native_spheres
    from vlm_tamp.mcvamp_contact_numerical import NativeScene
    worker,chain=local_worker
    record,metadata,_,_,world=calibrated_robot
    model=finger_environment_model(metadata,world)
    q=chain.reference_joints.copy()
    for increment in (0.,.12,-.08):
        q[0]=chain.reference_joints[0]+increment
        spheres=native_spheres(q);base,first,second=spheres[[37,40,45],:3]
        u,v=first-base,second-base;a,b,c,r=model['coefficients'][1]
        point=world[:3,:3]@(base+a*u+b*v+c*np.cross(u,v))+world[:3,3]
        scene=scene_for(point,'stationary')
        for checked in (True,False):
            with NativeScene(scene,record,metadata,metadata['joint_positions'],.1,
                             articulated=False,check_fingers=checked) as native:
                path=tmp_path/'observed.npz';np.savez(path,**native.ik_clouds_world)
                worker.configure_scene(str(path),hashlib.sha256(path.read_bytes()).hexdigest(),
                                       finger_model=native.finger_model)
                result=worker.collision_diagnostics(q[None])
                cost=float(np.max(result['environment_cost_per_sphere']))
                assert cost==pytest.approx(r if checked else 0.,abs=3e-6)
                assert native.validate(q)['accepted']==(not checked)
