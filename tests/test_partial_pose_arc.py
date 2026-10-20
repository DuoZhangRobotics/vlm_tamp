from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.arc_contact_scene import reconcile_static
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.partial_pose_arc import PartialPoseArc,PalmClearanceObjective
from vlm_tamp.hinge_uncertainty import rotate_cloud,hinge_sensitivity
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/mcvamp-convex-frozen-grasp-state11-v2-20261015/open_microwave-state011'


def test_parallel_hinge_offset_causes_expected_prediction_error():
    points=np.array([[.25,0,1.],[0,.3,1.1]])
    nominal=rotate_cloud(points,[0,0,1],[0,0,1],np.pi/2)
    perturbed=rotate_cloud(points,[.01,0,1],[0,0,1],np.pi/2)
    assert np.allclose(np.linalg.norm(perturbed-nominal,axis=1),np.sqrt(2)*.01)
    result=hinge_sensitivity(points,[0,0,1],[0,0,1],[90.])
    assert not result['uncertainty_calibrated_from_data']
    assert not result['execution_veto']


def test_role_reconciliation_needs_current_moving_support_and_handle_membership():
    previous=np.array([[0,0,1],[.002,0,1],[.006,0,1],[.05,0,1],[0,.002,1]])
    moving=np.array([[0,0,1]])
    def selected(points):return points[:,1]==0
    static,removed=reconcile_static(previous,[[.1,0,1]],moving,[[0,0,1]],selected)
    assert np.array_equal(removed,previous[:2])
    assert len(static)==4
    assert any(np.array_equal(point,[.1,0,1]) for point in static)
    # Merely being close to the gripper or handle cannot erase a point whose
    # current moving association or depth correspondence is absent.
    assert any(np.array_equal(point,[.006,0,1]) for point in static)
    assert any(np.array_equal(point,[0,.002,1]) for point in static)


def test_partial_pose_projection_preserves_orientation_freedom():
    if not SOURCE.exists():pytest.skip('Retained Panda calibration required')
    inputs=read_json(SOURCE/'plan-00/horizon-1/planning-input.json')
    chain=RobotChain(read_json(SOURCE/'robot-chain.json'));q=np.asarray(inputs['reference_joints'])
    tcp=pose_matrix(chain.forward(q));meta=inputs['metadata']
    inverse=np.linalg.inv(tcp)
    pads=[np.asarray(v['vertices_world']).mean(0) for v in meta['robot_collision_volumes_world']
          if v['robot_geom_name'].endswith('_pad_collision')]
    pad=(inverse[:3,:3]@np.mean(pads,axis=0)+inverse[:3,3])
    model=PartialPoseArc(chain,inputs['task_spec']['joint_origin_world_m'],inputs['task_spec']['joint_axis_world'],q,pad)
    N=model.nullspace(q,.05);rng=np.random.default_rng(0);rotations=[]
    for _ in range(12):
        solved,info=model.project(q+N@rng.normal(0,.18,7),.05)
        if not info['accepted']:continue
        matrix=pose_matrix(chain.forward(solved))
        point=matrix[:3,3]+matrix[:3,:3]@pad
        assert np.linalg.norm(point-model.point(.05))<2e-5
        rotations.append(matrix[:3,:3])
    assert len(rotations)>=2
    assert max(Rotation.from_matrix(rotations[0].T@r).magnitude() for r in rotations)>.05


def test_palm_clearance_objective_gradient_matches_cost_difference():
    if not SOURCE.exists():pytest.skip('Retained Panda calibration required')
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from diagnose_partial_pose_arc import load_scene
    from vlm_tamp.mcvamp_microwave import NativeScene,clear_native_cache
    scene,record,meta=load_scene(SOURCE);q=np.asarray(meta['joint_positions'])
    with NativeScene(scene,record,meta,q,np.pi/2,aligned=False) as context:
        objective=PalmClearanceObjective(context,margin=.005,arm=False)
        residual,J=objective.evaluate(q,0.)
        assert np.linalg.norm(residual)>0
        analytic=J.T@residual;numeric=[]
        for j in range(7):
            step=np.zeros(7);step[j]=1e-6
            a=objective.evaluate(q+step,0.)[0];b=objective.evaluate(q-step,0.)[0]
            numeric.append((a@a-b@b)/(4e-6))
        assert np.allclose(analytic,numeric,atol=2e-6,rtol=.02)
    clear_native_cache()
