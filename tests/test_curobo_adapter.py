"""Robot-frame, joint-limit, and alternative-posture admission regressions."""

import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.rekep_libero.curobo_model import export_model
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK, joint_reserve, posture_cost

ROOT = Path(__file__).resolve().parents[1]


def record():
    # Robot-only calibration from a preserved episode, no object geometry.
    return json.loads((ROOT/'results/physical-tasks-v11-eval/goal00-state011/robot-chain.json').read_text())


def urdf_forward(path, q):
    matrix = np.eye(4)
    for joint in ET.parse(path).getroot().findall('joint'):
        origin = joint.find('origin')
        step = np.eye(4)
        step[:3, 3] = np.fromstring(origin.attrib['xyz'], sep=' ')
        step[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.attrib['rpy'], sep=' ')).as_matrix()
        matrix = matrix@step
        if joint.attrib['type'] == 'revolute':
            index = int(joint.attrib['name'].replace('joint', ''))-1
            change = np.eye(4)
            change[:3, :3] = Rotation.from_rotvec([0, 0, q[index]]).as_matrix()
            matrix = matrix@change
    return matrix


def test_exported_urdf_matches_calibrated_chain_across_seven_joints(tmp_path):
    robot_record = record()
    chain = RobotChain(robot_record)
    export_model(robot_record, tmp_path, .025)
    low, high = chain.limits.T
    for q in [chain.reference_joints, *np.random.default_rng(4).uniform(low+.03, high-.03, (50, 7))]:
        assert np.allclose(urdf_forward(tmp_path/'calibrated-panda.urdf', q),
                           pose_matrix(chain.forward(q)), atol=2e-12)
    for index, joint in enumerate(ET.parse(tmp_path/'calibrated-panda.urdf').getroot().findall('joint')[:7]):
        assert float(joint.find('limit').attrib['lower']) == pytest.approx(low[index]+.025)
        assert float(joint.find('limit').attrib['upper']) == pytest.approx(high[index]-.025)


@pytest.mark.parametrize('margin', [0, -.1, .3])
def test_invalid_reserve_rejected(tmp_path, margin):
    with pytest.raises(ValueError):
        export_model(record(), tmp_path, margin)


def test_ik_proposals_filtered_by_exact_fk_reserve_and_self_collision():
    solver = object.__new__(TaskIK)
    solver.chain = RobotChain(record())
    q = solver.chain.reference_joints.copy()
    solver.reserve, solver.events = .025, []
    solver.config = {'ik_position_tolerance_m': .001, 'ik_orientation_tolerance_rad': .01}
    invalid = q.copy()
    invalid[5] = solver.chain.limits[5, 1]-.0001
    wrong_pose = q.copy()
    wrong_pose[2] += .3
    solver.self_valid = lambda value: True
    solver.client = SimpleNamespace(solve=lambda pose, seed: {
        'solutions': [invalid.tolist(), wrong_pose.tolist(), q.tolist(), q.tolist()],
        'success': [True]*4, 'elapsed_seconds': .01, 'peak_gpu_allocated_bytes': 10})
    candidates = solver.solve_many(solver.chain.forward(q), q)
    assert len(candidates) == 1
    assert solver.events[-1]['rejected'] == {'solver': 0, 'bounds': 1, 'residual': 1, 'duplicate': 1, 'self_collision': 0}
    solver.self_valid = lambda value: False
    assert solver.solve_many(solver.chain.forward(q), q) == []


def test_posture_cost_penalizes_stops_and_point_ik_keeps_reserve():
    solver = object.__new__(TaskIK)
    solver.chain = RobotChain(record())
    solver.reserve = .025
    solver.config = {'ik_position_tolerance_m': .001}
    solver.self_valid = lambda q: True
    q = solver.chain.reference_joints.copy()
    local = np.array([.01, 0, -.01])
    pose = pose_matrix(solver.chain.forward(q))
    target = pose[:3, 3]+pose[:3, :3]@local+np.array([.01, .01, .01])
    solution = solver.solve_point(target, local, q)
    assert solution is not None
    assert joint_reserve(np.asarray(solution['joints']), solver.chain.limits) >= solver.reserve-1e-8
    risky = q.copy()
    risky[5] = solver.chain.limits[5, 1]-.0001
    assert posture_cost(risky, risky, solver.chain.limits) > posture_cost(q, q, solver.chain.limits)


def test_scene_rejection_of_one_ik_does_not_discard_same_grasp(monkeypatch, tmp_path):
    from vlm_tamp.rekep_libero import grasp_selection_curobo as selection
    pose = np.array([0., 0., 0., 0., 0., 0., 1.])
    monkeypatch.setattr(selection, 'candidate_variants', lambda *args: [(4, 'test', pose)])
    proposals = [{'joints': [0]*7}, {'joints': [1]*7}]
    tested = []

    def admission(grasp, *args):
        tested.append(grasp['joints'])
        if len(tested) == 1:
            return None, {'reason': 'full_arm_approach_collision'}
        return {'solution_joints': grasp['joints']}, None

    monkeypatch.setattr(selection, 'admit_posture', admission)
    ik = SimpleNamespace(config={'maximum_ik_postures_per_grasp': 16}, solve_many=lambda *args, **kwargs: proposals)
    saved = {}
    chosen = selection.choose_grasp({'scores': np.ones(5)}, np.zeros((3, 3)),
        {'source_anchor_world_m': [0, 0, 0], 'task_type': 'pick_place'}, {}, ik, [0]*7,
        tmp_path, None, None, [0, 0], lambda path, value: saved.update({path.name: value}))
    assert chosen['ik_alternative'] == 1
    assert chosen['index'] == 4
    assert chosen['tested_ik_postures'] == 2
    assert saved['grasp-admission.json']['contact_policy_changed_from_v11'] is False


def test_approach_ends_at_exact_posture_used_for_closure_admission():
    from vlm_tamp.rekep_libero.task_ik_v5 import TaskIK as CPUReferenceIK
    solver = object.__new__(TaskIK)
    solver.chain = RobotChain(record())
    solver.config = {'ik_seed_count': 2, 'ik_maximum_evaluations': 100,
                     'ik_position_tolerance_m': .001, 'ik_orientation_tolerance_rad': .01}
    solver.self_valid = lambda q: True
    solver.solve = lambda pose, seed, seed_count=None: CPUReferenceIK.solve(solver, pose, seed, seed_count)
    target = solver.chain.reference_joints.copy()
    initial = target.copy()
    initial[0] += .05
    grasp = {'joints': target.tolist(), 'pose_world_xyzw': solver.chain.forward(target).tolist()}
    result = solver.approach(initial, grasp)
    assert result['accepted']
    assert result['joint_postures'][-1] == grasp['joints']
    assert result['endpoint_matches_selected_grasp']
    for q, pose in zip(result['joint_postures'], result['poses_world_xyzw']):
        assert np.allclose(pose_matrix(solver.chain.forward(q)), pose_matrix(pose), atol=1e-10)
