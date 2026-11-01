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
    return json.loads((ROOT/'tests/fixtures/calibrated_panda.json').read_text())


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


@pytest.mark.parametrize('margin', [-.1, .3])
def test_invalid_reserve_rejected(tmp_path, margin):
    with pytest.raises(ValueError):
        export_model(record(), tmp_path, margin)


def fake_solver():
    solver=object.__new__(TaskIK)
    solver.chain=RobotChain(record());solver.reserve=0.;solver.events=[]
    solver.config={'ik_position_tolerance_m':.001,'ik_orientation_tolerance_rad':.01}
    solver.self_valid=lambda q: True
    return solver


def test_failed_optimizer_flag_is_not_rescued_by_cpu_fk():
    solver=fake_solver();q=solver.chain.reference_joints.copy()
    solver.client=SimpleNamespace(solve=lambda *a:{'solutions':[q.tolist()],'success':[False],
        'elapsed_seconds':.01,'peak_gpu_allocated_bytes':0})
    assert solver.solve_many(solver.chain.forward(q),q)==[]
    assert solver.events[-1]['admitted_failed_flag_candidates']==0


def test_solver_successes_are_unchanged_and_not_cpu_filtered():
    solver=fake_solver();q=solver.chain.reference_joints.copy()
    rounded=q.copy();rounded[5]=float(np.float32(solver.chain.limits[5,1]))
    nearby=q.copy();nearby[2]+=.001
    solver.self_valid=lambda value: (_ for _ in ()).throw(AssertionError('Native planner owns collision admission'))
    solver.client=SimpleNamespace(solve=lambda *a:{
        'solutions':[rounded.tolist(),nearby.tolist(),q.tolist(),q.tolist()],
        'success':[True]*4,'elapsed_seconds':.01,'peak_gpu_allocated_bytes':0})
    candidates=solver.solve_many(solver.chain.forward(q),q)
    assert len(candidates)==3  # Only the exact duplicate is removed.
    for proposal in (q,nearby,rounded):
        assert any(np.array_equal(row['joints'],proposal) for row in candidates)
    assert all(not row['candidate_modified'] for row in candidates)
    assert candidates[0]['joints']==q.tolist()


def test_point_compatibility_entry_uses_curobo_and_preserves_its_failure():
    solver=fake_solver();q=solver.chain.reference_joints.copy();calls=[]
    def request(operation, **kw):
        calls.append((operation,kw))
        return {'solutions':[q.tolist()],'success':[False],'elapsed_seconds':.01,'backend':'curobo_v2'}
    solver.client=SimpleNamespace(request=request)
    assert solver.solve_point([.1,.2,.3],[0.,0.,.01],q) is None
    assert calls[0][0]=='solve_contact' and calls[0][1]['local'] is True
    assert posture_cost(q,q,solver.chain.limits)==0.
    at_stop=solver.chain.limits[:,1]
    assert posture_cost(at_stop,at_stop,solver.chain.limits)==0.


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
