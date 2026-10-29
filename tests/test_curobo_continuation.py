"""Full-arc ownership and admission regressions, without GPU or simulation."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from vlm_tamp.curobo_v2_arc import CuroboV2ArcProposer
from vlm_tamp.partial_pose_arc import full_arc_plan


def planner(tmp_path, monkeypatch, fail_second=False, reject_edge=False):
    import vlm_tamp.curobo_v2_arc as module
    goal = np.radians(6.)
    calls = []
    context = SimpleNamespace(reference_q=np.zeros(7), grasp_constraint='point',
        scene=SimpleNamespace(config={}), record={}, progress=lambda q: q[0],
        validate=lambda q: {'accepted': True},
        constraint_errors=lambda q: {'accepted': not (reject_edge and q[0] > np.radians(4.))})

    def connect(start, target, **kwargs):
        calls.append(('connect', start.copy(), target.copy()))
        return np.vstack([start, target]), {'exact_solution': True, 'status': 'solved', 'total_seconds': .001}

    def wrist(ctx, path, requested_goal):
        assert ctx is context and requested_goal == goal
        return True, {'maximum_wrist_excursion_deg': 0.}

    context.plan = connect
    monkeypatch.setattr(module, 'path_wrist_admitted', wrist)
    proposer = object.__new__(CuroboV2ArcProposer)
    proposer.context, proposer.folder, proposer.goal = context, tmp_path, goal
    proposer.solver_mode, proposer.arc_id, proposer.steps = 'local_continuation', 1, 2
    proposer.step_degrees, proposer.local_dt, proposer.speed_degrees_per_second = 3., .2, 15.

    def candidates(seed, angle, **kwargs):
        calls.append(('IK', seed.copy(), angle))
        if fail_second and angle > np.radians(4.):
            return [], [{'accepted': False, 'reason': 'scene_collision'}]
        q = seed.copy(); q[0] = angle
        return [q], [{'accepted': True}]

    proposer.candidates = candidates
    proposer.close = lambda: calls.append(('close',))
    return context, proposer, goal, calls


def test_local_sequence_uses_previous_posture_and_returns_only_complete_arc(tmp_path, monkeypatch):
    context, proposer, goal, calls = planner(tmp_path, monkeypatch)
    path, result = full_arc_plan(context, goal, seconds=10., goal_proposer=proposer)
    assert result['exact_solution'] and path[-1, 0] == goal
    requests = [r for r in calls if r[0] == 'IK']
    assert requests[0][1][0] == 0.
    assert requests[1][1][0] == requests[0][2]
    assert calls[-1] == ('close',)
    assert result['execution_during_search'] is False
    assert result['arc_anchor_reset_between_steps'] is False


@pytest.mark.parametrize('failure', ['no_goal', 'invalid_edge'])
def test_partial_progress_is_never_returned_for_execution(tmp_path, monkeypatch, failure):
    context, proposer, goal, calls = planner(tmp_path, monkeypatch,
        fail_second=failure == 'no_goal', reject_edge=failure == 'invalid_edge')
    path, result = full_arc_plan(context, goal, seconds=10., goal_proposer=proposer)
    assert path.shape == (0, 7)
    assert not result['exact_solution']
    assert result['failure']['reached_angle_deg'] == pytest.approx(3.)
    assert result['failure']['attempted_angle_deg'] == pytest.approx(6.)
    assert result['failure']['motion_executed'] is False
    assert calls[-1] == ('close',)


def test_exhausted_budget_does_not_start_another_gpu_query(tmp_path, monkeypatch):
    context, proposer, goal, calls = planner(tmp_path, monkeypatch)
    path, result = full_arc_plan(context, goal, seconds=0., goal_proposer=proposer)
    assert len(path) == 0 and result['status'] == 'search_budget_exhausted'
    assert calls == [('close',)]


def test_worker_scope_closes_when_candidate_generation_raises(tmp_path, monkeypatch):
    context, proposer, goal, calls = planner(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError('numerical worker error')
    proposer.candidates = fail
    with pytest.raises(RuntimeError, match='numerical worker error'):
        full_arc_plan(context, goal, seconds=10., goal_proposer=proposer)
    assert calls[-1] == ('close',)


def test_path_starts_with_native_admitted_projection_not_raw_measurement(tmp_path, monkeypatch):
    context, proposer, goal, calls = planner(tmp_path, monkeypatch)
    def connect(start, target, **kwargs):
        projected = start.copy()
        if start[0] == 0.:
            projected[0] = .001
        return np.vstack([projected, target]), {'exact_solution': True, 'status': 'solved'}
    context.plan = connect
    context.constraint_errors = lambda q: {'accepted': q[0] > 0.}
    path, result = full_arc_plan(context, goal, seconds=10., goal_proposer=proposer)
    assert result['exact_solution']
    assert path[0, 0] == .001
    assert result['start_projection_max_joint_delta_rad'] == .001
    assert all(context.constraint_errors(q)['accepted'] for q in path)


def test_constructor_closes_configured_arc_if_frame_query_fails(tmp_path):
    calls = []
    def request(operation, **kwargs):
        calls.append(operation)
        if operation == 'configure_arc':
            return {'arc_id': 4}
        if operation == 'arc_forward':
            raise RuntimeError('frame query failed')
        return {'arc_ended': True}
    q = np.zeros(7)
    chain = SimpleNamespace(reference_joints=q, limits=np.tile([-3., 3.], (7, 1)),
        forward=lambda q: [0., 0., 1., 0., 0., 0., 1.])
    context = SimpleNamespace(chain=chain, origin=np.zeros(3), axis=np.array([0., 0., 1.]),
        reference_q=q, pad_local=np.zeros(3), grasp_constraint='point',
        scene=SimpleNamespace(config={}))
    ik = SimpleNamespace(client=SimpleNamespace(backend='v2', request=request),
        _v2_arc_spheres=({}, {'sphere_count': 0, 'groups': []}),_v2_arc_aperture=())
    (tmp_path/'scene-reference.json').write_text(json.dumps({'arrays_path': 'unused', 'arrays_sha256': 'unused'}))
    with pytest.raises(RuntimeError, match='frame query failed'):
        CuroboV2ArcProposer(ik, context, tmp_path, .1)
    assert calls == ['configure_arc', 'arc_forward', 'end_arc']


@pytest.mark.parametrize('reject_global_path', [False, True])
def test_fallback_replans_whole_arc_and_requires_native_admission(tmp_path, monkeypatch, reject_global_path):
    context, proposer, goal, calls = planner(tmp_path, monkeypatch, fail_second=True)
    local_candidates = proposer.candidates
    def candidates(seed, angle, **kwargs):
        if seed[0] == 0. and angle == goal:
            q = seed.copy(); q[0] = angle
            return [q], [{'accepted': True}]
        return local_candidates(seed, angle, **kwargs)
    proposer.candidates = candidates
    proposer.global_fallback_enabled = True
    proposer.configure_request = {}
    proposer.ik = SimpleNamespace(client=SimpleNamespace(request=lambda *args, **kwargs: {'arc_id': 2}))
    if reject_global_path:
        context.constraint_errors = lambda q: {'accepted': q[0] < goal}
    path, result = full_arc_plan(context, goal, seconds=10., goal_proposer=proposer)
    assert result['global_fallback']['attempted']
    assert result['local_failure']['reached_angle_deg'] == pytest.approx(3.)
    assert sum(c[0] == 'close' for c in calls) == 2
    if reject_global_path:
        assert len(path) == 0 and not result['exact_solution']
        assert result['global_queries'][0]['status'] == 'complete_path_admission_failed'
    else:
        assert result['exact_solution'] and result['global_fallback']['solved']
        np.testing.assert_allclose(path[:, 0], [0., goal])
