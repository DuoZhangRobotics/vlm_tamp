"""Full-arc first, explicit continuation fallback, and unmodified planner output."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from vlm_tamp.curobo_v2_arc import CuroboV2ArcProposer
from vlm_tamp.partial_pose_arc import full_arc_plan


def planner(tmp_path, fail_full=False, fail_second=False):
    goal=np.radians(6.);calls=[]
    def forbidden(*a,**kw):
        raise AssertionError('Do not recheck a native planner path')
    context=SimpleNamespace(reference_q=np.zeros(7),grasp_constraint='point',
        scene=SimpleNamespace(config={}),record={},progress=lambda q:q[0],
        validate=forbidden,constraint_errors=forbidden)
    def connect(start,target,**kwargs):
        calls.append(('connect',start.copy(),target.copy()))
        return np.vstack([start,target]),{'exact_solution':True,'status':'solved','total_seconds':.001}
    context.plan=connect
    proposer=object.__new__(CuroboV2ArcProposer)
    proposer.context,proposer.folder,proposer.goal=context,tmp_path,goal
    proposer.solver_mode,proposer.arc_id,proposer.steps='global',1,2
    proposer.step_degrees,proposer.local_dt,proposer.speed_degrees_per_second=3.,.2,15.
    proposer.continuation_fallback_enabled=True;proposer.configure_request={}
    proposer.ik=SimpleNamespace(client=SimpleNamespace(request=lambda *a,**kw:{'arc_id':2}))
    def candidates(seed,angle,**kwargs):
        calls.append(('IK',seed.copy(),angle,proposer.solver_mode))
        if fail_full and proposer.solver_mode=='global':return [],[{'accepted':False,'reason':'curobo_ik_failed'}]
        if fail_second and angle>np.radians(4.):return [],[{'accepted':False,'reason':'curobo_ik_failed'}]
        q=seed.copy();q[0]=angle
        return [q],[{'accepted':True}]
    proposer.candidates=candidates;proposer.close=lambda:calls.append(('close',))
    return context,proposer,goal,calls


def test_full_requested_arc_is_the_first_and_only_query_when_solved(tmp_path):
    context,proposer,goal,calls=planner(tmp_path)
    path,result=full_arc_plan(context,goal,seconds=10.,goal_proposer=proposer)
    queries=[r for r in calls if r[0]=='IK'];connections=[r for r in calls if r[0]=='connect']
    assert len(queries)==len(connections)==1
    assert queries[0][2]==goal and queries[0][3]=='global'
    np.testing.assert_array_equal(path[:,0],[0.,goal])
    assert result['solved_by']=='full_arc' and not result['continuation_fallback']['attempted']
    assert not result['additional_path_checks'] and calls[-1]==('close',)


def test_continuation_only_follows_failed_full_arc_and_uses_previous_posture(tmp_path):
    context,proposer,goal,calls=planner(tmp_path,fail_full=True)
    path,result=full_arc_plan(context,goal,seconds=10.,goal_proposer=proposer)
    queries=[r for r in calls if r[0]=='IK']
    assert [r[2] for r in queries]==[goal,goal/2,goal]
    assert queries[1][1][0]==0. and queries[2][1][0]==goal/2
    assert result['solved_by']=='continuation_fallback'
    assert result['continuation_fallback']['attempted']
    assert result['full_arc_attempt']['status']=='full_arc_no_ik_goal'
    assert path[-1,0]==goal


def test_failed_fallback_returns_no_partial_motion(tmp_path):
    context,proposer,goal,calls=planner(tmp_path,fail_full=True,fail_second=True)
    path,result=full_arc_plan(context,goal,seconds=10.,goal_proposer=proposer)
    assert path.shape==(0,7) and not result['exact_solution']
    assert result['failure']['reached_angle_deg']==pytest.approx(3.)
    assert result['failure']['motion_executed'] is False


def test_exhausted_budget_does_not_start_another_gpu_query(tmp_path):
    context,proposer,goal,calls=planner(tmp_path)
    path,result=full_arc_plan(context,goal,seconds=0.,goal_proposer=proposer)
    assert len(path)==0 and result['status']=='search_budget_exhausted'
    assert calls==[('close',)]


def test_worker_scope_closes_when_candidate_generation_raises(tmp_path):
    context,proposer,goal,calls=planner(tmp_path)
    def fail(*a,**kw):raise RuntimeError('numerical worker error')
    proposer.candidates=fail
    with pytest.raises(RuntimeError,match='numerical worker error'):
        full_arc_plan(context,goal,seconds=10.,goal_proposer=proposer)
    assert calls[-1]==('close',)


def test_native_returned_path_is_kept_without_projection_or_rechecking(tmp_path):
    context,proposer,goal,calls=planner(tmp_path)
    supplied=np.array([[.001,0,0,0,0,0,0],[goal,0,0,0,0,0,0]])
    context.plan=lambda *a,**kw:(supplied,{'exact_solution':True,'status':'solved'})
    path,result=full_arc_plan(context,goal,seconds=10.,goal_proposer=proposer)
    assert path is supplied and result['exact_solution']


def test_constructor_closes_configured_arc_if_frame_query_fails(tmp_path):
    calls = []
    def request(operation, **kwargs):
        calls.append(operation)
        if operation == 'configure_arc':
            assert kwargs['position_tolerance'] == .001
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




def test_gpu_endpoint_is_not_resolved_clipped_or_rescued(tmp_path):
    """A valid GPU endpoint reaches admission verbatim; failure stays failure."""
    examined = []
    def forbidden(*args, **kwargs):
        raise AssertionError('The cuRobo handoff must not run a second IK/projection')
    def validate(q):
        examined.append(q.copy())
        return {'accepted': True}
    chain = SimpleNamespace(forward=lambda q: [0., 0., 0., 0., 0., 0., 1.])
    proposer = object.__new__(CuroboV2ArcProposer)
    proposer.context = SimpleNamespace(axis=np.array([0., 0., 1.]), chain=chain,
        reference_tcp=np.eye(4), scene=SimpleNamespace(config={}),
        project=forbidden, validate=validate, progress=lambda q: .105)
    proposer.model = SimpleNamespace(reference_rotation=np.eye(3), grasp_constraint='point',
        point=lambda a: np.zeros(3), project=forbidden,
        residual=forbidden,
        position_tolerance_m=.006, angular_weighted_tolerance=1e-6)
    proposer.folder, proposer.closed, proposer.arc_id = tmp_path, False, 1
    proposer.goal, proposer.queries, proposer.ik_position_tolerance_m = .1, [], .001
    q = np.array([.1, -.2, .3, -.4, .5, -.6, .7])
    gpu = {'solutions': [q.tolist(), (q+.2).tolist()], 'success': [True, False],
        'elapsed_seconds': .1, 'solver_seconds': .1}
    goals, checks = proposer.candidates(np.zeros(7), .1, gpu_result=gpu)
    np.testing.assert_array_equal(goals[0], q)
    assert examined == []  # Only the native planner owns endpoint collision checking.
    assert len(goals) == 1
    assert checks[0]['candidate_modified'] is False
    # Point accuracy owns the goal tolerance; the old independent .002-rad
    # progress veto would have rejected this otherwise valid endpoint.
    assert checks[0]['progress_error_rad'] == pytest.approx(.005)
    assert checks[1]['reason'] == 'curobo_ik_failed'
    assert all(row['equality_polish_seconds'] == 0. for row in checks)
