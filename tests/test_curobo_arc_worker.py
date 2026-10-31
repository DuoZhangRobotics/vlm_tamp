"""CPU protocol checks for arc lifetime, geometry ownership, and local seeds."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from vlm_tamp.rekep_libero.curobo_v2_worker import Worker, normalize_ik_optimizer_config


def test_legacy_particle_costs_are_preserved_in_supported_schema():
    original={'rollout':{'cost_cfg':{
        'bound_cfg':{'cost_type':'TELEPORT','weight':[50.,1.]},
        'multi_link_pose_cfg':{'weight':[1000.,10.]}},
        'constraint_cfg':{'scene_collision_cfg':{'weight':500.}}}}
    result=normalize_ik_optimizer_config(original)
    assert result['rollout']['cost_cfg']=={
        'cspace_cfg':{'cost_type':'POSITION','weight':[50.,1.]},
        'tool_pose_cfg':{'weight':[1000.,10.]}}
    assert result['rollout']['constraint_cfg']==original['rollout']['constraint_cfg']
    assert 'bound_cfg' in original['rollout']['cost_cfg']
    assert normalize_ik_optimizer_config(result)==result


def test_conflicting_legacy_and_current_costs_are_not_silently_dropped():
    with pytest.raises(ValueError,match='Ambiguous'):
        normalize_ik_optimizer_config({'rollout':{'cost_cfg':{'bound_cfg':{},'cspace_cfg':{}}}})


@pytest.fixture
def worker_scene(monkeypatch, tmp_path):
    meshes, solvers = [], []

    class Mesh:
        @staticmethod
        def from_pointcloud(points, *, pitch, name):
            result = SimpleNamespace(points=points.copy(), pitch=pitch, name=name, faces=[0, 1, 2])
            meshes.append(result)
            return result

    class Pose:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @staticmethod
        def from_list(values, **kwargs):
            return values

    class Solver:
        def __init__(self, options):
            self.options, self.destroy_count, self.calls = options, 0, []
            self.config = SimpleNamespace(use_lm_seed=True, exit_early=True)
            self.seed_ik_solver = SimpleNamespace(action_min=torch.full((7,), -1.),
                                                 action_max=torch.full((7,), 1.))
            self.scene_collision_checker = SimpleNamespace(
                data=SimpleNamespace(update_obstacle_pose=lambda *args: None))

        def destroy(self):
            self.destroy_count += 1

        def update_tool_pose_criteria(self, criteria):
            self.criteria = criteria

        def solve_pose(self, goal, **kwargs):
            self.calls.append(kwargs)
            seeds = kwargs['seed_config']
            count = seeds.shape[1]
            return SimpleNamespace(solution=seeds, success=torch.ones(count, dtype=torch.bool),
                position_error=torch.zeros(count), rotation_error=torch.zeros(count),
                feasible=torch.ones(count, dtype=torch.bool), solve_time=.01)

        def compute_kinematics(self, state):
            count = len(state.position)
            return SimpleNamespace(robot_spheres=torch.zeros((count, 1, 4)),
                                   tool_poses=SimpleNamespace(position=torch.zeros((count, 3))))

    monkeypatch.setitem(sys.modules, 'curobo', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'curobo.scene', SimpleNamespace(Mesh=Mesh,
                                                                  Scene=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'curobo.motion_retargeter',
                        SimpleNamespace(ToolPoseCriteria=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'curobo.types', SimpleNamespace(
        Pose=Pose, GoalToolPose=SimpleNamespace(from_poses=lambda poses: poses)))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    worker = Worker()
    worker.config, worker.seed, worker.count = {}, 17, 64
    worker.robot_config = {'kinematics': {}}
    worker.solver = SimpleNamespace(grasp_solver_untouched=True)
    worker.device = SimpleNamespace(to_device=lambda value: torch.as_tensor(value))
    worker.joint_state = lambda value: SimpleNamespace(position=torch.as_tensor(value))

    def create_solver(robot, **options):
        solver = Solver(options)
        solvers.append(solver)
        return solver

    worker.create_solver = create_solver
    path = tmp_path/'scene.npz'
    np.savez(path, static_points=np.array([[2., 0., 0.], [2., .1, 0.]]),
             moving_reference=np.array([[.5, 0., 0.], [.5, .1, 0.]]),
             contact_reference=np.array([[9., 9., 9.]]))
    config = dict(collision_spheres={'palm': []}, pad_local=[0., 0., 0.],
        scene_arrays_path=str(path), scene_arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        origin=[0., 0., 0.], axis=[0., 0., 1.], solver_mode='local_continuation')
    return worker, config, meshes, solvers


def solve(worker, arc_id=None):
    return worker.solve_arc(point=[.5, .1, 0.], reference_quaternion_xyzw=[0., 0., 0., 1.],
                            seed=[-.99, .99, 0., .1, .2, .3, .4], angle=.05, arc_id=arc_id)


def test_local_optimizer_is_reused_only_within_one_arc_and_rng_resets(worker_scene):
    worker, config, meshes, solvers = worker_scene
    setup = worker.configure_arc(**config)
    initial = solve(worker, setup['arc_id'])
    second = solve(worker, setup['arc_id'])
    assert len(solvers) == 1
    assert initial['optimizer_generation'] == second['optimizer_generation'] == 1
    assert not initial['optimizer_reused_within_arc']
    assert second['optimizer_reused_within_arc']
    assert not second['query_scoped_optimizer']
    first_seeds, second_seeds = [call['seed_config'].numpy() for call in solvers[0].calls]
    assert first_seeds.shape == (1, 64, 7)
    assert np.allclose(first_seeds[0, 0], [-.99, .99, 0., .1, .2, .3, .4])
    assert np.max(np.abs(first_seeds)) <= 1.
    assert np.max(np.abs(first_seeds - first_seeds[:, :1])) < .1
    assert not np.array_equal(first_seeds, second_seeds)
    assert worker.end_arc(setup['arc_id'])['arc_ended']
    assert solvers[0].destroy_count == 1
    with pytest.raises(ValueError, match='No configured arc'):
        solve(worker, setup['arc_id'])
    with pytest.raises(ValueError, match='No configured arc'):
        worker.arc_forward([[0.]*7], arc_id=setup['arc_id'])
    replacement = worker.configure_arc(**config)
    assert replacement['arc_id'] > setup['arc_id']
    assert replacement['optimizer_generation'] == 2
    solve(worker, replacement['arc_id'])
    assert np.array_equal(first_seeds, solvers[-1].calls[0]['seed_config'].numpy())
    assert len(meshes) == 2  # CPU meshes survive solver disposal.
    assert worker.solver.grasp_solver_untouched


def test_reconfigure_invalidates_old_ids_and_requires_local_solve_id(worker_scene):
    worker, config, _, solvers = worker_scene
    first = worker.configure_arc(**config)
    second = worker.configure_arc(**config)
    assert solvers[0].destroy_count == 1
    for operation in (lambda: solve(worker, first['arc_id']),
                      lambda: worker.arc_forward([[0.]*7], arc_id=first['arc_id']),
                      lambda: worker.end_arc(first['arc_id'])):
        with pytest.raises(ValueError, match='Stale or invalid'):
            operation()
    with pytest.raises(ValueError, match='arc_id is required'):
        solve(worker)
    assert solvers[-1].destroy_count == 0
    assert worker.arc_forward([[0.]*7], arc_id=second['arc_id'])['contact_points_world'] == [[0., 0., 0.]]


def test_local_geometry_ownership_and_optimizer_settings(worker_scene):
    worker, config, meshes, solvers = worker_scene
    setup = worker.configure_arc(**config)
    assert setup['observed_points'] == {'stationary': 2, 'moving': 2}
    assert setup['mesh_pitch_by_role_m'] == {'stationary': .005, 'moving': .002}
    assert meshes[0].pitch == .005 and meshes[1].pitch == .002
    assert np.max(meshes[1].points) < 1.
    options = solvers[0].options
    assert options['optimization_dt'] == .2
    assert options['velocity_regularization_weight'] == .001
    assert options['seed_velocity_weight'] == 0.
    assert not solvers[0].config.use_lm_seed
    assert not solvers[0].config.exit_early


def test_global_mode_keeps_query_reset_and_uses_only_current_collision_clouds(worker_scene):
    worker, config, meshes, solvers = worker_scene
    config.pop('solver_mode')
    setup = worker.configure_arc(**config)
    assert setup['solver_mode'] == 'global'
    assert setup['observed_points']['moving'] == 2
    assert setup['current_moving_surface_only']
    assert np.max(meshes[1].points)<1.  # The distant contact reference is metadata.
    assert setup['mesh_pitch_by_role_m']['moving'] == .005
    first, second = solve(worker), solve(worker)
    assert len(solvers) == 2 and solvers[0].destroy_count == 1
    assert first['optimizer_generation'] == 1 and second['optimizer_generation'] == 2
    assert second['query_scoped_optimizer'] and second['solver_recreated']
    assert not second['all_seeds_local']


@pytest.mark.parametrize('mode',['local_continuation','global'])
def test_cleared_clouds_do_not_reuse_a_previous_moving_obstacle(worker_scene,mode):
    worker,config,meshes,solvers=worker_scene
    worker.configure_arc(**config)
    path=Path(config['scene_arrays_path'])
    np.savez(path,static_points=np.empty((0,3)),moving_reference=np.empty((0,3)),
        contact_reference=np.array([[9.,9.,9.]]))
    setup=worker.configure_arc(**{**config,'solver_mode':mode,
        'scene_arrays_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    assert setup['observed_points']=={'stationary':0,'moving':0}
    assert worker.moving_mesh_name is None
    assert len(meshes)==2  # No placeholder or metadata mesh was manufactured.
    solve(worker,setup['arc_id'])


@pytest.mark.parametrize('change', [
    {'local_dt': 0.}, {'local_dt': float('nan')}, {'mesh_pitch': -.01},
    {'moving_mesh_pitch': float('inf')}, {'solver_mode': 'unsupported'},
    {'local_seed_std': -.1}, {'local_velocity_weight': float('nan')},
    {'axis': [0., 0., 0.]}, {'origin': [0., float('inf'), 0.]},
    {'scene_arrays_sha256': 'incorrect'},
])
def test_invalid_arc_configuration_never_creates_solver(worker_scene, change):
    worker, config, _, solvers = worker_scene
    with pytest.raises(ValueError):
        worker.configure_arc(**{**config, **change})
    assert solvers == []
