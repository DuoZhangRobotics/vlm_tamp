"""Reproducible two-task ReKep/GraspGen physical LIBERO experiment."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def dump(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')


def reference_indices(program):
    """Track used landmarks while retaining the original numbered-array indexing."""
    used = {x for x in program.grasp_keypoints + program.release_keypoints if x >= 0}
    for stage in program.stages:
        for constraint in stage.subgoal_constraints + stage.path_constraints:
            for node in ast.walk(constraint.definition):
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'keypoints':
                    first = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
                    if isinstance(first, ast.Slice):
                        used.update(range(program.num_keypoints))
                    else:
                        value = ast.literal_eval(first)
                        used.update(value if isinstance(value, (list, tuple)) else [value])
    return np.asarray(sorted(used), int)


def track_references(perception, bundle, response, observation_dir, used):
    subset = dict(bundle)
    for name in ('points_world', 'pixels_yx', 'groups', 'features'):
        subset[name] = bundle[name][used]
    tracked = perception.track(subset, Path(observation_dir) / 'obs.npz', response['metadata'])
    result = dict(bundle, cameras=tracked['cameras'], meta=tracked['meta'])
    for name in ('points_world', 'pixels_yx'):
        result[name] = bundle[name].copy()
        result[name][used] = tracked[name]
    result['tracking'] = tracked['tracking']
    return result


class EpisodeFailure(RuntimeError):
    def __init__(self, stage, detail):
        self.stage, self.detail = stage, str(detail)
        super().__init__(self.detail)


def retained_grasp(before_pose, after_pose, before_point, after_point, finger_qpos, *, require_motion=True):
    """Visual rigid co-motion and nonempty closure, without simulator attachment."""
    from vlm_tamp.rekep_libero.geometry import pose_matrix, transform_points
    before, after = pose_matrix(before_pose), pose_matrix(after_pose)
    motion = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    expected = transform_points([before_point], after @ np.linalg.inv(before))[0]
    rigid_error = float(np.linalg.norm(np.asarray(after_point) - expected))
    observed_motion = float(np.linalg.norm(np.asarray(after_point) - np.asarray(before_point)))
    gap = float(np.sum(np.abs(finger_qpos)))
    closure = .004 < gap < .078
    tolerance = max(.002, .25 * motion)
    follows = rigid_error <= tolerance
    if require_motion:
        follows = follows and motion >= .005 and observed_motion >= .5 * motion
    return {'verified': bool(closure and follows), 'tcp_motion_m': motion,
            'source_motion_m': observed_motion, 'rigid_prediction_error_m': rigid_error,
            'maximum_rigid_error_m': tolerance, 'physical_gap_m': gap,
            'source': 'physical nonempty closure and RGB-D rigid co-motion'}


def run_episode(config, task, state, output, perception, grasp_model, *, development=False, allow_paid=False):
    from vlm_tamp.rekep_libero.api import ApiError, BudgetError
    from vlm_tamp.rekep_libero.geometry import ObservedSDF, pose_matrix, transform_points
    from vlm_tamp.rekep_libero.planning import PlanningAdapter
    from vlm_tamp.rekep_libero.program import generate_program, ProgramError
    from vlm_tamp.rekep_libero.sim_client import SimClient

    output = Path(output)
    if (output / 'episode.json').exists():
        return json.loads((output / 'episode.json').read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Incomplete episode already exists; preserve it and inspect before replay: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    episode_id = output.name
    record = {'episode_id': episode_id, 'task_type': task['task_type'], 'task_index': task['task_index'],
              'initial_state': state, 'seed': config['seed'], 'development': development,
              'success': False, 'status': 'running', 'failure_stage': None, 'control_steps': 0,
              'physical_grasp_attempts': 0, 'completed_stages': 0, 'stages': [],
              'provided_inventory': False, 'ground_truth_used_by_policy': False,
              'assisted_grasping': False, 'scope': config['scope']}
    dump(output / 'episode-start.json', record)
    start = time.monotonic()
    response, bundle, sim = None, None, None
    held_id, held_cloud, attachment_verified = -1, None, False
    attached_indices = np.empty(0, int)
    current_failure_stage = 'execution'

    def limits():
        if time.monotonic() - start > config['max_episode_seconds']:
            raise EpisodeFailure('execution', 'episode_wall_time_limit')
        if response and response.get('step_limit_reached'):
            raise EpisodeFailure('execution', '600_control_step_limit')

    def observe(label):
        nonlocal response, bundle
        folder = output / label
        response = sim.observe(folder)
        bundle = track_references(perception, bundle, response, folder, used)
        dump(folder / 'tracking.json', {'referenced_indices': used, 'diagnostics': bundle['tracking'],
                                      'points_world': bundle['points_world']})
        return folder

    def build_sdf(label):
        sdf = ObservedSDF.from_rgbd(list(bundle['cameras'].values()), voxel_size=config['voxel_size_m'])
        known = response['metadata'].get('robot_collision_points_world', [])
        volumes = response['metadata'].get('robot_collision_volumes_world', [])
        if volumes:
            sdf.exclude_known_robot_volumes(volumes)
        if len(known):
            sdf.exclude_known_robot(np.asarray(known))
        if attachment_verified and held_cloud is not None:
            sdf.exclude_attached_region(held_cloud)
        sdf.save(output / (label + '-sdf.npz'))
        return sdf

    def send_pose(pose, gripper=0, max_steps=20):
        nonlocal response
        limits()
        response = sim.action(np.asarray(pose).tolist(), gripper=gripper, max_steps=max_steps)
        record['control_steps'] = response['control_steps']
        record['success'] = record['success'] or bool(response['evaluation']['task_success'])
        if not response.get('reached_pose') and gripper == 0:
            raise EpisodeFailure('execution', 'controller_did_not_reach_checked_waypoint')

    def attachment_cost(index):
        return 0.0 if index == held_id and attachment_verified else 1.0

    try:
        sim = SimClient(config['sim_python'], timeout=180, log_path=output / 'simulator.log')
        response = sim.reset(task['task_index'], state, output / 'initial', seed=config['seed'])
        record['instruction'] = response['metadata']['instruction']
        # Establish actual open-gripper geometry before predicting detector grasps.
        send_pose(response['metadata']['eef_pose_world_xyzw'], gripper=-1, max_steps=60)
        response = sim.observe(output / 'open-gripper')
        current_failure_stage = 'perception'
        import torch
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        bundle = perception.prepare(output / 'open-gripper/obs.npz', response['metadata'], output / 'perception')
        record['num_keypoints'] = bundle['num_keypoints']
        current_failure_stage = 'program'
        api_config = {**config, 'allow_paid': allow_paid, 'scene_id': episode_id,
                      'ledger_dir': str(ROOT / config['ledger_dir']),
                      'prior_spend_usd': config['prior_spend'], 'budget_usd': config['budget']}
        program = generate_program(bundle['numbered_image'], record['instruction'], task['task_type'],
                                   bundle['num_keypoints'], output / 'program', api_config, attachment_cost)
        program.validate_values(response['metadata']['eef_pose_world_xyzw'][:3], bundle['points_world'])
        used = reference_indices(program)
        record['program'] = program.metadata()
        for stage in program.stages:
            limits()
            stage_log = {'index': stage.index, 'grasp_keypoint': stage.grasp_keypoint,
                         'release_keypoint': stage.release_keypoint, 'plans': [], 'completed': False}
            record['stages'].append(stage_log)
            np.random.seed(config['seed'] + stage.index)
            calibration = response['metadata']['robot_calibration']
            planner = PlanningAdapter(sim, calibration, collision_margin=config['collision_margin_m'])
            dump(output / 'planner-adaptations.json', planner.metadata)
            if stage.grasp_keypoint >= 0:
                current_failure_stage = 'grasp'
                grasp_dir = output / f'stage{stage.index}-grasps'
                sdf = build_sdf(f'stage{stage.index}-grasp')
                selection = grasp_model.prepare_grasps(
                    perception, bundle, stage.grasp_keypoint, grasp_dir, calibration, sim, sdf,
                    articulated=task['task_type'] == 'open_middle_drawer')
                stage_log['grasp_selection'] = selection
                if not selection['accepted']:
                    raise EpisodeFailure('grasp', selection['failure'])
                cloud = np.load(grasp_dir / 'selected_cloud.npy', allow_pickle=False)
                target = np.asarray(selection['selected_tcp_pose'])
                pregrasp = target.copy()
                pregrasp[:3] -= pose_matrix(target)[:3, 2] * .06
                empty = SimpleNamespace(subgoal_constraints=[], path_constraints=[])
                current_failure_stage = 'planning'
                for suffix, goal in [('pregrasp', pregrasp), ('approach', target)]:
                    planned = planner.solve_stage(empty, response['metadata']['eef_pose_world_xyzw'],
                                                  response['metadata']['joint_positions'], bundle['points_world'],
                                                  np.zeros(len(bundle['points_world']), bool), sdf,
                                                  fixed_grasp_tcp_pose=goal, target_points=cloud)
                    dump(output / f'stage{stage.index}-{suffix}-plan.json', planned)
                    stage_log['plans'].append({'kind': suffix, 'accepted': planned['accepted'],
                                               'failure': planned.get('failure')})
                    if not planned['accepted']:
                        raise EpisodeFailure('planning', planned['failure'])
                    current_failure_stage = 'execution'
                    for waypoint in planned['poses_world_xyzw'][1:]:
                        send_pose(waypoint)
                record['physical_grasp_attempts'] += 1
                send_pose(target, gripper=1, max_steps=120)
                current_failure_stage = 'tracking'
                observe(f'stage{stage.index}-closed')
                closed_pose = np.asarray(response['metadata']['eef_pose_world_xyzw'])
                closed_point = bundle['points_world'][stage.grasp_keypoint].copy()
                closed_qpos = np.asarray(response['metadata']['gripper_qpos'])
                # Retention is tested by a small physical retraction along the
                # detector's approach, after checking geometry and pull costs.
                probe = closed_pose.copy()
                probe[:3] -= pose_matrix(closed_pose)[:3, 2] * .01
                provisional_points = bundle['points_world'].copy()
                part = bundle['groups'] == bundle['groups'][stage.grasp_keypoint]
                if task['task_type'] == 'open_middle_drawer':
                    # A cabinet-wide RGB mask cannot establish a rigid drawer.
                    # Predict movement of the selected handle landmark only;
                    # additional drawer landmarks need observed co-motion.
                    part = np.zeros(len(provisional_points), bool)
                    part[stage.grasp_keypoint] = True
                provisional_points[part] += probe[:3] - closed_pose[:3]
                next_stage = program.stages[stage.index] if stage.index < program.num_stages else None
                if next_stage:
                    program.set_attachment_cost(lambda idx: 0. if idx == stage.grasp_keypoint else 1.)
                    try:
                        residuals = [fn(probe[:3], provisional_points) for fn in next_stage.path_constraints]
                    finally:
                        program.set_attachment_cost(attachment_cost)
                    if any(x > .0001 for x in residuals):
                        raise EpisodeFailure('grasp', 'retention_probe_violates_generated_pull_constraint')
                current_failure_stage = 'planning'
                sdf = build_sdf(f'stage{stage.index}-retention')
                planned = planner.solve_stage(empty, closed_pose, response['metadata']['joint_positions'],
                                               bundle['points_world'], part, sdf,
                                               fixed_grasp_tcp_pose=probe, target_points=cloud)
                dump(output / f'stage{stage.index}-retention-plan.json', planned)
                if not planned['accepted']:
                    raise EpisodeFailure('planning', planned['failure'])
                for waypoint in planned['poses_world_xyzw'][1:]:
                    send_pose(waypoint, gripper=1, max_steps=5)
                current_failure_stage = 'tracking'
                observe(f'stage{stage.index}-retained')
                moved_pose = np.asarray(response['metadata']['eef_pose_world_xyzw'])
                moved_point = bundle['points_world'][stage.grasp_keypoint]
                retention = retained_grasp(closed_pose, moved_pose, closed_point, moved_point,
                                          response['metadata']['gripper_qpos'])
                attachment_verified = retention['verified']
                stage_log['retention'] = dict(retention, closed_qpos=closed_qpos)
                if not attachment_verified:
                    raise EpisodeFailure('grasp', 'physical_grasp_retention_not_verified')
                held_id, held_cloud = stage.grasp_keypoint, cloud
                if task['task_type'] == 'open_middle_drawer':
                    delta = moved_pose[:3] - closed_pose[:3]
                    changes = bundle['points_world'] - provisional_points + (probe[:3] - closed_pose[:3]) * part[:, None]
                    agrees = np.linalg.norm(changes - delta, axis=1) <= .008
                    moved = np.linalg.norm(changes, axis=1) >= .005
                    attached_indices = np.union1d([held_id], used[agrees[used] & moved[used]])
                else:
                    attached_indices = np.flatnonzero(part)
                held_cloud = transform_points(held_cloud, pose_matrix(moved_pose) @ np.linalg.inv(pose_matrix(closed_pose)))
                stage_log['visually_attached_landmark_indices'] = attached_indices
                stage_log['completed'] = True
            else:
                current_failure_stage = 'planning'
                for attempt in range(config['max_stage_replans']):
                    limits()
                    actual = response['metadata']['eef_pose_world_xyzw']
                    residuals = [fn(actual[:3], bundle['points_world']) for fn in stage.subgoal_constraints]
                    actual_paths = [fn(actual[:3], bundle['points_world']) for fn in stage.path_constraints]
                    stage_log['actual_goal_residuals'] = residuals
                    stage_log['actual_path_residuals'] = actual_paths
                    if any(x > .0001 for x in actual_paths):
                        raise EpisodeFailure('planning', 'observed_path_constraint_violation')
                    if all(x <= .0001 for x in residuals):
                        stage_log['completed'] = True
                        break
                    sdf = build_sdf(f'stage{stage.index}-replan{attempt}')
                    movable = np.zeros(len(bundle['points_world']), bool)
                    if held_id >= 0:
                        movable[attached_indices] = True
                    planned = planner.solve_stage(stage, actual, response['metadata']['joint_positions'],
                                                   bundle['points_world'], movable, sdf, target_points=held_cloud)
                    dump(output / f'stage{stage.index}-replan{attempt}-plan.json', planned)
                    stage_log['plans'].append({'kind': 'manipulation', 'accepted': planned['accepted'],
                                               'failure': planned.get('failure')})
                    if not planned['accepted']:
                        raise EpisodeFailure('planning', planned['failure'])
                    current_failure_stage = 'execution'
                    for waypoint in planned['poses_world_xyzw'][1:6]:
                        send_pose(waypoint, gripper=1 if held_id >= 0 else 0, max_steps=5)
                    current_failure_stage = 'tracking'
                    old_tcp = np.asarray(actual)
                    old_source = bundle['points_world'][held_id].copy() if held_id >= 0 else None
                    observe(f'stage{stage.index}-replan{attempt}-observation')
                    if held_cloud is not None:
                        new_tcp = np.asarray(response['metadata']['eef_pose_world_xyzw'])
                        retention = retained_grasp(old_tcp, new_tcp, old_source, bundle['points_world'][held_id],
                                                   response['metadata']['gripper_qpos'], require_motion=False)
                        stage_log.setdefault('retention_checks', []).append(retention)
                        attachment_verified = retention['verified']
                        if not attachment_verified:
                            raise EpisodeFailure('grasp', 'grasp_lost_during_manipulation')
                        held_cloud = transform_points(held_cloud, pose_matrix(new_tcp) @ np.linalg.inv(pose_matrix(old_tcp)))
                    current_failure_stage = 'planning'
                if not stage_log['completed']:
                    raise EpisodeFailure('planning', 'stage_replan_limit_without_satisfied_goal')
                if stage.release_keypoint >= 0:
                    send_pose(response['metadata']['eef_pose_world_xyzw'], gripper=-1, max_steps=60)
                    held_id, held_cloud, attachment_verified = -1, None, False
                    observe(f'stage{stage.index}-released')
            record['completed_stages'] += 1
            dump(output / 'episode-progress.json', record)
        record['status'] = 'success' if record['success'] else 'failed'
        if not record['success']:
            record.update(failure_stage='execution', failure_detail='program_finished_without_LIBERO_success')
    except EpisodeFailure as error:
        record.update(status='success' if record['success'] else 'failed', failure_stage=error.stage,
                      failure_detail=error.detail)
    except (ProgramError, ApiError) as error:
        category = 'budget' if isinstance(error, BudgetError) else 'api' if isinstance(error, ApiError) else 'program'
        record.update(status='failed', failure_stage=category,
                      failure_detail=str(error))
    except Exception as error:
        record.update(status='failed', failure_stage=current_failure_stage, failure_detail=str(error),
                      exception_type=type(error).__name__)
        (output / 'error_traceback.txt').write_text(traceback.format_exc())
    finally:
        if sim is not None:
            try:
                final = sim.close()
                if isinstance(final, dict):
                    record['control_steps'] = final.get('control_steps', record['control_steps'])
                    record['success'] = record['success'] or bool(final.get('evaluation', {}).get('task_success'))
            except Exception as error:
                record['close_error'] = str(error)
        record['elapsed_seconds'] = time.monotonic() - start
        video = output / 'initial/rollout.mp4'
        record['video'] = str(video.resolve()) if video.exists() else None
        if record['success']:
            record['status'] = 'success'
        dump(output / 'episode.json', record)
    print(json.dumps({k: record.get(k) for k in ('episode_id', 'success', 'failure_stage', 'failure_detail',
                                                'control_steps', 'physical_grasp_attempts')}), flush=True)
    return record


def freeze(config, destination):
    files = [ROOT / 'configs/rekep_libero_v1.json', ROOT / 'requirements-rekep-libero.txt', Path(__file__).resolve()]
    files += sorted((ROOT / 'src/vlm_tamp/rekep_libero').glob('*.py'))
    for name in ('keypoint_proposal.py', 'subgoal_solver.py', 'path_solver.py', 'utils.py',
                 'transform_utils.py', 'vlm_query/prompt_template.txt'):
        files.append(ROOT / 'third_party/ReKep' / name)
    files += [ROOT / 'results/rekep-libero-setup/runtime.json',
              ROOT / 'results/rekep-libero-setup/requirements-installed.txt']
    files += sorted((ROOT / '.runtime/graspgen-models/checkpoints').glob('*'))
    files += [ROOT / '.runtime/torch-hub/checkpoints/dinov2_vits14_pretrain.pth',
              ROOT / 'third_party/MobileSAM/weights/mobile_sam.pt']
    for folder in ('grasp_gen', 'config', 'assets/panda_gripper'):
        files += [p for p in (ROOT / 'third_party/GraspGen' / folder).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
    for folder in ('third_party/MobileSAM/mobile_sam', 'third_party/dinov2/dinov2'):
        files += [p for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts]
    def file_sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    manifest = {'config': config, 'files_sha256': {str(p.relative_to(ROOT)): file_sha(p) for p in files},
                'frozen_before_evaluation': True}
    destination = Path(destination)
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous != manifest:
            raise RuntimeError('Frozen implementation/configuration changed; use a new versioned run')
    else:
        dump(destination, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rekep_libero_v1.json')
    parser.add_argument('--phase', choices=['development', 'evaluation', 'all', 'report'], default='all')
    parser.add_argument('--allow-paid', action='store_true', help='Use the authorized $2 ledger for uncached Luna requests')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    output = ROOT / config['output']
    output.mkdir(parents=True, exist_ok=True)
    development, evaluation = [], []
    for task in config['tasks']:
        fmt = f"libero_goal-{task['task_index']:02d}-state{{state:03d}}-seed{config['seed']}"
        development.append(output / ('dev-' + fmt.format(state=task['development_state'])))
        evaluation.extend(output / fmt.format(state=state) for state in config['evaluation_states'])
    dump(output / 'planned-episodes.json', {'development': [str(x) for x in development],
                                          'evaluation': [str(x) for x in evaluation], 'evaluation_denominator': 10})
    if args.phase != 'report':
        from vlm_tamp.rekep_libero.perception import Perception, GraspGen
        perception = Perception()
        grasp_model = GraspGen(ROOT / config['graspgen_model_config'])
        if args.phase in ('development', 'all'):
            for task, folder in zip(config['tasks'], development):
                run_episode(config, task, task['development_state'], folder, perception, grasp_model,
                            development=True, allow_paid=args.allow_paid)
        if args.phase in ('evaluation', 'all'):
            freeze(config, output / 'frozen-protocol.json')
            for task in config['tasks']:
                for state in config['evaluation_states']:
                    folder = output / f"libero_goal-{task['task_index']:02d}-state{state:03d}-seed{config['seed']}"
                    run_episode(config, task, state, folder, perception, grasp_model, allow_paid=args.allow_paid)
    from vlm_tamp.rekep_libero.report import build_report
    result = build_report(evaluation, ROOT / 'results/rekep-libero-report', expected_episodes=10,
                          ledger_path=ROOT / config['ledger_dir'] / 'ledger.json',
                          expected_episode_paths=evaluation, development_dirs=development)
    print(json.dumps({k: result[k] for k in ('evaluation_planned', 'evaluation_successes',
                                           'evaluation_physical_grasp_attempts', 'failures')}), flush=True)


if __name__ == '__main__':
    main()
