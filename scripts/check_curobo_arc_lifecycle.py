"""GPU regression for arc-scoped optimizer reuse and reset on saved observations.

Run A (two local increments), B (an unrelated saved failed arc), then A again
through one persistent TaskIK worker. This checks numerical reproducibility,
native endpoint admission, and stale/closed arc rejection; it issues no robot
commands and does not instantiate the simulator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.runtime/mc-vamp-deps')]

import numpy as np
from scipy.spatial.transform import Rotation
import vamp

from benchmark_curobo_v2 import load_saved_arc
from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.curobo_v2_arc import CuroboV2ArcProposer
from vlm_tamp.mcvamp_contact_numerical import NativeScene, clear_native_cache
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def source_snapshot(output):
    sources = sorted({
        *ROOT.glob('src/**/*.py'), *ROOT.glob('src/**/*.cc'),
        *ROOT.glob('src/**/*.hh'), Path(__file__).resolve(),
        ROOT / 'scripts/benchmark_curobo_v2.py',
    })
    hashes = {}
    for source in sources:
        name = source.relative_to(ROOT)
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(name)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / 'source-hashes.json', hashes)
    return hashes


def solve_payload(proposer, seed, angle):
    rotation = (Rotation.from_rotvec(float(angle) * proposer.context.axis).as_matrix()
                @ proposer.model.reference_rotation)
    return {
        'point': proposer.model.point(angle).tolist(),
        'reference_quaternion_xyzw': Rotation.from_matrix(rotation).as_quat().tolist(),
        'seed': np.asarray(seed).tolist(),
        'angle': float(angle),
        'mode': proposer.model.grasp_constraint,
    }


def rejection_probe(client, operation, **payload):
    before = time.perf_counter()
    try:
        result = client.request(operation, **payload)
    except RuntimeError as error:
        return {'rejected': True, 'error': str(error),
                'wall_seconds': time.perf_counter() - before}
    return {'rejected': False, 'response': result,
            'wall_seconds': time.perf_counter() - before}


def endpoint_comparison(first, second, tolerance):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    matching_shape = a.shape == b.shape
    error = float(np.max(np.abs(a - b))) if matching_shape and a.size else None
    return {'first_shape': list(a.shape), 'second_shape': list(b.shape),
            'maximum_absolute_difference_rad': error,
            'tolerance_rad': float(tolerance),
            'accepted': bool(matching_shape and error is not None and error < tolerance)}


def run_query(ik, source, record, target, label, *, angle=None,
              previous_arc_id=None, second_increment=False):
    target.mkdir()
    scene, metadata, start, inputs, reference = load_saved_arc(source, record)
    options = {**reference['configuration'], **inputs['search_options']}
    goal = float(inputs['goal_angle_rad'])
    angle = goal if angle is None else float(angle)
    write_json(target / 'scene-reference.json', reference)
    write_json(target / 'planning-input.json', inputs)
    row = {
        'label': label, 'source': str(source),
        'source_arrays_sha256': reference['arrays_sha256'],
        'source_arrays_hash_verified': True,
        'original_status': read_json(source / 'result.json')['status'],
        'query_angle_rad': angle, 'full_arc_goal_rad': goal,
        'grasp_constraint': options['grasp_constraint'],
        'worker_pid': ik.client.process.pid,
    }
    before = time.perf_counter()
    with NativeScene(scene, record, metadata, start, goal, aligned=False,
                     grasp_constraint=options['grasp_constraint']) as context:
        context.motion_options = options
        proposer = CuroboV2ArcProposer(ik, context, target, goal)
        row['setup'] = proposer.setup
        arc_id = proposer.arc_id
        row['arc_id'] = arc_id
        request = solve_payload(proposer, start, angle)
        try:
            if previous_arc_id is not None:
                row['stale_solve'] = rejection_probe(
                    ik.client, 'solve_arc', arc_id=previous_arc_id, **request)
                row['stale_forward'] = rejection_probe(
                    ik.client, 'arc_forward', arc_id=previous_arc_id,
                    joints=[start.tolist()])
            row['missing_id_solve'] = rejection_probe(ik.client, 'solve_arc', **request)
            goals, checks = proposer.candidates(start, angle)
            row['admitted_endpoints'] = [q.tolist() for q in goals]
            row['admitted_count'] = len(goals)
            row['admission_checks'] = checks
            row['first_query'] = proposer.queries[-1]
            row['native_admission_rechecked'] = bool(
                goals and all(context.validate(q)['accepted'] for q in goals))
            if second_increment and goals:
                later, later_checks = proposer.candidates(goals[0], 2 * angle)
                row['second_increment'] = {
                    'angle_rad': 2 * angle,
                    'admitted_count': len(later),
                    'admitted_endpoints': [q.tolist() for q in later],
                    'admission_checks': later_checks,
                    'query': proposer.queries[-1],
                    'arc_id_unchanged': proposer.arc_id == arc_id,
                }
            write_json(target / 'native-scene.json', context.record)
        finally:
            row['close_result'] = proposer.close()
        row['closed_solve'] = rejection_probe(ik.client, 'solve_arc', arc_id=arc_id, **request)
        row['closed_forward'] = rejection_probe(ik.client, 'arc_forward', arc_id=arc_id,
                                                joints=[start.tolist()])
        row['worker_healthy_after_close'] = bool(ik.client.forward([start.tolist()]))
    row['wall_seconds'] = time.perf_counter() - before
    write_json(target / 'result.json', row)
    print(json.dumps({'query': label, 'arc_id': arc_id,
                      'admitted': row['admitted_count'],
                      'wall_seconds': row['wall_seconds']}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-a', type=Path, default=ROOT / (
        'results/curobo-v2-migration-20261022/physical-replay002-query-state/attempt01/010-arc'))
    parser.add_argument('--source-b', type=Path, default=ROOT / (
        'results/compact-luna-practice-20261022/practice-v1/attempt01/012-arc'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--angle-deg', type=float, default=3.)
    parser.add_argument('--tolerance-rad', type=float, default=1e-5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_snapshot(args.output)
    report = {
        'scope': __doc__, 'simulator_queried': False, 'robot_motion_issued': False,
        'evaluation_only_object_state_used': False,
        'observations': 'Retained RGB-D-derived scene points and robot-only calibration.',
        'queries': [], 'checks': {}, 'accepted': False,
    }
    started = time.perf_counter()
    ik = None
    try:
        record = read_json(args.source_a.parent / 'robot-chain.json')
        other = read_json(args.source_b.parent / 'robot-chain.json')
        robot_keys = ('reference_joints', 'joint_axes_world', 'joint_anchors_world_m',
                      'reference_tcp_world', 'joint_limits')
        report['checks']['robot_calibration_matches'] = all(
            np.array_equal(np.asarray(record[key]), np.asarray(other[key])) for key in robot_keys)
        if not report['checks']['robot_calibration_matches']:
            raise ValueError('A and B must use the same calibrated robot in one worker')
        write_json(args.output / 'robot-chain.json', record)
        config = read_json(args.source_a / 'scene-reference.json')['configuration']
        config['curobo_backend'] = 'v2'
        empty = vamp.Environment()
        before = time.perf_counter()
        ik = TaskIK(record, config,
                    lambda q: vamp.panda.validate(np.asarray(q).tolist(), empty, True),
                    output=args.output / 'ik')
        report['initialization_seconds'] = time.perf_counter() - before
        report['runtime'] = ik.client.metadata
        previous_id = None
        for label, source, angle, extra in [
            ('A-first', args.source_a, np.radians(args.angle_deg), True),
            ('B-intervening', args.source_b, None, False),
            ('A-repeat', args.source_a, np.radians(args.angle_deg), False),
        ]:
            row = run_query(ik, source, record, args.output / label, label,
                            angle=angle, previous_arc_id=previous_id,
                            second_increment=extra)
            report['queries'].append(row)
            previous_id = row['arc_id']
            write_json(args.output / 'results.json', report)
        first, middle, repeat = report['queries']
        report['admitted_endpoint_reproducibility'] = endpoint_comparison(
            first['admitted_endpoints'], repeat['admitted_endpoints'], args.tolerance_rad)
        report['gpu_solution_reproducibility'] = endpoint_comparison(
            first['first_query']['gpu_result']['solutions'],
            repeat['first_query']['gpu_result']['solutions'], args.tolerance_rad)
        ids = [row['arc_id'] for row in report['queries']]
        generations = [row['setup']['optimizer_generation'] for row in report['queries']]
        first_gpu = first['first_query']['gpu_result']
        second_gpu = first.get('second_increment', {}).get('query', {}).get('gpu_result', {})
        checks = report['checks']
        checks.update({
            'one_persistent_worker': len({row['worker_pid'] for row in report['queries']}) == 1,
            'arc_ids_advance': ids[0] < ids[1] < ids[2],
            'optimizer_generations_advance': generations[0] < generations[1] < generations[2],
            'optimizer_recreated_for_each_arc': all(
                row['setup']['solver_recreated'] for row in report['queries']),
            'optimizer_lifetime_one_arc': all(
                row['setup']['optimizer_lifetime'] == 'one_arc' for row in report['queries']),
            'first_query_has_fresh_state': all(
                not row['first_query']['gpu_result']['optimizer_reused_within_arc']
                for row in report['queries']),
            'second_query_reuses_same_optimizer': bool(
                second_gpu.get('optimizer_reused_within_arc')
                and second_gpu.get('optimizer_generation') == first_gpu['optimizer_generation']),
            'A_first_admitted': first['admitted_count'] > 0,
            'A_repeat_admitted': repeat['admitted_count'] > 0,
            'A_native_admission_rechecked': first['native_admission_rechecked']
                                            and repeat['native_admission_rechecked'],
            'same_arc_second_increment_admitted': first.get('second_increment', {}).get('admitted_count', 0) > 0,
            'same_arc_id_reused': first.get('second_increment', {}).get('arc_id_unchanged', False),
            'admitted_endpoints_reproduce': report['admitted_endpoint_reproducibility']['accepted'],
            'raw_gpu_solutions_reproduce': report['gpu_solution_reproducibility']['accepted'],
            'GPU_acceptance_flags_reproduce': first['first_query']['gpu_result']['success']
                                              == repeat['first_query']['gpu_result']['success'],
            'stale_solve_rejected': all(row['stale_solve']['rejected'] for row in (middle, repeat)),
            'stale_forward_rejected': all(row['stale_forward']['rejected'] for row in (middle, repeat)),
            'missing_ID_solve_rejected': all(row['missing_id_solve']['rejected'] for row in report['queries']),
            'closed_solve_rejected': all(row['closed_solve']['rejected'] for row in report['queries']),
            'closed_forward_rejected': all(row['closed_forward']['rejected'] for row in report['queries']),
            'worker_healthy_after_arc_close': all(
                row['worker_healthy_after_close'] for row in report['queries']),
        })
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
    finally:
        if ik is not None:
            ik.close()
        clear_native_cache()
        report['checks']['source_unchanged_during_test'] = all(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in sources.items())
        report['wall_seconds'] = time.perf_counter() - started
        report['accepted'] = 'error' not in report and bool(report['checks']) and all(report['checks'].values())
        write_json(args.output / 'results.json', report)
        print(json.dumps({key: report[key] for key in ('accepted', 'checks', 'wall_seconds')}
                         | {'error': report.get('error')}), flush=True)
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
