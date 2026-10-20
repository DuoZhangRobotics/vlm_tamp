"""Paired microwave diagnosis with frozen grasps and estimated/oracle hinges.

The oracle arm intentionally uses simulator hinge geometry from a previous
episode. Its results are diagnostics, not sensor-only benchmark results.
The existing executor and its frozen experiment sources remain unchanged.
"""

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]

from try_rekep_bowl_v3 import save
from vlm_tamp.rekep_libero import mechanism_execution as executor
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.mechanism_geometry import Mechanism
from vlm_tamp.rekep_libero.perception import Perception


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NoNewGrasps:
    def generate(self, *args, **kwargs):
        raise RuntimeError('This paired diagnostic must reuse the frozen GraspGen candidates')


def oracle_geometry(spec, reference):
    diagnostics = read(reference/'initial/evaluation_contact_diagnostics.json')
    opened = read(reference/'opened/metadata.json')['control_steps']
    step = next(row for row in diagnostics['steps'] if row['step'] == opened)
    joint = next(row for row in step['evaluation_only_articulation_state']
                 if row['joint_name'] == 'microwave_1_microjoint')
    axis = np.asarray(joint['axis_world'], float)
    axis /= np.linalg.norm(axis)
    # Preserve the estimated opening/closing sign; replace only axis geometry.
    axis *= 1 if axis @ np.asarray(spec['joint_axis_world']) >= 0 else -1
    result = copy.deepcopy(spec)
    result.update(joint_axis_world=axis.tolist(), joint_origin_world_m=joint['anchor_world_m'],
                  simulator_scene_state_used=True,
                  geometry_source='ORACLE DIAGNOSTIC: native simulator hinge; other geometry remains RGB-D')
    return result, {
        'oracle_hinge_used': True, 'oracle_progress_used': False,
        'native_joint_name': joint['joint_name'], 'reference_step': opened,
        'reference_file': str((reference/'initial/evaluation_contact_diagnostics.json').relative_to(ROOT)),
        'reference_sha256': digest(reference/'initial/evaluation_contact_diagnostics.json'),
        'fields_replaced': ['joint_axis_world', 'joint_origin_world_m'],
        'travel_direction_and_target_unchanged': True,
    }


@contextmanager
def fixed_grasp_hooks(mode, reference):
    original_geometry, original_selector = executor.ground_geometry, executor.choose_grasp

    def geometry(*args, **kwargs):
        mechanism, panel_mask = original_geometry(*args, **kwargs)
        folder = Path(args[-1])
        save(folder/'estimated-task-spec.json', mechanism.spec)
        if mode == 'oracle':
            spec, evidence = oracle_geometry(mechanism.spec, reference)
            save(folder/'oracle-hinge-input.json', evidence)
            save(folder/'task-spec.json', spec)
            mechanism = Mechanism(spec)
        return mechanism, panel_mask

    def selector(generated, cloud, calibration, ik, initial, sim, scene, mechanism,
                 open_qpos, output, save_record):
        chosen = copy.deepcopy(read(reference/'chosen-grasp.json'))
        q = np.asarray(chosen['solution_joints'])
        expected = pose_matrix(chosen['fk_pose_world_xyzw'])
        actual = pose_matrix(ik.chain.forward(q))
        position_error = float(np.linalg.norm(expected[:3, 3]-actual[:3, 3]))
        rotation_error = float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())
        start_error = float(np.max(np.abs(np.asarray(chosen['approach']['joint_postures'][0])-initial)))
        if position_error > 2e-5 or rotation_error > 2e-4 or start_error > 1e-6 or not ik.self_valid(q):
            raise ValueError('Frozen grasp or approach does not match the current calibrated robot')
        save_record(Path(output)/'fixed-grasp-evidence.json', {
            'reference': str((reference/'chosen-grasp.json').relative_to(ROOT)),
            'reference_sha256': digest(reference/'chosen-grasp.json'),
            'grasp_pose_posture_and_approach_reused_exactly': True,
            'new_grasp_selection_performed': False,
            'baseline_task_path_is_historical_only': True,
            'postclosure_path_recomputed_with_current_hinge': True,
            'fk_position_error_m': position_error, 'fk_orientation_error_rad': rotation_error,
            'approach_initial_joint_error_rad': start_error,
        })
        save_record(Path(output)/'grasp-admission.json', {
            'selected': chosen, 'rejected_candidates': [], 'diagnostic_frozen_selection': True,
            'note': 'Reuse original grasp/approach; normal per-command checks and postclosure preflight remain active',
        })
        return chosen

    executor.ground_geometry, executor.choose_grasp = geometry, selector
    try:
        yield
    finally:
        executor.ground_geometry, executor.choose_grasp = original_geometry, original_selector


def episode_audit(folder, reference):
    summary = read(folder/'summary.json')
    actions = read(folder/'initial/actions.json')
    baseline = read(reference/'initial/actions.json')
    closure_step = read(reference/'closed/metadata.json')['control_steps']
    prefix = actions[:closure_step]
    actual = np.asarray([a['joint_positions'] for a in prefix])
    original = np.asarray([a['joint_positions'] for a in baseline[:len(prefix)]])
    complete = len(prefix) == closure_step
    joint_error = float(np.max(np.abs(actual-original))) if len(prefix) else None
    target_error = max((float(np.max(np.abs(np.asarray(a['target_pose_world_xyzw'])-
                                           b['target_pose_world_xyzw'])))
                        for a, b in zip(prefix, baseline)), default=0.)
    states = read(folder/'initial/evaluation_contact_diagnostics.json')['steps']
    values = [next(j['position'] for j in s['evaluation_only_articulation_state']
                   if j['joint_name'] == 'microwave_1_microjoint') for s in states]
    opening = summary['motion'] == 'open'
    recomputed = [(q < -1.3 if opening else q > -.005) for q in values]
    if any(flag != a['evaluation_task_success'] for flag, a in zip(recomputed, actions)):
        raise ValueError('Native microwave goal disagrees with post-run state replay')
    audit = {
        'all_approach_and_closure_steps_executed': complete,
        'baseline_closure_control_step': closure_step,
        'maximum_prefix_joint_state_difference_rad': joint_error,
        'maximum_prefix_pose_command_difference': target_error,
        'native_initial_joint_rad': values[0], 'native_final_joint_rad': values[-1],
        'native_final_open_angle_deg': float(abs(np.degrees(values[-1]))),
        'native_success_ever': any(recomputed), 'native_success_final': recomputed[-1],
        'native_predicate_replay_agrees': True,
        'physical_attachment_injected': False,
    }
    save(folder/'paired-audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/microwave-hinge-diagnosis')
    parser.add_argument('--reference', default='results/articulated-physics-pilot')
    args = parser.parse_args()
    output, reference = ROOT/args.output, ROOT/args.reference
    output.mkdir(parents=True, exist_ok=False)
    config = read(ROOT/'configs/articulated_libero.json')
    config['scene_mode'] = 'diagnostic'
    tasks = [r for r in config['tasks'] if r['label'] in {'open_microwave', 'close_microwave'}]
    planned = [{'task': t['label'], 'name': t['name'], 'state': 1, 'hinge_mode': mode}
               for t in tasks for mode in ['estimated', 'oracle']]
    save(output/'manifest.json', {'planned_episodes': planned, 'all_in_denominator': True,
         'seed': 0, 'maximum_control_steps': 600, 'benchmark_result': False})
    save(output/'configuration.json', config)
    frozen = read(reference/'protocol.json')['sources_sha256']
    if any(digest(ROOT/p) != sha for p, sha in frozen.items()):
        raise ValueError('The original execution sources changed')
    sources = {**frozen, str(Path(__file__).relative_to(ROOT)): digest(__file__)}
    for path in sources:
        destination = output/'source-snapshot'/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, destination)
    ledger = ROOT/'results/articulated-api-v1/ledger.json'
    ledger_before = digest(ledger)
    save(output/'protocol.json', {
        'sources_sha256': sources, 'reference': str(reference.relative_to(ROOT)),
        'purpose': 'isolate hinge geometry while fixing original grasps, postures, and approach',
        'oracle_fields': ['joint_axis_world', 'joint_origin_world_m'],
        'oracle_progress_used': False, 'physical_contacts_enabled': True,
        'assisted_grasping': False, 'scene_mode': 'diagnostic',
        'all_original_tracking_and_gripper_gates_retained': True,
        'api_ledger_sha256_before': ledger_before, 'new_api_calls': 0,
        'new_graspgen_inference_calls': 0, 'benchmark_result': False,
    })
    perception = Perception()
    results = []
    for planned_episode in planned:
        task, mode = planned_episode['task'], planned_episode['hinge_mode']
        folder = output/mode/f'{task}-state001'
        baseline = reference/f'{task}-state001'
        with fixed_grasp_hooks(mode, baseline):
            summary = executor.run_episode(task, planned_episode['name'], 1, folder,
                config, perception, NoNewGrasps(), reuse_run=reference)
        summary.update(hinge_mode=mode, diagnostic_only=True,
                       simulator_scene_state_used=(mode == 'oracle'), oracle_progress_used=False,
                       fixed_baseline_grasp=True, new_api_calls=0)
        save(folder/'summary.json', summary)
        audit = episode_audit(folder, baseline)
        results.append({**summary, 'paired_audit': audit})
        if digest(ledger) != ledger_before:
            raise ValueError('API ledger changed during a cache-only diagnostic')
        save(output/'summary.json', {'planned_episodes': len(planned),
             'completed_episodes': len(results), 'episodes': results,
             'new_api_calls': 0, 'api_ledger_unchanged': True, 'benchmark_result': False})
        print(json.dumps({'task': task, 'hinge_mode': mode,
                         'native_success': summary['official_task_success'],
                         'failure_stage': summary['failure_stage'], 'paired_audit': audit}), flush=True)


if __name__ == '__main__':
    main()
