"""Compare contact rules on preserved successful grasps, without policy oracle inputs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.runtime/mc-vamp-deps')]
from try_rekep_bowl_v3 import read_observation_v3, save
from vlm_tamp.rekep_libero.collision_paths_v7 import estimated_contact_aperture
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.rekep_libero.scene_state_v7 import ObservedScene as SceneV7
from vlm_tamp.rekep_libero.scene_state_v8 import ObservedScene as SceneV8
from vlm_tamp.rekep_libero.sim_client_v8 import SimClientV8
from vlm_tamp.rekep_libero.task_ik_v6 import TaskIK


def read(path):
    return json.loads(path.read_text())


def evaluation_contacts(path):
    """Offline explanation only. Never used to label regions or select poses."""
    data = read(path)
    if not data['evaluation_only'] or not data['excluded_from_policy_input']:
        raise ValueError('Contact diagnostic must be evaluation-only')
    counts, forces = Counter(), {}
    for step in data['steps']:
        for item in step['robot_involved_contacts']:
            pair = ' / '.join(item['geom_names'])
            counts[pair] += 1
            forces[pair] = max(forces.get(pair, 0.), float(item['normal_force_n']))
    return [{'pair': key, 'recorded_contacts': count, 'peak_normal_force_n': forces[key]}
            for key, count in counts.most_common()]


def audit(task, state_index, output, config):
    kind = 'bowl' if task == 8 else 'drawer'
    previous = ROOT / f'results/physical-tasks-v6-eval/{kind}/goal{task:02d}-state{state_index:03d}'
    source = ROOT / f'results/physical-tasks-v7.2-eval/goal{task:02d}-state{state_index:03d}'
    metadata = read(source / 'opened/metadata.json')
    cameras, _ = read_observation_v3(source / 'opened', metadata)
    spec = read(source / 'task-spec.json')
    cloud = np.load(source / 'grasp-cloud.npy', allow_pickle=False)
    parent = np.load(source / 'parent-mask.npy', allow_pickle=False) if task == 0 else None
    target = np.load(source / 'target-cloud.npy', allow_pickle=False) if task == 8 else None
    args = (cameras, metadata, spec, cloud, np.load(source / 'source-mask.npy'),
            'agentview' if task == 8 else 'frontview', parent, target)
    scenes = {'v7': SceneV7(*args, config), 'v8': SceneV8(*args, config)}
    selected = read(previous / 'chosen-grasp.json')
    sim = SimClientV8(log_path=output / f'{kind}-simulator.log')
    rows = {}
    try:
        sim.reset(task, state_index, output / f'{kind}-robot-model', seed=0)
        import vamp
        environment = vamp.Environment()
        ik = TaskIK(read(source / 'robot-chain.json'), config,
                    lambda q: vamp.panda.validate(np.asarray(q).tolist(), environment, True))
        grasp = ik.solve(selected['tcp_pose_world_xyzw'], metadata['joint_positions'])
        if grasp is None:
            raise ValueError('Preserved successful grasp has no nominal IK solution')
        aperture = estimated_contact_aperture(sim, grasp['joints'], scenes['v8'].region.contact_reference,
                                             metadata['gripper_qpos'])
        for phase, qpos in [('open', metadata['gripper_qpos']), ('predicted_closed', aperture)]:
            if qpos is not None:
                rows[phase] = {name: scene.joints(sim, grasp['joints'], gripper_qpos=qpos, stage='grasp')
                               for name, scene in scenes.items()}
        closed = read(previous / 'closed/metadata.json')
        transform = np.asarray(read(previous / 'closure-surface-registration.json')['transform_initial_to_closed_world'])
        if task == 0:
            centre = cloud.mean(0)
            offset = float((transform_points([centre], transform)[0] - centre) @ spec['pull_axis_world'])
            transform = scenes['v8'].drawer_transform(offset)
        rows['actual_closed_robot_on_observed_scene'] = {
            name: scene.check(closed['robot_collision_volumes_world'], closed['eef_pose_world_xyzw'],
                              stage='grasp', moving_transform=transform, check_payload=True)
            for name, scene in scenes.items()}
    finally:
        sim.close()
    result = {'task_index': task, 'initial_state': state_index,
        'scope': 'offline diagnostic at a previously successful pose; not a new episode or selection override',
        'previous_native_success': read(previous / 'summary.json')['official_task_success'],
        'known_success_pose_for_diagnostic_only': {'index': selected['index'], 'variant': selected['variant']},
        'collision_geometry_inputs': 'synchronized RGB-D, cached predicted masks, known robot geometry and proprioception',
        'simulator_scene_identity_used_by_checker': False,
        'contact_diagnostics_for_explanation_only': evaluation_contacts(previous / 'initial/evaluation_contact_diagnostics.json'),
        'checks': rows, 'new_api_calls': 0}
    save(output / f'{kind}-audit.json', result)
    print(json.dumps({'task': kind, 'checks': {phase: {version: {
        'accepted': value['accepted'], 'collisions': value['collisions']}
        for version, value in checks.items()} for phase, checks in rows.items()}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/contact-policy-v8-development/successful-contact-audit')
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve audit outputs')
    output.mkdir(parents=True)
    config = read(ROOT / 'configs/physical_tasks_v8.json')
    results = [audit(task, 11, output, config) for task in [8, 0]]
    save(output / 'summary.json', {'scope': 'offline contact correction audit', 'new_api_calls': 0,
         'evaluation_scene_ids_excluded_from_policy': True, 'tasks': results})


if __name__ == '__main__':
    main()
