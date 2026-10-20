"""Validate scene updates on preserved RGB-D drawer motion, without new actions."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from try_rekep_bowl_v3 import read_observation_v3, save
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.rekep_libero.scene_state_v7 import ObservedScene


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='results/physical-tasks-v7-development/pilot-03/goal00-state011')
    parser.add_argument('--trajectory', default='results/physical-tasks-v6-eval/drawer/goal00-state011')
    parser.add_argument('--output', default='results/physical-tasks-v7-development/moving-drawer-replay')
    args = parser.parse_args()
    source, trajectory, output = ROOT / args.scene, ROOT / args.trajectory, ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve existing replays')
    output.mkdir(parents=True)
    metadata = read(source / 'opened/metadata.json')
    cameras, _ = read_observation_v3(source / 'opened', metadata)
    spec = read(source / 'task-spec.json')
    reference = np.load(source / 'grasp-cloud.npy', allow_pickle=False)
    scene = ObservedScene(cameras, metadata, spec, reference, np.load(source / 'source-mask.npy'),
                          'frontview', np.load(source / 'parent-mask.npy'))
    centre = reference.mean(0)
    closure = read(trajectory / 'closure-surface-registration.json')['transform_initial_to_closed_world']
    offset = float((transform_points([centre], closure)[0] - centre) @ spec['pull_axis_world'])
    rows = []
    for index, feedback in enumerate(read(trajectory / 'drawer-feedback.json')):
        folder = 'probe-held' if index == 0 else f'pull-observation-{index - 1:02d}'
        metadata = read(trajectory / folder / 'metadata.json')
        cameras, _ = read_observation_v3(trajectory / folder, metadata)
        progress = offset + feedback['observed_progress_m']
        transform = scene.drawer_transform(progress)
        update = scene.refresh(cameras, metadata, transform)
        clearance = scene.check(metadata['robot_collision_volumes_world'], metadata['eef_pose_world_xyzw'],
                                stage='pull', moving_transform=transform, check_payload=True)
        rows.append({'observation': folder, 'observed_progress_m': progress, 'control_steps': feedback['control_steps'],
                     'original_tracking_valid': feedback['tracking_valid'], 'update': update, 'clearance': clearance})
        scene.save(output / f'scene-{index:02d}.npz')
    result = {'scope': 'new geometry checker on preserved v6 RGB-D motion; not new physical execution',
              'new_control_actions': 0, 'new_api_calls': 0, 'simulator_object_state_used': False,
              'observed_scene_updates': len(rows), 'all_updates_supported': all(row['update']['current_moving_supported_points'] >= 75 for row in rows),
              'maximum_observed_drawer_displacement_m': max(row['observed_progress_m'] for row in rows),
              'full_arm_clearance_passes': sum(row['clearance']['accepted'] for row in rows), 'rows': rows}
    save(output / 'summary.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))
    if not result['all_updates_supported']:
        raise ValueError('Observed drawer scene replay lost moving-panel support')


if __name__ == '__main__':
    main()
