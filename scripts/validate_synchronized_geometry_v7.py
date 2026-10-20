"""Validate synchronized RGB-D and isolated arm/gripper geometry in LIBERO."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from try_rekep_bowl_v3 import read_observation_v3, save
from vlm_tamp.rekep_libero.cache_evidence_v7 import cached_scene
from vlm_tamp.rekep_libero.scene_state_v7 import ObservedScene
from vlm_tamp.rekep_libero.sim_client_v7 import SimClientV7
from vlm_tamp.rekep_libero.task_geometry_v5 import compile_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/physical-tasks-v7-development/synchronized-camera-smoke')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'configs/physical_tasks_v7.json').read_text())
    sim = SimClientV7(log_path=output / 'simulator.log')
    try:
        state = sim.reset(8, 1, output / 'initial', seed=0)
        state = sim.action(state['metadata']['eef_pose_world_xyzw'], gripper=-1, max_steps=60, output_dir=output / 'opened')
        source = ROOT / 'results/rekep-libero-v2/libero_goal-08-state001-seed0'
        bundle, program, cloud, mask = cached_scene(source, output / 'opened', config)
        spec, grasp, target, _, masks = compile_scene(bundle, program, cloud, mask, config)
        cameras, _ = read_observation_v3(output / 'opened', state['metadata'])
        scene = ObservedScene(cameras, state['metadata'], spec, grasp, masks['source'],
                              bundle['main_camera'], masks.get('parent'), target, config)
        clearance = scene.check(state['metadata']['robot_collision_volumes_world'], state['metadata']['eef_pose_world_xyzw'])
        before = sim.fk(include_volumes=True)
        hypothetical = sim.fk(state['metadata']['joint_positions'], include_volumes=True, gripper_qpos=[.01, -.01])
        after = sim.fk(include_volumes=True)
        evidence = bundle['cache_evidence']
        camera_metadata = state['metadata']['cameras'].values()
        checks = {'initial_full_arm_clear': clearance['accepted'],
                  'cached_selected_rgbd_evidence_accepted': evidence['selected_evidence_accepted'],
                  'graspgen_input_cloud_identical': np.array_equal(grasp, np.load(ROOT / 'results/physical-tasks-v6-eval/bowl/goal08-state001/grasp-cloud.npy')),
                  'live_gripper_unchanged_by_hypothetical_fk': np.allclose(before['gripper_qpos'], after['gripper_qpos']),
                  'live_tcp_unchanged_by_hypothetical_fk': np.allclose(before['eef_pose_world_xyzw'], after['eef_pose_world_xyzw']),
                  'all_collision_parts_exported': len(hypothetical['robot_collision_volumes_world']) == 11,
                  'capture_policy_present': all('capture_policy' in camera for camera in camera_metadata),
                  'capture_times_agree': len({camera['capture_simulation_time_s'] for camera in camera_metadata}) == 1}
        save(output / 'result.json', {'checks': checks, 'clearance': clearance, 'scene': scene.record(), 'cache_evidence': evidence})
        print(json.dumps(checks), flush=True)
        if not all(checks.values()):
            raise AssertionError('Synchronized geometry validation failed')
    finally:
        closed = sim.close()
        print(json.dumps({key: closed.get(key) for key in ['control_steps', 'video_path']}), flush=True)


if __name__ == '__main__':
    main()
