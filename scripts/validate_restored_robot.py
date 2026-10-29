"""Small server interface checks, excluded from manipulation success rates."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.runtime/mc-vamp-deps')]
from vlm_tamp.coupled_scene import Scene
from vlm_tamp.coupled_storage import read_json, write_json, compact_episode
from vlm_tamp.hinged_contact_tools import NativeScene, clear_native_cache
from vlm_tamp.partial_pose_arc import PartialPoseArc
from vlm_tamp.profiled_sim_client import ProfiledClient
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from replay_subscription_episode import compress_observations


def validate(fixture, output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = read_json(fixture / 'opened/metadata.json')
    record = read_json(fixture / 'robot-chain.json')
    chain = RobotChain(record)
    q = np.asarray(metadata['joint_positions'])
    cameras = []
    with np.load(fixture / 'opened/obs.npz') as data:
        for name, cal in metadata['cameras'].items():
            rgb, depth, points = [data[name + suffix] for suffix in ('_rgb', '_depth', '_world_points')]
            transform = np.asarray(cal['camera_to_world'])
            optical = (points - transform[:3, 3]) @ transform[:3, :3]
            projected = optical @ np.asarray(cal['intrinsic']).T
            uv = projected[..., :2] / projected[..., 2:]
            yy, xx = np.indices(depth.shape)
            valid = np.isfinite(depth) & (depth > cal['near_m']) & (depth < cal['far_m'])
            r = {'camera': name, 'valid_pixels': int(valid.sum()), 'rgb_std': float(rgb.std()),
                 'depth_z_error_m': float(np.max(np.abs(optical[..., 2][valid] - depth[valid]))),
                 'depth_z_scaled_error': float(np.max(np.abs(optical[..., 2][valid] - depth[valid]) / np.maximum(1., depth[valid]))),
                 'pixel_roundtrip_error': float(np.max(np.abs(uv[valid] - np.stack([xx, yy], -1)[valid]))),
                 'capture_control_steps': cal['capture_control_steps']}
            assert r['valid_pixels'] > 100 and r['rgb_std'] > 1
            # World arrays are float32; background depth can exceed 400 m.
            # Bound error in units of depth instead of treating sky pixels as
            # close-range manipulation surfaces. Keep the pixel check separate.
            assert r['depth_z_scaled_error'] < 8 * np.finfo(np.float32).eps
            assert r['pixel_roundtrip_error'] < .01
            assert r['capture_control_steps'] == metadata['control_steps']
            cameras.append(r)
    write_json(output / 'rgbd-conventions.json', cameras)
    import vamp
    checks = []
    with ProfiledClient(output / 'simulator.log', timeout=300) as sim:
        sim.request('reset', task_name='KITCHEN_SCENE7_open_the_microwave', state_index=1,
                    output_dir=str(output / 'initial'), seed=0, suite_name='libero_90',
                    scene_variant='nominal', stop_on_success=True)
        robot_world = np.linalg.inv(np.asarray(record['calibration']['world_to_robot']))
        eef_tcp = np.linalg.inv(np.asarray(vamp.panda.eefk(chain.reference_joints))[0]) @ np.linalg.inv(robot_world) @ chain.reference_tcp
        low, high = chain.limits.T
        samples = np.vstack([q, np.random.default_rng(0).uniform(low + .1, high - .1, (24, 7))])
        for joints in samples:
            isolated = sim.robot_jacobian(joints)
            expected = pose_matrix(isolated['eef_pose_world_xyzw'])
            pose, jacobian = chain.forward(joints, jacobian=True)
            predicted = pose_matrix(pose)
            native = robot_world @ np.asarray(vamp.panda.eefk(joints))[0] @ eef_tcp
            checks.append({
                'poe_position_error_m': float(np.linalg.norm(expected[:3, 3] - predicted[:3, 3])),
                'poe_rotation_error_rad': float(Rotation.from_matrix(expected[:3, :3].T @ predicted[:3, :3]).magnitude()),
                'jacobian_absolute_error': float(np.max(np.abs(jacobian - isolated['pose_jacobian']))),
                'native_position_error_m': float(np.linalg.norm(expected[:3, 3] - native[:3, 3])),
                'native_rotation_error_rad': float(Rotation.from_matrix(expected[:3, :3].T @ native[:3, :3]).magnitude()),
            })
    maxima = {k: max(r[k] for r in checks) for k in checks[0]}
    assert maxima['poe_position_error_m'] < 1e-8 and maxima['jacobian_absolute_error'] < 1e-8
    assert maxima['native_position_error_m'] < .001 and maxima['native_rotation_error_rad'] < .005
    write_json(output / 'kinematic-checks.json', {'configurations': 25, 'maxima': maxima, 'checks': checks})
    pad = next(v for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name'].endswith('finger1_pad_collision'))
    point = np.asarray(pad['vertices_world']).mean(0)
    spec = {'joint_origin_world_m': (point + [-.25, 0, 0]).tolist(), 'joint_axis_world': [0, 0, 1],
            'panel_normal_world': [0, 1, 0], 'front_plane': {'point_world_m': (point + [0, -.1, 0]).tolist()}}
    allowed = Scene.from_points(np.empty((0, 3)), [point], [point], spec)
    blocked = Scene.from_points([point], [point], [point], spec)
    modes = []
    for mode in ('point', 'rigid', 'jaw_axis'):
        with NativeScene(allowed, record, metadata, q, .02, articulated=False, grasp_constraint=mode) as context:
            assert context.validate(q)['accepted']
            path, free = context.plan(q, q + [.01, 0, 0, 0, 0, 0, 0], constrained=False, seconds=3.)
            assert free['exact_solution'] and len(path) > 1
        with NativeScene(blocked, record, metadata, q, .02, articulated=False, grasp_constraint=mode) as context:
            rejection = context.validate(q)
            assert not rejection['accepted'] and rejection['last_rejected_surface'] == 0
        with NativeScene(allowed, record, metadata, q, .02, grasp_constraint=mode) as context:
            model = PartialPoseArc(chain, spec['joint_origin_world_m'], spec['joint_axis_world'], q,
                                   context.pad_local, grasp_constraint=mode)
            goal, projected = model.project(q, .02)
            assert projected['accepted']
            path, arc = context.plan(q, goal, seconds=3.)
            assert arc['exact_solution'] and len(path) > 1
            residual = max(float(np.linalg.norm(model.residual(v, context.progress(v))[:3])) for v in path)
            assert residual < .001
        modes.append({'mode': mode, 'bounded_finger_contact_allowed': True,
                      'stationary_collision_rejected': True, 'free_plan': free['status'],
                      'arc_plan': arc['status'], 'arc_maximum_position_residual_m': residual})
    clear_native_cache()
    result = {'scope': 'development interface smoke with explicit synthetic planning geometry; not task success',
              'cameras': len(cameras), 'kinematics': maxima, 'planning_modes': modes}
    write_json(output / 'summary.json', result)
    actions_file = output / 'initial/actions.json'
    if actions_file.exists() and read_json(actions_file):
        result['storage'] = compact_episode(output)
    else:
        result['storage'] = {'zero_control_step_session': True,
                             'initial_frame_retained': True, 'movie_not_required': True}
    result['lossless_sensor_compression'] = compress_observations(output)
    write_json(output / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    validate(args.fixture.resolve(), args.output.resolve())
