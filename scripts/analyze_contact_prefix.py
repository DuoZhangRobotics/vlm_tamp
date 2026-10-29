"""Locate stationary endpoints inside rejected configured gripper volumes.

This is an offline evaluator. It never edits a scene, runs a policy, or changes
planner admission. The retained native float32 geometry is the source of truth.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import vamp

from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.scene_observation_diagnostics import depth_support


def load_cameras(snapshot):
    observation = read_json(snapshot / 'observation.json')
    workspace = Path(observation['public_workspace'])
    public = observation['observation']
    calibration = read_json(workspace / public['camera_calibration'])
    with np.load(workspace / public['sensor_arrays']) as arrays:
        cameras = {name: {**meta, **{key: arrays[name + '_' + key] for key in
                    ('rgb', 'depth', 'world_points', 'known_robot_depth_prior') if name + '_' + key in arrays}}
                   for name, meta in calibration.items()}
    return cameras


def pixel_provenance(points, cameras):
    """Nearest retained raw endpoint per view, with robot-depth residual."""
    candidates = [[] for _ in points]
    if not len(points):
        return candidates
    low, high = points.min(0) - .015, points.max(0) + .015
    for name, view in cameras.items():
        xyz = np.asarray(view['world_points']).reshape(-1, 3)
        near = np.flatnonzero(np.isfinite(xyz).all(1) & (xyz >= low).all(1) & (xyz <= high).all(1))
        if not len(near):
            continue
        distance, indices = cKDTree(xyz[near]).query(points)
        h, w = view['depth'].shape[:2]
        for i, (d, j) in enumerate(zip(distance, indices)):
            pixel = int(near[j])
            y, x = divmod(pixel, w)
            depth = float(np.asarray(view['depth']).reshape(-1)[pixel])
            prior = view.get('known_robot_depth_prior')
            prior = float(np.asarray(prior).reshape(-1)[pixel]) if prior is not None else None
            candidates[i].append({'camera': name, 'pixel_xy': [x, y], 'distance_m': float(d),
                                  'depth_m': depth, 'known_robot_depth_m': prior,
                                  'robot_depth_residual_m': depth - prior if prior is not None else None})
    return [sorted(rows, key=lambda r: r['distance_m'])[:3] for rows in candidates]


def analyze(root, name):
    folder = root / 'probes' / name
    native = read_json(folder / 'native-scene.json')
    trace = read_json(folder / 'validation-trace.json')
    record = read_json(root / 'robot-chain.json')
    world_robot = np.linalg.inv(np.asarray(record['calibration']['world_to_robot']))
    with np.load(folder / 'native-geometry.npz') as arrays:
        stationary = arrays['stationary']
        planes = arrays['gripper_planes']
        offsets = arrays['gripper_plane_offsets']
        bounds = arrays['gripper_bounds']
    points_world = transform_points(stationary, world_robot)
    tree = cKDTree(stationary)
    parts = {1000 + p['sphere_range'][0]: (i, p) for i, p in enumerate(native['gripper_cover'])}
    counts, matches, all_ids = Counter(), [], set()
    for index, check in enumerate(trace):
        result = check['result']
        if result['accepted']:
            continue
        part, surface = int(result['last_rejected_part']), int(result['last_rejected_surface'])
        counts[f'{part}/{surface}'] += 1
        if surface != 0 or part not in parts:
            continue
        which, description = parts[part]
        # This uses the same native FK and float32 convex planes as the checker.
        eef = np.asarray(vamp.panda.eefk(np.asarray(check['joints'], dtype=np.float32)), dtype=np.float32)[0]
        centre = eef[:3, :3] @ bounds[which, :3] + eef[:3, 3]
        ids = np.asarray(tree.query_ball_point(centre, float(bounds[which, 3]) + 1e-6), dtype=int)
        local = (stationary[ids] - eef[:3, 3]) @ eef[:3, :3]
        part_planes = planes[offsets[which]:offsets[which + 1]]
        distances = np.max(local @ part_planes[:, :3].T + part_planes[:, 3], axis=1)
        inside = distances <= native['gripper_convex_point_padding_m']
        hit_ids, penetration = ids[inside], -distances[inside]
        all_ids.update(int(i) for i in hit_ids)
        matches.append({'trace_index': index, 'part': part, 'robot_geom_name': description['robot_geom_name'],
                        'progress_degrees': float(np.degrees(check['progress_rad'])),
                        'joints': check['joints'], 'stationary_indices': hit_ids,
                        'halfspace_penetration_m': penetration,
                        'matching_point_count': len(hit_ids)})
    selected = np.asarray(sorted(all_ids), dtype=int)
    points = points_world[selected]
    np.savez_compressed(folder / 'rejected-stationary-points.npz', native_indices=selected,
                        world_points=points, native_points=stationary[selected])
    snapshots = sorted((root / 'snapshots').glob('event*'))
    evidence = []
    seen_observations = set()
    for snapshot in snapshots:
        reference = read_json(snapshot / 'scene-reference.json')
        obs = read_json(snapshot / 'observation.json')
        row = {'event': snapshot.name, 'control_steps': obs['control_steps'],
               'role_distances': {}, 'scene_record': reference['scene_record']}
        with np.load(reference['arrays_path']) as arrays:
            for role in ('static_points', 'moving_reference', 'contact_reference'):
                distances = cKDTree(arrays[role]).query(points)[0] if len(points) else np.empty(0)
                row['role_distances'][role] = distances
        obs_id = obs['observation']['observation_id']
        if obs_id not in seen_observations and len(points):
            seen_observations.add(obs_id)
            cameras = load_cameras(snapshot)
            support, _, statuses = depth_support(points, cameras, maximum_points=max(1200, len(points)))
            row.update(depth_support=support, depth_status=statuses,
                       raw_pixel_provenance=pixel_provenance(points, cameras), observation_id=obs_id)
        evidence.append(row)
    report = {'probe': name, 'scope': 'offline evaluator; stationary points are not removed or retagged',
              'native_rejections_by_part_surface': dict(counts),
              'native_stationary_count': len(stationary), 'unique_rejected_stationary_points': len(selected),
              'world_points': points, 'native_stationary_indices': selected,
              'checks': matches, 'checks_without_recovered_points': sum(not r['matching_point_count'] for r in matches),
              'snapshots': evidence, 'planner_result': read_json(folder / 'result.json')['status']}
    write_json(folder / 'point-analysis.json', report)
    print({'probe': name, 'result': report['planner_result'], 'rejections': dict(counts),
           'unique_stationary_hits': len(selected), 'unrecovered_checks': report['checks_without_recovered_points'],
           'world_bounds': [points.min(0).tolist(), points.max(0).tolist()] if len(points) else None}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--probe', required=True)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.probe)
