"""Scientific figure from retained evaluator-only contact-prefix evidence."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from analyze_contact_prefix import load_cameras
from vlm_tamp.coupled_scene import RobotVolumes
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.rekep_libero.geometry import pose_matrix, transform_points
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain


def main(root):
    analysis = read_json(root / 'probes/event018/point-analysis.json')
    points = np.asarray(analysis['world_points'])
    native = read_json(root / 'evaluator-handle-provenance.json')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, event, step in zip(axes[0], ['event009', 'event016'], [259, 275]):
        view = load_cameras(root / 'snapshots' / event)['arm_top']
        optical = transform_points(points, np.linalg.inv(pose_matrix(view['camera_to_world'])))
        uv = optical @ np.asarray(view['intrinsic']).T
        uv = uv[:, :2] / uv[:, 2:]
        ax.imshow(view['rgb'])
        ax.scatter(uv[:, 0], uv[:, 1], s=90, facecolors='none', edgecolors='#e13535', linewidths=1.8)
        for i, p in enumerate(uv):
            ax.annotate(str(i + 1), p, xytext=(6, -7), textcoords='offset points', color='#e13535', weight='bold')
        center = uv.mean(0)
        ax.set_xlim(center[0] - 35, center[0] + 35)
        ax.set_ylim(center[1] + 35, center[1] - 35)
        ax.set_title(f'Fresh top-view RGB, step {step}')
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[1, 0]
    for row, color, label in zip(native['steps'][-2:], ['#238d79', '#326cbd'],
                                 ['Handle at step 259', 'Handle at step 275']):
        centre = row['handle_center_world_m']
        ax.add_patch(plt.Circle(centre[:2], native['native_handle_radius_m'], edgecolor=color,
                               facecolor='none', lw=2, label=label))
    record = read_json(root / 'robot-chain.json')
    metadata = read_json(root / 'snapshots/event017/metadata.json')
    robot = RobotVolumes(RobotChain(record), metadata['robot_collision_volumes_world'], metadata['joint_positions'])
    for q, style, label in [(metadata['joint_positions'], '--', 'Current finger 1'),
                            (analysis['checks'][0]['joints'], '-', 'Rejected finger 1')]:
        volume = next(v for v in robot.forward(q) if v['robot_geom_name'] == 'gripper0_finger1_collision')
        xy = np.asarray(volume['vertices_world'])[:, :2]
        hull = ConvexHull(xy).vertices
        outline = xy[np.r_[hull, hull[0]]]
        ax.plot(outline[:, 0], outline[:, 1], style, color='#444444', lw=1.4, label=label)
    ax.scatter(points[:, 0], points[:, 1], color='#e13535', s=35, zorder=5, label='Retained stationary points')
    ax.set_aspect('equal')
    ax.set_xlabel('World x (m)')
    ax.set_ylabel('World y (m)')
    ax.set_title('Top-down geometry near the handle')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, 1))
    ax.grid(alpha=.15)
    ax = axes[1, 1]
    contacts = read_json(root / 'attempt01/initial/evaluation_contact_diagnostics.json')['steps']
    angles = [-np.degrees(next(j['position'] for j in r['evaluation_only_articulation_state']
                             if j['joint_name'] == 'microwave_1_microjoint')) for r in contacts]
    ax.plot(np.arange(1, len(angles) + 1), angles, color='#326cbd', label='Native door opening')
    steps, model = [], []
    first_normal = None
    for event in ['event003', 'event005', 'event006', 'event009', 'event011', 'event016']:
        obs = read_json(root / 'snapshots' / event / 'observation.json')
        normal = np.asarray(obs['spec']['panel_normal_world'])
        if first_normal is None:
            first_normal = normal
        steps.append(obs['control_steps'])
        model.append(-np.degrees(np.arctan2(np.cross(first_normal, normal)[2], first_normal @ normal)))
    ax.step(steps, model, where='post', color='#238d79', label='Stored model rotation from grounding')
    ax.scatter(steps, model, color='#238d79', s=18)
    ax.axvline(259, color='#e13535', linestyle=':', label='Rebuild adds the stale points')
    ax.set_xlim(55, 280)
    ax.set_ylim(-1, 22)
    ax.set_xlabel('Control step')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('The model already lagged before the rebuild')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=.15)
    fig.suptitle('State 44: handle samples retained as stationary obstacles', fontsize=15)
    fig.supxlabel('Development replay with new RGB-D. Native handle geometry is evaluator-only; no planner inputs were changed.', fontsize=9)
    fig.savefig(root.parent / 'state044-stale-handle-points.png', dpi=180)
    fig.savefig(root.parent / 'state044-stale-handle-points.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
