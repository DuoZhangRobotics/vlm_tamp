"""Replay rejected configurations and build an RGB/3-D collision diagnostic.

Scene inputs are preserved RGB-D and predicted masks. Hypothetical robot FK
uses configured Panda geometry only. No rejected motion is physically executed.
"""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.runtime/mc-vamp-deps')]
from try_rekep_bowl_v3 import read_observation_v3, save
from vlm_tamp.rekep_libero.collision_paths_v7 import check_joint_path, drawer_transform, estimated_contact_aperture
from vlm_tamp.rekep_libero.geometry import pose_matrix, transform_points
from vlm_tamp.rekep_libero.scene_state_v7 import bounded_points
from vlm_tamp.rekep_libero.scene_state_v11 import ObservedScene
from vlm_tamp.rekep_libero.sim_client_v8 import SimClientV8
from vlm_tamp.rekep_libero.task_ik_v5 import manipulation_poses
from vlm_tamp.rekep_libero.task_ik_v6 import TaskIK


def read(path):
    return json.loads(path.read_text())


def load_scene(folder, config):
    metadata = read(folder / 'opened/metadata.json')
    cameras, _ = read_observation_v3(folder / 'opened', metadata)
    spec = read(folder / 'task-spec.json')
    scene = ObservedScene(cameras, metadata, spec, np.load(folder / 'grasp-cloud.npy'),
        np.load(folder / 'source-mask.npy'), 'frontview', np.load(folder / 'parent-mask.npy'), config=config)
    return scene, cameras, metadata, spec


def make_ik(folder, config):
    import vamp
    env = vamp.Environment()
    return TaskIK(read(folder / 'robot-chain.json'), config,
        lambda q: vamp.panda.validate(np.asarray(q).tolist(), env, True))


def replay(folder, row, sim, config, phase):
    scene, cameras, metadata, spec = load_scene(folder, config)
    ik = make_ik(folder, config)
    initial = metadata['joint_positions']
    grasp = ik.solve(row['tcp_pose_world_xyzw'], initial)
    if grasp is None:
        raise ValueError('Preserved rejected pose cannot be reproduced with the original IK seeds')
    target = grasp['pose_world_xyzw']
    open_q = metadata['gripper_qpos']
    final_geometry = sim.fk(grasp['joints'], include_volumes=True, gripper_qpos=open_q)
    final_clearance = scene.check(final_geometry['robot_collision_volumes_world'], target, stage='grasp')
    moving = np.eye(4)
    stage = 'grasp'
    if phase == 'grasp':
        q, qpos = np.asarray(grasp['joints']), open_q
        original = row['clearance']
    else:
        saved = row['path']
        if phase == 'approach':
            approach = ik.approach(np.asarray(initial), grasp)
            if not approach['accepted']:
                raise ValueError('Preserved approach does not reproduce')
            joint_path, qpos = np.asarray(approach['joint_postures']), open_q
            stage = 'approach'
        else:
            future = manipulation_poses(spec, target, config)
            pull = ik.path(future, grasp['joints'])
            if not pull['accepted']:
                raise ValueError('Preserved pull does not reproduce')
            joint_path = np.asarray([s['joints'] for s in pull['solutions']])
            qpos = estimated_contact_aperture(sim, grasp['joints'], scene.region.contact_reference, open_q)
            stage = 'pull'
        edge, weight = saved['edge'], saved['fraction']
        q = joint_path[edge] + weight * (joint_path[min(edge + 1, len(joint_path) - 1)] - joint_path[edge])
        original = saved['clearance']
        if phase == 'pull':
            moving = drawer_transform(scene, target)(ik.chain.forward(q))
    geometry = sim.fk(q.tolist(), include_volumes=True, gripper_qpos=qpos)
    geometry['hypothetical_arm_joints'] = q.tolist()
    result = scene.check(geometry['robot_collision_volumes_world'], geometry['eef_pose_world_xyzw'],
                          stage=stage, moving_transform=moving, check_payload=phase == 'pull')
    old_pairs = {(c['robot_geom_name'], c['scene_part']): c for c in original['collisions']}
    new_pairs = {(c['robot_geom_name'], c['scene_part']): c for c in result['collisions']}
    if old_pairs.keys() != new_pairs.keys():
        raise ValueError('Replay collision pairs differ from the preserved rejection')
    for pair in old_pairs:
        old, new = old_pairs[pair], new_pairs[pair]
        if old['blocked_observed_points'] != new['blocked_observed_points'] or abs(
                old['minimum_signed_point_distance_m'] - new['minimum_signed_point_distance_m']) > 1e-7:
            raise ValueError('Replay geometry does not match the recorded rejection')
    return scene, cameras, metadata, spec, ik, grasp, final_geometry, final_clearance, geometry, result, moving


def collision_points(scene, geometry, result, transform):
    moving = transform_points(scene.moving_reference, transform)
    contacts = transform_points(scene.region.contact_reference, transform)
    volumes = {v['robot_geom_name']: v for v in geometry['robot_collision_volumes_world']}
    rows = []
    for collision in result['collisions']:
        volume = volumes[collision['robot_geom_name']]
        points = scene.static_points if collision['scene_part'] == 'stationary_scene' else moving
        margin = collision['required_clearance_m']
        indices, distance = bounded_points(points, cKDTree(points), volume, margin)
        hit = points[indices]
        allowed = (scene.policy.robot_allowed(hit, volume, geometry['eef_pose_world_xyzw'], contacts, result['stage'])
                   if collision['scene_part'] == 'moving_source' else np.zeros(len(hit), bool))
        allowed |= scene.policy.local_panel_allowed(hit, volume, contacts, result['stage'], transform)
        selected = (distance <= margin) & ~allowed
        if int(selected.sum()) != collision['blocked_observed_points']:
            raise ValueError('Visualization point selection differs from checker')
        rows.append({'robot_part': collision['robot_geom_name'], 'scene_part': collision['scene_part'],
                     'points': hit[selected], 'signed_distances': distance[selected], 'required_clearance_m': margin})
    return rows


def sample_cloud(points, maximum):
    points = np.asarray(points)
    return points[::max(1, int(np.ceil(len(points) / maximum)))]


def triangle_geometry(volume):
    vertices = np.asarray(volume['vertices_world'])
    return vertices, ConvexHull(vertices).simplices


def add_robot(figure, geometry, blocked, *, opacity=.45):
    import plotly.graph_objects as go
    for volume in geometry['robot_collision_volumes_world']:
        vertices, triangles = triangle_geometry(volume)
        name = volume['robot_geom_name']
        color = blocked.get(name, '#228fbc' if name.startswith('gripper') else '#637585')
        figure.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], color=color,
            opacity=.65 if name in blocked else opacity, name=name, hoverinfo='name',
            showlegend=False, flatshading=True))


def figure_3d(scene, geometry, rows, transform, title, *, zoom=False, centre=None):
    import plotly.graph_objects as go
    figure = go.Figure()
    static, moving = scene.static_points, transform_points(scene.moving_reference, transform)
    if zoom:
        centre = np.asarray(centre)
        static = static[np.linalg.norm(static - centre, axis=1) < .08]
        moving = moving[np.linalg.norm(moving - centre, axis=1) < .12]
    else:
        static = static[(static[:, 0] > -.75) & (static[:, 0] < .6) & (static[:, 1] > -.6)
                        & (static[:, 1] < .6) & (static[:, 2] < 1.65)]
    for name, points, color in [('Other observed surfaces', static, '#99a6b1'), ('Selected moving drawer', moving, '#16966d')]:
        points = sample_cloud(points, 15000 if not zoom else 6000)
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
            marker={'size': 1.3 if not zoom else 2., 'color': color, 'opacity': .55}, name=name))
    blocked = {}
    for row in rows:
        penetrate = row['signed_distances'] < 0
        blocked[row['robot_part']] = '#bd343a' if penetrate.any() else '#d79224'
        for mask, color, label in [(penetrate, '#db253c', 'Inside robot volume'), (~penetrate, '#e1a027', 'Positive gap, buffer rejection')]:
            p, d = row['points'][mask], row['signed_distances'][mask] * 1000
            if not len(p):
                continue
            figure.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
                marker={'size': 4.5 if zoom else 3., 'color': color}, name=f'{label}: {row["robot_part"]}',
                customdata=d, hovertemplate='%{customdata:.4f} mm signed gap<extra>%{fullData.name}</extra>'))
    add_robot(figure, geometry, blocked)
    scene_options = {'aspectmode': 'data', 'xaxis_title': 'World x (m)', 'yaxis_title': 'World y (m)',
                     'zaxis_title': 'Height (m)', 'camera': {'eye': {'x': 1.4, 'y': 1.6, 'z': 1.0}}}
    if zoom:
        for index, axis in enumerate(['xaxis', 'yaxis', 'zaxis']):
            scene_options[axis] = {'range': [centre[index] - .055, centre[index] + .055], 'title': ['x (m)', 'y (m)', 'Height (m)'][index]}
    figure.update_layout(title=title, scene=scene_options, height=530, margin={'l': 0, 'r': 0, 'b': 0, 't': 40},
                         legend={'orientation': 'h', 'y': -.10}, paper_bgcolor='white')
    return figure


def project(points, camera):
    points = transform_points(points, np.linalg.inv(pose_matrix(camera['camera_to_world'])))
    pixels = points @ np.asarray(camera['intrinsic']).T
    return pixels[:, :2] / points[:, 2, None], points[:, 2]


def static_figure(destination, title, scene, cameras, geometry, rows, transform, final_geometry=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    points = np.vstack([row['points'] for row in rows])
    centre = points.mean(0)
    colors = {r['robot_part']: '#c1333e' if (r['signed_distances'] < 0).any() else '#d18a18' for r in rows}
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    ax = fig.add_subplot(221)
    camera = cameras['frontview']
    ax.imshow(camera['rgb'])
    for volume in geometry['robot_collision_volumes_world']:
        name = volume['robot_geom_name']
        if not name.startswith('gripper') and name not in colors:
            continue
        vertices, triangles = triangle_geometry(volume)
        pixel, depth = project(vertices, camera)
        edges = np.unique(np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0)
        edges = edges[(depth[edges] > 0).all(1)]
        ax.add_collection(LineCollection(pixel[edges], colors=colors.get(name, '#248db4'), linewidths=.6, alpha=.5))
    for row in rows:
        pixels, _ = project(row['points'], camera)
        ax.scatter(pixels[:, 0], pixels[:, 1], c=colors[row['robot_part']], s=10)
    ax.set(xlim=(0, camera['rgb'].shape[1]), ylim=(camera['rgb'].shape[0], 0),
           title='Initial scene RGB + hypothetical robot wireframe')
    ax.axis('off')
    ax3 = fig.add_subplot(222, projection='3d')
    static = sample_cloud(scene.static_points, 22000)
    selected = sample_cloud(transform_points(scene.moving_reference, transform), 7000)
    ax3.scatter(*static.T, c='#a9b1bb', s=.3, alpha=.17)
    ax3.scatter(*selected.T, c='#178863', s=.6, alpha=.6)
    for volume in geometry['robot_collision_volumes_world']:
        vertices, triangles = triangle_geometry(volume)
        name = volume['robot_geom_name']
        ax3.add_collection3d(Poly3DCollection(vertices[triangles], alpha=.38, linewidth=.08,
            facecolor=colors.get(name, '#2685a9' if name.startswith('gripper') else '#617486')))
    ax3.scatter(*points.T, c='#d73545' if any((r['signed_distances'] < 0).any() for r in rows) else '#d99722', s=12)
    ax3.set(xlim=(-.55, .45), ylim=(-.5, .40), zlim=(.85, 1.65), xlabel='x (m)', ylabel='y (m)', zlabel='Height (m)',
            title='Configured robot volumes and observed surfaces')
    ax3.set_box_aspect([1, .9, .8])
    ax3.view_init(elev=21, azim=55)
    for number, dims, labels in [(223, (0, 2), ('x (m)', 'Height (m)')), (224, (1, 2), ('y (m)', 'Height (m)'))]:
        axis = fig.add_subplot(number)
        near = scene.static_points[np.linalg.norm(scene.static_points - centre, axis=1) < .07]
        axis.scatter(near[:, dims[0]], near[:, dims[1]], c='#adb5bd', s=2, alpha=.35)
        names = {r['robot_part'] for r in rows}
        for volume in geometry['robot_collision_volumes_world']:
            if volume['robot_geom_name'] not in names:
                continue
            vertices, _ = triangle_geometry(volume)
            hull = ConvexHull(vertices[:, dims])
            polygon = vertices[hull.vertices][:, dims]
            axis.fill(polygon[:, 0], polygon[:, 1], color=colors[volume['robot_geom_name']], alpha=.18)
            axis.plot(*np.vstack([polygon, polygon[:1]]).T, color=colors[volume['robot_geom_name']], linewidth=1)
        for row in rows:
            axis.scatter(row['points'][:, dims[0]], row['points'][:, dims[1]], c=colors[row['robot_part']], s=18)
        axis.set(xlabel=labels[0], ylabel=labels[1], xlim=(centre[dims[0]]-.025, centre[dims[0]]+.025),
                 ylim=(centre[dims[1]]-.025, centre[dims[1]]+.025), title='Contact neighborhood — 5 cm window')
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.2)
    fig.suptitle(title, fontsize=15)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def multiseed_probe(folder, scene, sim, metadata, spec, ik, grasp, output, *, seeds=32):
    rng = np.random.default_rng(11)
    low, high = ik.chain.limits.T
    initial, reference = np.asarray(metadata['joint_positions']), np.asarray(grasp['joints'])
    starts = [initial, reference, ik.chain.reference_joints]
    starts += [np.clip(reference + rng.normal(0, .7, 7), low+.0001, high-.0001) for _ in range(seeds-7)]
    starts += [rng.uniform(low+.01, high-.01) for _ in range(4)]
    rows, distinct = [], []
    for number, start in enumerate(starts):
        solution = ik.solve(grasp['pose_world_xyzw'], start, seed_count=1)
        if solution is None:
            continue
        q = np.asarray(solution['joints'])
        if any(np.max(np.abs(q - previous)) < .05 for previous in distinct):
            continue
        distinct.append(q)
        row = {'seed_index': number, 'solution': solution,
               'minimum_joint_margin_rad': float(np.minimum(q-low, high-q).min()),
               'joint_distance_from_start_rad': float(np.linalg.norm(q-initial))}
        endpoint = scene.joints(sim, q, gripper_qpos=metadata['gripper_qpos'], stage='grasp')
        row['endpoint_accepted'] = endpoint['accepted']
        row['endpoint_collisions'] = endpoint['collisions']
        row['full_approach_and_pull_accepted'] = False
        if endpoint['accepted']:
            approach = ik.approach(initial, solution)
            row['approach_ik_accepted'] = approach['accepted']
            if approach['accepted']:
                target = np.asarray(solution['pose_world_xyzw'])
                checked = check_joint_path(scene, sim, ik, approach['joint_postures'],
                    gripper_qpos=metadata['gripper_qpos'], transform_at_pose=lambda pose: np.eye(4),
                    stage_at_pose=lambda pose: 'grasp' if np.linalg.norm(np.asarray(pose)[:3]-target[:3]) <= .012 else 'approach',
                    check_payload=False, config=ik.config)
                row['approach_check'] = checked
                if checked['accepted']:
                    pull = ik.path(manipulation_poses(spec, target, ik.config), q)
                    row['pull_ik_accepted'] = pull['accepted']
                    if pull['accepted']:
                        aperture = estimated_contact_aperture(sim, q, scene.region.contact_reference, metadata['gripper_qpos'])
                        if aperture is not None:
                            pull_check = check_joint_path(scene, sim, ik, [s['joints'] for s in pull['solutions']],
                                gripper_qpos=aperture, transform_at_pose=drawer_transform(scene, target),
                                stage_at_pose=lambda pose: 'pull', check_payload=True, config=ik.config)
                            row['pull_check'] = pull_check
                            row['full_approach_and_pull_accepted'] = pull_check['accepted']
        rows.append(row)
        save(output / 'multiseed-progress.json', {'seeds_attempted': number+1, 'distinct_solutions': len(rows), 'rows': rows})
    result = {'scope': 'counterfactual IK/path admission with existing CPU solver; no rejected motion executed',
        'curobo_used': False, 'seeds_attempted': len(starts), 'distinct_ik_solutions': len(rows),
        'endpoint_accepted': sum(row['endpoint_accepted'] for row in rows),
        'full_approach_and_pull_accepted': sum(row['full_approach_and_pull_accepted'] for row in rows),
        'ranking': 'complete approach/pull feasibility, then joint margin, then joint travel',
        'rows': sorted(rows, key=lambda row: (-int(row['full_approach_and_pull_accepted']), -row['minimum_joint_margin_rad'], row['joint_distance_from_start_rad'])),
        'physical_task_success_measured': False, 'new_api_calls': 0}
    save(output / 'multiseed-ik.json', result)
    return result


def joint_limit_figure(output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    folder = ROOT / 'results/physical-tasks-v11-eval/goal08-state021'
    action = read(folder / 'initial/actions.json')[-1]
    limits = np.asarray(read(folder / 'robot-chain.json')['joint_limits'])
    q, target = np.asarray(action['joint_positions']), np.asarray(action['nullspace_target_joints'])
    gap = np.degrees(limits[:, 1] - q)
    planned_gap = np.degrees(limits[:, 1] - target)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    axes[0].bar(np.arange(1, 8)-.18, planned_gap, width=.36, color='#268db4', label='Commanded margin')
    axes[0].bar(np.arange(1, 8)+.18, gap, width=.36, color='#ae4852', label='Measured margin')
    axes[0].set(xlabel='Arm joint', ylabel='Degrees below upper limit', title='All seven arm joints')
    axes[0].legend(fontsize=8)
    axes[1].barh(['Commanded joint 6', 'Measured joint 6'], [planned_gap[5], gap[5]], color=['#268db4', '#b9384c'])
    axes[1].axvline(0, color='#222', linewidth=1)
    axes[1].set(xlabel='Degrees below upper limit (negative = overshoot)', title='Joint 6 — enlarged scale')
    axes[1].grid(axis='x', alpha=.2)
    figure.savefig(output / 'joint-limit.png', dpi=170)
    plt.close(figure)
    result = {'joint_one_based': 6, 'limit_rad': float(limits[5, 1]), 'command_rad': float(target[5]),
              'measured_rad': float(q[5]), 'command_margin_deg': float(planned_gap[5]), 'measured_margin_deg': float(gap[5])}
    save(output / 'joint-limit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/rejected-grasp-visualization-v11')
    parser.add_argument('--ik-seeds', type=int, default=32)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError('Preserve previous diagnostics')
    output.mkdir(parents=True)
    ledger = ROOT / 'results/rekep-libero-api/ledger.json'
    budget_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    protocol = read(ROOT / 'results/physical-tasks-v11-eval/protocol.json')
    frozen_before = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in protocol['sources_sha256']}
    config = read(ROOT / 'configs/physical_tasks_v11.json')
    cases = [('deep-grasp', 11, 24, 'raw', 'full_arm_grasp_collision', 'grasp', 'Deep overlap at the final grasp'),
             ('upper-handle-approach', 11, 75, 'asset_contact_aligned_half_turn', 'full_arm_approach_collision', 'approach', 'Small finger overlap during approach'),
             ('palm-buffer-pull', 31, 178, 'asset_contact_aligned_half_turn', 'moving_payload_task_collision', 'pull', 'Positive palm gap rejected by the 1 mm buffer')]
    sim = SimClientV8(log_path=output / 'simulator.log')
    cards, results, render_inputs = [], [], []
    try:
        sim.reset(0, 11, output / 'robot-model', seed=0)
        for slug, state, index, variant, reason, phase, title in cases:
            folder = ROOT / f'results/physical-tasks-v11-eval/goal00-state{state:03d}'
            row = next(r for r in read(folder / 'grasp-admission.json')['rejected_candidates']
                       if r['index'] == index and r['variant'] == variant and r['reason'] == reason)
            scene, cameras, metadata, spec, ik, grasp, final_geometry, final_check, geometry, result, moving = replay(folder, row, sim, config, phase)
            rows = collision_points(scene, geometry, result, moving)
            render_inputs.append({'case': slug, 'state': state, 'joints': geometry['hypothetical_arm_joints'],
                'gripper_qpos': geometry['gripper_qpos'], 'pull_axis_world': spec['pull_axis_world'],
                'drawer_displacement_m': float(moving[:3, 3] @ np.asarray(spec['pull_axis_world']))})
            points = np.vstack([r['points'] for r in rows])
            static_figure(output / f'{slug}.png', title, scene, cameras, geometry, rows, moving, final_geometry)
            figures = [('rejected', figure_3d(scene, geometry, rows, moving, 'Rejected configuration — rotate to inspect')),
                       ('closeup', figure_3d(scene, geometry, rows, moving, 'Contact neighborhood', zoom=True, centre=points.mean(0))),
                       ('grasp', figure_3d(scene, final_geometry, [], np.eye(4), f'Final grasp pose — {"accepted" if final_check["accepted"] else "rejected"}'))]
            for suffix, figure in figures:
                (output / f'{slug}-{suffix}.json').write_text(figure.to_json())
            summary = {'case': slug, 'state': state, 'candidate': index, 'variant': variant, 'phase': phase,
                       'final_grasp_accepted': final_check['accepted'], 'replayed_rejection_matches_original': True,
                       'rejection': result, 'known_robot_geometry_only': True, 'scene_reconstruction_from_rgbd': True,
                       'simulator_scene_identity_used': False, 'physical_rejected_pose_executed': False}
            save(output / f'{slug}.json', summary)
            results.append(summary)
            explanation = ('These are substantial overlaps in the first IK posture: scene points extend roughly 52 mm into a forearm volume and 22 mm into the palm. Different arm IK can move the elbow, but a fixed TCP pose also fixes the palm/finger geometry.' if phase == 'grasp' else
                'The final grasp pose is accepted. A different pose along the initial approach is rejected for a single observed point about 0.696 mm inside a finger volume near the upper handle. This is a predicted overlap, not a physically observed collision.' if phase == 'approach' else
                'There is no predicted palm penetration here: the minimum gap is about 0.221 mm. The configured palm buffer is 1 mm, so this candidate is rejected for insufficient clearance. Orange points mark positive gaps, not collision penetration.')
            cards.append(f'<article><h2>{html.escape(title)}</h2><p>{explanation}</p><img class="static" src="{slug}.png">'
                f'<details><summary>Rotate the 3-D geometry and compare the final grasp</summary><div class="plots">'
                + ''.join(f'<div class="plot" id="{slug}-{suffix}" data-figure="{slug}-{suffix}.json"></div>' for suffix, _ in figures)
                + f'</div></details><p><a href="{slug}.json">Exact replay record</a></p></article>')
            print(json.dumps({'case': slug, 'replay_match': True, 'final_grasp_accepted': final_check['accepted'],
                              'collisions': [{k:v for k,v in c.items() if k != 'example_points_world_m'} for c in result['collisions']]}), flush=True)
            if slug == 'upper-handle-approach':
                probe = multiseed_probe(folder, scene, sim, metadata, spec, ik, grasp, output, seeds=args.ik_seeds)
                print(json.dumps({k:v for k,v in probe.items() if k != 'rows'}), flush=True)
    finally:
        sim.close()
    save(output / 'pose-render-input.json', render_inputs)
    joint = joint_limit_figure(output)
    from plotly.offline import get_plotlyjs
    (output / 'plotly.min.js').write_text(get_plotlyjs())
    probe = read(output / 'multiseed-ik.json')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inspecting rejected grasp and motion poses</title><script src="plotly.min.js"></script><style>
body{{margin:0;background:#f5f7fa;color:#203344;font:16px/1.6 system-ui}}main{{max-width:1200px;margin:auto;padding:28px}}
article,.note{{background:white;padding:22px;border:1px solid #d8e2e9;border-radius:12px;margin:24px 0}}h1{{line-height:1.2}}h2{{font-size:22px}}
.static{{width:100%}}.plots{{display:grid;grid-template-columns:1fr;gap:18px}}.plot{{min-height:550px}}summary{{cursor:pointer}}a{{color:#156383}}
</style><main><h1>Inspecting rejected grasp and motion poses</h1>
<p>Blue/gray meshes are the configured Panda collision volumes. Green points are the selected drawer; gray points are other observed RGB-D surfaces.
Red points are predicted inside robot volumes. Orange points have positive clearance but violate a configured buffer.</p>
<div class="note">Every rejection is replayed at the same seven-joint configuration and gripper aperture as the saved v11 decision.
The recorded point counts and signed distances match. Robot geometry comes from its configured model; scene geometry comes from RGB-D.
These poses were not executed physically. Initial scene photographs retain the robot's original pose; the colored wireframe is the hypothetical rejected pose.</div>
{''.join(cards)}<article><h2>Trying more IK solutions for the same grasp</h2>
<p>Existing CPU IK, {probe['seeds_attempted']} seeds: {probe['distinct_ik_solutions']} distinct solutions,
{probe['endpoint_accepted']} pass the endpoint scene check and {probe['full_approach_and_pull_accepted']} pass the complete sampled approach and pull checks.
This is a counterfactual planning test, not a new physical success. cuRobo is not installed or used in this diagnostic.</p>
<p><a href="multiseed-ik.json">All solutions and path checks</a></p></article>
<article><h2>The bowl joint-limit stop</h2><img class="static" src="joint-limit.png">
<p>Joint 6 was commanded {joint['command_margin_deg']:.5f}° below its upper limit, but the measured state was {-joint['measured_margin_deg']:.5f}° above it.
Our placement IK keeps only a 0.0001 rad joint margin and returns the first solution meeting its objective. More interior solutions and explicit ranking are appropriate;
controller tracking still needs a margin. <a href="joint-limit.json">Numbers</a></p></article>
<article><h2>Why cuRobo is a useful next candidate</h2><p>cuRobo supports parallel IK seeds, returning multiple solutions, environment collision checks and motion generation.
Our calibrated robot/TCP model and phase-specific contact rules must be preserved; an IK endpoint alone does not establish that its approach or drawer pull is feasible.</p>
<p><a href="https://curobo.org/_api/curobo.wrap.reacher.ik_solver.html">Official IK API</a> ·
<a href="https://curobo.org/get_started/2c_world_collision.html">Collision geometry documentation</a></p></article>
<article><h2>VLA comparison</h2><p>OpenVLA-OFT's LIBERO evaluation loop sends policy action chunks directly to the simulator and counts task completion.
It does not impose the geometric preflight veto used here. Thus small contact can occur during a successful episode. This is not evidence that every policy succeeds by bumping randomly,
or that VLA cannot have collision-aware execution. Safety-specific VLA work evaluates those concerns separately.</p>
<p><a href="https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py">Official evaluation loop</a> ·
<a href="https://arxiv.org/abs/2610.01083">WBAG VLA collision-avoidance framework</a></p></article>
<p><a href="summary.json">Diagnostic summary</a> · <a href="validation.json">Validation</a></p></main>
<script>document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{{if(!d.open)return;d.querySelectorAll('.plot').forEach(async el=>{{
if(el.dataset.ready)return;el.dataset.ready='yes';const response=await fetch(el.dataset.figure);const f=await response.json();Plotly.newPlot(el,f.data,f.layout,{{responsive:true,displaylogo:false}});
}})}}));</script></html>'''
    # Embed figure payloads: file:// pages cannot reliably fetch sibling JSON.
    payloads = {path.stem: json.loads(path.read_text()) for path in output.glob('*-rejected.json')}
    payloads.update({path.stem: json.loads(path.read_text()) for path in output.glob('*-closeup.json')})
    payloads.update({path.stem: json.loads(path.read_text()) for path in output.glob('*-grasp.json')})
    document = document.replace("const response=await fetch(el.dataset.figure);const f=await response.json();", "const f=FIGURES[el.id];")
    document = document.replace('<script>document.querySelectorAll', '<script>const FIGURES='+json.dumps(payloads)+';document.querySelectorAll')
    (output / 'index.html').write_text(document)
    unchanged = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name,digest in frozen_before.items())
    validation = {'all_rejected_pose_replays_match': all(row['replayed_rejection_matches_original'] for row in results),
        'frozen_v11_sources_unchanged': unchanged,
        'api_ledger_unchanged': hashlib.sha256(ledger.read_bytes()).hexdigest() == budget_hash,
        'new_api_calls': 0, 'new_api_spend_usd': 0, 'rejected_configurations_executed': 0,
        'curobo_used': False, 'primary_source_links_included': True}
    save(output / 'validation.json', validation)
    save(output / 'summary.json', {'cases': results, 'multiseed_probe': {k:v for k,v in probe.items() if k != 'rows'},
                                    'joint_limit': joint, 'validation': validation})
    print(json.dumps(validation), flush=True)
    if not unchanged:
        raise ValueError('Frozen benchmark sources changed')


if __name__ == '__main__':
    main()
