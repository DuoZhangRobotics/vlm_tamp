"""Compact signed-gap figures and an alternative-IK trajectory comparison."""

import base64
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]
import visualize_rejected_grasps_v11 as vis


def array(value):
    if isinstance(value, dict) and 'bdata' in value:
        result = np.frombuffer(base64.b64decode(value['bdata']), dtype=np.dtype(value['dtype']))
        if 'shape' in value:
            result = result.reshape(tuple(int(x) for x in value['shape'].replace(',', ' ').split()))
        return result
    return np.asarray(value)


def compact_comparison(output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from PIL import Image
    cases = [('deep-grasp', 'Deep overlap at grasp'), ('upper-handle-approach', 'Tiny overlap along approach'),
             ('palm-buffer-pull', 'Buffer only — no penetration')]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 6.5), constrained_layout=True,
                               gridspec_kw={'height_ratios': [2.2, 1.]})
    values = []
    for column, (slug, title) in enumerate(cases):
        png = Image.open(output/f'{slug}.png')
        axes[0, column].imshow(png.crop((190, 80, 835, 630)))
        axes[0, column].axis('off')
        axes[0, column].set_title(title, fontsize=12)
        data = vis.read(output/f'{slug}-rejected.json')
        case = vis.read(output/f'{slug}.json')
        worst = min(case['rejection']['collisions'], key=lambda c:c['minimum_signed_point_distance_m'])
        name = worst['robot_geom_name']
        cloud = next(t for t in data['data'] if t.get('customdata') is not None and t.get('name','').endswith(name))
        distances = array(cloud['customdata']).reshape(-1)
        index = int(np.argmin(distances))
        point = np.array([array(cloud[k])[index] for k in ['x','y','z']])
        mesh = next(t for t in data['data'] if t['type']=='mesh3d' and t['name']==name)
        vertices = np.c_[array(mesh['x']), array(mesh['y']), array(mesh['z'])]
        planes = ConvexHull(vertices).equations
        signed = float(np.max(planes[:, :3] @ point + planes[:, 3])) * 1000
        if abs(signed - worst['minimum_signed_point_distance_m']*1000) > 1e-4:
            raise ValueError('Compact normal-gap diagram does not match the 3-D checker')
        margin = worst['required_clearance_m']*1000
        ax = axes[1, column]
        interval = (-60, 4) if signed < -5 else (-1.4, 1.6)
        ax.set_xlim(*interval)
        ax.axvspan(interval[0], 0, facecolor='#c85058', alpha=.13)
        if margin:
            ax.axvspan(0, margin, facecolor='#d99724', alpha=.17)
            ax.axvline(margin, color='#c48824', linestyle='--', linewidth=1)
        ax.axvline(0, color='#333', linewidth=1.3)
        color = '#c33847' if signed < 0 else '#c6891b'
        ax.plot([0, signed], [.3, .3], color=color, linewidth=2)
        ax.scatter([signed], [.3], color=color, s=55)
        ax.annotate(f'{signed:+.3f} mm', (signed, .3), xytext=(0, 13), textcoords='offset points',
                    ha='center', fontsize=12, color=color)
        ax.text(0, .0, 'Robot boundary', ha='center', fontsize=8)
        if margin:
            ax.text(margin, -.28, '1 mm buffer', ha='center', fontsize=8, color='#a36b0d')
        ax.set_ylim(-.45, .8)
        ax.set_yticks([])
        ax.set_xlabel('Signed gap along the active robot face (mm)', fontsize=9)
        ax.grid(axis='x', alpha=.2)
        values.append({'case':slug, 'robot_part':name, 'signed_gap_mm':signed, 'required_buffer_mm':margin})
    figure.suptitle('Three different reasons for rejecting a robot configuration', fontsize=16)
    figure.savefig(output/'comparison.png', dpi=165)
    plt.close(figure)
    vis.save(output/'signed-gap-comparison.json', {'values':values, 'distances_match_original_3d_checker':True,
        'figures_are_hypothetical_robot_configurations_on_initial_scene_images':True})


def alternative_ik(output):
    import plotly.graph_objects as go
    folder = ROOT/'results/physical-tasks-v11-eval/goal00-state011'
    config = vis.read(ROOT/'configs/physical_tasks_v11.json')
    probe = vis.read(output/'multiseed-ik.json')
    winner = next(row for row in probe['rows'] if row['full_approach_and_pull_accepted'])
    scene, cameras, metadata, spec = vis.load_scene(folder, config)
    ik = vis.make_ik(folder, config)
    row = next(r for r in vis.read(folder/'grasp-admission.json')['rejected_candidates']
        if r['index']==75 and r['variant']=='asset_contact_aligned_half_turn' and r['reason']=='full_arm_approach_collision')
    original = ik.solve(row['tcp_pose_world_xyzw'], metadata['joint_positions'])
    original_approach = ik.approach(np.asarray(metadata['joint_positions']), original)
    new_approach = ik.approach(np.asarray(metadata['joint_positions']), winner['solution'])
    sim = vis.SimClientV8(log_path=output/'alternative-ik-simulator.log')
    try:
        sim.reset(0, 11, output/'alternative-ik-robot-model', seed=0)
        geometry = sim.fk(winner['solution']['joints'], include_volumes=True, gripper_qpos=metadata['gripper_qpos'])
        aperture = vis.estimated_contact_aperture(sim, winner['solution']['joints'], scene.region.contact_reference,
                                                metadata['gripper_qpos'])
        closure = scene.joints(sim, winner['solution']['joints'], gripper_qpos=aperture, stage='grasp')
    finally:
        sim.close()
    vis.save(output/'winning-ik-closure.json', {'predicted_closure_check':closure,
        'scope':'hypothetical isolated robot geometry; not an executed grasp', 'new_api_calls':0})
    figure = vis.figure_3d(scene, geometry, [], np.eye(4), 'Same grasp, different IK: planned TCP approach routes')
    for name, approach, color in [('Original IK route — rejected', original_approach, '#ca3749'),
                                  ('Alternative IK route — passes sampled checks', new_approach, '#158964')]:
        path = np.asarray(approach['poses_world_xyzw'])[:, :3]
        figure.add_trace(go.Scatter3d(x=path[:,0],y=path[:,1],z=path[:,2],mode='lines+markers',
            line={'color':color,'width':6},marker={'size':3,'color':color},name=name))
    (output/'alternative-ik-route.json').write_text(figure.to_json())
    details = {'same_requested_grasp_pose':True, 'alternative_seed_index':winner['seed_index'],
        'minimum_endpoint_joint_margin_rad':winner['minimum_joint_margin_rad'],
        'predicted_closure_accepted':closure['accepted'],
        'sampled_approach_accepted':winner['approach_check']['accepted'],
        'sampled_pull_accepted':winner['pull_check']['accepted'],
        'physical_task_success_measured':False, 'curobo_used':False}
    vis.save(output/'alternative-ik-route-record.json', details)
    return figure, details


def main():
    output = ROOT/'results/rejected-grasp-visualization-v11'
    if (output/'comparison.png').exists():
        raise FileExistsError('Preserve existing supplements')
    compact_comparison(output)
    figure, details = alternative_ik(output)
    path = output/'index.html'
    document = path.read_text()
    document = document.replace('<h1>Inspecting rejected grasp and motion poses</h1>',
        '<h1>Inspecting rejected grasp and motion poses</h1><img class="static" src="comparison.png">')
    marker = '<p><a href="multiseed-ik.json">All solutions and path checks</a></p>'
    document = document.replace(marker, marker+f'<p>The passing alternative also passes predicted gripper closure. '
        f'Its smallest endpoint joint reserve is {details["minimum_endpoint_joint_margin_rad"]:.3f} rad. '
        'Both routes target the same grasp pose; this demonstrates a choice-of-IK issue for this case.</p>'
        '<details><summary>Compare the two planned approach routes in 3-D</summary>'
        '<div class="plot" id="alternative-ik-route"></div></details>')
    document = document.replace('const FIGURES=', 'const ALTERNATIVE_IK='+figure.to_json()+';const FIGURES=')
    document = document.replace('const f=FIGURES[el.id];', "const f=el.id==='alternative-ik-route'?ALTERNATIVE_IK:FIGURES[el.id];")
    path.write_text(document)
    print(json.dumps(details))


if __name__=='__main__':
    main()
