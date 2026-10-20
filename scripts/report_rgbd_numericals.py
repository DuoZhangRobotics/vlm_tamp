"""Inspectable RGB-D pilot report with offline labels and interactive 3D views."""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vlm_tamp.dataset import read_records
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_numeric_evaluation import capture_checks, grade_case
from vlm_tamp.rgbd_numeric_trial import check_protocol
from vlm_tamp.rgbd_numericals import POINT_FIELDS, json_text, pixel_index


LABELS = dict(zip(POINT_FIELDS, ['S', 'G', 'A', 'B', 'L', 'H', 'T']))
COLORS = ['#ff65a5', '#00dfbf', '#ffb84d', '#ffb84d', '#95a7ff', '#95a7ff', '#80eaff']


def overlay(source, blanks, metrics, destination):
    image = Image.open(source).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for index, key in enumerate(POINT_FIELDS):
        uv = blanks[key] if blanks else None
        if uv is None:
            continue
        x, y = pixel_index(uv, (height, width))
        correct = metrics['anchors'][index]['diagnostic'] == 'correct_instance'
        draw.ellipse([x-5, y-5, x+5, y+5], fill=COLORS[index], outline='black', width=2)
        draw.text((x+7, y-10), LABELS[key] + ('' if correct else '!'), fill='white', stroke_width=2,
                  stroke_fill='black')
    image.save(destination)


def depth_picture(depth, destination):
    value = np.clip((depth-.6)/2.5, 0, 1)
    palette = np.array([[30, 40, 150], [0, 200, 190], [255, 235, 100]], dtype=float)
    low = np.minimum((value*2).astype(int), 1)
    fraction = np.minimum(value*2-low, 1)[..., None]
    rgb = palette[low]*(1-fraction)+palette[low+1]*fraction
    Image.fromarray(rgb.astype('uint8')).save(destination)


def cloud_data(rgb, cloud, numeric):
    positions, colors = cloud[::7, ::7].reshape(-1, 3), rgb[::7, ::7].reshape(-1, 3)
    anchors = []
    if numeric:
        for index, key in enumerate(POINT_FIELDS):
            landmark = numeric['landmarks'][key]
            if landmark['status'] == 'measured':
                anchors.append({'name': LABELS[key], 'position': landmark['position_m'], 'color': COLORS[index]})
    center = np.mean([anchor['position'] for anchor in anchors], axis=0) if anchors else np.array([0., 0., .9])
    keep = np.linalg.norm(positions-center, axis=1) < .55
    positions, colors = positions[keep], colors[keep]
    lines = []
    if numeric and numeric['grasp']['status'] == 'candidate':
        grasp = numeric['grasp']
        p, r = np.array(grasp['position_m']), np.array(grasp['rotation_matrix'])
        for index, color in enumerate(['#ff6666', '#66ee88', '#66aaff']):
            lines.append({'a': p.tolist(), 'b': (p+r[:, index]*.045).tolist(), 'color': color})
        lines.append({'a': p.tolist(), 'b': grasp['pregrasp_position_m'], 'color': '#ffffff'})
    if numeric and numeric['support']['status'] == 'estimated':
        p = np.array(numeric['support']['plane_point_world_m'])
        n = np.array(numeric['support']['normal_world'])
        lines.append({'a': p.tolist(), 'b': (p+n*.06).tolist(), 'color': '#fce680'})
    return {'points': np.round(positions, 6).tolist(), 'colors': colors.tolist(),
            'center': center.tolist(), 'anchors': anchors, 'lines': lines}


VIEWER = r'''
for (const canvas of document.querySelectorAll('canvas.cloud')) {
 const data=JSON.parse(document.getElementById(canvas.dataset.scene).textContent);
 const ctx=canvas.getContext('2d'); let yaw=.3, pitch=.8, zoom=800, dragging=false, last;
 function draw(){
  const w=canvas.width=700,h=canvas.height=450; ctx.fillStyle='#0a1320';ctx.fillRect(0,0,w,h);
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  function project(p){const x=p[0]-data.center[0],y=p[1]-data.center[1],z=p[2]-data.center[2];
   const a=cy*x-sy*y,b=sy*x+cy*y;return [w/2+zoom*a,h/2-zoom*(cp*z-sp*b),sp*z+cp*b];}
  const pts=data.points.map((p,i)=>[...project(p),data.colors[i]]).sort((a,b)=>a[2]-b[2]);
  for(const p of pts){ctx.fillStyle=`rgb(${p[3].join(',')})`;ctx.fillRect(p[0],p[1],2.5,2.5);}
  for(const line of data.lines){const a=project(line.a),b=project(line.b);ctx.strokeStyle=line.color;
   ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
  for(const anchor of data.anchors){const p=project(anchor.position);ctx.fillStyle=anchor.color;
   ctx.beginPath();ctx.arc(p[0],p[1],5,0,7);ctx.fill();ctx.fillStyle='white';ctx.font='15px system-ui';
   ctx.fillText(anchor.name,p[0]+8,p[1]-5);}
  ctx.fillStyle='#c6d5e8';ctx.font='13px system-ui';ctx.fillText('World coordinates in meters · visible depth surfaces',12,24);
 }
 canvas.addEventListener('pointerdown',e=>{dragging=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{if(!dragging)return;yaw+=(e.clientX-last[0])*.008;
  pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw();});
 canvas.addEventListener('pointerup',()=>dragging=false);
 canvas.addEventListener('pointercancel',()=>dragging=false);
 canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(200,Math.min(3000,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
 draw();
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', default='results/rgbd-numerical-independent-axis')
    parser.add_argument('--output', default='results/rgbd-numerical-report')
    parser.add_argument('--config', default='configs/rgbd_numeric_independent_axis.json')
    args = parser.parse_args()
    trial, output = Path(args.trial), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    manifest = json.loads((trial / 'manifest.json').read_text())
    usage = manifest.get('source_ledger', manifest['ledger'])
    config, protocol, records = check_protocol(ROOT / args.config)
    _, source = read_records(ROOT / config['dataset'])
    rows = {row['case_id']: row for row in map(json.loads, (trial / 'predictions.jsonl').read_text().splitlines())}
    tasks, diagnostics, captures = [], Counter(), {}
    for case in config['cases']:
        record = records[case['example_id']]
        row = rows.get(case['id'], {'status': 'not_run', 'blanks': None, 'numericals': None})
        capture = ROOT / config['depth_dataset'] / case['example_id']
        depth, cloud = [np.load(capture / name, allow_pickle=False) for name in ('depth_meters.npy', 'world_points.npy')]
        calibration = json.loads((capture / 'calibration.json').read_text())
        captures[case['example_id']] = capture_checks(depth, cloud, calibration)
        metrics = grade_case(record, source, case, row)
        diagnostics.update(anchor['diagnostic'] for anchor in metrics['anchors'])
        rgb_path = trial / case['id'] / 'original.png'
        if not rgb_path.exists():
            rgb_path = capture / 'rgb.png'
        rgb = np.asarray(Image.open(rgb_path))
        overlay(rgb_path, row['blanks'], metrics, assets / (case['id']+'-points.png'))
        depth_picture(depth, assets / (case['id']+'-depth.png'))
        viewer = cloud_data(rgb, cloud, row['numericals'])
        task = {**case, 'instruction': protocol['requests'][len(tasks)]['payload']['instruction'],
                'status': row['status'], 'blanks': row['blanks'], 'numericals': row['numericals'], 'metrics': metrics}
        (assets / (case['id']+'-numericals.json')).write_text(json_text(task)+'\n')
        tasks.append((task, viewer))
    checks = [check for task, _ in tasks for check in task['metrics']['relation_checks']]
    summary = {'model': config['model'], 'cases': len(tasks), 'unique_scenes': len(captures),
               'valid_forms': sum(task['status'] == 'valid' for task, _ in tasks),
               'semantic_complete_goals': sum(task['metrics']['semantic_complete_goal'] for task, _ in tasks),
               'anchor_diagnostics': dict(diagnostics),
               'depth_lifted_landmarks': sum(anchor['depth_status'] == 'measured'
                                            for task, _ in tasks for anchor in task['metrics']['anchors']),
               'measured_landmarks_on_correct_instance': sum(anchor['depth_status'] == 'measured'
                                                           and anchor['diagnostic'] == 'correct_instance'
                                                           for task, _ in tasks for anchor in task['metrics']['anchors']),
               'grasp_candidates': sum(task['metrics']['grasp_candidate'] for task, _ in tasks),
               'correctly_bound_grasp_candidates': sum(task['metrics']['candidate_landmarks_on_correct_object']
                                                       for task, _ in tasks),
               'correct_path_choices': sum(task['metrics']['path_choice_correct'] for task, _ in tasks),
               'relation_self_checks': len(checks), 'passing_relation_self_checks': sum(c['passed'] for c in checks),
               'maximum_jacobian_error': max((c['maximum_jacobian_error'] for c in checks), default=None),
               'capture_checks': captures, 'ledger': usage, 'new_compilation_ledger': manifest['ledger'],
               'preserved_artifacts': len(protocol['preserved_artifacts']),
               'joint_space_compiled': False, 'ik_tested': False, 'robot_execution_tested': False,
               'grasp_stability_verified': False, 'human_reviewed_affordance_labels': False,
               'development_sample': True, 'tasks': [task for task, _ in tasks]}
    (output / 'summary.json').write_text(json_text(summary)+'\n')
    esc = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Depth-based numerical grounding</title><style>body{font:16px system-ui;line-height:1.55;max-width:1450px;margin:auto;padding:24px;background:#111826;color:#edf2ff}.panel,article{background:#1b2638;padding:22px;margin:22px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:18px}img,canvas{width:100%;height:auto}canvas{touch-action:none}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #405169;text-align:left}a{color:#79d5ff}.ok{color:#74dfae}.bad{color:#ffa39d}h2{font-size:22px}</style>',
             '<h1>Depth-based numerical grounding</h1><p>Mini fills image landmarks and method choices. Calibrated LIBERO depth supplies 3D points; a fixed geometry library proposes grasp frames and numerical relations.</p>',
             f'<div class="panel"><strong>{summary["valid_forms"]}/6 valid replies · {summary["semantic_complete_goals"]}/6 complete object/target bindings · {summary["grasp_candidates"]}/6 grasp-frame proposals.</strong><p>{summary["depth_lifted_landmarks"]}/42 landmarks pass depth checks; {summary["measured_landmarks_on_correct_instance"]} of those are on the correct object. Original API usage: ${usage["estimated_usage_cost_usd"]:.5f} across {usage["api_attempts"]} calls. Independent-axis compilation reuses those replies with zero additional calls.</p></div>',
             '<div class="panel"><strong>What is established:</strong> aligned depth capture, numerical estimates, and formula/Jacobian checks. <strong>Still unresolved:</strong> robot TCP conversion, object-bottom offsets, IK, collisions, stable contact and physical execution. A grasp proposal is not a successful grasp; a numerical point target is not a verified whole-object placement.</div>',
             '<p>Four previously tested scenes, six development cases. The two upright variants add an explicit instruction; they are not original LIBERO path requirements. No inventory, masks, crops, DINOv2, or simulator answers enter inference. Depth is consumed locally after the RGB reply. Depth at discontinuities is rejected without moving the selected point.</p>',
             '<p>Labels: S source identity · G grasp center · A/B closing edges · L/H object axis · T target support. “!” means the unchanged point missed the expected object under held-back masks. Those masks grade whole objects, not human-reviewed grasp parts.</p>']
    for task, viewer in tasks:
        identifier, metric = task['id'], task['metrics']
        parts += [f'<article id="{esc(identifier)}"><h2>{esc(task["instruction"])}</h2>',
                  f'<p>Object/target binding: <strong>{"correct" if metric["semantic_complete_goal"] else "incorrect or unresolved"}</strong> · Path choice: {"correct" if metric["path_choice_correct"] else "incorrect or unresolved"} · Grasp frame: {"proposed" if metric["grasp_candidate"] else "unresolved"}</p>',
                  f'<div class="grid"><div><h3>RGB landmarks</h3><img src="assets/{identifier}-points.png" alt="Selected landmarks on original RGB"></div><div><h3>Aligned depth</h3><img src="assets/{identifier}-depth.png" alt="Aligned metric depth"><small>Blue → yellow: 0.6 → 3.1 meters.</small></div><div><h3>Observed 3D geometry</h3><canvas class="cloud" data-scene="scene-{identifier}" aria-label="Interactive 3D surface and grasp-frame view"></canvas><small>Drag to rotate; scroll to zoom. Grasp axes: red x, green y, blue z. White line: proposed approach. Yellow: local support normal.</small></div></div>',
                  f'<script type="application/json" id="scene-{identifier}">{json_text(viewer).replace("<", "&lt;")}</script>',
                  '<table><tr><th>Landmark</th><th>Object binding</th><th>Depth estimate</th></tr>']
        for anchor in metric['anchors']:
            point = task['numericals']['landmarks'][anchor['field']] if task['numericals'] else {}
            value = ', '.join(f'{number:.4f}' for number in point['position_m']) if 'position_m' in point else point.get('reason', 'unresolved')
            parts.append(f'<tr><td>{LABELS[anchor["field"]]}</td><td>{esc(anchor["diagnostic"])}</td><td>{esc(value)}</td></tr>')
        parts += ['</table>', f'<p>Object-axis proxy error: {metric["object_axis_canonical_z_proxy_error_degrees"]}° · Support-normal proxy error: {metric["support_normal_canonical_z_proxy_error_degrees"]}°. These use held-back object-frame axes, not reviewed part labels.</p>',
                  f'<details><summary>VLM-filled blanks</summary><pre>{esc(json.dumps(task["blanks"],indent=2))}</pre></details>',
                  f'<details><summary>Numerical estimates and unresolved values</summary><pre>{esc(json.dumps(task["numericals"],indent=2))}</pre></details>',
                  f'<details><summary>Equation and Jacobian checks</summary><pre>{esc(json.dumps(metric["relation_checks"],indent=2))}</pre></details>',
                  f'<a href="assets/{identifier}-numericals.json">Download case results</a></article>']
    parts += [f'<details class="panel"><summary>Exact prompt</summary><pre>{esc(manifest["prompt_text"])}</pre></details>',
              '<script>'+VIEWER+'</script></html>']
    (output / 'index.html').write_text('\n'.join(parts))
    print({key: value for key, value in summary.items() if key not in {'tasks', 'capture_checks'}})


if __name__ == '__main__':
    main()
