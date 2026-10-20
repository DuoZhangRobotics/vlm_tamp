"""Calibrated hinge-axis and requested/measured TCP overlays on saved scenes."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
COLORS = {'estimated': '#f45cda', 'native': '#26e3e8', 'target': '#ffe052', 'actual': '#ff593e'}
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT = ImageFont.truetype(FONT_PATH, 15)
TITLE = ImageFont.truetype(FONT_PATH, 18)


def read(path):
    return json.loads(Path(path).read_text())


def project(points, camera):
    points = np.atleast_2d(points)
    transform = np.asarray(camera['camera_to_world'])
    local = (points-transform[:3, 3]) @ transform[:3, :3]
    if np.any(local[:, 2] <= 0):
        raise ValueError('Audit annotation projects behind the camera')
    image = local @ np.asarray(camera['intrinsic']).T
    return image[:, :2]/image[:, 2:3]


def episode(folder):
    spec = read(folder/'estimated-task-spec.json')
    actions = read(folder/'initial/actions.json')
    contacts = read(folder/'initial/evaluation_contact_diagnostics.json')['steps']
    initial = read(folder/'initial/metadata.json')
    native = next(j for j in contacts[0]['evaluation_only_articulation_state']
                  if j['joint_name'] == 'microwave_1_microjoint')
    return {'folder': folder, 'spec': spec, 'actions': {a['step']: a for a in actions},
            'native': native, 'native_steps': {s['step']: next(j for j in s['evaluation_only_articulation_state']
                if j['joint_name'] == native['joint_name']) for s in contacts},
            'initial': initial, 'target': read(folder/'chosen-grasp.json')['fk_pose_world_xyzw'][:3],
            'frames': sorted((folder/'initial/frames').glob('step*.jpg')),
            'summary': read(folder/'summary.json')}


def annotation(rgb, data, camera, step, label, actual=None, gap=None):
    image = rgb.copy().convert('RGB')
    draw = ImageDraw.Draw(image)
    native = data['native']
    native_origin = np.asarray(native['anchor_world_m'])
    for kind, origin, axis in [
            ('native', native_origin, native['axis_world']),
            ('estimated', data['spec']['joint_origin_world_m'], data['spec']['joint_axis_world'])]:
        origin, axis = np.asarray(origin, float), np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        # Draw equivalent axis segments at the same door height for comparison.
        centre = origin + axis*((native_origin-origin) @ axis)
        pixels = project([centre-.105*axis, centre+.105*axis], camera)
        points = [tuple(p) for p in pixels]
        draw.line(points, fill='black', width=8)
        draw.line(points, fill=COLORS[kind], width=4)
        for x, y in points:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=COLORS[kind])
    target = np.asarray(data['target'])
    tx, ty = project(target, camera)[0]
    draw.ellipse((tx-7, ty-7, tx+7, ty+7), outline='black', width=5)
    draw.ellipse((tx-7, ty-7, tx+7, ty+7), outline=COLORS['target'], width=3)
    if actual is None:
        actual = data['initial']['eef_pose_world_xyzw'][:3] if step == 0 else data['actions'][step]['eef_pose_world_xyzw'][:3]
    ax, ay = project(actual, camera)[0]
    draw.line([(ax-6, ay), (ax+6, ay)], fill='black', width=6)
    draw.line([(ax, ay-6), (ax, ay+6)], fill='black', width=6)
    draw.line([(ax-6, ay), (ax+6, ay)], fill=COLORS['actual'], width=3)
    draw.line([(ax, ay-6), (ax, ay+6)], fill=COLORS['actual'], width=3)
    if step >= 150:
        draw.line([(tx, ty), (ax, ay)], fill=COLORS['actual'], width=2)
    if gap is None:
        q = data['initial']['gripper_qpos'] if step == 0 else data['actions'][step]['gripper_qpos']
        gap = sum(abs(v) for v in q)
    error = 1000*float(np.linalg.norm(np.asarray(actual)-target))
    result = Image.new('RGB', (512, 644), '#10151e')
    result.paste(image, (0, 72))
    text = ImageDraw.Draw(result)
    text.text((10, 3), label, font=TITLE, fill='white')
    text.text((10, 28), 'Estimated hinge', font=FONT, fill=COLORS['estimated'])
    text.text((215, 28), 'True hinge', font=FONT, fill=COLORS['native'])
    text.text((10, 49), 'Requested TCP', font=FONT, fill=COLORS['target'])
    text.text((215, 49), 'Measured TCP', font=FONT, fill=COLORS['actual'])
    angle = abs(np.degrees(data['native_steps'].get(step, native)['position']))
    text.text((10, 589), f'Step {step} | Door opening {angle:.1f} deg', font=FONT, fill='white')
    text.text((10, 614), f'Distance to grasp {error:.1f} mm | Finger gap {gap*1000:.1f} mm', font=FONT, fill='white')
    return result


def video(panels, labels, path):
    size = (512*len(panels), 644)
    encoder = subprocess.Popen(['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{size[0]}x{size[1]}', '-framerate', '4',
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2',
        '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    try:
        for index in range(max(len(p['frames']) for p in panels)):
            canvas = Image.new('RGB', size)
            for offset, (data, label) in enumerate(zip(panels, labels)):
                frame = data['frames'][min(index, len(data['frames'])-1)]
                step = int(frame.stem[4:])
                with Image.open(frame) as rgb:
                    panel = annotation(rgb, data, data['initial']['cameras']['agentview'], step, label)
                canvas.paste(panel, (offset*512, 0))
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait(timeout=60) != 0:
            raise RuntimeError('Annotation video encoding failed')
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/microwave-hinge-visual-audit')
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'results/microwave-hinge-diagnosis'
    panels, facts, cards = [], [], []
    for task in ['open_microwave', 'close_microwave']:
        estimated = episode(base/'estimated'/f'{task}-state001')
        native = episode(base/'oracle'/f'{task}-state001')
        filename = f'{task}-hinges.mp4'
        video([estimated, native], ['Estimated hinge used', 'True hinge used'], output/filename)
        camera_name = 'frontview' if task == 'open_microwave' else 'agentview'
        meta = read(estimated['folder']/'at-grasp/metadata.json')
        step = meta['control_steps']
        with Image.open(estimated['folder']/'at-grasp'/f'{camera_name}.png') as rgb:
            panel = annotation(rgb, estimated, meta['cameras'][camera_name], step,
                               task.replace('_', ' ')+' | Before closure')
        panels.append(panel)
        target = np.asarray(estimated['target'])
        for stage in ['at-grasp', 'closed']:
            meta = read(estimated['folder']/stage/'metadata.json')
            step = meta['control_steps']
            action = estimated['actions'][step]
            contacts = read(estimated['folder']/'initial/evaluation_contact_diagnostics.json')['steps'][step-1]
            facts.append({'task': task, 'stage': stage, 'step': step,
                'requested_grasp_tcp_world_m': target.tolist(),
                'measured_tcp_world_m': meta['eef_pose_world_xyzw'][:3],
                'distance_to_requested_grasp_mm': float(np.linalg.norm(np.asarray(meta['eef_pose_world_xyzw'][:3])-target)*1000),
                'tcp_reached': action['tcp_pose_reached'],
                'finger_gap_mm': sum(abs(v) for v in meta['gripper_qpos'])*1000,
                'native_positive_contacts': [c for c in contacts['robot_involved_contacts'] if c['normal_force_n']>.1]})
        cards.append(f'<h2>{task.replace("_", " ")}</h2><video controls src="{filename}"></video>')
    closure = Image.new('RGB', (1024, 644))
    for index, panel in enumerate(panels):
        closure.paste(panel, (512*index, 0))
    closure.save(output/'preclosure-grasp-errors.png')
    hook = ROOT/'results/microwave-hook-contact-diagnosis/open_microwave-state001'
    if (hook/'summary.json').is_file():
        data = episode(hook)
        video([data], ['True hinge | pinch-width veto removed'], output/'hook-contact-opening.mp4')
        cards.append('<h2>Additional hook-contact diagnosis</h2><video controls src="hook-contact-opening.mp4"></video>')
    save = {'audit_only': True, 'native_geometry_used_only_for_overlay_or_explicit_oracle_runs': True,
            'legend': COLORS, 'hinge_axis_segment_length_m': .21,
            'axes_drawn_through_scene_without_occlusion': True,
            'tcp_is_controller_grip_site_not_fingertip_contact': True,
            'facts': facts,
            'interpretation': 'Original executor begins closure after timeout even when grasp TCP was not reached. Bilateral pinch contact is not a task-success requirement.',
            'vla_hook_explanation': 'Mechanically plausible; external published clip has no available contact/action logs to verify the exact mechanism.',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'audit.json').write_text(json.dumps(save, indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><html><meta charset="utf-8">'
        '<title>Hinge and grasp visual audit</title><style>body{font:17px system-ui;max-width:1100px;'
        'margin:30px auto;padding:20px;line-height:1.5}video,img{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Hinge axes and missed grasp poses</h1><p>Magenta: estimated hinge. Cyan: native hinge. '
        'Yellow circle: requested grasp TCP. Red cross: measured TCP. Lines show calibrated projections '
        'through the scene, including occluded axes. These are evaluation annotations.</p>'
        '<p>The original runner begins closure even when its grasp pose was not reached. '
        'A closed-gripper pinch is not required by LIBERO\'s opening predicate.</p>'
        '<img src="preclosure-grasp-errors.png"><p><a href="audit.json">Measurements and native contacts</a></p>'
        + ''.join(cards)+'</html>')
    print(json.dumps({'output': str(output), 'preclosure_pose_errors_mm': [
        f['distance_to_requested_grasp_mm'] for f in facts if f['stage']=='at-grasp'],
        'hook_video_included': (output/'hook-contact-opening.mp4').exists()}))


if __name__ == '__main__':
    main()
