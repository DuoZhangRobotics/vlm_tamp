"""Summarize paired hinge trials and render local comparison videos."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def contacts_at_closure(folder):
    step = read(folder/'closed/metadata.json')['control_steps']
    actions = read(folder/'initial/actions.json')
    rows = read(folder/'initial/evaluation_contact_diagnostics.json')['steps']
    result = {'closure_step': step, 'gripper_gap_at_closure_m':
              sum(abs(v) for v in actions[step-1]['gripper_qpos']),
              'gripper_gap_at_final_m': sum(abs(v) for v in actions[-1]['gripper_qpos']),
              'native_joint_at_closure_rad': next(j['position'] for row in rows if row['step'] == step
                   for j in row['evaluation_only_articulation_state']
                   if j['joint_name'] == 'microwave_1_microjoint'),
              'microwave_contact_steps_after_closure': 0,
              'bilateral_finger_microwave_contact_steps_after_closure': 0,
              'object_level_contact_labels_do_not_certify_handle_membership': True}
    for row in rows:
        if row['step'] < step:
            continue
        names = set()
        for contact in row['robot_involved_contacts']:
            if contact['normal_force_n'] <= 1e-6:
                continue
            if any(name.startswith('microwave_1_') for name in contact['geom_names']):
                names.update(n for n in contact['geom_names'] if n.startswith('gripper0_'))
        result['microwave_contact_steps_after_closure'] += bool(names)
        both = all(any(f'finger{i}' in n for n in names) for i in [1, 2])
        result['bilateral_finger_microwave_contact_steps_after_closure'] += both
    return result


def duration(path):
    command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)]
    return float(json.loads(subprocess.check_output(command))['format']['duration'])


def comparison_video(left, right, output):
    total = max(duration(left), duration(right))
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    chains = []
    for index, caption in enumerate(['Estimated hinge', 'True hinge - diagnostic']):
        chains.append(f'[{index}:v]fps=4,pad=iw:ih+42:0:42:black,'
                      f'drawtext=fontfile={font}:text=\'{caption}\':fontcolor=white:'
                      'fontsize=20:x=12:y=10,tpad=stop_mode=clone:stop_duration=120'
                      f'[v{index}]')
    chains.append('[v0][v1]hstack=inputs=2[v]')
    command = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', str(left), '-i', str(right),
               '-filter_complex', ';'.join(chains), '-map', '[v]', '-t', str(total),
               '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
               '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/microwave-hinge-diagnosis')
    parser.add_argument('--output', default='results/microwave-hinge-diagnosis-report')
    args = parser.parse_args()
    base, output = ROOT/args.run, ROOT/args.output
    manifest, protocol = read(base/'manifest.json'), read(base/'protocol.json')
    if len(read(base/'summary.json')['episodes']) != len(manifest['planned_episodes']):
        raise ValueError('Every planned episode must finish before reporting')
    output.mkdir(parents=True, exist_ok=False)
    records, groups = [], {}
    for row in manifest['planned_episodes']:
        folder = base/row['hinge_mode']/f'{row["task"]}-state001'
        summary, audit = read(folder/'summary.json'), read(folder/'paired-audit.json')
        record = {**summary, 'paired_audit': audit,
                  'physical_contact_audit': contacts_at_closure(folder),
                  'episode': str(folder.relative_to(ROOT)),
                  'geometry_model_uses_simulator_hinge': row['hinge_mode'] == 'oracle'}
        record['rgbd_feedback_with_diagnostic_scope'] = [
            {**feedback, 'simulator_scene_state_used': row['hinge_mode'] == 'oracle',
             'scope_note': 'RGB-D measurements fitted using the selected estimated/oracle hinge model'}
            for feedback in read(folder/'mechanism-feedback.json')]
        records.append(record)
        groups.setdefault(row['task'], {})[row['hinge_mode']] = record
    charts, cards, paired = [], [], []
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for task, modes in groups.items():
        figure, axes = plt.subplots(1, 1, figsize=(7.5, 3.5), constrained_layout=True)
        for mode, record in modes.items():
            folder = ROOT/record['episode']
            states = read(folder/'initial/evaluation_contact_diagnostics.json')['steps']
            angles = [abs(np.degrees(next(j['position'] for j in s['evaluation_only_articulation_state']
                                         if j['joint_name'] == 'microwave_1_microjoint'))) for s in states]
            axes.plot([s['step'] for s in states], angles, label=mode)
        threshold = np.degrees(1.3 if task == 'open_microwave' else .005)
        axes.axhline(threshold, color='gray', linestyle='--', label='Native success boundary')
        axes.axvline(modes['estimated']['paired_audit']['baseline_closure_control_step'],
                     color='gray', linestyle=':', label='Grasp closure completed')
        axes.set(xlabel='Control step', ylabel='Native door opening angle (degrees)',
                 title=task.replace('_', ' ').capitalize())
        axes.grid(alpha=.2)
        axes.legend(fontsize=8)
        chart = output/f'{task}-native-angle.png'
        figure.savefig(chart, dpi=150)
        plt.close(figure)
        charts.append(chart.name)
        video = output/f'{task}-comparison.mp4'
        comparison_video(ROOT/modes['estimated']['episode']/'initial/rollout.mp4',
                         ROOT/modes['oracle']['episode']/'initial/rollout.mp4', video)
        pair = {'task': task, 'estimated_native_success': modes['estimated']['official_task_success'],
                'oracle_native_success': modes['oracle']['official_task_success'],
                'estimated_final_open_angle_deg': modes['estimated']['paired_audit']['native_final_open_angle_deg'],
                'oracle_final_open_angle_deg': modes['oracle']['paired_audit']['native_final_open_angle_deg']}
        paired.append(pair)
        cards.append(f'<article><h2>{html.escape(task.replace("_", " "))}</h2>'
                     f'<video controls preload="metadata" src="{video.name}"></video>'
                     f'<img src="{chart.name}" alt="Native door angle versus control steps">'
                     f'<pre>{html.escape(json.dumps(pair, indent=2))}</pre></article>')
    unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == sha
                    for p, sha in protocol['sources_sha256'].items())
    prefix = all(r['paired_audit']['all_approach_and_closure_steps_executed']
                 and r['paired_audit']['maximum_prefix_joint_state_difference_rad'] < 1e-10
                 and r['paired_audit']['maximum_prefix_pose_command_difference'] == 0 for r in records)
    ledger_unchanged = hashlib.sha256((ROOT/'results/articulated-api-v1/ledger.json').read_bytes()).hexdigest() == protocol['api_ledger_sha256_before']
    report = {'planned_episodes': len(manifest['planned_episodes']), 'completed_episodes': len(records),
              'diagnostic_only': True, 'benchmark_success_rates_replaced': False,
              'all_grasp_approach_and_closure_prefixes_identical_to_baseline': prefix,
              'frozen_sources_unchanged': unchanged, 'api_ledger_unchanged': ledger_unchanged,
              'new_api_calls': 0, 'paired_results': paired, 'episodes': records,
              'limitations': ['One initial state per task; not a generalization evaluation',
                              'Oracle intentionally supplies native hinge geometry',
                              'Scene collision predictions do not veto this physics diagnostic',
                              'Raw component feedback inherits sensor-only metadata; normalized feedback above marks oracle geometry use',
                              'Tracking and gripper-gap gates are retained',
                              'A failed oracle trial does not establish hinge geometry is irrelevant']}
    published = [
        {'model': '3DThinkVLA', 'task': 'Open the middle drawer of the cabinet',
         'setting': 'LIBERO simulation, authors published episode 14',
         'source_page': 'https://3dthinkvla.github.io/',
         'video_url': 'https://3dthinkvla.github.io/SImResults/LIBERO/rollout_open_the_middle_drawer_of_the_cabinet_episode14_success.mp4'},
        {'model': 'FLOWER', 'task': 'Open the microwave',
         'setting': 'Real robot, authors representative evaluation video',
         'source_page': 'https://intuitive-robots.github.io/flower_vla/',
         'video_url': 'https://intuitive-robots.github.io/flower_vla/static/video/real_vids/open_microwave/primary_camera.mp4'},
    ]
    (output/'published-vla-videos.json').write_text(json.dumps({
        'external_author_rollouts': published, 'run_locally': False,
        'matched_state_comparison': False, 'remote_media_downloaded': False}, indent=2)+'\n')
    (output/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    html_text = ('<!doctype html><html><meta charset="utf-8"><title>Microwave hinge diagnosis</title>'
                 '<style>body{font:17px system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.5}'
                 'video,img{width:100%}article{margin:35px 0}pre{white-space:pre-wrap}</style>'
                 '<h1>Microwave hinge diagnosis</h1><p>Same official state, fixed original grasp, IK posture, '
                 'and approach. The oracle changes only hinge axis and location. Native physics, tracking '
                 'and retention gates remain active. These four trials are diagnostics, not sensor-only '
                 'benchmark results.</p><p>Videos show estimated hinge on the left and true hinge on the '
                 'right. When one rollout stops first, its last frame is held for comparison.</p>'
                 f'<p>Identical approach/closure motion: {prefix}. Original sources unchanged: {unchanged}. '
                 f'No model requests: {ledger_unchanged}.</p><a href="evaluation.json">Full measurements</a>'
                 + ''.join(cards)+'<h2>Published VLA opening examples</h2>'
                 '<p>These are external authors\' clips, not locally run or matched-state comparisons.</p>'
                 + ''.join(f'<article><h3>{item["model"]}: {item["task"]}</h3>'
                           f'<p>{item["setting"]}. <a href="{item["source_page"]}">Authors\' source</a></p>'
                           f'<video controls preload="metadata" src="{item["video_url"]}"></video></article>'
                           for item in published)+'</html>')
    (output/'index.html').write_text(html_text)
    if not (unchanged and prefix and ledger_unchanged):
        raise ValueError('Diagnostic integrity check failed; report preserves the discrepancy')
    print(json.dumps({k: report[k] for k in ['paired_results',
        'all_grasp_approach_and_closure_prefixes_identical_to_baseline', 'frozen_sources_unchanged', 'api_ledger_unchanged']}))


if __name__ == '__main__':
    main()
