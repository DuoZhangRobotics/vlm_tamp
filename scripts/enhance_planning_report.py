"""Add observed-input evidence and early-stop movies without changing policy data."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def failure_category(row):
    if row['native_success_ever']:
        return 'native success'
    stage = row.get('failure_stage') or ''
    message = row.get('failure') or ''
    if stage == 'grounding':
        return 'incomplete API response' if 'did not complete' in message else 'task grounding'
    if stage == 'sensor_geometry':
        return 'RGB-D contact / hinge geometry'
    if stage == 'complete_motion_grasp_selection':
        return 'grasp admission / initial path planning'
    if 'replan' in stage:
        return 'contact path replanning'
    if stage == 'physical_grasp':
        return 'physical grasp / retention'
    if stage == 'approach':
        return 'grasp approach execution'
    if stage == 'planned_joint_tracking':
        return 'trajectory tracking / scene guard'
    return stage or 'goal estimate / execution'


def role_svg(choices, filename):
    def rectangle(box, color):
        x0, y0, x1, y1 = map(float, box)
        return f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="{color}" stroke-width="3"/>'
    shapes = rectangle(choices['moving_panel_box_xyxy'], '#41ceee')
    shapes += rectangle(choices['contact_box_xyxy'], '#ffd44f')
    points = choices.get('hinge_pixels_xy') or []
    if points:
        line = ' '.join(f'{float(x)},{float(y)}' for x, y in points)
        shapes += f'<polyline points="{line}" fill="none" stroke="#fa71de" stroke-width="3"/>'
    x, y = choices['grasp_pixel_xy']
    shapes += f'<circle cx="{float(x)}" cy="{float(y)}" r="5" fill="#ffd44f" stroke="#111" stroke-width="1"/>'
    return ('<svg viewBox="0 0 512 512" style="max-width:512px;width:100%" role="img" '
            'aria-label="VLM selected contact, panel and hinge pixels">'
            f'<image href="{html.escape(filename, quote=True)}" width="512" height="512"/>{shapes}</svg>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--diagnostics')
    args = parser.parse_args()
    output = ROOT/args.report
    data = json.loads((output/'evaluation.json').read_text())
    cards, media, counts = [], [], Counter()
    retained = reclaimed = 0
    for row in data['episodes']:
        folder = (ROOT/row['episode']).resolve()
        if not folder.is_relative_to(ROOT/'results'):
            raise ValueError('Report episode must be inside project results')
        task, state = row['task'], row['initial_state']
        stem = f'{task}-state{state:03d}'
        category = failure_category(row)
        counts[(row['split'], category)] += 1
        payload = {'task': task, 'state': state, 'split': row['split'], 'category': category}
        body = f'<p>{html.escape(row.get("failure") or row.get("status", ""))}</p>'
        ground = folder/'grounding/grounding.json'
        if ground.exists():
            record = json.loads(ground.read_text())
            choices = record['choices']
            camera = choices['camera']
            source = folder/'grounding'/camera/'rgb.png'
            target = output/(stem+'-rgb.png')
            if source.exists():
                shutil.copyfile(source, target)
                body += role_svg(choices, target.name)
                body += '<p>Gold: selected contact and grasp pixel. Cyan: selected door box. Magenta: selected hinge edge.</p>'
            payload.update(camera=camera, choices=choices,
                           provided_inventory=record.get('provided_inventory'),
                           simulator_scene_state_used=record.get('simulator_scene_state_used'))
        else:
            source = folder/'grounding/role-choice-sheet.png'
            if source.exists():
                target = output/(stem+'-input.png')
                shutil.copyfile(source, target)
                body += f'<p>No completed role selection. The RGB input sheet is shown below.</p><img src="{target.name}">'
        movie = output/(stem+'.mp4')
        raw = folder/'initial/rollout.mp4'
        if not movie.exists() and raw.exists():
            shutil.copyfile(raw, movie)
            body += f'<p>Episode stopped before a task model was available; raw observation video:</p><video controls preload="metadata" src="{movie.name}"></video>'
            payload['raw_early_stop_movie'] = movie.name
        if movie.exists():
            payload['movie'] = movie.name
        retention = folder/'retention.json'
        if retention.exists():
            storage = json.loads(retention.read_text())
            retained += storage['bytes_after_cleanup']
            reclaimed += storage['bytes_reclaimed_this_pass'] + storage.get('prior_bytes_reclaimed', 0)
        cards.append(f'<details><summary>{html.escape(task)} · state {state} · {row["split"]} · '
                     f'{html.escape(category)}</summary>{body}</details>')
        media.append(payload)
    rows = ''.join(f'<tr><td>{split}</td><td>{html.escape(category)}</td><td>{number}</td></tr>'
                   for (split, category), number in sorted(counts.items()))
    diagnostic_html = ''
    diagnostic_files = []
    if args.diagnostics:
        source_root = ROOT/args.diagnostics
        for name in ['native-open-state011/summary.json', 'native-open-state011/configurations.json',
                     'close-state011-handle.json', 'grasp-admission.json']:
            source = source_root/name
            if not source.exists():
                continue
            target = output/('diagnostic-'+name.replace('/', '-'))
            shutil.copyfile(source, target)
            diagnostic_files.append(target.name)
        native = source_root/'native-open-state011/summary.json'
        if native.exists():
            audit = json.loads(native.read_text())
            c = audit['counts']
            diagnostic_html += ('<h3>Collision-model audit: opening state 11</h3>'
                f'<p>After replaying five settling actions and all {audit["setup_actions_replayed"]} setup actions, '
                f'an isolated simulator-state copy checked {c["queried"]} rejected grasp configurations. '
                f'{c["native_no_robust_contacts"]} had no native penetration deeper than 0.1mm. '
                f'All {c["inferred_only_native_clear"]} of {c["inferred_only_queried"]} configurations rejected solely by the inferred panel volume were clear under that check.</p>'
                '<p>This confirms conservative admission rejections in this scene. The penetration audit does not certify the 3mm planning clearance, a collision-free approach or a successful grasp. '
                'The audit executed no grasps, changed no benchmark outcomes and supplied no native geometry to the policy.</p>')
        handle = source_root/'close-state011-handle.json'
        if handle.exists():
            audit = json.loads(handle.read_text())
            diagnostic_html += ('<h3>Selected-surface audit: closing state 11</h3>'
                f'<p>The selected surface had {1000*audit["observed_support_height_m"]:.1f}mm of robust height, '
                f'against {1000*audit["required_full_pad_height_with_margins_m"]:.1f}mm required by the finite-pad check. '
                'No grasp was attempted. Inspect the selected RGB contact box below.</p>')
        diagnostic_html += '<p>'+ ' · '.join(f'<a href="{name}">{name}</a>' for name in diagnostic_files)+'</p>'
    block = ('<!-- observed-input-evidence:start --><section><h2>Failure causes and observed inputs</h2>'
             '<p>These overlays show the actual RGB selections used by the frozen policy. '
             'They contain no simulator object labels or poses. API completion failures are separate from motion failures.</p>'
             '<table><tr><th>Split</th><th>Outcome / failure phase</th><th>Count</th></tr>'+rows+'</table>'
             f'<p>Retained episode payloads: {retained/2**30:.2f} GiB. Intermediate data reclaimed automatically: {reclaimed/2**30:.2f} GiB.</p>'+diagnostic_html
             +''.join(cards)+'<p><a href="media-manifest.json">Input evidence, failure grouping and rendering provenance</a></p>'
             '</section><!-- observed-input-evidence:end -->')
    index = output/'index.html'
    document = index.read_text()
    if '<!-- observed-input-evidence:start -->' in document:
        raise ValueError('This report already has the observed-input evidence section')
    index.write_text(document.replace('</html>', block+'</html>'))
    record = {'policy_data_changed': False, 'all_planned_cases_preserved': True,
              'postprocessor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_evaluation_sha256': hashlib.sha256((output/'evaluation.json').read_bytes()).hexdigest(),
              'evaluation_only_diagnostic_files': diagnostic_files,
              'retained_episode_bytes': retained, 'reclaimed_intermediate_bytes': reclaimed,
              'failure_counts': [{'split': split, 'category': category, 'count': count}
                                 for (split, category), count in sorted(counts.items())], 'episodes': media}
    (output/'media-manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    shutil.copyfile(__file__, output/'render-observed-inputs.py')
    print(json.dumps({'media_episodes': len(media), 'failure_counts': record['failure_counts'],
                      'retained_GiB': round(retained/2**30, 2)}))


if __name__ == '__main__':
    main()
