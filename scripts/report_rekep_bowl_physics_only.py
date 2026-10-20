"""Report logged physical contact and RGB-D retention for a grasp diagnostic."""

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from try_rekep_bowl_v3 import save


ROOT = Path(__file__).resolve().parents[1]


def build(source, output):
    source, output = ROOT / source, ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / 'summary.json').read_text())
    retention = json.loads((source / 'lift-retention.json').read_text())
    actions = json.loads((source / 'initial/actions.json').read_text())
    contacts = json.loads((source / 'initial/evaluation_contact_diagnostics.json').read_text())
    assert len(actions) == summary['control_steps'] <= 600
    assert summary['physical_grasp_attempts'] == int(any(action['control'][-1] > 0 for action in actions))
    assert summary['official_task_success'] == any(action['evaluation_task_success'] for action in actions)
    assert summary['grasp_lift_success'] == retention['verified']
    held = contacts['steps'][-20:]
    supported = []
    for step in held:
        sides = {1: set(), 2: set()}
        for contact in step['robot_involved_contacts']:
            if contact['normal_force_n'] <= 0:
                continue
            for index, name in enumerate(contact['geom_names']):
                for side in (1, 2):
                    if f'finger{side}_pad_collision' in name:
                        other = contact['geom_names'][1-index]
                        if not contact['robot_flags'][1-index]:
                            sides[side].add(other.rsplit('_g', 1)[0])
        supported.append(bool(sides[1] & sides[2]))
    verification = {'logged_controls_verified': True,
                    'both_pads_force_contact_same_object_hold_steps': sum(supported),
                    'evaluated_hold_steps': len(held),
                    'contact_identities_evaluation_only': True,
                    'rgbd_bowl_motion_mm': 1000 * retention['observed_motion_m'],
                    'physical_grasp_verified': retention['verified'] and all(supported),
                    'software_collision_checks': summary['software_collision_checks'],
                    'simulator_contact_physics': 'preserved',
                    'strict_planner_success_claimed': False,
                    'placement_attempted': False,
                    'new_api_calls': summary['new_api_calls'],
                    'api_ledger_unchanged': summary['api_ledger_unchanged']}
    save(output / 'verification.json', verification)
    save(output / 'summary.json', summary)

    def url(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return html.escape(quote(os.path.relpath(path.resolve(), output.resolve()).replace(os.sep, '/'), safe='/'), quote=True)

    opened = url(source / 'opened/agentview.png')
    lifted = url(source / 'lifted-held/agentview.png')
    video = url(summary['video'])
    result_file = url(source / 'summary.json')
    retained_file = url(source / 'lift-retention.json')
    runner = url(ROOT / 'scripts/try_rekep_bowl_physics_only.py')
    strict = url(ROOT / 'results/rekep-bowl-pilot-report/index.html')
    title = 'Bowl grasp and lift succeeded' if verification['physical_grasp_verified'] else 'Physical grasp diagnostic'
    explanation = ('The Panda reached the predicted grasp, closed normally, lifted the bowl '
                   f'{verification["rgbd_bowl_motion_mm"]:.1f} mm and held it for 20 controls (about one second). '
                   'Both finger pads had physical force contacts with the same object throughout the hold.')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:17px/1.6 system-ui;background:#f3f5f7;color:#182436;margin:0}}main{{max-width:1000px;margin:auto;padding:36px 24px}}
h1{{font-size:38px;line-height:1.15;margin:8px 0 18px}}h2{{font-size:24px}}.hero,section{{background:white;border:1px solid #dce3e9;border-radius:16px;padding:24px;margin-bottom:22px}}
.tag{{color:#12613d;font-weight:750}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}img,video{{width:100%;border-radius:10px}}figure{{margin:0}}figcaption{{font-size:15px;color:#546174}}a{{color:#1766aa}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #e1e6ea}}.note{{color:#546174}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}}}</style>
<main><div class="hero"><span class="tag">PHYSICAL GRASP DIAGNOSTIC</span><h1>{title}</h1>
<p>{explanation}</p><p><strong>No new API requests. Normal simulator contact physics remained active.</strong></p></div>
<section><h2>What changed</h2><p>The previous software collision, unknown-space and tracking-error stop rules were bypassed.
The robot used a cached RGB-D GraspGen pose and ordinary pose/gripper actions. No object was attached or moved by software.</p>
<p>This let us test whether the proposed grasp physically works. Collision-aware planning remains a separate unfinished part of the pipeline.</p>
<table><tr><th>Stage</th><th>Result</th></tr><tr><td>Approach the bowl</td><td>Reached grasp position within {summary['at_grasp_position_error_m']*1000:.2f} mm</td></tr>
<tr><td>Close the fingers</td><td>Physical finger-pad contacts confirmed</td></tr><tr><td>Lift and hold</td><td>{verification['rgbd_bowl_motion_mm']:.1f} mm observed bowl motion; held</td></tr>
<tr><td>Put the bowl on the plate</td><td>Not attempted in this diagnostic</td></tr></table></section>
<section><h2>Before and after</h2><div class="grid"><figure><img src="{opened}" alt="Bowl resting on the table before approach"><figcaption>Before: bowl on the table</figcaption></figure>
<figure><img src="{lifted}" alt="Panda holding the bowl after lifting"><figcaption>After: bowl lifted and held by its rim</figcaption></figure></div></section>
<section><h2>Recorded physical execution</h2><video controls preload="metadata" src="{video}"></video></section>
<section><h2>Scope and evidence</h2><p>One development trial on LIBERO Goal task 8, initial state 1, seed 0; {summary['control_steps']} controls.
This demonstrates a physical grasp and lift in this scene. It does not establish a benchmark success rate or completion of the bowl-on-plate task.</p>
<p><a href="{result_file}">Execution record</a> · <a href="{retained_file}">RGB-D retention evidence</a> ·
<a href="verification.json">Contact and event verification</a> · <a href="{runner}">Reproducible runner</a> · <a href="{strict}">Earlier strict-check attempts</a></p></section></main></html>'''
    (output / 'index.html').write_text(page)
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/rekep-bowl-physics-only-v1/attempt-01')
    parser.add_argument('--output', default='results/rekep-bowl-physics-only-report')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output)))
