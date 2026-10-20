"""Post-run FK, joint-reserve, contact, and native-success evidence.

This audit reads simulator scene contact labels only after episodes finish.
They are never fed back to grasp selection or control.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', help='Append audit links to this generated HTML report')
    args = parser.parse_args()
    base, output = Path(args.input), Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing audits')
    configuration = read(base/'configuration.json')
    rows, frames = [], []
    for task, state in read(base/'manifest.json')['planned_episodes']:
        folder = base/f'goal{task:02d}-state{state:03d}'
        summary, commands = read(folder/'summary.json'), read(folder/'command-log.json')
        limits = np.asarray(read(folder/'robot-chain.json')['joint_limits'])
        frame = read(folder/'curobo-frame-validation.json')
        frames.extend(frame['samples'])
        chosen = read(folder/'chosen-grasp.json') if (folder/'chosen-grasp.json').exists() else None
        queries = read(folder/'curobo-queries.json')
        command_reserve = [float(np.minimum(np.asarray(c['requested_joint_posture'])-limits[:, 0],
                            limits[:, 1]-np.asarray(c['requested_joint_posture'])).min()) for c in commands]
        actual_reserve = [float(np.minimum(np.asarray(c['actual_joints'])-limits[:, 0],
                            limits[:, 1]-np.asarray(c['actual_joints'])).min()) for c in commands]
        last_check = read(folder/'collision-checks.json')[-1] if (folder/'collision-checks.json').exists() else None
        actual_boundaries = [c for c in read(folder/'collision-checks.json') if c['kind']=='actual_action_boundary'] if last_check else []
        contacts = read(folder/'initial/evaluation_contact_diagnostics.json')
        if not contacts['evaluation_only'] or not contacts['excluded_from_policy_input']:
            raise ValueError('Simulator scene contact identities must be evaluation-only')
        final_contacts = contacts['steps'][-1]['robot_involved_contacts'] if contacts['steps'] else []
        native = [a['evaluation_task_success'] for a in read(folder/'initial/actions.json')]
        assert any(native) == summary['official_task_success']
        row = {'task': task, 'state': state, 'native_success': summary['official_task_success'],
            'observed_camera_names': list(read(folder/'opened/metadata.json')['cameras']),
            'failure_stage': summary['failure_stage'], 'failure': summary.get('failure'),
            'fk_frame_check_passed': frame['accepted'],
            'selected_grasp_matches_approach_endpoint': (chosen is None or
                np.array_equal(chosen['solution_joints'], chosen['approach']['joint_postures'][-1])),
            'minimum_command_joint_reserve_rad': min(command_reserve, default=None),
            'minimum_measured_joint_reserve_at_command_boundary_rad': min(actual_reserve, default=None),
            'command_joint_reserve_passed': not command_reserve or min(command_reserve)>=configuration['command_joint_reserve_rad']-1e-5,
            'maximum_tcp_command_tracking_error_m': max((float(np.linalg.norm(np.asarray(c['requested_pose_world_xyzw'])[:3]-np.asarray(c['actual_pose_world_xyzw'])[:3])) for c in commands),default=None),
            'gpu_ik_queries': len(queries), 'gpu_ik_seconds': sum(q['elapsed_seconds'] for q in queries),
            'peak_curobo_gpu_allocated_bytes': max((q['peak_gpu_allocated_bytes'] for q in queries),default=0),
            'maximum_distinct_robot_valid_ik_in_one_query': max((q['distinct_robot_valid'] for q in queries),default=0),
            'last_collision_check': last_check, 'actual_boundaries_checked': len(actual_boundaries),
            'evaluation_only_final_native_contacts': final_contacts}
        rows.append(row)
    checks = {'all_frame_checks_passed': all(r['fk_frame_check_passed'] for r in rows),
              'all_approaches_end_at_admitted_posture': all(r['selected_grasp_matches_approach_endpoint'] for r in rows),
              'all_commands_have_joint_reserve': all(r['command_joint_reserve_passed'] for r in rows)}
    audit = {'scope': 'post-execution analysis; simulator contact identities excluded from policy input',
             'auditor_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'checks': checks, 'all_checks_passed': all(checks.values()),
             'frame_samples': len(frames),
             'maximum_frame_position_error_m': max(f['position_error_m'] for f in frames),
             'maximum_frame_orientation_error_rad': max(f['orientation_error_rad'] for f in frames),
             'maximum_isolated_sim_position_error_m': max(f.get('isolated_sim_position_error_m',0) for f in frames),
             'maximum_isolated_sim_orientation_error_rad': max(f.get('isolated_sim_orientation_error_rad',0) for f in frames),
             'episodes': rows, 'collision_free_guarantee': False}
    output.write_text(json.dumps(audit,indent=2)+'\n')
    if args.report:
        report = Path(args.report)
        document = report.read_text()
        if 'id="curobo-execution-audit"' in document:
            raise ValueError('Report already contains this audit')
        contacts_path = output.parent/'executed-contact-audit.json'
        contacts_audit = read(contacts_path)
        count = contacts_audit['successful_episodes_with_non_finger_or_other_object_contacts']
        successes = sum(row['native_success'] for row in rows)
        note = (f'{count} of {successes} successful episodes had positive-force non-finger or other-object '
                'robot contacts under the post-execution object-label audit. Cabinet-level labels do not '
                'establish contact with the exact selected handle region.')
        block = ('<div class="box" id="curobo-execution-audit"><b>Measured execution checks</b>'
            f'<p>{len(frames)} GPU FK samples; maximum position mismatch '
            f'{audit["maximum_frame_position_error_m"]*1e6:.3f} micrometres. '
            'Approach endpoint consistency and the 0.025 rad command reserve were audited independently.</p>'
            f'<p>{html.escape(note)} Simulator scene contact labels were used only for this post-run audit.</p>'
            f'<p>Observed reconstruction uses {len(rows[0]["observed_camera_names"])} calibrated RGB-D views, '
            'including extra robot-coverage views, as in v11. This is a multiview simulation adaptation; '
            'simulator object meshes and poses are excluded from planning.</p>'
            f'<p><a href="{html.escape(output.name)}">IK and execution audit</a> · '
            '<a href="executed-contact-audit.json">Physical contact audit</a></p></div>')
        report.write_text(document.replace('</body>', block+'</body>'))
    print(json.dumps({k:v for k,v in audit.items() if k!='episodes'}))
    if not audit['all_checks_passed']:
        raise ValueError('cuRobo execution invariants failed')


if __name__ == '__main__':
    main()
