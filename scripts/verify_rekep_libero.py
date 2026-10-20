"""Audit actual episode evidence, frozen inputs, API accounting and preservation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run_dir='results/rekep-libero-v1', report_dir='results/rekep-libero-report'):
    folder = ROOT / run_dir
    plan = json.loads((folder / 'planned-episodes.json').read_text())
    frozen = json.loads((folder / 'frozen-protocol.json').read_text())
    preserved = json.loads((ROOT / 'results/rekep-libero-setup/preserved_files.json').read_text())
    changed = [name for name, expected in frozen['files_sha256'].items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != expected]
    old_changed = [name for name, expected in preserved.items()
                   if not (ROOT / name).is_file() or sha(ROOT / name) != expected]
    assert not changed, 'Frozen files changed: ' + repr(changed)
    assert not old_changed, 'Earlier artifacts changed: ' + repr(old_changed)
    assert len(plan['evaluation']) == 10
    counts, failures, rejections = Counter(), Counter(), Counter()
    rows = []
    for name in plan['evaluation']:
        episode = Path(name)
        assert (episode / 'episode.json').is_file(), 'Missing planned episode: ' + episode.name
        row = json.loads((episode / 'episode.json').read_text())
        assert row['development'] is False and row['initial_state'] in [1, 11, 21, 31, 41]
        assert row['seed'] == 0 and 0 <= row['control_steps'] <= 600
        assert not row['ground_truth_used_by_policy'] and not row['provided_inventory'] and not row['assisted_grasping']
        assert row['video'] and Path(row['video']).is_file()
        actions = json.loads((episode / 'initial/actions.json').read_text())
        assert len(actions) == row['control_steps']
        actual_success = any(action['evaluation_task_success'] for action in actions)
        assert row['success'] == actual_success, 'Success lacks physical task-predicate evidence'
        counts['episodes'] += 1
        counts['physical_control_steps'] += row['control_steps']
        counts['official_successes'] += int(row['success'])
        counts['physical_grasp_attempts'] += row['physical_grasp_attempts']
        counts['perception_completed'] += int((episode / 'perception/perception.json').exists())
        counts['program_received'] += int((episode / 'program/response.json').exists())
        counts['program_validated'] += int((episode / 'program/metadata.json').exists())
        counts['planned_motion_calls'] += len(list(episode.glob('stage*-*-plan.json')))
        for path in episode.glob('stage*-grasps/grasp_generation.json'):
            generated = json.loads(path.read_text())
            assert generated['requested_candidates'] == 200 and generated['retained_candidates'] <= 100
            assert generated['tries'] == 1 and not generated['contacts_provided_by_model']
            counts['grasp_generation_runs'] += 1
            counts['raw_grasp_candidates'] += generated['generated_candidates']
        for path in episode.glob('stage*-grasps/grasp_selection.json'):
            selected = json.loads(path.read_text())
            rejections.update(x['reason'] for x in selected['rejected_candidates'])
            counts['feasible_grasp_candidates'] += len(selected['feasible_candidates'])
        if not row['success']:
            failures[row['failure_stage']] += 1
        rows.append({k: row.get(k) for k in ('episode_id', 'success', 'failure_stage', 'failure_detail',
                                            'control_steps', 'physical_grasp_attempts')})
    ledger = json.loads((ROOT / 'results/rekep-libero-api/ledger.json').read_text())
    assert ledger['api_attempts'] <= 12 and ledger['cumulative_conservative_spend_usd'] <= 2
    for entry in ledger['requests'].values():
        request = json.loads((ROOT / 'results/rekep-libero-api' / ('request-' + entry['request_sha256'] + '.json')).read_text())
        assert request['model'] == 'gpt-6-luna' and request['reasoning'] == {'effort': 'high'}
        assert request['max_output_tokens'] in [6000, 12000] and request['store'] is False
        assert len(request['input'][0]['content']) == 2
        canonical = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False)
        assert hashlib.sha256(canonical.encode()).hexdigest() == entry['request_sha256']
    result = {'verification_passed': True, 'frozen_files_verified': len(frozen['files_sha256']),
              'older_files_preserved': len(preserved), 'counts': dict(counts),
              'failure_counts': dict(failures), 'grasp_rejection_counts': dict(rejections),
              'api_attempts': ledger['api_attempts'],
              'cumulative_conservative_spend_usd': ledger['cumulative_conservative_spend_usd'],
              'new_conservative_spend_usd': ledger['experiment_conservative_spend_usd'],
              'episodes': rows, 'success_criterion': 'Recorded physical LIBERO task-success predicate',
              'limitation': 'Failures before manipulation remain in denominator; no full-method ranking follows'}
    destination = ROOT / report_dir / 'verification.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    report = destination.with_name('index.html')
    if report.exists():
        document = report.read_text()
        start_marker, end_marker = '<!-- evidence-audit-start -->', '<!-- evidence-audit-end -->'
        if start_marker in document:
            before, remaining = document.split(start_marker, 1)
            _, after = remaining.split(end_marker, 1)
            document = before + after
        note = (start_marker + '<section><h2>Verified pipeline coverage</h2><p>'
                f"{counts['program_validated']}/10 programs validated; "
                f"{counts['grasp_generation_runs']}/10 episodes generated model grasps; "
                f"{counts['feasible_grasp_candidates']} feasible candidates survived filtering; "
                f"{counts['physical_grasp_attempts']} physical grasp attempts; "
                f"{counts['planned_motion_calls']} task motion-planning calls.</p>"
                '<p>Failures before manipulation measure where this adaptation stops. They do not establish '
                "ReKep's performance with a completed perception/control stack or its ranking against VLAs. "
                'The development diagnostic found unobserved moving-arm space under the declared cameras; '
                'unknown space remained blocked after exact current-robot exclusion.</p>'
                '<p><a href="verification.json">Independent evidence and preservation audit</a></p></section>'
                + end_marker)
        report.write_text(document.replace('</html>', note + '</html>'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/rekep-libero-v1')
    parser.add_argument('--report', default='results/rekep-libero-report')
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.report), indent=2))
