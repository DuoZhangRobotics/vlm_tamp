"""Summarize evaluation-only physical contact pairs after an experiment."""

import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base, output = Path(args.input), Path(args.output)
    manifest = json.loads((base / 'manifest.json').read_text())
    episodes = []
    for task, state in manifest['planned_episodes']:
        folder = base / f'goal{task:02d}-state{state:03d}'
        summary = json.loads((folder / 'summary.json').read_text())
        diagnostic = json.loads((folder / 'initial/evaluation_contact_diagnostics.json').read_text())
        if not diagnostic['evaluation_only'] or not diagnostic['excluded_from_policy_input']:
            raise ValueError('Scene-identity contacts must remain evaluation-only')
        counts, peaks, unexpected = Counter(), {}, Counter()
        for step in diagnostic['steps']:
            for item in step['robot_involved_contacts']:
                pair = ' / '.join(item['geom_names'])
                counts[pair] += 1
                peaks[pair] = max(peaks.get(pair, 0.), float(item['normal_force_n']))
                if sum(item['robot_flags']) != 1:
                    continue
                robot = next(name for name, flag in zip(item['geom_names'], item['robot_flags']) if flag)
                scene = next(name for name, flag in zip(item['geom_names'], item['robot_flags']) if not flag)
                prefix = 'akita_black_bowl_1_' if task == 8 else 'wooden_cabinet_1_'
                if ('finger' not in robot or not scene.startswith(prefix)) and item['normal_force_n'] > 1e-6:
                    unexpected[pair] += 1
        episodes.append({'task': task, 'state': state, 'native_task_success': summary['official_task_success'],
            'contact_pairs': [{'pair': pair, 'contact_entries': count, 'peak_normal_force_n': peaks[pair]}
                              for pair, count in counts.most_common()],
            'non_finger_or_other_object_contact_entries_with_positive_force': dict(unexpected)})
    result = {'scope': 'post-execution evaluation diagnostics only; no input to policy, selection or control',
              'scene_identity_labels_evaluation_only': True,
              'drawer_cabinet_pairs_are_not_a_certificate_of_local_region_membership': True,
              'episodes': episodes,
              'successful_episodes_with_non_finger_or_other_object_contacts': sum(
                  row['native_task_success'] and bool(row['non_finger_or_other_object_contact_entries_with_positive_force'])
                  for row in episodes)}
    if output.exists():
        raise FileExistsError('Preserve contact audits')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'episodes'}))


if __name__ == '__main__':
    main()
