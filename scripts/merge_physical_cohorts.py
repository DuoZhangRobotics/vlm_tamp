"""Combine disjoint experiment partitions with identical frozen sources/config."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    parts, output = [Path(p).resolve() for p in args.inputs], Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Preserve experiment outputs')
    config, protocol = read(parts[0] / 'configuration.json'), read(parts[0] / 'protocol.json')
    episodes, planned = [], []
    for part in parts:
        if read(part / 'configuration.json') != config:
            raise ValueError('Partition configurations differ')
        candidate = read(part / 'protocol.json')
        if candidate['sources_sha256'] != protocol['sources_sha256']:
            raise ValueError('Partition frozen sources differ')
        for name, digest in protocol['sources_sha256'].items():
            if hashlib.sha256((part / 'source-snapshot' / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Partition source snapshot was changed')
        for task, state in read(part / 'manifest.json')['planned_episodes']:
            pair = [task, state]
            if pair in planned:
                raise ValueError('Duplicate episode across partitions')
            planned.append(pair)
            summary = read(part / f'goal{task:02d}-state{state:03d}/summary.json')
            episodes.append(summary)
        if not read(part / 'summary.json')['api_ledger_unchanged']:
            raise ValueError('A partition changed the API ledger')
    if sorted(planned) != sorted([[task, state] for task in config['tasks'] for state in config['initial_states']]):
        raise ValueError('Partitions do not cover the full official cohort')
    output.mkdir(parents=True)
    for part in parts:
        for task, state in read(part / 'manifest.json')['planned_episodes']:
            name = f'goal{task:02d}-state{state:03d}'
            (output / name).symlink_to(part / name, target_is_directory=True)
    shutil.copytree(parts[0] / 'source-snapshot', output / 'source-snapshot')
    protocol['partitioned_execution'] = {'inputs': [str(p) for p in parts],
        'same_configuration_and_source_hashes': True, 'episode_outputs_rewritten': False,
        'scheduling': 'two independent simulator processes, one task per partition'}
    write(output / 'protocol.json', protocol)
    write(output / 'configuration.json', config)
    write(output / 'manifest.json', {'planned_episodes': planned, 'all_in_denominator': True,
          'seed': config['seed'], 'maximum_control_steps': config['maximum_control_steps']})
    write(output / 'summary.json', {'planned_episodes': len(planned), 'completed_episodes': len(episodes),
          'official_task_successes': sum(row['official_task_success'] for row in episodes),
          'api_ledger_unchanged': True, 'episodes': episodes})
    print(json.dumps({'episodes': len(episodes), 'native_successes': sum(row['official_task_success'] for row in episodes),
                      'same_frozen_sources_and_configuration': True}))


if __name__ == '__main__':
    main()
