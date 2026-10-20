"""Validate frozen sources, native predicates, contact evidence and preservation."""

import argparse
from collections import Counter
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from try_rekep_bowl_v3 import save


def read(path):
    return json.loads(path.read_text())


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for name, value in attrs if name in {'href', 'src'} and value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/physical-tasks-v8-eval')
    parser.add_argument('--report', default='results/task-grounding-contact-v8-report')
    parser.add_argument('--tests', default='results/contact-policy-v10-development/regression-tests.xml')
    args = parser.parse_args()
    base, report = ROOT / args.input, ROOT / args.report
    manifest, protocol = read(base / 'manifest.json'), read(base / 'protocol.json')
    before = read(ROOT / 'results/contact-policy-v8-development/preserved-before.json')
    changed = [name for name, digest in before.items() if not (ROOT / name).is_file()
               or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    freeze = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
              and hashlib.sha256((base / 'source-snapshot' / name).read_bytes()).hexdigest() == digest
              for name, digest in protocol['sources_sha256'].items()}
    episodes, permissions, nonblocking, failures = [], Counter(), Counter(), Counter()
    for task, state in manifest['planned_episodes']:
        folder = base / f'goal{task:02d}-state{state:03d}'
        summary, actions = read(folder / 'summary.json'), read(folder / 'initial/actions.json')
        native = any(action['evaluation_task_success'] for action in actions)
        if native != summary['official_task_success']:
            raise ValueError('Recorded native predicate disagrees with summary')
        if bool(actions[-1]['evaluation_task_success']) != summary['final_task_success']:
            raise ValueError('Final-state success disagrees with action log')
        if not (summary['control_steps'] == len(actions) <= 600):
            raise ValueError('Control budget/count mismatch')
        if (folder / 'collision-checks.json').is_file():
            for record in read(folder / 'collision-checks.json'):
                result = record['result']
                for item in result.get('contact_permissions_applied', []):
                    permissions[item['rule']] += item['points']
                for item in result.get('nonblocking_positive_clearance', []):
                    nonblocking[item['robot_geom_name']] += item['points']
                if not result['accepted']:
                    failures.update(f"{item['robot_geom_name']} / {item['scene_part']}" for item in
                                    result.get('clearance', result).get('collisions', []))
        episodes.append({'task': task, 'state': state, 'native_success': native,
                         'final_success': summary['final_task_success'],
                         'grasp_admitted': summary['full_task_grasp_admission'],
                         'reached_closure': (folder / 'closed').is_dir(), 'control_steps': len(actions),
                         'failure_stage': summary['failure_stage']})
    parsed = Links()
    parsed.feed((report / 'index.html').read_text())
    local = [unquote(urlparse(link).path) for link in parsed.links if not urlparse(link).scheme and not link.startswith('#')]
    missing = [link for link in local if not (report / link).is_file()]
    ledger = read(ROOT / 'results/rekep-libero-api/ledger.json')
    smoke = read(ROOT / 'results/contact-policy-v8-development/gripper-geometry-smoke/result.json')
    test_suites = ET.parse(ROOT / args.tests).getroot().findall('testsuite')
    test_counts = {name: sum(int(suite.attrib.get(name, 0)) for suite in test_suites)
                   for name in ['tests', 'failures', 'errors', 'skipped']}
    checks = {'all_prior_files_preserved': not changed, 'frozen_sources_and_snapshots_match': all(freeze.values()),
              'all_planned_episodes_present': len(episodes) == 10,
              'native_success_recomputed': True, 'all_control_budgets_passed': True,
              'report_links_valid': not missing, 'measured_gripper_geometry_smoke_passed': smoke['all_checks_passed'],
              'api_ledger_identical_to_before': hashlib.sha256((ROOT / 'results/rekep-libero-api/ledger.json').read_bytes()).hexdigest()
                                               == before['results/rekep-libero-api/ledger.json'],
              'api_cap_respected': ledger['cumulative_conservative_spend_usd'] <= ledger['budget_usd']}
    checks['regression_tests_passed'] = test_counts['tests'] > 0 and test_counts['failures'] == test_counts['errors'] == 0
    result = {'checks': checks, 'all_checks_passed': all(checks.values()),
              'previous_files_preserved': len(before), 'changed_prior_files': changed,
              'source_files_frozen': len(freeze), 'local_report_links_checked': len(local), 'missing_links': missing,
              'actual_boundary_permitted_points_by_rule': dict(permissions),
              'nonblocking_positive_gap_points_by_part': dict(nonblocking),
              'runtime_blocking_pairs': dict(failures), 'episodes': episodes,
              'new_api_calls': 0, 'new_api_spend_usd': 0,
              'cumulative_api_spend_usd': ledger['cumulative_conservative_spend_usd'],
              'remaining_api_budget_usd': ledger['remaining_usd']}
    result['regression_tests'] = test_counts
    result['validator_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(report / 'validation.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ['episodes', 'actual_boundary_permitted_points_by_rule', 'nonblocking_positive_gap_points_by_part']}))
    if not result['all_checks_passed']:
        raise ValueError('Experiment validation failed')


if __name__ == '__main__':
    main()
