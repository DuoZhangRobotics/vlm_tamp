"""Publish the corrected evaluation beside the preserved original development."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from vlm_tamp.rekep_libero.report import build_report
    config = json.loads((ROOT / 'configs/rekep_libero_v2.json').read_text())
    evaluations, development = [], []
    for task in config['tasks']:
        template = f"libero_goal-{task['task_index']:02d}-state{{state:03d}}-seed{config['seed']}"
        evaluations += [ROOT / config['output'] / template.format(state=state)
                        for state in config['evaluation_states']]
        development.append(ROOT / 'results/rekep-libero-v1' /
                           ('dev-' + template.format(state=task['development_state'])))
    result = build_report(evaluations, ROOT / 'results/rekep-libero-strict-report', expected_episodes=10,
                          expected_episode_paths=evaluations, development_dirs=development,
                          ledger_path=ROOT / 'results/rekep-libero-api/ledger.json')
    print(json.dumps({key: result[key] for key in
                     ('evaluation_planned', 'evaluation_successes', 'evaluation_physical_grasp_attempts', 'failures')}))


if __name__ == '__main__':
    main()
