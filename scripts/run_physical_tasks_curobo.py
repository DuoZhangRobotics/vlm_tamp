"""Reproducible cuRobo IK comparison using v11's physical/contact protocol."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]

from try_rekep_bowl_v3 import save
from vlm_tamp.rekep_libero.grasp_selection_curobo import choose_grasp
from vlm_tamp.rekep_libero.physical_execution import run_episode
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', type=int, nargs='+', default=[8, 0])
    parser.add_argument('--states', type=int, nargs='+', default=[1, 11, 21, 31, 41])
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new output directory; existing results are preserved')
    output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT/'configs/physical_tasks_curobo.json'
    config = json.loads(config_path.read_text())
    planned = [(task, index) for task in args.tasks for index in args.states]
    if len(set(planned)) != len(planned):
        raise ValueError('Duplicate planned episodes')
    save(output/'configuration.json', config)
    sources = sorted((ROOT/'src/vlm_tamp/rekep_libero').glob('*.py'))
    sources += [Path(__file__), ROOT/'scripts/try_rekep_bowl_v3.py', config_path,
                ROOT/'src/vlm_tamp/drawer_constraints.py', ROOT/'src/vlm_tamp/drawer_local_plane.py',
                ROOT/'src/vlm_tamp/pose_constraints.py', ROOT/'tests/test_curobo_adapter.py',
                ROOT/'docs/curobo_libero.md', ROOT/'requirements-curobo-libero.txt',
                ROOT/'scripts/setup_curobo_libero.sh', ROOT/'scripts/report_curobo_libero.py']
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        target = output/'source-snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = digest(path)
    ledger = ROOT/'results/rekep-libero-api/ledger.json'
    ledger_hash = digest(ledger)
    protocol = {'sources_sha256': hashes, 'api_ledger_before_sha256': ledger_hash,
        'development_only': args.development, 'all_official_states_previously_inspected': True,
        'partition_selection': {'tasks': args.tasks, 'initial_states': args.states},
        'no_held_out_generalization_claim': True, 'all_in_denominator': True,
        'curobo_ik_used': True, 'curobo_motiongen_used': False,
        'curobo_revision': config['curobo_revision'], 'scene_collision_backend': 'unchanged v11 observed surfaces and task contact permissions',
        'collision_free_guarantee': False, 'simulator_scene_ground_truth_in_policy': False,
        'physical_contacts_enabled': True, 'assisted_grasping': False,
        'api_calls': 0, 'graspgen_inference_calls': 0,
        'thread_limits': {name: os.environ.get(name) for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}}
    save(output/'protocol.json', protocol)
    save(output/'manifest.json', {'planned_episodes': planned, 'all_in_denominator': True,
        'seed': config['seed'], 'maximum_control_steps': config['maximum_control_steps']})
    summaries = []
    for task, index in planned:
        summaries.append(run_episode(task, index, output/f'goal{task:02d}-state{index:03d}', config, None,
                                     ik_factory=TaskIK, grasp_selector=choose_grasp))
        save(output/'summary.json', {'planned_episodes': len(planned), 'completed_episodes': len(summaries),
            'official_task_successes': sum(row['official_task_success'] for row in summaries),
            'api_ledger_unchanged': digest(ledger) == ledger_hash, 'episodes': summaries})


if __name__ == '__main__':
    main()
