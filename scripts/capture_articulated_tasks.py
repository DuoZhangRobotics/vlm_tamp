"""Capture RGB-D for new official tasks without reading scene ground truth."""

import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vlm_tamp.rekep_libero.sim_client_articulated import ArticulatedClient

TASKS = [
    ('open_top_drawer', 'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet'),
    ('close_top_drawer', 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet'),
    ('open_microwave', 'KITCHEN_SCENE7_open_the_microwave'),
    ('close_microwave', 'KITCHEN_SCENE6_close_the_microwave'),
    ('turn_on_stove', 'KITCHEN_SCENE3_turn_on_the_stove'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/articulated-development/captures-v3')
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    base = ROOT/args.output
    base.mkdir(parents=True, exist_ok=False)
    summaries = []
    for label, task in TASKS:
        if args.tasks and label not in args.tasks:
            continue
        folder = base/label
        folder.mkdir()
        sim = ArticulatedClient(folder/'simulator.log')
        try:
            state = sim.reset(task, 0, folder/'initial')
            state = sim.action(state['metadata']['eef_pose_world_xyzw'], -1, 60, folder/'opened')
            summaries.append({'task': task, 'label': label, 'status': 'captured',
                              'instruction': state['metadata']['instruction']})
        except Exception as error:
            summaries.append({'task': task, 'label': label, 'status': 'failed', 'error': str(error)})
        finally:
            sim.close()
        (base/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == '__main__':
    main()
