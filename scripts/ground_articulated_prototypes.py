"""First-state role/primitive grounding for the new official tasks."""

import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from capture_articulated_tasks import TASKS
from vlm_tamp.rekep_libero.articulated_grounding import ground
from vlm_tamp.rekep_libero.perception import Perception


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures', default='results/articulated-development/captures-v3')
    parser.add_argument('--output', default='results/articulated-development/grounding-v2')
    args = parser.parse_args()
    perception = Perception()
    for label, task in TASKS:
        folder = ROOT/args.captures/label/'opened'
        output = ROOT/args.output/label
        try:
            metadata = json.loads((folder/'metadata.json').read_text())
            _, choices, cloud, _ = ground(perception, folder, metadata, output, f'prototype-{label}-state0')
            print(json.dumps({'task': label, 'choices': choices, 'contact_points': len(cloud)}), flush=True)
        except Exception as error:
            output.mkdir(parents=True, exist_ok=True)
            (output/'failure.json').write_text(json.dumps({'error': str(error)}, indent=2)+'\n')
            print(json.dumps({'task': label, 'failed': str(error)}), flush=True)


if __name__=='__main__':
    main()
