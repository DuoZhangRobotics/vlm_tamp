"""Prepare or run a frozen, image-only Molmo2 contact-point experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

from vlm_tamp.pointing_molmo2 import Molmo2Pointer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ('results/luna-communication-rerun-20261102/trials/'
                 'baseline-r1-close_microwave-state017-600/'
                 'agent-workspace/robot_data/attempt01')
REVISION = 'e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b'
PROMPTS = {
    'locate_door': (
        'Image 1 shows the wider scene. Image 2 shows the same physical state from another camera. '
        'Point to the microwave door in image 2. Only mark image 2.'),
    'closing_contact': (
        'Image 1 shows the wider scene. Image 2 shows the same physical state from another camera. '
        'The robot\'s task is to close the microwave door by pushing with a finger, without pinching '
        'or pulling a handle. Point to ONE visible contact location in image 2 where a finger '
        'could push the door toward its closed configuration. The marked surface must actually '
        'be visible in image 2. A reachable robot posture is not part of this test; do not assume '
        'one has been solved. If image 2 does not support a useful mark, say no suitable visible '
        'point. Only mark image 2.'),
}


def digest(path: Path) -> str:
    """Identify the exact public input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(study: Path) -> None:
    """Freeze all queries before inference; no answers or geometry enter the inputs."""
    if (study / 'protocol.json').exists():
        raise FileExistsError('Preserve the frozen protocol')
    cases = []
    views = [(name, 'arm_lower_right') for name in ['obs001', 'obs029', 'obs041', 'obs051', 'obs068']]
    views.append(('obs001', 'wrist'))
    for observation, camera in views:
        folder = study / 'inputs' / f'{observation}-{camera}'
        folder.mkdir(parents=True)
        images = []
        for name, source_name in [('context.png', 'agentview.png'), ('contact.png', camera + '.png')]:
            source = SOURCE / observation / source_name
            destination = folder / name
            shutil.copyfile(source, destination)
            images.append({'path': str(destination.relative_to(study)), 'sha256': digest(destination),
                           'source': str(source.relative_to(ROOT))})
        for query, prompt in PROMPTS.items():
            cases.append({'id': f'{observation}-{camera}-{query}', 'observation': observation,
                          'camera': camera, 'query': query, 'prompt': prompt, 'images': images})
    protocol = {
        'model': 'allenai/Molmo2-8B', 'revision': REVISION, 'dtype': 'bfloat16',
        'quantization': None, 'decoding': {'do_sample': False, 'max_new_tokens': 512},
        'cases': cases, 'window_is_an_allowed_image_space_contact': True,
        'scope': 'Five recorded instants from one state-17 episode, plus its initial wrist view. '
                 'The other-side camera is a recorded diagnostic, not the two-camera benchmark. '
                 'No robot motion, contact execution, SAM, depth, native geometry, previous '
                 'answers or successful coordinates are supplied to Molmo2. Each query is fresh.',
        'comparison_scope': 'Initial side-view image bytes match prior Luna/Sol tests. Molmo2 uses '
                            'its native pointing prompt/output, not the earlier agent/tool/JSON '
                            'protocol; this is not a matched model-performance comparison.',
        'supplied_queries': 'Operator-supplied generic object and closing-contact requests; '
                            'these are not Luna-authored discoveries.',
    }
    (study / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')


def run(study: Path, checkpoint: Path) -> None:
    """Load once and save every fresh request, including non-pointing responses."""
    if 'VLM_TAMP_CAPACITY_STATE' in os.environ:
        Path(os.environ['VLM_TAMP_CAPACITY_STATE']).write_text('{"waiting": false}\n')
    protocol = json.loads((study / 'protocol.json').read_text())
    (study / 'queries').mkdir()
    started = time.perf_counter()
    pointer = Molmo2Pointer(checkpoint)
    environment = {name: importlib.metadata.version(name) for name in
                   ['torch', 'torchvision', 'transformers', 'accelerate', 'safetensors']}
    (study / 'load.json').write_text(json.dumps({
        'model_load_seconds': pointer.load_seconds,
        'startup_through_load_seconds': time.perf_counter() - started,
        'packages': environment, 'model_revision': protocol['revision'],
        'checkpoint': str(checkpoint), 'dtype': protocol['dtype'],
    }, indent=2) + '\n')
    print('Model loaded', pointer.load_seconds, flush=True)
    for case in protocol['cases']:
        images = [study / row['path'] for row in case['images']]
        for path, metadata in zip(images, case['images']):
            if digest(path) != metadata['sha256']:
                raise ValueError(f'Frozen image changed: {path}')
        folder = study / 'queries' / case['id']
        folder.mkdir()
        (folder / 'prompt.txt').write_text(case['prompt'] + '\n')
        result = pointer.point(images, case['prompt'])
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        (folder / 'raw.txt').write_text(result['raw_text'] + '\n')
        print(case['id'], result['status'], result['raw_text'],
              round(result['total_seconds'], 3), flush=True)
    (study / 'completion.json').write_text(json.dumps({
        'status': 'complete', 'queries': len(protocol['cases']),
        'load_and_queries_seconds': time.perf_counter() - started,
        'robot_motion_executed': False,
    }, indent=2) + '\n')


def main() -> None:
    """Run only the explicitly requested preparation or inference stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / '.runtime/models/Molmo2-8B')
    args = parser.parse_args()
    args.study.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare(args.study)
    else:
        run(args.study, args.checkpoint)


if __name__ == '__main__':
    main()
