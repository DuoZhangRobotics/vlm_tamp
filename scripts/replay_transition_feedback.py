"""Paired offline Luna judgments on identical recorded public evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time

from vlm_tamp.actor_structured import OPERATIONS, VISUAL_TOOLS, decision_schema, run_decision
from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.feedback_transition import compact_transition_context
from vlm_tamp.subscription_agent import clean_environment
from run_restriction_study import source_hashes

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ('microwave17_stroke', 'luna-native-repair-20261108/trials/01-close_microwave-s017-transient_docs', 31,
     'away', None),
    ('microwave17_final', 'luna-native-repair-20261108/trials/01-close_microwave-s017-transient_docs', 49,
     'toward', None),
    ('microwave49_approach', 'luna-native-repair-20261108/trials/02-close_microwave-s049-transient_docs', 14,
     'away', None),
    ('bowl_first_lift', 'luna-tool-documentation-20261107/trials/04-place_bowl-s000-persistent_docs', 29,
     None, 'not_established'),
    ('bowl_second_lift', 'luna-tool-documentation-20261107/trials/04-place_bowl-s000-persistent_docs', 56,
     None, 'maintained'),
)
ASSESSMENT_INSTRUCTIONS = """
This is a recorded-input test. Your returned action will NOT be executed.
In addition to the usual operation, arguments_json, expected_effect and note, return assessment.
Assess the latest executed interaction using only the supplied public evidence.
object_goal_change is toward, away, no_change or uncertain relative to the stated task goal.
contact_retention is maintained, lost, not_established or uncertain. Distinguish an empty
grasp from losing a previously established grasp. Unknown measurements are not zero motion.
Use uncertain when the available evidence does not establish the answer. Explain briefly
which observations support your assessment. A terminal episode still requires finish.
"""


def probe_schema(operations: list[str]) -> dict:
    """Ask the same explicit judgments in both conditions without exposing evaluator labels."""
    schema = decision_schema(operations)
    schema['required'].append('assessment')
    schema['properties']['assessment'] = {
        'type': 'object', 'additionalProperties': False,
        'required': ['object_goal_change', 'contact_retention', 'evidence'],
        'properties': {
            'object_goal_change': {'type': 'string', 'enum': ['toward', 'away', 'no_change', 'uncertain']},
            'contact_retention': {'type': 'string', 'enum': ['maintained', 'lost', 'not_established', 'uncertain']},
            'evidence': {'type': 'string'},
        },
    }
    return schema


def main() -> None:
    """Keep every predeclared repeat, including uncertainty and unsupported claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'source_sha256': source_hashes(),
        'model': 'gpt-6-luna', 'reasoning': 'high', 'repeats_per_condition_per_case': 2,
        'conditions': ['existing', 'compact'], 'cases': [],
        'robot_motion_executed': False, 'task_success_measured': False,
        'selection': 'Five predeclared diagnostic contexts; development cases, not an unbiased sample.',
        'comparison': 'Same original images, instructions, evidence source, tool choices and assessment questions; only executed-receipt presentation differs.',
        'evaluation_scope': 'Private post-episode labels used only by this scoring script. Uncertain is an abstention, not a false assertion. No oracle labels enter prompts.',
    }
    selected = []
    for name, relative, number, direction, contact in CASES:
        run = ROOT / 'results' / relative
        folder = run / 'decisions' / f'{number:03d}'
        context = read_json(folder / 'input.json')
        images = [run / 'live-images' / item['path'] for item in context['images']]
        for item, path in zip(context['images'], images):
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Source image hash mismatch: ' + str(path))
        compact = compact_transition_context(context)
        if compact['images'] != context['images']:
            raise ValueError('Presentation changed the image manifest')
        protocol['cases'].append({'name': name, 'source': str(folder),
            'control_step': context['current_observation']['control_steps'],
            'evaluator_direction': direction, 'evaluator_contact': contact,
            'original_bytes': len(json.dumps(context, separators=(',', ':')).encode()),
            'compact_bytes': len(json.dumps(compact, separators=(',', ':')).encode()),
            'images': context['images']})
        selected.append((folder, context, compact, images))
    write_json(output / 'protocol.json', protocol)
    if args.prepare_only:
        return
    rows = []
    for index, (folder, original, compact, images) in enumerate(selected):
        if source_hashes() != protocol['source_sha256']:
            raise RuntimeError('Frozen recorded-input comparison sources changed')
        prefix = (folder / 'prompt.txt').read_text().rsplit('\n', 1)[0] + ASSESSMENT_INSTRUCTIONS
        operations = [name for name in OPERATIONS if name in original['tools']]
        operations += ['help', 'inspect_image', *sorted(VISUAL_TOOLS if original['visual_tools_available'] else [])]
        if original.get('terminal_summary_requested'):
            operations = ['finish']
        for repeat in range(2):
            conditions = ['existing', 'compact'] if (index + repeat) % 2 == 0 else ['compact', 'existing']
            for condition in conditions:
                payload = original if condition == 'existing' else compact
                name = f'{index + 1:02d}-r{repeat + 1}-{condition}'
                destination = output / name
                logging.info('Offline judgment %s; no robot connection', name)
                decision = run_decision(prefix + '\n' + json.dumps(payload, separators=(',', ':')),
                    images, destination, clean_environment(), probe_schema(operations),
                    'gpt-6-luna', 'high', time.monotonic() + 300)
                write_json(destination / 'input.json', payload)
                if decision is None:
                    raise TimeoutError('Offline judgment exceeded its 300-second allowance')
                case = protocol['cases'][index]
                scores = {}
                for field, label in [('object_goal_change', case['evaluator_direction']),
                                     ('contact_retention', case['evaluator_contact'])]:
                    if label is None:
                        continue
                    answer = decision['assessment'][field]
                    scores[field] = {'expected': label, 'answer': answer, 'correct': answer == label,
                                     'abstained': answer == 'uncertain',
                                     'false_assertion': answer not in {label, 'uncertain'}}
                rows.append({'context': index + 1, 'name': case['name'], 'repeat': repeat + 1,
                    'condition': condition, 'folder': name, 'source': str(folder),
                    'decision': decision, 'scores': scores, 'timing': read_json(destination / 'timing.json'),
                    'robot_motion_executed': False})
                write_json(output / 'summary.json', {'complete': False, 'comparisons': rows})
    if source_hashes() != protocol['source_sha256']:
        raise RuntimeError('Frozen recorded-input comparison sources changed')
    write_json(output / 'summary.json', {'complete': True, 'comparisons': rows})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main()
