"""Public-only reflection contract and bounded local experiment memory."""
import json
from pathlib import Path
import shutil

from .checkpoint_diagnostic import latest_observation, read_json, sha256, verify_packet, write_json


ACTOR_PROMPT = """Continue the active robotics episode using tools.txt and robot.py.
Read the approved example, experience/guidelines.json, and
checkpoint-prefix/checkpoint.json and checkpoint-prefix/history.jsonl. Inspect
the current images. The reset and recorded prefix have already been physically
executed. Do not reset: continue from the current robot state and model. The
episode has 600 total control steps, including those already used, and a
30-minute wall budget including replay. Check episode_status for remaining time.

If experience/recent exists, it contains a previous, completed attempt at this
same practice checkpoint. Paths inside that record resolve relative to its
record/ directory, not the current robot_data/. Its terminal outcome describes
that previous attempt only. An optional reflection is a candidate hypothesis
written by the same model after that attempt; it is not a verified fact. Use
the available evidence and decide your own next actions.

Work autonomously with the public observations and numerical tools. Choose
regions, grasps, probes, model updates, release/regrasp and replanning yourself.
Stop all motion immediately on native task success. Otherwise continue useful
investigation and interaction until a real control or wall budget is exhausted;
do not use meaningless motion to consume budget. A planning failure is not an
episode termination. Do not seek hidden implementation, private evaluation,
other project results or external model services. Briefly record evidence,
changes you tried, uncertainty and findings in findings.txt. Do not provide a
detailed reasoning transcript. All model reasoning uses this subscription session.
"""

REVIEW_PROMPT = """Review one completed robot practice attempt from public evidence.
First read inputs/retry-start.json: it specifies where the NEXT acting attempt
begins and its remaining control budget. Its prefix has already been executed;
changes to actions before that boundary and another reset are unavailable.
inputs/checkpoint.json describes the preceding attempt's TERMINAL snapshot,
not the starting state of the next attempt. Your proposed change and its scope
must apply from the stated retry start, with any further preconditions explicit.
Then read inputs/tools.txt, inputs/example/readme.txt, inputs/experience/guidelines.json,
inputs/history.jsonl and inputs/outcome.json. Other public artifacts and images
are under inputs/; paths in the record resolve relative to that directory.
Some older arrays may be unavailable, as listed in inputs/checkpoint.json.
There is no live robot service. Do not execute robot operations or invent results.

Write reflection.json with exactly these fields:
- evidence: 1 to 6 objects, each containing event (integer event number), pointer
  (RFC 6901 JSON pointer into that event), and value (the exact JSON value there).
- observation: a concise discrepancy or useful regularity supported by evidence.
- hypothesis: a possible explanation, distinguished from what was observed.
- next_change: one change to test on another attempt at the same checkpoint.
- test: an observable outcome that would support or contradict that change.
- uncertainty: alternatives, missing evidence and remaining uncertainty.
- scope: where this candidate lesson might apply and where it is untested.

Keep the six text fields together under 350 words. Cite only accessible evidence.
The official task-success boolean is authoritative. Completion of a tool call
is not proof of task success. Your output is a candidate for another practice
attempt, not a verified portable rule. Do not write a detailed reasoning transcript.
Do not modify inputs or obtain hidden simulator geometry or other agent answers.
"""


def pointer_value(value, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('Evidence requires a non-root JSON pointer')
    for token in pointer[1:].split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def validate_reflection(value, rows):
    fields = {'evidence', 'observation', 'hypothesis', 'next_change', 'test', 'uncertainty', 'scope'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('Reflection fields differ from the public contract')
    words = 0
    for key in fields - {'evidence'}:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError('Reflection text must be nonempty')
        words += len(value[key].split())
    if words > 350:
        raise ValueError('Reflection exceeds the declared word budget')
    if not isinstance(value['evidence'], list) or not 1 <= len(value['evidence']) <= 6:
        raise ValueError('Use one to six evidence citations')
    events = {row['sequence']: row for row in rows}
    for item in value['evidence']:
        if not isinstance(item, dict) or set(item) != {'event', 'pointer', 'value'} or type(item['event']) is not int:
            raise ValueError('Invalid evidence citation structure')
        if item['event'] not in events:
            raise ValueError('Evidence cites an unavailable event')
        try:
            expected = pointer_value(events[item['event']], item['pointer'])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError('Evidence pointer does not resolve') from error
        if json.dumps(expected, sort_keys=True, allow_nan=False) != json.dumps(item['value'], sort_keys=True, allow_nan=False):
            raise ValueError('Evidence value is not an exact copy of the cited public field')
    return {'passed': True, 'narrative_words': words, 'citations_checked': len(value['evidence']),
            'scope': 'Format, provenance and exact cited values only; causal explanation is not certified.'}


def tree_hashes(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def remaining_actor_seconds(ready, now):
    remaining = ready['deadline_monotonic'] - now
    if remaining <= 0:
        raise TimeoutError('Replay exhausted the total episode wall budget')
    return remaining


def retry_start(rows, cutoff, *, control_limit=600):
    """Extract the public configured state at the fixed restart boundary.

This experiment restarts at a completed grounding/model-edit snapshot. Preserve
that estimate even when it is wrong; no earlier or evaluator-corrected value is
substituted. A motion-only cutoff needs a separately recorded current model.
"""
    prefix = [row for row in rows if row['sequence'] <= cutoff]
    if not prefix or prefix[-1]['sequence'] != cutoff:
        raise ValueError('Retry boundary is missing from the public history')
    response = prefix[-1]['response']
    if not response.get('ok') or 'model' not in response.get('result', {}):
        raise ValueError('Retry boundary must include a public current-model snapshot')
    observation = latest_observation(prefix)
    steps = observation['control_steps']
    if steps >= control_limit or observation.get('task_success'):
        raise ValueError('The retry start must be a nonterminal public state')
    instruction = next(row['response']['result']['instruction'] for row in prefix
        if row['request']['operation'] == 'reset' and row['response'].get('ok'))
    return {'scope': 'Starting state of the next acting attempt, extracted only from the fixed public prefix.',
        'task_instruction': instruction, 'last_completed_event': cutoff,
        'control_steps': steps, 'remaining_control_steps': control_limit - steps,
        'current_observation': observation, 'current_model': response['result']['model'],
        'current_geometry': response['result'],
        'earlier_prefix_actions_already_executed': True, 'another_reset_available': False,
        'paths_resolve_relative_to': 'this inputs directory',
        'wall_budget': '1800 seconds including replay; the live actor checks its remaining time with episode_status.'}


def prepare_review_packet(packet, manifest, destination, cutoff):
    """Add retry context only to the reviewer; leave actor experience unchanged."""
    packet, destination = Path(packet), Path(destination)
    if any(path.is_symlink() for path in packet.rglob('*')):
        raise ValueError('Review input must not contain symlinks')
    verify_packet(packet, manifest)
    rows = [json.loads(line) for line in (packet / 'history.jsonl').read_text().splitlines()]
    start = retry_start(rows, cutoff)
    shutil.copytree(packet, destination)
    write_json(destination / 'retry-start.json', start)
    terminal = read_json(destination / 'checkpoint.json')
    terminal['scope'] = ('Terminal snapshot of the preceding completed attempt. '
                         'The next actor starts from retry-start.json, not from this terminal snapshot.')
    write_json(destination / 'checkpoint.json', terminal)
    return {**manifest, 'packet_sha256': tree_hashes(destination),
        'review_only_restart_context': True, 'restart_cutoff': cutoff,
        'actor_record_unchanged': verify_packet(packet, manifest)['passed']}
