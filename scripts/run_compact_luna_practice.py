"""Continuous Luna practice with compact public tools and no separate reviewer."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUIDE = ROOT / 'data/agent_memory/luna-scene-motion-v1/guidelines.json'


def prompt(attempts, episode_seconds, session_seconds):
    return f"""Practice opening the microwave using physical robot interaction.
Use this one continuing session for {attempts} attempts from clean resets of the
same development scene. Each attempt allows 600 control steps and
{episode_seconds} seconds; the whole session allows {session_seconds} seconds.

Read tools.txt, example/readme.txt and experience/guidelines.json once. The
supplied guidance is earlier Luna-authored experience, not a guaranteed recipe.
Choose your own observations, geometry, contacts, experiments and corrections.
Begin with reset. Re-measure geometry and make fresh plans after every reset.

Use python3 robot.py help OPERATION before first using an unfamiliar tool.
CLI replies are compact; full_response points to all original fields. Read that
file when details matter instead of reissuing the operation. Python call returns
full data for analysis; brief returns a compact display. Use NumPy/SciPy/Pillow
when useful, printing only relevant results. Open the current preview images;
request other views or full sensor data when needed. Avoid dumping directory
trees, entire histories or repeated manuals into the conversation.

Stop motion immediately on task_success=true. Otherwise continue useful work
until the actual control or wall budget is exhausted. Planning failures do not
end an episode. Do not consume budget with meaningless motion. After a terminal
episode, save its results and reset for the next granted attempt. Follow the
current learning_phase from practice_status: establish repeatable success
before considering efficiency. Complete the granted attempts within the budget.

Before each attempt, save practice/attemptNN/plan.json with one hypothesis,
the intended change or repeatability check, and a predicted observation. After
it, save practice/attemptNN/findings.txt and update your own guidelines.json.
Keep notes concise: evidence, what changed, observed outcome, uncertainty and
applicability. guidelines.json needs a nonempty principles list of conditional
lessons and uncertainty; keep recorded coordinates and paths in findings only.
Preserve failures and distinguish hypotheses from verified facts.
Maintain a short working-state.txt with the current attempt, important evidence
references and unresolved question so details can be recovered without rereading
everything. This is your own practice; no separate review agent supplies lessons.

Finish with findings.txt and practice-summary.json recording every attempt,
best_supported_method, remaining limitations and stop_reason. Record task-success
booleans accurately. Same-scene practice does not establish generalization.
Use only this public workspace and supplied tools; no private implementation,
simulator ground truth, other agents or external model services.
"""


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def summarize(run):
    evaluation = read(run / 'evaluation-private.json', {'attempts': []})
    metrics = defaultdict(lambda: {'calls': 0, 'seconds': 0.})
    events = run / 'tool-events.jsonl'
    for line in events.read_text().splitlines() if events.exists() else []:
        row = json.loads(line)
        operation = row['request']['operation']
        metrics[operation]['calls'] += 1
        metrics[operation]['seconds'] += row['wall_seconds']
    agent = read(run / 'agent-exit.json', {})
    tool_seconds = sum(m['seconds'] for m in metrics.values())
    interface = run / 'agent-workspace/robot_data/interface-events.jsonl'
    displays = [json.loads(s) for s in interface.read_text().splitlines()] if interface.exists() else []
    raw = sum(r['raw_bytes'] for r in displays)
    compact = sum(r['display_bytes'] for r in displays)
    source_hashes = read(run / 'source-hashes.json', {})
    changed = [name for name, value in source_hashes.items()
               if not (ROOT / name).exists() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value]
    audit = read(run / 'audit.json', {})
    return {
        'mode': 'continuous_self_practice', 'separate_reflection_reviewer': False,
        'task': 'KITCHEN_SCENE7_open_the_microwave',
        'same_scene_repetitions': True, 'held_out_evaluation': False,
        'agent_exit': agent, 'attempts': evaluation['attempts'],
        'native_successes': sum(bool(a['official_task_success']) for a in evaluation['attempts']),
        'actor_wall_seconds': agent.get('wall_seconds'), 'robot_service_seconds': tool_seconds,
        'other_actor_seconds': agent.get('wall_seconds', 0.) - tool_seconds,
        'timing_scope': 'Other actor time includes model inference, shell analysis, transport and scheduling; it is not measured pure reasoning time.',
        'operations': dict(metrics),
        'compact_presentations': len(displays), 'full_reply_bytes': raw, 'display_bytes': compact,
        'display_reduction_fraction': 1. - compact / raw if raw else None,
        'display_scope': 'Bytes of selected CLI/brief presentations versus their complete JSON replies, not model input tokens or all shell output.',
        'source_integrity': {'passed': not changed, 'checked': len(source_hashes), 'changed': changed},
        'execution_audit': audit.get('trajectory_audit', []),
        'api_ledgers_unchanged': audit.get('api_ledgers_unchanged_since_checkpoint'),
        'learner_findings': 'agent-workspace/findings.txt',
        'learner_guidelines': 'agent-workspace/guidelines.json',
        'automatic_memory_promotion': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--state', type=int, default=45)
    parser.add_argument('--attempts', type=int, default=5)
    parser.add_argument('--agent-seconds', type=int, default=5400)
    parser.add_argument('--episode-seconds', type=int, default=1800)
    parser.add_argument('--guidelines-source', type=Path, default=DEFAULT_GUIDE)
    parser.add_argument('--numerical-backend', choices=['legacy','curobo_v2'], default='curobo_v2')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.state < 50 or not 2 <= args.attempts <= 20:
        raise ValueError('Use a state 0..49 and 2..20 practice attempts')
    if args.agent_seconds <= 0 or args.episode_seconds <= 0:
        raise ValueError('Use positive time budgets')
    run = args.output.resolve()
    if not run.is_relative_to(ROOT / 'results') or run == ROOT / 'results':
        raise ValueError('Use a new results/<study>/<run> directory')
    prompt_path = run.parent / 'inputs' / (run.name + '-prompt.txt')
    text = prompt(args.attempts, args.episode_seconds, args.agent_seconds)
    command = [sys.executable, str(ROOT / 'scripts/run_subscription_agent_pilot.py'),
        '--output', str(run), '--state', str(args.state), '--attempts', str(args.attempts),
        '--initial-attempts', str(args.attempts), '--learning-practice',
        '--agent-seconds', str(args.agent_seconds), '--episode-seconds', str(args.episode_seconds),
        '--stop-on-success', '--persistent-episode', '--surface-contact-tools',
        '--numerical-backend', args.numerical_backend,
        '--compact-interface', '--fit-diagnostics', '--abstract-experience',
        '--guidelines-source', str(args.guidelines_source.resolve()), '--prompt-file', str(prompt_path)]
    registration = {'task': 'KITCHEN_SCENE7_open_the_microwave', 'suite': 'libero_90',
        'state': args.state, 'seed': 0, 'scene_variant': 'nominal',
        'maximum_attempts': args.attempts, 'control_steps_per_attempt': 600,
        'episode_seconds': args.episode_seconds, 'session_seconds': args.agent_seconds,
        'model': 'gpt-6-luna', 'reasoning': 'high', 'same_learner_context': True,
        'separate_reflection_reviewer': False, 'prior_completed_episode_records_supplied': False,
        'guidelines_source': str(args.guidelines_source.resolve()),
        'guidelines_sha256': hashlib.sha256(args.guidelines_source.read_bytes()).hexdigest(),
        'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'development_state_previously_observed': True, 'generalization_test': False,
        'numerical_backend': args.numerical_backend,
        'numerical_policy_changes': args.numerical_backend != 'legacy', 'command': command}
    if args.dry_run:
        print(json.dumps({**registration, 'prompt': text}, indent=2))
        return
    registration_path = run.parent / (run.name + '-registration.json')
    if run.exists() or registration_path.exists():
        raise ValueError('Existing practice evidence; use a new output directory')
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(text)
    write(registration_path, registration)
    status_path = run.parent / (run.name + '-status.json')
    write(status_path, {'phase': 'running', 'started_unix': time.time(), 'run': str(run)})
    started = time.monotonic()
    with (run.parent / (run.name + '-launcher.log')).open('w') as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    summary = summarize(run)
    summary.update(launcher_returncode=completed.returncode, total_wall_seconds=time.monotonic()-started)
    write(run / 'learning-summary.json', summary)
    write(status_path, {'phase': 'complete' if completed.returncode == 0 else 'error',
        'returncode': completed.returncode, 'attempts': len(summary['attempts']),
        'successes': summary['native_successes'], 'completed_unix': time.time()})
    print(json.dumps({'run': str(run), 'attempts': len(summary['attempts']),
        'successes': summary['native_successes'], 'returncode': completed.returncode,
        'source_integrity': summary['source_integrity']}, indent=2))
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
