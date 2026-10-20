"""Replay saved articulated joint states through LIBERO's goal predicates.

This is an evaluation-only audit of the five single-joint pilot tasks. It reads
the native predicate bodies and default ranges from the pinned LIBERO checkout;
it neither advances the simulator nor supplies native geometry to the policy.
"""

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'third_party/LIBERO/libero/libero/envs/objects/articulated_objects.py'
BDDL = ROOT / 'third_party/LIBERO/libero/libero/bddl_files/libero_90'
RULES = {
    'open_top_drawer': ('WoodenCabinet', 'is_open', 'Open',
                        'wooden_cabinet_1_top_region', 'wooden_cabinet_1_top_level', 'm'),
    'close_top_drawer': ('WoodenCabinet', 'is_close', 'Close',
                         'wooden_cabinet_1_top_region', 'wooden_cabinet_1_top_level', 'm'),
    'open_microwave': ('Microwave', 'is_open', 'Open',
                       'microwave_1', 'microwave_1_microjoint', 'rad'),
    'close_microwave': ('Microwave', 'is_close', 'Close',
                        'microwave_1', 'microwave_1_microjoint', 'rad'),
    'turn_on_stove': ('FlatStove', 'turn_on', 'Turnon',
                      'flat_stove_1', 'flat_stove_1_button', 'rad'),
}


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_predicate(class_name, method_name):
    """Load only native range assignments and the selected predicate function."""
    tree = ast.parse(NATIVE.read_text(), filename=str(NATIVE))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    ranges = {}
    for node in ast.walk(constructor):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript)
                and ast.unparse(target.value) == "self.object_properties['articulation']"):
            ranges[ast.literal_eval(target.slice)] = ast.literal_eval(node.value)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(NATIVE), 'exec'), namespace)
    native_object = SimpleNamespace(
        object_properties={'articulation': ranges, 'vis_site_names': {}}, naming_prefix='audit_')
    comparison = next(n for n in ast.walk(method)
                      if isinstance(n, ast.Compare) and ast.unparse(n.left) == 'qpos')
    if len(comparison.ops) != 1 or len(comparison.comparators) != 1:
        raise ValueError('Expected a single scalar native goal comparison')
    operation = {ast.Lt: '<', ast.Gt: '>', ast.GtE: '>='}[type(comparison.ops[0])]
    threshold = eval(compile(ast.Expression(comparison.comparators[0]), str(NATIVE), 'eval'),
                     {'self': native_object, 'min': min, 'max': max})
    def evaluate(qpos):
        return bool(namespace[method_name](native_object, qpos))
    return evaluate, {'class': class_name, 'method': method_name,
                      'operator': operation, 'threshold': threshold,
                      'source_line': method.lineno, 'default_ranges': ranges}


def verify_single_goal(path, predicate, target):
    # Fail closed if a future task adds another goal or changes its target.
    source = re.sub(r';[^\n]*', '', path.read_text())
    tokens = re.findall(r'\(|\)|[^\s()]+', source)
    stack, roots = [], []
    for token in tokens:
        if token == '(':
            value = []
            (stack[-1] if stack else roots).append(value)
            stack.append(value)
        elif token == ')':
            stack.pop()
        else:
            stack[-1].append(token)
    if stack or len(roots) != 1:
        raise ValueError('Unexpected task definition syntax')
    goal = next(n for n in roots[0] if isinstance(n, list) and n and n[0] == ':goal')
    if goal != [':goal', ['And', [predicate, target]]]:
        raise ValueError(f'Unsupported multi-joint or changed goal: {goal}')
    return goal


def audit_episode(folder, planned, mode, fingerprints):
    paths = {name: folder / filename for name, filename in {
        'summary': 'summary.json', 'actions': 'initial/actions.json',
        'states': 'initial/evaluation_contact_diagnostics.json'}.items()}
    summary, actions, diagnostics = [read(paths[key]) for key in ['summary', 'actions', 'states']]
    if not diagnostics.get('evaluation_only') or not diagnostics.get('excluded_from_policy_input'):
        raise ValueError('Native states must remain explicitly evaluation-only')
    class_name, method, predicate, target, joint, units = RULES[planned['task']]
    goal_path = BDDL / f'{planned["name"]}.bddl'
    goal = verify_single_goal(goal_path, predicate, target)
    evaluate, rule = native_predicate(class_name, method)
    states = {s['step']: s for s in diagnostics['steps']}
    if len(states) != len(actions) or len(actions) != summary['control_steps']:
        raise ValueError('State, action, and summary lengths disagree')
    values, successes, success_steps = [], [], []
    for action in actions:
        joints = [j for j in states[action['step']]['evaluation_only_articulation_state']
                  if j['joint_name'] == joint]
        if len(joints) != 1 or not math.isfinite(joints[0]['position']):
            raise ValueError('Missing or invalid evaluated task joint')
        qpos = joints[0]['position']
        success = evaluate(qpos)
        if success != action['evaluation_task_success']:
            raise ValueError(f'Native predicate replay disagrees at {folder}:{action["step"]}')
        values.append(qpos)
        successes.append(success)
        if success:
            success_steps.append(action['step'])
    ever, final = any(successes), bool(successes and successes[-1])
    if ever != summary['official_task_success'] or final != summary['final_task_success']:
        raise ValueError('Summary disagrees with replayed physical outcome')
    for path in [*paths.values(), goal_path]:
        fingerprints[str(path.relative_to(ROOT))] = sha256(path)
    first, last = values[0], values[-1]
    signed_margin = last-rule['threshold']
    if rule['operator'] == '<':
        signed_margin *= -1
    return {
        'mode': mode, 'task': planned['task'], 'state': planned['state'],
        'episode': str(folder.relative_to(ROOT)), 'evaluation_only': True,
        'native_goal': goal, 'joint_name': joint, 'joint_units': units,
        'native_predicate': rule, 'first_logged_joint_position': first,
        'final_joint_position': last, 'logged_joint_displacement': last-first,
        'final_angle_from_closed_deg': abs(math.degrees(last)) if 'microwave' in planned['task'] else None,
        'signed_goal_margin': signed_margin,
        'additional_motion_to_goal_boundary': max(0., -signed_margin),
        'native_success_ever': ever, 'native_success_final': final,
        'native_success_steps': success_steps, 'control_steps': len(actions),
        'controller_status': summary['status'],
        'controller_stop_stage': summary.get('failure_stage'),
        'controller_stop_reason': summary.get('failure'),
        'native_success_despite_controller_failure': ever and summary['status'] == 'failed',
        'all_step_predicates_agree': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', default='results/articulated-strict-final')
    parser.add_argument('--diagnostic', default='results/articulated-physics-pilot')
    parser.add_argument('--output', default='results/articulated-postrun-audit')
    args = parser.parse_args()
    rows, fingerprints = [], {str(NATIVE.relative_to(ROOT)): sha256(NATIVE)}
    for mode in ['strict', 'diagnostic']:
        base = ROOT / getattr(args, mode)
        manifest_path = base / 'manifest.json'
        manifest = read(manifest_path)
        fingerprints[str(manifest_path.relative_to(ROOT))] = sha256(manifest_path)
        for planned in manifest['planned_episodes']:
            folder = base / f'{planned["task"]}-state{planned["state"]:03d}'
            rows.append(audit_episode(folder, planned, mode, fingerprints))
    counts = {}
    for mode in ['strict', 'diagnostic']:
        selected = [r for r in rows if r['mode'] == mode]
        counts[mode] = {
            'planned_episodes': len(selected),
            'native_success_ever': sum(r['native_success_ever'] for r in selected),
            'native_success_final': sum(r['native_success_final'] for r in selected),
            'controller_stop_stages': dict(Counter(r['controller_stop_stage'] or 'completed' for r in selected)),
        }
    audit = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'post-run replay of the five single-joint pilot goals; no new simulation controls',
        'success_criterion': 'native goal true at any logged control step; final state also reported',
        'native_geometry_used_for_policy': False, 'new_api_requests': 0,
        'post_release_stability_test_performed': False,
        'all_logged_steps_replayed': sum(r['control_steps'] for r in rows),
        'native_predicate_replay_matches_all_logs': all(r['all_step_predicates_agree'] for r in rows),
        'modes': counts, 'episodes': rows, 'source_and_inputs_sha256': fingerprints,
        'audit_source_sha256': sha256(Path(__file__)),
    }
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    columns = ['mode', 'task', 'state', 'native_success_ever', 'native_success_final',
               'final_joint_position', 'joint_units', 'final_angle_from_closed_deg',
               'controller_stop_stage', 'controller_stop_reason']
    with (output / 'outcomes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: audit[key] for key in
                      ['all_logged_steps_replayed', 'native_predicate_replay_matches_all_logs', 'modes']}))


if __name__ == '__main__':
    main()
