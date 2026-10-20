"""Single-variable oracle-hinge opening diagnostic without a pinch-width veto.

An AST adaptation changes only the condition guarding the gripper-gap rejection.
Physical contacts, RGB-D co-motion, IK, manifold and control limits stay active.
All original execution source files remain untouched.
"""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]

from diagnose_microwave_hinge import NoNewGrasps, digest, episode_audit, fixed_grasp_hooks, read
from try_rekep_bowl_v3 import save
from vlm_tamp.rekep_libero import mechanism_execution as executor
from vlm_tamp.rekep_libero.perception import Perception


def adapt_executor(output):
    original = inspect.getsource(executor.run_episode)
    tree = ast.parse(original)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(
                isinstance(n, ast.Constant) and n.value == 'Gripper gap does not support retained contact'
                for n in ast.walk(node)):
            # This uniquely identifies the small local if, never an enclosing try.
            if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
                matches.append(node)
    if len(matches) != 1 or ast.unparse(matches[0].test) != 'not 0.002 < gap < 0.079':
        raise ValueError('Expected exactly the frozen pinch-width gate')
    matches[0].test = ast.copy_location(ast.Constant(False), matches[0].test)
    tree.body[0].name = 'run_episode_without_pinch_width_veto'
    ast.fix_missing_locations(tree)
    adapted = ast.unparse(tree)+'\n'
    path = output/'adapted-executor.py'
    path.write_text(adapted)
    namespace = dict(vars(executor))
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['run_episode_without_pinch_width_veto'], hashlib.sha256(adapted.encode()).hexdigest()


def main():
    output = ROOT/'results/microwave-hook-contact-diagnosis'
    reference = ROOT/'results/articulated-physics-pilot'
    output.mkdir(parents=True, exist_ok=False)
    config = read(ROOT/'configs/articulated_libero.json')
    config['scene_mode'] = 'diagnostic'
    label = 'open_microwave'
    task = next(row['name'] for row in config['tasks'] if row['label'] == label)
    ledger = ROOT/'results/articulated-api-v1/ledger.json'
    ledger_hash = digest(ledger)
    frozen = read(reference/'protocol.json')['sources_sha256']
    if any(digest(ROOT/p) != sha for p, sha in frozen.items()):
        raise ValueError('Frozen executor sources changed')
    save(output/'manifest.json', {'planned_episodes': [{'task': label, 'name': task, 'state': 1}],
         'all_in_denominator': True, 'diagnostic_only': True, 'maximum_control_steps': 600})
    save(output/'protocol.json', {
        'diagnostic_only': True, 'oracle_hinge_used': True, 'oracle_progress_used': False,
        'additional_change': 'Disable only 0.002 < finger-gap < 0.079 pinch-width rejection',
        'rgbd_tracking_and_comotion_vetoes_retained': True,
        'joint_manifold_self_collision_and_control_limits_retained': True,
        'physical_contacts_enabled': True, 'assisted_grasping': False,
        'fixed_original_grasp_and_approach': True, 'new_api_calls': 0,
        'api_ledger_sha256_before': ledger_hash, 'frozen_sources_sha256': frozen,
        'runner_sha256': digest(__file__),
    })
    perception = Perception()
    folder = output/f'{label}-state001'
    with fixed_grasp_hooks('oracle', reference/f'{label}-state001'):
        adapted, adapted_hash = adapt_executor(output)
        summary = adapted(label, task, 1, folder, config, perception, NoNewGrasps(), reuse_run=reference)
    summary.update(diagnostic_only=True, simulator_scene_state_used=True, oracle_hinge_used=True,
                   oracle_progress_used=False, pinch_width_gate_disabled=True,
                   assisted_grasping=False, new_api_calls=0)
    save(folder/'summary.json', summary)
    audit = episode_audit(folder, reference/f'{label}-state001')
    actions = read(folder/'initial/actions.json')
    feedback = read(folder/'mechanism-feedback.json')
    for item in feedback:
        item.update(gripper_gap_m=sum(abs(x) for x in actions[item['control_steps']-1]['gripper_qpos']),
                    simulator_scene_state_used=True, oracle_hinge_used=True,
                    physical_measurement_source='RGB-D, fitted using oracle hinge geometry')
    save(output/'feedback-with-diagnostic-scope.json', feedback)
    intact = all(digest(ROOT/p) == sha for p, sha in frozen.items())
    if digest(ledger) != ledger_hash or not intact:
        raise ValueError('Frozen sources or API ledger changed')
    record = {'planned_episodes': 1, 'completed_episodes': 1, 'episode': summary,
              'paired_audit': audit, 'frozen_sources_unchanged': intact,
              'api_ledger_unchanged': True, 'new_api_calls': 0,
              'adapted_executor_sha256': adapted_hash, 'diagnostic_only': True}
    save(output/'summary.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
