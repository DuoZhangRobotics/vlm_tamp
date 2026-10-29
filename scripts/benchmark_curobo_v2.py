"""Compare v2 goals on retained observed scenes; no simulator or robot motion."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
import vamp
from vlm_tamp.arc_contact_scene import ArcContactScene
from vlm_tamp.coupled_scene import RobotVolumes
from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.curobo_v2_arc import CuroboV2ArcProposer
from vlm_tamp.mcvamp_contact_numerical import NativeScene, clear_native_cache
from vlm_tamp.partial_pose_arc import full_arc_plan
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def load_saved_arc(folder, record):
    reference = read_json(folder/'scene-reference.json')
    inputs = read_json(folder/'planning-input.json')
    path = Path(reference['arrays_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['arrays_sha256']:
        raise ValueError('Saved scene input hash changed')
    chain = RobotChain(record)
    robot = RobotVolumes(chain, reference['configured_robot_reference_volumes'],
                         reference['configured_robot_reference_joints'])
    with np.load(path, allow_pickle=False) as arrays:
        scene = ArcContactScene.from_points(arrays['static_points'], arrays['moving_reference'],
            arrays['contact_reference'], reference['spec'], config=reference['configuration'],
            robot=robot, reference_volumes=robot.reference_volumes)
        for name in ['static_points', 'moving_reference', 'contact_reference', 'inferred_panel_reference']:
            setattr(scene, name, arrays[name].copy())
    scene.role_reconciliation = copy.deepcopy(reference['scene_record']['handle_role_reconciliation'])
    metadata = copy.deepcopy(inputs['metadata'])
    start = np.asarray(inputs['reference_joints'])
    metadata['robot_collision_volumes_world'] = [v for v in robot.forward(start)
        if 'gripper' not in v['robot_geom_name']]+metadata['robot_collision_volumes_world']
    return scene, metadata, start, inputs, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/compact-luna-practice-20261022/practice-v1/attempt01')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events', type=int, nargs='*')
    parser.add_argument('--goals-only', action='store_true')
    parser.add_argument('--arc-solver-mode', choices=['local_continuation', 'global'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for source in [*ROOT.glob('src/vlm_tamp/*.py'), *ROOT.glob('src/vlm_tamp/rekep_libero/*.py'), Path(__file__)]:
        name = source.resolve().relative_to(ROOT)
        dest = args.output/'source'/name;dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        source_hashes[str(name)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(args.output/'source-hashes.json', source_hashes)
    record = read_json(args.source/'robot-chain.json')
    config = read_json(args.source/'007-arc/scene-reference.json')['configuration']
    config['curobo_backend'] = 'v2'
    empty = vamp.Environment()
    started = time.perf_counter()
    ik = TaskIK(record, config, lambda q: vamp.panda.validate(np.asarray(q).tolist(), empty, True),
                output=args.output/'ik')
    report = {'scope': __doc__, 'initialization_seconds': time.perf_counter()-started,
              'source': str(args.source), 'runtime': ik.client.metadata,
              'simulator_queried': False, 'robot_motion_issued': False, 'queries': []}
    try:
        for folder in sorted(args.source.glob('*-arc')):
            event = int(folder.name.split('-')[0])
            if args.events and event not in args.events:
                continue
            target = args.output/folder.name
            target.mkdir()
            scene, metadata, start, inputs, reference = load_saved_arc(folder, record)
            options = {**reference['configuration'], **inputs['search_options']}
            if args.arc_solver_mode:
                options['curobo_v2_arc_solver_mode'] = args.arc_solver_mode
                inputs['search_options']['curobo_v2_arc_solver_mode'] = args.arc_solver_mode
            goal = float(inputs['goal_angle_rad'])
            write_json(target/'scene-reference.json', reference)
            write_json(target/'planning-input.json', inputs)
            original = read_json(folder/'result.json')
            row = {'event': event, 'original_status': original['status'],
                   'original_seconds': original.get('total_seconds'), 'goal_rad': goal,
                   'grasp_constraint': options['grasp_constraint'], 'source_arrays_hash_verified': True}
            before = time.perf_counter()
            try:
                with NativeScene(scene, record, metadata, start, goal, aligned=False,
                                 grasp_constraint=options['grasp_constraint']) as context:
                    context.motion_options = options
                    with CuroboV2ArcProposer(ik, context, target, goal) as proposer:
                        row['scene_setup'] = proposer.setup
                        if args.goals_only:
                            _, goals, checks = proposer.generate(goal)
                            result = {'status': 'goals_admitted' if goals else 'no_admissible_goals',
                                      'admissible_goals': len(goals), 'goal_proposals': checks}
                            row['solved'] = None
                        else:
                            path, result = full_arc_plan(context, goal,
                                seconds=float(options.get('arc_search_seconds', 20.)),
                                proposals=int(options.get('arc_goal_proposals', 32)), goal_proposer=proposer)
                            row['solved'] = result['status'] == 'solved'
                            if len(path):
                                if not all(context.validate(q)['accepted'] and context.constraint_errors(q)['accepted'] for q in path):
                                    raise ValueError('Returned path fails native sampled-state validation')
                                np.save(target/'path.npy', path)
                                row['path_states'] = len(path)
                        row.update(status=result['status'], result=result,
                                   gpu_queries=len(proposer.queries),
                                   gpu_seconds=sum(q['gpu_seconds'] for q in proposer.queries))
                        write_json(target/'native-scene.json', context.record)
                        write_json(target/'result.json', result)
            except Exception as error:
                row.update(status='adapter_error', error=f'{type(error).__name__}: {error}', solved=False)
            row['wall_seconds'] = time.perf_counter()-before
            report['queries'].append(row)
            write_json(args.output/'results.json', report)
            print(json.dumps({k: v for k, v in row.items() if k in
                {'event', 'status', 'solved', 'wall_seconds', 'gpu_seconds', 'error'}}), flush=True)
            if row['status'] == 'adapter_error':
                raise RuntimeError(row['error'])
    finally:
        ik.close()
        clear_native_cache()


if __name__ == '__main__':
    main()
