"""Offline matched estimator replay: identical sensor frames, no physical actions.

The selected service implementation processes retained command endpoints. Its
executor is replaced only in this evaluator process with those recorded robot
states. Both versions receive exactly the same RGB-D and proprioception; native
object geometry is never read. Original planning requests run unchanged.
"""
import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze_contact_prefix import load_cameras
from diagnose_contact_prefix import compress_diagnostic_json, probe
from replay_subscription_episode import source_commands
from vlm_tamp.arc_contact_scene import ArcContactScene
from vlm_tamp.coupled_scene import RobotVolumes
from vlm_tamp.coupled_storage import preserve_scene, read_json, write_json
from vlm_tamp.mcvamp_contact_numerical import clear_native_cache
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain

ROOT = Path(__file__).resolve().parents[1]


def run(source, output, service_source):
    source, output, service_source = source.resolve(), output.resolve(), service_source.resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('matched_scene_service', service_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = read_json(source / 'summary.json')
    events = read_json(source / 'replay-events.json')
    protocol = read_json(source / 'protocol.json')
    historical = Path(protocol['source'])
    commands = {c['event']: c for c in source_commands(historical, 1)}
    plan_kinds = {}
    import json
    for line in (historical / 'tool-events.jsonl').read_text().splitlines():
        event = json.loads(line)
        plan_kinds[event['sequence']] = event['request']['operation']
    initial = source / 'snapshots/event003'
    reference = read_json(initial / 'scene-reference.json')
    observation = read_json(initial / 'observation.json')
    record = read_json(source / 'robot-chain.json')
    write_json(output / 'robot-chain.json', record)
    service = module.RobotService.__new__(module.RobotService)
    service.output = output
    service.workspace = output / 'public'
    service.public = service.workspace / 'robot_data'
    (service.public / 'attempt01').mkdir(parents=True)
    service.episode = output / 'attempt01'
    service.episode.mkdir()
    service.attempt, service.calls, service.epoch = 1, 3, observation['geometry_epoch']
    service.record, service.config = record, reference['configuration']
    service.ik = SimpleNamespace(chain=RobotChain(record))
    robot = RobotVolumes(service.ik.chain, reference['configured_robot_reference_volumes'],
                         reference['configured_robot_reference_joints'])
    with np.load(reference['arrays_path']) as arrays:
        service.scene = ArcContactScene.from_points(arrays['static_points'], arrays['moving_reference'],
            arrays['contact_reference'], reference['spec'], config=service.config,
            robot=robot, reference_volumes=robot.reference_volumes)
    service.scene.role_reconciliation = copy.deepcopy(reference['scene_record']['handle_role_reconciliation'])
    service.spec, service.cloud = service.scene.spec, service.scene.contact_reference
    service.choices, service.mask = observation['choices'], None
    service.anchors = [service.spec['source_anchor_world_m']]
    service.contact_scene_kind = 'articulated'
    service.plans, service.events, service.executions = {}, [], []
    service.tracker = service.tracking_result = None
    service.fit_diagnostics = service.stop_on_success = False
    service.sim = object()
    service.applied_pose = np.eye(4)
    sensor_hashes = {}

    def capture(event):
        snapshot = source / 'snapshots' / f'event{event:03d}'
        obs = read_json(snapshot / 'observation.json')
        metadata = read_json(snapshot / 'metadata.json')
        service.state = {'control_steps': obs['control_steps'], 'metadata': metadata}
        service.cameras = load_cameras(snapshot)
        for name, view in service.cameras.items():
            for key in ('robot_depth_prior', 'near_m', 'far_m'):
                if key in metadata['cameras'][name]:
                    view[key] = metadata['cameras'][name][key]
        public = copy.deepcopy(obs['observation'])
        workspace = Path(obs['public_workspace'])
        for key in ('sensor_arrays', 'camera_calibration'):
            path = workspace / public[key]
            public[key] = str(path)
            sensor_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        service.latest_observation = public
        return public

    capture(3)
    traces = []
    try:
        for event in events:
            seq, request = event['sequence'], event['request']
            if seq <= 3:
                continue
            operation, arguments = request['operation'], request.get('arguments', {})
            service.calls = seq
            if seq in commands:
                command = commands[seq]
                endpoint = read_json(source / f'snapshots/event{seq:03d}/observation.json')
                metadata = read_json(source / f'snapshots/event{seq:03d}/metadata.json')

                def recorded_execution(sim, state, refs, **kwargs):
                    assert np.array_equal(refs, command['references'])
                    assert kwargs['gripper'] == command['gripper']
                    assert state['control_steps'] + len(refs) == endpoint['control_steps']
                    return {'control_steps': endpoint['control_steps'], 'metadata': metadata}, {
                        'issued_commands': len(refs), 'planned_commands': len(refs),
                        'all_commands_issued': True, 'terminated_on_simulator_success': False,
                        'offline_recorded_endpoint': True}

                module.execute_batch = recorded_execution
                service.capture = lambda: capture(seq)
                label = arguments['plan_id'] if operation == 'execute' else ('close' if arguments.get('close', True) else 'open')
                arc = label.startswith('p') and plan_kinds[int(label[1:])] == 'plan_arc'
                result = service.issue(command['references'], command['gripper'], label, articulated=arc)
            elif operation == 'start_tracking':
                result = service.start_tracking()
            elif operation == 'measure_tracking':
                result = service.measure_tracking()
            elif operation == 'apply_tracking':
                result = service.apply_tracking(**arguments)
            elif operation == 'inspect_scene':
                result = service.safe_scene_evidence()
            else:
                raise ValueError(f'Unexpected retained prefix operation {seq}: {operation}')
            traces.append({'event': seq, 'operation': operation, 'control_steps': service.state['control_steps'],
                           'result': result, 'spec': copy.deepcopy(service.spec),
                           'geometry_epoch': service.epoch, 'scene_record': service.scene.record()})
            write_json(output / 'estimator-events.json', traces)
            folder = output / 'snapshots' / f'event{seq:03d}'
            folder.mkdir(parents=True)
            preserve_scene(service.scene, folder)
            print(f'{output.name}: event {seq}, {operation}, epoch {service.epoch}', flush=True)
        probes = []
        for prior in baseline['probes']:
            name = Path(prior['folder']).name
            result = probe(service.scene, record, service.state['metadata'], service.config,
                           prior['arguments'], output / 'probes' / name)
            probes.append(result)
            print(f'{output.name}: {name} {result["status"]}', flush=True)
        write_json(output / 'summary.json', {
            'scope': 'development matched offline estimator test; no physical execution or agent',
            'source': str(source), 'service_source': str(service_source),
            'service_sha256': hashlib.sha256(service_source.read_bytes()).hexdigest(),
            'sensor_sha256': sensor_hashes, 'native_object_geometry_used': False,
            'baseline_queries': baseline['probes'], 'queries': probes,
            'final_spec': service.spec, 'final_scene_record': service.scene.record()})
    finally:
        clear_native_cache()
        compression = compress_diagnostic_json(output)
        write_json(output / 'compression.json', compression)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--service-source', type=Path, default=ROOT / 'scripts/serve_subscription_robot.py')
    args = parser.parse_args()
    run(args.source, args.output, args.service_source)
