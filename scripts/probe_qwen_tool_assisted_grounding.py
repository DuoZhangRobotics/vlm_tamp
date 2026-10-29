"""Two bounded offline grounding sessions with public analysis tools.

No robot service, actuator, evaluator coordinates, cross-case memory or teacher
is exposed. A private observer evaluates proposals only after each actor ends.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from vlm_tamp.local_agent import LocalChatClient, PublicTools, function, parse_call, write_json
from vlm_tamp.robot_contracts import GROUND_HELP, operation_schemas, validate_operation
from vlm_tamp.subscription_agent import verify_isolation
from vlm_tamp.hinged_contact_tools import ground_handle
from validate_robot_io_geometry import load_case, choices, template
from evaluate_qwen_sampling_geometry import evaluate
from report_subscription_pilot import archive_workspace

CAMERAS = ('frontview', 'agentview')
SIZES = {name: {'width': 512, 'height': 512} for name in CAMERAS}
FINISH = function('finish_grounding', 'Finish this offline analysis. An accepted fit is only a candidate, not proof of a correct hinge or robot success.',
    {'chosen_ground_call': {'type': 'integer', 'minimum': 0, 'maximum': 3,
        'description': 'One-based accepted ground-call index; use 0 if unresolved.'},
     'brief_reason': {'type': 'string'}}, ['chosen_ground_call', 'brief_reason'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OfflineTools(PublicTools):
    def __init__(self, workspace, cameras, metadata):
        super().__init__(workspace, live=False, extra_readonly=[workspace / 'observation', workspace / 'tools', workspace / 'tools.txt'])
        self.active = {'ground'}
        self.image_sizes = copy.deepcopy(SIZES)
        self.cameras, self.metadata = cameras, metadata
        self.ground_results = []
        self.finished = None

    def schemas(self):
        return super().schemas() + [FINISH]

    def dispatch(self, name, args):
        if name == 'finish_grounding':
            chosen = args['chosen_ground_call']
            if chosen and (chosen > len(self.ground_results) or not self.ground_results[chosen - 1]['reply'].get('ok')):
                raise ValueError('Select an accepted ground call, or 0 for unresolved.')
            self.finished = copy.deepcopy(args)
            return {'ok': True, 'offline_analysis_finished': True}, []
        if name != 'ground':
            return super().dispatch(name, args)
        if len(self.ground_results) >= 3:
            raise ValueError('The three-call grounding budget is exhausted; finish with a candidate or unresolved.')
        validate_operation('ground', args, SIZES)
        started = time.monotonic()
        regions = args['regions']
        try:
            mechanism, cloud, mask, anchors, evidence = ground_handle(
                self.cameras, choices(regions), template(regions['camera']), self.metadata)
            reply = {'ok': True, 'ground_call': len(self.ground_results) + 1,
                'fit_accepted': True, 'physical_motion_issued': False,
                'estimated_hinge_origin_world_m': mechanism.spec['joint_origin_world_m'],
                'estimated_hinge_axis_world': mechanism.spec['joint_axis_world'],
                'contact_cloud_points': len(cloud),
                'motion_direction_check': evidence.get('motion_direction_check'),
                'interpretation': 'Accepted by the observed-depth fit. This does not verify the hinge hypothesis or establish task success.'}
        except Exception as error:
            reply = {'ok': False, 'ground_call': len(self.ground_results) + 1,
                     'error_type': type(error).__name__, 'error': str(error),
                     'physical_motion_issued': False}
        self.ground_results.append({'arguments': copy.deepcopy(args), 'reply': reply,
                                   'geometry_seconds': time.monotonic() - started})
        return reply, []


def prepare_observation(work, case, *, trial_root=None):
    source = (Path(trial_root) if trial_root is not None else ROOT / 'results/astra-luna-comparison-20261023/trials' / case) / 'agent-workspace/robot_data/attempt01/obs001'
    obs = work / 'observation'; obs.mkdir()
    calibration = json.loads((source / 'calibration.json').read_text())
    write_json(obs / 'calibration.json', {name: calibration[name] for name in CAMERAS})
    with np.load(source / 'sensors.npz') as data:
        fields = {key: data[key] for key in data.files if any(key.startswith(name + '_') for name in CAMERAS)}
        np.savez_compressed(obs / 'sensors.npz', **fields)
    for camera in CAMERAS:
        shutil.copyfile(source / (camera + '.png'), obs / (camera + '.png'))
    write_json(obs / 'manifest.json', {'image_sizes': SIZES,
        'coordinate_convention': 'Original image pixels, top-left origin, x right, y down.',
        'sensors': 'observation/sensors.npz', 'sensor_fields': list(fields),
        'sensor_array_metadata': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
            'indexing': 'array[y, x]' if value.ndim == 2 else 'array[y, x, channel]',
            'channels': ['x', 'y', 'z'] if key.endswith('_world_points') else ['red', 'green', 'blue'] if key.endswith('_rgb') else None}
            for key, value in fields.items()},
        'calibration': 'observation/calibration.json',
        'scope': 'Current public RGB-D and camera calibration; no native object poses or correct-answer annotations.'})
    tool_dir = work / 'tools'; (tool_dir / 'schemas').mkdir(parents=True)
    (tool_dir / 'ground.txt').write_text(GROUND_HELP)
    write_json(tool_dir / 'schema-index.json', {'operations': ['ground']})
    write_json(tool_dir / 'schemas/ground.json', operation_schemas(SIZES)['ground'])
    (work / 'tools.txt').write_text('Offline tools: read_file, view_image, run_python, write_file, ground, finish_grounding.\nGround fits observed depth without motion. Python can inspect public arrays, crop images and compute coordinates.\nAll paths are workspace-relative. observation/ and tools/ are read-only. Save generated images under scratch/; /tmp is isolated and not shared with the image viewer or later calls. Array layouts are in observation/manifest.json.\n')
    return source


def run_case(root, ident, case, endpoint, *, max_turns=8, retest=False,
             model_name='qwen9b', trial_root=None, case_loader=load_case, observer=evaluate):
    out = root / ident; out.mkdir()
    work = Path(tempfile.mkdtemp(prefix='vlm-qwen-ground-public-'))
    write_json(out / 'workspace-location.json', {'path': str(work), 'archive': 'agent-workspace'})
    source = prepare_observation(work, case, trial_root=trial_root)
    seen = set(json.loads((root.parent / 'protocol.json').read_text())['image_sha256'].values())
    images = {name: digest(work / 'observation' / (name + '.png')) for name in CAMERAS}
    if any(value in seen for value in images.values()):
        raise RuntimeError('Fresh observation repeats a prior Qwen image; stop before generation')
    cameras, metadata, provenance = case_loader(case)
    readonly = [work / 'observation', work / 'tools', work / 'tools.txt']
    isolation = verify_isolation(work, [ROOT / 'src/vlm_tamp/handle_contact.py',
        ROOT / 'results/astra-luna-comparison-20261023/grounding-diagnostics-private.json',
        root.parent / 'protocol.json', out / 'workspace-location.json'], readonly_paths=readonly)
    write_json(out / 'isolation.json', isolation)
    public_hashes = {str(p.relative_to(work)): digest(p) for folder in [work / 'observation', work / 'tools'] for p in folder.rglob('*') if p.is_file()}
    public_hashes['tools.txt'] = digest(work / 'tools.txt')
    write_json(out / 'input-hashes.json', public_hashes)
    protocol = {'id': ident, 'model': model_name, 'source_private': str(source.relative_to(ROOT)), 'images_sha256': images,
        'fresh_to_qwen_image_hashes': not retest, 'same_asset_development_observation': True,
        'retest_after_documentation_and_turn_budget_change': retest,
        'maximum_model_turns': max_turns, 'maximum_ground_calls': 3, 'seconds': 1200,
        'output_tokens': 32768, 'temperature': 1., 'presence_penalty': 1.5,
        'geometry_provenance': provenance, 'private_answers_supplied': False,
        'physical_motion': False, 'cross_case_memory': False, 'reflection_reviewer': False}
    write_json(out / 'protocol.json', protocol)
    public_tools = OfflineTools(work, cameras, metadata)
    client = LocalChatClient(endpoint, model_name, out / 'model-calls', seed=17,
                             temperature=1., presence_penalty=1.5)
    instruction = (
        'Task: ground the handle and hinged panel for opening the microwave in the current observations. '
        'This is an offline analysis; no robot motion is available. You may inspect the supplied original images, '
        'use Python to crop or analyze them, and inspect public depth arrays/calibration in observation/manifest.json. '
        'Python has NumPy, SciPy and Pillow. All paths must be workspace-relative. observation/ and tools/ are '
        'read-only. Save generated files under scratch/ (already exists); /tmp is isolated and cannot be used '
        'with view_image or later Python calls. Array shapes, dtypes and indexing are in observation/manifest.json. '
        'Inspect generated crops using view_image; crop coordinates must be '
        'converted back to original-image pixels before ground. You choose the analysis; no correct coordinates are supplied. '
        'Use ground to check your hypotheses against depth. Fit acceptance does not establish a correct hinge. '
        f'At most three ground calls and {max_turns} model turns are available. Call finish_grounding with the accepted '
        'ground-call index you choose, or 0 if unresolved. Explain uncertainty briefly.\nContract:\n' + GROUND_HELP)
    content = [{'type': 'text', 'text': instruction}]
    for camera in CAMERAS:
        content.extend([{'type': 'text', 'text': camera + ': observation/' + camera + '.png, 512 by 512'},
                        {'type': 'local_image', 'path': 'observation/' + camera + '.png'}])
    messages = [{'role': 'system', 'content': 'Use the supplied public tools to inspect evidence and submit an offline grounding candidate. Keep explanations concise and distinguish a numerical fit from verified physical success.'},
                {'role': 'user', 'content': content}]
    started, events = time.monotonic(), []
    reason, error = 'turn_budget_exhausted', None
    try:
        for turn in range(1, max_turns + 1):
            remaining = 1200 - (time.monotonic() - started)
            if remaining <= 0:
                reason = 'wall_budget_exhausted'; break
            schemas = public_tools.schemas()
            write_json(root / 'status.json', {'phase': 'running', 'case': ident, 'turn': turn,
                'current_call': f'{root.name}/{ident}/model-calls/call{turn:03d}'})
            response, timing = client.complete(messages, schemas, work, timeout=min(900, remaining))
            messages.append(response)
            if timing['finish_reason'] == 'length':
                reason = 'model_output_limit'; break
            if timing['finish_reason'] not in {'stop', 'tool_calls'}:
                reason = 'unexpected_finish_reason'; break
            calls = response.get('tool_calls', [])
            if not calls:
                if public_tools.ground_results:
                    reason = 'final_text_after_grounding'; break
                messages.append({'role': 'user', 'content': 'No grounding call has been submitted. Use the available analysis/ground tools, or finish_grounding with 0 if unresolved.'})
                continue
            extra_messages = []
            for call in calls:
                begin = time.monotonic()
                event = {'turn': turn, 'call': call}
                try:
                    if public_tools.finished is not None or time.monotonic() - started >= 1200:
                        raise RuntimeError('Offline session is terminal; additional tool call was not run.')
                    name, arguments = parse_call(call, schemas)
                    event.update(schema_valid=True, operation=name)
                    result, extra = public_tools.dispatch(name, arguments)
                    extra_messages.extend(extra)
                except Exception as exc:
                    result = {'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)[:4000]}
                    if hasattr(exc, 'request_diagnostic'):
                        result['request_diagnostic'] = exc.request_diagnostic
                event.update(result=result, seconds=time.monotonic() - begin)
                events.append(event)
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)})
                with (out / 'tool-events.jsonl').open('a') as log:
                    log.write(json.dumps(event) + '\n')
            messages.extend(extra_messages)
            write_json(out / 'conversation.json', messages)
            write_json(out / 'ground-results.json', public_tools.ground_results)
            if public_tools.finished is not None:
                reason = 'actor_finished'; break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        reason = 'runtime_error'
    finally:
        write_json(out / 'conversation.json', messages)
        write_json(out / 'ground-results.json', public_tools.ground_results)
        result = {'id': ident, 'termination_reason': reason, 'error': error,
            'actor_wall_seconds': time.monotonic() - started,
            'model_seconds': sum(c['wall_seconds'] for c in client.calls),
            'model_calls': len(client.calls), 'tool_seconds': sum(e['seconds'] for e in events),
            'geometry_seconds': sum(g['geometry_seconds'] for g in public_tools.ground_results),
            'context_admission_seconds': sum((c.get('context_budget') or {}).get('tokenization_seconds', 0) for c in client.calls),
            'tool_calls': len(events), 'ground_calls': len(public_tools.ground_results),
            'accepted_fits': sum(g['reply'].get('ok', False) for g in public_tools.ground_results),
            'actor_selection': public_tools.finished,
            'output_tokens': sum((c.get('usage') or {}).get('completion_tokens', 0) for c in client.calls),
            'output_limit_hits': sum(c.get('finish_reason') == 'length' for c in client.calls),
            'input_hashes_match': all(digest(work / p) == h for p, h in public_hashes.items()),
            'isolation_passed': isolation['passed'], 'physical_motion': False}
        write_json(out / 'summary.json', result)
        archive_workspace(work, out / 'agent-workspace')
        shutil.rmtree(work)
    # Observer-only native scoring happens after the actor has ended and its
    # public workspace is gone. Nothing from this assessment enters messages.
    observer_rows = [{'id': f'{ident}-ground{i}', 'contract_valid': True,
                      'source_private': {'case': case}, 'arguments': item['arguments']}
                     for i, item in enumerate(public_tools.ground_results, 1)]
    geometry = observer(observer_rows, out / 'geometry-private.json') if observer_rows else []
    result['private_evaluation_seconds'] = sum(g.get('geometry_seconds', 0) for g in geometry)
    result['observer_geometry'] = [{k: g.get(k) for k in ['id', 'geometry_accepted', 'native_hinge_line_error_mm', 'error']} for g in geometry]
    write_json(out / 'summary.json', result)
    write_json(root / 'status.json', {'phase': 'case_complete', 'case': ident})
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8127/v1')
    parser.add_argument('--preflight-only', action='store_true', help='Validate the isolated public tool interface without model generation.')
    parser.add_argument('--retest', action='store_true', help='The single registered documentation/16-turn retest; original artifacts stay intact.')
    args = parser.parse_args(); parent = args.run.resolve()
    if args.preflight_only:
        work = Path(tempfile.mkdtemp(prefix='vlm-qwen-ground-preflight-'))
        try:
            prepare_observation(work, 'state020-r1-luna')
            isolation = verify_isolation(work, [ROOT / 'src/vlm_tamp/handle_contact.py',
                ROOT / 'results/astra-luna-comparison-20261023/grounding-diagnostics-private.json'],
                readonly_paths=[work / 'observation', work / 'tools', work / 'tools.txt'])
            cameras, metadata, _ = load_case('state020-r1-luna')
            adapter = OfflineTools(work, cameras, metadata)
            computation, _ = adapter.dispatch('run_python', {'code':
                "import json,numpy as np;from PIL import Image;im=Image.open('observation/frontview.png');"
                "im.crop((0,0,64,64)).save('scratch/crop.png');data=np.load('observation/sensors.npz');"
                "print(json.dumps({'image_size':im.size,'sensor_fields':len(data.files),'crop_saved':True}))"})
            crop, images = adapter.dispatch('view_image', {'path': 'scratch/crop.png'})
            schema_names = [x['function']['name'] for x in adapter.schemas()]
            result = {'isolation': isolation, 'python': computation, 'crop': crop,
                      'image_messages': images, 'tools': schema_names, 'model_calls': 0,
                      'passed': isolation['passed'] and computation.get('ok') and crop['width'] == 64 and 'execute' not in schema_names}
            write_json(parent / ('tool-assisted-retest-preflight.json' if args.retest else 'tool-assisted-preflight.json'), result)
            if not result['passed']:
                raise RuntimeError('Offline tool preflight failed')
            print(json.dumps({'passed': True, 'model_calls': 0, 'tools': schema_names}), flush=True)
        finally:
            shutil.rmtree(work)
        return
    gate = json.loads((parent / 'study/summary.json').read_text())
    if not gate['tool_assisted_gate_passed']:
        raise ValueError('Registered completion/schema gate was not met; no tool-assisted calls authorized by this protocol')
    if args.retest and not (parent / 'tool-assisted-retest-protocol.json').is_file():
        raise ValueError('Register the bounded retest protocol before generation')
    root = parent / ('tool-assisted-retest' if args.retest else 'tool-assisted'); root.mkdir(exist_ok=False)
    sources = [Path(__file__), ROOT / 'src/vlm_tamp/local_agent.py', ROOT / 'src/vlm_tamp/local_context_budget.py',
        ROOT / 'src/vlm_tamp/handle_contact.py', ROOT / 'scripts/validate_robot_io_geometry.py',
        ROOT / 'scripts/evaluate_qwen_sampling_geometry.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    write_json(root / 'source-hashes.json', hashes)
    for name in hashes:
        destination = root / 'source-snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    rows = []
    for ident, case in [('fresh01', 'state020-r1-luna'), ('fresh02', 'state035-r1-luna')]:
        rows.append(run_case(root, ident, case, args.endpoint, max_turns=16 if args.retest else 8, retest=args.retest))
        write_json(root / 'summary.json', {'cases': rows, 'scope': 'Offline tool-assisted feasibility on the same asset. Initial cases were fresh to Qwen; this retest reuses them with clearer paths/layouts and more turns. No practice transfer or physical success established.' if args.retest else 'Offline tool-assisted feasibility; same asset, fresh to Qwen. No practice transfer or physical success established.'})
    write_json(root / 'source-audit.json', {'passed': all(digest(ROOT / p) == h for p, h in hashes.items())})
    write_json(root / 'status.json', {'phase': 'complete', 'cases': len(rows)})


if __name__ == '__main__':
    main()
