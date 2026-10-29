"""Historical 4k local-model diagnostic on public contracts and recorded evidence.

No physical actions, training, reviewer model, or later episode outcomes are
supplied to Qwen. These are purposeful development checks, not a benchmark.
This preserves the registered 2026-10-24 allowance; new runs default to 32k
through run_local_agent_pilot.py or retest_qwen_output_budget.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.local_agent import LocalChatClient, function, parse_call, write_json
from vlm_tamp.compact_robot_interface import compact_response

SOURCE = ROOT / 'results/astra-luna-comparison-20261023/trials'


def events(case):
    return [json.loads(x) for x in (SOURCE / case / 'tool-events.jsonl').read_text().splitlines()]


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    public = out / 'public'
    public.mkdir()
    original = SOURCE / 'state039-r1-luna/agent-workspace'
    action_schema = function('submit_request', 'Record your proposed next robot request without executing it.',
        {'operation': {'type': 'string'}, 'arguments': {'type': 'object'}, 'brief_reason': {'type': 'string'}},
        ['operation', 'arguments', 'brief_reason'])
    assessment_schema = function('submit_assessment', 'Record an evidence-based assessment of the supplied reply.',
        {'physical_motion_issued': {'type': 'boolean'}, 'object_motion_established': {'type': 'boolean'},
         'geometric_unreachability_established': {'type': 'boolean'}, 'brief_reason': {'type': 'string'}},
        ['physical_motion_issued', 'object_motion_established', 'geometric_unreachability_established', 'brief_reason'])
    cases = [{'id': 'open_approach', 'kind': 'contract', 'schema': action_schema,
        'text': 'The current grasp plan is p003. Propose one request to execute its approach with the fingers open. '\
                'Use submit_request; no request will actually execute.\nContract:\n' + (original / 'tools/execute.txt').read_text(),
        'images': [], 'expected_private': {'operation': 'execute', 'plan_id': 'p003', 'gripper': -1}}]
    budget = next(x for x in events('state024-r1-luna') if 'insufficient_grasp_budget' in json.dumps(x['response']))
    cases.append({'id': 'budget_feedback', 'kind': 'feedback', 'schema': assessment_schema,
        'text': 'Interpret the following robot reply using submit_assessment. Distinguish what happened from what remains untested.\n' + json.dumps(budget['response']),
        'images': [], 'source_private': {'case': 'state024-r1-luna', 'event': budget['sequence']},
        'expected_private': {'physical_motion_issued': False, 'object_motion_established': False, 'geometric_unreachability_established': False}})
    motion = next(x for x in events('state039-r1-luna') if x['request']['operation'] == 'execute' and
                  compact_response(x['response']).get('result', {}).get('scene_pose_observation', {}).get('source') == 'unresolved_rgbd_motion')
    # Only public compact feedback is transmitted, without later decisions or evaluator geometry.
    cases.append({'id': 'motion_feedback', 'kind': 'feedback', 'schema': assessment_schema,
        'text': 'Interpret the following robot reply using submit_assessment. Distinguish robot execution from established object motion.\n' + json.dumps(compact_response(motion['response'])),
        'images': [], 'source_private': {'case': 'state039-r1-luna', 'event': motion['sequence']},
        'expected_private': {'physical_motion_issued': True, 'object_motion_established': False, 'geometric_unreachability_established': False}})
    for index, name in enumerate(['state039-r1-luna', 'state024-r1-luna', 'state049-r2-luna'], 1):
        row = next(x for x in events(name) if x['request']['operation'] == 'reset')
        images = []
        for camera in ['frontview', 'agentview']:
            relative = f'vision{index}/{camera}.png'
            target = public / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(SOURCE / name / 'agent-workspace' / row['response']['result']['images'][camera], target)
            images.append({'camera': camera, 'path': relative, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        cases.append({'id': f'vision{index}', 'kind': 'vision', 'schema': action_schema,
            'text': 'Task: open the microwave. The two images are current frontview and agentview, each 512 by 512. '\
                    'Propose a ground request from your interpretation of the images using submit_request. '\
                    'Use one supplied camera and its original pixel coordinates. Explain uncertainty briefly. '\
                    'This is an offline proposal; no physical action or geometry fit will run.\nContract:\n' + (original / 'tools/ground.txt').read_text(),
            'images': images, 'source_private': {'case': name, 'event': row['sequence']},
            'expected_private': {'geometric_correctness': 'Requires evaluator review; valid JSON alone does not establish a correct hinge.'}})
    write_json(out / 'cases-private.json', cases)
    protocol = {'model': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'scope': 'Purposefully selected development capability checks, not held-out performance or learning.',
        'case_count': len(cases), 'repetitions': 2, 'max_output_tokens': 4096,
        'temperature': .6, 'seeds': [17, 18], 'thinking': True, 'tool_choice': 'auto',
        'constrained_output_decoding': False, 'model_retries': 0, 'physical_robot_calls': 0,
        'gate': 'Proceed to the separate simulator smoke test if all six contract/feedback responses pass and all six image responses have valid ground-request structure. Visual accuracy is reviewed separately.',
        'private_rubric_sent_to_model': False, 'later_history_sent_to_model': False}
    sources = [Path(__file__), ROOT / 'src/vlm_tamp/local_agent.py', ROOT / 'src/vlm_tamp/compact_robot_interface.py']
    protocol['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protocol['cases_sha256'] = hashlib.sha256((out / 'cases-private.json').read_bytes()).hexdigest()
    for source in sources:
        target = out / 'source-snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(out / 'protocol.json', protocol)
    return cases


def grade(case, name, args):
    if case['kind'] == 'contract':
        return args.get('operation') == 'execute' and args.get('arguments', {}).get('plan_id') == 'p003' and args.get('arguments', {}).get('gripper') == -1
    if case['kind'] == 'feedback':
        return all(args.get(k) is v for k, v in case['expected_private'].items())
    if args.get('operation') != 'ground':
        return False
    regions = args.get('arguments', {}).get('regions', {})
    if regions.get('camera') not in {'frontview', 'agentview'}:
        return False
    for key in ['contact_box_xyxy', 'moving_panel_box_xyxy']:
        box = regions.get(key)
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (float, int)) and 0 <= v <= 511 for v in box):
            return False
        if not (box[0] < box[2] and box[1] < box[3]):
            return False
    hinge = regions.get('hinge_pixels_xy')
    direction = regions.get('initial_image_direction_xy')
    return (isinstance(hinge, list) and len(hinge) == 2 and
            all(isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and 0 <= v <= 511 for v in p) for p in hinge)
            and isinstance(direction, list) and len(direction) == 2 and
            all(isinstance(v, (int, float)) for v in direction) and any(direction))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8127/v1')
    args = parser.parse_args()
    out = args.output.resolve()
    cases = prepare(out)
    results = []
    for case in cases:
        for replicate in [1, 2]:
            key = f'{case["id"]}-r{replicate}'
            destination = out / 'responses' / key
            content = [{'type': 'text', 'text': case['text']}]
            for picture in case['images']:
                content.extend([{'type': 'text', 'text': picture['camera']}, {'type': 'local_image', 'path': picture['path']}])
            messages = [{'role': 'system', 'content': 'Use the provided function to submit your answer. Base it only on the supplied public evidence. Keep explanations brief.'},
                        {'role': 'user', 'content': content}]
            # Preserve this historical diagnostic's registered 4k budget.
            client = LocalChatClient(args.endpoint, 'qwen9b', destination, seed=16+replicate,max_tokens=4096,
                                     temperature=.6,presence_penalty=None)
            result = {'id': key, 'case': case['id'], 'replicate': replicate, 'kind': case['kind'],
                      'passed': False, 'schema_valid': False}
            try:
                response, timing = client.complete(messages, [case['schema']], out / 'public', timeout=240)
                calls = response.get('tool_calls', [])
                if len(calls) != 1:
                    raise ValueError(f'Expected one submission; got {len(calls)}')
                name, payload = parse_call(calls[0], [case['schema']])
                result.update(schema_valid=True, answer=payload, passed=grade(case, name, payload), timing=timing)
            except Exception as error:
                result['error'] = f'{type(error).__name__}: {error}'
                result['timing'] = client.calls[-1] if client.calls else None
            results.append(result)
            write_json(destination / 'result.json', result)
            write_json(out / 'results.json', results)
            print(json.dumps({'id': key, 'passed': result['passed'], 'seconds': (result.get('timing') or {}).get('wall_seconds'), 'error': result.get('error')}), flush=True)
    write_json(out / 'summary.json', {'responses': len(results), 'schema_valid': sum(x['schema_valid'] for x in results),
        'contract_feedback_passes': sum(x['passed'] for x in results if x['kind'] != 'vision'),
        'vision_structure_passes': sum(x['passed'] for x in results if x['kind'] == 'vision'),
        'passed_smoke_gate': len(results) == 12 and all(x['passed'] for x in results),
        'vision_geometry_accuracy': 'Not scored by the structural gate; inspect recorded image annotations.',
        'physical_success': 'Not measured in this offline diagnostic.'})


if __name__ == '__main__':
    main()
