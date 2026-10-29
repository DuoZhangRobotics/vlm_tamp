"""Historical 4k follow-up: isolate image-count and coordinate-unit effects.

One visible handle, eight fresh responses; no robot actions or answer hints.
The approximate human box is evaluator-only and is never sent to the model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.local_agent import LocalChatClient, function, parse_call, write_json


def box_metrics(box, reference):
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
        return {'valid_box': False}
    l, t, r, b = box
    if not (0 <= l < r <= 512 and 0 <= t < b <= 512):
        return {'valid_box': False}
    x1, y1, x2, y2 = reference
    overlap = max(0, min(r, x2)-max(l, x1)) * max(0, min(b, y2)-max(t, y1))
    union = (r-l)*(b-t) + (x2-x1)*(y2-y1) - overlap
    return {'valid_box': True, 'iou_approximate_human_box': overlap/union,
            'centre_inside_approximate_handle': x1 <= (l+r)/2 <= x2 and y1 <= (t+b)/2 <= y2,
            'centre_error_pixels': (((l+r-x1-x2)/2)**2 + ((t+b-y1-y2)/2)**2)**.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8127/v1')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    public = out / 'public'
    public.mkdir()
    source = ROOT / 'results/astra-luna-comparison-20261023/trials/state039-r1-luna/agent-workspace/robot_data/attempt01/obs001'
    for camera in ['frontview', 'agentview']:
        shutil.copyfile(source / (camera + '.png'), public / (camera + '.png'))
    reference = [317, 219, 342, 320]
    jobs = [{'views': views, 'units': units, 'replicate': rep}
            for views in [1, 2] for units in ['pixels', 'normalized_1000'] for rep in [1, 2]]
    random.Random(20261024).shuffle(jobs)
    protocol = {'question': 'Does a simpler visual query fail because of multiple views or coordinate units?',
        'model': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'design': '2 image-count conditions x 2 coordinate conventions x 2 fresh-context repetitions, one development scene.',
        'schedule': jobs, 'temperature': .6, 'thinking': True, 'max_output_tokens': 4096,
        'retries': 0, 'training': False, 'physical_motion': False,
        'reference_box_private': reference, 'reference_source': 'Approximate human annotation of the visible frontview handle, registered before querying.',
        'pass_definition': 'Converted box centre inside the approximate human handle box and IoU >= 0.30.',
        'reference_sent_to_model': False, 'image_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in public.glob('*.png')},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / 'src/vlm_tamp/local_agent.py']}}
    write_json(out / 'protocol-private.json', protocol)
    results = []
    for job in jobs:
        ident = f'{job["views"]}view-{job["units"]}-r{job["replicate"]}'
        limit = 511 if job['units'] == 'pixels' else 1000
        convention = ('Use original 512 by 512 image pixel coordinates, from 0 to 511.' if job['units'] == 'pixels' else
                      'Use coordinates normalized from 0 to 1000 along each image axis: left/top is 0, right/bottom is 1000.')
        schema = function('submit_box', 'Submit the tight bounding box of the requested visible object.',
            {'box_xyxy': {'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': limit}, 'minItems': 4, 'maxItems': 4},
             'brief_description': {'type': 'string'}}, ['box_xyxy', 'brief_description'])
        text = ('Locate the microwave door handle in the image labelled frontview. Return its tight bounding box [left, top, right, bottom] with submit_box. '
                'The origin is the top-left corner; x increases rightward and y increases downward. ' + convention +
                ' Give coordinates for the frontview image only.' +
                (' The second image, labelled agentview, is another view of the same scene; do not return coordinates in that second image.' if job['views'] == 2 else ''))
        content = [{'type': 'text', 'text': text}, {'type': 'text', 'text': 'frontview'}, {'type': 'local_image', 'path': 'frontview.png'}]
        if job['views'] == 2:
            content.extend([{'type': 'text', 'text': 'agentview'}, {'type': 'local_image', 'path': 'agentview.png'}])
        # Preserve this historical diagnostic's registered 4k budget.
        client = LocalChatClient(args.endpoint, 'qwen9b', out / 'responses' / ident, seed=16+job['replicate'],max_tokens=4096,
                                 temperature=.6,presence_penalty=None)
        result = {'id': ident, **job, 'passed': False}
        try:
            response, timing = client.complete([{'role': 'system', 'content': 'Use the provided function to submit your answer. Base it only on the supplied images. Keep explanations brief.'},
                                                {'role': 'user', 'content': content}], [schema], public, timeout=240)
            calls = response.get('tool_calls', [])
            if len(calls) != 1:
                raise ValueError(f'Expected one submission; got {len(calls)}')
            _, answer = parse_call(calls[0], [schema])
            box = answer['box_xyxy']
            pixels = [x*.512 for x in box] if job['units'] == 'normalized_1000' else box
            metrics = box_metrics(pixels, reference)
            result.update(answer=answer, box_pixels=pixels, metrics=metrics, timing=timing,
                          passed=metrics.get('centre_inside_approximate_handle', False) and metrics.get('iou_approximate_human_box', 0) >= .30)
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
            result['timing'] = client.calls[-1] if client.calls else None
        results.append(result)
        write_json(out / 'responses' / ident / 'result.json', result)
        write_json(out / 'results.json', results)
        print(json.dumps(result), flush=True)
    write_json(out / 'summary.json', {'responses': len(results), 'passes': sum(x['passed'] for x in results),
        'by_condition': {f'{v}view-{u}': {'passed': sum(x['passed'] for x in results if x['views'] == v and x['units'] == u), 'responses': 2}
                         for v in [1, 2] for u in ['pixels', 'normalized_1000']},
        'scope': 'Single-frame handle-box diagnostic with an approximate human reference; not physical manipulation or held-out transfer.'})


if __name__ == '__main__':
    main()
