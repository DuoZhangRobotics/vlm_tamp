"""Read-only, fresh-agent comparison of recorded door-motion interpretations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from vlm_tamp.subscription_agent import clean_environment, codex_arguments, verify_isolation


REFERENCE_PROMPT = """Before deciding, identify the fixed appliance opening/front,
the moving door, and their attachment edge in the images. Describe where the
door would lie when closed relative to the fixed appliance. Then assess whether
the second image is closer to or farther from that configuration. Base the
answer on that relationship, not merely screen-left/right movement or panel size.
If a required reference is obscured, state that uncertainty. No closed-state
example, motion direction, expected answer, or object joint measurement is supplied.
"""

POINT_TOOL = '''"""Read exact selected RGB-D pixels; no object model or task inference."""
import json
from pathlib import Path
import sys
import numpy as np

capture, camera = sys.argv[1:3]
if capture not in {'before', 'after'}:
    raise ValueError('capture must be before or after')
pixels = json.loads(sys.argv[3])
results = []
with np.load(Path('inputs') / capture / 'sensors.npz') as arrays:
    points = arrays[camera + '_world_points']
    depth = arrays[camera + '_depth']
    known_robot = arrays[camera + '_known_robot_depth_prior']
    for pixel in pixels:
        x, y = pixel
        if type(x) is not int or type(y) is not int:
            raise ValueError('Pixel coordinates must be integers')
        if not (0 <= x < points.shape[1] and 0 <= y < points.shape[0]):
            raise ValueError('Pixel is outside the original image')
        point = points[y, x]
        finite = bool(np.isfinite(point).all() and np.isfinite(depth[y, x]))
        results.append({'pixel_xy': pixel, 'valid_finite_depth': finite,
            'point_world_m': point.tolist() if finite else None,
            'depth_m': float(depth[y, x]) if finite else None,
            'robot_depth_prior_m': float(known_robot[y, x]) if np.isfinite(known_robot[y, x]) else None,
            'robot_depth_prior_scope': 'Expected robot-only depth at this ray; null means no predicted robot surface. This is not an object identity or occlusion label.'})
print(json.dumps({'capture': capture, 'camera': camera, 'samples': results}))
'''


def write_json(path: Path, value: object) -> None:
    """Write a compact, inspectable experiment record."""
    path.write_text(json.dumps(value, indent=2) + '\n')


def main() -> None:
    """Prepare one declared condition and execute a fresh isolated Luna session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--before', type=int, default=1)
    parser.add_argument('--after', type=int, default=10)
    parser.add_argument('--condition', choices=['pair', 'reference', 'multiview_rgbd'], required=True)
    parser.add_argument('--seconds', type=int, default=240)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.study.resolve() / args.condition
    out.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='vlm-luna-goal-reference-'))
    write_json(out / 'active-workspace.json', {'workspace': str(work)})
    public = {'image_coordinates': 'original pixels, origin top-left; x right, y down',
              'chronology': ['before', 'after'], 'captures': {}}
    provenance = []
    source_steps = {}
    for label, observation in [('before', args.before), ('after', args.after)]:
        source = run / f'agent-workspace/robot_data/attempt01/obs{observation:03d}'
        destination = work / 'inputs' / label
        destination.mkdir(parents=True)
        cameras = sorted(p.stem for p in source.glob('*.png')) if args.condition == 'multiview_rgbd' else ['agentview']
        paths = [source / (camera + '.png') for camera in cameras]
        if args.condition == 'multiview_rgbd':
            paths += [source / 'sensors.npz', source / 'calibration.json']
        for path in paths:
            target = destination / path.name
            shutil.copyfile(path, target)
            provenance.append({'public_file': str(target.relative_to(work)),
                               'source_file': str(path),
                               'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        public['captures'][label] = {'images': {camera: f'inputs/{label}/{camera}.png' for camera in cameras}}
        source_steps[label] = json.loads((source / 'robot.json').read_text())['control_steps']
    public['main_pair'] = ['inputs/before/agentview.png', 'inputs/after/agentview.png']
    if args.condition == 'multiview_rgbd':
        public['rgbd'] = {'arrays': 'inputs/{before|after}/sensors.npz',
                         'keys': '{camera}_rgb, {camera}_depth, {camera}_world_points, {camera}_known_robot_depth_prior',
                         'units': 'metres; XYZ points share world coordinates across captures; known_robot_depth_prior is floating-point robot-only depth, NaN where no robot surface is predicted',
                         'indexing': 'array[y, x]; RGB files use the same original pixels',
                         'calibration': 'inputs/{before|after}/calibration.json',
                         'availability': 'Only listed views exist at each timestamp; no missing view is synthesized. Wrist camera moves with the arm.'}
        (work / 'sample_points.py').write_text(POINT_TOOL)
    write_json(work / 'observations.json', public)
    write_json(out / 'input-provenance.json', provenance)
    evaluator = json.loads((run / 'evaluator-diagnostics.json').read_text())
    angles = {60: evaluator['initial_open_deg']}
    for motion in evaluator['motions']:
        angles[motion['start_control']] = motion['opening_before_deg']
        angles[motion['end_control']] = motion['opening_after_deg']
    change = angles[source_steps['after']] - angles[source_steps['before']]
    expected = 'opening' if change > 0 else 'closing' if change < 0 else 'unchanged'
    write_json(out / 'evaluator-only-labels.json', {'expected_motion': expected,
        'source_control_steps': source_steps,
        'opening_before_deg': angles[source_steps['before']],
        'opening_after_deg': angles[source_steps['after']], 'native_object_data_supplied': False})
    forbidden = [out / 'evaluator-only-labels.json', run / 'evaluation-private.json',
                 Path(__file__).resolve(), Path.home() / '.codex/sessions']
    isolation = verify_isolation(work, forbidden, readonly_paths=[work / 'inputs'])
    write_json(out / 'isolation.json', isolation)
    prompt = """This is a read-only visual interpretation test of a robot arm operating
a microwave door. Read observations.json, then use your image-view tool to open
the two original main_pair images individually, in their listed before/after order.
Is the DOOR being opened, closed, unchanged, or is the evidence uncertain?
Base the answer on the observations. Do not read other project code, past results,
agent histories or anything outside this public workspace. No robot actions exist.
"""
    if args.condition != 'pair':
        prompt += REFERENCE_PROMPT
    if args.condition == 'multiview_rgbd':
        prompt += """Additional recorded views and measured RGB-D/calibration are listed
in observations.json. Inspect useful additional views. Use numerical measurements
to check the relationship you identified: select visible fixed-opening and door
pixels yourself. sample_points.py returns exact selected world points with no
object semantics, hidden object poses or direction labels. Example syntax:
python3 sample_points.py before CAMERA '[[x,y], [x,y]]'
Replace CAMERA and coordinates with your own selections. NumPy/SciPy are available
for further calculations on the public arrays. Do not assume nonexistent after
views. Report which measurements actually support your answer or why they do not.
"""
    prompt += """Write answers.json with these fields:
{"motion":"opening|closing|unchanged|uncertain", "confidence":"low|medium|high",
 "visible_evidence":"brief observable evidence", "fixed_goal_reference":"identified reference or unresolved",
 "closed_configuration":"observed relationship that closure would require, or unresolved",
 "measurements_used":[], "additional_evidence_needed":"if any"}.
Give the answer you find, including uncertainty; no particular answer is expected.
"""
    (out / 'prompt.txt').write_text(prompt)
    arguments = codex_arguments(work, out / 'agent-last.txt', model='gpt-6-luna',
                                reasoning='high', readonly_paths=[work / 'inputs'], ephemeral=False)
    write_json(out / 'agent-command.json', arguments)
    started = time.monotonic()
    with (out / 'agent-events.jsonl').open('w') as log, (out / 'agent-stderr.log').open('w') as errors:
        process = subprocess.Popen(arguments, stdin=subprocess.PIPE, text=True, cwd=work,
                                   env=clean_environment(), stdout=log, stderr=errors)
        write_json(out / 'process.json', {'pid': process.pid, 'started': time.time()})
        try:
            process.stdin.write(prompt)
            process.stdin.close()
            returncode = process.wait(timeout=args.seconds)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            write_json(out / 'agent-exit.json', {'returncode': process.returncode,
                'wall_seconds': time.monotonic() - started, 'robot_actions_available': False})
            shutil.copytree(work, out / 'agent-workspace',
                            ignore=shutil.ignore_patterns('.analysis_site', '.codex', '.git', '.aws', '.agents'))
            shutil.rmtree(work)
    if returncode:
        raise RuntimeError(f'Luna exited with {returncode}; inspect {out}')
    answer = json.loads((out / 'agent-workspace/answers.json').read_text())
    write_json(out / 'evaluation.json', {'condition': args.condition, 'answer': answer,
        'expected_motion': expected, 'correct': answer['motion'] == expected,
        'isolation_passed': isolation['passed'], 'robot_motion_executed': False})
    print(json.dumps({'condition': args.condition, 'answer': answer,
                      'correct': answer['motion'] == expected}, indent=2))


if __name__ == '__main__':
    main()
