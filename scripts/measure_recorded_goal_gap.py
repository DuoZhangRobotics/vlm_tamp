"""Track frozen model-selected references and measure their observed RGB-D gap."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from vlm_tamp.goal_gap import landmark_sample, measured_gap
from vlm_tamp.observed_video import sampled_steps
from vlm_tamp.part_motion import CoTrackerBackend

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: object) -> None:
    """Save explicit missing measurements as null, never nonfinite JSON values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def capture_catalog(source: Path) -> list[dict]:
    """Read source capture timestamps without consulting object state."""
    return [{'observation_id': int(p.name[3:]), 'control_step': json.loads((p/'robot.json').read_text())['control_steps'],
             'folder': p} for p in sorted(source.glob('obs*'))]


def dense_camera_sequence(run: Path, catalog: list[dict], output: Path) -> tuple[list[dict], list[dict]]:
    """Decode every recorded external-camera frame in memory, retaining exact capture RGB."""
    initial = run/'attempt01/initial'
    movie = initial/'rollout.mp4'
    clock = json.loads((initial/'video-frames.json').read_text())
    by_step: dict[int, list[dict]] = defaultdict(list)
    for row in catalog:by_step[row['control_step']].append(row)
    steps = sorted(step for step in set(clock['source_control_steps']) | set(by_step)
                   if min(by_step) <= step <= max(by_step))
    indices = {step:i for i,step in enumerate(clock['presentation_control_steps'])}
    expression = '+'.join(f'eq(n\\,{indices[step]})' for step in steps)
    encoder = shutil.which('ffmpeg')
    if not encoder:raise RuntimeError('The recorded-video decoder is unavailable')
    with Image.open(catalog[0]['folder']/'agentview.png') as image:
        width,height = image.size
    views,records = [],[]
    movie_hash = hashlib.sha256(movie.read_bytes()).hexdigest()
    with (output/'dense-decode.log').open('w') as log:
        process = subprocess.Popen([encoder,'-nostdin','-loglevel','error','-threads','1',
            '-i',str(movie),'-vf','select='+expression,'-vsync','0','-frames:v',str(len(steps)),
            '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=log)
        try:
            for step in steps:
                payload = process.stdout.read(width*height*3)
                if len(payload) != width*height*3:
                    raise RuntimeError('Dense video decode ended early; inspect dense-decode.log')
                if step in by_step:
                    for capture in by_step[step]:
                        path = capture['folder']/'agentview.png'
                        rgb = np.asarray(Image.open(path).convert('RGB')).copy()
                        records.append({'control_step':step,'observation_id':capture['observation_id'],
                            'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'source':'exact RGB-D capture'})
                        views.append({'rgb':rgb})
                else:
                    rgb = np.frombuffer(payload,dtype=np.uint8).reshape(height,width,3).copy()
                    records.append({'control_step':step,'observation_id':None,'source':'recorded MP4 RGB only',
                        'source_video':str(movie),'source_video_sha256':movie_hash,
                        'video_frame_index':indices[step],'rgb_sha256':hashlib.sha256(payload).hexdigest()})
                    views.append({'rgb':rgb})
            process.stdout.close()
            if process.wait(timeout=30):raise RuntimeError('Dense video decoder failed; inspect dense-decode.log')
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=20)
    write(output/'agentview-tracking-inputs.json',records)
    return views,records


def camera_sequence(run: Path, catalog: list[dict], camera: str, output: Path) -> tuple[list[dict], list[dict]]:
    """Use exact capture RGB at depth times and recorded video between external views."""
    by_step: dict[int, list[dict]] = defaultdict(list)
    for row in catalog:by_step[row['control_step']].append(row)
    records = []
    if camera == 'agentview':
        initial = run / 'attempt01/initial'
        clock = json.loads((initial/'video-frames.json').read_text())
        available = sorted(set(clock['source_control_steps']) | set(by_step))
        duplicates = len(catalog) - len(by_step)
        steps = sampled_steps(available, sorted(by_step), 64-duplicates)
        middle = [step for step in steps if step not in by_step]
        folder = output / 'tracking-rgb'
        folder.mkdir()
        if middle:
            indices = {step: i for i, step in enumerate(clock['presentation_control_steps'])}
            expression = '+'.join(f'eq(n\\,{indices[step]})' for step in middle)
            encoder = shutil.which('ffmpeg')
            if not encoder:raise RuntimeError('The recorded-video decoder is unavailable')
            subprocess.run([encoder, '-nostdin', '-loglevel', 'error', '-threads', '1',
                '-i', str(initial/'rollout.mp4'), '-vf', 'select='+expression, '-vsync', '0',
                '-frames:v', str(len(middle)), str(folder/'decoded-%03d.png')], check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
        decoded = {step: folder/f'decoded-{i+1:03d}.png' for i, step in enumerate(middle)}
        for step in steps:
            if step in by_step:
                records.extend({'control_step': step, 'observation_id': r['observation_id'],
                                'path': r['folder']/(camera+'.png'), 'source': 'exact RGB-D capture'}
                               for r in by_step[step])
            else:
                records.append({'control_step': step, 'observation_id': None,
                                'path': decoded[step], 'source': 'recorded MP4 RGB only'})
    else:
        records = [{'control_step': r['control_step'], 'observation_id': r['observation_id'],
                    'path': r['folder']/(camera+'.png'), 'source': 'exact RGB-D capture'} for r in catalog]
    views = [{'rgb': np.asarray(Image.open(row['path']).convert('RGB')).copy()} for row in records]
    manifest = [{**r, 'path': str(r['path']),
                 'sha256': hashlib.sha256(r['path'].read_bytes()).hexdigest()} for r in records]
    write(output/(camera+'-tracking-inputs.json'), manifest)
    return views, records


def load_depth(folder: Path, camera: str) -> dict | None:
    """Load only that camera's saved measured depth and calibration-derived world points."""
    if not (folder/'sensors.npz').exists():return None
    with np.load(folder/'sensors.npz', allow_pickle=False) as arrays:
        return {key: arrays[camera+'_'+key] for key in ['depth', 'world_points', 'known_robot_depth_prior']}


def measure_case(study: Path, state: int, tracker: CoTrackerBackend | None,
                 *, use_saved_tracks: bool = False, all_recorded_rgb: bool = False) -> None:
    """Track selected material-point hypotheses with no semantic repair or alternate backend."""
    started = time.perf_counter()
    selection = study / f'state{state:03d}'
    output = study/'dense_tracking'/selection.name if all_recorded_rgb else selection
    output.mkdir(parents=True,exist_ok=True)
    reference_path = selection/'agent-workspace/references.json'
    frozen = json.loads((selection/'reference-freeze.json').read_text())
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != frozen['sha256']:
        raise ValueError('The reference choices changed after selection')
    references = json.loads(reference_path.read_text())
    previous_path = output/'gap.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    run = ROOT / (f'results/luna-molmo2-integration-v3-20261103/trials/'
                  f'baseline-r1-close_microwave-state{state:03d}-600')
    source = run/'agent-workspace/robot_data/attempt01'
    catalog = capture_catalog(source)
    by_id = {r['observation_id']: r for r in catalog}
    roles = ['moving', 'fixed_closed']
    if any(references.get(role) is None for role in roles):
        write(output/'gap.json', {'status': 'reference_pair_unresolved', 'references': references,
            'samples': [], 'robot_motion_executed': False,
            'reason': 'The actor did not supply both visible reference hypotheses; no substitute point was chosen.'})
        return
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for role in roles:
        selected = references[role]
        camera, observation_id = selected['camera'], selected['observation_id']
        if camera not in ['agentview', 'wrist'] or observation_id not in by_id:
            raise ValueError('Reference does not identify a supplied camera/capture')
        pixel = np.asarray(selected['pixel_xy'], float)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError('Reference coordinates are not two finite original-image pixels')
        groups[camera, observation_id].append(role)
    observations = {r['observation_id']: {'observation_id': r['observation_id'],
                                         'control_step': r['control_step']} for r in catalog}
    camera_data = {}
    timings = []
    role_references = {}
    for (camera, observation_id), group_roles in groups.items():
        if camera not in camera_data:
            if use_saved_tracks:
                frames = json.loads((output/(camera+'-tracking-inputs.json')).read_text())
                checked_videos = set()
                for frame in frames:
                    if 'path' in frame and hashlib.sha256(Path(frame['path']).read_bytes()).hexdigest() != frame['sha256']:
                        raise ValueError('Saved tracker input changed')
                    if frame.get('source_video') and frame['source_video'] not in checked_videos:
                        if hashlib.sha256(Path(frame['source_video']).read_bytes()).hexdigest() != frame['source_video_sha256']:
                            raise ValueError('Recorded source video changed')
                        checked_videos.add(frame['source_video'])
                camera_data[camera] = (None, frames)
            elif all_recorded_rgb and camera=='agentview':
                camera_data[camera] = dense_camera_sequence(run,catalog,output)
            else:
                camera_data[camera] = camera_sequence(run, catalog, camera, output)
        views, frames = camera_data[camera]
        reference_index = next(i for i, r in enumerate(frames) if r['observation_id'] == observation_id)
        queries = np.asarray([references[role]['pixel_xy'] for role in group_roles], np.float32)
        track_path = output/f'{camera}-ref{observation_id:03d}-tracks.npz'
        if use_saved_tracks:
            with np.load(track_path, allow_pickle=False) as arrays:
                tracks, visible = arrays['tracks'], arrays['visible']
            earlier = [t for t in previous.get('tracking', []) if t['camera']==camera
                       and t['reference_observation_id']==observation_id]
            timing = {k:v for k,v in earlier[0].items() if k in ['model_load','inference']} if earlier else {
                'timing_unavailable': 'The prior process failed while reading depth before persisting this group timing.'}
        else:
            tracks, visible, timing = tracker.predict(views, queries, reference_index=reference_index)
            np.savez_compressed(track_path, tracks=tracks, visible=visible)
            write(output/f'{camera}-ref{observation_id:03d}-tracking-time.json', timing)
        timings.append({'camera': camera, 'reference_observation_id': observation_id, 'roles': group_roles,
                        'rgb_frames': len(frames), 'reference_frame_index': reference_index, **timing})
        reference_depth = load_depth(by_id[observation_id]['folder'], camera)
        for role in group_roles:
            role_references[role] = landmark_sample(reference_depth, references[role]['pixel_xy'])
        for index, frame in enumerate(frames):
            identifier = frame['observation_id']
            if identifier is None:continue
            view = load_depth(by_id[identifier]['folder'], camera)
            for j, role in enumerate(group_roles):
                sample = landmark_sample(view, tracks[index, j], bool(visible[index, j]))
                sample.update(camera=camera, reference_observation_id=observation_id)
                observations[identifier][role] = sample
    rows = []
    for capture in catalog:
        row = observations[capture['observation_id']]
        row.update(measured_gap(row['moving'], row['fixed_closed']))
        for role in roles:
            current, initial = row[role], role_references[role]
            current['world_displacement_from_reference_m'] = (
                float(np.linalg.norm(np.asarray(current['raw_world_m'])-initial['raw_world_m']))
                if current['supported'] and initial['supported'] else None)
        rows.append(row)
    report = {'status': 'measured_hypothesis', 'references': references,
        'selection_confident': references.get('can_select'), 'reference_samples': role_references,
        'samples': rows, 'supported_capture_count': sum(r['supported'] for r in rows),
        'capture_count': len(rows), 'recorded_depth_capture_count':sum((r['folder']/'sensors.npz').exists() for r in catalog),
        'tracker': 'CoTracker3 scaled_offline', 'tracking': timings,
        'all_recorded_external_rgb': all_recorded_rgb,
        'used_existing_tracker_outputs': use_saved_tracks,
        'original_tracking_and_measurement_seconds': previous.get('total_seconds') if use_saved_tracks else None,
        'total_seconds': time.perf_counter()-started, 'robot_motion_executed': False,
        'closing_direction_hardcoded': False, 'native_state_used_for_measurement': False,
        'scope': 'Offline tracks of actor-selected material-point hypotheses, lifted through measured depth. '
                 'No held-position substitution, inferred plane, point snapping, alternate tracker or depth interpolation. '
                 'Semantic correspondence to the closed configuration is not guaranteed by a small gap.',
        'fixed_reference_displacement_scope': 'World displacement may reflect appliance-base motion, a wrong semantic selection, or tracking error; no automatic re-anchoring.'}
    write(output/'gap.json', report)
    print(json.dumps({'state': state, 'supported': report['supported_capture_count'],
                      'captures': len(rows), 'tracking_seconds': report['total_seconds']}), flush=True)


def main() -> None:
    """Measure only after both independent agents have frozen their references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--use-saved-tracks', action='store_true',
                        help='Explicitly remeasure already saved tracks; fail if outputs or hashes are missing')
    parser.add_argument('--all-recorded-rgb',action='store_true',
                        help='Separate dense condition with the same frozen landmarks and tracker')
    args = parser.parse_args()
    if not (args.study/'selection-complete.json').exists():
        raise RuntimeError('Reference selection must complete before measuring either case')
    if not args.use_saved_tracks:
        from supervise_luna_dashboard_study import wait_for_capacity
        wait_for_capacity(args.study/('dense-tracking-resources' if args.all_recorded_rgb else 'tracking-resources'),
                          'recorded-goal-gap-tracking')
    tracker = None if args.use_saved_tracks else CoTrackerBackend(device='cuda')
    try:
        for state in [17, 49]:measure_case(args.study, state, tracker, use_saved_tracks=args.use_saved_tracks,
                                          all_recorded_rgb=args.all_recorded_rgb)
    finally:
        if tracker is not None:tracker.close()
    write(args.study/('dense-measurement-complete.json' if args.all_recorded_rgb else 'measurement-complete.json'), {'complete': True})


if __name__ == '__main__':main()
