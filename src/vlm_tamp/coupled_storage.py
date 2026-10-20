"""Bounded episode storage, preserving outcomes and lossless planning evidence."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
POLICY = {'version': 1, 'raw_observations': 'opened and latest',
          'videos': 'verified movie plus exact frame/control-step index',
          'planner_inputs': 'temporary pickle removed after worker exits',
          'collision_geometry': 'lossless arrays, deduplicated within each episode',
          'large_diagnostics': 'lossless gzip; outcomes/actions/constraints stay readable',
          'automatic_post_episode_cleanup': True}
_SCENES = weakref.WeakKeyDictionary()


def read_json(path):
    path = Path(path)
    if not path.exists() and path.suffix == '.json':
        path = path.with_suffix('.json.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.new')
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()) + '\n')
    os.replace(temporary, path)


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def preserve_scene(scene, output):
    """Save portable numeric evidence once per scene, rather than repeated pickles."""
    output = Path(output)
    cached = _SCENES.get(scene)
    if cached is None:
        episode = next((p for p in [output, *output.parents] if (p/'robot-chain.json').exists()), output.parent)
        arrays = {key: np.asarray(getattr(scene, key)) for key in
                  ['static_points', 'moving_reference', 'contact_reference', 'inferred_panel_reference']}
        digest = hashlib.sha256()
        for name, data in arrays.items():
            digest.update(name.encode())
            digest.update(str((data.shape, data.dtype.str)).encode())
            digest.update(data.tobytes())
        token = digest.hexdigest()
        folder = episode/'collision-scenes'
        folder.mkdir(exist_ok=True)
        target = folder/(token+'.npz')
        if not target.exists():
            np.savez_compressed(target, **arrays)
        cached = {'arrays_path': str(target.resolve()), 'arrays_sha256': _digest(target),
                  'scene_record': scene.record(), 'spec': scene.spec, 'configuration': scene.config,
                  'configured_robot_reference_volumes': scene.robot.reference_volumes if scene.robot is not None else [],
                  'configured_robot_reference_joints': getattr(scene.robot, 'reference_joints', None) if scene.robot is not None else None,
                  'lossless_numeric_arrays': True, 'simulator_scene_geometry_used': False}
        _SCENES[scene] = cached
    write_json(output/'scene-reference.json', cached)


def _movie_index(folder):
    """Only discard JPEGs once the corresponding movie is readable and complete."""
    import cv2
    frames = sorted((folder/'initial/frames').glob('step*.jpg'))
    index_path = folder/'initial/video-frames.json'
    if not frames:
        return []
    movie = folder/'initial/rollout.mp4'
    def complete(path):
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened() or round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frames):
                return False
            first, _ = capture.read()
            capture.set(cv2.CAP_PROP_POS_FRAMES, len(frames)-1)
            last, _ = capture.read()
            return first and last
        finally:
            capture.release()
    if not complete(movie):
        temporary = movie.with_name('rollout-retention.tmp.mp4')
        try:
            subprocess.run(['ffmpeg', '-nostdin', '-y', '-loglevel', 'error', '-framerate', '4',
                '-pattern_type', 'glob', '-i', str(folder/'initial/frames/step*.jpg'),
                '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(temporary)], check=True, timeout=120,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if not complete(temporary):
                raise RuntimeError('Movie verification failed; raw frames were retained')
            os.replace(temporary, movie)
        finally:
            temporary.unlink(missing_ok=True)
    write_json(index_path, {'movie': 'rollout.mp4', 'control_steps': [int(p.stem[4:]) for p in frames],
                            'frame_count_verified': len(frames), 'frames_per_second': 4})
    return frames


def video_frames(folder):
    """Yield (control step, PIL image) from raw frames or the compact movie."""
    from PIL import Image
    folder = Path(folder)
    frames = sorted((folder/'initial/frames').glob('step*.jpg'))
    if frames:
        for frame in frames:
            with Image.open(frame) as image:
                yield int(frame.stem[4:]), image.convert('RGB').copy()
        return
    index = folder/'initial/video-frames.json'
    if not index.exists():
        return
    import cv2
    record = read_json(index)
    capture = cv2.VideoCapture(str(folder/'initial'/record['movie']))
    try:
        for step in record['control_steps']:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Retained movie misses a recorded control frame')
            yield step, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def compact_episode(folder, *, allowed_root=RESULTS):
    """Call only after all episode workers have exited and summary is saved."""
    folder, allowed_root = Path(folder).resolve(), Path(allowed_root).resolve()
    if folder == allowed_root or not folder.is_relative_to(allowed_root):
        raise ValueError('Cleanup requires an episode beneath the results root')
    if not (folder/'summary.json').exists():
        raise ValueError('An unfinished episode cannot be pruned')
    previous = read_json(folder/'retention.json') if (folder/'retention.json').exists() else None
    snapshots = []
    for path in folder.glob('*/obs.npz'):
        if path.is_symlink() or path.parent.is_symlink():
            continue
        metadata = read_json(path.parent/'metadata.json')
        snapshots.append((metadata['control_steps'], path.parent.name == 'final', path.parent))
    keep = {folder/'opened'}
    if snapshots:
        keep.add(max(snapshots, key=lambda row: (row[0], row[1], row[2].name))[2])
    # Verify video before deleting anything; a failed encoder leaves evidence intact.
    frames = set(_movie_index(folder))
    removed, archived = [], []
    before = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    readable = {'summary.json', 'protocol.json', 'manifest.json', 'actions.json',
                'evaluation_contact_diagnostics.json', 'tracking.json', 'observations.json',
                'replans.json', 'grasp-search.json', 'task-spec.json'}
    for base, directories, names in os.walk(folder, followlinks=False):
        directories[:] = [d for d in directories if not (Path(base)/d).is_symlink()]
        for name in names:
            path = Path(base)/name
            if path.is_symlink() or not path.is_file():
                continue
            delete = path in frames or name == 'trusted-local-input.pkl'
            if path.parent not in keep:
                delete |= name in {'obs.npz', 'robot_visual_meshes.npz'}
                delete |= path.suffix == '.png' and name.startswith('arm_')
            if delete:
                removed.append({'path': str(path.relative_to(folder)), 'bytes': path.stat().st_size})
                path.unlink()
            elif (path.suffix == '.json' and path.stat().st_size > 1024*1024
                  and name not in readable and path.parent not in keep
                  and path != folder/'initial/metadata.json'):
                destination = path.with_suffix('.json.gz')
                temporary = destination.with_suffix('.gz.tmp')
                digest = _digest(path)
                if destination.exists():
                    raise ValueError('Refusing to overwrite an existing diagnostic archive')
                try:
                    with path.open('rb') as src, gzip.open(temporary, 'wb', compresslevel=1) as dest:
                        shutil.copyfileobj(src, dest)
                    value = hashlib.sha256()
                    with gzip.open(temporary, 'rb') as stream:
                        for chunk in iter(lambda: stream.read(1024*1024), b''):
                            value.update(chunk)
                    if value.hexdigest() != digest:
                        raise RuntimeError('Archive verification failed; original retained')
                    archived.append({'path': str(path.relative_to(folder)), 'sha256': digest,
                                     'before_bytes': path.stat().st_size, 'after_bytes': temporary.stat().st_size})
                    os.replace(temporary, destination)
                    path.unlink()
                finally:
                    temporary.unlink(missing_ok=True)
    after = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    record = {'policy': POLICY, 'raw_observations_retained': sorted(p.name for p in keep if (p/'obs.npz').exists()),
              'bytes_before_cleanup': before, 'bytes_after_cleanup': after,
              'bytes_reclaimed_this_pass': before-after, 'removed': removed, 'archived': archived}
    if previous:
        record['prior_bytes_reclaimed'] = previous.get('prior_bytes_reclaimed', 0)+previous['bytes_reclaimed_this_pass']
    write_json(folder/'retention.json', record)
    return {k: v for k, v in record.items() if k not in {'removed', 'archived', 'policy'}}
