"""Encode recorded robot views on the simulator control clock."""
from bisect import bisect_right
from pathlib import Path
import json
import shutil
import subprocess

from PIL import Image


CONTROL_HZ = 20


def frame_timeline(paths, control_hz=CONTROL_HZ):
    paths = sorted(map(Path, paths), key=lambda p: int(p.stem[4:]))
    source_steps = [int(p.stem[4:]) for p in paths]
    if not source_steps:
        return paths, [], {}
    ticks = list(range(source_steps[0], source_steps[-1]+1))
    indices = [bisect_right(source_steps, tick)-1 for tick in ticks]
    return paths, indices, {
        'schema_version': 2, 'movie': 'rollout.mp4',
        'control_steps': [source_steps[i] for i in indices],
        'presentation_control_steps': ticks, 'source_control_steps': source_steps,
        'frame_count_verified': len(ticks), 'frames_per_second': control_hz,
        'control_hz': control_hz, 'timebase': 'simulator_control_steps',
        'missing_capture_policy': 'hold_previous_recorded_image',
        'simulated_elapsed_seconds': (source_steps[-1]-source_steps[0])/control_hz,
        'encoded_duration_seconds': len(ticks)/control_hz,
        'initial_frame_included': True,
    }


def encode_control_video(paths, movie, *, encoder=None, control_hz=CONTROL_HZ):
    """Hold sparse captures for their actual duration; never invent observations."""
    paths, indices, record = frame_timeline(paths, control_hz)
    if not paths:
        return None
    movie = Path(movie)
    encoder = encoder or shutil.which('ffmpeg')
    if not encoder:
        raise RuntimeError('Existing ffmpeg encoder unavailable')
    with Image.open(paths[0]) as first:
        width, height = first.size
    with (movie.parent/'video_encoder.log').open('w') as log:
        process = subprocess.Popen([encoder, '-nostdin', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
            '-framerate', str(control_hz), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)],
            stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            previous, pixels = None, None
            for index in indices:
                if index != previous:
                    with Image.open(paths[index]) as image:
                        pixels = image.convert('RGB').tobytes()
                    previous = index
                process.stdin.write(pixels)
            process.stdin.close()
            if process.wait(timeout=60):
                raise RuntimeError('Video encoder failed; see video_encoder.log')
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    record['movie'] = movie.name
    (movie.parent/'video-frames.json').write_text(json.dumps(record, indent=2)+'\n')
    return record
