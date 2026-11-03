"""Luna-authored visual requests grounded in the two recorded RGB cameras."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from .part_perception import checked_folder, overlay

CAMERAS = ('agentview', 'wrist')


def point(tools: Any, event_id: int, observation_id: int, camera: str, query: str) -> dict[str, Any]:
    """Read RGB only; return points without changing a scene, mask or robot state."""
    started = time.perf_counter()
    record = tools._record(observation_id)
    if camera not in CAMERAS or any(name not in record['cameras'] for name in CAMERAS):
        raise ValueError('point uses the recorded agentview and wrist cameras; select one of them')
    folder = checked_folder(tools.output, f'{event_id:03d}-point')
    source = Path(record['sensors']).parent
    paths = [source / f'{name}.png' for name in CAMERAS]
    images = [np.asarray(Image.open(path).convert('RGB')) for path in paths]
    target = CAMERAS.index(camera) + 1
    prompt = (
        f'Point to: {query}\n'
        f'Image 1 is agentview. Image 2 is wrist. Only point in image {target}.')
    prepared = time.perf_counter()
    model = tools.pointer.point(paths, prompt)
    inferred = time.perf_counter()
    points = []
    for item in model['points']:
        name = CAMERAS[item['image_index'] - 1]
        x, y = map(int, np.rint(item['pixel_xy']))
        height, width = images[item['image_index'] - 1].shape[:2]
        in_image = 0 <= x < width and 0 <= y < height
        points.append({**item, 'camera': name, 'observation_id': observation_id,
                       'contact_pixel_xy': [x, y] if in_image else None,
                       'matches_requested_camera': name == camera,
                       'pixel_conversion': 'nearest integer pixel; raw floating point retained',
                       'pixel_status': 'in_image' if in_image else 'outside_pixel_center_grid'})
    pictures = []
    for name, rgb in zip(CAMERAS, images):
        selected = [p['pixel_xy'] for p in points if p['camera'] == name]
        if selected:
            picture = folder / f'{name}-points.png'
            overlay(picture, {'rgb': rgb}, np.zeros(rgb.shape[:2], bool), points=selected, point_radius=7)
            pictures.append(tools._relative(picture))
    request_file = folder / 'model-request.json'
    request_file.write_text(json.dumps({
        'query': query, 'prompt': prompt, 'observation_id': observation_id, 'camera': camera,
        'images': [{'image_index': i + 1, 'camera': name,
                    'path': tools._relative(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for i, (name, path) in enumerate(zip(CAMERAS, paths))],
        'simulator_object_geometry_supplied': False,
    }, indent=2) + '\n')
    model_file = folder / 'model-output.json'
    model_file.write_text(json.dumps(model, indent=2) + '\n')
    response = {
        'pointing_backend': model['backend'], 'status': model['status'],
        'query': query, 'observation_id': observation_id, 'camera': camera, 'points': points,
        'images': pictures, 'raw_text': model['raw_text'],
        'request_file': tools._relative(request_file), 'model_output_file': tools._relative(model_file),
        'worker_pid': model['worker_pid'], 'worker_request_sequence': model['worker_request_sequence'],
        'timing_seconds': {'rgb_read': prepared - started, 'model_rpc': inferred - prepared,
                           'preprocessing': model['preprocessing_seconds'],
                           'generation': model['generation_seconds'],
                           'overlay_and_records': time.perf_counter() - inferred,
                           'total': time.perf_counter() - started},
        'robot_motion_executed': False, 'planner_model_updated': False,
        'scope': 'Image-space proposals from your query. No depth, masks, grasp pose, hinge or '
                 'push direction is inferred by this tool. Window marks are not excluded.',
    }
    (folder / 'result.json').write_text(json.dumps(response, indent=2) + '\n')
    return response
