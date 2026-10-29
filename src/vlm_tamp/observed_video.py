"""Bounded, timestamped RGB clips from the simulator's ordinary camera log.

This module reads sensor images only. It never reads object state, interpolates
depth, chooses a task strategy or changes robot controls.
"""
import hashlib
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from PIL import Image


def sampled_steps(available,anchors,maximum=64):
    """Spread samples over control time while retaining every depth anchor."""
    available=sorted(set(int(v) for v in available));anchors=sorted(set(int(v) for v in anchors))
    if not anchors or anchors[0]==anchors[-1]:raise ValueError('Choose captures at two or more distinct control steps')
    if any(v not in available for v in anchors):raise ValueError('A depth capture has no aligned recorded agentview RGB frame')
    available=[v for v in available if anchors[0]<=v<=anchors[-1]]
    if type(maximum) is not int or not 2<=maximum<=64:raise ValueError('Use a video frame budget from 2 to 64')
    if len(anchors)>maximum:raise ValueError('Video frame budget must retain every supplied depth timestamp')
    if len(available)<=maximum:return available
    chosen=set(anchors);values=np.asarray(available)
    distance=np.min(np.abs(values[:,None]-np.asarray(anchors)[None,:]),axis=1)
    while len(chosen)<maximum:
        index=int(np.argmax(distance));step=available[index];chosen.add(step)
        distance=np.minimum(distance,np.abs(values-step))
    return sorted(chosen)


def recorded_clip(root,anchors,maximum,folder,workspace):
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('Live agentview RGB recording is unavailable')
    paths={}
    for path in root.glob('step*.jpg'):
        match=re.fullmatch(r'step(\d+)\.jpg',path.name)
        if match and not path.is_symlink():paths[int(match.group(1))]=path
    selected=sampled_steps(paths,anchors,maximum)
    folder=Path(folder);folder.mkdir(exist_ok=False);workspace=Path(workspace)
    views=[];frames=[]
    for step in selected:
        payload=paths[step].read_bytes();sha=hashlib.sha256(payload).hexdigest()
        destination=folder/paths[step].name;destination.write_bytes(payload)
        with Image.open(BytesIO(payload)) as image:rgb=np.asarray(image.convert('RGB')).copy()
        views.append({'rgb':rgb})
        frames.append({'control_step':step,'image':str(destination.relative_to(workspace)),'sha256':sha})
    return views,frames,{'available_rgb_frames':sum(anchors[0]<=s<=anchors[-1] for s in paths),
        'selected_rgb_frames':len(selected),'maximum_rgb_frames':maximum,
        'sampling':'All available frames when within budget; otherwise temporal spacing with every depth anchor retained.',
        'rgb_source':'Recorded agentview JPEGs; no new simulator render or intermediate depth.'}
