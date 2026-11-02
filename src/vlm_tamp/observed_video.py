"""Bounded, timestamped RGB clips from the simulator's ordinary camera log.

This module reads sensor images only. It never reads object state, interpolates
depth, chooses a task strategy or changes robot controls.
"""
import hashlib
import json
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


def inspect_recording(episode: Path, folder: Path, workspace: Path, start: int, end: int,
                      maximum: int = 8, region: list[int] | None = None) -> dict:
    """Retrieve a bounded chronological RGB strip from the existing recording."""
    import shutil
    import subprocess
    from PIL import ImageDraw
    from .control_video import CONTROL_HZ
    if not 0 <= start <= end or not 2 <= maximum <= 12:
        raise ValueError('Use an ordered nonnegative control interval and 2..12 frames')
    initial=Path(episode)/'initial'
    paths={int(p.stem[4:]):p for p in (initial/'frames').glob('step*.jpg')
        if re.fullmatch(r'step\d+\.jpg',p.name) and not p.is_symlink()}
    live=bool(paths)
    clock=None
    if live:
        available=sorted(paths)
    elif (initial/'video-frames.json').exists() and (initial/'rollout.mp4').exists():
        clock=json.loads((initial/'video-frames.json').read_text())
        available=sorted(set(clock['source_control_steps']))
    else:
        return {'available':False,'reason':'no_recorded_rgb','robot_motion_issued':False,'new_sensor_capture':False}
    interval=[step for step in available if start<=step<=end]
    if not interval:
        return {'available':False,'reason':'no_recorded_frame_in_requested_interval',
            'available_control_interval':[available[0],available[-1]],'new_sensor_capture':False,
            'robot_motion_issued':False}
    selected=interval if len(interval)<=maximum else sampled_steps(interval,[interval[0],interval[-1]],maximum)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);workspace=Path(workspace)
    if not live:
        encoder=shutil.which('ffmpeg')
        if not encoder:raise RuntimeError('Existing ffmpeg is unavailable for recorded video inspection')
        indices={step:i for i,step in enumerate(clock['presentation_control_steps'])}
        selection='+'.join(f'eq(n\\,{indices[step]})' for step in selected)
        completed=subprocess.run([encoder,'-nostdin','-loglevel','error','-threads','1',
            '-i',str(initial/'rollout.mp4'),'-vf','select='+selection,'-vsync','0',
            '-frames:v',str(len(selected)),str(folder/'decoded-%02d.png')],capture_output=True,text=True,timeout=30)
        if completed.returncode:raise RuntimeError('Recorded video decode failed: '+completed.stderr[-1000:])
        paths={step:folder/f'decoded-{i+1:02d}.png' for i,step in enumerate(selected)}
    images=[];frames=[]
    for step in selected:
        with Image.open(paths[step]) as opened:
            image=opened.convert('RGB');width,height=image.size
        box=[0,0,width,height] if region is None else region
        if (len(box)!=4 or any(type(v) is not int for v in box)
                or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height):
            raise ValueError('Use a crop within the recorded original image')
        image=image.crop(box);destination=folder/f'step{step:06d}.png';image.save(destination)
        frames.append({'control_step':step,'simulated_seconds':step/CONTROL_HZ,
            'image':str(destination.relative_to(workspace)),'original_size':[width,height],
            'region_xyxy':box,'source_sha256':hashlib.sha256(paths[step].read_bytes()).hexdigest()})
        images.append(image)
    columns=min(4,len(images));tile=384;header=32
    canvas=Image.new('RGB',(columns*tile,((len(images)+columns-1)//columns)*(tile+header)),(244,247,250))
    draw=ImageDraw.Draw(canvas)
    for i,(image,frame) in enumerate(zip(images,frames)):
        image.thumbnail((tile,tile),Image.Resampling.LANCZOS)
        x=(i%columns)*tile;y=(i//columns)*(tile+header)
        draw.text((x+6,y+8),f"agentview | control {frame['control_step']} | {frame['simulated_seconds']:.2f} s",fill=(20,35,50))
        canvas.paste(image,(x,y+header))
    canvas.save(folder/'sequence.png')
    result={'available':True,'camera':'agentview','requested_control_interval':[start,end],
        'available_control_interval':[available[0],available[-1]],'recorded_frames_in_interval':len(interval),
        'frames':frames,'selected_frame_count':len(frames),'comparison_image':str((folder/'sequence.png').relative_to(workspace)),
        'source':'live recorded JPEGs' if live else 'completed recorded MP4',
        'new_sensor_capture':False,'robot_motion_issued':False,'depth_available':False,
        'interpretation':'Chronological recorded RGB only. Compare moving and fixed parts; no task direction is supplied. Use original current RGB-D observations for geometry pixels.'}
    (folder/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
