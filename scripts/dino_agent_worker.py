"""One optional local DINOv3 query in the existing isolated GPU environment."""
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
from vlm_tamp.dinov3_features import Dinov3Features,MODEL_REVISION,MODEL_ID
from vlm_tamp.dino_agent_tools import propose_landmarks,match_landmarks


def view(path,camera):
    with np.load(path,allow_pickle=False) as data:
        result={key:data[camera+'_'+key].copy() for key in ['rgb','depth','world_points']}
        result['robot']=data[camera+'_known_robot_depth_prior'].copy() if camera+'_known_robot_depth_prior' in data else None
    return result


def overlay(rgb,points,path):
    image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
    for label,point in points:
        x,y=map(int,point);draw.ellipse((x-4,y-4,x+4,y+4),fill='#ffdf00',outline='black',width=1)
        draw.text((x+6,y-8),str(label),fill='white',stroke_width=2,stroke_fill='black')
    image.save(path)


def main():
    request=json.load(sys.stdin);start=time.perf_counter()
    snapshot=ROOT/'.runtime/hf-cache'/('models--'+MODEL_ID.replace('/','--'))/'snapshots'/MODEL_REVISION
    with contextlib.redirect_stdout(sys.stderr):
        engine=Dinov3Features(checkpoint_dir=snapshot)
    loaded=time.perf_counter();current=view(request['current_sensors'],request['camera'])
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=True)
    first=engine.extract(current['rgb']);extract_seconds=time.perf_counter()-loaded
    if request['operation']=='landmarks':
        result=propose_landmarks(first['features'],current['world_points'],current['depth'],
            box=request.get('region_xyxy'),count=request.get('count',8),robot_depth=current['robot'])
        overlay(current['rgb'],[(p['id'],p['pixel_xy']) for p in result['candidates']],output/'landmarks.png')
        images=['landmarks.png']
    elif request['operation']=='match':
        reference=view(request['reference_sensors'],request['reference_camera'])
        before=time.perf_counter();second=engine.extract(reference['rgb']);extract_seconds+=time.perf_counter()-before
        result=match_landmarks(second['features'],first['features'],reference['world_points'],current['world_points'],
            reference['depth'],current['depth'],request['pixels_xy'],box=request.get('region_xyxy'),
            reference_robot=reference['robot'],current_robot=current['robot'])
        overlay(reference['rgb'],[(r['id'],r['reference_pixel_xy']) for r in result['matches']],output/'reference.png')
        overlay(current['rgb'],[(r['id'],r['candidates'][0]['pixel_xy']) for r in result['matches'] if r['candidates']],output/'matches.png')
        images=['reference.png','matches.png']
    else:raise ValueError('Unknown DINO operation')
    import torch
    result.update(model=MODEL_ID,revision=MODEL_REVISION,weights_sha256=engine.metadata['weights_sha256'],
        rgb_only_encoder=True,depth_used_only_for_output_geometry=True,
        load_seconds=loaded-start,feature_seconds=extract_seconds,total_seconds=time.perf_counter()-start,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,images=images)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
