"""Offline sensor-only diagnostics of the remaining regression failures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from PIL import Image,ImageDraw
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.cohort_execution import read_cohort_observation
from vlm_tamp.rekep_libero.mechanism_geometry import project
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.mcvamp_mechanism import calibrated_frames
import vamp
out=ROOT/'results/reliability-diagnosis-20261016';out.mkdir(exist_ok=True)
rows=[]
for task,state in [('close_top_drawer',11),('close_top_drawer',41),('turn_on_stove',21),('turn_on_stove',31)]:
    run='mcvamp-mechanisms-camera-retry-20261016' if task=='close_top_drawer' and state==41 else 'mcvamp-mechanisms-frozen-20261015-b'
    folder=ROOT/'results'/run/f'{task}-state{state:03d}'
    spec=read_json(folder/'task-spec.json');meta=read_json(folder/'opened/metadata.json')
    cameras,_=read_cohort_observation(folder/'opened/obs.npz',meta);cam=cameras[spec['camera']]
    pic=Image.fromarray(cam['rgb']);draw=ImageDraw.Draw(pic)
    cloud=np.load(folder/'grasp-cloud.npy');record={'task':task,'state':state}
    def dots(points,color):
        for x,y in project(points,cam):
            if np.isfinite([x,y]).all():draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
    dots(cloud[::max(1,len(cloud)//500)],'#1bca7b')
    if task=='close_top_drawer':
        probe=next(folder.glob('candidate-*/probe/native-geometry.npz'))
        with np.load(probe) as a:arrays={k:a[k] for k in a.files}
        frame,_,_=calibrated_frames(read_json(folder/'robot-chain.json'))
        cfg=arrays['configuration'];q=cfg[44:51]
        eef=np.asarray(vamp.panda.eefk(q),float)[0];eef[:3,3]+=.025*cfg[3:6]
        local=transform_points(arrays['stationary'],np.linalg.inv(eef));hit=np.zeros(len(local),bool)
        for i,b in enumerate(arrays['gripper_bounds']):
            if not b[4]:continue
            lo,hi=arrays['gripper_plane_offsets'][i:i+2];plane=arrays['gripper_planes'][lo:hi]
            hit|=(np.max(local@plane[:,:3].T+plane[:,3],axis=1)<=0)
        points=transform_points(arrays['stationary'][hit],frame);dots(points,'#ff2233')
        normal=np.asarray(spec['panel_normal_world']);origin=np.asarray(spec['front_plane']['point_world_m'])
        record.update(collided_stationary_points=len(points),plane_distance_m=((points-origin)@normal).tolist(),collision_bounds_world_m=[points.min(0).tolist(),points.max(0).tolist()] if len(points) else None)
        np.save(out/f'{task}-{state}-collision-points.npy',points)
    else:
        import json
        choices=json.loads((folder/'grounding/response.txt').read_text())
        draw.rectangle(choices['moving_panel_box_xyxy'],outline='#ff9933',width=2)
        x,y=choices['grasp_pixel_xy'];draw.ellipse((x-4,y-4,x+4,y+4),outline='red',width=2)
        x,y=project([spec['joint_origin_world_m']],cam)[0];draw.line((x-5,y,x+5,y),fill='cyan',width=2);draw.line((x,y-5,x,y+5),fill='cyan',width=2)
        record.update(choices=choices,old_pivot=spec['joint_origin_world_m'])
    pic.save(out/f'{task}-{state}.png');rows.append(record)
write_json(out/'audit.json',rows)
print(rows)
