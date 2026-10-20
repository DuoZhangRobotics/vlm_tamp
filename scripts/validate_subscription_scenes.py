"""Sensor/scene smoke checks on an already-used development state; no policy trial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.profiled_sim_client import ProfiledClient
from vlm_tamp.subscription_scene_variants import VARIANTS
from vlm_tamp.rgbd_numericals import world_points

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=(ROOT/a.output).resolve();out.mkdir(exist_ok=False);records=[];base=None
    for name in VARIANTS:
        folder=out/name;folder.mkdir();sim=ProfiledClient(folder/'simulator.log')
        try:
            state=sim.request('reset',task_name='KITCHEN_SCENE7_open_the_microwave',state_index=8,
                output_dir=str(folder/'initial'),seed=0,suite_name='libero_90',scene_variant=name)
            state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
            assert not state['evaluation']['task_success']
            meta=state['metadata'];path=folder/'opened/obs.npz'
            with np.load(path) as arr:
                errors={}
                for cam in ['agentview','frontview','wrist']:
                    cal=meta['cameras'][cam];actual=arr[cam+'_world_points'];depth=arr[cam+'_depth']
                    restored=world_points(depth,cal['intrinsic'],cal['camera_to_world'])
                    errors[cam]=float(np.nanmax(np.abs(actual-restored)))
            assert all(x<2e-6 for x in errors.values())
            hashes={k:hashlib.sha256((folder/'opened'/f'{k}.png').read_bytes()).hexdigest() for k in ['agentview','frontview','wrist']}
            if name=='nominal':
                prior=json.loads((ROOT/'results/subscription-dino-study-20261019/runs/state008-no_dino/study-case-result.json').read_text())['input_images']
                assert hashes==prior,'Nominal observation differs from frozen baseline'
                base=hashes
            else:assert hashes['frontview']!=base['frontview'],'Scene variant has no visual effect'
            records.append({'variant':name,'state':8,'native_initial_success':False,'rgb_hashes':hashes,
                'calibrated_world_max_errors_m':errors,'nominal_matches_previous_rgb':hashes==base if name=='nominal' else None,
                'robot_joints':meta['joint_positions'],'settling_actions':5,'open_gripper_control_steps':60})
        finally:
            sim.close()
            reclaimed=0
            for f in folder.rglob('obs.npz'):reclaimed+=f.stat().st_size;f.unlink()
            # Video duplicates are unnecessary for a no-task-motion setup check.
            for f in folder.rglob('*.mp4'):reclaimed+=f.stat().st_size;f.unlink()
            for frame_dir in folder.rglob('frames'):
                import shutil
                reclaimed+=sum(f.stat().st_size for f in frame_dir.rglob('*') if f.is_file());shutil.rmtree(frame_dir)
            (folder/'cleanup.json').write_text(json.dumps({'bytes_reclaimed':reclaimed,'retained':'RGB, calibration, scene initialization record and logs'},indent=2))
        (out/'summary.json').write_text(json.dumps({'records':records,'state_used_only_for_infrastructure_validation':8,'policy_runs':0},indent=2))
        print(json.dumps(records[-1]),flush=True)
    sheet=Image.new('RGB',(1536,1072),'white');draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(VARIANTS):
        x=(i%3)*512;y=(i//3)*536;draw.text((x+8,y+7),name,fill='black')
        sheet.paste(Image.open(out/name/'opened/frontview.png'),(x,y+24))
    sheet.save(out/'scene-sheet.png')
if __name__=='__main__':main()
