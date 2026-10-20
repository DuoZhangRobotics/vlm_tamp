"""Render short simulation-time previews from verified retained movies."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.coupled_storage import read_json,write_json,video_frames


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',default='results/mcvamp-drawer-stove-report/previews');args=p.parse_args()
    out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=True);selected={}
    for root in args.runs:
        for path in sorted((ROOT/root).glob('*-state*/summary.json')):
            row=read_json(path)
            if row['official_task_success'] and row['task'] not in selected:selected[row['task']]=(path.parent,row)
    rows=[]
    for label,(folder,row) in selected.items():
        target=out/(folder.name+'.mp4');steps=[]
        if not target.exists():
            encoder=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','512x512','-framerate','20','-i','pipe:0','-an','-c:v','libx264','-threads','2',
                '-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
            try:
                for step,image in video_frames(folder):
                    if step<60:continue
                    encoder.stdin.write(image.tobytes());steps.append(step)
                encoder.stdin.close();assert encoder.wait(timeout=60)==0
                assert steps==list(range(60,row['control_steps']+1))
            finally:
                if encoder.poll() is None:encoder.kill();encoder.wait()
        rows.append({'task':label,'state':row['initial_state'],'native_success':True,'frames_per_second':20,
            'control_steps':[60,row['control_steps']],'every_joint_frame_preserved':True,
            'planning_pauses_omitted':True,'original':str(folder/'initial/rollout.mp4'),'preview':str(target)})
    write_json(out/'provenance.json',{'previews':rows,'time_is_simulation_time_not_pipeline_wall_time':True})
    print({'previews':list(selected)})

if __name__=='__main__':main()
