"""Short simulation-time previews; original diagnostic movies remain intact."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.coupled_storage import read_json,write_json,video_frames


def main():
    source=ROOT/'results/all-prior-tasks-frozen-v3-20261015'
    output=ROOT/'results/all-prior-tasks-report/previews';output.mkdir(exist_ok=True)
    rows=[]
    for case in ['bowl_on_plate-state001','open_middle_drawer-state001','open_microwave-state011']:
        folder=source/case;summary=read_json(folder/'summary.json');assert summary['official_task_success']
        movie=output/(case+'.mp4')
        expected=list(range(60,summary['control_steps']+1))
        if not movie.exists():
            encoder=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','512x512','-framerate','20','-i','pipe:0','-an','-c:v','libx264','-threads','2',
                '-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],stdin=subprocess.PIPE)
            steps=[]
            try:
                for step,image in video_frames(folder):
                    if step<60:continue
                    encoder.stdin.write(image.tobytes());steps.append(step)
                encoder.stdin.close();assert encoder.wait(timeout=60)==0
                assert steps==expected,'Preview requires every joint-control frame'
            finally:
                if encoder.poll() is None:encoder.kill();encoder.wait()
        rows.append({'case':case,'native_success':True,'control_steps':[60,summary['control_steps']],
            'frames_per_second':20,'every_joint_control_frame_preserved':True,
            'planning_pauses_omitted':True,'original_diagnostic_movie_preserved':str(folder/'initial/rollout.mp4'),
            'preview':str(movie)})
    write_json(output/'provenance.json',{'previews':rows,'video_time_is_simulation_time_not_end_to_end_wall_time':True})
    print({'previews':[r['case'] for r in rows]})


if __name__=='__main__':main()
