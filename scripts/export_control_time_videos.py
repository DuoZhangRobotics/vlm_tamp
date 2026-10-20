"""Present retained simulator frames at their recorded 20 Hz control timing."""
import hashlib
import json
from pathlib import Path
import subprocess

import cv2

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/overnight-long-horizon-report-20261020/media'
SOURCES=[
    'grasp-horizon-physical-20261020/synced-point-90-r1/attempt01/initial/rollout.mp4',
    'grasp-horizon-physical-20261020/synced-point-15-r1/attempt01/initial/rollout.mp4',
    'adaptive-family-payload-boundary-20261020/bowl_on_plate-state001/initial/rollout.mp4',
    'adaptive-family-payload-boundary-20261020/open_middle_drawer-state001/initial/rollout.mp4',
    'adaptive-family-payload-boundary-20261020/turn_on_stove-state001/initial/rollout.mp4',
    'subscription-scene-sync-practice-20261020/attempt01/initial/rollout.mp4',
    'subscription-prepared-sequence-practice-20261020/attempt08/initial/rollout.mp4',
    'frozen-prepared-method-transfer-20261020/state011/attempt01/initial/rollout.mp4',
]


def export(relative):
    source=ROOT/'results'/relative
    index=source.parent/'video-frames.json'
    if not source.exists() or not index.exists():return None
    destination=OUTPUT/'--'.join(Path(relative).parts)
    record=destination.with_suffix('.json')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        assert json.loads(record.read_text())['source_sha256']==digest
        return str(destination)
    steps=json.loads(index.read_text())['control_steps']
    assert steps and all(b>a for a,b in zip(steps,steps[1:]))
    cap=cv2.VideoCapture(str(source))
    assert round(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(steps)
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command=['ffmpeg','-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{width}x{height}','-r','20','-i','-','-an','-c:v','libx264','-threads','1',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    count=0
    with destination.with_suffix('.encoder.log').open('w') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for i,step in enumerate(steps):
                okay,frame=cap.read();assert okay
                repeats=steps[i+1]-step if i+1<len(steps) else 1
                for _ in range(repeats):
                    process.stdin.write(frame.tobytes());count+=1
        finally:
            cap.release();process.stdin.close()
        assert process.wait()==0
    check=cv2.VideoCapture(str(destination))
    assert round(check.get(cv2.CAP_PROP_FRAME_COUNT))==count
    assert abs(check.get(cv2.CAP_PROP_FPS)-20)<1e-6
    check.release()
    record.write_text(json.dumps({'source':str(source),'source_sha256':digest,
        'source_control_step_index':str(index),'output_frames':count,'fps':20,
        'control_steps':[steps[0],steps[-1]],'duration_seconds':count/20,
        'presentation':'Repeat retained frames across unsampled control steps; no synthesized motion.',
        'planning_and_model_pauses_included':False,'original_evidence_unchanged':True},indent=2)+'\n')
    return str(destination)


if __name__=='__main__':
    OUTPUT.mkdir(exist_ok=True)
    print(json.dumps({'videos':[p for source in SOURCES if (p:=export(source))]},indent=2))
