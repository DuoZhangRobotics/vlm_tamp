"""Validate optional DINO tools on existing observations, without robot actions."""
import json
from pathlib import Path
import subprocess
import os

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'results/subscription-agent-fit-diagnostics-20261019/agent-workspace/robot_data/attempt02'
    output=ROOT/'results/dino-agent-tool-smoke-20261019';output.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for k in list(env):
        if k.endswith(('_API_KEY','_TOKEN','_SECRET')):env.pop(k,None)
    def query(name,request):
        folder=output/name
        with (output/(name+'.log')).open('w') as log:
            p=subprocess.run([str(ROOT/'.runtime/envs/dinov3/bin/python'),str(ROOT/'scripts/dino_agent_worker.py')],
                input=json.dumps({**request,'output':str(folder)}),text=True,capture_output=False,
                stdout=subprocess.PIPE,stderr=log,env=env,cwd=ROOT,timeout=240,check=True)
        return json.loads(p.stdout.strip().splitlines()[-1])
    initial=str(source/'obs014/sensors.npz');final=str(source/'obs038/sensors.npz')
    proposals=query('landmarks',{'operation':'landmarks','current_sensors':initial,'camera':'frontview',
        'region_xyxy':[165,197,346,312],'count':8})
    pixels=[r['pixel_xy'] for r in proposals['candidates']]
    identity=query('identity',{'operation':'match','current_sensors':initial,'reference_sensors':initial,
        'camera':'frontview','reference_camera':'frontview','pixels_xy':pixels,'region_xyxy':[165,197,346,312]})
    moved=query('moved',{'operation':'match','current_sensors':final,'reference_sensors':initial,
        'camera':'frontview','reference_camera':'frontview','pixels_xy':pixels})
    errors=[]
    for a,row in zip(pixels,identity['matches']):
        b=row['candidates'][0]['pixel_xy'];errors.append(((a[0]-b[0])**2+(a[1]-b[1])**2)**.5)
    result={'candidate_count':len(pixels),'identity_match_errors_px':errors,
        'identity_passed':bool(errors and max(errors)<=4),
        'changed_view_matches':len(moved['matches']),
        'changed_view_ambiguous_count':sum(r.get('ambiguous',True) for r in moved['matches']),
        'changed_view_is_qualitative_check_not_ground_truth_accuracy':True,
        'physical_actions_executed':0,'model_api_calls':0,
        'weights_sha256':proposals['weights_sha256'],'revision':proposals['revision'],
        'peak_gpu_allocated_mb':max(r['peak_gpu_allocated_mb'] for r in [proposals,identity,moved]),
        'total_worker_seconds':sum(r['total_seconds'] for r in [proposals,identity,moved]),
        'dense_features_saved_to_disk':False}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['identity_passed']:raise RuntimeError('DINO coordinate contract smoke test failed')


if __name__=='__main__':main()
