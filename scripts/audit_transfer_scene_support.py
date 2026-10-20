"""Reconstruct saved model poses and compare them with retained RGB-D only."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from try_rekep_bowl_v3 import read_observation_v3
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.scene_observation_diagnostics import depth_support,overlay


def main():
    study=ROOT/'results/frozen-prepared-method-transfer-20261020'
    out=ROOT/'results/overnight-long-horizon-report-20261020/scene-support'
    out.mkdir(exist_ok=True);rows=[]
    for state in [1,11,21,31]:
        case=study/f'state{state:03d}'
        events=[json.loads(x) for x in (case/'tool-events.jsonl').read_text().splitlines()]
        exports=[e for e in events if e['response'].get('ok') and
                 e['response'].get('result',{}).get('point_clouds')]
        first,last=exports[0],exports[-1]
        episode=case/'attempt01'
        cameras,_=read_observation_v3(episode/'opened',episode/'opened/metadata.json')
        with np.load(case/'agent-workspace'/first['response']['result']['point_clouds']) as data:
            control=depth_support(data['moving'],cameras)[0]
        result={'state':state,'initial_model_control':control,
            'native_success':json.loads((episode/'summary.json').read_text())['official_task_success'],
            'simulator_geometry_used_for_estimation':False}
        if not result['native_success']:
            with np.load(case/'agent-workspace'/last['response']['result']['point_clouds']) as data:
                moving=data['moving'].copy();contact=data['contact'].copy()
            applied=[]
            for path in sorted(episode.glob('*-scene-pose-update.json')):
                number=int(path.name.split('-')[0])
                if number<=last['sequence']:continue
                update=json.loads(path.read_text());T=np.asarray(update['transform_world'])
                moving=transform_points(moving,T);contact=transform_points(contact,T)
                applied.append({'operation':number,'source':update['source']})
            retained=sorted(episode.glob('*-observation/obs.npz'))
            latest=retained[-1].parent
            cameras,_=read_observation_v3(latest,latest/'metadata.json')
            model,points,status=depth_support(moving,cameras)
            contact_info,_,_=depth_support(contact,cameras)
            destination=out/f'state{state:03d}-model-overlay.png'
            overlay(points,status,cameras['frontview'],destination)
            result.update(final_model=model,final_contact_model=contact_info,applied_updates=applied,
                overlay=str(destination),observation=str(latest))
        rows.append(result)
    (out/'audit.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{'state':r['state'],'native_success':r['native_success'],
        'initial_support':r['initial_model_control']['supported_fraction'],
        'final_support':r.get('final_model',{}).get('supported_fraction'),
        'final_free_space_conflict':r.get('final_model',{}).get('multi_view_free_space_conflict_fraction'),
        'contact_free_space_conflict':r.get('final_contact_model',{}).get('multi_view_free_space_conflict_fraction')} for r in rows],indent=2))


if __name__=='__main__':main()
