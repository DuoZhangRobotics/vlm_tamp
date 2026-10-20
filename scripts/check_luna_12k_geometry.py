"""Check completed RGB choices against the existing sensor-only geometry stage."""
from pathlib import Path
import traceback

import numpy as np

from try_rekep_bowl_v3 import read_observation_v3
from vlm_tamp.coupled_pipeline import handle_anchor_menu
from vlm_tamp.coupled_resources import fresh_task_template
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.handle_contact import ground_handle

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'results/microwave-luna-12k-20261015'
    summary=read_json(root/'summary.json');rows=[]
    for entry in summary['results']:
        label=f'close_microwave-state{entry["state"]:03d}'
        folder=root/label/'numerical';folder.mkdir(exist_ok=True)
        row={'state':entry['state'],'simulator_scene_state_used':False,'new_api_calls':0,
             'stage':'current sensor-only handle/hinge geometry; not physical execution'}
        try:
            choices=read_json(root/label/'choices.json')
            observation=Path(entry['source']).parent/'opened'
            metadata=read_json(observation/'metadata.json')
            cameras,_=read_observation_v3(observation,metadata)
            template=fresh_task_template(read_json(ROOT/'results/articulated-physics-pilot/close_microwave-state001/task-spec.json'),choices)
            mechanism,cloud,mask,anchors,evidence=ground_handle(cameras,choices,template,metadata)
            anchors,support=handle_anchor_menu(anchors[0][:2],cloud,metadata)
            write_json(folder/'task-spec.json',mechanism.spec)
            write_json(folder/'geometry.json',evidence);write_json(folder/'pad-support.json',support)
            np.save(folder/'handle-cloud.npy',cloud);np.save(folder/'panel-mask.npy',mask)
            row.update(status='geometry_and_pad_support_passed',handle_points=len(cloud),anchors_world=anchors)
        except Exception as error:
            row.update(status='geometry_failed',failure=str(error))
            (folder/'error.txt').write_text(traceback.format_exc())
        rows.append(row);write_json(folder/'summary.json',row)
        print({k:v for k,v in row.items() if k!='anchors_world'},flush=True)
    write_json(root/'numerical-summary.json',{'planned':len(rows),'passed':sum(r['status']=='geometry_and_pad_support_passed' for r in rows),
        'results':rows,'physical_execution':False,'schema_validity_is_not_visual_or_physical_success':True})


if __name__=='__main__':main()
