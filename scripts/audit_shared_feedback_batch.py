"""Batch small, evaluation-only object-motion audits of completed episodes."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from audit_shared_feedback_object_motion import audit
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.family_transfer import TASKS as MECHANISMS
from run_all_prior_tasks import TASKS as OTHERS


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    tasks={t['label']:t for t in [*OTHERS,*MECHANISMS]};out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    existing={read_json(p)['source']:p for p in out.rglob('object-motion.json')};rows=[]
    for name in a.runs:
        root=ROOT/name
        for entry in read_json(root/'manifest.json')['episodes']:
            folder=root/f'{entry["task"]}-state{entry["state"]:03d}-{entry["variant"]}'
            if not (folder/'summary.json').exists():continue
            summary=read_json(folder/'summary.json')
            if 'controller_variant' not in summary:continue
            if str(folder) in existing:result=read_json(existing[str(folder)])
            else:
                print(json.dumps({'started':str(folder)}),flush=True)
                result=audit(folder,out/root.name/folder.name,tasks[entry['task']])
            rows.append(result)
            write_json(out/'summary.json',{'completed':len(rows),'all_replays_match':all(r['accepted_as_original_episode_measurement'] for r in rows),'episodes':rows})
            print(json.dumps({'completed':str(folder),'matches':result['accepted_as_original_episode_measurement']}),flush=True)

if __name__=='__main__':main()
