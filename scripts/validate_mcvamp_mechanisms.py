"""Independent saved-reference, source, movie and native-outcome audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.coupled_storage import read_json,write_json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();episodes=[];source_checks=[];keys=set()
    for run in args.runs:
        root=(ROOT/run).resolve();protocol=read_json(root/'protocol.json')
        current=all(sha(ROOT/n)==h for n,h in protocol['sources_sha256'].items())
        snapshot=all(sha(root/'source-snapshot'/n)==h for n,h in protocol['sources_sha256'].items())
        source_checks.append({'run':str(root),'current_sources_match':current,'snapshot_matches':snapshot})
        for entry in read_json(root/'manifest.json')['episodes']:
            key=(entry['task'],entry['state']);assert key not in keys;keys.add(key)
            folder=root/f'{key[0]}-state{key[1]:03d}';summary=read_json(folder/'summary.json')
            paths=sorted(folder.glob('path-*/references.npy'))
            refs=np.vstack([np.load(p) for p in paths]) if paths else np.empty((0,7))
            actions=read_json(folder/'initial/actions.json');joint=[a for a in actions if 'target_joints' in a]
            issued=np.asarray([a['target_joints'] for a in joint],float).reshape(-1,7)
            tracking=read_json(folder/'tracking.json');tracked=np.asarray([a['reference_joints'] for a in tracking],float).reshape(-1,7)
            match=bool(np.array_equal(refs,issued) and np.array_equal(refs,tracked))
            assert match,(key,'reference mismatch')
            assert all(e.get('execution_veto_checks')==0 for e in tracking)
            assert summary['control_steps']==len(actions)<=600
            assert summary['official_task_success']==any(a['evaluation_task_success'] for a in actions)
            assert summary['final_task_success']==bool(actions and actions[-1]['evaluation_task_success'])
            index=read_json(folder/'initial/video-frames.json');capture=cv2.VideoCapture(str(folder/'initial/rollout.mp4'))
            try:
                count=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)));first,_=capture.read()
                capture.set(cv2.CAP_PROP_POS_FRAMES,count-1);last,_=capture.read()
            finally:capture.release()
            assert first and last and count==index['frame_count_verified']==len(index['control_steps'])
            assert (folder/'retention.json').exists()
            episodes.append({'task':key[0],'state':key[1],'native_success':summary['official_task_success'],
                'all_saved_references_issued':match,'joint_references':len(refs),'execution_vetoes':0,
                'movie_verified':True,'movie_frames':count,'cleanup_completed':True})
    historical=read_json(ROOT/'results/all-prior-tasks-frozen-v3-20261015/protocol.json')['sources_sha256']
    baseline_unchanged=all(sha(ROOT/n)==h for n,h in historical.items())
    ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json');assert ledger['cumulative_conservative_spend_usd']<=2
    assert baseline_unchanged and all(r['current_sources_match'] and r['snapshot_matches'] for r in source_checks)
    result={'passed':True,'episodes':episodes,'sources':source_checks,'historical_baseline_sources_unchanged':baseline_unchanged,
        'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'api_cap_usd':2.,
        'successes':sum(e['native_success'] for e in episodes)}
    write_json(ROOT/args.output,result);print(json.dumps({'passed':True,'episodes':len(episodes),'successes':result['successes']}))

if __name__=='__main__':main()
