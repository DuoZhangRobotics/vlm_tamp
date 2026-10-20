"""Independent native-outcome, command, video, source and cleanup validation."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_episode(folder):
    summary=read_json(folder/'summary.json');actions=read_json(folder/'initial/actions.json')
    if (folder/'approach-references.npy').exists():
        selected=read_json(folder/'chosen-grasp.json')
        arrays=[np.load(folder/'approach-references.npy'),np.repeat(np.asarray(selected['joints'])[None],40,axis=0)]
        arrays.extend(np.load(p) for p in sorted(folder.glob('plan-*/execution-references.npy')))
    else:arrays=[np.load(p) for p in sorted(folder.glob('path-*/references.npy'))]
    refs=np.vstack(arrays) if arrays else np.empty((0,7))
    actual=np.asarray([x['target_joints'] for x in actions if 'target_joints' in x]).reshape(-1,7)
    tracking=read_json(folder/'tracking.json');tracked=np.asarray([x['reference_joints'] for x in tracking]).reshape(-1,7)
    assert np.array_equal(refs,actual) and np.array_equal(refs,tracked),(folder,'reference mismatch')
    assert all(x['execution_veto_checks']==0 for x in tracking)
    assert len(actions)==summary['control_steps']<=600
    assert summary['official_task_success']==any(x['evaluation_task_success'] for x in actions)
    assert summary['final_task_success']==bool(actions and actions[-1]['evaluation_task_success'])
    index=read_json(folder/'initial/video-frames.json');cap=cv2.VideoCapture(str(folder/'initial/rollout.mp4'))
    frames=int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)));first,_=cap.read();cap.set(cv2.CAP_PROP_POS_FRAMES,frames-1);last,_=cap.read();cap.release()
    assert first and last and frames==index['frame_count_verified']==len(index['control_steps'])
    retention=read_json(folder/'retention.json');assert summary['storage']['bytes_after_cleanup']>0
    assert all(x['arrays_equal'] for x in retention.get('lossless_after_run_recompression',[]))
    return {'folder':str(folder),'task':summary['task'],'state':summary['initial_state'],
        'native_success':summary['official_task_success'],'final_native_success':summary['final_task_success'],
        'all_planned_references_issued':True,'joint_commands':len(refs),'execution_vetoes':0,
        'movie_verified':True,'cleanup_completed':True,'reclaimed_bytes':summary['storage']['bytes_reclaimed_this_pass']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-pending',action='store_true');a=p.parse_args()
    episodes=[];sources=[];pending=[]
    for name in a.runs:
        root=(ROOT/name).resolve();protocol=read_json(root/'protocol.json');hashes=protocol['sources_sha256']
        assert all(sha(root/'source-snapshot'/n)==h for n,h in hashes.items()),('Snapshot mismatch',name)
        sources.append({'run':name,'snapshot_matches':True,'current_sources_match':all((ROOT/n).exists() and sha(ROOT/n)==h for n,h in hashes.items())})
        for row in read_json(root/'manifest.json')['episodes']:
            folder=root/f'{row["task"]}-state{row["state"]:03d}'
            if not (folder/'summary.json').exists():
                pending.append({'run':name,**row});assert a.allow_pending,('Pending episode',row);continue
            episodes.append(check_episode(folder))
        if not pending:assert not (root/'temporary-proposal-cache').exists()
    historical=read_json(ROOT/'results/all-prior-tasks-frozen-v3-20261015/protocol.json')['sources_sha256']
    assert all(sha(ROOT/n)==h for n,h in historical.items())
    ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json');assert ledger['cumulative_conservative_spend_usd']<=2.
    result={'passed':not pending,'completed_evidence_checks_passed':True,'pending':pending,'episodes':episodes,
        'source_snapshots':sources,'historical_pipeline_sources_unchanged':True,'execution_vetoes':0,
        'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'api_cap_usd':2.,
        'total_reclaimed_bytes':sum(x['reclaimed_bytes'] for x in episodes)}
    write_json(ROOT/a.output,result);print({'passed':result['passed'],'episodes':len(episodes),'pending':len(pending)})
if __name__=='__main__':main()
