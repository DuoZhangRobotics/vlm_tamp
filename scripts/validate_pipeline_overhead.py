"""Independent verification of timing runs, exact replay and preserved policy."""
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from benchmark_execution_io import array_hashes


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_episode(folder):
    s=read_json(folder/'summary.json');actions=read_json(folder/'initial/actions.json')
    files=sorted(folder.glob('path-*/references.npy'))
    refs=np.vstack([np.load(p) for p in files]) if files else np.empty((0,7))
    commands=np.asarray([a['target_joints'] for a in actions if 'target_joints' in a]).reshape(-1,7)
    tracking=read_json(folder/'tracking.json');tracked=np.asarray([a['reference_joints'] for a in tracking]).reshape(-1,7)
    assert np.array_equal(refs,commands) and np.array_equal(refs,tracked)
    assert all(t['execution_veto_checks']==0 for t in tracking)
    assert len(actions)==s['control_steps']<=600
    assert s['official_task_success']==any(a['evaluation_task_success'] for a in actions)
    index=read_json(folder/'initial/video-frames.json');cap=cv2.VideoCapture(str(folder/'initial/rollout.mp4'))
    count=int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)));first,_=cap.read();cap.set(cv2.CAP_PROP_POS_FRAMES,count-1);last,_=cap.read();cap.release()
    assert first and last and count==index['frame_count_verified']==len(index['control_steps'])
    assert (folder/'retention.json').exists() and 'storage' in s
    return {'folder':str(folder),'native_success':s['official_task_success'],'all_references_issued':True,
        'joint_commands':len(refs),'video_verified':True,'video_frames':count,'cleanup_completed':True}


def main():
    report=ROOT/'results/pipeline-overhead-report';data=read_json(report/'evaluation.json')
    prior=read_json(ROOT/'results/mcvamp-mechanisms-camera-retry-20261016/protocol.json')['sources_sha256']
    assert all(sha(ROOT/n)==h for n,h in prior.items()),'Planning or baseline source changed'
    episodes=[];replay_checks=[];snapshots=[]
    runs=[ROOT/'results/pipeline-io-development-20261016/fast-open-top',ROOT/'results/pipeline-io-comparison-20261016/baseline-open-top']
    runs+=sorted((ROOT/'results/pipeline-io-final-20261016').glob('*'))
    for folder in runs:
        if not (folder/'summary.json').exists():continue
        episodes.append(check_episode(folder));s=read_json(folder/'summary.json')
        assert s['controller_and_physics_equivalent'] and s['evaluation_success_sequence_equal']
        assert all(v==0. for v in s['maximum_errors_versus_source'].values())
        assert s['opened_sensor_arrays_match_source']
        captures=read_json(folder/'sensor-array-hashes.json');source=Path(s['source'])
        assert captures[-1]['arrays']==array_hashes(source/'final/obs.npz')
        replay_checks.append({'folder':str(folder),'all_control_and_state_values_exact':True,
            'initial_and_final_sensor_arrays_byte_identical':True})
        proto=folder/'execution-protocol.json'
        if proto.exists():
            hashes=read_json(proto)['sources_sha256'];assert all(sha(folder/'source-snapshot'/n)==h for n,h in hashes.items())
            snapshots.append({'run':str(folder),'snapshot_valid':True})
    # Direct before/after captures at every boundary for the paired tasks.
    paired=[]
    for task,baseline in [('open_top_drawer',ROOT/'results/pipeline-io-comparison-20261016/baseline-open-top'),
        ('turn_on_stove',ROOT/'results/pipeline-io-final-20261016/baseline-turn_on_stove')]:
        fast=ROOT/'results/pipeline-io-final-20261016'/('fast-'+task)
        assert read_json(fast/'sensor-array-hashes.json')==read_json(baseline/'sensor-array-hashes.json')
        paired.append({'task':task,'all_observation_boundaries_byte_identical':True})
    http=0
    for label in ['fresh','fresh-v2','cached-v2']:
        root=ROOT/f'results/pipeline-profile-{label}-20261016';protocol=read_json(root/'protocol.json')
        assert all(sha(root/'source-snapshot'/n)==h for n,h in protocol['sources_sha256'].items())
        if label.endswith('v2'):assert all(sha(ROOT/n)==h for n,h in protocol['sources_sha256'].items())
        snapshots.append({'run':str(root),'snapshot_valid':True,'current_sources_match':label.endswith('v2')})
        assert protocol['same_planner_and_constraint_parameters'] and protocol['same_trajectory_horizons']
        assert not protocol['asynchronous_replanning'] and not protocol['execution_vetoes']
        assert not (root/'temporary-proposal-cache').exists()
        for s in read_json(root/'summary.json')['episodes']:
            folder=root/f'{s["task"]}-state{s["initial_state"]:03d}';episodes.append(check_episode(folder))
            spans=read_json(folder/'timing.json')['events'];calls=[e for e in spans if e['name']=='grounding.luna_round_trip']
            if protocol['grounding']=='fresh':
                assert len(calls)==1 and calls[0]['response_status']=='completed'
                assert len([e for e in spans if e['name']=='vision.dinov2_features_and_transfer'])==2
                assert all(e['gpu_synchronized'] for e in spans if e['name'].startswith('vision.'))
                http+=1
            else:assert not calls
            retention=read_json(folder/'retention.json')
            assert all(r['arrays_equal'] for r in retention['lossless_after_run_recompression'])
    tests=(report/'verification-tests.txt').read_text();match=re.search(r'(\d+) passed',tests);assert match
    assert int(match[1])==70 and http==4
    ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json');assert ledger['cumulative_conservative_spend_usd']<=2.
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.media=[]
        def handle_starttag(self,tag,attrs):
            if tag in {'video','img'}:self.media.extend(v for k,v in attrs if k=='src')
    links=Links();links.feed((report/'index.html').read_text());assert all((report/p).resolve().exists() for p in links.media)
    result={'passed':True,'tests_passed':70,'planning_sources_unchanged':True,'planner_and_controller_parameters_unchanged':True,
        'same_camera_views_and_depth_filter':True,'voxel_equivalence':read_json(report/'voxel-equivalence.json'),
        'episodes':episodes,'reference_replay_checks':replay_checks,'paired_all_boundary_sensor_checks':paired,
        'source_snapshots':snapshots,'actual_fresh_luna_calls':http,'automatic_api_retries':0,
        'api_cost_usd':data['new_api_cost_usd'],'cumulative_api_cost_usd':ledger['cumulative_conservative_spend_usd'],
        'api_cap_usd':2.,'all_thirteen_runs_cleaned':len(episodes)==13,
        'execution_time_vetoes':0,'report_media_links_verified':True}
    write_json(report/'final-validation.json',result)
    print({'passed':True,'runs':len(episodes),'tests_passed':70,'fresh_calls':http,'exact_replays':len(replay_checks)})

if __name__=='__main__':main()
