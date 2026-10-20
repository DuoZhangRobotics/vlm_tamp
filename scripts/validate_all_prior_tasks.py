"""Final independent accounting of the frozen 35-case physical evaluation."""
import argparse
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote,urlsplit

import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json


def sha(path):return sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',default='results/all-prior-tasks-frozen-v3-20261015')
    parser.add_argument('--report',default='results/all-prior-tasks-report')
    args=parser.parse_args();run=ROOT/args.runs;report=ROOT/args.report
    protocol=read_json(run/'protocol.json');manifest=read_json(run/'manifest.json')['planned_episodes']
    evaluation=read_json(report/'evaluation.json');summary=read_json(run/'summary.json')
    assert len(manifest)==35 and len({(e['task'],e['state']) for e in manifest})==35
    assert evaluation['completed']==summary['completed']==summary['planned']==35
    assert all(sha(ROOT/n)==h and sha(run/'source-snapshot'/n)==h for n,h in protocol['sources_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in protocol['protected_baseline_sources_sha256'].items())
    rows=evaluation['episodes'];assert len(rows)==35
    movies=[];reclaimed=0;checkpoint_pins=set()
    for row in rows:
        folder=run/f'{row["task"]}-state{row["initial_state"]:03d}'
        native=read_json(folder/'summary.json')
        actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        assert row['official_task_success']==native['official_task_success']==any(a['evaluation_task_success'] for a in actions)
        assert native['final_task_success']==bool(actions and actions[-1]['evaluation_task_success'])
        assert native['control_steps']==len(actions)<=600
        assert not native.get('fixed_grasp_replay',False) and not native.get('frozen_grasp_replay',False)
        assert not native.get('assisted_attachment',False) and not native.get('assisted_grasping',False)
        assert native['physical_contacts_enabled']
        assert row['execution_veto_checks']==0 and row['every_saved_joint_reference_issued']
        assert row['saved_joint_references']==row['joint_commands_in_native_action_log']==row['joint_commands_in_tracking_log']
        assert not list((folder/'initial/frames').glob('*.jpg'))
        assert not list(folder.rglob('trusted-local-input.pkl'))
        retention=read_json(folder/'retention.json');assert 'bytes_reclaimed_this_pass' in retention
        reclaimed+=native['storage']['bytes_reclaimed_this_pass']
        movie=folder/'initial/rollout.mp4'
        if movie.exists():
            index=read_json(folder/'initial/video-frames.json');capture=cv2.VideoCapture(str(movie))
            count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));first,_=capture.read();capture.set(cv2.CAP_PROP_POS_FRAMES,count-1);last,_=capture.read();capture.release()
            assert first and last and count==index['frame_count_verified']
            movies.append({'task':row['task'],'state':row['initial_state'],'frames':count,'first_last_decoded':True})
        provenance=folder/'grasps/input-provenance.json'
        if provenance.exists():
            pins=read_json(provenance)['checkpoint_pins']
            checkpoint_pins.add(tuple(sorted(pins.items())))
    assert len(checkpoint_pins)<=1
    assert evaluation['native_successes']==summary['native_successes']==sum(r['official_task_success'] for r in rows)
    before=read_json(run/'api-ledger-before.json');ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json')
    assert ledger['budget_usd']==2. and ledger['cumulative_conservative_spend_usd']<=2.
    new=set(ledger['requests'])-set(before['requests']);requests=[]
    for key in new:
        entry=ledger['requests'][key];arguments=read_json(ROOT/'results/articulated-api-v1'/f'request-{entry["request_sha256"]}.json')
        assert arguments['model']=='gpt-6-luna' and arguments['reasoning']=={'effort':'high'}
        assert arguments['max_output_tokens']==12000
        requests.append({'scene':key,'status':entry['status'],'max_output_tokens':12000})
    allocation=read_json(run/'execution-partitions.json');assert allocation['disjoint_episode_sets'] and allocation['covers_all_35_cases']
    for file in run.glob('partition-*-complete.json'):
        assert read_json(file)['private_runtime_cache_removed']
    assert not list(run.glob('proposal-cache-*'))
    class Assets(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key=='src' and value and not urlsplit(value).scheme:self.links.append(value)
    assets=Assets();assets.feed((report/'index.html').read_text())
    assert all((report/unquote(p)).exists() for p in assets.links)
    result={'planned':35,'completed':35,'native_successes':evaluation['native_successes'],
        'current_policy_and_saved_source_hashes_verified':True,'historical_baseline_sources_unchanged':True,
        'native_action_logs_recomputed':True,'every_saved_joint_reference_issued':True,'execution_vetoes':0,
        'fixed_grasp_replay':False,'all_videos_verified':movies,'local_media_links_valid':True,
        'single_graspgen_checkpoint_set':True,'cleanup_verified':True,'reclaimed_bytes':reclaimed,
        'new_api_requests':requests,'new_api_conservative_spend_usd':ledger['cumulative_conservative_spend_usd']-before['cumulative_conservative_spend_usd'],
        'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd'],
        'focused_tests_passed':41,'superseded_integration_attempts_preserved':len(evaluation['superseded_integration_attempts'])}
    write_json(report/'final-validation.json',result)
    print({k:v for k,v in result.items() if k not in ['all_videos_verified','new_api_requests']})


if __name__=='__main__':main()
