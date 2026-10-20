"""Post-run bookkeeping audit; never imported by an execution policy."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.rekep_libero.scene_state_v7 import voxel_points


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_partitions(folder):
    """Check that the frozen byte key never aliased different point roles."""
    chain=folder/'robot-chain.json'
    if not chain.exists():return {'scene_contexts_checked':0,'partition_aliases':[]}
    robot_world=np.asarray(read_json(chain)['calibration']['world_to_robot'])
    contexts={};seen=set();aliases=[]
    for path in [*folder.rglob('scene-reference.json'),*folder.rglob('scene-reference.json.gz')]:
        record=read_json(path);spec=record['spec']
        signature=(record['arrays_sha256'],str(spec['front_plane']),str(spec['panel_normal_world']),
            spec.get('measured_panel_surface_half_band_m',.032))
        if signature in seen:continue
        seen.add(signature)
        with np.load(record['arrays_path']) as data:
            static,moving,contact=[data[k] for k in ['static_points','moving_reference','contact_reference']]
        distance=cKDTree(contact).query(moving)[0]
        low,high=np.quantile(contact[:,2],[.01,.99]);normal=np.asarray(spec['panel_normal_world'],float)
        normal/=np.linalg.norm(normal)
        plane=np.abs((moving-np.asarray(spec['front_plane']['point_world_m']))@normal)
        selected=(distance<.028)|((distance<.065)&(plane<spec.get('measured_panel_surface_half_band_m',.032))
            &(moving[:,2]>=low-.015)&(moving[:,2]<=high+.015))
        arrays=[np.ascontiguousarray(transform_points(voxel_points(points,spacing),robot_world),dtype=np.float32)
            for points,spacing in [(static,.0075),(moving[~selected],.005),
                (np.vstack([moving[selected],contact]),.003)]]
        key=hashlib.sha256(b''.join(a.tobytes() for a in arrays)+str(.0025).encode()).hexdigest()
        counts=tuple(len(a) for a in arrays)
        if key in contexts and contexts[key]!=counts:aliases.append({'key':key,'previous':contexts[key],'current':counts})
        contexts[key]=counts
    return {'scene_contexts_checked':len(seen),'partition_aliases':aliases}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    args=parser.parse_args();run=ROOT/args.run
    protocol=read_json(run/'protocol.json');summary=read_json(run/'summary.json');manifest=read_json(run/'manifest.json')
    if summary['completed']!=len(manifest['planned_episodes']):raise ValueError('Episode set is incomplete')
    checks={
        'all_planned_episodes_recorded':summary['planned']==summary['completed']==len(manifest['planned_episodes']),
        'saved_sources_match_frozen_hashes':all(digest(run/'source-snapshot'/p)==h for p,h in protocol['sources_sha256'].items()),
        'baseline_sources_unchanged':all(digest(ROOT/p)==h for p,h in protocol['baseline_frozen_sources_sha256'].items()),
        'api_ledger_unchanged':digest(ROOT/'results/articulated-api-v1/ledger.json')==protocol['api_ledger_sha256_before']}
    episodes=[]
    for planned in manifest['planned_episodes']:
        folder=run/f'{planned["task"]}-state{planned["state"]:03d}'
        row=read_json(folder/'summary.json');actions=read_json(folder/'initial/actions.json')
        events=read_json(folder/'tracking.json')
        phases=read_json(folder/'phases.json') if (folder/'phases.json').exists() else []
        references=[]
        for phase in phases:
            q=np.load(folder/f'plan-{phase["phase"]:02d}/execution-references.npy')
            references.append(bool(phase['all_commands_issued'] and len(q)==phase['issued_commands']==phase['planned_commands']))
        partition=cache_partitions(folder)
        checked={'case':folder.name,'native_success_matches_actions':row['official_task_success']==any(a['evaluation_task_success'] for a in actions),
            'final_native_success_matches':row['final_task_success']==bool(actions and actions[-1]['evaluation_task_success']),
            'native_outcome_label_matches':row['task_outcome']==('success' if row['official_task_success'] else 'failure'),
            'all_issued_contact_trajectories_completed':all(references),
            'issued_contact_trajectories':len(phases),'zero_execution_vetoes':all(e['execution_veto_checks']==0 for e in events),
            'control_limit_respected':len(actions)<=600,'cleanup_record_present':(folder/'retention.json').exists(),
            'at_most_two_raw_observations':len(list(folder.glob('*/obs.npz')))<=2,
            'raw_frames_pruned':not any((folder/'initial/frames').glob('*.jpg')),
            **partition}
        episodes.append(checked)
    good=all(checks.values()) and all(all(v for k,v in e.items() if isinstance(v,bool)) and not e['partition_aliases'] for e in episodes)
    result={'passed':good,'checks':checks,'episodes':episodes,'post_run_only':True,
        'no_collision_or_constraint_execution_checks_added':True}
    write_json(run/'audit.json',result);print({'passed':good,'episodes':len(episodes)})
    if not good:raise SystemExit(1)

if __name__=='__main__':main()
