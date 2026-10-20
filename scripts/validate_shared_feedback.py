"""Check nominal/issued references, native outcomes, videos and frozen sources."""
import argparse
import hashlib
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json


def verify_episode(folder):
    summary=read_json(folder/'summary.json');actions=read_json(folder/'initial/actions.json')
    expected=[];nominal=[];latency=[];phases=[]
    for segment in sorted((folder/'execution-segments').glob('*')):
        refs=np.load(segment/'nominal.npy');issued=np.load(segment/'issued.npy')
        trace=read_json(segment/'tracking.json');execution=read_json(segment/'execution.json')
        assert len(refs)==len(issued)==len(trace)==execution['issued_commands']
        assert execution['all_commands_issued']
        assert np.array_equal(issued,np.asarray([e['reference_joints'] for e in trace]))
        assert np.array_equal(refs,np.asarray([e['nominal_reference_joints'] for e in trace]))
        assert all(e['execution_veto_checks']==0 for e in trace)
        assert [e['reference_index'] for e in trace]==list(range(len(refs)))
        expected.extend(issued);nominal.extend(refs)
        if 'correction_compute_seconds' in execution:
            latency.append(execution['correction_compute_seconds']/max(1,len(refs)))
        phases.append({'phase':segment.name,'first_step':trace[0]['control_step'] if trace else None,
            'last_step':trace[-1]['control_step'] if trace else None,'references':len(refs)})
    actual=np.asarray([a['target_joints'] for a in actions if 'target_joints' in a]).reshape(-1,7)
    expected=np.asarray(expected).reshape(-1,7)
    assert np.array_equal(actual,expected),'Issued commands differ from native action log'
    assert len(actions)==summary['control_steps']<=600
    success=any(a['evaluation_task_success'] for a in actions)
    final=bool(actions and actions[-1]['evaluation_task_success'])
    assert success==summary['official_task_success'] and final==summary['final_task_success']
    idx=read_json(folder/'initial/video-frames.json');capture=cv2.VideoCapture(str(folder/'initial/rollout.mp4'))
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));first,_=capture.read();capture.set(cv2.CAP_PROP_POS_FRAMES,count-1);last,_=capture.read();capture.release()
    assert first and last and count==len(idx['control_steps'])==idx['frame_count_verified']
    contacts=read_json(folder/'initial/evaluation_contact_diagnostics.json')['steps']
    forces=[max([c['normal_force_n'] for c in row['robot_involved_contacts']],default=0.) for row in contacts]
    observed=[]
    if (folder/'feedback-depth-tracking.json').exists():observed=read_json(folder/'feedback-depth-tracking.json')
    return {'task':summary['task'],'state':summary['initial_state'],'variant':summary['controller_variant'],
        'native_success':success,'final_native_success':final,'control_steps':len(actions),
        'all_nominal_samples_issued_once':True,'issued_joint_commands':len(actual),'execution_vetoes':0,
        'movie_verified':True,'phases':phases,'peak_single_contact_normal_force_n':max(forces,default=0.),
        'mean_correction_compute_ms':float(np.mean(latency)*1000) if latency else None,
        'depth_updates':len(observed),'accepted_depth_updates':sum(x['tracking']['accepted'] for x in observed),
        'contact_mode':read_json(folder/'execution-contact.json').get('mode') if (folder/'execution-contact.json').exists() else None,
        'failure_stage':summary.get('failure_stage'),'failure':summary.get('failure'),
        'reclaimed_bytes':summary['storage']['bytes_reclaimed_this_pass']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--allow-pending',action='store_true');a=p.parse_args()
    root=ROOT/a.run;protocol=read_json(root/'protocol.json');manifest=read_json(root/'manifest.json')['episodes']
    for name,digest in protocol['sources_sha256'].items():
        assert hashlib.sha256((root/'source-snapshot'/name).read_bytes()).hexdigest()==digest
    cases=[];pending=[]
    for e in manifest:
        folder=root/f'{e["task"]}-state{e["state"]:03d}-{e["variant"]}'
        if not (folder/'summary.json').exists() or 'controller_variant' not in read_json(folder/'summary.json'):
            pending.append(e);continue
        cases.append(verify_episode(folder))
    assert not pending or a.allow_pending,'Pending manifest cases'
    result={'passed':not pending,'completed_evidence_passed':True,'planned':len(manifest),'completed':len(cases),
            'pending':pending,'episodes':cases,'source_snapshot_verified':True,
            'current_sources_match_snapshot':all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in protocol['sources_sha256'].items())}
    write_json(root/'verification.json',result)
    print({'passed':result['passed'],'completed':len(cases),'pending':len(pending)})

if __name__=='__main__':main()
