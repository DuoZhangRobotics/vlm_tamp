"""Verify frozen strategy choices, feedback boundaries and actual simulator actions."""
import argparse
import hashlib
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.strategy_decision import parse_decision
from vlm_tamp.recovery_decision import parse_decision as parse_recovery
from vlm_tamp.region_decision import parse_decision as parse_region
from vlm_tamp.rekep_libero.articulated_grounding import response_text

FORBIDDEN={'evaluation','evaluation_task_success','official_task_success','final_task_success',
           'simulator_object_pose','object_ground_truth','true_joint_axis','object_ids','state_index'}


def check_feedback(value):
    if isinstance(value,dict):
        assert not FORBIDDEN.intersection(value),FORBIDDEN.intersection(value)
        for child in value.values():check_feedback(child)
    elif isinstance(value,list):
        for child in value:check_feedback(child)


def verify_episode(folder):
    summary=read_json(folder/'summary.json');actions=read_json(folder/'initial/actions.json')
    assert len(actions)==summary['control_steps']<=600
    assert not actions or not actions[0]['evaluation_task_success'], 'Episode starts already successful'
    assert any(row['evaluation_task_success'] for row in actions)==summary['official_task_success']
    assert bool(actions and actions[-1]['evaluation_task_success'])==summary['final_task_success']
    issued=[];correction=[]
    for segment in sorted((folder/'execution-segments').glob('*')):
        nominal=np.load(segment/'nominal.npy');commands=np.load(segment/'issued.npy')
        trace=read_json(segment/'tracking.json');record=read_json(segment/'execution.json')
        assert len(nominal)==len(commands)==len(trace)==record['issued_commands']
        assert record['all_commands_issued']
        assert np.array_equal(commands,np.asarray([r['reference_joints'] for r in trace]))
        assert np.array_equal(nominal,np.asarray([r['nominal_reference_joints'] for r in trace]))
        assert all(r['execution_veto_checks']==0 for r in trace)
        if trace and 'reference_index' in trace[0]:assert [r['reference_index'] for r in trace]==list(range(len(trace)))
        issued.extend(commands);correction.extend(np.max(np.abs(commands-nominal),axis=1))
    actual=np.asarray([a['target_joints'] for a in actions if 'target_joints' in a]).reshape(-1,7)
    assert np.array_equal(actual,np.asarray(issued).reshape(-1,7))
    decisions=[]
    for path in sorted((folder/'decisions').glob('*/decision.json')):
        packet=read_json(path.parent/'feedback.json');check_feedback(packet)
        version=read_json(path).get('version')
        parse=parse_region if version=='strategy-v3' else parse_recovery if version=='strategy-v2' else parse_decision
        choice=parse(path.read_text())
        if (path.parent/'response.json').exists():
            raw=read_json(path.parent/'response.json');assert raw['status']=='completed'
            assert parse(response_text(raw))==choice
            account=read_json(path.parent/'api-accounting.json')
            assert account['max_output_tokens']==32000 and account['reasoning_effort']=='high'
        result=read_json(path.parent/'tool-result.json') if (path.parent/'tool-result.json').exists() else None
        if (path.parent/'contact-menu.json').exists():
            menu=read_json(path.parent/'contact-menu.json')
            assert menu['selection_evidence']['mode_order_source']=='caller'
            assert all(row['mode']==choice['contact_mode'] for row in menu['candidates'])
        if choice['version'] in ('strategy-v2','strategy-v3') and result:
            if choice['step_mm'] is not None:
                assert all(0<x<=choice['step_mm']/1000+1e-10 for x in result.get('executed_plan_distances_m',[]))
            selected=path.parent/'chosen-grasp.json'
            if selected.exists():
                grasp=read_json(selected)
                assert abs(grasp['probe']['progress']-choice['step_mm']/1000)<1e-10
                if choice['posture_rank'] is not None:assert grasp['ik_rank']==choice['posture_rank']
            if choice['operation']=='release_reposition' and result.get('release_executed'):
                assert result.get('retreat_executed') is not None
        decisions.append({'round':path.parent.name,'choice':choice,'result':result,'folder':str(path.parent)})
    assert len(decisions)<=summary.get('maximum_decision_rounds',5)
    video=folder/'initial/rollout.mp4';capture=cv2.VideoCapture(str(video))
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));first,_=capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES,count-1);last,_=capture.read();capture.release()
    index=read_json(folder/'initial/video-frames.json');assert first and last and count==index['frame_count_verified']
    return {'task':summary['task'],'state':summary['initial_state'],'policy':summary['strategy_policy'],
        'success':summary['official_task_success'],'final_success':summary['final_task_success'],
        'control_steps':len(actions),'decisions':decisions,'issued_commands':len(issued),
        'complete_trajectory_execution_verified':True,'feedback_boundary_verified':True,
        'no_automatic_mode_switch_verified':True,'video_verified':True,
        'maximum_joint_reference_correction_rad':max(correction,default=0.),
        'failure':summary.get('failure'),'failure_stage':summary.get('failure_stage'),
        'seconds':summary['episode_wall_seconds'],'folder':str(folder)}


def verify_run(root):
    protocol=read_json(root/'protocol.json');manifest=read_json(root/'manifest.json')['episodes']
    for path,digest in protocol['sources_sha256'].items():
        assert hashlib.sha256((root/'source-snapshot'/path).read_bytes()).hexdigest()==digest
    rows=[];pending=[]
    for entry in manifest:
        folder=root/f"{entry['task']}-state{entry['state']:03d}-{entry['policy']}"
        if not (folder/'summary.json').exists():pending.append(entry);continue
        rows.append(verify_episode(folder))
    pairs=[]
    for key in {(r['task'],r['state']) for r in rows}:
        group=[r for r in rows if (r['task'],r['state'])==key]
        if len(group)<2:continue
        fields={};availability={}
        for file in ['opened/agentview.png','opened/frontview.png','grasp-cloud.npy',
                     'grasps/grasp_candidates.npz','task-spec.json','grounding/response.json',
                     'grounding/validation-repair/response.json']:
            present=[(Path(row['folder'])/file).exists() for row in group]
            availability[file]={'present_in':sum(present),'paired_episodes':len(group)}
            if not all(present):
                fields[file]=None if not any(present) else False
                continue
            values=[hashlib.sha256((Path(row['folder'])/file).read_bytes()).hexdigest() for row in group]
            # NPZ container timestamps can differ; compare the actual arrays.
            if file.endswith('.npz'):
                with np.load(Path(group[0]['folder'])/file) as a:base={k:a[k].copy() for k in a.files}
                equal=True
                for row in group[1:]:
                    with np.load(Path(row['folder'])/file) as a:
                        equal &= set(a.files)==set(base) and all(np.array_equal(a[k],base[k]) for k in base)
                fields[file]=bool(equal)
            elif file.endswith('.json'):
                base=read_json(Path(group[0]['folder'])/file)
                fields[file]=all(read_json(Path(row['folder'])/file)==base for row in group[1:])
            else:fields[file]=len(set(values))==1
        pairs.append({'task':key[0],'state':key[1],'input_equality':fields,'availability':availability,
            'all_identical':all(v is not False for v in fields.values())})
    return {'completed':len(rows),'planned':len(manifest),'pending':pending,'episodes':rows,
        'paired_inputs':pairs,'source_snapshots_verified':True,'all_pairs_identical':all(p['all_identical'] for p in pairs)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    root=ROOT/a.run;result=verify_run(root);write_json(root/'verification.json',result)
    print({k:v for k,v in result.items() if k not in {'episodes','paired_inputs'}})


if __name__=='__main__':main()
