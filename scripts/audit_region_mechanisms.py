"""POST-EXECUTION evaluation; never imported by a policy or motion tool.

Compare sensor-derived estimates with the requested native drawer joint only
after workers have closed. These oracle values are report-only.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain

def audit(folder):
    folder=Path(folder);summary=read_json(folder/'summary.json')
    raw=read_json(folder/'initial/evaluation_contact_diagnostics.json')
    assert raw['evaluation_only'] and raw['excluded_from_policy_input']
    steps={r['step']:r for r in raw['steps']}
    def joint(step):return next(x for x in steps[step]['evaluation_only_articulation_state'] if x['joint_name'].endswith('_bottom_level'))
    first,last=min(steps),max(steps)
    spec=read_json(folder/'current-task-spec.json') if (folder/'current-task-spec.json').exists() else read_json(folder/'task-spec.json')
    predicted=np.asarray(spec['joint_axis_world'],float);native=np.asarray(joint(last)['axis_world'])
    dot=float(np.clip(predicted@native/(np.linalg.norm(predicted)*np.linalg.norm(native)),-1,1))
    phases=[];commands=[];chain=RobotChain(read_json(folder/'robot-chain.json'))
    actions=read_json(folder/'initial/actions.json')
    for f in sorted((folder/'execution-segments').glob('*-manipulate/tracking.json')):
        records=read_json(f)
        if not records or 'normal_world' not in records[0]:continue
        a=min(r['control_step'] for r in records)-1;b=max(r['control_step'] for r in records)
        if a in steps:
            phases.append({'segment':f.parent.name,'before_step':a,'after_step':b,
                'native_joint_delta_m':joint(b)['position']-joint(a)['position']})
            nominal=np.load(f.parent/'nominal.npy');issued=np.load(f.parent/'issued.npy')
            before=np.asarray(actions[a-1]['eef_pose_world_xyzw'][:3]);after=np.asarray(actions[b-1]['eef_pose_world_xyzw'][:3])
            commands.append({'segment':f.parent.name,'nominal_terminal_translation_m':(chain.forward(nominal[-1])[:3]-before).tolist(),
                'issued_terminal_translation_m':(chain.forward(issued[-1])[:3]-before).tolist(),
                'actual_hand_translation_m':(after-before).tolist(),'native_drawer_joint_delta_m':joint(b)['position']-joint(a)['position']})
    history=read_json(folder/'strategy-history.json')
    estimates=[x['result']['object_progress_delta_m'] for x in history
               if x['result'].get('manipulation_executed') and x['result'].get('object_progress_delta_m') is not None]
    return {'folder':str(folder),'state':summary['initial_state'],'policy':summary['strategy_policy'],
        'evaluation_only':True,'provided_to_policy':False,'native_task_success':summary['official_task_success'],
        'native_joint_name':joint(last)['joint_name'],'native_positive_joint_axis_world':native.tolist(),
        'last_estimated_axis_world':predicted.tolist(),'last_axis_line_error_deg':float(np.degrees(np.arccos(abs(dot)))),
        'last_axis_direction_error_to_positive_joint_deg':float(np.degrees(np.arccos(dot))),
        'native_initial_position_m':joint(first)['position'],'native_final_position_m':joint(last)['position'],
        'native_total_joint_delta_m':joint(last)['position']-joint(first)['position'],
        'native_manipulation_joint_delta_m':sum(r['native_joint_delta_m'] for r in phases),
        'reported_projected_manipulation_progress_m':sum(estimates),'manipulation_segments':phases,
        'last_four_manipulation_commands_and_motion':commands[-4:],
        'last_robot_contacts_evaluation_only':[{k:c[k] for k in ('geom_names','distance_m','normal_force_n')}
            for c in steps[last]['robot_involved_contacts']],
        'scope':'Reported progress can refer to revised regions; joint deltas always refer to the requested bottom drawer. Last-axis error describes the final selected region hypothesis.'}

def main():
    roots=[ROOT/'results'/name for name in ('region-recovery-frozen-v2-20261018','region-recovery-separate-20261018',
        'region-recovery-continuation-v2-20261018')]
    rows=[]
    for root in roots:
        for p in root.glob('*/summary.json'):
            rows.append(audit(p.parent))
    out=ROOT/'results/region-recovery-report';out.mkdir(exist_ok=True)
    write_json(out/'mechanism-evaluation.json',{'evaluation_only':True,'never_used_for_control':True,'episodes':rows})
    print({'evaluation_only_episodes':len(rows)})

if __name__=='__main__':main()
