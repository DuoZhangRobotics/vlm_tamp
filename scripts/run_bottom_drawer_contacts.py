"""Physical contact-mode diagnostics, separate from the frozen benchmark.

Proposals use only retained RGB-D and known robot geometry. Scene-collision
admission is deliberately bypassed to test actual physics; robot IK and joint
limits remain. Every issued reference is executed. No object state assignment,
native object geometry, assisted grasp or success feedback enters the policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp.coupled_storage import read_json,write_json,compact_episode
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.profiled_sim_client import ProfiledClient,execute_batch
from vlm_tamp.rekep_libero.geometry import pose_matrix,matrix_pose,transform_points,interpolate_path
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK
from vlm_tamp.timed_execution import schedule

MODES=['detector_full_open','frontal_full_open','tilted_partial_pinch','tilted_parallel_hook','tilted_parallel_feedback']


def unit(x):return np.asarray(x,float)/np.linalg.norm(x)


def proposals(source):
    spec=read_json(source/'task-spec.json');meta=read_json(source/'opened/metadata.json');cloud=np.load(source/'grasp-cloud.npy')
    normal=unit(spec['joint_axis_world']);up=np.array([0.,0.,1.]);side=unit(np.cross(normal,up))
    centre=np.median(cloud,axis=0);axial=(cloud-centre)@side;lo,hi=np.quantile(axial,[.25,.75]);central=cloud[(axial>=lo)&(axial<=hi)]
    basis=np.column_stack([side,normal,up]);bounds=np.quantile(central@basis,[.02,.98],axis=0)
    centre=basis@bounds.mean(0);extent=bounds[1]-bounds[0]
    inverse=np.linalg.inv(pose_matrix(meta['eef_pose_world_xyzw']))
    pads=[transform_points(v['vertices_world'],inverse) for v in meta['robot_collision_volumes_world'] if v['robot_geom_name'].endswith('_pad_collision')]
    pad_local=np.mean([v.mean(0) for v in pads],axis=0)
    pad_depth=float(np.ptp(pads[0][:,1]));pad_height=float(np.ptp(pads[0][:,2]))
    admission=read_json(source/'grasp-admission.json');first=next(r for r in admission['rejected_candidates'] if r.get('joints'))
    result=[]
    for mode in MODES:
        if mode=='detector_full_open':
            pose=np.asarray(first['pose']);q=first['joints'];half=.04
        else:
            if mode=='frontal_full_open':jaw=up;approach=-normal;contact=centre;half=.04
            else:
                approach=unit(-normal-up)  # One declared 45-degree contact template.
                jaw=side if mode.startswith('tilted_parallel') else unit(normal-up)
                contact=centre.copy()
                if mode.startswith('tilted_parallel'):
                    contact-=normal*(extent[1]/2+pad_depth/2-.001)
                    contact-=up*(pad_height/2)
                    length=float(np.ptp(np.quantile(cloud@side,[.05,.95])))
                    half=float(np.clip(length/4,.008,.032))
                else:half=float(np.clip(extent[1]/2+.004,.008,.032))
            y=unit(np.cross(approach,jaw));jaw=unit(np.cross(y,approach))
            matrix=np.eye(4);matrix[:3,:3]=np.column_stack([jaw,y,approach]);matrix[:3,3]=contact-matrix[:3,:3]@pad_local
            pose=matrix_pose(matrix);q=None
        # Existing Panda formatter uses 0.2 normalized increments: each free
        # closing step changes the commanded half opening by 4 mm.
        jaw_steps=int(np.clip(np.rint((.04-half)/.004),0,9));actual_half=.04-.004*jaw_steps
        result.append({'mode':mode,'pose_world_xyzw':pose.tolist(),'saved_detector_joints':q,
            'jaw_closing_steps_before_approach':jaw_steps,'commanded_half_aperture_m':actual_half,
            'pinch_after_contact':not mode.startswith('tilted_parallel'),'pull_axis_world':normal.tolist(),
            'pull_distance_m':float(spec['target_progress']),'observed_handle_center_world_m':centre.tolist(),
            'observed_central_extents_m':extent.tolist(),'pad_center_tcp_m':pad_local.tolist(),
            'source':'measured RGB-D bar plus configured robot pads; no simulator object geometry'})
    return result


def episode(source,proposal,folder):
    folder.mkdir(parents=True,exist_ok=False);sim=ProfiledClient(folder/'simulator.log',fast=True)
    ik=None;events=[];phases=[];state=None;stage='reset';record={'mode':proposal['mode'],'status':'failed',
        'diagnostic_only':True,'planning_scene_collision_admission':False,'execution_veto_checks':0,
        'object_state_assigned':False,'assisted_attachment':False,'simulator_scene_geometry_used_for_policy':False,
        'api_calls':0,'initial_state':int(source.name.split('state')[-1]),'task':'open_bottom_drawer'}
    def issue(qs,label,gripper,*,dense=False,hold=20):
        nonlocal state
        refs=np.asarray(qs) if dense else schedule(np.asarray(qs),maximum_joint_speed=.30,endpoint_hold_steps=hold)
        if len(refs)>600-state['control_steps']:raise ValueError('Whole diagnostic segment exceeds 600-step episode budget')
        target=folder/f'path-{len(phases):02d}-{label}';target.mkdir();np.save(target/'references.npy',refs)
        state,done=execute_batch(sim,state,refs,gripper=gripper,event_sink=events.append)
        phases.append({'label':label,'gripper_command':gripper,**{k:v for k,v in done.items() if k!='events'}})
        write_json(folder/'phases.json',phases)
    try:
        old=read_json(source/'summary.json');state=sim.reset(old['official_task_name'],old['initial_state'],folder/'initial',suite_name=old['suite'])
        state=sim.action(state['metadata']['eef_pose_world_xyzw'],gripper=-1,max_steps=60,output_dir=folder/'opened')
        # Require the same calibrated robot start; source measurements are from
        # this official deterministic reset, not a hand-entered target pose.
        saved=read_json(source/'opened/metadata.json')
        if np.max(np.abs(np.asarray(state['metadata']['joint_positions'])-saved['joint_positions']))>2e-4:
            raise ValueError('Diagnostic robot reset differs from the retained measured scene')
        from try_rekep_bowl_v3 import read_observation_v3
        cameras,_=read_observation_v3(folder/'opened',state['metadata'])
        with np.load(source/'opened/obs.npz') as old_arrays:
            rgb_match=all(np.array_equal(cameras[n]['rgb'],old_arrays[n+'_rgb']) for n in ['agentview','frontview'])
        if not rgb_match:raise ValueError('Diagnostic images differ from the retained RGB-D scene')
        write_json(folder/'input-provenance.json',{'source':str(source),'both_rgb_views_identical':True,'new_api_calls':0})
        config=read_json(ROOT/'results/reliability-mechanisms-frozen-20261016/protocol.json')['configuration']
        robot=sim.robot_chain();write_json(folder/'robot-chain.json',robot)
        import vamp
        empty=vamp.Environment();ik=TaskIK(robot,config,lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True),output=folder,sim=sim)
        stage='robot_valid_ik_path';q0=np.asarray(state['metadata']['joint_positions']);pose=np.asarray(proposal['pose_world_xyzw'])
        if proposal['saved_detector_joints'] is not None:q=np.asarray(proposal['saved_detector_joints'])
        else:
            solutions=ik.solve_many(pose,q0,maximum=8)
            if not solutions:raise ValueError('Contact template has no robot-valid IK')
            q=np.asarray(solutions[0]['joints'])
        pre=pose.copy();pre[:3]-=.08*pose_matrix(pose)[:3,2]
        reverse=ik.path(interpolate_path([pose,pre]),q)
        if not reverse['accepted']:raise ValueError('No continuous IK path from contact to pregrasp')
        contact_path=np.vstack([q,[s['joints'] for s in reverse['solutions']]])[::-1]
        pull=pose.copy();pull[:3]+=proposal['pull_distance_m']*np.asarray(proposal['pull_axis_world'])
        continued=ik.path(interpolate_path([pose,pull]),q)
        if not continued['accepted']:raise ValueError('No continuous IK path for the intended pull')
        pulling=np.vstack([q,[s['joints'] for s in continued['solutions']]])
        approach=np.vstack([q0,contact_path])
        estimated=len(schedule(approach,maximum_joint_speed=.30,endpoint_hold_steps=40))+len(schedule(pulling,maximum_joint_speed=.30,endpoint_hold_steps=20))
        estimated+=proposal['jaw_closing_steps_before_approach']+10+30+12
        if estimated>540:raise ValueError('Complete diagnostic plan exceeds remaining episode budget')
        write_json(folder/'proposal.json',proposal);write_json(folder/'plan.json',{'estimated_reference_steps':estimated,
            'full_path_robot_valid_ik':True,'scene_collision_admission_disabled':True,'native_object_geometry_used':False})
        state=sim.enable_joint_tracking(kp=300.)
        if proposal['jaw_closing_steps_before_approach']:
            issue(np.repeat(q0[None],proposal['jaw_closing_steps_before_approach'],axis=0),'set_partial_opening',1,dense=True)
            issue(np.repeat(q0[None],10,axis=0),'settle_partial_opening',0,dense=True)
        record['measured_half_aperture_before_approach_m']=float(np.mean(np.abs(state['metadata']['gripper_qpos'])))
        stage='approach_and_contact';issue(approach,'approach_and_contact',0,hold=40)
        sim.observe(folder/'contact')
        stage='contact_mode';issue(np.repeat(q[None],30,axis=0),'pinch' if proposal['pinch_after_contact'] else 'hold_hook',1 if proposal['pinch_after_contact'] else 0,dense=True)
        if proposal['mode']=='tilted_parallel_feedback':
            # Start the motion from the achieved hand pose. A small downward
            # position bias supplies compliant contact preload through the
            # unchanged joint servo. Its 3 mm scale is the existing depth-grid
            # resolution, shared across cases; no native drawer state is read.
            stage='measured_hand_reference_update'
            measured=np.asarray(state['metadata']['eef_pose_world_xyzw'])
            reference=measured.copy();reference[2]-=config['scene_voxel_spacing_m']
            end=reference.copy();end[:3]+=proposal['pull_distance_m']*np.asarray(proposal['pull_axis_world'])
            continued=ik.path(interpolate_path([reference,end]),state['metadata']['joint_positions'])
            if not continued['accepted']:raise ValueError('No robot-valid path after measured hand reference update')
            pulling=np.vstack([state['metadata']['joint_positions'],[s['joints'] for s in continued['solutions']]])
            write_json(folder/'feedback-reference.json',{'measured_hand_pose':measured.tolist(),'updated_reference':reference.tolist(),
                'updated_goal':end.tolist(),'preload_m':config['scene_voxel_spacing_m'],
                'native_object_state_used':False,'update_after_completed_contact_segment':True,'execution_veto':False})
        stage='complete_pull';issue(pulling,'pull',1 if proposal['pinch_after_contact'] else 0,hold=20)
        final_q=np.asarray(pulling[-1]);issue(np.repeat(final_q[None],12,axis=0),'final_hold',1 if proposal['pinch_after_contact'] else 0,dense=True)
        sim.observe(folder/'final');record['status']='completed_all_commands'
    except Exception as error:
        record.update(failure_stage=stage,failure=str(error));(folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if ik:ik.close()
        sim.close();actions=read_json(folder/'initial/actions.json') if (folder/'initial/actions.json').exists() else []
        record.update(official_task_success=any(a['evaluation_task_success'] for a in actions),
            final_task_success=bool(actions and actions[-1]['evaluation_task_success']),control_steps=len(actions),
            all_issued_trajectories_completed=all(p['all_commands_issued'] for p in phases),
            completed_trajectories=len(phases),planned_reference_count=len(events))
        write_json(folder/'tracking.json',events);write_json(folder/'summary.json',record)
        record['storage']=compact_episode(folder);write_json(folder/'summary.json',record);compress_retained_observations(folder)
    return read_json(folder/'summary.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--state',type=int,default=5)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES);args=parser.parse_args()
    source=ROOT/'results/family-transfer-frozen-20261016'/f'open_bottom_drawer-state{args.state:03d}'
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False);choices=[p for p in proposals(source) if p['mode'] in args.modes]
    write_json(out/'manifest.json',{'episodes':[{'mode':p['mode'],'state':args.state} for p in choices],
        'development_diagnostics':True,'all_cases_in_denominator':True,'frozen_benchmark_unchanged':True,
        'planning_collision_checks_bypassed_for_diagnosis':True,'no_execution_vetoes':True,'no_simulator_geometry_for_proposal':True})
    write_json(out/'proposals.json',choices);before=hashlib.sha256((ROOT/'results/articulated-api-v1/ledger.json').read_bytes()).hexdigest()
    rows=[]
    for proposal in choices:
        print(json.dumps({'started':proposal['mode']}),flush=True)
        result=episode(source,proposal,out/proposal['mode']);rows.append(result)
        write_json(out/'summary.json',{'planned':len(choices),'completed':len(rows),'native_successes':sum(r['official_task_success'] for r in rows),'episodes':rows})
        print(json.dumps(result),flush=True)
    assert before==hashlib.sha256((ROOT/'results/articulated-api-v1/ledger.json').read_bytes()).hexdigest()


if __name__=='__main__':main()
