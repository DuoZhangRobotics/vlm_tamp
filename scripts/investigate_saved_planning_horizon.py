"""Matched, planning-only horizon sweep from a saved post-grasp query."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.mcvamp_microwave import NativeScene,clear_native_cache
from vlm_tamp.partial_pose_arc import full_arc_plan
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.timed_execution import schedule


def reconstruct(query):
    inputs=read_json(query/'planning-input.json');saved=read_json(query/'scene-reference.json')
    episode=next(p for p in query.parents if (p/'robot-chain.json').exists())
    record=read_json(episode/'robot-chain.json');chain=RobotChain(record)
    robot_reference=saved.get('configured_robot_reference_joints')
    inferred=robot_reference is None
    if inferred:robot_reference=record['reference_joints']
    robot=RobotVolumes(chain,saved['configured_robot_reference_volumes'],robot_reference)
    start=np.asarray(inputs['reference_joints'])
    predicted={v['robot_geom_name']:v for v in robot.forward(start)}
    palm_errors=[]
    for actual in inputs['metadata']['robot_collision_volumes_world']:
        if actual['robot_geom_name'].endswith('hand_collision'):
            expected=predicted[actual['robot_geom_name']]
            palm_errors.append(float(np.max(np.abs(np.asarray(expected['vertices_world'])-actual['vertices_world']))))
    if not palm_errors or max(palm_errors)>1e-5:
        raise ValueError('Cannot infer the legacy scene robot reference: palm reconstruction mismatch')
    array_path=Path(saved['arrays_path'])
    if hashlib.sha256(array_path.read_bytes()).hexdigest()!=saved['arrays_sha256']:
        raise ValueError('Saved scene arrays changed')
    with np.load(array_path) as arrays:
        scene=Scene.from_points(arrays['static_points'],arrays['moving_reference'],arrays['contact_reference'],
            saved['spec'],config=saved['configuration'],robot=robot,reference_volumes=saved['configured_robot_reference_volumes'])
    options=dict(saved['configuration']);options.update(inputs.get('search_options',{}))
    if 'search_options' not in inputs:
        event_id=int(query.name.split('-')[0]);run=episode.parent
        event=next(json.loads(line) for line in (run/'tool-events.jsonl').read_text().splitlines()
                   if json.loads(line)['sequence']==event_id)
        if event['request']['operation']!='plan_arc':raise ValueError('Legacy query has no matching arc request')
        options['hinge_refinement_accepted']=bool(event['request']['arguments'].get('refined',False))
    evidence={'legacy_reference_inferred':inferred,'palm_vertex_max_error_m':max(palm_errors),
        'scene_arrays_sha256':saved['arrays_sha256'],'scene_arrays_path':str(array_path),
        'original_request_degrees':float(np.degrees(inputs['goal_angle_rad'])),
        'search_options_recovered':{k:options.get(k) for k in ['motion_policy','hinge_refinement_accepted','arc_search_seconds','arc_goal_proposals','arc_seed']}}
    return scene,record,inputs,start,options,evidence


def run_query(scene,record,inputs,start,options,degrees,seconds,proposals,seed,folder):
    folder.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    with NativeScene(scene,record,inputs['metadata'],start,np.radians(degrees),aligned=False) as context:
        context.motion_options=dict(options)
        path,result=full_arc_plan(context,np.radians(degrees),seconds=seconds,proposals=proposals,seed=seed)
        result['exact_solution']=bool(len(path)>1 and result['status']=='solved')
        row={'goal_degrees':degrees,'seconds_budget':seconds,'goal_proposals_budget':proposals,
            'goal_sampler_seed':seed,'motion_policy':options.get('motion_policy'),
            'refined':bool(options.get('hinge_refinement_accepted')),'solved':result['exact_solution'],
            'status':result['status'],'wall_seconds':time.perf_counter()-began,
            'admissible_goals':result.get('admissible_goals',0),
            'global_queries':len(result.get('global_queries',[])),
            'continuation_layers':len(result.get('continuation_layers',[])),
            'path_states':len(path),'physical_execution':False}
        if len(path):
            progress=np.asarray([context.progress(q) for q in path])
            np.save(folder/'path.npy',path);np.save(folder/'door-progress.npy',progress)
            row.update(terminal_degrees=float(np.degrees(progress[-1])),
                sampled_path_valid=all(context.validate(q)['accepted'] for q in path),
                scheduled_control_steps=len(schedule(path,maximum_joint_speed=.35,endpoint_hold_steps=6)))
        else:
            row['last_attempted_continuation_degrees']=float(np.degrees(result['last_progress_rad'])) if 'last_progress_rad' in result else None
        write_json(folder/'result.json',result);write_json(folder/'native-scene.json',context.record)
        write_json(folder/'query.json',row)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query',required=True);p.add_argument('--output',required=True)
    p.add_argument('--angles',type=float,nargs='+',default=[15,20,30,45,60,90])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--proposals',type=int,default=32)
    p.add_argument('--full-wrist-policy',action='store_true')
    args=p.parse_args();query=(ROOT/args.query).resolve();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'results') or out.exists():raise ValueError('Use a new results directory')
    out.mkdir(parents=True)
    scene,record,inputs,start,options,evidence=reconstruct(query)
    if args.full_wrist_policy:options['motion_policy']='full'
    write_json(out/'protocol.json',{'source_query':str(query),'reconstruction':evidence,
        'angles_degrees':args.angles,'goal_sampler_seeds':args.seeds,'seconds_budget':args.seconds,
        'goal_proposals':args.proposals,'motion_policy':options.get('motion_policy'),
        'refined':options.get('hinge_refinement_accepted'),'collision_rules_changed':False,
        'robot_start_identical_for_every_query':True,'cloud_identical_for_every_query':True,
        'physical_execution':False,'model_api_calls':0,'interpretation':'Planner component evidence, not physical task success or proof of infeasibility.'})
    original=read_json(query/'native-scene.json')
    with NativeScene(scene,record,inputs['metadata'],start,inputs['goal_angle_rad'],aligned=False) as check:
        keys=['static_points','moving_noncontact_points','selected_contact_points','point_radius_m']
        matches={k:check.record[k]==original[k] for k in keys}
        if not all(matches.values()):raise ValueError('Reconstructed collision context does not match recorded context')
        if not check.validate(start)['accepted']:raise ValueError('Recorded start no longer valid')
        old_path=np.load(query/'path.npy') if (query/'path.npy').exists() else None
        old_valid=all(check.validate(q)['accepted'] for q in old_path) if old_path is not None else None
        if old_valid is False:raise ValueError('Recorded successful path no longer valid')
        write_json(out/'reconstruction-check.json',{'passed':True,'context_fields_match':matches,'original_path_valid':old_valid,**evidence})
    rows=[]
    try:
        for seed in args.seeds:
            for angle in args.angles:
                folder=out/f'angle-{angle:g}-seed-{seed}'
                row=run_query(scene,record,inputs,start,options,angle,args.seconds,args.proposals,seed,folder)
                rows.append(row);write_json(out/'summary.json',{'completed':len(rows),'planned':len(args.seeds)*len(args.angles),'rows':rows})
                print(json.dumps(row),flush=True)
    finally:clear_native_cache()


if __name__=='__main__':main()
