"""Replay one recorded contact-planning query with old and repaired deadlines.

Only recorded RGB-D, robot geometry and recorded public motion fits are used.
No simulator or model agent is started and no generated trajectory is executed.
"""
import argparse
import copy
from collections import Counter
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from vlm_tamp.contact_skill_service import planning_stage
from vlm_tamp.contact_skills import Budget,SkillStop,build_contact_scene,plan_contact_stroke,refresh_collision_observations,unit
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.hinged_contact_tools import TaskIK,tool_configuration
from vlm_tamp.mcvamp_contact_numerical import clear_native_cache
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain

ROOT=Path(__file__).resolve().parents[1]


def observation(source,number,substitutions):
    root=source/f'agent-workspace/robot_data/attempt01/obs{number:03d}'
    metadata=read_json(source/f'attempt01/{number:03d}-observation/metadata.json')
    if not (root/'sensors.npz').exists():
        # Completed runs retain public requested frames, not every internal
        # feedback array. A later observe at the identical physical step can
        # supply current depth, but is explicitly a reconstruction.
        candidates=[]
        for arrays in root.parent.glob('obs*/sensors.npz'):
            robot=read_json(arrays.parent/'robot.json')
            if (robot['control_steps']==metadata['control_steps']
                    and np.allclose(robot['joint_positions'],metadata['joint_positions'],atol=1e-10,rtol=0)):
                candidates.append(arrays.parent)
        if not candidates:raise FileNotFoundError(f'No retained depth at control {metadata["control_steps"]}')
        replacement=sorted(candidates)[0]
        substitutions.append({'requested_observation':number,'used_observation':replacement.name,
            'same_control_step':metadata['control_steps'],'same_joint_state':True,
            'original_depth_arrays_unavailable':True})
        root=replacement
    cameras={}
    with np.load(root/'sensors.npz') as arrays:
        for name,calibration in read_json(root/'calibration.json').items():
            if name not in metadata['cameras']:continue
            cameras[name]={**calibration,'robot_depth_prior':metadata['cameras'][name]['robot_depth_prior'],
                **{key:arrays[f'{name}_{key}'] for key in ('rgb','depth','world_points','known_robot_depth_prior')}}
    return cameras,metadata


def run(source,event_number,geometry_number,output):
    import json,vamp
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    event=next(json.loads(line) for line in (source/'tool-events.jsonl').read_text().splitlines()
        if json.loads(line)['sequence']==event_number)
    args=event['request']['arguments'];saved=event['response']['result']
    record=read_json(source/'attempt01/robot-chain.json');chain=RobotChain(record)
    config=tool_configuration(read_json(ROOT/'configs/articulated_libero.json'))
    config.update(curobo_backend='v2',arc_goal_backend='curobo_v2')
    substitutions=[];missing_updates=[]
    cameras,meta=observation(source,args['observation_id'],substitutions)
    geometry=source/f'agent-workspace/robot_data/attempt01/geometry{geometry_number:03d}'
    with np.load(geometry/'points.npz') as points:
        joint=Scene.from_points(points['static'],points['moving'],points['contact'],read_json(geometry/'model.json'),
            config=config,robot=RobotVolumes(chain,meta['robot_collision_volumes_world'],meta['joint_positions']),
            reference_volumes=meta['robot_collision_volumes_world'])
    scene,_,_=build_contact_scene(cameras,meta,chain,args['camera'],args['region_xyxy'],args.get('target_region_xyxy'),
        args['contact_surface'],args.get('direction_world'),args.get('joint_degrees'),joint,skill_kind='push')
    point=np.asarray(saved['contact']['contact_world_m']);local=np.asarray(saved['contact']['contact_local_m'])
    for stage in saved['stages']:
        fit=stage['object_motion']
        if not fit['accepted']:raise ValueError('Only accepted recorded public pose fits can reconstruct this query')
        transform=np.asarray(fit['transform']);point=transform_points([point],transform)[0]
        scene=copy.copy(scene);scene.moving_reference=transform_points(scene.moving_reference,transform)
        scene.contact_reference=transform_points(scene.contact_reference,transform)
        try:cameras,meta=observation(source,stage['artifact_id'],substitutions)
        except FileNotFoundError:
            missing_updates.append(stage['artifact_id'])
            if stage is saved['stages'][-1]:raise
        else:scene,_=refresh_collision_observations(scene,cameras,meta)
    scene.skill_collision_policy='veto'
    length=float(args.get('stroke_length_m',.01));sign=np.sign(args['joint_degrees'])
    angle=sign*min(abs(np.radians(args['joint_degrees'])),length/max(np.linalg.norm(np.cross(scene.axis,point-scene.origin)),.01),np.pi/2)
    delta=scene.origin+Rotation.from_rotvec(scene.axis*angle).as_matrix()@(point-scene.origin)-point
    empty=vamp.Environment();phase=time.perf_counter()
    ik=TaskIK(record,config,lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True),output=output)
    report={'scope':'same recorded public observations and accepted fits; no physical execution or task-success claim',
        'source':str(source),'event':event_number,'ik_worker_startup_seconds':time.perf_counter()-phase,'queries':[],
        'observation_substitutions':substitutions,'intermediate_occupancy_updates_unavailable':missing_updates,
        'exact_original_collision_scene_reproduced':not (substitutions or missing_updates),
        'comparison_scope':'both deadline settings receive the same reconstructed observed scene at the recorded final robot state'}
    try:
        # Warm the worker equally before comparing the deadlines.
        ik.solve_many(chain.forward(meta['joint_positions']),meta['joint_positions'],maximum=8)
        for label,seconds in [('old_implicit',args['budget_seconds']/24),('independent_default',5.)]:
            clear_native_cache();row={'policy':label,'allowance_seconds':seconds};timing=Counter();started=time.perf_counter()
            try:
                with planning_stage(Budget(args['budget_seconds']),seconds,'contact_stroke',row,timing,'stroke_planning_seconds') as (budget,nested):
                    path,search=plan_contact_stroke(ik,scene,record,meta,local,delta,budget,angle=angle,timing=nested)
                row.update(status='planned',path_waypoints=len(path),posture_search=search)
                np.save(output/f'{label}-path.npy',path)
            except SkillStop as error:row.update(status=error.reason,diagnostic=error.evidence)
            row['wall_seconds']=time.perf_counter()-started;report['queries'].append(row)
            write_json(output/'result.json',report);print(f"{label}: {row['status']} in {row['wall_seconds']:.3f}s",flush=True)
    finally:ik.close();clear_native_cache()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--event',type=int,required=True);parser.add_argument('--geometry',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.source,args.event,args.geometry,args.output)
