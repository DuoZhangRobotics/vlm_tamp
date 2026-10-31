"""Recheck retained observed scenes without simulation or new model calls."""
import argparse
from pathlib import Path
import time

import numpy as np

from vlm_tamp.contact_collision_evidence import current_collision_witness
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.current_robot_scene import clear_current_robot_scene


def check(source,events,output,plan_arcs=False):
    source=Path(source);rows=[];record=read_json(source/'attempt01/robot-chain.json');chain=RobotChain(record)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    ik=None
    try:
        for event in events:
            folder=source/'attempt01'/f'{event:03d}-arc'
            inp=read_json(folder/'planning-input.json');ref=read_json(folder/'scene-reference.json')
            old=read_json(folder/'result.json');policy=read_json(folder/'native-scene.json')
            q=np.asarray(inp['reference_joints']);meta=inp['metadata']
            robot=RobotVolumes(chain,ref['configured_robot_reference_volumes'],ref['configured_robot_reference_joints'])
            meta={**meta,'robot_collision_volumes_world':
                [v for v in robot.forward(q) if 'gripper' not in v['robot_geom_name']]+meta['robot_collision_volumes_world']}
            with np.load(ref['arrays_path']) as arrays:
                scene=Scene.from_points(arrays['static_points'],arrays['moving_reference'],arrays['contact_reference'],
                    ref['spec'],config=ref['configuration'],robot=robot,reference_volumes=ref['configured_robot_reference_volumes'])
            scene.skill_contact_surface=policy['skill_contact_surface']
            original_scene=scene
            scene=clear_current_robot_scene(scene,meta)
            phase=time.perf_counter()
            with NativeScene(scene,record,meta,q,inp['goal_angle_rad'],aligned=False,
                    grasp_constraint=old.get('grasp_constraint','point')) as native:
                actual=native.validate(q)
                row={'event':event,'original_start':old['start'],'repaired_start':actual,
                    'original_overlap_evidence':current_collision_witness(original_scene,meta,old['start']),
                    'current_robot_point_filter':native.record['current_robot_point_filter'],
                    'total_seconds':time.perf_counter()-phase,'physical_motion_issued':False,
                    'source_scene_arrays_before_current_robot_filter':ref['arrays_path'],'same_reference_joints':q.tolist()}
                if not actual['accepted']:row['remaining_rejection']=current_collision_witness(scene,meta,actual)
            if plan_arcs and actual['accepted']:
                from vlm_tamp.arc_execution import contact_arc
                from vlm_tamp.hinged_contact_tools import TaskIK
                import vamp
                options={**ref['configuration'],**inp['search_options']}
                if ik is None:
                    worker=Path(output).parent/'arc-ik';worker.mkdir(exist_ok=False)
                    empty=vamp.Environment();phase=time.perf_counter()
                    ik=TaskIK(record,options,lambda joints:vamp.panda.validate(np.asarray(joints).tolist(),empty,True),output=worker)
                    row['ik_worker_startup_seconds']=time.perf_counter()-phase
                destination=Path(output).parent/f'arc{event:03d}'
                phase=time.perf_counter()
                try:
                    path,report=contact_arc(scene,record,meta,q,inp['goal_angle_rad'],destination,options,ik=ik)
                except (ValueError,RuntimeError) as error:
                    report=read_json(destination/'result.json') if (destination/'result.json').exists() else {}
                    row['arc_error']=str(error)
                row['arc_replan']={k:report.get(k) for k in ('status','exact_solution','goal_generation_seconds',
                    'native_connection_seconds','total_seconds','failure')}
                row['arc_replan']['wall_seconds']=time.perf_counter()-phase
            rows.append(row);clear_native_cache();write_json(Path(output),{'rows':rows,'in_progress':True})
            print(f"Event {event}: {old['start']['accepted']} -> {actual['accepted']}; arc={row.get('arc_replan',{}).get('status','not requested')}",flush=True)
    finally:
        if ik is not None:ik.close()
    result={'source_run':str(source),'scope':'saved point clouds and measured robot configurations with current-robot filtering; admission and optional planning, no physical execution or task-success claim',
        'rows':rows,'all_saved_starts_admitted':all(r['repaired_start']['accepted'] for r in rows)}
    write_json(Path(output),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--events',type=int,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-arcs',action='store_true');args=parser.parse_args()
    check(args.source,args.events,args.output,args.plan_arcs)
