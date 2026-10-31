"""Saved-state regressions plus actual cuRobo tests of empty cleaned roles."""
import argparse
from pathlib import Path
import time

import numpy as np

from check_saved_contact_starts import check
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_storage import preserve_scene,read_json,write_json
from vlm_tamp.current_robot_scene import clear_current_robot_scene
from vlm_tamp.curobo_v2_arc import CuroboV2ArcProposer
from vlm_tamp.hinged_contact_tools import TaskIK
from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain


def empty_world_checks(source,event,output):
    import vamp
    folder=Path(source)/'attempt01'/f'{event:03d}-arc'
    inp=read_json(folder/'planning-input.json');ref=read_json(folder/'scene-reference.json')
    record=read_json(Path(source)/'attempt01/robot-chain.json');chain=RobotChain(record)
    q=np.asarray(inp['reference_joints']);meta=inp['metadata']
    robot=RobotVolumes(chain,ref['configured_robot_reference_volumes'],ref['configured_robot_reference_joints'])
    meta={**meta,'robot_collision_volumes_world':
        [v for v in robot.forward(q) if 'gripper' not in v['robot_geom_name']]+meta['robot_collision_volumes_world']}
    palm=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
    point=np.asarray(palm['vertices_world']).mean(0,keepdims=True)
    config={**ref['configuration'],**inp['search_options']}
    raw=Scene.from_points(point,point,point,ref['spec'],config=config,robot=robot,
        reference_volumes=ref['configured_robot_reference_volumes'])
    output=Path(output);output.mkdir();empty=vamp.Environment()
    ik=TaskIK(record,config,lambda x:vamp.panda.validate(np.asarray(x).tolist(),empty,True),output=output)
    reports=[]
    try:
        for mode in ['local_continuation','global']:
            scene=clear_current_robot_scene(raw,meta)
            scene.config={**scene.config,'curobo_v2_arc_solver_mode':mode}
            destination=output/mode;destination.mkdir();preserve_scene(scene,destination)
            with NativeScene(scene,record,meta,q,np.radians(1),aligned=False) as native:
                assert native.validate(q)['accepted']
                with CuroboV2ArcProposer(ik,native,destination,np.radians(1)) as proposer:
                    assert proposer.setup['observed_points']=={'stationary':0,'moving':0}
                    goals,checks=proposer.candidates(q,0.,maximum=1,deadline=time.perf_counter()+10.)
                    reports.append({'mode':mode,'observed_points':proposer.setup['observed_points'],
                        'contact_reference_points_retained_as_metadata':len(scene.contact_reference),
                        'native_start_accepted':True,'ik_query_completed':True,'admitted_candidates':len(goals),
                        'physical_motion_issued':False})
            clear_native_cache()
        write_json(output/'result.json',{'scope':'synthetic empty collision-role regression using recorded robot geometry',
            'raw_contact_reference_not_reintroduced_as_obstacle':True,'queries':reports})
    finally:ik.close();clear_native_cache()
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--events',type=int,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--empty-world-checks',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    check(args.source,args.events,args.output/'saved-checks.json',plan_arcs=True)
    if args.empty_world_checks:print(empty_world_checks(args.source,args.events[0],args.output/'empty-world'))
