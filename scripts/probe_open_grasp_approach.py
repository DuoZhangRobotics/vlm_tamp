"""Replay one saved open-hand RRTC query without simulator or model calls."""
import argparse
from pathlib import Path
import numpy as np
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.hinged_contact_tools import tool_configuration
from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,nargs='+',default=[3000]);parser.add_argument('--seconds',type=float,default=2.)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    p=args.source;meta=read_json(p/'attempt01/001-observation/metadata.json')
    record=read_json(p/'attempt01/robot-chain.json');spec=read_json(p/'attempt01/002-grounding/model.json')
    goal=read_json(p/'attempt01/003-grasp/selection/grasp-attempts.json')[0]['joints']
    chain=RobotChain(record);robot=RobotVolumes(chain,meta['robot_collision_volumes_world'],meta['joint_positions'])
    with np.load(p/'workspace/robot_data/attempt01/geometry002/points.npz') as a:
        scene=Scene.from_points(a['static'],a['moving'],a['contact'],spec,
            config=tool_configuration(),robot=robot)
    scene.skill_contact_surface='fingertips';results=[]
    write_json(args.output/'request.json',{'source':str(p.resolve()),'start':meta['joint_positions'],'goal':goal,
        'maximum_seconds_per_query':args.seconds,'iteration_limits':args.iterations,'physical_motion_issued':False})
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False,output=args.output/'inputs') as context:
        for limit in args.iterations:
            path,result=context.plan(meta['joint_positions'],goal,constrained=False,seconds=args.seconds,iterations=limit)
            results.append(result);np.save(args.output/f'path-{limit}.npy',path)
            print({k:result[k] for k in ['status','solve_seconds','iterations','state_checks']},flush=True)
    write_json(args.output/'results.json',results);clear_native_cache()


if __name__=='__main__':main()
