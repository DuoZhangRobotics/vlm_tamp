"""Audit all learned hand poses against the observed scene without solving IK."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.hinged_contact_tools import tool_configuration
from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.graspgen_selection import learned_poses


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=args.source;args.output.mkdir(exist_ok=False)
    meta=read_json(p/'attempt01/001-observation/metadata.json');record=read_json(p/'attempt01/robot-chain.json')
    spec=read_json(p/'attempt01/002-grounding/model.json');chain=RobotChain(record)
    robot=RobotVolumes(chain,meta['robot_collision_volumes_world'],meta['joint_positions'])
    with np.load(p/'workspace/robot_data/attempt01/geometry002/points.npz') as a:
        scene=Scene.from_points(a['static'],a['moving'],a['contact'],spec,config=tool_configuration(),robot=robot)
    scene.skill_contact_surface='fingertips'
    file=p/'attempt01/003-grasp/generation/grasp_candidates.npz'
    with np.load(file) as a:generated={key:a[key] for key in a.files}
    attempts=read_json(p/'attempt01/003-grasp/selection/grasp-attempts.json');rows=[]
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False,output=args.output/'inputs') as context:
        for rank,row in enumerate(learned_poses(generated,meta['robot_calibration'])):
            began=time.perf_counter();result=context.validate_hand_pose(row['pose']);elapsed=time.perf_counter()-began
            saved=next((a for a in attempts if a['graspgen_index']==row['graspgen_index']),{})
            recorded=saved.get('ik_endpoint_checks',[])
            # Forward kinematics of a saved solution is not a fresh IK solve.
            at_recorded=context.validate_hand_pose(chain.forward(recorded[0]['joints'])) if recorded else None
            item={'rank':rank+1,'model_index':row['graspgen_index'],'score':row['graspgen_score'],
                'seconds':elapsed,'hand_at_requested_pose':result,'hand_at_saved_ik_pose':at_recorded,
                'previous_full_robot_first_blocking_part':recorded[0]['open_aperture']['blocking_robot_part'] if recorded else None}
            rows.append(item)
            print({'rank':rank+1,'hand_clear':result['accepted'],'part':result['blocking_robot_part'],
                'milliseconds':elapsed*1000},flush=True)
    write_json(args.output/'results.json',{'source':str(p.resolve()),'poses_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
        'ik_queries':0,'physical_motion_issued':False,'rows':rows,
        'nominal_hand_filter_seconds':sum(r['seconds'] for r in rows)})
    clear_native_cache()


if __name__=='__main__':main()
