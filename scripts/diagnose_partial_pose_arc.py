"""Saved RGB-D replay: reconcile handle roles and search a full free-wrist arc."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from try_rekep_bowl_v3 import read_observation_v3
from run_coupled_door import fresh_mask,observed_angle
from vlm_tamp.arc_contact_scene import ArcContactScene
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_storage import read_json,write_json,preserve_scene
from vlm_tamp.mcvamp_microwave import NativeScene,clear_native_cache
from vlm_tamp.partial_pose_arc import full_arc_plan
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain


def load_scene(source):
    inputs=read_json(source/'plan-00/horizon-1/planning-input.json')
    spec=inputs['task_spec'];record=read_json(source/'robot-chain.json')
    meta=read_json(source/'closed/metadata.json');cameras,_=read_observation_v3(source/'closed',meta)
    before=read_json(source/'scene-reference.json');arrays=np.load(before['arrays_path'])
    previous=Scene.from_points(arrays['static_points'],arrays['moving_reference'],arrays['contact_reference'],before['spec'])
    current=np.load(read_json(source/'plan-00/horizon-1/scene-reference.json')['arrays_path'])
    cloud=current['contact_reference']
    angle=observed_angle(np.asarray(spec['joint_origin_world_m']),np.asarray(spec['joint_axis_world']),
                         np.median(previous.contact_reference,axis=0),np.median(cloud,axis=0))
    mask,choices=fresh_mask(cameras,spec,cloud,previous,angle)
    config=read_json(source.parent/'protocol.json')['config']
    # Match the existing 8 mm selected-surface fusion tolerance. The former
    # 6 mm nearest-sample association left a thin observed handle shell static.
    config['handle_association_m']=.008
    robot=RobotVolumes(RobotChain(record),meta['robot_collision_volumes_world'],meta['joint_positions'])
    scene=ArcContactScene(cameras,meta,spec,cloud,mask,choices,config,robot=robot)
    scene.retain_static(previous)
    return scene,record,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--proposals',type=int,default=160)
    args=parser.parse_args();output=ROOT/args.output;output.mkdir(exist_ok=False)
    names=['scripts/diagnose_partial_pose_arc.py','src/vlm_tamp/partial_pose_arc.py',
           'src/vlm_tamp/arc_contact_scene.py','src/vlm_tamp/mcvamp_microwave.py',
           'src/vlm_tamp/native/mcvamp_microwave.cc']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    for name in names:
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    write_json(output/'protocol.json',{'source_hashes':hashes,'arguments':vars(args),
        'goal_angle_deg':90.,'physical_execution':False,'new_api_calls':0,
        'frozen_before_search':True,'simulator_scene_geometry_used_by_policy':False})
    source=ROOT/'results/mcvamp-convex-frozen-grasp-state11-v2-20261015/open_microwave-state011'
    scene,record,meta=load_scene(source)
    write_json(output/'robot-chain.json',record);preserve_scene(scene,output)
    write_json(output/'planning-input.json',{'metadata':{key:meta[key] for key in
        ['joint_positions','eef_pose_world_xyzw','robot_collision_volumes_world']},
        'task_spec':scene.spec,'reference_joints':meta['joint_positions'],'goal_angle_rad':np.pi/2})
    before=time.perf_counter()
    with NativeScene(scene,record,meta,meta['joint_positions'],np.pi/2,aligned=False,output=output/'native') as context:
        path,result=full_arc_plan(context,np.pi/2,seconds=args.seconds,proposals=args.proposals)
        result.update(source=str(source),physical_execution=False,role_reconciliation=scene.role_reconciliation,
            start_scene=context.validate(meta['joint_positions']),new_api_calls=0)
        if len(path):
            np.save(output/'path.npy',path);np.save(output/'progress.npy',[context.progress(q) for q in path])
    clear_native_cache();result['wall_seconds']=time.perf_counter()-before
    result['source_hashes_unchanged']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                                         for name,digest in hashes.items())
    write_json(output/'summary.json',result)
    print({k:v for k,v in result.items() if k not in ['goal_proposals','global_queries','continuation_layers']},flush=True)


if __name__=='__main__':main()
