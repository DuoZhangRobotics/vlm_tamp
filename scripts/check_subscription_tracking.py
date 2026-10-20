"""Replay the pilot's failed geometry update from saved sensors, without motion."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from serve_subscription_robot import RobotService
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_storage import read_json


def main():
    out=ROOT/'results/subscription-agent-pilot-20261018'
    public=out/'agent-workspace/robot_data/attempt02'
    service=RobotService.__new__(RobotService)
    service.spec=read_json(out/'attempt02/023-grounding/model.json')
    service.spec['camera']='frontview'
    service.choices=read_json(out/'attempt02/023-grounding/agent-regions.json')
    service.config=read_json(ROOT/'configs/articulated_libero.json')
    service.config.update(complete_planar_panel=False,scene_voxel_spacing_m=.003,handle_association_m=.008)
    service.state={'metadata':read_json(out/'attempt02/030-observation/metadata.json'),'control_steps':260}
    service.ik=SimpleNamespace(chain=RobotChain(read_json(out/'attempt02/robot-chain.json')))
    service.cameras={}
    with np.load(public/'obs030/sensors.npz') as arrays:
        for name,camera in service.state['metadata']['cameras'].items():
            view=dict(camera)
            for key in ['rgb','depth','world_points','known_robot_depth_prior']:
                if name+'_'+key in arrays:view[key]=arrays[name+'_'+key]
            service.cameras[name]=view
    with np.load(public/'geometry023/points.npz') as arrays:
        service.cloud=arrays['contact'];moving=arrays['moving'];static=arrays['static']
    service.anchors=[np.asarray(service.spec['source_anchor_world_m'])]
    robot=RobotVolumes(service.ik.chain,service.state['metadata']['robot_collision_volumes_world'],
                       service.state['metadata']['joint_positions'])
    service.scene=Scene.from_points(static,moving,service.cloud,service.spec,config=service.config,robot=robot)
    service.mask=np.zeros((512,512),bool);service.applied_pose=np.eye(4);service.epoch=0
    service.tracking_result=read_json(out/'attempt02/031-tracking.json')
    service.episode=out/'tracking-interface-regression';service.episode.mkdir(exist_ok=True);service.calls=1
    service.export_geometry=lambda evidence=None: {'camera':service.spec['camera'],
        'contact_points':len(service.cloud),'moving_points':len(service.scene.moving_reference),
        'static_points':len(service.scene.static_points),'geometry_epoch':service.epoch}
    result=service.apply_tracking(hinge_world_m=service.spec['joint_origin_world_m'])
    result={'passed':True,'result':result,'physical_actions_executed':0}
    print(json.dumps(result))
    (service.episode/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
