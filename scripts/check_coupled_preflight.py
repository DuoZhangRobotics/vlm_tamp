"""Whole-motion feasibility on preserved RGB-D; no robot execution or API requests."""

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from try_rekep_bowl_v3 import read_observation_v3,save
from vlm_tamp.handle_contact import ground_handle
from vlm_tamp.coupled_pipeline import terminal_angle,complete_motion_selection,supported_handle_anchors
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['open_microwave','close_microwave'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--contact-model',choices=['rigid','handle_point','point_contact','jaw_contact'],default='point_contact')
    parser.add_argument('--complete-panel',action='store_true')
    args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    version='pilot-v20' if args.task=='open_microwave' else 'pilot-v19'
    source=ROOT/'results/handle-contact-development'/version/(args.task+'-state001')
    config=json.loads((ROOT/'configs/articulated_libero.json').read_text())
    config['scene_voxel_spacing_m']=.003
    config['observed_clearance_margin_m']=.003
    config['noncontact_tool_clearance_m']=.002
    if args.complete_panel:
        config.update(complete_planar_panel=True,panel_completion_spacing_m=.005,
            allow_palm_panel_brace=False,allow_palm_handle_contact=False,
            allow_pad_handle_contact=True,allow_finger_handle_contact=True,
            maximum_target_contact_penetration_m=.003)
    cameras,metadata=read_observation_v3(source/'opened',source/'opened/metadata.json')
    choices=json.loads((source/'grounding/grounding.json').read_text())['choices']
    spec=json.loads((source/'task-spec.json').read_text())
    mechanism,cloud,mask,anchors,evidence=ground_handle(cameras,choices,spec,metadata)
    spec=mechanism.spec
    anchors=[np.r_[anchors[0][:2],np.quantile(cloud[:,2],f)] for f in [.20,.40,.60,.80,.93]]
    if args.contact_model=='jaw_contact':
        anchors,support=supported_handle_anchors(anchors,cloud,metadata)
        save(output/'pad-support.json',support)
    angle,goal_evidence=terminal_angle(cameras,metadata,spec,cloud)
    save(output/'goal.json',{'angle_rad':angle,**goal_evidence});save(output/'geometry.json',evidence)
    record=json.loads((source/'robot-chain.json').read_text());chain=RobotChain(record)
    opened_robot=RobotVolumes(chain,metadata['robot_collision_volumes_world'],metadata['joint_positions'])
    closed=json.loads((source/'closed/metadata.json').read_text())
    closed_robot=RobotVolumes(chain,closed['robot_collision_volumes_world'],closed['joint_positions'])
    open_scene=Scene(cameras,metadata,spec,cloud,mask,choices,config,robot=opened_robot)
    closed_scene=Scene(cameras,metadata,spec,cloud,mask,choices,config,robot=closed_robot)
    save(output/'scene.json',closed_scene.record())
    reference=ROOT/'results/articulated-physics-pilot'/(args.task+'-state001')/'grasps/grasp_candidates.npz'
    with np.load(reference) as data:
        generated={k:data[k].copy() for k in ['poses_world_base','scores','retained_indices']}
    import vamp
    empty=vamp.Environment();ik=None
    try:
        ik=TaskIK(record,config,lambda q:vamp.panda.validate(np.asarray(q).tolist(),empty,True),output=output)
        chosen,problem,path=complete_motion_selection(generated,anchors,metadata,ik,open_scene,closed_scene,
            spec,angle,output,save,contact_model=args.contact_model)
        np.save(output/'complete-path.npy',path)
        save(output/'chosen-grasp.json',{k:v for k,v in chosen.items() if k!='pose'})
        result={'status':'complete_constrained_path_found','states':len(path),'goal_angle_rad':angle,
                'native_task_success_measured':False,'physical_execution':False,'new_api_calls':0,
                'contact_model':args.contact_model}
    except Exception as error:
        result={'status':'preflight_failed','error':str(error),'new_api_calls':0,'physical_execution':False,
                'traceback':traceback.format_exc()}
    finally:
        if ik:ik.close()
    save(output/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
