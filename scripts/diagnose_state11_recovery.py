"""Audit the stopped grasp and test RGB-D landmark hinge recovery offline."""
from pathlib import Path
import numpy as np

from try_rekep_bowl_v3 import read_observation_v3
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.rigid_hinge_tracking import RGBDHingeTracker

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'results/state11-recovery-diagnosis-20261015';output.mkdir(exist_ok=True)
    case=ROOT/'results/short-planned-hinge-state11-20261015/open_microwave-state011'
    reference=ROOT/'results/mcvamp-convex-frozen-grasp-state11-v2-20261015/open_microwave-state011/closed'
    metadata=read_json(reference/'metadata.json');before,_=read_observation_v3(reference,metadata)
    inputs=read_json(case/'plan-00/horizon-0/planning-input.json')
    if np.max(np.abs(np.asarray(metadata['joint_positions'])-inputs['reference_joints']))>1e-7:
        raise ValueError('Reference observation is not the same physical closed-grasp state')
    arrays=np.load(read_json(case/'plan-00/horizon-0/scene-reference.json')['arrays_path'])
    tracker=RGBDHingeTracker(before,arrays['moving_reference'],prior=inputs['task_spec']['joint_origin_world_m'],
        reference_step=239,output=output/'landmark-reference.npz')
    target=case/'observation-02';meta=read_json(target/'metadata.json');after,_=read_observation_v3(target,meta)
    result=tracker.observe(after,control_step=303,output=output/'landmark-matches.npz')
    write_json(output/'rigid-hinge.json',result)
    print(result)


if __name__=='__main__':main()
