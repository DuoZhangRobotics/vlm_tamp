"""Constraint/frame and non-oracle recovery checks for the new mechanism adapter."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vlm_tamp.mechanism_surface import normalize_sheet_choices,observed_control_rotation
from vlm_tamp.mcvamp_mechanism import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.geometry import pose_matrix,matrix_pose
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.task_ik_v5 import TaskIK

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/all-prior-tasks-frozen-v3-20261015'


def test_sheet_coordinates_require_a_unique_consistent_camera_offset():
    raw=json.loads((BASE/'close_top_drawer-state041/grounding/response.txt').read_text())
    choices,repair=normalize_sheet_choices(raw)
    assert repair['offset_xy']==[0,560]
    assert choices['grasp_pixel_xy']==[288,258]
    assert choices['contact_box_xyxy']==[250,247,325,269]
    assert normalize_sheet_choices(choices)==(choices,None)
    raw['grasp_pixel_xy']=[288,20]
    with pytest.raises(ValueError):normalize_sheet_choices(raw)


def test_angular_observation_recovers_sign_and_rejects_symmetric_geometry():
    rng=np.random.default_rng(9);points=rng.uniform([-.025,-.004,.01],[.028,.004,.035],(500,3))
    theta=np.deg2rad(-8);changed=Rotation.from_rotvec([0,0,theta]).apply(points)
    fit=observed_control_rotation(points,changed,[0,0,0],[0,0,1])
    assert fit['valid'] and abs(fit['angle_rad']-theta)<np.deg2rad(.5)
    angles=np.linspace(-np.pi,np.pi,1000,endpoint=False)
    symmetric=np.c_[.03*np.cos(angles),.03*np.sin(angles),np.zeros(len(angles))]
    assert not observed_control_rotation(symmetric,symmetric,[0,0,0],[0,0,1])['valid']


@pytest.mark.parametrize('family',['prismatic','revolute'])
def test_native_constraint_agrees_with_geometric_motion_and_projects(family):
    record=json.loads((BASE/'open_middle_drawer-state001/robot-chain.json').read_text())
    metadata=json.loads((BASE/'open_middle_drawer-state001/opened/metadata.json').read_text())
    chain=RobotChain(record);q=np.asarray(metadata['joint_positions']);pose=pose_matrix(chain.forward(q))
    axis=np.array([1.,0,0]) if family=='prismatic' else np.array([0.,0,1.])
    origin=pose[:3,3]+[.04,0,0]
    scene=SimpleNamespace(spec={'joint_family':family},origin=origin,axis=axis,
        static_points=np.empty((0,3)),moving_reference=np.empty((0,3)),contact_reference=np.array([pose[:3,3]]),
        selected_contact_mask=np.array([],bool))
    config={'ik_seed_count':4,'ik_maximum_evaluations':120,'ik_position_tolerance_m':1e-5,'ik_orientation_tolerance_rad':1e-4}
    ik=TaskIK(record,config)
    amount=.01 if family=='prismatic' else .08
    transform=np.eye(4)
    if family=='prismatic':transform[:3,3]=amount*axis
    else:
        transform[:3,:3]=Rotation.from_rotvec(amount*axis).as_matrix()
        transform[:3,3]=origin-transform[:3,:3]@origin
    target=ik.solve(matrix_pose(transform@pose),q)
    assert target is not None
    with NativeScene(scene,record,metadata,q,amount) as native:
        assert np.linalg.norm(native.residual(q))<1e-6
        endpoint,check=native.project(target['joints'])
        assert check['accepted']
        assert abs(native.progress(endpoint)-amount)<2e-4
        assert np.linalg.norm(native.residual(endpoint))<2e-6
        perturb=q.copy();perturb[0]+=.05
        assert np.linalg.norm(native.residual(perturb))>1e-4
        projected,check=native.project(perturb)
        assert check['accepted'] and np.linalg.norm(native.residual(projected))<2e-6
    clear_native_cache()


def test_translating_surface_remains_obstacle_to_palm_and_finger_contact_is_local():
    record=json.loads((BASE/'open_middle_drawer-state001/robot-chain.json').read_text())
    metadata=json.loads((BASE/'open_middle_drawer-state001/opened/metadata.json').read_text())
    chain=RobotChain(record);q=np.asarray(metadata['joint_positions']);pose=pose_matrix(chain.forward(q))
    volumes=metadata['robot_collision_volumes_world']
    palm=next(v for v in volumes if v['robot_geom_name']=='gripper0_hand_collision')
    pad=next(v for v in volumes if v['robot_geom_name'].endswith('finger1_pad_collision'))
    palm_point=np.asarray(palm['vertices_world']).mean(0);pad_point=np.asarray(pad['vertices_world']).mean(0)
    config={'ik_seed_count':4,'ik_maximum_evaluations':120,'ik_position_tolerance_m':1e-5,'ik_orientation_tolerance_rad':1e-4}
    ik=TaskIK(record,config);axis=np.array([1.,0,0]);target=pose.copy();target[:3,3]+=.10*axis
    solution=ik.solve(matrix_pose(target),q);assert solution is not None
    def scene(point,contact):
        return SimpleNamespace(spec={'joint_family':'prismatic'},origin=pose[:3,3],axis=axis,
            static_points=np.empty((0,3)),moving_reference=np.asarray([point]),contact_reference=np.asarray([point]),
            selected_contact_mask=np.array([contact]))
    with NativeScene(scene(palm_point,False),record,metadata,q,.10) as native:
        assert not native.validate(q)['accepted']
        projected,info=native.project(solution['joints']);assert info['accepted']
        assert not native.validate(projected)['accepted']
    with NativeScene(scene(pad_point,True),record,metadata,q,.10) as native:
        assert native.validate(q)['accepted']
        projected,info=native.project(solution['joints']);assert info['accepted']
        assert native.validate(projected)['accepted']
    # Permitting a selected contact point never exempts the palm at that point.
    with NativeScene(scene(palm_point,True),record,metadata,q,.10) as native:
        assert not native.validate(q)['accepted']
    clear_native_cache()


def test_repaired_rgb_choice_supplies_selected_camera_to_perception(tmp_path,monkeypatch):
    from vlm_tamp import mechanism_surface,cohort_execution
    resources=mechanism_surface.MechanismResources(tmp_path)
    resources.perception=object()
    checked=[]
    def prepare(perception,bundle,choices,output):
        assert bundle['main_camera']==choices['camera']
        assert bundle['meta']['main_camera']==choices['camera']
        checked.append(choices['camera']);return choices
    monkeypatch.setattr(mechanism_surface,'prepare_contact',prepare)
    monkeypatch.setattr(cohort_execution,'read_cohort_observation',lambda *args:({},{}))
    choices=resources.grounding('close_top_drawer',41,BASE/'close_top_drawer-state041/opened',
        {'instruction':'close the top drawer'},tmp_path/'grounding')
    assert choices['grasp_pixel_xy']==[288,258]
    assert checked==['frontview'] and resources.new_api_calls==0
    assert (tmp_path/'grounding/coordinate-repair.json').exists()
