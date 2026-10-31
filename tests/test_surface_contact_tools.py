from pathlib import Path
import numpy as np
import pytest

from vlm_tamp.surface_contact_tools import contact_snapshot
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.rekep_libero.scene_state_v7 import RENDER_PRIOR_SOURCE

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def surface_scene():
    folder=ROOT/'results/prepared-final-all-tasks-20261020/open_microwave-state001'
    if not (folder/'opened/metadata.json').exists():pytest.skip('Calibrated robot fixture unavailable')
    metadata=read_json(folder/'opened/metadata.json');chain=RobotChain(read_json(folder/'robot-chain.json'))
    yy,xx=np.indices((40,40))
    points=np.stack([.60+xx*.003,np.full((40,40),.70),1.6+yy*.003],axis=-1)
    camera=np.eye(4);camera[:3,3]=[.65,.3,1.65]
    view={'world_points':points,'depth':np.ones((40,40)), 'camera_to_world':camera,
        'known_robot_depth_prior':np.zeros((40,40)),
        'robot_depth_prior':{'source':RENDER_PRIOR_SOURCE,'scene_geometry_used':False,
            'units':'meters','depth_convention':'optical-axis z'}}
    scene,record=contact_snapshot({'test':view},metadata,'test',[12,12,28,28],chain,radius_m=.015)
    return scene,record,metadata,read_json(folder/'robot-chain.json')


def test_surface_contact_preserves_every_other_observed_point_and_needs_no_handle(surface_scene):
    scene,record,_,_=surface_scene
    assert len(scene.static_points)>len(scene.contact_reference)>0
    assert record['grasp_or_attachment_required'] is False
    assert record['palm_arm_and_other_surface_checks_retained']
    assert not record['object_motion_predicted'] and not record['simulator_object_geometry_used']
    center=np.asarray(record['point_world_m'])
    assert np.linalg.norm(scene.contact_reference-center,axis=1).max()<=.0186
    assert len(record['configured_pad_centers_in_tcp'])==2
    assert np.allclose(np.abs(record['normal_toward_camera_world']),[0,1,0])


def test_generic_surface_scene_can_use_native_free_motion_planning(surface_scene,tmp_path):
    from types import SimpleNamespace
    from vlm_tamp.surface_contact_tools import plan_pose
    from vlm_tamp.hinged_contact_tools import clear_native_cache
    scene,_,metadata,chain_record=surface_scene
    ik=SimpleNamespace(solve_many=lambda *a,**k:[{'joints':metadata['joint_positions']}])
    path,result=plan_pose(ik,scene,chain_record,metadata,
        metadata['eef_pose_world_xyzw'],tmp_path/'plan')
    assert result['exact_solution'] and len(path)>=2
    assert not result['grasp_required'] and not result['articulation_constraint_used']
    clear_native_cache()


def test_surface_scene_cannot_silently_use_an_old_snapshot_after_motion():
    from serve_subscription_robot import RobotService
    robot=RobotService.__new__(RobotService)
    robot.contact_scene_kind='surface';robot.surface_reference_step=100
    robot.state={'control_steps':101}
    with pytest.raises(ValueError,match='before the last motion'):robot._check_surface_freshness()
    robot.state['control_steps']=100
    robot._check_surface_freshness()


def test_contact_point_conversion_uses_calibrated_offset_under_rotation():
    from scipy.spatial.transform import Rotation
    from vlm_tamp.surface_contact_tools import contact_pose_candidates
    from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points
    original=[.2,-.1,1.,*Rotation.from_euler('z',35,degrees=True).as_quat()]
    local=np.array([[x,y,z] for x in [.02,.025] for y in [-.005,.005] for z in [.04,.05]])
    metadata={'eef_pose_world_xyzw':original,'gripper_qpos':[.04,-.04],
        'robot_collision_volumes_world':[{'robot_geom_name':'gripper0_finger1_pad_collision',
            'vertices_world':transform_points(local,pose_matrix(original))}]}
    point=np.array([.3,.2,1.1]);rotation=Rotation.from_euler('y',90,degrees=True)
    candidate=contact_pose_candidates(metadata,point,rotation.as_quat())[0]
    target=pose_matrix(candidate['tcp_pose_world_xyzw'])
    actual=target[:3,3]+target[:3,:3]@np.array(candidate['contact_local_m'])
    assert np.allclose(actual,point)
    assert np.linalg.norm(target[:3,3]-point)>.04
    assert candidate['gripper_aperture_qpos']==[.04,-.04]
    from vlm_tamp.robot_contracts import validate_operation
    validate_operation('plan_contact_pose',{'point_world_m':point.tolist()})
