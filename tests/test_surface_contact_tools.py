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
