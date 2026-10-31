from itertools import product
from types import SimpleNamespace
import numpy as np

from vlm_tamp.coupled_scene import Scene
from vlm_tamp.current_robot_scene import clear_current_robot_scene,points_outside_current_robot
from vlm_tamp.contact_skills import predicted_scene
from vlm_tamp.scene_pose_sync import transport_scene
from vlm_tamp.coupled_storage import preserve_scene,read_json


def metadata(center=0.):
    offset=np.array([center,0.,0.]);normals=np.vstack([np.eye(3),-np.eye(3)])
    return {'joint_positions':[center,0,0,0,0,0,0],'gripper_qpos':[.04,-.04],
        'robot_collision_volumes_world':[{'type':'convex','robot_geom_name':'gripper0_hand_collision',
            'vertices_world':np.asarray(list(product([-.01,.01],repeat=3)))+offset,
            'planes_world':np.c_[normals,-.01-normals@offset]}]}


def scene(static=(),moving=((0.,0.,0.),(.04,0.,0.)),contact=((0.,0.,0.),)):
    spec={'joint_origin_world_m':[0,0,0],'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':[0,0,0]},'panel_normal_world':[0,1,0]}
    return Scene.from_points(static,moving,contact,spec)


def test_only_exact_current_robot_occupancy_is_removed_without_padding():
    points=np.array([[0.,0,0],[.01,0,0],[.010001,0,0],[.1,0,0]])
    kept,report=points_outside_current_robot(points,metadata())
    assert np.array_equal(kept,points[2:]) and report['removed_points']==2
    assert np.array_equal(points[:,0],[0.,.01,.010001,.1])
    assert not kept.flags.writeable


def test_sphere_and_overlapping_robot_parts_count_removed_points_once():
    meta=metadata();meta['robot_collision_volumes_world'].append(
        {'type':'sphere','robot_geom_name':'another_link','center_world':[0,0,0],'radius':.02})
    kept,report=points_outside_current_robot([[0,0,0],[.015,0,0],[.021,0,0]],meta)
    assert report['removed_points']==2 and len(kept)==1
    assert sum(report['points_inside_each_robot_part'].values())==3


def test_query_copy_cleans_collision_clouds_and_indices_preserving_perception():
    raw=scene(static=[[0,0,0],[.05,0,0]])
    query=clear_current_robot_scene(raw,metadata(),406)
    assert len(raw.static_points)==2 and len(raw.moving_reference)==2
    assert np.array_equal(query.static_points,[[.05,0,0]])
    assert np.array_equal(query.moving_reference,[[.04,0,0]])
    assert query.static_tree.query([0,0,0])[0]==.05
    assert query.collision_tree.query([0,0,0])[0]==.04
    assert np.array_equal(query.contact_reference,raw.contact_reference)
    assert query.current_robot_filter['contact_reference_is_association_only']
    assert query.current_robot_filter['reference_control_step']==406
    assert query.current_robot_filter['padding_m']==0.
    assert not hasattr(raw,'current_robot_filter')


def test_current_pose_changes_do_not_accumulate_holes_in_the_raw_scene():
    raw=scene(static=[[0,0,0],[.1,0,0]])
    first=clear_current_robot_scene(raw,metadata(0),1)
    second=clear_current_robot_scene(raw,metadata(.1),2)
    assert np.array_equal(first.static_points,[[.1,0,0]])
    assert np.array_equal(second.static_points,[[0,0,0]])
    assert len(raw.static_points)==2


def test_hypothetical_moving_geometry_is_not_cleared_against_the_current_robot():
    raw=scene(moving=[[.04,0,0]],contact=[[.04,0,0]])
    query=clear_current_robot_scene(raw,metadata(),10)
    predicted=predicted_scene(query,[-.04,0,0],0.)
    assert np.allclose(predicted.moving_reference,[[0,0,0]])
    assert predicted.current_robot_filter['reference_control_step']==10
    assert not predicted.current_robot_filter['future_candidate_poses_used']


def test_empty_obstacle_role_survives_transport_and_export(tmp_path):
    raw=scene(moving=[[0,0,0]],contact=[[0,0,0]])
    query=clear_current_robot_scene(raw,metadata(),10)
    assert query.moving_reference.shape==(0,3)
    transform=np.eye(4);transform[0,3]=.02
    future=transport_scene(query,transform,source='hand_motion_prediction',reference_step=10)
    assert future.moving_reference.shape==(0,3)
    assert len(future.contact_reference)==1
    folder=tmp_path/'arc';folder.mkdir();preserve_scene(future,folder)
    ref=read_json(folder/'scene-reference.json')
    assert ref['current_robot_point_filter']['reference_control_step']==10
    with np.load(ref['arrays_path']) as arrays:
        assert arrays['moving_reference'].shape==(0,3)
        assert len(arrays['contact_reference'])==1  # Association metadata, never extra obstacles.


def test_free_scene_entry_cleans_the_query_and_preserves_perception():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('current_scene_service',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    service=module.RobotService.__new__(module.RobotService)
    service.scene=scene(static=[[0,0,0],[.05,0,0]]);service.config={}
    service.state={'metadata':metadata(),'control_steps':406}
    query=service._query_scene()
    assert np.array_equal(query.static_points,[[.05,0,0]])
    assert query.current_robot_filter['reference_control_step']==406
    assert len(service.scene.static_points)==2


def test_grasp_preparation_clears_actual_geometry_not_the_hypothetical_open_hand(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('grasp_current_scene_service',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    service=module.RobotService.__new__(module.RobotService)
    measured=metadata();measured['gripper_qpos']=[.02,-.02]
    service.scene=scene(static=[[0,0,0],[.04,0,0]])
    service.state={'metadata':measured,'control_steps':10};service.config={'grasp_selection_method':'graspgen_direct'}
    service.contact_scene_kind='articulated';service.spec={};service.cloud=np.array([[0,0,0],[.04,0,0]])
    service.anchors=[];service.record={};service.ik=SimpleNamespace(events=[],chain=None)
    service.folder=lambda _:tmp_path/'grasp';service.open_gripper_reference_qpos=[.04,-.04]
    service.last_gripper_command=1
    received=[]
    service.resources=SimpleNamespace(grasps=lambda points,*a,**k:received.append(points.copy()))
    future=metadata(.04)['robot_collision_volumes_world']
    service.sim=SimpleNamespace(fk=lambda *a,**k:{'robot_collision_volumes_world':future})
    monkeypatch.setattr(module,'require_grasp_budget',lambda *a:{})
    monkeypatch.setattr(module,'RobotVolumes',lambda *a:SimpleNamespace())
    def select(generated,anchors,state,ik,record,opened,closed,*args):
        assert np.array_equal(opened.static_points,[[.04,0,0]])
        assert opened.current_robot_filter['gripper_qpos']==[.02,-.02]
        assert state['metadata']['gripper_qpos']==[.04,-.04]
        return {'joints':[0.]*7,'approach':np.zeros((2,7))}
    monkeypatch.setattr(module,'select_grasp',select)
    service._select_contact(3.,8,0)
    assert np.array_equal(received[0],[[.04,0,0]])
    assert len(service.cloud)==2 and len(service.scene.static_points)==2
