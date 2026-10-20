"""Regression evidence for sensor-only contact-role and goal-estimation repairs."""
import json
from pathlib import Path
import numpy as np
from vlm_tamp.reliability_geometry import local_face_mask,drawer_closing_target,refine_control_axis
from vlm_tamp.cohort_execution import read_cohort_observation
from vlm_tamp.coupled_storage import read_json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mcvamp-mechanisms-frozen-20261015-b'


def test_local_contact_face_does_not_exempt_unrelated_surfaces():
    contact=np.array([[0,0,.025]])
    points=np.array([[.01,0,0],[.01,0,-.012],[.10,0,0],[.01,0,.02]])
    assert local_face_mask(points,contact,np.zeros(3),np.array([0,0,1])).tolist()==[True,False,False,False]


def test_saved_rejected_drawer_points_are_the_same_observed_face():
    for i,run in [(11,BASE),(41,ROOT/'results/mcvamp-mechanisms-camera-retry-20261016')]:
        folder=run/f'close_top_drawer-state{i:03d}';spec=read_json(folder/'task-spec.json');cloud=np.load(folder/'grasp-cloud.npy')
        points=np.load(ROOT/f'results/reliability-diagnosis-20261016/close_top_drawer-{i}-collision-points.npy')
        assert len(points)>0 and local_face_mask(points,cloud,np.array(spec['front_plane']['point_world_m']),np.array(spec['panel_normal_world'])).all()


def test_closing_distance_uses_observed_face_gap(monkeypatch):
    from vlm_tamp import reliability_geometry as geometry
    x,z=np.meshgrid(np.linspace(-.12,.12,80),np.linspace(-.24,-.05,80));points=np.c_[x.ravel(),np.full(x.size,-.168),z.ravel()]
    monkeypatch.setattr(geometry,'observed_points',lambda *args,**kw:(points,{}))
    spec={'target_progress':.15,'front_plane':{'point_world_m':[0,0,0]},'panel_normal_world':[0,1,0]}
    estimated=drawer_closing_target({}, {},spec,np.zeros((1,3)),{'robot_self_depth_tolerance_m':.0001})
    assert abs(estimated['target_progress']-.178)<1e-8 and spec['target_progress']==.15
    assert estimated['closing_target_estimate']['simulator_joint_state_used'] is False


def test_stove_recovery_uses_raised_control_not_burner_box_center():
    folder=BASE/'turn_on_stove-state021';spec=read_json(folder/'task-spec.json');meta=read_json(folder/'opened/metadata.json')
    cameras,_=read_cohort_observation(folder/'opened/obs.npz',meta);choices=json.loads((folder/'grounding/response.txt').read_text())
    result,cloud=refine_control_axis(cameras,choices,spec,np.load(folder/'grasp-cloud.npy'))
    origin=np.asarray(spec['front_plane']['point_world_m']);normal=np.asarray(spec['panel_normal_world'])
    assert len(cloud)>100 and np.quantile((cloud-origin)@normal,.01)>.009
    assert result['control_axis_refinement']['source']=='raised connected RGB-D control component'
    assert np.linalg.norm(np.array(result['joint_origin_world_m'])-spec['joint_origin_world_m'])>.04
    assert result['control_axis_refinement']['simulator_object_truth_used'] is False


def test_historical_points_are_cleared_only_in_current_observed_free_space():
    from vlm_tamp.reliable_arc_scene import observed_free_space
    depth=np.full((15,15),2.);camera={'depth':depth,'camera_to_world':np.eye(4),'intrinsic':np.array([[10,0,7],[0,10,7],[0,0,1.]])}
    points=np.array([[0,0,1.],[0,0,2.],[0,0,3.],[10,0,1.]])
    assert observed_free_space(points,{'test':camera}).tolist()==[True,False,False,False]
    depth[6,7]=.5
    assert not observed_free_space(points[:1],{'test':camera})[0]


def test_bad_microwave_contact_recovers_supported_bar_without_simulator_pose():
    from vlm_tamp.reliable_handle import ground_handle
    from vlm_tamp.coupled_resources import fresh_task_template
    folder=ROOT/'results/all-prior-tasks-frozen-v3-20261015/close_microwave-state011'
    meta=read_json(folder/'opened/metadata.json');cameras,_=read_cohort_observation(folder/'opened/obs.npz',meta)
    choices=json.loads((folder/'grounding/response.txt').read_text());original=json.loads(json.dumps(choices))
    template=fresh_task_template(read_json(ROOT/'results/articulated-physics-pilot/close_microwave-state001/task-spec.json'),choices)
    _,cloud,_,anchors,evidence=ground_handle(cameras,choices,template,meta)
    assert choices==original
    assert np.ptp(cloud[:,2])>.10 and min(a[2] for a in anchors)>.95
    assert evidence['handle_recovery']['original_height_m']<.005
    assert evidence['central_bar_refinement']['radial_residual_m']<.001
    assert evidence['handle_recovery']['simulator_geometry_used'] is False


def test_placement_contact_is_limited_to_fingers_selected_plate_and_phase():
    from vlm_tamp.reliable_bowl_fallback import PlacementContactPolicy
    volumes=[{'robot_geom_name':'gripper0_finger1_collision'},{'robot_geom_name':'gripper0_hand_collision'}]
    policy=PlacementContactPolicy(volumes,{'task_type':'pick_place'},{},np.array([[0.,0.,.91]]))
    points=np.array([[0,0,.91],[0,0,.90],[.03,0,.91]])
    assert policy.local_panel_allowed(points,volumes[0],np.empty((0,3)),'place',np.eye(4)).tolist()==[True,False,False]
    assert not policy.local_panel_allowed(points,volumes[1],np.empty((0,3)),'place',np.eye(4)).any()
    assert not policy.local_panel_allowed(points,volumes[0],np.empty((0,3)),'carry',np.eye(4)).any()
