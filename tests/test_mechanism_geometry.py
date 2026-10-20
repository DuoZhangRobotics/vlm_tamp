"""Manifold residuals and camera conversion are independent of simulator assets."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vlm_tamp.rekep_libero.geometry import matrix_pose,pose_matrix,interpolate_path
from vlm_tamp.rekep_libero.mechanism_geometry import Mechanism,ray_plane,project


def test_prismatic_line_tracks_inward_axis_and_rejects_sideways_or_rotation():
    mechanism=Mechanism({'joint_family':'prismatic','joint_axis_world':[0,-1,0],
                         'joint_origin_world_m':[0,0,0],'target_progress':.15})
    grasp=[.2,.1,1.,0,0,0,1]
    end=mechanism.pose(grasp,.15)
    assert np.allclose(end[:3],[.2,-.05,1.])
    assert mechanism.residual(grasp,end)['position_residual_m']<1e-12
    wrong=end.copy();wrong[0]+=.03
    assert mechanism.residual(grasp,wrong)['position_residual_m']==pytest.approx(.03)
    wrong=end.copy();wrong[3:]=Rotation.from_rotvec([.2,0,0]).as_quat()
    assert mechanism.residual(grasp,wrong)['orientation_residual_rad']==pytest.approx(.2)


def test_hinge_circle_requires_arc_interior_even_when_endpoints_match():
    mechanism=Mechanism({'joint_family':'revolute','joint_axis_world':[0,0,1],
                         'joint_origin_world_m':[0,0,1],'target_progress':np.pi/2})
    grasp=[.2,0,1,0,0,0,1]
    path=mechanism.path(grasp)
    assert len(path)>10
    assert all(mechanism.residual(grasp,p)['position_residual_m']<1e-12 for p in path)
    assert all(mechanism.residual(grasp,p)['orientation_residual_rad']<1e-12 for p in path)
    end=mechanism.pose(grasp,np.pi/2)
    straight=interpolate_path([grasp,end],position_step=.005,rotation_step=.01)
    assert max(mechanism.residual(grasp,p)['position_residual_m'] for p in straight)>.058
    assert np.allclose(pose_matrix(end)[:3,3],[0,.2,1])


def test_hinge_with_offset_world_origin_keeps_radius_and_orientation():
    mechanism=Mechanism({'joint_family':'revolute','joint_axis_world':[0,-1,0],
                         'joint_origin_world_m':[.3,.2,1.1],'target_progress':np.pi/3})
    transform=np.eye(4);transform[:3,3]=[.4,.25,1.2]
    transform[:3,:3]=Rotation.from_euler('xyz',[.3,.2,.1]).as_matrix()
    grasp=matrix_pose(transform)
    for progress in np.linspace(0,np.pi/3,10):
        pose=mechanism.pose(grasp,progress)
        assert mechanism.progress_of_pose(grasp,pose)==pytest.approx(progress)
        assert mechanism.residual(grasp,pose)['position_residual_m']<1e-10


def test_hinge_ray_project_roundtrip_uses_calibration_and_measured_plane():
    camera={'camera_to_world':np.eye(4),'intrinsic':np.array([[400,0,256],[0,400,256],[0,0,1]])}
    pixel=np.array([315,188])
    point=ray_plane(pixel,camera,np.array([0,0,1.2]),np.array([0,0,1]))
    assert point[2]==pytest.approx(1.2)
    assert np.allclose(project([point],camera)[0],pixel)
    with pytest.raises(ValueError):
        ray_plane(pixel,camera,np.array([0,0,-1]),np.array([0,0,1]))


def test_invalid_mechanism_progress_is_rejected():
    mechanism=Mechanism({'joint_family':'prismatic','joint_axis_world':[0,1,0],
                         'joint_origin_world_m':[0,0,1],'target_progress':.15})
    for value in [float('nan'),-.1,.3]:
        with pytest.raises(ValueError):
            mechanism.transform(value)


def test_known_robot_geometry_ids_preserve_unnamed_gaps_and_exclude_scene_names():
    from types import SimpleNamespace
    from vlm_tamp.rekep_libero.articulated_kinematics import RobotModelNames
    ids={'floor':0,'robot0_arm':4,'cabinet':6,'gripper0_finger':8}
    model=SimpleNamespace(geom_names=tuple(ids),geom_name2id=lambda name:ids[name],
                          _model=SimpleNamespace(ngeom=10),sentinel='unchanged')
    proxy=RobotModelNames(model,('robot0_','gripper0_'))
    assert len(proxy.geom_names)==10
    assert proxy.geom_names[4]=='robot0_arm'
    assert proxy.geom_names[8]=='gripper0_finger'
    assert proxy.geom_names[0] is None and proxy.geom_names[6] is None
    assert proxy.sentinel=='unchanged'


def test_grasp_search_covers_other_poses_before_exhausting_redundant_ik_roots():
    from vlm_tamp.rekep_libero.mechanism_planning import ranked_postures
    batches=[(0,{'name':'first'},[{'q':0},{'q':1},{'q':2}]),
             (1,{'name':'second'},[{'q':3},{'q':4}])]
    rows=list(ranked_postures(batches))
    assert [(ordinal,rank) for ordinal,_,rank,_ in rows[:4]]==[(0,0),(1,0),(0,1),(1,1)]


@pytest.mark.parametrize('mode,enforced,admitted',[('strict',True,False),('diagnostic',False,True)])
def test_diagnostic_mode_retains_collision_prediction_and_labels_nonveto(monkeypatch,mode,enforced,admitted):
    from vlm_tamp.rekep_libero.mechanism_scene import MechanismScene,SceneV8
    prediction={'accepted':False,'collisions':[{'part':'finger','penetration_m':.003}]}
    monkeypatch.setattr(SceneV8,'check',lambda *args,**kwargs:dict(prediction))
    scene=object.__new__(MechanismScene)
    scene.config={'scene_mode':mode};scene.prediction_events=[]
    result=scene.check([],np.array([0,0,0,0,0,0,1]))
    assert result['accepted'] is admitted
    assert result['scene_admission_enforced'] is enforced
    assert result['predicted_scene_admission_accepted'] is False
    assert result['collisions']==prediction['collisions']
    assert len(scene.prediction_events)==1


def test_panel_probe_gate_is_grasp_independent_and_ignores_parked_robot_collisions():
    from types import SimpleNamespace
    from vlm_tamp.rekep_libero.mechanism_planning import panel_probe_admission
    mechanism=Mechanism({'joint_family':'prismatic','joint_axis_world':[0,1,0],
                         'joint_origin_world_m':[0,0,1],'target_progress':.15})
    scene=SimpleNamespace(config={'scene_mode':'strict'},check=lambda *a,**k:{
        'collisions':[{'robot_geom_name':'robot0_link3_collision'}]})
    metadata={'robot_collision_volumes_world':[],'eef_pose_world_xyzw':[0,0,1,0,0,0,1]}
    assert panel_probe_admission(scene,metadata,mechanism)['accepted']
    scene.check=lambda *a,**k:{'collisions':[{'robot_geom_name':'predicted_moving_source_envelope'}]}
    assert not panel_probe_admission(scene,metadata,mechanism)['accepted']
    scene.config['scene_mode']='diagnostic'
    result=panel_probe_admission(scene,metadata,mechanism)
    assert result['accepted'] and not result['predicted_panel_admission_accepted']
