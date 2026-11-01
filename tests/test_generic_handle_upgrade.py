import copy
from types import SimpleNamespace
import numpy as np


def test_current_handle_fit_upgrades_generic_joint_without_losing_axis(tmp_path,monkeypatch):
    import serve_subscription_robot as module
    from vlm_tamp.rekep_libero.mechanism_geometry import Mechanism
    robot=module.RobotService.__new__(module.RobotService)
    robot.workspace=tmp_path/'workspace';robot.public=robot.workspace/'robot_data'
    (robot.public/'attempt01').mkdir(parents=True)
    robot.episode=tmp_path/'episode';robot.episode.mkdir()
    robot.attempt=1;robot.calls=1;robot.epoch=0;robot.plans={};robot.config={};robot.record={}
    fixed={'generic_part_joint_model':True,'joint_origin_world_m':[.1,.2,1.],
        'joint_axis_world':[0.,0.,-1.],'joint_estimate_id':'measured-joint',
        'front_plane':{'point_world_m':[.3,.2,1.]},'panel_normal_world':[1.,0.,0.],
        'motion':'signed relative rotation; caller chooses task direction'}
    robot.task_spec={'motion':'close'}
    robot.spec=copy.deepcopy(fixed);robot.contact_scene_kind='articulated'
    robot.state={'control_steps':20,'metadata':{'joint_positions':[0.]*7,'gripper_qpos':[.04,-.04],
        'robot_collision_volumes_world':[],'eef_pose_world_xyzw':[0,0,1,0,0,0,1]}}
    robot.cameras={'view':{'rgb':np.zeros((100,100,3),np.uint8)}}
    robot.sim=SimpleNamespace(fk=lambda *a,**k:{'robot_collision_volumes_world':[]})
    robot.ik=SimpleNamespace(chain=object(),events=[])
    robot.resources=SimpleNamespace(grasps=lambda *a,**k:{})
    robot.episode_status=lambda:{'remaining_wall_seconds':300.}
    robot.rebuild=lambda:setattr(robot,'scene',SimpleNamespace(robot=object()))
    # This fixture tests the handle upgrade, not the independently tested
    # current-robot point filter; its scene has no collision clouds.
    robot._query_scene=lambda policy:robot.scene
    robot.export_geometry=lambda evidence:{'model':copy.deepcopy(robot.spec)}
    import vlm_tamp.explicit_grounding as explicit
    def fit(cameras,metadata,choices,*,fixed_joint,**kwargs):
        assert fixed_joint['joint_axis_world']==fixed['joint_axis_world']
        return {'spec':{**copy.deepcopy(fixed_joint),'camera':choices['camera'],'generic_part_joint_model':False},
            'cloud':np.array([[.3,.2,1.],[.3,.2,1.1]]),'mask':np.ones((100,100),bool),
            'anchors':[[.3,.2,1.]],'evidence':{}}
    monkeypatch.setattr(explicit,'ground_pixels',fit)
    monkeypatch.setattr(explicit,'selected_scene',lambda *a:SimpleNamespace(robot=object()))
    monkeypatch.setattr(module,'RobotVolumes',lambda *a:SimpleNamespace(reference_volumes=[]))
    monkeypatch.setattr(module,'ArcContactScene',lambda *a,**k:SimpleNamespace(robot=k['robot']))
    called=[]
    def select(*args):
        called.append(args[7])
        return {'approach':np.zeros((2,7)),'joints':[0.]*7,'pose_world_xyzw':[0,0,1,0,0,0,1],
            'source':'observed test handle','search_seconds':0.,'budget_timing_assumptions':{'approach_controls':8}}
    monkeypatch.setattr(module,'select_grasp',select)
    result=robot.refresh_contact('view',[10,20,20,80],[5,10,80,90],[15,50])
    assert result['fixed_joint_preserved']
    assert robot.spec['joint_origin_world_m']==fixed['joint_origin_world_m']
    assert robot.spec['joint_axis_world']==fixed['joint_axis_world']
    assert robot.spec['joint_estimate_id']=='measured-joint'
    assert 'observed_handle_radius_m' not in robot.spec
    robot.calls+=1
    plan=robot.plan_grasp()
    assert plan['kind']=='approach' and plan['expected_executor']=='execute'
    assert len(called)==1
