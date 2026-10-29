from vlm_tamp.planning_feedback import summarize_arc_failure
from vlm_tamp.planning_feedback import summarize_pose_failure


def test_feedback_separates_ik_projection_from_collision_and_unknown_values():
    data={'status':'search_budget_exhausted','goal_angle_rad':1.5,'grasp_constraint':'rigid',
          'start':{'accepted':True},'admissible_goals':0,
          'goal_proposals':[
              {'projection':{'accepted':False,'position_error_m':.1,'jaw_inequality_error':.15}},
              {'projection':{'accepted':True,'position_error_m':0.,'jaw_inequality_error':0.},
               'scene':{'accepted':False}}]}
    result=summarize_arc_failure(data)
    assert result['goal_projection_successes']==1
    assert result['projected_goals_rejected_by_collision']==1
    assert result['start_collision_free']
    assert not result['simulator_object_state_used']
    assert result['last_attempted_continuation_angle_deg'] is None
    empty=summarize_arc_failure({'status':'invalid_start','start':{'accepted':False}})
    assert empty['minimum_endpoint_position_error_m'] is None
    assert not empty['start_collision_free']


def test_failed_arc_exposes_current_diagnostics_without_private_path(tmp_path,monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import pytest
    source=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('feedback_service_test',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot=module.RobotService.__new__(module.RobotService)
    robot.episode=tmp_path/'private';robot.episode.mkdir()
    robot.workspace=tmp_path/'public';robot.public=robot.workspace/'robot_data'
    (robot.public/'attempt01').mkdir(parents=True)
    robot.calls=3;robot.attempt=1;robot.config={}
    robot.ik=robot.scene=robot.record=robot.spec=None
    robot.spec={'joint_origin_world_m':[0.,0.,0.],'joint_axis_world':[0.,0.,1.]}
    robot.epoch=0
    robot.state={'metadata':{'joint_positions':[0.]*7}}
    def fail(*args,**kwargs):
        folder=args[6];folder.mkdir()
        (folder/'result.json').write_text(json.dumps({'status':'search_budget_exhausted','goal_angle_rad':1.5}))
        raise ValueError('No complete numerical plan')
    monkeypatch.setattr(module,'measured_robot_scene',lambda service:service.scene)
    monkeypatch.setattr(module,'plan_signed_arc',fail)
    with pytest.raises(ValueError) as caught:
        robot.plan_arc(90,grasp_constraint='rigid')
    details=caught.value.planning_diagnostic
    assert details['status']=='search_budget_exhausted'
    assert (robot.workspace/details['full_diagnostic']).exists()
    assert str(robot.episode) not in json.dumps(details)


def test_collision_feedback_names_geometry_and_does_not_reuse_stale_part_for_self_collision():
    result={'start':{'self_rejections':0},'goal_proposals':[
        {'projection':{'accepted':True},'scene':{'accepted':False,'self_rejections':0,
            'last_rejected_surface':1,'last_rejected_part':1000}},
        {'projection':{'accepted':True},'scene':{'accepted':False,'self_rejections':1,
            'last_rejected_surface':1,'last_rejected_part':1000}}]}
    native={'gripper_cover':[{'sphere_range':[0,10],'robot_geom_name':'gripper0_hand_collision'}]}
    summary=summarize_arc_failure(result,native)
    assert summary['first_blocking_surface_by_rejected_goal']=={
        'moving_noncontact_geometry':1,'robot_self_collision':1}
    assert summary['first_blocking_robot_part_by_rejected_goal']['gripper0_hand_collision']==1


def test_free_pose_feedback_identifies_goal_collision_before_path_search():
    result=summarize_pose_failure([{'result':{'status':'invalid_endpoint','endpoints':{
        'start':{'accepted':True},'goal':{'accepted':False,'last_rejected_part':2006,'last_rejected_surface':0}}}}])
    assert result['module']=='native_scene.endpoint_admission'
    assert result['robot_valid_ik_solutions']==1
    assert result['endpoint_rejection_examples'][0]['robot_part']=='robot0_link6_collision'
    assert result['endpoint_rejection_examples'][0]['reason']=='stationary_observed_geometry'
    assert not result['physical_motion_issued']
    assert summarize_pose_failure([])['module']=='full_pose_ik'
