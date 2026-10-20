import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def service(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('sequence_service_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot=module.RobotService.__new__(module.RobotService)
    robot.episode=tmp_path/'episode';robot.episode.mkdir()
    robot.workspace=tmp_path/'workspace';robot.public=robot.workspace/'robot_data'
    (robot.public/'attempt01').mkdir(parents=True)
    robot.attempt=1;robot.calls=1;robot.epoch=0;robot.plans={}
    robot.state={'control_steps':0,'metadata':{'joint_positions':[0.]*7}}
    robot.stop_on_success=True
    return module,robot


def install_sequence(robot):
    path=np.array([[0.]*7,[.01]*7])
    robot.plans['test']={'kind':'sequence','path':path,'control_steps':0,'epoch':0,
        'extra':{'grasp_joints':path[-1],'manipulation':path+.01}}


def test_sequence_stops_immediately_on_success_and_does_not_revalidate_between_stages(tmp_path):
    _,robot=service(tmp_path);install_sequence(robot);calls=[]
    def issue(refs,gripper,label,**kwargs):
        calls.append((label,gripper))
        robot.epoch+=1;robot.state['control_steps']+=len(refs)
        success=len(calls)==2
        return {'task_success':success,'episode_done':success,'observation':{}}
    robot.issue=issue
    result=robot.execute_sequence('test',close_steps=3,hold=0)
    assert len(calls)==2 and result['task_success']
    assert result['remaining_stages_discarded_on_success']==1
    assert [r['stage'] for r in result['stages']]==['approach','closure']


def test_sequence_checks_full_budget_before_issuing_any_stage(tmp_path):
    _,robot=service(tmp_path);install_sequence(robot)
    robot.state['control_steps']=599;robot.plans['test']['control_steps']=599
    robot.issue=lambda *a,**k:pytest.fail('An over-budget sequence must not begin')
    with pytest.raises(ValueError,match='Complete sequence needs'):
        robot.execute_sequence('test')


@pytest.mark.parametrize('gripper',[0,1])
def test_neutral_and_closing_arc_commands_both_synchronize_geometry(tmp_path,monkeypatch,gripper):
    module,robot=service(tmp_path)
    robot.config={'scene_pose_sync':'predicted'};robot.plans={'arc':{'kind':'arc'}}
    robot.sim=object();robot.events=[];robot.executions=[];robot.tracker=None
    robot.ik=SimpleNamespace(chain=None);robot.scene=object();robot.anchors=[[0,0,0]]
    robot.stop_on_success=False;robot.capture=lambda:{'control_steps':5}
    def execute(*a,**k):
        return {'control_steps':5,'metadata':{'joint_positions':[0.]*7}},{
            'issued_commands':5,'planned_commands':5,'all_commands_issued':True}
    monkeypatch.setattr(module,'execute_batch',execute)
    monkeypatch.setattr(module,'hand_motion',lambda *a:(np.eye(4),.1))
    monkeypatch.setattr(module,'transport_scene',lambda *a,**k:SimpleNamespace(
        spec={},contact_reference=np.zeros((3,3)),pose_update={'source':k['source']}))
    result=robot.issue(np.zeros((5,7)),gripper,'arc')
    assert robot.epoch==1
    assert result['scene_pose_update']['source']=='hand_motion_prediction'


def test_compact_success_retains_full_numerical_evidence(tmp_path):
    _,robot=service(tmp_path)
    full={'status':'solved','goal_proposals':[{'accepted':True}]*32,'global_queries':[{'exact_solution':True}]}
    result=robot.keep_plan(np.zeros((2,7)),full,'arc')
    assert result['result']['goal_proposals_tried']==32
    assert result['result']['global_queries']==1
    saved=json.loads((robot.workspace/result['planning_details']).read_text())
    assert saved==full
    assert robot.plans[result['plan_id']]['result']==full
