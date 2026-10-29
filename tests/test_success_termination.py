"""Official success ends motion, without exposing simulator geometry."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def test_success_closes_episode_and_blocks_retreat(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('success_robot_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot=module.RobotService.__new__(module.RobotService)
    robot.episode=tmp_path;robot.calls=1;robot.state={'control_steps':60,'metadata':{}}
    robot.stop_on_success=True;robot.episode_success=False;robot.sim=object()
    robot.practice_budget=None
    robot.events=[];robot.executions=[];robot.dino_tools=False;robot.completion_checks=False
    def execute(sim,state,refs,**kwargs):
        assert kwargs['stop_on_success']
        return {'control_steps':62,'metadata':{}},{'events':[],'issued_commands':2,'planned_commands':5,
            'all_commands_issued':False,'terminated_on_simulator_success':True}
    monkeypatch.setattr(module,'execute_batch',execute)
    robot.latest_observation={'task_success':True,'episode_done':True,'images':{}}
    robot.capture=lambda:robot.latest_observation
    closed=[]
    robot.finish_attempt=lambda reason:closed.append(reason)
    refs=np.zeros((5,7))
    result=robot.issue(refs,1,'test')
    assert result['task_success'] and result['episode_done']
    assert closed==['simulator_success']
    assert np.load(tmp_path/'001-issued-test.npy').shape==(2,7)
    assert np.load(tmp_path/'001-requested-test.npy').shape==(5,7)
    for operation,arguments in [('gripper',{}),('execute',{'plan_id':'unused'}),('plan_joint',{'joints':[0.]*7})]:
        with pytest.raises(ValueError,match='already ended'):
            robot.dispatch(operation,arguments)
    assert robot.dispatch('observe',{})==robot.latest_observation
    assert robot.dispatch('finish',{})['task_success']
    assert closed==['simulator_success']
