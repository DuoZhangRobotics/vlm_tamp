from types import SimpleNamespace
import importlib.util
from pathlib import Path
import numpy as np
import pytest

from vlm_tamp.execution_limits import checked_control_steps,control_limit,remaining_controls
from vlm_tamp.profiled_sim_client import execute_batch
from vlm_tamp.hinged_contact_tools import grasp_budget_check


def test_extended_budget_survives_client_and_grasp_precheck():
    state={'control_steps':593,'maximum_control_steps':1200}
    assert remaining_controls(state)==607
    assert grasp_budget_check(state,{})['remaining_control_steps']==607
    calls=[]
    def issue(refs,**kwargs):
        calls.append(len(refs))
        return {'control_steps':633,'maximum_control_steps':1200,
                'trajectory':{'events':[],'issued_commands':40}}
    new,_=execute_batch(SimpleNamespace(joint_trajectory=issue),state,np.zeros((40,7)))
    assert calls==[40] and remaining_controls(new)==567
    with pytest.raises(ValueError,match='remaining budget'):
        execute_batch(SimpleNamespace(joint_trajectory=issue),{**state,'maximum_control_steps':600},np.zeros((40,7)))


def test_service_does_not_terminate_at_old_limit(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('budget_service_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot=module.RobotService.__new__(module.RobotService)
    robot.output=tmp_path;robot.maximum_control_steps=1200;robot.state={'control_steps':600,'maximum_control_steps':1200}
    status=robot.episode_status()
    assert not status['episode_done'] and status['remaining_control_steps']==600
    robot.state['control_steps']=1200
    assert robot.episode_status()['termination_reason']=='control_budget_exhausted'


@pytest.mark.parametrize('value',[0,-1,True,1200.5])
def test_control_budget_rejects_invalid_input(value):
    with pytest.raises(ValueError):checked_control_steps(value)
