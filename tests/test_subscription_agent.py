"""Research-contamination and billing boundaries for the subscription pilot."""
import importlib.util
from pathlib import Path
import shutil

import pytest

from vlm_tamp.subscription_agent import clean_environment, codex_arguments, verify_isolation


def service_module():
    path=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('subscription_robot_for_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_child_does_not_inherit_paid_api_keys_or_project_import_path(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','not-a-real-key')
    monkeypatch.setenv('CODEX_API_KEY','not-a-real-key')
    monkeypatch.setenv('PYTHONPATH','/private/solutions')
    monkeypatch.setenv('CODEX_THREAD_ID','prior-conversation')
    env=clean_environment()
    assert not {'OPENAI_API_KEY','CODEX_API_KEY','PYTHONPATH','CODEX_THREAD_ID'} & env.keys()
    args=codex_arguments('/tmp/public','/tmp/private-final')
    assert 'forced_login_method="chatgpt"' in args
    assert args[args.index('-m')+1]=='gpt-6-luna'
    assert '--ignore-user-config' in args
    assert 'project_doc_max_bytes=0' in args


@pytest.mark.integration
def test_os_boundary_blocks_direct_and_symlink_reads(tmp_path):
    if not shutil.which('codex'):
        pytest.skip('Codex executable required')
    private=tmp_path/'private';private.mkdir()
    canary=private/'solution.txt';canary.write_text('a hidden solution')
    public=tmp_path/'public';public.mkdir()
    link=public/'solution-link.txt';link.symlink_to(canary)
    result=verify_isolation(public,[canary,link])
    assert result['passed']
    assert all(not r['readable'] for r in result['reads'])


def test_public_robot_state_does_not_contain_evaluator_or_inventory():
    module=service_module()
    state={'control_steps':17,'evaluation':{'task_success':True},
           'metadata':{'joint_positions':[0]*7,'eef_pose_world_xyzw':[0]*7,
                       'gripper_qpos':[.04,-.04],'native_hinge':[1,2,3],
                       'object_inventory':['hidden'], 'source_path':'/private/solution'}}
    result=module.public_state(state)
    assert set(result)=={'control_steps','remaining_control_steps','joint_positions',
                         'eef_pose_world_xyzw','gripper_qpos'}
    assert result['remaining_control_steps']==583


def test_grounded_model_keeps_camera_required_by_scene_update(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    module=service_module()
    robot=module.RobotService.__new__(module.RobotService)
    robot.sim=object();robot.cameras={'frontview':{'rgb':np.zeros((512,512,3),dtype=np.uint8)}}
    robot.state={'metadata':{}};robot.epoch=0;robot.calls=1;robot.episode=tmp_path
    robot.rebuild=lambda:None;robot.export_geometry=lambda evidence:robot.spec
    def fit(cameras,choices,template,metadata):
        return SimpleNamespace(spec=template),np.zeros((100,3)),np.zeros((512,512)),[],{'cylinder_radius_m':.01}
    monkeypatch.setattr(module,'ground_handle',fit)
    result=robot.ground({'camera':'frontview','moving_panel_box_xyxy':[10,10,90,90],
        'contact_box_xyxy':[70,20,80,80],'hinge_pixels_xy':[[10,10],[10,90]]})
    assert result['camera']=='frontview'


def test_failed_tracking_update_preserves_previous_geometry():
    import numpy as np
    module=service_module()
    robot=module.RobotService.__new__(module.RobotService)
    robot.tracking_result={'pose_accepted':True,'rotation_xy':np.eye(2).tolist(),'translation_xy_m':[.01,0]}
    old=np.array([[.2,.3,.4]])
    robot.cloud=old;robot.spec={'camera':'frontview'};robot.mask=object();robot.choices={}
    robot.scene=object();robot.applied_pose=np.eye(4);robot.epoch=4;robot.anchors=[]
    def fail(pose,hinge):
        robot.cloud=old+.01;robot.spec={'camera':'broken'};robot.epoch=5
        raise ValueError('Synthetic scene rebuild failure')
    robot._apply_tracking_pose=fail
    with pytest.raises(ValueError):robot.apply_tracking()
    assert robot.cloud is old
    assert robot.spec=={'camera':'frontview'}
    assert robot.epoch==4
