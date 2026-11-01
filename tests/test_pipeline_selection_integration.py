"""Regressions for information lost between selection, planning and episode state."""
import json
from types import SimpleNamespace
import numpy as np
import pytest

from vlm_tamp.robot_contracts import validate_operation


def test_push_mask_hole_is_not_filled_by_multiview_role_expansion(monkeypatch):
    import vlm_tamp.contact_skills as skills
    import vlm_tamp.surface_contact_tools as surfaces
    monkeypatch.setattr(skills,'RobotVolumes',lambda *a,**k:None)
    monkeypatch.setattr(surfaces,'RobotVolumes',lambda *a,**k:None)
    y,x=np.indices((11,11));world=np.stack([(x-5)*.003,(y-5)*.003,np.ones_like(x)],axis=-1)
    camera={'world_points':world,'depth':np.ones((11,11)),'camera_to_world':np.eye(4),
        'intrinsic':np.array([[1/.003,0.,5.],[0.,1/.003,5.],[0.,0.,1.]])}
    mask=x<=7;mask[5,5]=False
    occupied=np.vstack([world.reshape(-1,3),[0.,0.,1.005]])
    metadata={'joint_positions':[0.]*7,'robot_collision_volumes_world':[],
        'eef_pose_world_xyzw':[0.,0.,.9,0.,0.,0.,1.]}
    scene,evidence,_=skills.build_contact_scene({'view':camera},metadata,None,'view',[0,0,11,11],
        [0,0,11,11],'fingertips',[0,0,1],observed=(occupied,{}),contact_mask=mask,target_mask=mask,contact_pixel_xy=[4,4],contact_name='pad')
    assert evidence['contact_mask_used'] and evidence['target_mask_used']
    for point in [[0,0,1],[0,0,1.005],[.015,0,1]]:
        assert np.any(np.all(scene.static_points==point,axis=1))
        assert not np.any(np.all(scene.moving_reference==point,axis=1))
    assert len(scene.static_points)+len(scene.moving_reference)==len(occupied)


@pytest.mark.parametrize('operation',['push','hook_pull','select_surface_contact'])
def test_same_registered_contact_is_callable_without_another_box(operation):
    args={'camera':'view','contact_pixel_xy':[4,4],'contact_part_id':'part-selected'}
    if operation!='select_surface_contact':args.update(observation_id=1,direction_world=[1,0,0],contact_name='pad')
    assert validate_operation(operation,args)==args


def test_failed_simulator_reset_is_terminal_infrastructure_not_a_task_failure(tmp_path,monkeypatch):
    import serve_subscription_robot as module
    workspace=tmp_path/'workspace';workspace.mkdir();closed=[]
    class Client:
        def __init__(self,*args):pass
        def request(self,*args,**kwargs):raise RuntimeError('EGL initialization unavailable')
        def close(self):closed.append(True)
    monkeypatch.setattr(module,'ProfiledClient',Client)
    monkeypatch.setattr(module,'CoupledResources',lambda *a:SimpleNamespace())
    robot=module.RobotService(tmp_path/'out',workspace,17,1,stop_on_success=True,persistent_episode=True)
    with pytest.raises(RuntimeError,match='EGL'):robot.reset()
    status=robot.episode_status()
    assert closed==[True] and robot.sim is None
    assert status['episode_done'] and not status['valid_task_attempt']
    assert status['termination_reason']=='simulator_initialization_failed'
    assert status['control_steps']==0 and not status['task_success']
    assert robot.dispatch('observe',{})['infrastructure_failure']['stage']=='simulator_reset'
    assert robot.dispatch('finish',{'claim':'incomplete','reason':'initialization failed'})['finished']
    assert robot.results==[]
    marker=json.loads((robot.output/'operator-interruption.json').read_text())
    assert marker['excluded_from_completed_task_outcomes']
