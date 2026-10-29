import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from vlm_tamp.signed_contact_motion import signed_scene,plan_signed_arc
from vlm_tamp.observed_joint_scene import current_part_points,install_joint_part
from vlm_tamp.robot_contracts import validate_operation
from vlm_tamp.timed_execution import schedule_contact,gripper_arm_reference
from vlm_tamp.joint_feedback import verify_feedback_joint
from scipy.spatial.transform import Rotation


def test_joint_revision_uses_separate_observed_motion_not_hand_motion():
    rng=np.random.default_rng(6)
    first=np.c_[rng.uniform(.12,.3,80),rng.uniform(-.05,.05,80),rng.uniform(.9,1.1,80)]
    second=first@Rotation.from_euler('z',4,degrees=True).as_matrix().T
    third=second@Rotation.from_euler('z',4,degrees=True).as_matrix().T
    records=[{'reference_observation_id':1,'current_observation_id':2,'first':first,'second':second},
             {'reference_observation_id':2,'current_observation_id':3,'first':second,'second':third}]
    result=verify_feedback_joint(records,[.04,0,1])
    assert result['motion_supported'] and result['new_model_predicts_better']
    assert np.linalg.norm(np.asarray(result['estimate']['joint_origin_world_m'])[:2])<1e-6
    records[0]['second']=first;records[1]['first']=first;records[1]['second']=first
    quiet=verify_feedback_joint(records,[.04,0,1])
    assert not quiet['motion_supported'] and not quiet['new_model_predicts_better']


def test_signed_large_request_is_segmented_without_mutating_observed_scene(tmp_path,monkeypatch):
    import vlm_tamp.arc_execution as arc
    import vlm_tamp.signed_contact_motion as module
    scene=SimpleNamespace(axis=np.array([0.,0.,1.]),spec={'joint_axis_world':[0,0,1]},
        robot=SimpleNamespace(forward=lambda q:[]))
    calls=[]
    def plan(query,record,meta,start,angle,folder,config,**kwargs):
        calls.append((query.axis.copy(),angle))
        end=np.asarray(start).copy();end[0]+=angle*query.axis[2]
        return np.vstack([start,end]),{'status':'solved','exact_solution':True}
    monkeypatch.setattr(arc,'contact_arc',plan)
    monkeypatch.setattr(module,'preserve_scene',lambda *a:None)
    monkeypatch.setattr(module,'predicted_rotation',lambda scene,angle:copy.copy(scene))
    ik=SimpleNamespace(chain=SimpleNamespace(forward=lambda q:[0,0,0,0,0,0,1]))
    path,report=plan_signed_arc(ik,scene,{}, {},np.zeros(7),-225,tmp_path/'plan',{'arc_search_seconds':10})
    assert [np.degrees(a) for _,a in calls]==pytest.approx([90,90,45])
    assert all(axis[2]==-1 for axis,_ in calls)
    assert path[-1,0]==pytest.approx(np.radians(-225))
    assert np.array_equal(scene.axis,[0,0,1]) and scene.spec['joint_axis_world']==[0,0,1]
    assert report['progress_unwrapped_by_segments']


def test_extended_contact_requests_keep_the_controller_joint_limit():
    validate_operation('push',{'observation_id':1,'camera':'view','region_xyxy':[0,0,10,10],
        'joint_degrees':-150,'joint_model_epoch':1,'contact_speed_m_s':.25,'stroke_length_m':.05})
    class Chain:
        def forward(self,q):return np.r_[q[:3],[0,0,0,1]]
    path=np.zeros((2,7));path[1,0]=.1;path[1,6]=1.
    refs=schedule_contact(path,Chain(),np.zeros(3),contact_speed_m_s=.25)
    assert np.max(np.abs(np.diff(np.vstack([path[0],refs]),axis=0)))*20<=1+1e-10


def test_large_measured_overshoot_has_an_inward_legal_target():
    limits=np.tile([-3.,3.],(7,1));q=np.zeros(7);q[5]=3.1
    target,report=gripper_arm_reference(q,limits)
    assert target[5]==3. and report['performed']
    assert report['controller_limits_actual_reference_increment']
    assert q[5]==3.1


def part_engine():
    yy,xx=np.indices((5,5));points=np.stack([xx*.025,yy*.025,np.ones_like(xx)],axis=-1).astype(float)
    view={'world_points':points,'depth':np.ones((5,5)),'camera_to_world':np.eye(4)}
    part={'observation_id':1,'camera':'view','mask':np.ones((5,5),bool)}
    engine=SimpleNamespace(parts={'part':part},tracks={},_view=lambda *a:view,
        _record=lambda oid:{'control_steps':10 if oid==1 else 20})
    return engine,points,view


def test_part_pose_needs_current_mask_or_independent_track_support():
    engine,points,_=part_engine()
    current,_=current_part_points(engine,'part','tracks',10)
    assert len(current)==25
    with pytest.raises(ValueError,match='current part mask'):
        current_part_points(engine,'part','tracks',20)
    shifted=points.reshape(-1,3)+[.03,0,0]
    engine.tracks['tracks']={'part_id':'part','observation_ids':[1,2],
        'world':np.stack([points.reshape(-1,3),shifted]),'valid':np.ones((2,25),bool)}
    current,evidence=current_part_points(engine,'part','tracks',20)
    assert np.allclose(current,shifted) and evidence['pose_fit']['accepted']


def test_registered_joint_installs_without_any_handle_fit(monkeypatch):
    import vlm_tamp.observed_joint_scene as module
    engine,points,view=part_engine();occupied=np.vstack([points.reshape(-1,3),[2.,2.,2.]])
    monkeypatch.setattr(module,'observed_points',lambda *a,**k:(occupied,{}))
    monkeypatch.setattr(module,'RobotVolumes',lambda *a,**k:SimpleNamespace(reference_volumes=[]))
    service=SimpleNamespace(_perception_tools=engine,cameras={'view':view},epoch=0,config={},
        ik=SimpleNamespace(chain=None),state={'control_steps':10,'metadata':{'joint_positions':[0.]*7,'robot_collision_volumes_world':[]}})
    service.export_geometry=lambda evidence:{'model':service.spec,'geometry_epoch':service.epoch}
    estimate={'estimate_id':'joint1','part_id':'part','track_id':'tracks', 'motion_supported':True,
        'estimate':{'joint_origin_world_m':[0,0,1],'axis_world_unsigned':[0,0,1]}}
    result=install_joint_part(service,estimate,motion_sign=-1)
    assert result['handle_model_required'] is False
    assert service.spec['generic_part_joint_model'] and 'observed_handle_radius_m' not in service.spec
    assert np.array_equal(service.scene.axis,[0,0,-1])
    assert len(service.scene.static_points)==1 and service.scene_reference_control_step==10


@pytest.mark.parametrize('lose_contact',[False,True])
def test_sequence_replans_from_actual_closure_and_returns_slip_recovery(tmp_path,monkeypatch,lose_contact):
    import vlm_tamp.sequence_execution as module
    from test_robot_sequence import service,install_sequence
    _,robot=service(tmp_path);install_sequence(robot);robot.record={};robot.scene=object()
    robot.folder=lambda name:tmp_path/f'{robot.calls}-{name}'
    monkeypatch.setattr(module,'measured_robot_scene',lambda s:s.scene)
    monkeypatch.setattr(module,'grasp_observation',lambda *a,**k:{'status':'grasp_not_verified'})
    planned=[]
    def plan(ik,scene,record,meta,start,degrees,*a,**k):
        assert meta['gripper_qpos']==[.012,-.009]
        planned.append(degrees);target=np.asarray(start).copy();target[0]+=.01
        return np.vstack([start,target]),{'status':'solved'}
    monkeypatch.setattr(module,'plan_signed_arc',plan)
    def issue(refs,gripper,label,**kwargs):
        robot.state['control_steps']+=len(refs);robot.state['metadata']['joint_positions']=refs[-1].tolist()
        if label.endswith('closure'):robot.state['metadata']['gripper_qpos']=[.012,-.009]
        slip=lose_contact and label.endswith('manipulation')
        return {'task_success':False,'episode_done':False,'observation':{},
            'grasp_evidence':{'status':'gripper_gap_collapsed_during_manipulation' if slip else 'grasp_not_verified'},
            'motion_effect':{'observed_region_rotation_deg':15.,'status':'measured_part_motion'}}
    robot.issue=issue
    result=robot.execute_sequence('test',close_steps=3,hold=0,calibrate_joint=False)
    assert result['post_closure_replanning'] and result['measured_closure_aperture_qpos']==[.012,-.009]
    assert planned==([15.] if lose_contact else [15.,15.])
    if lose_contact:
        assert result['stop_reason']=='grasp_engagement_lost' and result['recovery_required']
        assert result['remaining_control_steps']>0


def test_automatic_joint_revision_uses_longer_baseline_and_third_check(tmp_path,monkeypatch):
    import vlm_tamp.sequence_execution as module
    import vlm_tamp.joint_feedback as feedback
    from test_robot_sequence import service,install_sequence
    _,robot=service(tmp_path);install_sequence(robot);robot.record={};robot.scene=object();robot.cameras={}
    robot.folder=lambda name:tmp_path/f'{robot.calls}-{name}'
    monkeypatch.setattr(module,'measured_robot_scene',lambda s:s.scene)
    monkeypatch.setattr(module,'grasp_observation',lambda *a,**k:{'status':'grasp_not_verified'})
    planned=[];revisions=[];count=[0];robot.motion_feedback=[]
    points=np.random.default_rng(3).uniform([.1,-.1,.9],[.3,.1,1.1],(50,3))
    moved=lambda n:points@Rotation.from_euler('z',3*n,degrees=True).as_matrix().T
    class Tracker:
        def __init__(self,start):self.start=start;self.observe()
        def observe(self,*a,**k):self.last_pairs=(moved(self.start),moved(count[0]))
    def plan(ik,scene,record,meta,start,degrees,*a,**k):
        planned.append(degrees);end=np.asarray(start).copy();end[0]+=.01
        return np.vstack([start,end]),{'status':'solved'}
    def revise(service,pairs):
        revisions.append(count[0])
        np.testing.assert_allclose(pairs[0]['first'],moved(0))
        np.testing.assert_allclose(pairs[0]['second'],moved(2))
        np.testing.assert_allclose(pairs[1]['first'],moved(2))
        np.testing.assert_allclose(pairs[1]['second'],moved(3))
        return {'motion_supported':True,'applied':True}
    monkeypatch.setattr(module,'plan_signed_arc',plan)
    monkeypatch.setattr(feedback,'revise_from_feedback',revise)
    def issue(refs,gripper,label,**kwargs):
        robot.state['control_steps']+=len(refs);robot.state['metadata']['joint_positions']=refs[-1].tolist()
        if label.endswith('manipulation'):
            count[0]+=1;n=count[0]
            robot.motion_feedback.append({'label':label,'reference_observation_id':n+1,'current_observation_id':n+2,
                'first':moved(n-1),'second':moved(n),'tracker':Tracker(n-1)})
        return {'task_success':False,'episode_done':False,'observation':{},'grasp_evidence':{},
            'motion_effect':{'observed_region_rotation_deg':planned[-1] if planned else 0.}}
    robot.issue=issue
    result=robot.execute_sequence('test',close_steps=3,hold=0)
    assert revisions==[3]
    assert planned==[3.,3.,3.,15.,6.]
    assert result['stop_reason']=='requested_reference_motion_complete'


def test_arc_search_allowance_is_explicit_and_bounded_by_episode_time(tmp_path,monkeypatch):
    from test_robot_sequence import service
    module,robot=service(tmp_path);robot.scene=object();robot.record={}
    robot.spec={'joint_origin_world_m':[0,0,1],'joint_axis_world':[0,0,1]}
    robot.config={'arc_search_seconds':20.};robot.episode_status=lambda:{'remaining_wall_seconds':40.}
    monkeypatch.setattr(module,'measured_robot_scene',lambda s:s.scene)
    budgets=[]
    def plan(*args,config):
        budgets.append(config['arc_search_seconds'])
        return np.zeros((2,7)),{'status':'solved'}
    monkeypatch.setattr(module,'plan_signed_arc',plan)
    validate_operation('plan_arc',{'degrees':30,'budget_seconds':60})
    result=robot.plan_arc(30,budget_seconds=60)
    assert budgets==[40.] and result['constraints']['arc_search_budget_seconds']==40.
    assert robot.config['arc_search_seconds']==20.
