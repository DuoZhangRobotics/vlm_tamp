"""Exercise complete skill control flow without GPU or simulator startup."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

import vlm_tamp.contact_skill_service as module
import vlm_tamp.contact_skills as geometry


class Chain:
    def forward(self,q):return np.r_[np.asarray(q)[:3],[0,0,0,1.]]


@pytest.mark.parametrize('reuse',[False,True])
@pytest.mark.parametrize('native_success',[False,True])
@pytest.mark.parametrize('unresolved_pulse',[False,True])
def test_advisory_release_flow_and_native_success_termination(tmp_path,monkeypatch,reuse,native_success,unresolved_pulse):
    chain=Chain()
    scene=SimpleNamespace(static_points=np.empty((0,3)),moving_reference=np.array([[0.,0.,0.]]),
        contact_reference=np.array([[0.,0.,0.]]),axis=np.array([0.,0.,1.]),origin=np.zeros(3))
    evidence={'point_world_m':[0.,0.,0.],'normal_toward_camera_world':[-1.,0.,0.],
        'direction_world':[1.,0.,0.],'contact_surface':'fingertips','joint_associated':False}
    monkeypatch.setattr(module,'build_contact_scene',lambda *a,**k:(scene,evidence.copy(),np.zeros((12,3))))
    monkeypatch.setattr(module,'_overlay',lambda *a,**k:None)
    monkeypatch.setattr(module,'momentum_status',lambda:{'available':True,'validated':False,'validation_reason':'not_validated'})
    empty=np.empty((0,7));one=np.zeros((1,7))
    engagement={'path':np.zeros((2,7)),'references':empty if reuse else np.zeros((2,7)),
        'free_references':empty if reuse else one,'insertion_references':empty if reuse else one,
        'contact_local':np.zeros(3),'contact_world':np.zeros(3),'rotation':np.eye(3),
        'insertion_direction':np.array([1.,0.,0.]),'contact_name':'configured_pad',
        'attempts':[],'candidate_index':0,'timing_seconds':{},'total_seconds':0.,
        'native_contact_policy':{},'reuse_existing_contact':reuse}
    monkeypatch.setattr(module,'plan_engagement',lambda *a,**k:engagement)
    monkeypatch.setattr(module,'refresh_collision_observations',lambda s,*a:(s,{'source':'test observation'}))
    class Tracker:
        def __init__(self,cameras,points,step):
            self.step=step;self.tracker=SimpleNamespace(references={n:{'pixels':np.zeros((20,1,2))} for n in cameras})
        def observe(self,cameras,step):
            if unresolved_pulse and service.last_label.startswith('push-stroke'):
                return {'accepted':False,'pairs':7,'reason':'insufficient_visible_matches'}
            transform=np.eye(4);transform[0,3]=.0015*(step-self.step)
            return {'accepted':True,'transform':transform.tolist(),'source':'fixture measured motion'}
    monkeypatch.setattr(module,'ContactTracker',Tracker)
    class Check:
        def __init__(self,s):self.scene=s;self.record={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):
            return {'accepted':getattr(self.scene,'skill_collision_policy','veto')!='advisory',
                'static_rejections':1,'joint_bound_violations':[],'timed_out':False,
                'blocking_robot_part':'configured_finger','blocking_surface':'observed stationary geometry'}
    make_check=lambda s,*a,**k:Check(s)
    monkeypatch.setattr(module,'skill_native_scene',make_check)
    monkeypatch.setattr(geometry,'skill_native_scene',make_check)

    class Service:
        def __init__(self):
            self.workspace=tmp_path/'workspace';self.workspace.mkdir()
            self.public=self.workspace/'robot_data';self.public.mkdir()
            self.episode=tmp_path/'attempt01';self.episode.mkdir()
            self.sim=object();self.ik=SimpleNamespace(chain=chain,self_valid=lambda q:True,
                solve_many=lambda pose,seed,**kwargs:[{'joints':np.r_[pose[:3],np.zeros(4)]}])
            self.state={'control_steps':60,'metadata':{'joint_positions':[0.]*7,
                'eef_pose_world_xyzw':[0,0,0,0,0,0,1],'gripper_qpos':[.04,-.04]}}
            self.latest_observation={'observation_id':1,'control_steps':60}
            self.cameras={'test':{}};self.scene=scene;self.spec={'joint_axis_world':[0,0,-1]};self.record={}
            self.anchors=[[0,0,0]];self.scene_reference_control_step=60
            self.calls=1;self.attempt=1;self.epoch=0;self.events=[]
            self.retained_articulated_model=None;self.contact_scene_kind='articulated'
        def episode_status(self):return {'remaining_wall_seconds':1000}
        def issue(self,refs,gripper,label,**kwargs):
            assert gripper==0 and len(refs)>0
            self.last_label=label
            for q in refs:self.events.append({'measured_joints':q.tolist()})
            self.state['control_steps']+=len(refs)
            self.state['metadata']['joint_positions']=refs[-1].tolist()
            self.state['metadata']['eef_pose_world_xyzw']=chain.forward(refs[-1]).tolist()
            success=native_success and label.startswith('push-stroke')
            if success:self.sim=None;self.ik=None
            return {'execution':{'issued_commands':len(refs),'all_commands_issued':True},
                'observation':{'control_steps':self.state['control_steps'],'observation_id':self.calls},
                'task_success':success,'episode_done':success}
        def capture(self):raise AssertionError('Advisory scene warnings must not force a perception recovery gate')
    service=Service()
    result=module.run_skill(service,'push',observation_id=1,camera='test',region_xyxy=[1,1,6,6],
        mode='push_and_release',contact_speed_m_s=.12,collision_policy='advisory')
    names=[s['stage'] for s in result['stages']]
    assert result['robot_motion_executed']
    assert result['environment_collision_predictions']['count']>0
    if native_success:
        assert result['task_success'] and result['stop_reason']=='native_task_success'
        assert 'push-release-coasting' not in names
        pulse=result['stages'][-1]
        assert pulse['requested_contact_speed_m_s']==.12
        assert pulse['requested_motion_m']==pytest.approx(.03)
        assert 'planning_timing_seconds' in pulse
    else:
        assert result['status']=='push_and_release_complete'
        assert service.scene is None and service.spec is None
        assert service.retained_articulated_scene['reference_control_step']==60
        assert result['retained_scene_reference_control_step']==60
        assert result['next_geometry_interfaces'][0].startswith('apply_part_motion')
        assert names[-2:]==['push-release-withdrawal','push-release-coasting']
        assert result['stages'][-1]['execution']['issued_commands']==20
        if unresolved_pulse:
            assert result['release_motion_reference']=='before_pulse'
            assert result['motion_warnings'][0]['reason']=='object_motion_unresolved'
    assert ('free-approach' in names)==(not reuse)
