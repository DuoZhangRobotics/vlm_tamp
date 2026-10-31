from types import SimpleNamespace
import copy

import numpy as np
import pytest

import vlm_tamp.contact_skill_service as service_module
from vlm_tamp.coupled_scene import Scene
from vlm_tamp.push_contact_state import clear_motion_context,observed_contact_cloud
from vlm_tamp.robot_contracts import validate_operation
from vlm_tamp.rekep_libero.scene_state_v7 import voxel_points


class Chain:
    def forward(self,q):return np.r_[np.asarray(q)[:3],[0.,0.,0.,1.]]


@pytest.fixture
def contact_service(tmp_path,monkeypatch):
    spec={'joint_origin_world_m':[-.1,0,0],'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':[0,0,0]},'panel_normal_world':[-1,0,0]}
    target=np.array([[0.,0,0],[0,.01,0],[0,0,.01]])
    raw=Scene.from_points([[1.,1.,1.]],target,target,spec)
    counts={'builds':0,'searches':0,'strokes':0}
    def build(*args,**kwargs):
        counts['builds']+=1;scene=copy.copy(raw);scene.skill_contact_surface=args[6]
        joint=args[8] is not None
        return scene,{'point_world_m':[0.,0,0],'normal_toward_camera_world':[-1.,0,0],
            'direction_world':[0.,1.,0] if joint else list(args[7] or [1.,0,0]),
            'contact_surface':args[6],'joint_associated':joint},target
    def engage(*args,**kwargs):
        counts['searches']+=1;empty=np.empty((0,7))
        return {'path':np.zeros((2,7)),'references':empty,'free_references':empty,'insertion_references':empty,
            'contact_local':np.zeros(3),'contact_world':np.zeros(3),'rotation':np.eye(3),
            'insertion_direction':np.array([1.,0,0]),'contact_name':'gripper0_finger1_collision_tip_support',
            'reuse_existing_contact':True,'timing_seconds':{},'attempts':[],'total_seconds':0.}
    def plan(ik,scene,record,metadata,local,delta,budget,**kwargs):
        counts['strokes']+=1
        assert scene.current_robot_filter['reference_control_step']==svc.state['control_steps']
        start=np.asarray(metadata['joint_positions']);goal=start.copy();goal[:3]+=delta
        return np.vstack([start,goal]),{'orientation_candidate':0}
    class Tracker:
        def __init__(self,*args):
            self.reference=svc.object_position.copy()
            self.tracker=SimpleNamespace(references={'test':{'pixels':np.zeros((20,2))}})
        def observe(self,*args):
            if svc.fail_tracking:return {'accepted':False,'reason':'inconsistent_matches','pairs':10}
            transform=np.eye(4);transform[:3,3]=svc.object_position-self.reference
            return {'accepted':True,'transform':transform.tolist(),'pairs':20}
    monkeypatch.setattr(service_module,'build_contact_scene',build)
    monkeypatch.setattr(service_module,'plan_engagement',engage)
    monkeypatch.setattr(service_module,'plan_contact_stroke',plan)
    monkeypatch.setattr(service_module,'observed_contact_cloud',lambda service:((target,{}),False))
    monkeypatch.setattr(service_module,'ContactTracker',Tracker)
    monkeypatch.setattr(service_module,'_overlay',lambda *a,**k:None)
    monkeypatch.setattr(service_module,'momentum_status',lambda:{'available':True})
    class Service:
        def __init__(self):
            self.workspace=tmp_path/'workspace';self.public=self.workspace/'robot_data'
            self.episode=tmp_path/'attempt01';self.episode.mkdir()
            self.sim=object();self.ik=SimpleNamespace(chain=Chain());self.record={};self.config={}
            self.state={'control_steps':60,'metadata':{'joint_positions':[0.]*7,'gripper_qpos':[.04,-.04],
                'eef_pose_world_xyzw':[0,0,0,0,0,0,1],'robot_collision_volumes_world':[]}}
            self.latest_observation={'observation_id':1,'control_steps':60};self.cameras={'test':{}}
            self.scene=raw;self.spec=spec;self.anchors=[];self.scene_reference_control_step=60
            self.contact_scene_kind='articulated';self.retained_articulated_model=None
            self.calls=0;self.attempt=1;self.epoch=0;self.events=[]
            self.object_position=np.zeros(3);self.object_fraction=1.;self.fail_tracking=False
            self.tracking_lag_m=0.
        def episode_status(self):return {'remaining_wall_seconds':1000,'episode_done':False}
        def capture(self):
            self.latest_observation={'observation_id':self.calls,'control_steps':self.state['control_steps']}
            return self.latest_observation
        def issue(self,refs,gripper,label,**kwargs):
            clear_motion_context(self)
            before=np.asarray(self.state['metadata']['joint_positions'])
            actual=refs[-1].copy();delta=actual[:3]-before[:3];length=np.linalg.norm(delta)
            if length:actual[:3]-=min(self.tracking_lag_m,length)*delta/length
            self.object_position+=self.object_fraction*(actual[:3]-before[:3])
            self.state['metadata']['joint_positions']=actual.tolist()
            self.state['metadata']['eef_pose_world_xyzw']=self.ik.chain.forward(actual).tolist()
            self.state['control_steps']+=len(refs)
            self.events.extend({'measured_joints':q.tolist()} for q in refs)
            return {'execution':{'issued_commands':len(refs),'all_commands_issued':True},
                'observation':self.capture(),'episode_done':False,'task_success':False}
    svc=Service()
    return svc,counts,raw


def call(service,*,continued=False,**kwargs):
    service.calls+=1
    args={'direction_world':[1.,0,0],'distance_m':.01}
    if not continued:args.update(observation_id=service.latest_observation['observation_id'],camera='test',region_xyxy=[1,1,10,10])
    args.update(kwargs)
    return service_module.run_skill(service,'push',continuation=continued,**args)


def test_continued_push_reuses_contact_scene_and_reports_hand_vs_object_motion(contact_service):
    service,counts,raw=contact_service
    initial=call(service)
    assert initial['continue_push_available'] and not initial['scene_refresh_required']
    assert initial['grasp_scene_refresh_required'] and service.scene is None
    service.object_fraction=.8
    result=call(service,continued=True,distance_m=.03)
    assert result['stop_reason']=='requested_stroke_complete'
    assert result['commanded_contact_distance_m']==pytest.approx(.03)
    assert result['measured_contact_progress_m']==pytest.approx(.03)
    assert result['observed_progress']==pytest.approx(.024)
    assert result['full_scene_builds']==result['contact_searches']==0
    assert result['contact_reused'] and result['scene_reused']
    assert result['timing_seconds'].get('perception_seconds',0)==0
    assert counts['builds']==counts['searches']==1
    assert service.active_push_contact['scene'].static_points is raw.static_points
    assert result['continuation']['contact_id']==initial['continuation']['contact_id']
    assert all(not stage['scene_update']['full_scene_rebuilt'] for stage in result['stages'])


def test_preview_execution_does_not_repeat_scene_build_or_contact_search(contact_service):
    service,counts,_=contact_service
    preview=call(service,preview=True,budget_seconds=30.)
    assert not preview['robot_motion_executed'] and counts['builds']==counts['searches']==1
    result=call(service,budget_seconds=60.)
    assert result['preview_reused'] and result['robot_motion_executed']
    assert result['full_scene_builds']==result['contact_searches']==0
    assert counts['builds']==counts['searches']==1


def test_continuation_is_one_requested_stroke_by_default(contact_service):
    service,counts,_=contact_service;call(service);before=counts['strokes']
    service.tracking_lag_m=.0025
    result=call(service,continued=True,distance_m=.03)
    assert counts['strokes']==before+1
    assert result['measured_contact_progress_m']==pytest.approx(.0275)
    assert result['stop_reason']=='requested_stroke_complete'


def test_explicit_small_chunks_do_not_accumulate_tracking_loss(contact_service):
    service,counts,_=contact_service;call(service)
    service.tracking_lag_m=.0025
    result=call(service,continued=True,distance_m=.03,stroke_length_m=.01)
    assert result['measured_contact_progress_m']==pytest.approx(.0275)
    assert result['stop_reason']=='requested_stroke_complete'
    assert len(result['stages'])==4


def test_changed_model_does_not_reuse_old_preview(contact_service):
    service,counts,_=contact_service
    call(service,preview=True);service.epoch+=1
    result=call(service)
    assert not result['preview_reused'] and counts['builds']==2 and counts['searches']==2


def test_observe_alone_keeps_contact_but_unrelated_motion_clears_it(contact_service):
    service,counts,_=contact_service;call(service)
    service.calls+=1;service.capture()
    assert call(service,continued=True)['continue_push_available']
    q=np.asarray(service.state['metadata']['joint_positions']);service.issue(q[None],0,'unrelated')
    with pytest.raises(ValueError,match='No retained push contact'):call(service,continued=True)
    assert counts['builds']==1


def test_unresolved_motion_does_not_retain_a_stale_continuation(contact_service):
    service,_,_=contact_service;call(service);service.fail_tracking=True
    result=call(service,continued=True)
    assert result['stop_reason']=='object_motion_unresolved'
    assert not result['continue_push_available'] and service.active_push_contact is None


def test_ineffective_strokes_accumulate_across_continuation_calls(contact_service):
    service,_,_=contact_service;call(service);service.object_fraction=0.
    first=call(service,continued=True)
    assert first['continue_push_available'] and first['continuation']['ineffective_strokes']==1
    second=call(service,continued=True)
    assert second['stop_reason']=='no_progress_after_stroke_budget'
    assert not second['continue_push_available'] and service.active_push_contact is None


def test_bad_hinge_does_not_discard_measured_pose_for_direction_only_continuation(contact_service):
    service,counts,_=contact_service
    initial=call(service,direction_world=None,joint_degrees=15.,joint_model_epoch=0)
    assert initial['stop_reason']=='motion_inconsistent_with_fixed_joint'
    assert initial['continue_push_available'] and not initial['joint_model_supported']
    result=call(service,continued=True,direction_world=[0,1,0])
    assert result['stop_reason']=='requested_stroke_complete'
    assert counts['builds']==counts['searches']==1


def test_continuation_schema_needs_only_direction_and_distance():
    validate_operation('continue_push',{'direction_world':[1,0,0],'distance_m':.02})
    with pytest.raises(ValueError):validate_operation('continue_push',{'direction_world':[1,0,0]})


def test_observation_cache_is_one_entry_per_current_capture(monkeypatch):
    import vlm_tamp.push_contact_state as module
    calls=[];monkeypatch.setattr(module,'observed_points',lambda *a,**k:(calls.append(1) or np.zeros((2,3)),{}))
    service=SimpleNamespace(attempt=1,latest_observation={'observation_id':1},state={'control_steps':60,'metadata':{}},cameras={})
    first,reused=observed_contact_cloud(service);assert not reused
    second,reused=observed_contact_cloud(service);assert reused and first is second and len(calls)==1
    service.latest_observation={'observation_id':2}
    third,reused=observed_contact_cloud(service);assert not reused and len(calls)==2
    assert service.contact_observation_cache['value'] is third


@pytest.mark.parametrize('spacing',[.002,.003,.1])
def test_fast_voxels_preserve_exact_original_samples_and_order(spacing):
    rng=np.random.default_rng(17);points=rng.normal(size=(2000,3));points=np.vstack([points,points[:700],points[:700]+spacing*.01,[np.nan,0,0]])
    finite=points[np.isfinite(points).all(1)];cells=np.floor(finite/spacing).astype(np.int64)
    expected=finite[np.unique(cells,axis=0,return_index=True)[1]]
    assert np.array_equal(voxel_points(points,spacing),expected)
    assert voxel_points([],spacing).shape==(0,3)
