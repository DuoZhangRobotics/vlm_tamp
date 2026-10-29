import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from vlm_tamp.observed_video import sampled_steps
from vlm_tamp.perception_tools import PerceptionTools
from vlm_tamp.robot_contracts import RequestValidationError,validate_operation
from vlm_tamp.compact_robot_interface import compact_response


def test_temporal_sampling_preserves_depth_anchors_and_excludes_other_times():
    anchors=[10,31,48,66,67]
    assert sampled_steps(range(100),anchors,64)==list(range(10,68))
    sampled=sampled_steps(range(100),anchors,8)
    assert len(sampled)==8 and set(anchors)<=set(sampled)
    assert sampled==sorted(set(sampled)) and min(sampled)==10 and max(sampled)==67
    assert sampled_steps(range(100),anchors,8)==sampled
    with pytest.raises(ValueError,match='aligned'):sampled_steps([10,20,30],[10,21,30])
    with pytest.raises(ValueError,match='distinct'):sampled_steps([10],[10,10])
    with pytest.raises(ValueError,match='every'):sampled_steps(range(100),anchors,3)


def video_engine(tmp_path):
    public=tmp_path/'public';public.mkdir();video=tmp_path/'video';video.mkdir()
    records={};views=[];h,w=64,80;y,x=np.indices((h,w))
    for i,step in enumerate([10,20,30],1):
        folder=public/f'obs{i}';folder.mkdir()
        depth=np.full((h,w),1.+.1*i);world=np.stack([x/80.,y/80.,depth],-1)
        rgb=np.full((h,w,3),step,np.uint8)
        np.savez_compressed(folder/'sensors.npz',agentview_rgb=rgb,agentview_depth=depth,
            agentview_world_points=world,agentview_known_robot_depth_prior=np.zeros((h,w)))
        (folder/'calibration.json').write_text(json.dumps({'agentview':{'intrinsic':np.eye(3).tolist(),'camera_to_world':np.eye(4).tolist()}}))
        records[i]={'sensors':folder/'sensors.npz','calibration':folder/'calibration.json',
            'cameras':['agentview'],'attempt':1,'control_steps':step}
        views.append(world)
    for step in range(5,36):Image.fromarray(np.full((h,w,3),step,np.uint8)).save(video/f'step{step:04d}.jpg')
    class Tracker:
        def predict(self,frames,queries,reference_index):
            assert all(set(v)=={'rgb'} for v in frames)
            self.reference_index=reference_index;self.frame_count=len(frames)
            return np.tile(queries[None],(len(frames),1,1)),np.ones((len(frames),len(queries)),bool),{}
        def close(self):pass
    tracker=Tracker();engine=PerceptionTools(records,public/'perception',public,tracker=tracker,video_root=video)
    mask=np.zeros((h,w),bool);mask[10:50,10:60]=True
    engine.parts['part1']={'observation_id':2,'camera':'agentview','mask':mask}
    return engine,tracker,views


def test_live_video_lifts_only_exact_depth_timestamps(tmp_path):
    engine,tracker,views=video_engine(tmp_path)
    result=engine.track_video_motion(1,'part1',[1,2,3],max_rgb_frames=8)
    assert tracker.frame_count==8
    manifest=json.loads((engine.workspace/result['video_manifest']).read_text())
    steps=[f['control_step'] for f in manifest['frames']]
    assert steps[0]==10 and steps[-1]==30 and 20 in steps
    assert tracker.reference_index==steps.index(20)
    with np.load(engine.workspace/result['track_file']) as arrays:
        assert arrays['world_points'].shape[0]==3 and arrays['video_pixels'].shape[0]==8
        queries=arrays['agentview_queries'];xy=np.rint(queries).astype(int)
        for i in range(3):np.testing.assert_allclose(arrays['world_points'][i],views[i][xy[:,1],xy[:,0]])
    assert not result['robot_motion_executed'] and not result['planner_model_updated']
    compact=compact_response({'ok':True,'result':result})['result']
    assert compact['video_sampling']['selected_rgb_frames']==8
    assert compact['camera_reports'][0]['depth_anchors']==3
    assert compact['depth_control_steps']==[10,20,30]
    assert compact['video_manifest']==result['video_manifest']
    engine.close()


def model_service():
    from serve_subscription_robot import RobotService
    robot=RobotService.__new__(RobotService)
    robot.part_motion_tools=True;robot.spec={'joint_origin_world_m':[0.,0.,1.],
        'joint_axis_world':[0.,0.,-1.],'contact_marker':'retained'}
    robot.scene=object();robot.epoch=7;robot.state={'control_steps':30};robot.contact_scene_kind='articulated'
    value={'estimate':{'accepted':True,'joint_origin_world_m':[.1,.2,1.],'axis_world_unsigned':[0.,0.,1.]},
        'motion_supported':True,'status':'motion_supported','fit_observations':[1,2],
        'verify_observations':[2,3],'track_id':'tracks1','part_id':'part1'}
    robot._perception_tools=SimpleNamespace(estimates={'joint1':value},_record=lambda i:{'attempt':1,'control_steps':i*10})
    robot.plans={'old':{'kind':'arc','control_steps':30,'epoch':7}}
    robot._update_joint_parameters=lambda:None
    robot.export_geometry=lambda:{'model':copy.deepcopy(robot.spec),'geometry_epoch':robot.epoch}
    return robot,value


def test_estimate_application_preserves_direction_and_invalidates_old_plans():
    robot,_=model_service();result=robot.apply_joint_estimate('joint1')
    assert robot.spec['joint_origin_world_m']==[.1,.2,1.]
    assert robot.spec['joint_axis_world']==[0.,0.,-1.] and robot.spec['contact_marker']=='retained'
    assert robot.epoch==8 and result['hinge_update_applied'] and result['motion_supported']
    assert result['task_direction_verified_by_motion_fit'] is False
    assert result['axis_location_change_m']>=0
    assert 'unsigned axis' in result['direction_note']
    assert not result['robot_motion_executed'] and not result['provisional']
    with pytest.raises(ValueError,match='fresh plan'):robot.execute_plan('old')
    compact=compact_response({'ok':True,'result':result})['result']
    assert compact['applied_estimate_id']=='joint1' and compact['invalidated_old_plans']
    assert compact['source_control_steps']==[10,20,30]


@pytest.mark.parametrize('status',['motion_estimate_unverified','verification_inconclusive'])
def test_provisional_application_is_explicit(status):
    robot,value=model_service();value.update(status=status,motion_supported=False)
    with pytest.raises(ValueError,match='provisional'):robot.apply_joint_estimate('joint1')
    assert robot.epoch==7
    result=robot.apply_joint_estimate('joint1',allow_unverified=True)
    assert result['provisional'] and not result['motion_supported']


def test_contradicted_stale_and_unknown_estimates_do_not_change_model():
    robot,value=model_service();before=copy.deepcopy(robot.spec)
    value.update(status='verification_failed',motion_supported=False)
    with pytest.raises(ValueError,match='disagrees'):robot.apply_joint_estimate('joint1',allow_unverified=True)
    value.update(status='motion_supported',motion_supported=True);robot.state['control_steps']=40
    with pytest.raises(ValueError,match='stale'):robot.apply_joint_estimate('joint1')
    with pytest.raises(ValueError,match='active episode'):robot.apply_joint_estimate('other')
    assert robot.spec==before and robot.epoch==7


def test_failed_scene_rebuild_rolls_back_model_and_epoch():
    robot,_=model_service();old_spec=robot.spec;old_scene=robot.scene
    def fail():raise RuntimeError('No valid rebuilt scene')
    robot._update_joint_parameters=fail
    with pytest.raises(RuntimeError):robot.apply_joint_estimate('joint1')
    assert robot.spec is old_spec and robot.scene is old_scene and robot.epoch==7


def test_fixed_world_joint_can_be_reused_without_claiming_current_pose_evidence():
    robot,_=model_service();robot.state['control_steps']=40
    result=robot.apply_joint_estimate('joint1',reuse_fixed_joint=True)
    assert robot.spec['joint_origin_world_m']==[.1,.2,1.]
    assert result['historical_fixed_joint_reused']
    assert result['source_control_steps']==[10,20,30]
    assert not result['robot_motion_executed']


@pytest.mark.parametrize('override,expected',[(None,-1),(1,1),(-1,-1)])
def test_generic_joint_reinstall_keeps_retained_signed_axis(monkeypatch,override,expected):
    robot,_=model_service();robot.retained_articulated_model=robot.spec
    robot.spec=None;robot.scene=None;robot.contact_scene_kind='surface'
    received=[]
    def install(service,value,part,sign):
        received.append(sign);return {}
    monkeypatch.setattr('vlm_tamp.observed_joint_scene.install_joint_part',install)
    result=robot.apply_joint_estimate('joint1',motion_sign=override)
    assert received==[expected]
    assert result['signed_direction_preserved']==(override is None)


def test_manual_hinge_replacement_clears_measured_provenance():
    robot,_=model_service();robot.apply_joint_estimate('joint1')
    robot.set_model(axis_world=[0.,0.,1.])
    assert robot.spec['joint_estimation_evidence']['task_direction_updated_after_application']
    robot.set_model(hinge_world_m=[.3,.4,1.])
    assert 'joint_estimation_evidence' not in robot.spec
    assert robot.spec['hinge_evidence']['status']=='manual_hypothesis_not_independently_verified'


def test_new_request_contracts_reject_ambiguous_video_and_application_arguments():
    validate_operation('track_video_motion',{'part_id':'p','observation_ids':[1,3],'max_rgb_frames':64})
    validate_operation('apply_joint_estimate',{'estimate_id':'j','allow_unverified':True})
    with pytest.raises(RequestValidationError):validate_operation('track_video_motion',{'part_id':'p','observation_ids':[1,3],'max_rgb_frames':1000})
    with pytest.raises(RequestValidationError):validate_operation('apply_joint_estimate',{'estimate_id':'j','allow_unverified':'true'})
