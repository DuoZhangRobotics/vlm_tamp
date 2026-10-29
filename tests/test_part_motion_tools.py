import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.part_perception import association,compare_projected_mask
from vlm_tamp.part_motion import estimate_and_verify,lift_tracks,verify_upright_axis
from vlm_tamp.perception_tools import PerceptionTools
from vlm_tamp.robot_contracts import validate_operation,RequestValidationError


def view(*,depth=1.,offset=0.):
    h,w=64,80;K=np.array([[80,0,40],[0,80,32],[0,0,1.]])
    y,x=np.indices((h,w));world=np.stack([(x-40)*depth/80+offset,(y-32)*depth/80,np.full((h,w),depth)],-1)
    transform=np.eye(4);transform[0,3]=offset
    return {'rgb':np.zeros((h,w,3),np.uint8),'depth':np.full((h,w),depth),
            'world_points':world,'intrinsic':K,'camera_to_world':transform,
            'known_robot_depth_prior':np.zeros((h,w))}


def moving_points(second_pivot=None):
    rng=np.random.default_rng(10);pivot=np.array([.2,-.1])
    first=np.c_[rng.uniform(.38,.55,100),rng.uniform(-.12,.04,100),rng.uniform(.9,1.2,100)]
    def rotate(points,center,degrees):
        a=np.radians(degrees);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        result=points.copy();result[:,:2]=(points[:,:2]-center)@R.T+center;return result
    second=rotate(first,pivot,5);third=rotate(second,pivot if second_pivot is None else second_pivot,6)
    return np.stack([first,second,third]),np.ones((3,100),bool)


def test_association_uses_calibration_and_excludes_robot_pixels():
    source=view();target=view(offset=.05);points=source['world_points'][20:40,20:40].reshape(-1,3)
    target['known_robot_depth_prior'][25:30,20:25]=1.
    mask,report=association(points,target)
    expected=np.zeros((64,80),bool);expected[20:40,16:36]=True;expected[25:30,20:25]=False
    np.testing.assert_array_equal(mask,expected)
    assert report['robot_or_invalid_seed_pixels_removed']==25


def test_occlusion_is_not_reported_as_free_space_or_fused_surface():
    source=view();points=source['world_points'][20:30,20:30].reshape(-1,3)
    mask,hidden=association(points,view(depth=.8))
    assert not mask.any() and hidden['occluded_source_points']==100 and hidden['free_space_conflict_source_points']==0
    mask,free=association(points,view(depth=1.2))
    assert not mask.any() and free['occluded_source_points']==0 and free['free_space_conflict_source_points']==100


def test_mask_comparison_distinguishes_disagreement_and_unobservability():
    a=np.zeros((20,20),bool);a[2:10,2:10]=True
    b=np.zeros_like(a);b[10:18,10:18]=True
    assert compare_projected_mask(a,b)['interpretation']=='selected_masks_disagree'
    assert compare_projected_mask(np.zeros_like(a),b)['interpretation']=='insufficient_visible_correspondence'


def test_depth_lifting_handles_camera_motion_and_discards_invisible_tracks():
    views=[view(),view(offset=.05)]
    pixels=np.array([[[20,20],[30,30],[40,40]],[[16,20],[26,30],[36,40]]],float)
    visibility=np.ones((2,3),bool);visibility[1,1]=False
    views[1]['known_robot_depth_prior'][40,36]=1.
    world,valid=lift_tracks(views,pixels,visibility)
    np.testing.assert_allclose(world[0,0],world[1,0],atol=1e-12)
    assert valid[1].tolist()==[True,False,False]
    assert np.isnan(world[1,1:]).all()


def test_motion_recovers_axis_without_old_guess_and_checks_later_interval():
    world,valid=moving_points();result=estimate_and_verify(world,valid,[0,1],[1,2])
    np.testing.assert_allclose(result['estimate']['joint_origin_world_m'][:2],[.2,-.1],atol=1e-10)
    assert result['motion_supported'] and result['status']=='motion_supported'
    assert result['verification']['axis_prediction_rms_m']<1e-10
    assert 'no old hinge prior' in result['estimate']['initialization']


def test_wrong_static_axis_is_rejected_by_actual_later_motion():
    world,valid=moving_points()
    result=verify_upright_axis(world,valid,1,2,[.32,-.1,1.])
    assert not result['accepted'] and result['reason']=='axis_disagrees_with_held_out_motion'
    assert result['axis_prediction_rms_m']>.01


def test_changed_motion_axis_fails_later_interval_check():
    world,valid=moving_points(second_pivot=[.45,.2]);result=estimate_and_verify(world,valid,[0,1],[1,2])
    assert result['estimate']['accepted'] and not result['motion_supported']
    assert result['status']=='verification_failed'


@pytest.mark.parametrize('missing_evidence',['stationary','lost_tracks'])
def test_missing_later_motion_is_inconclusive_and_preserves_the_candidate(missing_evidence):
    from vlm_tamp.compact_robot_interface import compact_response
    world,valid=moving_points()
    if missing_evidence=='stationary':world[2]=world[1]
    else:valid[2]=False
    result=estimate_and_verify(world,valid,[0,1],[1,2])
    assert result['estimate']['accepted'] and not result['motion_supported']
    np.testing.assert_allclose(result['estimate']['joint_origin_world_m'][:2],[.2,-.1],atol=1e-10)
    assert result['status']=='verification_inconclusive'
    assert result['verification']['reason']=='verification_motion_not_observable'
    result['estimate_id']='joint1'
    compact=compact_response({'ok':True,'result':result})['result']
    assert compact['status']=='verification_inconclusive' and not compact['motion_supported']
    assert compact['estimate']['correspondences']==result['estimate']['correspondences']


def test_later_near_axis_points_can_verify_an_axis_initialized_from_a_wider_view():
    from vlm_tamp.part_motion import fit_upright
    from vlm_tamp.rigid_hinge_tracking import estimate_rigid_hinge
    outer,valid=moving_points();pivot=np.array([.2,-.1])
    angles=np.linspace(0,2*np.pi,80,endpoint=False)
    inner=np.c_[pivot[0]+.04*np.cos(angles),pivot[1]+.04*np.sin(angles),np.ones(80)]
    frames=[]
    for degrees in [0,5,11]:
        a=np.radians(degrees);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        p=inner.copy();p[:,:2]=(inner[:,:2]-pivot)@R.T+pivot;frames.append(p)
    world=np.concatenate([outer,np.stack(frames)],axis=1);visibility=np.ones((3,180),bool)
    visibility[0,100:]=False;visibility[2,:100]=False
    later=fit_upright(world,visibility,1,2)
    assert later['pose_accepted'] and later['accepted'] and later['radius_m']<.06
    original=estimate_rigid_hinge(world[1,100:],world[2,100:],prior=None)
    assert original['pose_accepted'] and not original['accepted']
    result=estimate_and_verify(world,visibility,[0,1],[1,2])
    assert result['motion_supported'] and result['verification']['axis_prediction_rms_m']<1e-10


@pytest.mark.parametrize('translation',[False,True])
def test_stationary_or_pure_translation_does_not_produce_confident_hinge(translation):
    world,valid=moving_points();world[1]=world[0];world[2]=world[0]
    if translation:world[1,:,:2]+=[.02,.01];world[2,:,:2]+=[.04,.02]
    result=estimate_and_verify(world,valid,[0,1],[1,2])
    assert not result['motion_supported'] and not result['estimate']['accepted']


def test_verification_cannot_reuse_training_interval():
    world,valid=moving_points()
    with pytest.raises(ValueError,match='later motion interval'):estimate_and_verify(world,valid,[0,1],[0,1])


class FakeSegmenter:
    def predict(self,view,key,points,labels,box):
        mask=np.zeros(view['rgb'].shape[:2],bool);mask[20:40,20:40]=True
        return np.array([mask]),[.9],{'model_load':0.}
    def close(self):pass


def engine(tmp_path):
    public=tmp_path/'public';public.mkdir();frame=public/'obs';frame.mkdir()
    v=view();arrays={'a_'+k:v[k] for k in ['rgb','depth','world_points','known_robot_depth_prior']}
    np.savez_compressed(frame/'sensors.npz',**arrays)
    (frame/'calibration.json').write_text(json.dumps({'a':{k:np.asarray(v[k]).tolist() for k in ['intrinsic','camera_to_world']}}))
    record={'sensors':frame/'sensors.npz','cameras':['a'],'attempt':1,
            'sensor_sha256':hashlib.sha256((frame/'sensors.npz').read_bytes()).hexdigest()}
    return PerceptionTools({1:record},public/'perception',public,segmenter=FakeSegmenter()),record


def test_part_ids_reuse_measured_masks_and_reject_unknown_references(tmp_path):
    tools,record=engine(tmp_path)
    result=tools.segment_parts(2,1,'a',box_xyxy=[10,10,50,50],label='caller hypothesis')
    part=result['parts'][0];assert part['usable_depth_pixels']==400
    association=tools.associate_views(3,part['part_id'],['a'])
    assert association['views'][0]['supported_pixels']==400
    with pytest.raises(ValueError,match='Unknown part'):tools.associate_views(4,'another-episode',['a'])
    tools.close()


def test_modified_sensor_file_cannot_enter_perception(tmp_path):
    tools,record=engine(tmp_path);Path(record['sensors']).write_bytes(b'not the captured sensors')
    with pytest.raises(ValueError,match='changed after capture'):
        tools.segment_parts(2,1,'a',box_xyxy=[10,10,50,50])


def test_repeated_observation_camera_reuses_the_loaded_sensor_snapshot(tmp_path,monkeypatch):
    tools,_=engine(tmp_path);original_load=np.load;loads=[]
    def load(path,*args,**kwargs):
        loads.append(str(path));return original_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',load)
    first=tools._view(1,'a');again=tools._view(1,'a')
    assert first is again
    assert len(loads)==1
    assert list(tools.cache)==[(1,'a')]


def test_cache_keeps_observations_and_cameras_separate(tmp_path):
    tools,record=engine(tmp_path)
    first=view();other=view(offset=.05);other['rgb'][:]=17
    arrays={camera+'_'+name:value[name] for camera,value in [('a',first),('b',other)]
            for name in ['rgb','depth','world_points','known_robot_depth_prior']}
    np.savez_compressed(record['sensors'],**arrays)
    calibration=Path(record['sensors']).with_name('calibration.json')
    calibration.write_text(json.dumps({camera:{name:np.asarray(value[name]).tolist() for name in ['intrinsic','camera_to_world']}
        for camera,value in [('a',first),('b',other)]}))
    record.update(cameras=['a','b'],sensor_sha256=hashlib.sha256(Path(record['sensors']).read_bytes()).hexdigest())
    later=view(depth=1.2);later['rgb'][:]=29
    folder=tools.workspace/'later';folder.mkdir()
    np.savez_compressed(folder/'sensors.npz',**{'a_'+name:later[name] for name in ['rgb','depth','world_points','known_robot_depth_prior']})
    (folder/'calibration.json').write_text(json.dumps({'a':{name:np.asarray(later[name]).tolist() for name in ['intrinsic','camera_to_world']}}))
    tools.observations[2]={'sensors':folder/'sensors.npz','cameras':['a'],'attempt':1}
    a=tools._view(1,'a');b=tools._view(1,'b');c=tools._view(2,'a')
    assert a['rgb'][0,0,0]==0 and b['rgb'][0,0,0]==17 and c['rgb'][0,0,0]==29
    assert a is tools._view(1,'a') and b is tools._view(1,'b') and c is tools._view(2,'a')
    assert set(tools.cache)=={(1,'a'),(1,'b'),(2,'a')}
    np.testing.assert_allclose(c['depth'],1.2)


def test_schema_and_compaction_preserve_measurement_scope():
    from vlm_tamp.compact_robot_interface import compact_response
    validate_operation('track_part_motion',{'part_id':'p','observation_ids':[1,2],'backend':'cotracker3'})
    with pytest.raises(RequestValidationError):validate_operation('segment_parts',{'observation_id':1,'camera':'a','points_xy':[[float('nan'),1]]})
    world,valid=moving_points();value=estimate_and_verify(world,valid,[0,1])
    value.update(estimate_id='joint1',semantic_identity_verified=False,physical_contact_verified=False)
    result=compact_response({'ok':True,'result':value})['result']
    assert result['status']=='motion_estimate_unverified' and not result['motion_supported']
    assert not result['semantic_identity_verified'] and not result['physical_contact_verified']
    assert result['estimate']['joint_origin_world_m']==value['estimate']['joint_origin_world_m']


def test_service_feature_is_explicitly_enabled_before_loading_any_backend():
    from serve_subscription_robot import RobotService
    service=RobotService.__new__(RobotService);service.sim=object()
    with pytest.raises(ValueError,match='not enabled'):service._part_tool('segment_parts',{})


def test_lk_can_track_backward_from_a_later_visible_reference():
    import cv2
    from vlm_tamp.part_motion import lk_tracks
    rng=np.random.default_rng(3);rgb=rng.integers(0,256,size=(64,80,3),dtype=np.uint8)
    frames=[]
    for shift in [0,2,4]:
        v=view();v['rgb']=cv2.warpAffine(rgb,np.array([[1.,0,shift],[0,1,0]]),(80,64));frames.append(v)
    points=np.array([[34,25],[44,35],[54,40]],np.float32)
    tracks,visible=lk_tracks(frames,points,reference_index=2)
    assert visible.all()
    np.testing.assert_allclose(tracks[0],points-[4,0],atol=.06)
    np.testing.assert_array_equal(tracks[2],points)


def test_cotracker_receives_query_frame_and_backward_flag_without_extra_model_loading():
    import torch
    from vlm_tamp.part_motion import CoTrackerBackend
    calls={}
    def model(video,queries,backward_tracking):
        calls.update(queries=queries.clone(),backward=backward_tracking)
        return queries[:,:,1:][:,None].expand(1,len(video[0]),-1,-1),torch.ones((1,len(video[0]),len(queries[0])),dtype=torch.bool)
    backend=CoTrackerBackend(device='cpu');backend.model=model
    tracks,visible,_=backend.predict([view(),view(),view()],np.array([[20,20]],np.float32),reference_index=2)
    assert calls['backward'] and calls['queries'][0,0,0].item()==2
    assert tracks.shape==(3,1,2) and visible.all()


def test_registry_accepts_later_reference_and_preserves_chronological_frames(tmp_path):
    tools,record=engine(tmp_path);tools.observations.update({2:dict(record),3:dict(record)})
    part=tools.segment_parts(4,3,'a',box_xyxy=[10,10,50,50])['parts'][0]['part_id']
    result=tools.track_part_motion(5,part,[1,2,3],backend='lk')
    assert result['reference_observation_id']==3 and result['reference_frame_index']==2
    assert result['observation_ids']==[1,2,3]
    from vlm_tamp.compact_robot_interface import compact_response
    compact=compact_response({'ok':True,'result':result})['result']
    for full,shown in zip(result['interval_motion'],compact['interval_motion']):
        assert full['fit_correspondences']==full['fit_inliers']==0
        assert full['fit_rms_m'] is None
        assert 'six_moving_landmarks' in full['fit_failed_checks']
        for key in ['fit_correspondences','fit_inliers','fit_rms_m','fit_failed_checks']:
            assert shown[key]==full[key]
    with pytest.raises(ValueError,match='ordered distinct'):
        tools.track_part_motion(6,part,[3,2,1],backend='lk')
