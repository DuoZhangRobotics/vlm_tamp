import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from vlm_tamp.region_decision import parse_decision,pixels,validate_region
from vlm_tamp.region_episode import RegionTools,search_coverage
from vlm_tamp.coupled_storage import read_json,write_json

def region():return {'camera':'agentview','contact_uv':[.5,.5],'contact_box_uv':[.45,.45,.55,.55],
    'moving_panel_box_uv':[.3,.3,.8,.8],'contact_kind':'surface'}
def decision():return {'version':'strategy-v3','operation':'revise_region','contact_mode':None,
    'approach_tilt_deg':None,'motion_extent':None,'step_mm':None,'posture_rank':None,'retreat_mm':None,
    'region':region(),'evidence':'The current region needs review.','expected_observation':'A new sensor-supported region.'}

def test_coordinates_are_camera_local_and_do_not_use_sheet_offsets():
    point,box,panel=pixels(region(),512,512)
    assert point==[256,256] and box==[230,230,281,281]
    other=region();other['camera']='frontview'
    assert pixels(other,512,512)==(point,box,panel)

@pytest.mark.parametrize('key,value',[('contact_uv',[347,171]),('contact_uv',[True,.5]),
    ('contact_uv',[.9,.9]),('contact_box_uv',[.6,.6,.4,.4]),('camera','simulator'),('contact_kind','ignore_collision')])
def test_region_schema_rejects_unbounded_or_inconsistent_requests(key,value):
    d=decision();d['region'][key]=value
    with pytest.raises(ValueError):parse_decision(json.dumps(d))

def test_revision_cannot_smuggle_motion_or_collision_policy():
    d=decision();d['step_mm']=25
    with pytest.raises(ValueError):parse_decision(json.dumps(d))
    d=decision();d['region']['ignore_parts']=['arm']
    with pytest.raises(ValueError):parse_decision(json.dumps(d))

def test_revision_requires_release_and_does_not_move_robot(tmp_path):
    s=object.__new__(RegionTools);s.current_choice={'contact_mode':'pinch'};s.region_epoch=0
    s.sim=SimpleNamespace(action=Mock())
    result=s.dispatch(decision(),tmp_path)
    assert result['status']=='region_revision_requires_release' and not result['manipulation_executed']
    s.sim.action.assert_not_called()

def test_failed_revision_is_transactional(tmp_path,monkeypatch):
    from vlm_tamp import region_episode as module
    s=object.__new__(RegionTools);s.current_choice=None;s.region_epoch=2;s.spec={'target_progress':.1}
    s.cloud=np.zeros((30,3));s.progress=SimpleNamespace(progress=.05);s.resources=SimpleNamespace(grasps=Mock())
    monkeypatch.setattr(module,'revised_geometry',Mock(side_effect=ValueError('No depth')))
    result=s.dispatch(decision(),tmp_path)
    assert result['status']=='region_revision_rejected' and s.region_epoch==2 and s.progress.progress==.05
    assert s.spec=={'target_progress':.1};s.resources.grasps.assert_not_called()

def test_coverage_reports_tested_postures_not_cumulative_counter_sums(tmp_path):
    write_json(tmp_path/'contact-menu.json',{'candidates':[{'approach_tilt_deg':30}]})
    write_json(tmp_path/'contact-mode-admission.json',{'attempts':[
        {'candidate':0,'ik_rank':0,'reason':'contact_pose_collision','diagnostic':{'static_rejections':1}},
        {'candidate':0,'ik_rank':1,'reason':'contact_pose_collision','diagnostic':{'static_rejections':2}}]})
    d={'operation':'try_contact','contact_mode':'push','step_mm':5,'posture_rank':None}
    c=search_coverage(tmp_path,d,2,60)
    assert c['tested_contact_postures']==[{'tilt_deg':30.,'ik_ranks':[0,1],
        'rejection_records':{'contact_pose_collision':2}}]
    assert 'static_rejections' not in json.dumps(c)

def test_native_collision_preview_matches_real_rejection_without_changing_it(tmp_path):
    from vlm_tamp.cohort_execution import read_cohort_observation
    from vlm_tamp.reliability_geometry import SurfaceScene
    from vlm_tamp.mcvamp_mechanism import NativeScene,clear_native_cache
    from vlm_tamp.collision_evidence import capture_rejections,save_capture
    root=Path(__file__).resolve().parents[1];source=root/'results/vlm-recovery-frozen-20261018/close_bottom_drawer-state017-vlm'
    meta=read_json(source/'opened/metadata.json');views,_=read_cohort_observation(source/'opened/obs.npz',meta)
    config=read_json(source.parent/'protocol.json')['configuration']['close_bottom_drawer']
    config={**config,'collision_evidence_views':True};spec=read_json(source/'task-spec.json')
    scene=SurfaceScene(views,meta,spec,np.load(source/'grasp-cloud.npy'),np.load(source/'panel-mask.npy'),config)
    record=read_json(source/'robot-chain.json');q0=np.asarray(meta['joint_positions'])
    rejected=read_json(source/'decisions/round-01/grasp-admission.json') if (source/'decisions/round-01/grasp-admission.json').exists() else None
    if rejected is None:
        source=root/'results/astra-recovery-physical-20261018/close_bottom_drawer-state017-shared'
        rejected=read_json(source/'decisions/round-03/grasp-admission.json')
    row=next(r for r in rejected['rejected_candidates'] if r.get('reason')=='open_grasp_scene_collision')
    q=np.asarray(row['joints'])
    with NativeScene(scene,record,meta,q0,.03,articulated=False) as base:original=base.validate(q)
    with capture_rejections() as captured:
        with NativeScene(scene,record,meta,q0,.03,articulated=False) as instrumented:actual=instrumented.validate(q)
    assert actual==original and not actual['accepted']
    report=save_capture(captured,views,tmp_path)
    assert len(report['representative_samples'])==1
    sample=report['representative_samples'][0]
    assert sample['reported_part']==actual['last_rejected_part'] and 1<=sample['reported_part']<37
    assert sample['reported_surface']==0 and sample['witness_verified']
    assert sample['sphere_clearance_or_max_halfspace_residual_m']<=0
    assert not sample['simulator_object_truth_used']
    clear_native_cache()

def test_surface_frame_does_not_rotate_with_arbitrary_patch_aspect_ratio():
    from vlm_tamp.shared_contact import contact_frame
    spec={'joint_family':'prismatic','joint_axis_world':[0,1,0],
          'panel_normal_world':[0,-1,0],'contact_kind':'surface'}
    for width,height in ((.04,.015),(.015,.04)):
        cloud=np.array([[x,0,z] for x in np.linspace(-width,width,8) for z in np.linspace(-height,height,8)])
        frame=contact_frame(cloud,spec)
        assert abs(frame['bar'][2])<1e-10 and frame['across'][2]>.99

def test_current_region_overlay_tracks_motion_instead_of_old_pixels(monkeypatch):
    from vlm_tamp import region_episode as module
    monkeypatch.setattr(module,'project',lambda points,camera:np.asarray(points)[:,:2]*511)
    cloud=np.array([[x,y,1.] for x in (.2,.3) for y in (.3,.4)])
    scene=SimpleNamespace(moving_reference=cloud.copy(),panel_origin=np.array([0,0,1.]),panel_normal=np.array([0,0,1.]))
    session=SimpleNamespace(selection=region(),views={'agentview':{'rgb':np.zeros((512,512,3),np.uint8)}},
        cloud=cloud.copy(),scene=scene,spec={'source_anchor_world_m':[.25,.35,1.]},state={'control_steps':100})
    before=module.projected_region(session)
    session.cloud[:,0]+=.1;scene.moving_reference[:,0]+=.1;session.spec['source_anchor_world_m'][0]+=.1
    session.state['control_steps']=110
    after=module.projected_region(session)
    assert np.allclose(np.array(after['contact_uv'])-before['contact_uv'],[.1,0])
    assert np.allclose(np.array(after['moving_panel_box_uv'])-before['moving_panel_box_uv'],[.1,0,.1,0])
    assert after['projection_at_control_step']==110 and not after['occlusion_tested']
