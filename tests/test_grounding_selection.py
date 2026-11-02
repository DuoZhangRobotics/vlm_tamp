"""Grounding must separate a selected vertical handle from nearby support surfaces."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.handle_contact import ground_panel, projected_selection
from vlm_tamp.reliable_handle import ground_handle
from vlm_tamp.rekep_libero.mechanism_geometry import project
from vlm_tamp.robot_contracts import validate_operation


def observed_scene():
    # Calibrated point rendering with a vertical door, round bar and a dense
    # horizontal table. All geometry passed to the fitter is measured pixels.
    origin=np.array([.55,-1.,1.5]);forward=np.array([.1,0.,1.05])-origin
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    transform=np.eye(4);transform[:3,:3]=np.column_stack([right,down,forward]);transform[:3,3]=origin
    intrinsic=np.array([[620.,0.,256.],[0.,620.,256.],[0.,0.,1.]])
    camera={'intrinsic':intrinsic,'camera_to_world':transform,'rgb':np.zeros((512,512,3),np.uint8)}
    x,z=np.meshgrid(np.linspace(-.12,.32,350),np.linspace(.95,1.23,240))
    panel=np.c_[x.ravel(),np.zeros(x.size),z.ravel()]
    x,y=np.meshgrid(np.linspace(-.3,.5,500),np.linspace(-.6,.3,500))
    table=np.c_[x.ravel(),y.ravel(),np.full(x.size,.9)]
    theta,z=np.meshgrid(np.linspace(0,2*np.pi,160),np.linspace(.99,1.18,200))
    bar=np.c_[.25+.01*np.cos(theta.ravel()),-.05+.01*np.sin(theta.ravel()),z.ravel()]
    points=np.vstack([panel,table,bar]);uv=np.rint(project(points,camera)).astype(int)
    depth=(points-origin)@forward
    good=(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512)&(depth>0)
    uv=uv[good];points=points[good];depth=depth[good]
    flat=uv[:,1]*512+uv[:,0];order=np.lexsort((depth,flat));ordered=flat[order]
    unique=order[np.r_[True,ordered[1:]!=ordered[:-1]]]
    world=np.full((512*512,3),np.nan);metric=np.zeros(512*512)
    world[flat[unique]]=points[unique];metric[flat[unique]]=depth[unique]
    camera.update(world_points=world.reshape(512,512,3),depth=metric.reshape(512,512))
    image=project(bar,camera);lo,hi=image.min(0),image.max(0)
    panel_image=project(panel,camera);p0,p1=panel_image.min(0),panel_image.max(0)
    # Both coarse selections extend below the moving part onto the table.
    regions={'camera':'view','contact_box_xyxy':[lo[0]-3,lo[1]-3,hi[0]+3,hi[1]+50],
        'moving_panel_box_xyxy':[p0[0]-2,p0[1]-2,p1[0]+2,p1[1]+30],
        'hinge_pixels_xy':project([[-.12,0.,1.],[ -.12,0.,1.2]],camera).tolist(),
        'initial_image_direction_xy':[1.,0.],'motion':'close'}
    finite=np.isfinite(camera['world_points']).all(2);yy,xx=np.where(finite);cloud=camera['world_points'][finite]
    def pixel_at(p):
        i=int(np.argmin(np.linalg.norm(cloud-np.asarray(p),axis=1)));return [int(xx[i]),int(yy[i])]
    regions['contact_pixel_xy']=pixel_at([.25,-.06,1.08])
    regions['hinge_pixels_xy']=[pixel_at([-.11,0.,1.01]),pixel_at([-.11,0.,1.20])]
    return {'view':camera},regions


def test_table_in_coarse_contact_box_does_not_corrupt_bar_fit():
    cameras,regions=observed_scene()
    _,cloud,_,anchors,evidence=ground_handle(cameras,regions,{'joint_family':'revolute','target_progress':1.5})
    assert evidence['target_selection']['outside_observed_panel_height_removed']>100
    assert np.min(cloud[:,2])>.94
    assert evidence['cylinder_robust_residual_m']<.002
    np.testing.assert_allclose(anchors[0][:2],[.25,-.05],atol=.002)
    assert evidence['cylinder_radius_m']==pytest.approx(.01,abs=.002)


def test_door_only_fit_does_not_require_bar_or_claim_a_grasp():
    cameras,regions=observed_scene();regions.pop('contact_box_xyxy')
    validate_operation('ground',{'regions':{k:v for k,v in regions.items() if k!='motion'}})
    model,cloud,mask,_,evidence=ground_panel(cameras,regions,{'joint_family':'revolute','target_progress':1.5})
    assert model.spec['generic_part_joint_model']
    assert 'observed_handle_radius_m' not in model.spec
    assert not evidence['handle_model_available']
    assert len(cloud)>100 and mask.sum()>100


def test_door_only_mask_keeps_observed_protrusions_as_moving_geometry():
    from vlm_tamp.coupled_scene import moving_mask
    cameras,regions=observed_scene();camera=cameras['view'];world=camera['world_points']
    part=np.isfinite(world).all(2)&(world[:,:,2]>.94)
    regions.pop('contact_box_xyxy');regions['panel_mask_selected']=True
    model,cloud,mask,_,_=ground_panel(cameras,regions,{'joint_family':'revolute','target_progress':1.5},panel_mask=part)
    assert np.any(cloud[:,1]<-.02)
    bar=cloud[cloud[:,1]<-.02][::10]
    points=np.vstack([bar,[.25,0.,.9]])
    ownership=moving_mask(points,camera,model.spec,cloud,mask,regions)
    assert ownership[:-1].all() and not ownership[-1]


def test_selected_mask_holes_and_camera_visibility_constrain_projection():
    camera={'world_points':np.zeros((8,8,3)),'intrinsic':np.eye(3),'camera_to_world':np.eye(4)}
    mask=np.zeros((8,8),bool);mask[1,1]=True;mask[3,3]=True
    points=np.array([[1.,1.,1.],[2.,2.,1.],[3.,3.,1.],[-1.,-1.,-1.]])
    assert projected_selection(points,camera,[0,0,7,7],mask).tolist()==[True,False,True,False]


def service(tmp_path,monkeypatch):
    import serve_subscription_robot as module
    robot=module.RobotService.__new__(module.RobotService)
    robot.sim=object();robot.cameras,regions=observed_scene()
    robot.state={'control_steps':60,'metadata':{}};robot.epoch=0;robot.calls=1;robot.episode=tmp_path
    robot.ik=SimpleNamespace(chain=object());robot.config={};robot.task_spec={'motion':'close'}
    import vlm_tamp.explicit_grounding as explicit
    monkeypatch.setattr(explicit,'selected_scene',lambda *a,**k:object())
    robot.rebuild=lambda:None;robot.export_geometry=lambda evidence:{'model':copy.deepcopy(robot.spec),'fit_diagnostics':evidence}
    return module,robot,regions


def test_raw_contact_selection_does_not_invoke_a_cylinder_fit_or_fallback(tmp_path,monkeypatch):
    module,robot,regions=service(tmp_path,monkeypatch)
    import vlm_tamp.reliable_handle as legacy
    monkeypatch.setattr(legacy,'ground_handle',lambda *a,**k:pytest.fail('No cylinder fit'))
    result=robot.ground({k:v for k,v in regions.items() if k!='motion'})
    assert result['grounding_status']=='observed_contact_and_part'
    assert result['handle_model_available'] and not result['handle_fit']['shape_fit_performed']
    assert 'observed_handle_radius_m' not in robot.spec
    assert robot.epoch==1 and result['planner_model_updated']


def test_missing_exact_contact_depth_preserves_previous_model(tmp_path,monkeypatch):
    _,robot,regions=service(tmp_path,monkeypatch);robot.spec={'old':True}
    x,y=regions['contact_pixel_xy'];robot.cameras['view']['world_points'][y,x]=np.nan
    with pytest.raises(ValueError,match='selected contact pixel has no measured'):
        robot.ground({k:v for k,v in regions.items() if k!='motion'})
    assert robot.spec=={'old':True} and robot.epoch==0


def test_registered_masks_reach_fitter_and_stale_masks_do_not(tmp_path,monkeypatch):
    module,robot,regions=service(tmp_path,monkeypatch)
    mask=np.isfinite(robot.cameras['view']['world_points']).all(2)
    robot._perception_tools=SimpleNamespace(parts={'part':{'observation_id':1,'camera':'view','mask':mask}},
        _record=lambda oid:{'control_steps':60})
    called=[]
    import vlm_tamp.explicit_grounding as explicit
    original=explicit.ground_pixels
    def fit(cameras,metadata,choices,**kwargs):
        called.append(kwargs['panel_mask'])
        return original(cameras,metadata,choices,**kwargs)
    monkeypatch.setattr(explicit,'ground_pixels',fit)
    result=robot.ground({k:v for k,v in regions.items() if k!='motion'},moving_part_id='part')
    np.testing.assert_array_equal(called[0],mask)
    assert robot.choices['panel_mask_selected']
    assert result['selected_parts']['moving_part_id']=='part'
    robot.state['control_steps']=61
    with pytest.raises(ValueError,match='stale'):robot.ground({k:v for k,v in regions.items() if k!='motion'},moving_part_id='part')
    assert len(called)==1 and robot.epoch==1


def test_compact_feedback_keeps_partial_model_and_selection_evidence():
    from vlm_tamp.compact_robot_interface import compact_response
    raw={'grounding_status':'door_only','handle_fit':{'accepted':False,'reason':'flat patch'},
        'handle_model_available':False,'selected_parts':{'moving_part_id':'part'},
        'model':{'generic_part_joint_model':True},
        'fit_diagnostics':{'target_selection':{'outside_observed_panel_height_removed':400}}}
    result=compact_response({'ok':True,'result':raw})['result']
    assert all(result[key]==value for key,value in raw.items())


def test_contact_part_can_supply_its_box_without_new_pixel_selection(tmp_path,monkeypatch):
    module,robot,regions=service(tmp_path,monkeypatch);regions.pop('contact_box_xyxy');regions.pop('motion')
    world=robot.cameras['view']['world_points']
    mask=np.isfinite(world).all(2)&(world[:,:,1]<-.02)&(world[:,:,2]>.95)
    robot._perception_tools=SimpleNamespace(parts={'bar':{'observation_id':1,'camera':'view','mask':mask}},
        _record=lambda oid:{'control_steps':60})
    import vlm_tamp.explicit_grounding as explicit
    original=explicit.ground_pixels
    def fit(cameras,metadata,choices,**kwargs):
        from vlm_tamp.part_perception import box_of
        assert choices['contact_box_xyxy']==box_of(mask)
        np.testing.assert_array_equal(kwargs['contact_mask'],mask)
        return original(cameras,metadata,choices,**kwargs)
    monkeypatch.setattr(explicit,'ground_pixels',fit)
    args={'regions':regions,'contact_part_id':'bar'}
    validate_operation('ground',args)
    result=robot.ground(**args)
    assert result['handle_model_available'] and result['selected_parts']['contact_part_id']=='bar'
    assert 'contact_box_xyxy' not in regions


def test_selected_door_mask_is_preserved_in_the_planning_scene():
    from vlm_tamp.coupled_scene import moving_mask
    x,z=np.meshgrid(np.linspace(-.012,.012,9),np.linspace(1.012,.988,9))
    world=np.stack([x,np.zeros_like(x),z],axis=-1)
    transform=np.eye(4);transform[:3,:3]=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    transform[:3,3]=[0,-1,1]
    camera={'world_points':world,'camera_to_world':transform,
        'intrinsic':np.array([[1/.003,0,4],[0,1/.003,4],[0,0,1]])}
    mask=np.ones((9,9),bool);mask[4,4]=False
    spec={'front_plane':{'point_world_m':[0,0,1]},'panel_normal_world':[0,-1,0],
        'joint_origin_world_m':[-.04,0,1],'joint_axis_world':[0,0,1],
        'generic_part_joint_model':True,'source_anchor_world_m':[0,0,1]}
    choices={'moving_panel_box_xyxy':[0,0,8,8],'panel_mask_selected':True}
    points=np.array([[0,0,1],[.003,0,1],[0,-.2,1]])
    assert moving_mask(points,camera,spec,world[mask],mask,choices).tolist()==[False,True,False]
