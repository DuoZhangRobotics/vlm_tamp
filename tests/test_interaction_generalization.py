import copy
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vlm_tamp.interaction_contract import parse_interaction,backend_for,InteractionError
from vlm_tamp.interaction_geometry import fit_plane,fit_bar,fit_control,align_planes


def selection():
    return {'version':'interaction-v1','camera':'agentview',
      'contact':{'shape':'straight_bar','box_xyxy':[10,10,50,120],'pixel_xy':[30,60],'axis_pixels_xy':[[30,20],[30,110]]},
      'moving':{'shape':'planar_panel','box_xyxy':[5,5,200,200]},
      'mechanism':{'family':'revolute','axis_source':'hinge_edge','axis_pixels_xy':[[10,5],[10,190]],'support_box_xyxy':[20,170,150,190],
          'motion':'close','initial_image_direction_xy':[1,0],'turn_direction_view':None},
      'goal':{'relation':'aligned_with_reference_plane','reference_box_xyxy':[210,10,250,180]},
      'contact_roles':['grasp_selected_part','fingers_on_local_moving_surface'],'confidence':'medium','explanation':'visible bar and frame'}


@pytest.mark.parametrize('mutation',[
 lambda v:v.update(task_label='microwave'),
 lambda v:v['contact'].update(pixel_xy=[float('nan'),60]),
 lambda v:v['contact'].update(pixel_xy=[500,500]),
 lambda v:v['mechanism'].update(axis_pixels_xy=None),
 lambda v:v['goal'].update(reference_box_xyxy=None),
 lambda v:v.update(contact_roles=['grasp_selected_part','ignore_all_arm_collisions']),
 lambda v:v['mechanism'].update(family='free'),
 lambda v:v['goal'].update(relation='hand_clearance'),
])
def test_contract_rejects_invalid_and_unscoped_choices(mutation):
    value=selection();mutation(value)
    with pytest.raises(InteractionError):parse_interaction(json.dumps(value))


def test_dispatch_uses_geometry_and_records_unknown():
    value=parse_interaction(json.dumps(selection()));assert backend_for(value)=='hinge'
    value['contact']['shape']='unknown'
    with pytest.raises(InteractionError,match='unresolved'):backend_for(value)


@pytest.mark.parametrize('angles',[[0,0,0],[35,42,17],[90,0,0],[-30,70,125]])
def test_bar_fitting_is_rotation_equivariant_and_not_vertical_only(angles):
    theta,z=np.meshgrid(np.linspace(-1.8,1.8,36),np.linspace(-.12,.12,40))
    points=np.c_[.023*np.cos(theta.ravel()),.023*np.sin(theta.ravel()),z.ravel()]
    r=Rotation.from_euler('xyz',angles,degrees=True).as_matrix();t=np.array([.12,-.4,.9])
    points=points@r.T+t;ends=np.array([[0,0,-.1],[0,0,.1]])@r.T+t
    fit=fit_bar(points,ends)
    assert abs(fit['axis_world']@r[:,2])>.999
    assert abs(fit['radius_m']-.023)<1e-4
    assert np.linalg.norm(np.cross(fit['origin_world_m']-t,r[:,2]))<1e-4
    assert fit['world_up_prior'] is False


def test_plane_alignment_uses_selected_reference_in_any_world_direction():
    r=Rotation.from_euler('xyz',[20,30,70],degrees=True).as_matrix();n=r[:,2]
    a={'point':np.zeros(3),'normal':n,'uncertainty_m':.001}
    b={'point':.174*n+.5*r[:,0],'normal':n,'uncertainty_m':.002}
    progress,_,uncertainty=align_planes(a,b,n,'prismatic')
    assert abs(progress-.174)<1e-10
    assert abs(uncertainty-np.sqrt(5)*.001)<1e-10
    with pytest.raises(InteractionError):align_planes(a,b,-n,'prismatic')


def test_rotated_plane_fit_and_control_use_selected_support_not_world_height():
    x,y=np.meshgrid(np.linspace(-.2,.2,24),np.linspace(-.15,.15,24))
    plane=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    theta=np.linspace(0,2*np.pi,200);control=np.c_[.03*np.cos(theta),.02*np.sin(theta),np.full(200,.01)]
    r=Rotation.from_euler('xyz',[60,40,20],degrees=True).as_matrix();t=np.array([.3,-.2,.8])
    model=fit_plane(plane@r.T+t,camera_position=t+r[:,2])
    fitted,cloud=fit_control(control@r.T+t,model)
    assert abs(model['normal']@r[:,2])>.9999
    assert np.linalg.norm(fitted['origin_world_m']-t)<.001
    assert len(cloud)==200


def test_bar_does_not_accept_flat_background_patch():
    x,y=np.meshgrid(np.linspace(-.12,.12,30),np.linspace(-.12,.12,30))
    with pytest.raises(InteractionError):fit_bar(np.c_[x.ravel(),y.ravel(),np.zeros(x.size)])


def test_backend_selects_general_planner_for_rotated_geometry():
    from types import SimpleNamespace
    from vlm_tamp.interaction_execution import execution_backend
    value=selection()
    fit=SimpleNamespace(selection=value,spec={'joint_axis_world':[0,0,1]},
        evidence={'contact_model':{'axis_world':[0,0,1]}})
    assert execution_backend(fit)=='upright_bar_hinge'
    fit.spec['joint_axis_world']=[0,1,0]
    assert execution_backend(fit)=='mechanism'
    fit.spec['joint_axis_world']=[0,0,1];value['contact_roles']=['grasp_selected_part']
    assert execution_backend(fit)=='mechanism'


def test_constant_features_cannot_create_nan_or_unbounded_clustering():
    import torch
    from vlm_tamp.bounded_keypoints import normalized,bounded_kmeans
    values=normalized(torch.ones((50,6),dtype=torch.float64))
    assert torch.isfinite(values).all() and torch.count_nonzero(values)==0
    labels,centres,record=bounded_kmeans(values,5,maximum_iterations=7)
    assert record['iterations']<=7 and record['finite_centres']
    assert len(labels)==50 and len(torch.unique(labels))==1


def test_clustering_preserves_separated_spatial_groups():
    import torch
    from vlm_tamp.bounded_keypoints import bounded_kmeans
    data=torch.cat([torch.full((10,2),float(i)*10) for i in range(5)])
    labels,centres,record=bounded_kmeans(data,5)
    assert len(torch.unique(labels))==5 and record['converged']
    assert all(len(torch.unique(labels[i:i+10]))==1 for i in range(0,50,10))
