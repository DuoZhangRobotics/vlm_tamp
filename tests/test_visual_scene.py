"""Observed-only projection marks and non-executing robot pose visualizations."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vlm_tamp.visual_scene import (cloud_samples,projection_spec,project_world,pixel_coordinates,
    render_projection,scene_views,mark_scene_point,pose_from_axes,preview_hand)
from vlm_tamp.rekep_libero.geometry import pose_matrix


@pytest.fixture
def service(tmp_path: Path) -> SimpleNamespace:
    """Two measured camera arrays and one unrelated camera that must be excluded."""
    yy,xx=np.mgrid[:16,:16]
    points=np.stack([xx*.01,yy*.01,np.ones_like(xx)],axis=-1)
    view={'rgb':np.full((16,16,3),128,np.uint8),'depth':np.ones((16,16)),
          'world_points':points,'camera_to_world':np.eye(4),
          'intrinsic':np.array([[40,0,8],[0,40,8],[0,0,1]])}
    public=tmp_path/'robot_data';(public/'attempt01').mkdir(parents=True)
    (public/'attempt01/obs001').mkdir()
    for name in ('agentview','wrist'):Image.fromarray(view['rgb']).save(public/f'attempt01/obs001/{name}.png')
    cube=np.array([[x,y,z] for x in [-.01,.01] for y in [-.02,.02] for z in [.9,.92]])
    return SimpleNamespace(workspace=tmp_path,public=public,attempt=1,calls=1,
        cameras={'agentview':view,'wrist':view,'hidden_orbit':{**view,'world_points':points+1000}},
        latest_observation={'observation_id':1},state={'control_steps':60,'metadata':{
            'eef_pose_world_xyzw':[0,0,1,0,0,0,1],'gripper_qpos':[.04,-.04],
            'robot_collision_volumes_world':[{'robot_geom_name':'gripper0_hand_collision','vertices_world':cube}]}},
        visual_views={})


def test_extra_camera_does_not_enter_rendered_cloud(service: SimpleNamespace) -> None:
    cloud=cloud_samples(service.cameras)
    assert cloud['points'].max()==1
    assert set(cloud['sources'][:,0])=={0,1}
    assert len(cloud['points'])==128


@pytest.mark.parametrize('plane',['xy','yz','xz'])
def test_pixel_world_mapping_uses_the_declared_axes(plane: str) -> None:
    points=np.array([[0.,0.,0.],[1.,1.,1.]])
    spec=projection_spec(points,plane,[.5,.5,.5],1.)
    pixels=project_world(points,spec)
    for point,pixel in zip(points,pixels):
        coords=pixel_coordinates(np.rint(pixel).astype(int).tolist(),spec)
        for axis,value in coords.items():assert value==pytest.approx(point[axis],abs=spec['meters_per_pixel'])


def test_surface_mark_returns_original_sample_and_keeps_old_observation_identity(service: SimpleNamespace) -> None:
    result=scene_views(service)
    view=result['views'][0];folder,_=service.visual_views[view['view_id']]
    with np.load(folder/'xy-index.npz') as data:
        y,x=np.argwhere(data['index']>=0)[0]
        hy,hx=np.argwhere(data['index'][48:592,48:592]<0)[0]+48
    service.calls=2;service.latest_observation={'observation_id':8}
    mark=mark_scene_point(service,view['view_id'],[int(x),int(y)])
    source=service.cameras[mark['source_camera']]['world_points']
    sx,sy=mark['source_pixel_xy']
    np.testing.assert_allclose(mark['point_world_m'],source[sy,sx])
    assert mark['observation_id']==1 and not mark['current_observation']
    with pytest.raises(ValueError,match='No observed sample'):
        mark_scene_point(service,view['view_id'],[int(hx),int(hy)])


def test_free_space_mark_has_no_invented_surface_support(service: SimpleNamespace) -> None:
    result=scene_views(service,center_world_m=[0,0,0],span_m=2.)
    service.calls=2
    mark=mark_scene_point(service,result['views'][0]['view_id'],[320,320],mode='free_space',
        depth_view_id=result['views'][1]['view_id'],depth_pixel_xy=[400,400])
    assert mark['surface_support_claimed'] is False
    assert mark['second_coordinate_axis']=='Z'
    assert abs(mark['point_world_m'][0])<.01 and abs(mark['point_world_m'][1])<.01
    assert mark['point_world_m'][2]<-.2


def test_preview_only_transforms_robot_geometry_and_does_not_mutate_state(service: SimpleNamespace) -> None:
    before=np.array(service.state['metadata']['robot_collision_volumes_world'][0]['vertices_world'])
    result=preview_hand(service,position_world_m=[.1,0,1],approach_world=[0,0,1],jaw_world=[1,0,1])
    pose=pose_matrix(result['pose_world_xyzw'])
    np.testing.assert_allclose(pose[:3,:3],np.eye(3))
    np.testing.assert_allclose(service.state['metadata']['robot_collision_volumes_world'][0]['vertices_world'],before)
    assert len(result['images'])==2
    assert not result['robot_motion_issued'] and not result['ik_solved'] and not result['collision_checked']
    with pytest.raises(ValueError,match='parallel'):pose_from_axes([0,0,0],[0,0,1],[0,0,-1])


def test_projection_depth_buffer_selects_the_front_observed_sample() -> None:
    cloud={'points':np.array([[0,0,1.],[0,0,2.]]),'colors':np.array([[255,0,0],[0,255,0]],np.uint8)}
    spec=projection_spec(cloud['points'],'xy',[0,0,0],1.)
    _,index=render_projection(cloud,spec)
    assert set(index[index>=0])=={1}
