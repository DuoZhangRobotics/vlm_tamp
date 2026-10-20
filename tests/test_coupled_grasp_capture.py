from itertools import product
import numpy as np
import pytest
from vlm_tamp.coupled_grasp_capture import observed_pad_capture


def metadata(separation=.026):
    return {'eef_pose_world_xyzw':[0,0,0,0,0,0,1],
       'robot_collision_volumes_world':[{'type':'convex','robot_geom_name':f'finger{i}_pad_collision',
        'vertices_world':np.array(list(product([x-.003,x+.003],[-.008,.008],[-.008,.008])))}
        for i,x in enumerate([-separation/2,separation/2])]}


def surface(centre=.014):
    angles=np.linspace(0,2*np.pi,100)
    return np.c_[centre+.01*np.cos(angles),.01*np.sin(angles),np.zeros(100)]


def test_observed_capture_accepts_supported_surface_beyond_old_axis_distance():
    record=observed_pad_capture(metadata(),surface(),.01)
    assert record['observed_points_in_pad_channel']>=10
    assert record['maximum_axis_distance_m']>.014
    assert record['continued_observed_retention_required']
    assert not record['force_closure_certified']


def test_nearby_axis_bound_cannot_substitute_for_surface_inside_jaws():
    with pytest.raises(ValueError,match='outside'):
        observed_pad_capture(metadata(),surface()+[0,.1,0],.01)


def test_wide_open_fingers_cannot_be_called_a_captured_bar():
    with pytest.raises(ValueError,match='too far apart'):
        observed_pad_capture(metadata(.08),surface(0),.01)


def test_tilted_capture_uses_projected_cylinder_width():
    from scipy.spatial.transform import Rotation
    data=metadata(.04);rotation=Rotation.from_euler('y',45,degrees=True)
    data['eef_pose_world_xyzw']=[0,0,0,*rotation.as_quat()]
    for pad in data['robot_collision_volumes_world']:
        pad['vertices_world']=rotation.apply(pad['vertices_world'])
    record=observed_pad_capture(data,surface(0),.01)
    assert record['maximum_supported_separation_m']>.04
    assert record['observed_points_in_pad_channel']>=10
