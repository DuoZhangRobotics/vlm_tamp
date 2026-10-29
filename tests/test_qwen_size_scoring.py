"""Scientific scoring must not reward a different candidate than the actor chose."""
import numpy as np
import pytest

from compare_qwen_sizes import raw_hinge_pixels
from summarize_qwen_sizes import selected_error


def test_selected_error_uses_actor_choice_not_best_observer_fit():
    case={'actor_selection':{'chosen_ground_call':2},'observer_geometry':[
        {'id':'scene-ground1','native_hinge_line_error_mm':1.},
        {'id':'scene-ground2','native_hinge_line_error_mm':80.}]}
    assert selected_error(case)==80.
    case['actor_selection']['chosen_ground_call']=0
    assert selected_error(case) is None


@pytest.mark.parametrize('points,expected_angle',[
    ([[60,40],[60,60]],0.),
    ([[40,50],[60,50]],90.),
])
def test_projected_hinge_scoring_has_known_pixel_distance_and_angle(points,expected_angle):
    # Identity camera, focal length100, principal point(50,50). The native
    # line through(0,0,1) along camera-Y projects to x=50.
    camera={'intrinsic':[[100,0,50],[0,100,50],[0,0,1]],'camera_to_world':np.eye(4)}
    loader=lambda _:({'frontview':camera},{},{})
    rows=[{'id':'sample','contract_valid':True,'source_private':{'case':'scene'},
           'arguments':{'regions':{'camera':'frontview','hinge_pixels_xy':points}}}]
    native={'scene':{'native_hinge_anchor_world_m':[0,0,1],'native_axis_world':[0,1,0]}}
    result=raw_hinge_pixels(rows,native,loader)[0]
    assert result['mean_distance_to_projected_native_hinge_line_px']==pytest.approx(10.)
    assert result['image_line_angle_error_deg']==pytest.approx(expected_angle)
