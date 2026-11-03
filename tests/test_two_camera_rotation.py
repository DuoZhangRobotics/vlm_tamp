"""Two-camera observation boundaries and explicit hinge rotation semantics."""
from __future__ import annotations

import numpy as np
import pytest

from vlm_tamp.contact_orientation import selection_geometry
from vlm_tamp.joint_rotation import canonical_axis, signed_rotation
from vlm_tamp.robot_contracts import RequestValidationError, validate_operation


def test_named_rotation_matches_right_hand_rule_and_changes_with_axis_sign() -> None:
    """The same physical rotation has the opposite name from the opposite axis end."""
    axis = np.array([0., 0., 1.])
    radial = np.array([1., 0., 0.])
    ccw = np.cross(axis, radial) * signed_rotation(10., 'ccw')
    cw_from_other_end = np.cross(-axis, radial) * signed_rotation(10., 'cw')
    np.testing.assert_array_equal(ccw, cw_from_other_end)
    assert ccw[1] > 0
    assert signed_rotation(-10.) == -10.
    request = {'observation_id':1, 'camera':'wrist', 'contact_pixel_xy':[20,30],
               'contact_name':'pad', 'region_xyxy':[10,20,40,50],
               'joint_degrees':10., 'joint_model_epoch':1, 'rotation_direction':'cw'}
    validate_operation('push', request)
    request['joint_degrees'] = -10.
    with pytest.raises(RequestValidationError, match='positive'):
        validate_operation('push', request)


@pytest.mark.parametrize('axis,expected',[
    ([0,0,-1],[0,0,1]), ([-1,0,0],[1,0,0]), ([0,-1,0],[0,1,0]),
    ([.05,-.1,-1],[-.05,.1,1]), ([-1,.2,0],[1,-.2,0]),
    ([-1,1,0],[1,-1,0]),
])
def test_canonical_axis_preserves_line_and_handles_horizontal_hinges(axis,expected) -> None:
    original=np.asarray(axis,dtype=float);wanted=np.asarray(expected,dtype=float)
    first=canonical_axis(original);second=canonical_axis(-original)
    np.testing.assert_allclose(first,second)
    np.testing.assert_allclose(first,wanted/np.linalg.norm(wanted))
    assert np.linalg.norm(np.cross(first,original)) < 1e-12


def test_hinge_pixel_order_does_not_change_origin_or_cw_ccw_tangent() -> None:
    from test_explicit_contact import capture
    from vlm_tamp.explicit_grounding import ground_pixels
    view,choices=capture()
    first=ground_pixels({'test':view},{},choices)['spec']
    choices['hinge_pixels_xy'].reverse()
    second=ground_pixels({'test':view},{},choices)['spec']
    np.testing.assert_allclose(first['joint_axis_world'],second['joint_axis_world'])
    np.testing.assert_allclose(first['joint_origin_world_m'],second['joint_origin_world_m'])
    p=np.asarray(first['source_anchor_world_m'])
    for direction in ['cw','ccw']:
        a=np.cross(first['joint_axis_world'],p-first['joint_origin_world_m'])*signed_rotation(5.,direction)
        b=np.cross(second['joint_axis_world'],p-second['joint_origin_world_m'])*signed_rotation(5.,direction)
        np.testing.assert_allclose(a,b)


def test_opposite_observed_face_changes_normal_push_compatibility_without_veto() -> None:
    camera = np.eye(4)
    camera[1,3] = -1.
    views = {'wrist': {'camera_to_world':camera}}
    first = selection_geometry([1,0,0],[0,1,0],[1,1,0],[0,1,0],views)
    second = selection_geometry([1,0,0],[0,-1,0],[1,-1,0],[0,1,0],views)
    assert first['contact_effect']['motion_relative_to_face'] == 'away_from_selected_face'
    assert first['contact_effect']['opposite_side_camera_candidates'] == ['wrist']
    assert second['contact_effect']['motion_relative_to_face'] == 'into_selected_face'
    assert not first['contact_effect']['execution_veto_applied']


def test_worker_default_capture_cannot_render_auxiliary_views(monkeypatch) -> None:
    """The sensor boundary is enforced before arrays can reach planning or the actor."""
    from vlm_tamp.profiled_sim_worker import ProfiledWorker
    from vlm_tamp.coupled_worker import CoupledWorker
    worker=ProfiledWorker.__new__(ProfiledWorker)
    worker.stats={}
    monkeypatch.setattr(CoupledWorker,'_sensor_arrays',lambda self: list(self.selected_capture_cameras))
    assert worker._sensor_arrays() == ['agentview','wrist']
    monkeypatch.setattr(ProfiledWorker,'_observe_selected',lambda self, _:list(self.selected_capture_cameras))
    assert worker.observe('unused') == ['agentview','wrist']
    assert worker.observe('unused',['wrist']) == ['wrist']
    with pytest.raises(ValueError,match='only agentview and wrist'):
        worker.observe('unused',['frontview'])


def test_compact_observation_exposes_both_current_views() -> None:
    from vlm_tamp.compact_robot_interface import observation
    value=observation({'images':{'agentview':'a.png','wrist':'w.png'},
                       'observation_id':3,'control_steps':100})
    assert value['images'] == {'agentview':'a.png','wrist':'w.png'}
    assert value['observation_id'] == 3
