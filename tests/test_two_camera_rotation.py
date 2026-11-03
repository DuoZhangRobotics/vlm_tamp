"""Two-camera observation boundaries and explicit hinge rotation semantics."""
from __future__ import annotations

import numpy as np
import pytest

from vlm_tamp.contact_orientation import selection_geometry
from vlm_tamp.joint_rotation import signed_rotation
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
