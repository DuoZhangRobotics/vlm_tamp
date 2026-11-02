"""Native rejection pictures are explanatory and do not change planning state."""

from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from vlm_tamp.contact_rejection_view import render_contact_rejection


def test_rejection_picture_uses_reported_sphere_and_existing_observation(tmp_path: Path) -> None:
    """No extra planner validation, robot motion or input-image edit is performed."""
    y, x = np.indices((96, 96))
    points = np.stack(((x-48)/60., (y-48)/60., np.ones_like(x)), axis=-1)
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    view = {"rgb": rgb, "world_points": points, "camera_to_world": np.eye(4),
        "intrinsic": np.array([[60., 0, 48], [0, 60, 48], [0, 0, 1]])}
    scene = SimpleNamespace(static_points=np.array([[0., .3, 1.]]),
        moving_reference=np.empty((0, 3)), contact_reference=np.empty((0, 3)))
    diagnostic = {"attempts": [{"planning": {"endpoints": {"goal": {
        "accepted": False, "last_rejected_surface": 0,
        "blocking_sphere_world_m": [0., .3, .98, .03],
        "blocking_robot_part": "gripper0_hand_collision", "blocking_surface": "observed stationary geometry"}}}}]}
    original = copy.deepcopy(diagnostic)
    rows = render_contact_rejection(diagnostic, scene, {"view": view}, "view", np.array([.05, .3, 1.]), tmp_path)
    assert len(rows) == 1 and Path(rows[0]["image"]).is_file()
    assert rows[0]["sampled_overlap_m"] == pytest.approx(.01)
    assert rows[0]["additional_collision_admission_checks"] == 0
    assert rows[0]["physical_motion_depicted"] is False
    assert diagnostic == original
    assert not rgb.any()


def test_numerical_failure_without_a_native_witness_produces_no_collision_image(tmp_path: Path) -> None:
    """An unknown collision must not be invented to explain an IK failure."""
    assert render_contact_rejection({"attempts": []}, None, {}, "", np.zeros(3), tmp_path) == []
