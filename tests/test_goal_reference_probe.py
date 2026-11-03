"""Public RGB-D probes must preserve numeric depth and missing-value semantics."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from check_luna_goal_reference import POINT_TOOL


def test_robot_depth_prior_is_a_measurement_not_a_boolean(tmp_path):
    """NaN means no predicted robot surface; a finite prior stays in metres."""
    folder = tmp_path / 'inputs/before'
    folder.mkdir(parents=True)
    np.savez(folder / 'sensors.npz',
             agentview_world_points=np.array([[[1., 2., 3.], [4., 5., 6.]]]),
             agentview_depth=np.array([[3., 6.]]),
             agentview_known_robot_depth_prior=np.array([[np.nan, 1.25]]))
    tool = tmp_path / 'sample_points.py'
    tool.write_text(POINT_TOOL)
    process = subprocess.run(
        [sys.executable, str(tool), 'before', 'agentview', '[[0,0],[1,0]]'],
        cwd=tmp_path, check=True, text=True, capture_output=True)
    samples = json.loads(process.stdout)['samples']
    assert samples[0]['robot_depth_prior_m'] is None
    assert type(samples[1]['robot_depth_prior_m']) is float
    assert samples[1]['robot_depth_prior_m'] == 1.25
    assert samples[0]['point_world_m'] == [1., 2., 3.]
