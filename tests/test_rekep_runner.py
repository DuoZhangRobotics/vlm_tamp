"""Integration boundaries in the ReKep runner, using no remote inference."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from vlm_tamp.rekep_libero.program import parse_program
from vlm_tamp.rekep_libero.report import build_report


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_rekep_libero.py"
    spec = importlib.util.spec_from_file_location("rekep_rollout_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_constraint_references_keep_original_image_ids(runner):
    source = '''```python
num_stages = 2
def stage1_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(end_effector - keypoints[4])
def stage2_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(keypoints[4] - np.mean(keypoints[[1, 3]], axis=0))
def stage2_path_constraint1(end_effector, keypoints):
    return get_grasping_cost_by_keypoint_idx(4)
grasp_keypoints = [4, -1]
release_keypoints = [-1, 4]
```'''
    program = parse_program(source, 5, "pick_place")
    assert runner.reference_indices(program).tolist() == [1, 3, 4]


def test_tracking_updates_sparse_references_without_relabeling(runner, tmp_path):
    points = np.arange(15, dtype=float).reshape(5, 3)
    pixels = np.arange(10).reshape(5, 2)
    features = np.arange(20).reshape(5, 4)
    groups = np.array([3, 1, 9, 1, 3])
    original = {"points_world": points.copy(), "pixels_yx": pixels.copy(), "groups": groups.copy(),
                "features": features.copy(), "num_keypoints": 5, "cameras": {"old": {}}, "meta": {}}

    class Tracker:
        def track(self, subset, obs, meta):
            assert subset["points_world"].shape == (2, 3)
            assert np.array_equal(subset["features"], features[[1, 4]])
            return {"points_world": subset["points_world"] + 0.25,
                    "pixels_yx": subset["pixels_yx"] + 1, "cameras": {"new": {}},
                    "meta": meta, "tracking": [{"similarity": .99}, {"similarity": .98}]}

    result = runner.track_references(Tracker(), original, {"metadata": {"frame": "world"}},
                                     tmp_path, np.array([1, 4]))
    assert np.array_equal(result["points_world"][[0, 2, 3]], points[[0, 2, 3]])
    assert np.allclose(result["points_world"][[1, 4]], points[[1, 4]] + .25)
    assert np.array_equal(result["groups"], groups)
    assert np.array_equal(result["features"], features)
    assert np.array_equal(original["points_world"], points)
    assert result["num_keypoints"] == 5


def test_actual_rollout_artifacts_have_browser_links_and_thumbnails(tmp_path):
    case = tmp_path / "experiment/libero_goal-08-state001-seed0"
    for relative in ("perception", "program", "initial", "stage1-grasps"):
        (case / relative).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "blue").save(case / "perception/numbered.png")
    Image.new("RGB", (16, 16), "green").save(case / "stage1-grasps/grasp_overlay.png")
    (case / "initial/rollout.mp4").write_bytes(b"video asset")
    (case / "initial/actions.json").write_text("[]")
    (case / "program/output_raw.txt").write_text("generated functions")
    (case / "program/response.json").write_text("{}")
    (case / "stage1-approach-plan.json").write_text("{}")
    np.savez(case / "stage1-grasps/grasp_candidates.npz", scores=np.array([.8]))
    (case / "episode.json").write_text(json.dumps({"success": False, "task_type": "pick_place",
                                                  "status": "failed", "failure_stage": "execution",
                                                  "physical_grasp_attempts": 1,
                                                  "video": "initial/rollout.mp4"}))
    output = tmp_path / "report"
    summary = build_report([case], output, expected_episodes=10)
    document = (output / "index.html").read_text()
    assert summary["evaluation_physical_grasp_attempts"] == 1
    assert "0/10" in document
    for asset in ("perception/numbered.png", "stage1-grasps/grasp_overlay.png", "program/output_raw.txt",
                  "program/response.json", "initial/rollout.mp4", "initial/actions.json",
                  "stage1-approach-plan.json", "stage1-grasps/grasp_candidates.npz"):
        assert asset in document
    assert document.count('<img loading="lazy"') == 2


def test_retention_rejects_stationary_landmark_when_tcp_moves_seven_mm(runner):
    before = np.array([0, 0, 1, 0, 0, 0, 1], float)
    after = before.copy()
    after[0] += .007
    point = np.array([.02, 0, 1], float)
    result = runner.retained_grasp(before, after, point, point, [.01, .01])
    assert result["source_motion_m"] == 0
    assert not result["verified"]


def test_retention_rejects_fully_shut_empty_gripper(runner):
    before = np.array([0, 0, 1, 0, 0, 0, 1], float)
    after = before.copy()
    after[2] += .01
    point = np.array([.02, 0, 1], float)
    result = runner.retained_grasp(before, after, point, point + [0, 0, .01], [0, 0])
    assert result["rigid_prediction_error_m"] < 1e-10
    assert not result["verified"]


def test_retention_accepts_rigid_observed_motion_with_nonempty_finger_gap(runner):
    before = np.array([0, 0, 1, 0, 0, 0, 1], float)
    after = before.copy()
    after[2] += .01
    point = np.array([.02, 0, 1], float)
    result = runner.retained_grasp(before, after, point, point + [0, 0, .0105], [.012, .012])
    assert result["verified"]
    assert result["source_motion_m"] > .01


def test_continuous_retention_detects_slip_without_new_probe(runner):
    pose = np.array([0, 0, 1, 0, 0, 0, 1], float)
    point = np.array([.02, 0, 1], float)
    result = runner.retained_grasp(pose, pose, point, point + [.005, 0, 0], [.012, .012],
                                   require_motion=False)
    assert not result["verified"]
