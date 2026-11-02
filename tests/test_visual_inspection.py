import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from vlm_tamp.visual_inspection import dispatch, install, motion_comparison


def fixture(root, obs=1, step=60, attempt=1):
    folder = root / f"robot_data/attempt{attempt:02d}/obs{obs:03d}"
    folder.mkdir(parents=True)
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    Image.fromarray(rgb).save(folder / "frontview.png")
    (folder / "robot.json").write_text(json.dumps({"control_steps": step, "joint_positions": [1, 2, 3]}))
    (folder / "calibration.json").write_text(json.dumps({"frontview": {"intrinsic": np.eye(3).tolist()}}))
    (folder / "sensors.npz").write_bytes(b"opaque sensor sentinel; must not be opened by image retrieval")
    return rgb


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "robot_data").rglob("*") if p.is_file()}


def test_history_is_bounded_ordered_and_has_no_sensor_numbers(tmp_path):
    for i in [1, 4, 9]:
        fixture(tmp_path, i, 60 + i)
    raw = hashes(tmp_path)
    result = dispatch(tmp_path, "history", {"limit": 2})["result"]
    assert [r["observation_id"] for r in result["observations"]] == [4, 9]
    assert result["observations"][0]["control_steps"] == 64
    assert "joint_positions" not in json.dumps(result)
    assert result["earlier_than"] == 4
    assert [r["observation_id"] for r in dispatch(tmp_path, "history", {"before": 4})["result"]["observations"]] == [1]
    assert hashes(tmp_path) == raw


def test_motion_comparison_uses_only_fixed_existing_camera_images(tmp_path):
    for obs,step in [(1,60),(2,90)]:
        fixture(tmp_path,obs,step)
        folder=tmp_path/f'robot_data/attempt01/obs{obs:03d}'
        fixed={'camera_to_world':np.eye(4).tolist(),'intrinsic':np.eye(3).tolist()}
        wrist=json.loads(json.dumps(fixed))
        wrist['camera_to_world'][0][3]=obs*.1
        (folder/'calibration.json').write_text(json.dumps({'frontview':fixed,'wrist':wrist}))
        (folder/'wrist.png').write_bytes((folder/'frontview.png').read_bytes())
    original=hashes(tmp_path)
    result=motion_comparison(tmp_path,1,1,2)
    assert result['available'] and result['cameras']==['frontview']
    meta=json.loads((tmp_path/result['metadata_file']).read_text())
    assert [(v['observation_id'],v['control_steps']) for v in meta['views']]==[(1,60),(2,90)]
    assert not meta['semantic_interpretation_supplied'] and not result['new_sensor_capture']
    assert hashes(tmp_path)==original
    assert not (tmp_path/'requests').exists()


def test_motion_comparison_does_not_treat_missing_calibration_as_a_fixed_view(tmp_path):
    fixture(tmp_path,1);fixture(tmp_path,2)
    result=motion_comparison(tmp_path,1,1,2)
    assert not result['available'] and result['reason']=='no_common_fixed_camera'


def test_crops_are_exact_and_canvas_coordinates_recover_original_pixels(tmp_path):
    source = fixture(tmp_path)
    fixture(tmp_path, 4, 95)
    raw = hashes(tmp_path)
    response = dispatch(tmp_path, "inspect", {"views": [
        {"observation_id": 4, "camera": "frontview", "region_xyxy": [2, 1, 6, 3], "label": "after"},
        {"observation_id": 1, "camera": "frontview", "label": "before"}]})
    assert response["ok"]
    result = response["result"]
    meta = json.loads((tmp_path / result["metadata_file"]).read_text())
    first, second = meta["views"]
    assert [first["observation_id"], second["observation_id"]] == [4, 1]
    assert first["control_steps"] == 95 and second["control_steps"] == 60
    np.testing.assert_array_equal(np.array(Image.open(tmp_path / first["crop_image"])), source[1:3, 2:6])
    np.testing.assert_array_equal(np.array(Image.open(tmp_path / second["crop_image"])), source)
    canvas = np.array(Image.open(tmp_path / result["comparison_image"]))
    x0, y0, x1, y1 = first["canvas_bounds_xyxy"]
    transform = np.asarray(first["canvas_pixel_to_original_affine"])
    for x, y in [(x0, y0), (x1 - 1, y1 - 1), (x0 + 170, y0 + 100)]:
        xx, yy = np.floor(transform @ [x, y, 1] + .5).astype(int)
        np.testing.assert_array_equal(canvas[y, x], source[yy, xx])
    assert hashes(tmp_path) == raw
    assert not result["new_sensor_capture"] and not result["planner_state_changed"]
    assert not (tmp_path / "requests").exists()


@pytest.mark.parametrize("region", [[-.1, 0, 4, 4], [-1, 0, 4, 4], [0, 0, 99, 4],
                                   [2, 2, 2, 4], [True, 0, 4, 4], "all"])
def test_bad_coordinates_rejected_before_artifacts(tmp_path, region):
    fixture(tmp_path)
    response = dispatch(tmp_path, "inspect", {"views": [{"observation_id": 1, "camera": "frontview", "region_xyxy": region}]})
    assert not response["ok"]
    assert not (tmp_path / "visual_inspections").exists()


def test_missing_second_observation_does_not_create_partial_comparison(tmp_path):
    fixture(tmp_path)
    response = dispatch(tmp_path, "inspect", {"views": [{"observation_id": 1, "camera": "frontview"},
                                                       {"observation_id": 2, "camera": "frontview"}]})
    assert not response["ok"]
    assert not (tmp_path / "visual_inspections").exists()


def test_path_traversal_and_symlink_source_rejected(tmp_path):
    fixture(tmp_path)
    response = dispatch(tmp_path, "inspect", {"views": [{"observation_id": 1, "camera": "../../private"}]})
    assert not response["ok"]
    file = tmp_path / "robot_data/attempt01/obs001/frontview.png"
    original = tmp_path / "elsewhere.png"
    file.replace(original)
    file.symlink_to(original)
    response = dispatch(tmp_path, "inspect", {"views": [{"observation_id": 1, "camera": "frontview"}]})
    assert not response["ok"] and "symlink" in response["error"]


def test_current_attempt_default_and_explicit_old_attempt(tmp_path):
    fixture(tmp_path, 1, 60, 1)
    fixture(tmp_path, 7, 70, 2)
    assert dispatch(tmp_path, "history", {})["result"]["attempt"] == 2
    assert dispatch(tmp_path, "history", {"attempt": 1})["result"]["observations"][0]["observation_id"] == 1


def test_installer_only_adds_local_helper_and_documentation(tmp_path):
    (tmp_path / "tools.txt").write_text("Original tool index.\n")
    (tmp_path / "robot.py").write_text("robot client sentinel")
    result = install(tmp_path)
    assert result["enabled"] and not result["new_sensor_capture"]
    assert (tmp_path / "robot.py").read_text() == "robot client sentinel"
    assert (tmp_path / "tools.txt").read_text().startswith("Original tool index.\n")
    assert "image viewer" in (tmp_path / "tools/visual_inspection.txt").read_text()
