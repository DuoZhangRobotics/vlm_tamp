"""Prepare sensor-only clouds for an offline RoArtNet benchmark.

Target boxes are manually selected in RGB, not native object masks. This is an
assisted perception benchmark; it is not an autonomous object detector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def target_prompt(obs, camera):
    # Frozen from RGB inspection before running RoArtNet or loading joint labels.
    if camera == "agentview":
        return [0, 171, 195, 435], [[80, 280], [60, 217]], [1, 0]
    if obs == "obs001":
        return [148, 130, 413, 323], [[235, 180], [245, 260], [294, 150]], [1, 1, 0]
    return [134, 130, 414, 370], [[235, 180], [160, 270], [294, 150], [270, 415]], [1, 1, 0, 0]


def save_local(run, rows):
    import torch
    from vlm_tamp.rekep_libero.target_region_v3 import load_mobile_sam_predictor

    torch.set_num_threads(4)
    started = time.perf_counter()
    predictor = load_mobile_sam_predictor()
    torch.cuda.synchronize()
    startup = time.perf_counter() - started
    inputs, audit = [], []
    for row in rows:
        case = "-".join([row["trial"], row["observation"], row["camera"]])
        folder = ROOT / "results/astra-luna-comparison-20261023/trials" / row["trial"]
        folder = folder / "agent-workspace/robot_data/attempt01" / row["observation"]
        dest = run / "inputs" / case
        dest.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        camera = row["camera"]
        with np.load(folder / "sensors.npz", allow_pickle=False) as arrays:
            rgb = arrays[camera + "_rgb"]
            depth = arrays[camera + "_depth"].squeeze()
            world = arrays[camera + "_world_points"]
            robot_depth = arrays[camera + "_known_robot_depth_prior"].squeeze()
        calibration = json.loads((folder / "calibration.json").read_text())[camera]
        T = np.asarray(calibration["camera_to_world"])
        K = np.asarray(calibration["intrinsic"])
        optical = (world - T[:3, 3]) @ T[:3, :3]
        good = np.isfinite(optical).all(-1) & np.isfinite(depth) & (depth > 0)
        robot = np.isfinite(robot_depth) & (robot_depth > 0) & (np.abs(depth - robot_depth) < .0003)
        yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
        projected = optical @ K.T
        projected = projected[..., :2] / projected[..., 2:3]
        pixel_error = np.linalg.norm(projected - np.stack([xx, yy], -1), axis=-1)
        reprojection_max = float(np.max(pixel_error[good]))
        depth_max = float(np.max(np.abs(optical[..., 2][good] - depth[good])))
        if reprojection_max > .01 or depth_max > 1e-5:
            raise ValueError(f"Calibration mismatch for {case}: {reprojection_max}, {depth_max}")
        io_seconds = time.perf_counter() - started
        box, points, labels = target_prompt(row["observation"], camera)
        started = time.perf_counter()
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(box=np.asarray(box), point_coords=np.asarray(points),
                                             point_labels=np.asarray(labels), multimask_output=False)
        torch.cuda.synchronize()
        segment_seconds = time.perf_counter() - started
        mask = masks[0].astype(bool)
        variants = {"sam": mask}
        if row == {"trial": "state020-r1-luna", "observation": "obs001", "camera": "frontview"}:
            box_mask = np.zeros(mask.shape, bool)
            x0, y0, x1, y1 = box
            box_mask[y0:y1, x0:x1] = True
            variants.update(erode2=binary_erosion(mask, iterations=2),
                            dilate2=binary_dilation(mask, iterations=2), bbox=box_mask)
        for variant, selected in variants.items():
            selected = selected & good & ~robot
            pc = np.ascontiguousarray(optical[selected], dtype=np.float32)
            if len(pc) < 100:
                raise ValueError(f"Insufficient selected sensor points: {case}/{variant}")
            file = dest / f"{variant}.npz"
            np.savez_compressed(file, point_cloud=pc, rgb=rgb[selected], camera_to_world=T)
            preview = rgb.copy()
            preview[selected] = (.55 * preview[selected] + .45 * np.array([40, 220, 100])).astype(np.uint8)
            im = Image.fromarray(preview)
            draw = ImageDraw.Draw(im)
            draw.rectangle(box, outline="orange", width=2)
            im.save(dest / f"{variant}-mask.png")
            item = {"id": case + "-" + variant, "kind": "local", "trial": row["trial"],
                    "observation": row["observation"], "camera": camera, "variant": variant,
                    "input": str(file.relative_to(ROOT)), "input_sha256": digest(file),
                    "points": len(pc), "frame": "optical_camera_metres", "camera_to_world": T.tolist(),
                    "intrinsic": K.tolist(), "source_rgb": str((folder / (camera + '.png')).relative_to(ROOT)),
                    "source_sensors": str((folder / "sensors.npz").relative_to(ROOT)),
                    "mask_preview": str((dest / f"{variant}-mask.png").relative_to(ROOT)),
                    "timing": {"sensor_load_and_frame_check_seconds": io_seconds,
                               "segmentation_seconds": segment_seconds},
                    "manual_target_selection": {"box_xyxy": box, "points_xy": points, "labels": labels},
                    "predicted_mask_score_not_calibrated": float(scores[0]),
                    "robot_pixels_removed": int((variants[variant] & robot).sum()),
                    "calibration_max_pixel_error": reprojection_max, "calibration_max_depth_error_m": depth_max}
            inputs.append(item)
        audit.append({"case": case, "source_sha256": digest(folder / "sensors.npz"),
                      "calibration_sha256": digest(folder / "calibration.json"),
                      "object_label_input": False, "joint_label_input": False})
        write_json(run / "inputs-local.json", inputs)
        print(json.dumps({"case": case, "points": len(pc), "segmentation_seconds": segment_seconds}), flush=True)
    write_json(run / "input-audit.json", {"segmentation_load_seconds": startup,
        "segmentation_checkpoint_sha256": digest(ROOT / "third_party/MobileSAM/weights/mobile_sam.pt"),
        "object_selection": "manual RGB-only boxes/points then MobileSAM, same rule before any scoring",
        "cases": audit})
    return inputs


def save_author(run, protocol):
    archive = ROOT / ".runtime/datasets/RealArt-6/without_table/microwave.zip"
    examples = ROOT / ".runtime/datasets/RealArt-6/examples"
    items = []
    with zipfile.ZipFile(archive) as zipped:
        for state in protocol["states"]:
            for view in protocol["views"]:
                name = f"{state}_without_chaos/xyzrgb_{view:02}.npz"
                # Only six explicitly named, relative dataset members are extracted.
                original = examples / name
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(zipped.read(name))
                with np.load(original, allow_pickle=False) as data:
                    pc = np.asarray(data["point_cloud"], dtype=np.float32)
                    rgb = data["rgb"]
                    # Joint annotations deliberately not read by preparation/inference.
                if not np.isfinite(pc).all() or pc.ndim != 2 or pc.shape[1] != 3:
                    raise ValueError(f"Invalid author point cloud: {name}")
                case = f"author-state{state}-view{view:02}"
                dest = run / "inputs" / case
                dest.mkdir(parents=True, exist_ok=True)
                file = dest / "cloud.npz"
                np.savez_compressed(file, point_cloud=pc, rgb=rgb)
                items.append({"id": case, "kind": "author", "variant": "author-segmented",
                              "input": str(file.relative_to(ROOT)), "input_sha256": digest(file),
                              "points": len(pc), "frame": "optical_camera_metres",
                              "author_source_for_later_scoring": str(original.relative_to(ROOT)),
                              "author_source_sha256": digest(original)})
    write_json(run / "inputs-author.json", items)
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/roartnet-perception-20261025")
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    authors = save_author(run, protocol["author_examples"])
    local = save_local(run, protocol["local_examples"])
    write_json(run / "inputs.json", authors + local)
