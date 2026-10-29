"""Exploratory motion verification on saved public RGB-D, without native priors.

This uses the existing upright SE2 estimator and optical flow. It does not use
RoArtNet's hinge location to choose matches or fit the unconstrained pivot.
"""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlm_tamp.rigid_hinge_tracking import estimate_rigid_hinge


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(folder, names):
    started = time.perf_counter()
    views = {}
    with np.load(folder / "sensors.npz", allow_pickle=False) as data:
        for name in names:
            rgb = data[name + "_rgb"]
            depth = data[name + "_depth"].squeeze()
            prior = data[name + "_known_robot_depth_prior"].squeeze()
            views[name] = {"gray": cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                           "world": data[name + "_world_points"],
                           "robot": np.isfinite(prior) & (prior > 0) & (np.abs(depth - prior) < .0003)}
    return views, time.perf_counter() - started


def correspondences(a, b, center, radius):
    first, second, counts = [], [], []
    for name, before in a.items():
        after = b[name]
        xyz = before["world"]
        mask = np.isfinite(xyz).all(2) & ~before["robot"] & (np.linalg.norm(xyz - center, axis=-1) < radius)
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        original = cv2.goodFeaturesToTrack(before["gray"], maxCorners=350, qualityLevel=.006,
                                           minDistance=5, mask=mask * 255, blockSize=5)
        if original is None:
            counts.append({"camera": name, "matched": 0})
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(before["gray"], after["gray"], original, None,
            winSize=(25, 25), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005))
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(after["gray"], before["gray"], tracked, None,
                                                  winSize=(25, 25), maxLevel=3)
        x, y = original.reshape(-1, 2), tracked.reshape(-1, 2)
        valid = status.ravel().astype(bool) & reverse.ravel().astype(bool) & (np.linalg.norm(back.reshape(-1, 2) - x, axis=1) < .8)
        h, w = before["gray"].shape
        valid &= (y[:, 0] >= 2) & (y[:, 0] < w - 2) & (y[:, 1] >= 2) & (y[:, 1] < h - 2)
        n = 0
        for p, q in zip(x[valid], y[valid]):
            px, py = np.rint(p).astype(int)
            qx, qy = np.rint(q).astype(int)
            if after["robot"][qy, qx]:
                continue
            u, v = xyz[py, px], after["world"][qy, qx]
            if np.isfinite(u).all() and np.isfinite(v).all():
                first.append(u)
                second.append(v)
                n += 1
        counts.append({"camera": name, "matched": n})
    return np.asarray(first).reshape(-1, 3), np.asarray(second).reshape(-1, 3), counts


def main():
    run = ROOT / "results/roartnet-perception-20261025"
    output = run / "motion"
    output.mkdir(exist_ok=True)
    base = ROOT / "results/astra-luna-comparison-20261023/trials/state020-r1-luna/agent-workspace/robot_data/attempt01"
    folders = sorted(p for p in base.glob("obs*") if int(p.name[3:]) >= 5)
    # Region includes the observed target and room for its moving surface.
    # It is defined from sensor points, not a simulator body or fitted axis.
    with np.load(run / "inputs/state020-r1-luna-obs001-frontview/sam.npz") as data:
        transform = data["camera_to_world"]
        world = data["point_cloud"] @ transform[:3, :3].T + transform[:3, 3]
    center = np.mean(world, axis=0)
    radius = float(np.max(np.linalg.norm(world - center, axis=1)) + .25)
    names = ["frontview", "agentview", "arm_side", "arm_back", "arm_top"]
    write(output / "protocol.json", {"phase": "exploratory follow-up after static scores; not a held-out validation",
        "pairs": [[a.name, b.name] for a, b in zip(folders[:-1], folders[1:])], "cameras": names,
        "region_center_world_m": center.tolist(), "region_radius_m": radius,
        "prior": None, "matching": "Existing forward/backward Lucas-Kanade settings, public RGB-D and robot mask",
        "assumption": "One dominant moving rigid part; upright revolute motion; movement under 15cm per correspondence",
        "native_information_used": False})
    previous, previous_load = load(folders[0], names)
    results = []
    for a, b in zip(folders[:-1], folders[1:]):
        started = time.perf_counter()
        current, load_seconds = load(b, names)
        matched_at = time.perf_counter()
        first, second, counts = correspondences(previous, current, center, radius)
        matched_seconds = time.perf_counter() - matched_at
        fit_at = time.perf_counter()
        fit = estimate_rigid_hinge(first, second, prior=None, seed=42)
        fit_seconds = time.perf_counter() - fit_at
        pair = a.name + "-" + b.name
        np.savez_compressed(output / (pair + ".npz"), reference_world=first, current_world=second)
        row = {"id": pair, "before": a.name, "after": b.name, "fit": fit, "camera_counts": counts,
               "timing": {"load_seconds": load_seconds, "flow_seconds": matched_seconds, "fit_seconds": fit_seconds,
                          "total_seconds": time.perf_counter() - started}}
        results.append(row)
        write(output / "results.json", results)
        print(json.dumps({"id": pair, "accepted": fit["accepted"], "reason": fit["reason"],
                          "correspondences": fit["correspondences"], "seconds": row["timing"]["total_seconds"]}), flush=True)
        previous = current
    write(output / "completion.json", {"pairs": len(results), "first_frame_load_seconds": previous_load,
        "accepted": sum(r["fit"]["accepted"] for r in results), "native_references_read": False})


if __name__ == "__main__":
    main()
