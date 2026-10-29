"""Score already saved RoArtNet outputs using observer-only joint references."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlm_tamp.articulation_metrics import axis_errors


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference_for(item):
    if item["kind"] == "author":
        path = ROOT / item["author_source_for_later_scoring"]
        with np.load(path, allow_pickle=False) as data:
            joint = data["joints"][0]
        return {"origin": joint[:3].tolist(), "axis": joint[3:6].tolist(),
                "affordance": joint[6:9].tolist(), "frame": "optical_camera_metres",
                "source": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "supplied_to_estimator": False}
    trial = ROOT / "results/astra-luna-comparison-20261023/trials" / item["trial"]
    robot = read(trial / "agent-workspace/robot_data/attempt01" / item["observation"] / "robot.json")
    path = trial / "attempt01/initial/evaluation_contact_diagnostics.json"
    diagnostics = read(path)
    assert diagnostics["evaluation_only"] and diagnostics["excluded_from_policy_input"]
    records = [x for x in diagnostics["steps"] if x["step"] == robot["control_steps"]]
    if not records:
        raise ValueError(f"No native reference at exact observation step for {item['id']}")
    joints = [x for x in records[-1]["evaluation_only_articulation_state"] if x["joint_name"] == "microwave_1_microjoint"]
    if len(joints) != 1:
        raise ValueError("Ambiguous native evaluation joint")
    joint = joints[0]
    return {"origin": joint["anchor_world_m"], "axis": joint["axis_world"], "frame": "world_metres",
            "position_rad": joint["position"], "control_step": robot["control_steps"],
            "source": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "supplied_to_estimator": False}


def stats(rows):
    complete = [r for r in rows if r["status"] == "complete"]
    fields = ["origin_to_reference_line_mm", "reference_plane_error_mm", "shortest_line_distance_mm", "axis_angle_deg"]
    out = {"cases": len(rows), "complete": len(complete)}
    for field in fields:
        values = [r["errors"][field] for r in complete if r["errors"][field] is not None]
        out[field] = {"median": float(np.median(values)), "mean": float(np.mean(values)), "max": float(np.max(values))} if values else None
    if complete:
        keys = [k for k in complete[0]["prediction"]["timing"] if k.endswith("_seconds")]
        out["mean_timing_seconds"] = {key: float(np.mean([r["prediction"]["timing"][key] for r in complete])) for key in keys}
        out["median_inference_seconds"] = float(np.median([r["prediction"]["timing"]["total_inference_seconds"] for r in complete]))
        out["within_30mm_and_5deg"] = sum(r["errors"]["reference_plane_error_mm"] is not None
            and r["errors"]["reference_plane_error_mm"] <= 30 and r["errors"]["axis_angle_deg"] <= 5 for r in complete)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/roartnet-perception-20261025")
    args = parser.parse_args()
    run = args.run.resolve()
    items = {x["id"]: x for x in read(run / "inputs.json")}
    rows, references = [], {}
    # All predictions must already exist before private scoring starts.
    paths = sorted((run / "predictions").glob("*.json"))
    if len(paths) != 38 or any(read(p)["status"] != "complete" for p in paths):
        raise ValueError("Expected all 38 predeclared predictions before private scoring")
    for path in paths:
        row = read(path)
        item = items[row["input_id"]]
        if item["id"] not in references:
            references[item["id"]] = reference_for(item)
        reference = references[item["id"]]
        pred = row["prediction"]
        if item["kind"] == "author":
            origin, axis = pred["joint_origin_optical_m"], pred["joint_axis_optical"]
            affordance_error = np.linalg.norm(np.asarray(pred["affordance_optical_m"]) - reference["affordance"]) * 1000
            row["affordance_error_mm"] = float(affordance_error)
        else:
            origin, axis = pred["joint_origin_world_m"], pred["joint_axis_world"]
        row["errors"] = axis_errors(origin, axis, reference["origin"], reference["axis"])
        row["origin_delta_in_reference_frame_m"] = (np.asarray(origin) - reference["origin"]).tolist()
        row["reference"] = reference
        row["input"] = item
        row["prediction_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    write(run / "scored-private.json", rows)
    write(run / "references-private.json", references)
    summary = {"inferences": len(rows), "successful_computations": len(rows), "groups": {},
               "interpretation": "Complete computation is not physical success; static hypotheses are not motion-verified.",
               "comparison_limit": "One local microwave asset; five closed front views are nearly identical. No population-level ranking or unseen-asset generalization claim."}
    for profile in ["author-eval", "checkpoint"]:
        selected = [r for r in rows if r["profile"] == profile and r["seed"] == 42]
        summary["groups"][profile + "/author"] = stats([r for r in selected if r["kind"] == "author"])
        summary["groups"][profile + "/local"] = stats([r for r in selected if r["kind"] == "local" and r["variant"] == "sam"])
        summary["groups"][profile + "/closed-front"] = stats([r for r in selected if r["kind"] == "local" and r["variant"] == "sam"
            and r["input"]["camera"] == "frontview" and r["input"]["observation"] == "obs001"])
    summary["initial_luna_comparison"] = []
    old = {r["trial"]: r for r in read(ROOT / "results/astra-luna-comparison-20261023/grounding-diagnostics-private.json")}
    for r in rows:
        i = r["input"]
        if r["profile"] == "author-eval" and r["seed"] == 42 and i["kind"] == "local" and i["variant"] == "sam" and i["camera"] == "frontview" and i["observation"] == "obs001":
            o = old[i["trial"]]
            summary["initial_luna_comparison"].append({"trial": i["trial"], "luna_original_hinge_error_mm": o["hinge_axis_line_position_error_m"] * 1000,
                "roartnet_origin_to_line_mm": r["errors"]["origin_to_reference_line_mm"], "roartnet_plane_error_mm": r["errors"]["reference_plane_error_mm"]})
    summary["gpu_memory_peak_mib"] = max(r["gpu0_total_mib"] for r in [json.loads(line) for line in (run / "gpu-memory.jsonl").read_text().splitlines()] if r["gpu0_total_mib"] is not None)
    summary["gpu_memory_scope"] = "Sampled total GPU0 allocation during isolated benchmark; includes library contexts and caches, not minimum model memory."
    write(run / "summary.json", summary)
    with (run / "timing-and-accuracy.csv").open("w") as f:
        timing_keys = [k for k in rows[0]["prediction"]["timing"] if k.endswith("_seconds")]
        writer = csv.DictWriter(f, fieldnames=["id", "kind", "profile", "seed", "origin_to_reference_line_mm", "reference_plane_error_mm", "axis_angle_deg", *timing_keys])
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in ["id", "kind", "profile", "seed"]},
                **{k: r["errors"][k] for k in ["origin_to_reference_line_mm", "reference_plane_error_mm", "axis_angle_deg"]},
                **{k: r["prediction"]["timing"][k] for k in timing_keys}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
