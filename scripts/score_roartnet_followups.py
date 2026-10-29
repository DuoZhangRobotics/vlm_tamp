"""Private scoring after the independent cleanup and motion probes finish."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from evaluate_roartnet import reference_for
from vlm_tamp.articulation_metrics import axis_errors


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    run = ROOT / "results/roartnet-perception-20261025"
    if not all((run / x / "completion.json").is_file() for x in ["motion", "cleanup"]):
        raise ValueError("Both public-input probes must finish before private scoring")
    refs = read(run / "references-private.json")
    cleanup = read(run / "cleanup/predictions.json")
    for row in cleanup:
        ref, pred = refs[row["input_id"]], row["prediction"]
        row["errors"] = axis_errors(pred["joint_origin_world_m"], pred["joint_axis_world"], ref["origin"], ref["axis"])
    write(run / "cleanup/scored-private.json", cleanup)
    motion = read(run / "motion/results.json")
    for row in motion:
        references = {}
        for side in ["before", "after"]:
            references[side] = reference_for({"kind": "local", "trial": "state020-r1-luna",
                                             "observation": row[side], "id": row[side]})
        ref, fit = references["after"], row["fit"]
        row["native_rotation_deg_private"] = float(np.degrees(ref["position_rad"] - references["before"]["position_rad"]))
        row["references_private"] = references
        if fit["accepted"]:
            row["errors"] = axis_errors(fit["hinge_world_m"], [0, 0, 1], ref["origin"], ref["axis"])
            row["axis_direction_source"] = "upright SE2 assumption, not inferred 3D direction"
            row["rotation_error_deg"] = abs(float(np.degrees(fit["rotation_rad"])) - row["native_rotation_deg_private"])
    write(run / "motion/scored-private.json", motion)
    accepted = [r for r in motion if r["fit"]["accepted"]]
    write(run / "followup-summary.json", {
        "scope": "Exploratory diagnostics after original benchmark, same asset/trajectory; not held-out transfer",
        "cleanup_cases": len(cleanup),
        "motion_pairs": len(motion), "motion_accepted": len(accepted),
        "motion_error_mm": {"median": float(np.median([r["errors"]["reference_plane_error_mm"] for r in accepted])),
                            "min": min(r["errors"]["reference_plane_error_mm"] for r in accepted),
                            "max": max(r["errors"]["reference_plane_error_mm"] for r in accepted)},
        "motion_max_rotation_error_deg": max(r["rotation_error_deg"] for r in accepted),
        "motion_pair_mean_wall_seconds": float(np.mean([r["timing"]["total_seconds"] for r in motion])),
        "automatic_integration": False})
    for row in cleanup:
        print(row["input_id"], row["cleanup"]["retained_points"], row["errors"])
    print(read(run / "followup-summary.json"))


if __name__ == "__main__":
    main()
