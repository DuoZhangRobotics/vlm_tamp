"""Replay saved arc endpoints through native planning and validate every returned sample.

The goals JSON is a list of {"name": str, "joints": [seven joint angles]}.
This integration check needs a built native library and an archived arc scene;
it does not start the simulator, a model, or a GPU worker.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".runtime/mc-vamp-deps")]

import numpy as np

from benchmark_curobo_v2 import load_saved_arc
from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.mcvamp_contact_numerical import NativeScene, clear_native_cache
from vlm_tamp.partial_pose_arc import path_wrist_admitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-arc", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = read_json(args.saved_arc.parent / "robot-chain.json")
    scene, metadata, start, inputs, reference = load_saved_arc(args.saved_arc, record)
    angle = inputs["goal_angle_rad"]
    rows = []
    clear_native_cache()
    try:
        with NativeScene(scene, record, metadata, start, angle, aligned=False,
                         grasp_constraint="point") as context:
            context.motion_options = {**reference["configuration"], **inputs["search_options"]}
            for index, item in enumerate(read_json(args.goals)):
                goal = np.asarray(item["joints"], dtype=float)
                path, result = context.plan(start, goal, seconds=args.seconds, iterations=5000)
                row = {"name": item["name"], "result": result, "passed": False}
                if len(path):
                    wrist_ok, wrist = path_wrist_admitted(context, path, angle)
                    collision_checks = [context.validate(q) for q in path]
                    projections = [context.project(q) for q in path]
                    # A valid sample is already on the manifold. Projection must
                    # accept it without changing its float32 configuration.
                    unchanged = all(np.array_equal(projected, np.asarray(q, np.float32))
                                    for q, (projected, _) in zip(path, projections))
                    constraints_ok = all(check["accepted"] and np.isfinite(check["residual"])
                                         and check["residual"] <= 1e-6
                                         and check["position_error_m"] <= context.position_tolerance_m + 1e-8
                                         for _, check in projections)
                    collision_ok = all(check["accepted"] for check in collision_checks)
                    row.update(
                        dense_samples=len(path), wrist=wrist, wrist_admitted=bool(wrist_ok),
                        all_samples_collision_admitted=collision_ok,
                        all_samples_already_on_constraint=bool(unchanged and constraints_ok),
                        maximum_constraint_residual=max(check["residual"] for _, check in projections),
                        constraint_residual_units="scaled_dimensionless_acceptance_ratio",
                        maximum_position_error_m=max(check["position_error_m"] for _, check in projections),
                        position_tolerance_m=context.position_tolerance_m,
                        maximum_adjacent_joint_step_rad=float(np.max(np.abs(np.diff(path, axis=0)))),
                        passed=bool(result["exact_solution"] and wrist_ok and collision_ok
                                    and unchanged and constraints_ok),
                    )
                    np.save(args.output / f"path-{index:02d}.npy", path)
                rows.append(row)
                write_json(args.output / "results.json", rows)
                print(f"{item['name']}: {result['status']}; passed={row['passed']}; "
                      f"seconds={result['total_seconds']:.3f}", flush=True)
    finally:
        clear_native_cache()
    if not rows or not all(row["passed"] for row in rows):
        raise SystemExit("At least one saved endpoint failed strict native validation")


if __name__ == "__main__":
    main()
