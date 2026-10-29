"""Test the native positional tube against saved observed robot/scene geometry.

Uses numerical FK targets to place the contact point 2.5 mm and 3.5 mm off its
hinge circle. Also checks the Euclidean tube boundary, unchanged angular limits,
and that tolerance changes cannot admit a palm collision. No simulator or GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".runtime/mc-vamp-deps")]

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from benchmark_curobo_v2 import load_saved_arc
from vlm_tamp.coupled_storage import read_json, write_json
from vlm_tamp.mcvamp_contact_numerical import NativeScene, clear_native_cache
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.pose_constraints import skew
from vlm_tamp.rekep_libero.robot_chain_v4 import fixed_orientation_residual


def displaced_pose(context, offset, rotation=None):
    start = context.reference_q
    initial = pose_matrix(context.chain.forward(start))
    desired_rotation = initial[:3, :3] if rotation is None else rotation
    desired_point = context.anchor + np.asarray(offset)

    def residual(q, jacobian=False):
        pose, jac = context.chain.forward(q, jacobian=True)
        tcp = pose_matrix(pose)
        point = tcp[:3, 3] + tcp[:3, :3] @ context.pad_local
        angular, angular_jac = fixed_orientation_residual(desired_rotation, tcp[:3, :3], jac[3:])
        if jacobian:
            return np.vstack([jac[:3] - tcp[:3, :3] @ skew(context.pad_local) @ jac[3:],
                              .15 * angular_jac, 1e-6 * np.eye(7)])
        return np.r_[point - desired_point, .15 * angular, 1e-6 * (q - start)]

    bounds = context.chain.limits + [.01, -.01]
    result = least_squares(residual, start, jac=lambda q: residual(q, True),
                           bounds=bounds.T, max_nfev=200,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if np.linalg.norm(residual(result.x)[:6]) > 1e-7:
        raise ValueError(f"Synthetic FK target did not converge: {result.message}; residual={residual(result.x)}")
    return result.x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-arc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = read_json(args.saved_arc.parent / "robot-chain.json")
    scene, metadata, start, inputs, _ = load_saved_arc(args.saved_arc, record)
    scene.config["arc_position_tolerance_m"] = .003
    checks = []

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})
        write_json(args.output / "checks.json", checks)
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)

    clear_native_cache()
    try:
        with NativeScene(scene, record, metadata, start, inputs["goal_angle_rad"],
                         aligned=False) as context:
            radial = context.anchor - context.origin
            radial -= context.axis * (radial @ context.axis)
            radial /= np.linalg.norm(radial)
            for name, offset, expected in [
                ("2.5mm_height_inside", .0025 * context.axis, True),
                ("3.5mm_height_outside", .0035 * context.axis, False),
                ("2.5mm_radial_inside", .0025 * radial, True),
                ("2.5mm_each_axis_outside", .0025 * (radial + context.axis), False),
            ]:
                q = displaced_pose(context, offset)
                error = context.constraint_errors(q)
                check(name, error["accepted"] == expected
                      and abs(error["position_error_m"] - np.linalg.norm(offset)) < 1e-6,
                      error=error, joints=q.tolist())
                projected, projection = context.project(q)
                if expected:
                    check(name + "_projection_unchanged", projection["accepted"]
                          and np.array_equal(projected, np.asarray(q, np.float32)),
                          projection=projection)
                else:
                    check(name + "_projection_corrects", projection["accepted"]
                          and projection["position_error_m"] <= .003,
                          projection=projection)

            context.lib.mcvamp_set_grasp_constraint(context.handle, 1)
            initial = pose_matrix(context.chain.forward(start))
            target_rotation = Rotation.from_rotvec(.01 * context.axis).as_matrix() @ initial[:3, :3]
            q = displaced_pose(context, np.zeros(3), target_rotation)
            error = context.constraint_errors(q)
            check("angular_error_not_relaxed", not error["accepted"]
                  and error["position_error_m"] < 1e-6
                  and error["angular_weighted_error"] > 1e-4, error=error)
            context.lib.mcvamp_set_grasp_constraint(context.handle, 0)

        # Inject an observed point at the configured palm centre. It must remain
        # a static collision at both the old and new positional tolerances.
        palm = next(v for v in metadata["robot_collision_volumes_world"]
                    if v["robot_geom_name"] == "gripper0_hand_collision")
        point = np.mean(palm["vertices_world"], axis=0)
        scene.static_points = np.vstack([scene.static_points, point])
        with NativeScene(scene, record, metadata, start, inputs["goal_angle_rad"],
                         aligned=False) as context:
            results = []
            for tolerance in [1e-6, .003]:
                assert context.lib.mcvamp_set_position_tolerance(context.handle, tolerance)
                results.append(context.validate(start))
            check("palm_collision_unchanged", all(not r["accepted"] for r in results)
                  and all(r["last_rejected_part"] == 1000 for r in results)
                  and all(r["last_rejected_surface"] == 0 for r in results), results=results)
        write_json(args.output / "summary.json", {"all_passed": all(r["passed"] for r in checks),
                   "checks": len(checks), "position_tolerance_m": .003,
                   "saved_arc": str(args.saved_arc), "robot_motion_issued": False})
    finally:
        clear_native_cache()
    if not checks or not all(r["passed"] for r in checks):
        raise SystemExit("At least one positional tolerance check failed")


if __name__ == "__main__":
    main()
