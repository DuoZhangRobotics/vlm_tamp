"""Independent object-axis constraints derived from unchanged RGB-D blanks.

Object-frame relations do not require a grasp proposal. Binding the object-axis
frame to robot joint coordinates still requires a grasp/attachment transform.
"""

import numpy as np

from .pose_constraints import perpendicular_basis
from .rgbd_numericals import UPRIGHT_SUFFIX, compile_numericals


def compile_independent_axis(blanks, payload, depth, calibration, config):
    result = compile_numericals(blanks, payload, depth, calibration, config)
    result["schema_version"] = "rgbd-numerical-1.1"
    result["derivation"] = "same VLM blanks and geometry policy; object-axis constraints independent of grasp"
    axis = result["object_axis"]
    if blanks["path_constraint"] != "keep_upright" or axis["status"] != "estimated":
        return result
    direction = np.asarray(axis["axis_world"])
    basis = perpendicular_basis(direction)
    rotation = np.column_stack([basis, direction])
    origin = result["landmarks"]["axis_low_uv"]["position_m"]
    axis["estimated_axis_frame"] = {
        "position_world_m": origin, "rotation_world": rotation.tolist(),
        "definition": "z follows lower-to-upper observed surface chord; origin lower point; yaw arbitrary",
        "canonical_object_frame_verified": False,
    }
    common = {"axis_local": [0., 0., 1.], "axis_world": [0., 0., 1.],
              "scope": "path", "stage": "transport", "frame": "world",
              "state_pose": "estimated_object_axis_frame",
              "required_by_instruction": UPRIGHT_SUFFIX.strip() in payload["instruction"],
              "semantic_axis_verified": False, "robot_joint_binding_status": "unresolved",
              "attachment_transform_status": "unresolved"}
    result["relations"] = [spec for spec in result["relations"] if not spec["id"].startswith("upright_")]
    result["relations"].extend([
        {**common, "id": "upright_manifold", "type": "axis_alignment",
         "equality_residual_dimension": 2, "positive_axis_branch_inequality": True},
        {**common, "id": "upright_tolerance", "type": "axis_cone",
         "max_angle_degrees": config["upright_tolerance_degrees"],
         "tolerance_source": "configured policy; not inferred from RGB"},
    ])
    result["unresolved"] = [reason for reason in result["unresolved"]
                            if reason != "Upright relation selected but grasp-frame/axis geometry unresolved"]
    result["unresolved"].append("Object-axis residuals generated; robot/grasp attachment transform unresolved")
    return result
