"""Show native planner rejection witnesses; never admit, reject or repair a plan."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .rekep_libero.geometry import transform_points


def project(view: dict, point: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a known world point through the saved camera calibration."""
    optical = transform_points([point], np.linalg.inv(np.asarray(view["camera_to_world"])))[0]
    if optical[2] <= 0:
        return np.array([np.nan, np.nan]), float(optical[2])
    pixel = np.asarray(view["intrinsic"]) @ optical
    return pixel[:2] / pixel[2], float(optical[2])


def render_contact_rejection(
    diagnostic: dict,
    scene: Any,
    cameras: dict[str, dict],
    source_camera: str,
    target_point: np.ndarray,
    folder: Path,
) -> list[dict]:
    """Render one reported endpoint witness in the selected and a useful existing view.

    The nearest point comes from the planner's existing cloud. Camera ranking uses
    agreement with that already observed point, not a new observation or path test.
    """
    checks = [check for attempt in diagnostic.get("attempts", [])
        for check in attempt.get("planning", {}).get("endpoints", {}).values()
        if check.get("accepted") is False and check.get("blocking_sphere_world_m")]
    if not checks:
        return []
    check = checks[0]
    sphere = np.asarray(check["blocking_sphere_world_m"], float)
    clouds = {0: scene.static_points, 1: scene.moving_reference, 2: scene.contact_reference}
    cloud = np.asarray(clouds[int(check["last_rejected_surface"])])
    if not len(cloud):
        return []
    distances = np.linalg.norm(cloud-sphere[:3], axis=1)
    nearest = cloud[np.argmin(distances)]
    views = []
    for name, view in cameras.items():
        pixel, depth = project(view, nearest)
        h, w = view["rgb"].shape[:2]
        if depth <= 0 or not (0 <= pixel[0] < w and 0 <= pixel[1] < h):
            continue
        x, y = np.rint(pixel).astype(int)
        x, y = min(x, w-1), min(y, h-1)
        measured = np.asarray(view["world_points"])[y, x]
        if np.isfinite(measured).all():
            views.append((float(np.linalg.norm(measured-nearest)), name))
    views.sort()
    names = [source_camera] if source_camera in cameras else []
    names.extend(name for _, name in views if name not in names)
    records = []
    for name in names[:2]:
        view = cameras[name]
        picture = Image.fromarray(view["rgb"]).copy()
        draw = ImageDraw.Draw(picture)
        centre, depth = project(view, sphere[:3])
        if depth > 0:
            k = np.asarray(view["intrinsic"])
            radius = sphere[3]*np.array([k[0, 0], k[1, 1]])/depth
            draw.ellipse(tuple(centre-radius)+tuple(centre+radius), outline="#ff5555", width=3)
        for point, color in [(target_point, "#ffdc45"), (nearest, "#40bfff")]:
            pixel, depth = project(view, point)
            if depth > 0:
                x, y = pixel
                draw.line((x-5, y, x+5, y), fill=color, width=3)
                draw.line((x, y-5, x, y+5), fill=color, width=3)
        draw.rectangle((0, 0, picture.width, 57), fill="#152536")
        draw.text((7, 5), "REJECTED ENDPOINT - NOT EXECUTED", fill="white")
        draw.text((7, 22), "red: planner sphere | blue: observed point", fill="white")
        draw.text((7, 39), "yellow: requested touch projection", fill="#ffdc45")
        path = folder/f"rejection-{name}.png"
        picture.save(path)
        records.append({"camera": name, "image": str(path),
            "blocking_robot_part": check.get("blocking_robot_part"),
            "blocking_surface": check.get("blocking_surface"),
            "sphere_world_m": sphere.tolist(),
            "nearest_observed_point_world_m": nearest.tolist(),
            "nearest_distance_m": float(distances.min()),
            "sampled_overlap_m": float(sphere[3]-distances.min()),
            "additional_collision_admission_checks": 0,
            "physical_motion_depicted": False})
    return records
