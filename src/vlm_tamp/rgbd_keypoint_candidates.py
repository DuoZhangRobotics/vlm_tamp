"""Appearance/geometry candidate proposals without task or simulator identities.

Features must already be aligned to the original RGB/depth pixel grid. This
module runs no image model; it clusters RGB-derived regions and measured 3D
positions, then returns actual sensor pixels rather than cluster centroids.
"""

import numpy as np


DEFAULT_SETTINGS = {
    "candidates_per_region": 8,
    "max_regions": 24,
    "min_region_pixels": 12,
    "min_separation_m": 0.003,
    "bounds_min": None,
    "bounds_max": None,
    "minimum_depth_m": None,
    "maximum_depth_m": None,
}


def validated_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("Proposal settings must be a dictionary")
    if settings and set(settings) - set(DEFAULT_SETTINGS):
        raise ValueError("Unknown candidate proposal setting")
    config = {**DEFAULT_SETTINGS, **(settings or {})}
    for name in ("candidates_per_region", "max_regions", "min_region_pixels"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError("Proposal count settings must be positive integers")
    distance = config["min_separation_m"]
    if (isinstance(distance, bool) or not isinstance(distance, (int, float))
            or not np.isfinite(distance) or distance < 0):
        raise ValueError("Candidate separation must be finite and nonnegative")
    lower, upper = config["bounds_min"], config["bounds_max"]
    if (lower is None) != (upper is None):
        raise ValueError("Both world workspace bounds must be supplied together")
    if lower is not None:
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all()
                or not np.isfinite(upper).all() or np.any(lower >= upper)):
            raise ValueError("World workspace bounds must be finite ordered XYZ vectors")
        config["bounds_min"], config["bounds_max"] = lower.tolist(), upper.tolist()
    for name in ("minimum_depth_m", "maximum_depth_m"):
        value = config[name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value <= 0
        ):
            raise ValueError("Optical depth bounds must be positive finite meters")
    minimum, maximum = config["minimum_depth_m"], config["maximum_depth_m"]
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError("Minimum optical depth must be less than maximum depth")
    return config


def _regions(predicted_regions, shape):
    if isinstance(predicted_regions, list):
        regions = []
        for index, raw in enumerate(predicted_regions, 1):
            mask = np.asarray(raw)
            if mask.shape != shape or not np.issubdtype(mask.dtype, np.number) and mask.dtype != bool:
                raise ValueError("RGB-predicted masks must match the original image grid")
            if not np.isfinite(mask).all():
                raise ValueError("RGB-predicted masks must be finite")
            regions.append((index, mask.astype(bool)))
        return regions
    labels = np.asarray(predicted_regions)
    if labels.shape != shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Predicted region labels must be an integer map on the original image grid")
    if np.any(labels < 0):
        raise ValueError("Predicted region labels must be nonnegative")
    return [(int(label), labels == label) for label in np.unique(labels) if label != 0]


def _normalize_columns(values):
    shifted = values - values.min(axis=0)
    ranges = np.ptp(values, axis=0)
    return shifted / np.where(ranges > 1e-12, ranges, 1.)


def _cluster_region(features, positions, count):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    appearance = np.asarray(features, dtype=np.float64)
    centered = appearance - appearance.mean(axis=0)
    components = min(3, appearance.shape[1], len(appearance))
    if np.any(np.abs(centered) > 1e-12):
        reduced = PCA(n_components=components, svd_solver="randomized", random_state=0).fit_transform(appearance)
    else:
        reduced = np.zeros((len(appearance), components))
    combined = np.column_stack([_normalize_columns(reduced), _normalize_columns(positions)])
    count = min(count, len(combined), len(np.unique(combined, axis=0)))
    model = KMeans(n_clusters=count, random_state=0, n_init=10).fit(combined)
    selected = []
    for cluster in range(count):
        members = np.flatnonzero(model.labels_ == cluster)
        if not len(members):
            continue
        distance = np.sum((combined[members] - model.cluster_centers_[cluster]) ** 2, axis=1)
        selected.append(int(members[np.argmin(distance)]))
    return selected


def propose_candidates(features, world_points, predicted_regions, settings=None, *, depth_m=None):
    """Return deterministic original pixels; depth is optical z, never world z.

    Region zero is always excluded. Region masks may overlap and remain
    independent. There are no object, task, reference-mask, or inventory inputs.
    """
    config = validated_settings(settings)
    features, points = np.asarray(features), np.asarray(world_points, dtype=float)
    if (features.ndim != 3 or features.shape[2] < 1 or not np.issubdtype(features.dtype, np.number)
            or points.shape != (*features.shape[:2], 3)):
        raise ValueError("Dense features and world points must share the original HxW pixel grid")
    height, width = features.shape[:2]
    if not height or not width:
        raise ValueError("Original image grid cannot be empty")
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(features).all(axis=-1)
    if config["bounds_min"] is not None:
        valid &= ((points >= config["bounds_min"]) & (points <= config["bounds_max"])).all(axis=-1)
    if depth_m is None:
        if config["minimum_depth_m"] is not None or config["maximum_depth_m"] is not None:
            raise ValueError("Declared optical depth bounds require the aligned depth_m array")
    else:
        depth = np.asarray(depth_m, dtype=float)
        if depth.shape != (height, width):
            raise ValueError("Metric depth must share the original RGB pixel grid")
        valid &= np.isfinite(depth) & (depth > 0)
        if config["minimum_depth_m"] is not None:
            valid &= depth >= config["minimum_depth_m"]
        if config["maximum_depth_m"] is not None:
            valid &= depth <= config["maximum_depth_m"]
    eligible = []
    for region_id, mask in _regions(predicted_regions, (height, width)):
        region = mask & valid
        if np.count_nonzero(region) >= config["min_region_pixels"]:
            eligible.append((region_id, region))
    # Cap eligible regions in supplied/label order. No task or outcome-based ranking.
    proposed = []
    for region_id, region in eligible[:config["max_regions"]]:
        pixels = np.argwhere(region)
        positions = points[region]
        selected = _cluster_region(features[region], positions, config["candidates_per_region"])
        selected.sort(key=lambda index: tuple(pixels[index]))
        retained = []
        for index in selected:
            if any(np.linalg.norm(positions[index] - positions[other]) < config["min_separation_m"]
                   for other in retained):
                continue
            retained.append(index)
            y, x = (int(value) for value in pixels[index])
            proposed.append({"pixel_xy": [x, y], "uv": [(x + .5)/width, (y + .5)/height],
                             "position_world_m": positions[index].tolist(),
                             "predicted_region_id": region_id})
    proposed.sort(key=lambda item: (item["pixel_xy"][1], item["pixel_xy"][0], item["predicted_region_id"]))
    return [{"id": index, **item} for index, item in enumerate(proposed)]
