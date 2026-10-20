"""RGB-predicted multiview target masks seeded by visible cached object points.

Geometric proximity does not add unlabeled surface points to the target. Each
additional view must produce an RGB mask supported by depth-consistent visible
seeds and by the measured extent of the cached main-view object.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from .geometry import pose_matrix, transform_points


def load_mobile_sam_predictor(device='cuda'):
    """Load only the existing local segmentation checkpoint, without DINO."""
    from vlm_tamp.libero import ROOT
    sys.path.insert(0, str(ROOT / 'third_party/MobileSAM'))
    from mobile_sam import SamPredictor, sam_model_registry
    model = sam_model_registry['vit_t'](
        checkpoint=str(ROOT / 'third_party/MobileSAM/weights/mobile_sam.pt')).eval().to(device)
    return SamPredictor(model)


def visible_projected_points(world_points, camera, depth_tolerance=.004):
    """Only source points agreeing with observed optical depth become seeds."""
    points = np.asarray(world_points, float)
    depth = np.asarray(camera['depth'], float).squeeze()
    intrinsic = np.asarray(camera['intrinsic'], float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or depth.ndim != 2 or intrinsic.shape != (3, 3)):
        raise ValueError('Visible projection requires finite calibrated RGB-D points')
    if not np.isfinite(depth_tolerance) or not 0 < depth_tolerance <= .004:
        raise ValueError('Seed depth agreement must be positive and at most 4mm')
    optical = transform_points(points, np.linalg.inv(pose_matrix(camera['camera_to_world'])))
    z = optical[:, 2]
    projected = optical @ intrinsic.T
    pixels = np.rint(projected[:, :2] / np.where(z > 0, z, 1)[:, None]).astype(int)
    x, y = pixels[:, 0], pixels[:, 1]
    in_view = ((z > 0) & (x >= 0) & (y >= 0) & (x < depth.shape[1]) & (y < depth.shape[0]))
    indices = np.flatnonzero(in_view)
    measured = depth[y[indices], x[indices]]
    valid = np.isfinite(measured) & (measured > 0)
    errors = np.abs(measured - z[indices])
    visible = valid & (errors <= depth_tolerance)
    visible_indices = indices[visible]
    visible_pixels = pixels[visible_indices]
    if len(visible_pixels):
        _, keep = np.unique(visible_pixels, axis=0, return_index=True)
        visible_pixels, visible_indices = visible_pixels[np.sort(keep)], visible_indices[np.sort(keep)]
    diagnostics = {'source_points': len(points), 'projected_in_frustum': int(in_view.sum()),
                   'visible_unique_pixels': len(visible_pixels),
                   'occluded_source_points': int((valid & (measured < z[indices] - depth_tolerance)).sum()),
                   'depth_mismatched_source_points': int((valid & ~visible).sum()),
                   'invalid_projected_depth': int((~valid).sum()),
                   'depth_agreement_tolerance_m': depth_tolerance}
    return visible_pixels.astype(float), visible_indices, diagnostics


def _prompt_points(pixels, maximum=8):
    """Deterministic image-spread positives, beginning nearest the centroid."""
    pixels = np.asarray(pixels, float)
    if len(pixels) <= maximum:
        return pixels.copy()
    selected = [int(np.argmin(np.linalg.norm(pixels - pixels.mean(axis=0), axis=1)))]
    distance = np.full(len(pixels), np.inf)
    while len(selected) < maximum:
        distance = np.minimum(distance, np.linalg.norm(pixels - pixels[selected[-1]], axis=1))
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    return pixels[selected]


def _save_mask(out, name, camera, mask, positive_pixels):
    np.save(out / f'{name}-mask.npy', mask)
    rgb = np.asarray(camera['rgb'], np.uint8).copy()
    rgb[mask] = np.rint(.5 * rgb[mask] + .5 * np.asarray([0, 255, 0])).astype(np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for x, y in positive_pixels:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline='yellow', width=1)
    image.save(out / f'{name}-overlay.png')


def infer_multiview_target_cloud(bundle, source_mask, sam_predictor, outdir, *,
                                depth_tolerance=.004, extent_padding=.003,
                                maximum_mask_ratio=.25, minimum_bounded_fraction=.80,
                                minimum_seed_coverage=.80, minimum_mask_points=30,
                                minimum_visible_seeds=3):
    """Return ``(cloud, camera_masks, report)`` with all main points preserved.

    Extent checks reject an unrelated/tabletop RGB mask rather than converting a
    large mask into an object using geometric clipping alone. Accepted masks
    are finally clipped to the bounded measured reference extent.
    """
    cameras = bundle['cameras']
    main_name = bundle['main_camera']
    main = cameras[main_name]
    source_mask = np.asarray(source_mask, bool)
    points_main = np.asarray(main['world_points'], float)
    if source_mask.shape != np.asarray(main['depth']).squeeze().shape or points_main.shape != source_mask.shape + (3,):
        raise ValueError('Cached RGB mask must match calibrated main-view points')
    valid_main = (np.isfinite(points_main).all(axis=2)
                  & np.isfinite(main['depth']) & (np.asarray(main['depth']) > 0))
    if np.any(source_mask & ~valid_main):
        raise ValueError('Every cached main-view target pixel needs valid metric depth')
    reference = points_main[source_mask]
    if len(reference) < minimum_mask_points:
        raise ValueError('Cached RGB object mask has insufficient point support')
    if not np.isfinite(extent_padding) or not 0 <= extent_padding <= .004:
        raise ValueError('Measured reference extent padding must be at most 4mm')
    if not 0 < maximum_mask_ratio < .5 or not 0 < minimum_bounded_fraction <= 1 or not 0 < minimum_seed_coverage <= 1:
        raise ValueError('RGB mask admission thresholds are invalid')
    low, high = reference.min(axis=0) - extent_padding, reference.max(axis=0) + extent_padding
    masks = {main_name: source_mask.copy()}
    additions, rows = [], []
    out = None if outdir is None else Path(outdir)
    if out is not None:
        if (out / 'quality.json').exists():
            raise FileExistsError('Preserve an existing target-region report')
        out.mkdir(parents=True, exist_ok=True)
        _save_mask(out, main_name, main, source_mask, [])
        np.save(out / 'main-reference-cloud.npy', reference)
    for name, camera in cameras.items():
        if name == main_name:
            continue
        visible, _, diagnostics = visible_projected_points(reference, camera, depth_tolerance)
        row = {'camera': name, **diagnostics, 'accepted': False, 'candidate_masks': []}
        if len(visible) < minimum_visible_seeds:
            row['reason'] = 'insufficient_depth_visible_object_seeds'
            rows.append(row)
            continue
        prompts = _prompt_points(visible)
        row['positive_prompt_pixels_xy'] = prompts.tolist()
        points = np.asarray(camera['world_points'], float)
        depth = np.asarray(camera['depth'], float).squeeze()
        valid = np.isfinite(points).all(axis=2) & np.isfinite(depth) & (depth > 0)
        bounded = valid & (points >= low).all(axis=2) & (points <= high).all(axis=2)
        sam_predictor.set_image(np.asarray(camera['rgb'], np.uint8))
        predicted, scores, _ = sam_predictor.predict(point_coords=prompts,
            point_labels=np.ones(len(prompts), int), multimask_output=True)
        options = []
        visible_xy = visible.astype(int)
        for index, (predicted_mask, score) in enumerate(zip(predicted, scores)):
            mask = np.asarray(predicted_mask, bool)
            if mask.shape != depth.shape:
                raise ValueError('RGB segmentation prediction and depth shapes disagree')
            rgb_area = int(mask.sum())
            valid_area = int((mask & valid).sum())
            bounded_area = int((mask & bounded).sum())
            seed_coverage = float(np.mean(mask[visible_xy[:, 1], visible_xy[:, 0]]))
            bounded_fraction = bounded_area / max(1, valid_area)
            record = {'mask_index': index, 'score': float(score), 'rgb_mask_pixels': rgb_area,
                      'valid_depth_pixels': valid_area, 'bounded_pixels': bounded_area,
                      'bounded_fraction': bounded_fraction, 'visible_seed_coverage': seed_coverage}
            if not np.isfinite(score) or score < .65:
                record['reason'] = 'low_rgb_mask_confidence'
            elif rgb_area > maximum_mask_ratio * mask.size:
                record['reason'] = 'oversized_rgb_region'
            elif bounded_area < minimum_mask_points:
                record['reason'] = 'insufficient_bounded_metric_support'
            elif bounded_fraction < minimum_bounded_fraction:
                record['reason'] = 'rgb_mask_extends_outside_measured_object_extent'
            elif seed_coverage < minimum_seed_coverage:
                record['reason'] = 'rgb_mask_does_not_cover_visible_object_seeds'
            else:
                accepted_mask = mask & bounded
                options.append((seed_coverage, bounded_fraction, float(score), -rgb_area, index, accepted_mask))
                record['reason'] = None
            row['candidate_masks'].append(record)
        if not options:
            row['reason'] = 'no_supported_rgb_object_mask'
            rows.append(row)
            continue
        selected = max(options, key=lambda item: item[:5])
        mask = selected[-1]
        masks[name] = mask
        additions.append(points[mask])
        row.update(accepted=True, reason=None, selected_mask_index=selected[4],
                   selected_pixels=int(mask.sum()), selected_points=int(mask.sum()))
        rows.append(row)
        if out is not None:
            _save_mask(out, name, camera, mask, prompts)
    # Preserve every reference sample, then deduplicate only newly predicted
    # object samples. No unlabeled point can enter through nearest-neighbor
    # distance to the bowl/table boundary.
    cloud = reference.copy()
    seen = set(map(tuple, np.floor(reference / .001).astype(np.int64)))
    for additional in additions:
        selected = []
        for index, key in enumerate(map(tuple, np.floor(additional / .001).astype(np.int64))):
            if key not in seen:
                seen.add(key)
                selected.append(index)
        if selected:
            cloud = np.r_[cloud, additional[selected]]
    report = {'version': 'rgb-multiview-target-3.0', 'main_camera': main_name,
              'source': 'cached main RGB MobileSAM mask; depth-visible transferred positives and new RGB MobileSAM masks',
              'main_reference_points': len(reference), 'preserved_main_points': len(reference),
              'new_rgb_mask_points': len(cloud) - len(reference), 'total_points': len(cloud),
              'accepted_added_views': sum(row['accepted'] for row in rows), 'views': rows,
              'measured_reference_min_world': reference.min(axis=0).tolist(),
              'measured_reference_max_world': reference.max(axis=0).tolist(),
              'final_cloud_min_world': cloud.min(axis=0).tolist(), 'final_cloud_max_world': cloud.max(axis=0).tolist(),
              'extent_padding_m': extent_padding, 'depth_agreement_m': depth_tolerance,
              'maximum_rgb_mask_ratio': maximum_mask_ratio, 'minimum_bounded_fraction': minimum_bounded_fraction,
              'minimum_visible_seed_coverage': minimum_seed_coverage,
              'main_mask_sha256': hashlib.sha256(source_mask.tobytes()).hexdigest(),
              'no_blind_nearest_neighbor_cloud_fusion': True, 'no_simulator_inventory': True, 'new_api_calls': 0}
    if out is not None:
        np.save(out / 'target-cloud.npy', cloud)
        np.savez_compressed(out / 'predicted-target-masks.npz', **masks)
        (out / 'quality.json').write_text(json.dumps(report, indent=2) + '\n')
    return cloud, masks, report
