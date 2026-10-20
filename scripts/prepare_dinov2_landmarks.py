"""Actual local DINOv2 + RGB MobileSAM + aligned depth; no task/reference inputs."""

import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from vlm_tamp.dinov2_features import Dinov2Features, file_sha
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_keypoint_candidates import propose_candidates
from vlm_tamp.rgbd_landmark_choices import validate_candidates


def draw_numbered(rgb, candidates):
    scale = 3
    image = Image.fromarray(rgb).resize((rgb.shape[1]*scale, rgb.shape[0]*scale))
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=20)
    occupied = []
    offsets = [(8, -24), (8, 8), (-44, -24), (-44, 8), (12, -48), (12, 30), (-65, -44), (-65, 30)]
    for point in candidates:
        px, py = [(v+.5)*scale for v in point['pixel_xy']]
        label = str(point['id'])
        box = draw.textbbox((0, 0), label, font=font)
        width, height = box[2]+8, box[3]-box[1]+8
        options = []
        for dx, dy in offsets:
            x, y = max(0, min(image.width-width, px+dx)), max(0, min(image.height-height, py+dy))
            rect = [x, y, x+width, y+height]
            overlap = sum(max(0, min(rect[2], other[2])-max(rect[0], other[0]))
                          *max(0, min(rect[3], other[3])-max(rect[1], other[1])) for other in occupied)
            options.append((overlap, rect))
        _, rect = min(options, key=lambda item: item[0])
        occupied.append(rect)
        draw.line((px, py, (rect[0]+rect[2])/2, (rect[1]+rect[3])/2), fill='#ffe26b', width=2)
        draw.rectangle(rect, fill='#fff5cf', outline='black', width=1)
        draw.text((rect[0]+4, rect[1]+3-box[1]), label, font=font, fill='black')
        draw.ellipse([px-4, py-4, px+4, py+4], fill='#ffe26b', outline='black', width=1)
    return image


def predicted_masks(rgb, generator, settings):
    found = generator.generate(rgb)
    found = [item for item in found if settings['minimum_area'] <= item['area']
             <= rgb.shape[0]*rgb.shape[1]*settings['maximum_area_fraction']]
    found.sort(key=lambda item: (-item['area'], -item['predicted_iou'], item['bbox']))
    kept = []
    for item in found:
        mask = item['segmentation']
        duplicate = False
        for other in kept:
            intersection = np.logical_and(mask, other['segmentation']).sum()
            union = np.logical_or(mask, other['segmentation']).sum()
            if intersection/max(union, 1) >= settings['duplicate_iou']:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
        if len(kept) >= settings['maximum_regions']:
            break
    # Retain overlapping/nested RGB masks rather than erasing a potential part.
    return [item['segmentation'] for item in kept], [
        {'id': index+1, 'area_pixels': int(item['area']), 'bbox_xywh': item['bbox'],
         'predicted_iou': item['predicted_iou'], 'stability_score': item['stability_score']}
        for index, item in enumerate(kept)
    ]


def main():
    config = json.loads((ROOT / 'configs/rgbd_dinov2_choices_trial.json').read_text())
    output = ROOT / config['candidate_dataset']
    if output.exists():
        raise FileExistsError('Preserve previously generated landmark candidates')
    output.mkdir(parents=True)
    (output / 'selection_before_proposals.json').write_text(json.dumps(config, indent=2)+'\n')
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'third_party/MobileSAM'))
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    sam_checkpoint = ROOT / 'third_party/MobileSAM/weights/mobile_sam.pt'
    sam = sam_model_registry['vit_t'](checkpoint=str(sam_checkpoint)).eval().to('cuda')
    seg = config['segmentation']
    generator = SamAutomaticMaskGenerator(sam, points_per_side=seg['points_per_side'], points_per_batch=32,
                                         pred_iou_thresh=seg['pred_iou_thresh'],
                                         stability_score_thresh=seg['stability_score_thresh'], crop_n_layers=0,
                                         min_mask_region_area=30)
    encoder = Dinov2Features('cuda')
    captures = {}
    for case in config['cases']:
        identifier = case['example_id']
        if identifier in captures:
            continue
        source = ROOT / config['depth_dataset'] / identifier
        folder = output / identifier
        folder.mkdir()
        rgb = np.asarray(Image.open(source / 'rgb.png').convert('RGB'))
        depth = np.load(source / 'depth_meters.npy', allow_pickle=False)
        cloud = np.load(source / 'world_points.npy', allow_pickle=False)
        masks, mask_metadata = predicted_masks(rgb, generator, seg)
        encoded = encoder.extract(rgb)
        candidates = propose_candidates(encoded['features'], cloud, masks, config['proposal'], depth_m=depth)
        validate_candidates(candidates, rgb.shape[:2])
        np.save(folder / 'predicted_masks.npy', np.asarray(masks, dtype=bool))
        np.save(folder / 'patch_features.npy', encoded['patch_features'])
        (folder / 'candidates.json').write_text(json.dumps(candidates, indent=2)+'\n')
        (folder / 'original.png').write_bytes((source / 'rgb.png').read_bytes())
        draw_numbered(rgb, candidates).save(folder / 'numbered.png')
        metadata = {'example_id': identifier, 'model_inference_executed': True, 'offline_mock': False,
                    'encoder': encoded['metadata'], 'segmentation_source': 'MobileSAM from RGB only',
                    'segmentation_checkpoint_sha256': file_sha(sam_checkpoint),
                    'mask_records': mask_metadata, 'proposal_settings': config['proposal'],
                    'candidate_count': len(candidates), 'numbered_image_shape': [rgb.shape[0]*3, rgb.shape[1]*3],
                    'original_image_sha256': file_sha(source / 'rgb.png'),
                    'depth_sha256': file_sha(source / 'depth_meters.npy'),
                    'world_points_sha256': file_sha(source / 'world_points.npy'),
                    'provided_inventory': False, 'reference_masks_used': False,
                    'mask_zero_excluded': True, 'positive_masks_semantically_verified': False,
                    'cross_region_merge_used': False, 'frame': 'world', 'units': 'meters'}
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        captures[identifier] = metadata
        print(identifier, {'predicted_regions': len(masks), 'candidate_points': len(candidates)}, flush=True)
        del encoded
    (output / 'manifest.json').write_text(json.dumps({'config': config, 'captures': captures}, indent=2)+'\n')
    print({'actual_dinov2_scenes': len(captures), 'api_calls': 0})


if __name__ == '__main__':
    main()
