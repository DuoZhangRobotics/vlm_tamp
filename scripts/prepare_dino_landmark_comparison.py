"""Matched candidates from actual cached DINOv2/v3 features and shared RGB masks."""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from prepare_dinov2_landmarks import draw_numbered, predicted_masks
from vlm_tamp.dinov2_features import file_sha
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_keypoint_candidates import propose_candidates
from vlm_tamp.rgbd_landmark_choices import validate_candidates


def main():
    config = json.loads((ROOT / 'configs/rgbd_dino_comparison.json').read_text())
    output = ROOT / config['candidate_root']
    if output.exists():
        raise FileExistsError('Preserve an existing comparison candidate bundle')
    output.mkdir(parents=True)
    (output / 'selection_before_proposals.json').write_text(json.dumps(config, indent=2)+'\n')
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'third_party/MobileSAM'))
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    sam_checkpoint = ROOT / 'third_party/MobileSAM/weights/mobile_sam.pt'
    sam = sam_model_registry['vit_t'](checkpoint=str(sam_checkpoint)).eval().to('cuda')
    settings = config['segmentation']
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=settings['points_per_side'], points_per_batch=32,
        pred_iou_thresh=settings['pred_iou_thresh'], stability_score_thresh=settings['stability_score_thresh'],
        crop_n_layers=0, min_mask_region_area=30,
    )
    manifest = {'config': config, 'feature_inference_reused': True, 'model_mock': False,
                'api_calls': 0, 'reference_inputs': False, 'captures': {}}
    identifiers = list(dict.fromkeys(case['example_id'] for case in config['cases']))
    for identifier in identifiers:
        capture = ROOT / config['depth_dataset'] / identifier
        rgb = np.asarray(Image.open(capture / 'rgb.png').convert('RGB'))
        depth = np.load(capture / 'depth_meters.npy', allow_pickle=False)
        cloud = np.load(capture / 'world_points.npy', allow_pickle=False)
        masks, mask_metadata = predicted_masks(rgb, generator, settings)
        mask_folder = output / 'shared_masks' / identifier
        mask_folder.mkdir(parents=True)
        np.save(mask_folder / 'predicted_masks.npy', np.asarray(masks, dtype=bool))
        mask_sha = file_sha(mask_folder / 'predicted_masks.npy')
        for condition in config['conditions']:
            folder = ROOT / config['candidate_datasets'][condition] / identifier
            folder.mkdir(parents=True)
            source = ROOT / config['feature_sources'][condition] / identifier
            metadata = json.loads((source / 'metadata.json').read_text())
            if not metadata['model_inference_executed']:
                raise ValueError('Actual pretrained feature inference is required')
            patch = np.load(source / 'patch_features.npy', allow_pickle=False)
            if condition == 'dinov2':
                encoder = metadata['encoder']
                padded, original = encoder['padded_shape'], encoder['original_shape']
            else:
                encoder = metadata
                padded, original = encoder['padded_shape_hw'], encoder['image_shape_hw']
            if tuple(original) != rgb.shape[:2] or not np.isfinite(patch).all():
                raise ValueError('Cached features must align to original sensor dimensions')
            with torch.inference_mode():
                tensor = torch.from_numpy(patch.copy()).permute(2, 0, 1)[None].float().to('cuda')
                dense = functional.interpolate(tensor, size=tuple(padded), mode='bilinear', align_corners=False)
                dense = dense[0, :, :original[0], :original[1]].permute(1, 2, 0).cpu().numpy()
            candidates = propose_candidates(dense, cloud, masks, config['proposal'], depth_m=depth)
            validate_candidates(candidates, rgb.shape[:2])
            (folder / 'candidates.json').write_text(json.dumps(candidates, indent=2)+'\n')
            (folder / 'original.png').write_bytes((capture / 'rgb.png').read_bytes())
            draw_numbered(rgb, candidates).save(folder / 'numbered.png')
            np.save(folder / 'predicted_masks.npy', np.asarray(masks, dtype=bool))
            np.save(folder / 'patch_features.npy', patch)
            info = {'condition': condition, 'example_id': identifier, 'encoder': encoder,
                    'model_inference_executed': True, 'cached_real_features_used': True,
                    'feature_source_path': str(source), 'feature_source_metadata_sha256': file_sha(source / 'metadata.json'),
                    'source_patch_sha256': file_sha(source / 'patch_features.npy'), 'offline_mock': False,
                    'candidate_count': len(candidates), 'shared_predicted_masks_sha256': mask_sha,
                    'segmentation': settings, 'mask_records': mask_metadata,
                    'segmentation_checkpoint_sha256': file_sha(sam_checkpoint), 'proposal_settings': config['proposal'],
                    'source_image_sha256': file_sha(capture / 'rgb.png'),
                    'depth_sha256': file_sha(capture / 'depth_meters.npy'),
                    'world_points_sha256': file_sha(capture / 'world_points.npy'),
                    'provided_inventory': False, 'reference_masks_used': False, 'mask_zero_excluded': True,
                    'cross_region_merge_used': False, 'positive_masks_semantically_verified': False}
            (folder / 'metadata.json').write_text(json.dumps(info, indent=2)+'\n')
            manifest['captures'][condition+'/'+identifier] = info
            print(condition, identifier, {'shared_regions': len(masks), 'candidate_points': len(candidates)}, flush=True)
            del dense
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print({'actual_feature_model_scene_pairs': len(manifest['captures']), 'api_calls': 0})


if __name__ == '__main__':
    main()
