"""Actual local DINOv3/MobileSAM drawer proposals without reference labels."""
import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image

from prepare_dinov2_landmarks import draw_numbered, predicted_masks
from vlm_tamp.dinov3_features import DEFAULT_CACHE, MODEL_ID, Dinov3Features, file_hash
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_keypoint_candidates import propose_candidates
from vlm_tamp.rgbd_landmark_choices import validate_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/task_grounding_scope.json')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / config['candidate_dataset']
    if output.exists():
        raise FileExistsError('Preserve previously generated drawer candidates')
    checkpoints = list((DEFAULT_CACHE / ('models--' + MODEL_ID.replace('/', '--')) / 'snapshots').glob('*/model.safetensors'))
    if len(checkpoints) != 1:
        raise ValueError('Expected the existing approved official DINOv3 snapshot')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'third_party/MobileSAM'))
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
    checkpoint = ROOT / 'third_party/MobileSAM/weights/mobile_sam.pt'
    model = sam_model_registry['vit_t'](checkpoint=str(checkpoint)).eval().to('cuda')
    seg = config['segmentation']
    generator = SamAutomaticMaskGenerator(model, points_per_side=seg['points_per_side'], points_per_batch=32,
        pred_iou_thresh=seg['pred_iou_thresh'], stability_score_thresh=seg['stability_score_thresh'],
        crop_n_layers=0, min_mask_region_area=30)
    encoder = Dinov3Features(checkpoint_dir=checkpoints[0].parent)
    manifest = {'model_inference_executed':True,'offline_mock':False,'provided_inventory':False,
                'reference_masks_used':False,'api_calls':0,'config':config,'captures':{}}
    for identifier in dict.fromkeys(case['example_id'] for case in config['cases']):
        source = ROOT / config['dataset'] / identifier
        folder = output / identifier
        folder.mkdir()
        rgb = np.asarray(Image.open(source / 'rgb.png').convert('RGB'))
        depth = np.load(source / 'depth_meters.npy', allow_pickle=False)
        cloud = np.load(source / 'world_points.npy', allow_pickle=False)
        encoded = encoder.extract(rgb)
        masks, mask_records = predicted_masks(rgb, generator, seg)
        candidates = propose_candidates(encoded['features'], cloud, masks, config['proposal'], depth_m=depth)
        validate_candidates(candidates, rgb.shape[:2])
        np.save(folder / 'patch_features.npy', encoded['patch_features'])
        np.save(folder / 'predicted_masks.npy', np.asarray(masks,dtype=bool))
        (folder / 'original.png').write_bytes((source / 'rgb.png').read_bytes())
        draw_numbered(rgb,candidates).save(folder / 'numbered.png')
        (folder / 'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
        metadata = {'example_id':identifier,'encoder':encoded['metadata'],'model_inference_executed':True,
                    'offline_mock':False,'provided_inventory':False,'reference_masks_used':False,
                    'segmentation_source':'MobileSAM from RGB only','segmentation':seg,
                    'segmentation_checkpoint_sha256':file_hash(checkpoint),'mask_records':mask_records,
                    'proposal_settings':config['proposal'],'candidate_count':len(candidates),
                    'original_image_sha256':file_hash(source / 'rgb.png'),
                    'depth_sha256':file_hash(source / 'depth_meters.npy'),
                    'world_points_sha256':file_hash(source / 'world_points.npy'),
                    'mask_zero_excluded':True,'cross_region_merge_used':False,
                    'implementation_sha256':file_hash(__file__)}
        (folder / 'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        manifest['captures'][identifier] = metadata
        print(identifier,{'rgb_regions':len(masks),'candidates':len(candidates)},flush=True)
        del encoded
    (output / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
