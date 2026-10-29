"""RGB mask proposals and calibrated, depth-checked correspondence evidence.

No object inventory, joint model, native segmentation or task-specific pixels.
Labels belong to the caller; mask scores are not semantic correctness scores.
"""
from pathlib import Path
import hashlib
import time

import numpy as np

from .rigid_hinge_tracking import robot_mask
from .rekep_libero.target_region_v3 import visible_projected_points


def usable_pixels(view):
    depth=np.asarray(view['depth']).squeeze();world=np.asarray(view['world_points'])
    return np.isfinite(depth)&(depth>0)&np.isfinite(world).all(2)&~robot_mask(view)


def box_of(mask):
    y,x=np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()),int(y.max())] if len(x) else None


def association(points,view):
    """Only directly depth-supported projected pixels enter the target mask."""
    pixels,indices,report=visible_projected_points(points,view,depth_tolerance=.004)
    mask=np.zeros(np.asarray(view['depth']).squeeze().shape,bool)
    if len(pixels):
        xy=pixels.astype(int);good=usable_pixels(view)[xy[:,1],xy[:,0]]
        xy=xy[good];mask[xy[:,1],xy[:,0]]=True
        report['robot_or_invalid_seed_pixels_removed']=int((~good).sum())
    else:report['robot_or_invalid_seed_pixels_removed']=0
    report.update(supported_pixels=int(mask.sum()),supported_box_xyxy=box_of(mask),
        free_space_conflict_source_points=max(0,report['depth_mismatched_source_points']-report['occluded_source_points']),
        scope='Projection of the reference visible surface only; no hidden surface reconstruction or semantic identity claim.')
    return mask,report


def compare_projected_mask(projected,target_mask):
    shared=int(np.count_nonzero(projected&target_mask));count=int(projected.sum())
    return {'projected_supported_pixels':count,'pixels_inside_comparison_mask':shared,
        'projected_seed_coverage':shared/count if count else None,
        'comparison_mask_pixels':int(target_mask.sum()),
        'interpretation':'insufficient_visible_correspondence' if count<20 else
            'selected_masks_disagree' if shared/count<.5 else 'overlapping_observed_surface',
        'semantic_identity_verified':False,
        'scope':'Containment can also explain overlap; this is correspondence evidence, not proof of identical physical parts.'}


def overlay(path,view,mask,*,points=None):
    from PIL import Image,ImageDraw
    rgb=np.asarray(view['rgb'],np.uint8).copy()
    rgb[mask]=np.rint(.55*rgb[mask]+.45*np.array([30,220,130])).astype(np.uint8)
    image=Image.fromarray(rgb)
    if points is not None:
        draw=ImageDraw.Draw(image)
        for x,y in points:
            if np.isfinite([x,y]).all():draw.ellipse([x-2,y-2,x+2,y+2],outline='yellow',width=1)
    image.save(path)


class MobileSAMBackend:
    """Keep the existing local model and the latest image embedding warm."""
    def __init__(self,device='cuda'):
        self.device=device;self.predictor=None;self.image_key=None;self.load_seconds=0.

    def predict(self,view,key,points,labels,box):
        import torch
        timing={};start=time.perf_counter()
        if self.predictor is None:
            from .rekep_libero.target_region_v3 import load_mobile_sam_predictor
            self.predictor=load_mobile_sam_predictor(self.device)
            if self.device.startswith('cuda'):torch.cuda.synchronize()
            self.load_seconds=time.perf_counter()-start
            timing['model_load']=self.load_seconds
        else:timing['model_load']=0.
        start=time.perf_counter()
        rgb=np.asarray(view['rgb'],np.uint8)
        image_key=(key,rgb.shape,hashlib.sha256(rgb.tobytes()).hexdigest())
        if image_key!=self.image_key:
            self.predictor.set_image(rgb);self.image_key=image_key
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        timing['image_embedding']=time.perf_counter()-start
        start=time.perf_counter()
        masks,scores,_=self.predictor.predict(
            point_coords=np.asarray(points,float) if points else None,
            point_labels=np.asarray(labels,int) if points else None,
            box=np.asarray(box,float) if box is not None else None,multimask_output=True)
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        timing['mask_prediction']=time.perf_counter()-start
        return masks,scores,timing

    def close(self):
        self.predictor=None;self.image_key=None


def checked_folder(base,name):
    base=Path(base)
    if base.is_symlink():raise ValueError('Perception output directory cannot be a symlink')
    path=base/name
    if path.exists() or path.is_symlink():raise ValueError('Preserve existing perception artifacts')
    if not path.resolve().is_relative_to(base.resolve()):raise ValueError('Perception output left its directory')
    path.mkdir(parents=True)
    return path
