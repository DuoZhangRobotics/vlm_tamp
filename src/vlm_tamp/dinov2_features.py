"""Cached DINOv2 ViT-S/14 descriptors aligned to original RGB sensor pixels."""

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from .libero import ROOT


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


class Dinov2Features:
    """Load exact local official architecture and weights; no download fallback."""

    def __init__(self, device='cuda'):
        import torch

        self.torch, self.device = torch, torch.device(device)
        source = ROOT / 'third_party/dinov2'
        checkpoint = ROOT / '.runtime/torch-hub/checkpoints/dinov2_vits14_pretrain.pth'
        if not checkpoint.is_file() or not (source / 'hubconf.py').is_file():
            raise FileNotFoundError('Cached official DINOv2 source/checkpoint required')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('DINOv2 GPU runtime unavailable')
        # Construct only the architecture, then load the pinned local checkpoint.
        self.model = torch.hub.load(str(source), 'dinov2_vits14', source='local', pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model.eval().to(self.device)
        self.patch_size = 14
        self.metadata = {
            'model': 'dinov2_vits14', 'checkpoint_path': str(checkpoint),
            'checkpoint_sha256': file_sha(checkpoint),
            'repository_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            'patch_size': 14, 'channels': 384, 'device': str(self.device),
            'preprocessing': 'ImageNet mean/std; bottom/right edge padding to complete patches; no RGB resize',
            'model_input': 'RGB only', 'depth_enters_encoder': False,
            'model_inference_executed': True, 'weights_downloaded_in_this_run': False,
        }

    def extract(self, rgb):
        import torch.nn.functional as functional

        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('DINOv2 requires HxWx3 uint8 original RGB')
        height, width = rgb.shape[:2]
        if not height or not width:
            raise ValueError('Empty RGB')
        ph, pw = [(value+13)//14*14 for value in (height, width)]
        padded = np.pad(rgb, ((0, ph-height), (0, pw-width), (0, 0)), mode='edge')
        torch = self.torch
        image = torch.from_numpy(padded.copy()).permute(2, 0, 1)[None].to(self.device).float()/255
        mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        with torch.inference_mode():
            tokens = self.model.forward_features((image-mean)/std)['x_norm_patchtokens']
            if tuple(tokens.shape) != (1, (ph//14)*(pw//14), 384):
                raise ValueError('Unexpected DINOv2 patch token layout')
            patch = tokens.reshape(1, ph//14, pw//14, 384).permute(0, 3, 1, 2)
            dense = functional.interpolate(patch.float(), size=(ph, pw), mode='bilinear', align_corners=False)
            dense = dense[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
            patch_grid = patch[0].permute(1, 2, 0).float().cpu().numpy()
        if not np.isfinite(dense).all():
            raise ValueError('Nonfinite DINOv2 descriptors')
        return {'features': dense, 'patch_features': patch_grid,
                'metadata': {**self.metadata, 'original_shape': [height, width],
                             'padded_shape': [ph, pw], 'patch_grid_shape': [ph//14, pw//14]}}
