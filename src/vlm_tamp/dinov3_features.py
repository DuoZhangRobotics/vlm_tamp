"""Frozen official DINOv3 RGB features; no task labels or depth enter this module.

The download uses the user's existing Hugging Face login. It does not request
model access, accept license terms, execute remote code, or use model mirrors.
Torch and Transformers are imported lazily so the main simulator environment
can validate the data contract without installing this optional GPU runtime.
"""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
MODEL_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".runtime" / "hf-cache"


class Dinov3Unavailable(RuntimeError):
    """The official model or its isolated inference runtime is unavailable."""


def validate_rgb(rgb):
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError("DINOv3 needs a nonempty H x W x 3 RGB image")
    if array.dtype != np.uint8:
        raise ValueError("DINOv3 RGB pixels must be uint8 in [0,255]")
    return array


def padded_shape(height, width, patch_size=16):
    """Pad bottom/right to complete patches without resizing image coordinates."""
    values = (height, width, patch_size)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("Image dimensions and patch size must be positive integers")
    return tuple((value + patch_size - 1) // patch_size * patch_size for value in (height, width))


def patch_contract(config, height, width, sequence_length):
    """Validate token layout before excluding class and register tokens."""
    patch_size = config.get("patch_size")
    hidden_size = config.get("hidden_size")
    registers = config.get("num_register_tokens")
    if config.get("model_type") != "dinov3_vit" or patch_size != 16 or hidden_size != 384:
        raise ValueError("Expected the official DINOv3 ViT-S/16 RGB model configuration")
    if config.get("num_channels", 3) != 3 or registers != 4:
        raise ValueError("Unexpected DINOv3 RGB channels or register-token count")
    if padded_shape(height, width, patch_size) != (height, width):
        raise ValueError("Inference dimensions must contain complete DINOv3 patches")
    grid = (height // patch_size, width // patch_size)
    prefix = 1 + registers
    if sequence_length != prefix + grid[0] * grid[1]:
        raise ValueError("DINOv3 output token count does not match the padded pixel grid")
    return {"prefix_tokens": prefix, "grid_shape": grid, "channels": hidden_size}


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(directory):
    directory = Path(directory).resolve()
    config_file, weights_file = directory / "config.json", directory / "model.safetensors"
    if not config_file.is_file() or not weights_file.is_file():
        raise Dinov3Unavailable("An authorized DINOv3 snapshot needs config.json and model.safetensors")
    config = json.loads(config_file.read_text())
    patch_contract(config, 16, 16, 6)
    return {"checkpoint_directory": str(directory), "config": config,
            "config_sha256": file_hash(config_file), "weights_sha256": file_hash(weights_file)}


class Dinov3Features:
    """Load approved official weights and return RGB-aligned dense descriptors."""

    def __init__(self, checkpoint_dir=None, cache_dir=None, device="cuda"):
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise Dinov3Unavailable("Run DINOv3 in its isolated Python 3.11 GPU environment") from exc
        source = "authorized_local_directory"
        if checkpoint_dir is None:
            try:
                from huggingface_hub import snapshot_download
                checkpoint_dir = snapshot_download(
                    repo_id=MODEL_ID, revision=MODEL_REVISION,
                    cache_dir=str(cache_dir or DEFAULT_CACHE), token=True,
                    allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json",
                                    "LICENSE.md", "README.md"],
                )
            except Exception as exc:
                raise Dinov3Unavailable(
                    "The official DINOv3 checkpoint requires an approved Hugging Face login"
                ) from exc
            source = "official_huggingface_snapshot"
        self.metadata = checkpoint_metadata(checkpoint_dir)
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise Dinov3Unavailable("The isolated DINOv3 runtime cannot access a CUDA device")
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(
            str(checkpoint_dir), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa",
        ).eval().to(self.device)
        loaded_config = self.model.config.to_dict()
        patch_contract(loaded_config, 16, 16, 6)
        self.metadata.update(
            model_id=MODEL_ID, requested_revision=MODEL_REVISION, checkpoint_source=source,
            revision_pinned_download=source == "official_huggingface_snapshot",
            device=str(self.device), torch_version=str(torch.__version__),
            dtype="float32", trust_remote_code=False, training=False,
            rgb_only=True, depth_input=False, input_mean=list(IMAGENET_MEAN), input_std=list(IMAGENET_STD),
            implementation_sha256=file_hash(__file__),
        )

    def extract(self, rgb):
        import torch
        from torch.nn import functional as functional

        rgb = validate_rgb(rgb)
        height, width = rgb.shape[:2]
        padded_height, padded_width = padded_shape(height, width)
        # PIL-backed arrays may be read-only; own the buffer before Torch sees it.
        pixels = torch.from_numpy(np.array(rgb, copy=True, order="C")).to(self.device)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        pixels = functional.pad(pixels, (0, padded_width - width, 0, padded_height - height), mode="replicate")
        mean = pixels.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = pixels.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        with torch.inference_mode():
            output = self.model(pixel_values=(pixels - mean) / std)
            tokens = output.last_hidden_state
            contract = patch_contract(self.model.config.to_dict(), padded_height, padded_width, tokens.shape[1])
            if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[2] != contract["channels"]:
                raise ValueError("Unexpected DINOv3 patch-feature dimensions")
            patch_tokens = tokens[:, contract["prefix_tokens"]:]
            grid = patch_tokens.reshape(1, *contract["grid_shape"], contract["channels"]).permute(0, 3, 1, 2)
            dense = functional.interpolate(
                grid, size=(padded_height, padded_width), mode="bilinear", align_corners=False,
            )[:, :, :height, :width]
            if not torch.isfinite(grid).all() or not torch.isfinite(dense).all():
                raise ValueError("DINOv3 produced nonfinite visual descriptors")
            patch_features = grid[0].permute(1, 2, 0).float().cpu().numpy()
            features = dense[0].permute(1, 2, 0).float().cpu().numpy()
        metadata = {**self.metadata, "image_shape_hw": [height, width],
                    "padded_shape_hw": [padded_height, padded_width],
                    "padding_bottom_right_pixels": [padded_height - height, padded_width - width],
                    "grid_shape_hw": list(contract["grid_shape"]), "patch_size": 16,
                    "excluded_class_and_register_tokens": contract["prefix_tokens"],
                    "feature_channels": contract["channels"],
                    "image_coordinate_transform": "identity; bottom/right replicate padding only",
                    "dense_interpolation": "bilinear to padded image, align_corners=False, crop to original RGB",
                    "rgb_sha256": sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()}
        return {"features": features, "patch_features": patch_features, "metadata": metadata}
