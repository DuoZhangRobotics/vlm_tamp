"""Data-contract checks; these tests do not replace real checkpoint inference."""

import json

import numpy as np
import pytest

from vlm_tamp.dinov3_features import (
    Dinov3Unavailable, checkpoint_metadata, file_hash, padded_shape, patch_contract, validate_rgb,
)


CONFIG = {"model_type": "dinov3_vit", "patch_size": 16, "hidden_size": 384,
          "num_register_tokens": 4, "num_channels": 3}


def test_padding_preserves_original_pixel_coordinates_and_handles_rectangular_images():
    assert padded_shape(512, 512) == (512, 512)
    assert padded_shape(479, 641) == (480, 656)
    for bad in (0, -1, True, 3.5):
        with pytest.raises(ValueError):
            padded_shape(bad, 512)


def test_patch_layout_excludes_class_and_register_tokens_and_rejects_misalignment():
    contract = patch_contract(CONFIG, 480, 656, 5 + 30 * 41)
    assert contract == {"prefix_tokens": 5, "grid_shape": (30, 41), "channels": 384}
    for config, height, tokens in (
        ({**CONFIG, "num_register_tokens": 0}, 480, 1235),
        ({**CONFIG, "patch_size": 14}, 480, 1235),
        ({**CONFIG, "hidden_size": 768}, 480, 1235),
        (CONFIG, 479, 1235), (CONFIG, 480, 1234),
    ):
        with pytest.raises(ValueError):
            patch_contract(config, height, 656, tokens)


def test_image_contract_rejects_depth_grayscale_and_float_images():
    assert validate_rgb(np.zeros((20, 30, 3), dtype=np.uint8)).shape == (20, 30, 3)
    for bad in (np.zeros((20, 30)), np.zeros((20, 30, 4), dtype=np.uint8),
                np.zeros((20, 30, 3)), np.zeros((0, 30, 3), dtype=np.uint8)):
        with pytest.raises(ValueError):
            validate_rgb(bad)


def test_checkpoint_metadata_requires_files_and_checks_config(tmp_path):
    with pytest.raises(Dinov3Unavailable):
        checkpoint_metadata(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    # Byte fixture validates integrity bookkeeping, not a model or inference result.
    (tmp_path / "model.safetensors").write_bytes(b"integrity fixture only")
    metadata = checkpoint_metadata(tmp_path)
    assert metadata["weights_sha256"] == file_hash(tmp_path / "model.safetensors")
    assert metadata["config"] == CONFIG
    (tmp_path / "config.json").write_text(json.dumps({**CONFIG, "model_type": "other"}))
    with pytest.raises(ValueError):
        checkpoint_metadata(tmp_path)
