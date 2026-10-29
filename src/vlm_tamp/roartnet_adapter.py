"""Offline, metric adapter for the official RPMArt/RoArtNet implementation.

No simulator access, object labels or native hinge parameters are used here.
The estimator receives an already selected object cloud, a configured category
checkpoint and public camera calibration. It does not infer the joint type.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
import time

import numpy as np


# The exact optical -> SAPIEN camera rotation in RPMArt real_service.py.
OPTICAL_TO_MODEL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def validate_cloud(points, *, author_raw_fixture=False):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
        raise ValueError("Expected at least 100 object XYZ points in camera metres")
    if not np.isfinite(points).all():
        raise ValueError("Object XYZ must be finite; filter invalid depth before inference")
    # Preserve published author fixtures exactly as their own test loader does.
    # They contain many z=0 points, not just invalid all-zero depth sentinels.
    if not author_raw_fixture and np.any(points[:, 2] <= 0):
        raise ValueError("Optical-camera points must have positive forward depth")
    if np.max(np.ptp(points, axis=0)) < .02:
        raise ValueError("Object cloud has insufficient metric extent")
    return np.ascontiguousarray(points, dtype=np.float32)


def model_to_optical(points):
    return np.asarray(points) @ OPTICAL_TO_MODEL


def optical_to_world(points, camera_to_world, *, direction=False):
    transform = np.asarray(camera_to_world, float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera_to_world must be a finite 4x4 rigid transform")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)
            or not np.allclose(transform[3], [0., 0., 0., 1.])):
        raise ValueError("camera_to_world must preserve metric distances and handedness")
    result = np.asarray(points) @ rotation.T
    return result if direction else result + transform[:3, 3]


class RoArtNetEstimator:
    """Load unmodified upstream models, SHOT and inference; profile public calls.

    Global upstream functions are temporarily wrapped for profiling, so one
    instance must not be called concurrently in the same process.
    """
    def __init__(self, repository, checkpoint, *, profile="author-eval", device=0):
        started = time.perf_counter()
        self.repository, self.checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
        sys.path.insert(0, str(self.repository))
        import torch
        import cupy
        from omegaconf import OmegaConf
        import inference
        from models.roartnet import create_encoder, create_shot_encoder

        self.torch, self.cupy, self.upstream = torch, cupy, inference
        self.device, self.profile = device, profile
        torch.cuda.set_device(device)
        self.synchronize()
        cfg = OmegaConf.load(self.checkpoint / ".hydra/config.yaml")
        if profile == "author-eval":
            # Merge just the published inference voting settings, not its paths
            # or the deliberately missing dataset interpolation.
            override = OmegaConf.load(self.repository / "configs/eval_config.yaml")
            cfg.algorithm.voting = OmegaConf.merge(cfg.algorithm.voting, override.algorithm.voting)
        elif profile != "checkpoint":
            raise ValueError(f"Unknown inference profile: {profile}")
        if cfg.dataset.normalize != "none" or cfg.dataset.rgb or cfg.dataset.joint_num != 1:
            raise ValueError("This adapter currently supports the XYZ-only, one-joint, metric microwave checkpoint")
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.import_seconds = time.perf_counter() - started
        started = time.perf_counter()
        shot = cfg.algorithm.shot_encoder
        encoder = cfg.algorithm.encoder
        self.shot_encoder = create_shot_encoder(list(shot.hidden_dims), shot.feature_dim,
                                                shot.bn, shot.ln, shot.dropout)
        self.encoder = create_encoder(cfg.algorithm.sampling.tuple_more_num, shot.feature_dim,
                                      cfg.dataset.rgb, list(encoder.hidden_dims),
                                      cfg.algorithm.voting.rot_bin_num, cfg.dataset.joint_num,
                                      encoder.bn, encoder.ln, encoder.dropout)
        for model, name in [(self.shot_encoder, "shot_encoder_latest.pth"),
                            (self.encoder, "encoder_latest.pth")]:
            weights = torch.load(self.checkpoint / "weights" / name, map_location="cpu", weights_only=True)
            model.load_state_dict(weights, strict=True)
            model.to(device=f"cuda:{device}").eval()
        self.synchronize()
        self.model_load_seconds = time.perf_counter() - started
        self.kwargs = {k: self.config["dataset"][k] for k in
                       ["denoise", "normalize", "resolution", "receptive_field", "sample_points_num", "joint_num"]}
        self.kwargs.update(self.config["algorithm"]["sampling"])
        self.kwargs.update(self.config["algorithm"]["voting"])
        self.kwargs["device"] = device

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)
        with self.cupy.cuda.Device(self.device):
            self.cupy.cuda.get_current_stream().synchronize()

    @contextmanager
    def profiling(self):
        times, diagnostics, restore, handles = {}, {}, [], []
        translation_calls = 0

        def wrap(owner, name, stage, inspect=None):
            original = getattr(owner, name)

            def call(*args, **kwargs):
                self.synchronize()
                started = time.perf_counter()
                result = original(*args, **kwargs)
                self.synchronize()
                label = stage() if callable(stage) else stage
                times[label] = times.get(label, 0.) + time.perf_counter() - started
                if inspect:
                    inspect(result)
                return result
            setattr(owner, name, call)
            restore.append((owner, name, original))

        def inspect_quantized(result):
            diagnostics["quantized_points"] = len(result[1])

        def inspect_shot(result):
            array = np.asarray(result).reshape(-1, 352)
            diagnostics["shot_nonfinite_rows"] = int((~np.isfinite(array).all(1)).sum())
            diagnostics["shot_zero_rows_before_sanitization"] = int((array == 0).all(1).sum())

        def translation_label():
            nonlocal translation_calls
            translation_calls += 1
            return "joint_origin_voting_seconds" if translation_calls == 1 else "affordance_voting_seconds"

        def inspect_translation(result):
            votes = np.asarray(result[1])
            name = "joint_origin_votes" if translation_calls == 1 else "affordance_votes"
            diagnostics[name] = {"grid_shape": list(votes.shape), "sum": float(votes.sum()),
                                 "peak": float(votes.max()), "not_calibrated_probability": True}
            if not np.isfinite(votes).all() or not votes.max() > 0:
                raise ValueError(f"No finite positive support in {name}")

        def inspect_rotation(result):
            votes = np.asarray(result[2])
            diagnostics["rotation_votes"] = {"directions": len(votes), "sum": float(votes.sum()),
                                              "peak": float(votes.max()), "not_calibrated_probability": True}
            if not np.isfinite(votes).all() or not votes.max() > 0:
                raise ValueError("No finite positive support in rotation votes")

        def time_model(model, name):
            state = {}

            def before(*_):
                self.synchronize()
                state["start"] = time.perf_counter()

            def after(*_):
                self.synchronize()
                times[name] = time.perf_counter() - state["start"]
            handles.extend([model.register_forward_pre_hook(before), model.register_forward_hook(after)])

        try:
            wrap(self.upstream.ME.utils, "sparse_quantize", "quantization_seconds", inspect_quantized)
            wrap(self.upstream.shot, "estimate_normal", "normal_estimation_seconds")
            wrap(self.upstream.shot, "compute", "shot_descriptor_seconds", inspect_shot)
            wrap(self.upstream, "farthest_point_sample", "point_sampling_seconds")
            wrap(self.upstream, "voting_translation", translation_label, inspect_translation)
            wrap(self.upstream, "voting_rotation", "axis_direction_voting_seconds", inspect_rotation)
            time_model(self.shot_encoder, "shot_encoder_seconds")
            time_model(self.encoder, "tuple_encoder_seconds")
            yield times, diagnostics
        finally:
            for owner, name, original in reversed(restore):
                setattr(owner, name, original)
            for handle in handles:
                handle.remove()

    def predict(self, optical_points, *, camera_to_world=None, seed=42, author_raw_fixture=False):
        started = time.perf_counter()
        points = validate_cloud(optical_points, author_raw_fixture=author_raw_fixture)
        pc = np.ascontiguousarray(points @ OPTICAL_TO_MODEL.T, dtype=np.float32)
        self.upstream.setup_seed(seed)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        with self.profiling() as (stages, diagnostics):
            origins, directions, affordances = self.upstream.inference_fn(
                pc, None, self.shot_encoder, self.encoder, **self.kwargs)
        self.synchronize()
        origins, directions, affordances = [model_to_optical(x) for x in (origins, directions, affordances)]
        if not all(np.isfinite(x).all() for x in [origins, directions, affordances]):
            raise ValueError("RoArtNet produced a nonfinite geometry estimate")
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        output = {"joint_origin_optical_m": origins[0].tolist(), "joint_axis_optical": directions[0].tolist(),
                  "affordance_optical_m": affordances[0].tolist(), "joint_type": "revolute",
                  "joint_type_source": "configured microwave category; not predicted",
                  "verified_by_motion": False, "seed": seed, "profile": self.profile,
                  "input_points": len(points), "input_nonpositive_z_rows": int((points[:, 2] <= 0).sum()),
                  "diagnostics": diagnostics,
                  "torch_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                  "cupy_pool_retained_bytes": self.cupy.get_default_memory_pool().total_bytes()}
        if camera_to_world is not None:
            output.update(joint_origin_world_m=optical_to_world(origins[0], camera_to_world).tolist(),
                          joint_axis_world=optical_to_world(directions[0], camera_to_world, direction=True).tolist(),
                          affordance_world_m=optical_to_world(affordances[0], camera_to_world).tolist())
        total = time.perf_counter() - started
        output["timing"] = {**stages, "other_inference_seconds": max(0., total - sum(stages.values())),
                            "total_inference_seconds": total,
                            "timing_note": "Instrumented wall time with GPU synchronization at stage boundaries"}
        return output
