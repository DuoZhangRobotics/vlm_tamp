"""RGB-predicted masks, the native ReKep proposer, and learned Panda grasps.

All inputs are calibrated sensor arrays. Simulator segmentation and object
identities are intentionally absent from the interface.
"""

import ast
import hashlib
import importlib.util
import json
import sys
import time
import types
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from vlm_tamp.libero import ROOT

from .geometry import graspgen_to_tcp, pose_matrix, transform_points


NATIVE_PROPOSER_CONFIG = {
    'device': 'cuda', 'bounds_min': [-.65, -.9, .65], 'bounds_max': [.65, .9, 1.8],
    'min_dist_bt_keypoints': .06, 'num_candidates_per_mask': 5, 'max_mask_ratio': .5, 'seed': 0,
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _temporary_modules(modules):
    previous = {key: sys.modules.get(key) for key in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for key, module in previous.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


def _read_observation(obs_npz, meta):
    if isinstance(meta, (str, Path)):
        meta = json.loads(Path(meta).read_text())
    meta = dict(meta)
    if 'main_camera' not in meta:
        meta['main_camera'] = meta['primary_camera']
    if isinstance(obs_npz, (str, Path)):
        with np.load(obs_npz, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
    else:
        arrays = dict(obs_npz)
    cameras = {}
    for name, calibration in meta['cameras'].items():
        cameras[name] = {
            'rgb': np.asarray(arrays[f'{name}_rgb'], np.uint8),
            'depth': np.asarray(arrays[f'{name}_depth'], float).squeeze(),
            'world_points': np.asarray(arrays[f'{name}_world_points'], float),
            **calibration,
        }
        if cameras[name]['world_points'].shape != (*cameras[name]['depth'].shape, 3):
            raise ValueError('Observation depth and point cloud shapes disagree')
    return cameras, meta


def _native_proposer(config):
    import torch

    source = ROOT / 'third_party/ReKep'
    tree = ast.parse((source / 'utils.py').read_text())
    helper = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                  and item.name == 'filter_points_by_bounds')
    utilities = types.ModuleType('utils')
    utilities.np = np
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source / 'utils.py'), 'exec'), utilities.__dict__)
    if not hasattr(torch.amp, 'autocast'):
        torch.amp.autocast = torch.autocast
    original_load = torch.hub.load
    checkpoint = ROOT / '.runtime/torch-hub/checkpoints/dinov2_vits14_pretrain.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError('The pinned official DINOv2 checkpoint is required')

    def local_load(repository, name, *args, **kwargs):
        if repository != 'facebookresearch/dinov2' or name != 'dinov2_vits14':
            raise ValueError('Only the pinned native DINOv2 architecture is allowed')
        model = original_load(str(ROOT / 'third_party/dinov2'), name, source='local', pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        return model

    torch.hub.load = local_load
    try:
        with _temporary_modules({'utils': utilities}):
            spec = importlib.util.spec_from_file_location('rekep_libero_native_proposer', source / 'keypoint_proposal.py')
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
            proposer = native.KeypointProposer(dict(config))
    finally:
        torch.hub.load = original_load
    proposer.mean_shift.n_jobs = 4
    return proposer


class Perception:
    def __init__(self, device='cuda', proposer_config=None):
        import torch

        self.torch = torch
        self.device = device
        self.config = {**NATIVE_PROPOSER_CONFIG, **(proposer_config or {}), 'device': device}
        self.proposer = _native_proposer(self.config)
        sys.path.insert(0, str(ROOT / 'third_party/MobileSAM'))
        from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

        self.sam = sam_model_registry['vit_t'](
            checkpoint=str(ROOT / 'third_party/MobileSAM/weights/mobile_sam.pt')).eval().to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam, points_per_side=8, points_per_batch=32, pred_iou_thresh=.75,
            stability_score_thresh=.88, crop_n_layers=0, min_mask_region_area=30)
        self.point_predictor = SamPredictor(self.sam)
        self._capture = {}
        old_features = self.proposer._get_features
        old_project = self.proposer._project_keypoints_to_img

        def features(rgb, shape):
            result = old_features(rgb, shape)
            self._capture['features'] = result.float().cpu().numpy().reshape(shape['img_h'], shape['img_w'], -1)
            return result

        def project(rgb, pixels, groups, masks, dense):
            self._capture.update(pixels_yx=pixels.copy(), groups=groups.copy())
            return old_project(rgb, pixels, groups, masks, dense)

        self.proposer._get_features = features
        self.proposer._project_keypoints_to_img = project

    def prepare(self, obs_npz, meta, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cameras, meta = _read_observation(obs_npz, meta)
        name = meta['main_camera']
        camera = cameras[name]
        rgb, points = camera['rgb'], camera['world_points']
        candidates = self.mask_generator.generate(rgb)
        count = np.prod(rgb.shape[:2])
        valid_depth = np.isfinite(points).all(axis=2) & np.isfinite(camera['depth']) & (camera['depth'] > 0)
        candidates = [item for item in candidates if 90 <= item['area'] <= .45 * count]
        candidates.sort(key=lambda item: (-item['area'], -item['predicted_iou'], item['bbox']))
        # Large masks are painted first, so overlaps belong to the smaller RGB region.
        labels = np.zeros(rgb.shape[:2], np.int32)
        masks = []
        for item in candidates[:32]:
            mask = np.asarray(item['segmentation'], bool) & valid_depth
            if mask.sum() < 90:
                continue
            masks.append(mask)
            labels[mask] = len(masks)
        # Native PCA/k-means requires at least six points and nonconstant XYZ.
        for label in np.unique(labels):
            selected = points[labels == label]
            if len(selected) < 12 or not np.isfinite(selected).all() or np.any(np.ptp(selected, axis=0) < 1e-6):
                labels[labels == label] = 0
        if not masks or np.count_nonzero(labels) < 90:
            raise RuntimeError('perception_no_predicted_regions')
        with (out / 'native_proposer.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            keypoints, numbered = self.proposer.get_keypoints(rgb, points, labels)
        if len(keypoints) == 0 or not np.isfinite(keypoints).all():
            raise RuntimeError('perception_no_native_keypoints')
        pixels = self._capture['pixels_yx']
        groups = self._capture['groups']
        dense = self._capture['features']
        features = dense[tuple(pixels.T)]
        Image.fromarray(numbered).save(out / 'numbered.png')
        Image.fromarray(rgb).save(out / 'rgb.png')
        np.savez_compressed(out / 'keypoints.npz', points_world=keypoints, pixels_yx=pixels, groups=groups,
                            features=features, masks=np.asarray(masks, bool), labels=labels)
        record = {
            'main_camera': name, 'numbered_image': str((out / 'numbered.png').resolve()),
            'rgb_image': str((out / 'rgb.png').resolve()), 'keypoint_file': str((out / 'keypoints.npz').resolve()),
            'num_keypoints': len(keypoints), 'config': self.config, 'mask_source': 'RGB-only MobileSAM',
            'native_proposer_source_unchanged': True,
            'native_proposer_sha256': _sha(ROOT / 'third_party/ReKep/keypoint_proposal.py'),
            'mask_checkpoint_sha256': _sha(ROOT / 'third_party/MobileSAM/weights/mobile_sam.pt'),
            'dino_checkpoint_sha256': _sha(ROOT / '.runtime/torch-hub/checkpoints/dinov2_vits14_pretrain.pth'),
            'input_keys': sorted(cameras), 'frame': 'world', 'units': 'meters',
        }
        (out / 'perception.json').write_text(json.dumps(record, indent=2) + '\n')
        self._capture.pop('features', None)
        return {**record, 'points_world': keypoints, 'pixels_yx': pixels, 'groups': groups,
                'features': features, 'masks': np.asarray(masks, bool), 'labels': labels,
                'cameras': cameras, 'meta': meta}

    def selected_cloud(self, bundle, keypoint_id, *, articulated=False, out_dir=None):
        if not 0 <= keypoint_id < len(bundle['points_world']):
            raise ValueError('Invalid grasp keypoint index')
        camera = bundle['cameras'][bundle['main_camera']]
        y, x = bundle['pixels_yx'][keypoint_id]
        candidates = [mask for mask in bundle['masks'] if mask[y, x]]
        # Point-prompted RGB segmentation lets a small handle survive automatic
        # mask proposals; it does not use a simulator part label.
        self.point_predictor.set_image(camera['rgb'])
        point_masks, scores, _ = self.point_predictor.predict(
            point_coords=np.asarray([[x, y]], float), point_labels=np.ones(1, int), multimask_output=True)
        for mask, score in zip(point_masks, scores):
            if score >= .65 and mask[y, x] and 30 <= mask.sum() <= .45 * mask.size:
                candidates.append(mask)
        candidates = [mask for mask in candidates if mask.sum() >= 30]
        if not candidates:
            raise RuntimeError('grasp_no_rgb_region_at_keypoint')
        mask = min(candidates, key=lambda item: int(item.sum())).copy()
        points = camera['world_points']
        valid = np.isfinite(points).all(axis=2) & (camera['depth'] > 0)
        mask &= valid
        if articulated:
            # The handle is an independently moving local part. Whole-cabinet
            # masks cannot become its grasp input; this local crop is observable.
            mask &= np.linalg.norm(points - bundle['points_world'][keypoint_id], axis=2) <= .085
        cloud = points[mask]
        if len(cloud) < 30 or np.linalg.norm(np.ptp(cloud, axis=0)) < .012:
            raise RuntimeError('grasp_region_insufficient_metric_support')
        # Fuse only points projecting into the RGB-selected mask and agreeing
        # with its measured surface, avoiding a 3D radius that adds the cabinet.
        tree = cKDTree(cloud)
        for name, other in bundle['cameras'].items():
            if name == bundle['main_camera']:
                continue
            xyz = other['world_points'].reshape(-1, 3)
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            keep = tree.query(xyz)[0] <= .008
            cloud = np.r_[cloud, xyz[keep]]
        rounded = np.floor(cloud / .002).astype(int)
        cloud = cloud[np.unique(rounded, axis=0, return_index=True)[1]]
        record = {'keypoint_id': int(keypoint_id), 'points': len(cloud), 'mask_pixels': int(mask.sum()),
                  'articulated_local_part': bool(articulated), 'mask_source': 'RGB MobileSAM at VLM keypoint',
                  'no_simulator_identity': True}
        if out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / 'selected_cloud.npy', cloud)
            np.save(out / 'selected_mask.npy', mask)
            (out / 'selected_region.json').write_text(json.dumps(record, indent=2) + '\n')
        return cloud, mask, record

    def track(self, bundle, obs_npz, meta, *, search_radius=.12, minimum_similarity=.85):
        cameras, meta = _read_observation(obs_npz, meta)
        camera = cameras[bundle['main_camera']]
        transformed, _, _, _, shape = self.proposer._preprocess(
            camera['rgb'], camera['world_points'], np.zeros(camera['depth'].shape, np.int32))
        self.proposer._get_features(transformed, shape)
        dense = self._capture.pop('features')
        points = camera['world_points']
        normalized = dense / np.maximum(np.linalg.norm(dense, axis=2, keepdims=True), 1e-8)
        tracked, pixels, diagnostics = [], [], []
        for old, feature in zip(bundle['points_world'], bundle['features']):
            support = np.isfinite(points).all(axis=2) & (camera['depth'] > 0)
            support &= np.linalg.norm(points - old, axis=2) <= search_radius
            candidates = np.argwhere(support)
            if len(candidates) < 3:
                raise RuntimeError('tracking_lost_no_depth_support')
            feature = feature / max(np.linalg.norm(feature), 1e-8)
            similarities = normalized[tuple(candidates.T)] @ feature
            order = np.argsort(similarities)[::-1]
            best = int(order[0])
            y, x = candidates[best]
            similarity = float(similarities[best])
            if not np.isfinite(similarity) or similarity < minimum_similarity:
                raise RuntimeError('tracking_lost_dino_similarity')
            # A separated, equally strong match on another surface is ambiguous.
            competitor = points[tuple(candidates[order[1:]].T)]
            separate = np.linalg.norm(competitor - points[y, x], axis=1) > .025
            ambiguous = separate & (similarities[order[1:]] >= similarity - .01)
            if ambiguous.any():
                raise RuntimeError('tracking_lost_ambiguous_descriptor')
            tracked.append(points[y, x])
            pixels.append([y, x])
            diagnostics.append({'similarity': similarity, 'displacement_m': float(np.linalg.norm(points[y, x] - old))})
        return {**bundle, 'points_world': np.asarray(tracked), 'pixels_yx': np.asarray(pixels),
                'cameras': cameras, 'meta': meta, 'tracking': diagnostics}


class GraspGen:
    """One diffusion/discriminator pass; no analytic/OBB grasp fallback."""

    def __init__(self, config_path=None):
        import torch

        self.torch = torch
        sys.path.insert(0, str(ROOT / 'third_party/GraspGen'))
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

        self.config_path = Path(config_path or ROOT / '.runtime/graspgen-models/checkpoints/graspgen_franka_panda.yml')
        if not self.config_path.is_file():
            raise FileNotFoundError(f'Pinned Panda diffusion/discriminator config required: {self.config_path}')
        self.cfg = load_grasp_cfg(str(self.config_path))
        if self.cfg.data.gripper_name != 'franka_panda' or self.cfg.eval.model_name != 'diffusion-discriminator':
            raise ValueError('Only the Panda diffusion/discriminator checkpoint is supported')
        self.sampler = GraspGenSampler(self.cfg)

    def generate(self, cloud, out_dir, *, seed=0):
        torch = self.torch
        xyz = np.asarray(cloud, np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 30 or not np.isfinite(xyz).all():
            raise ValueError('GraspGen needs at least 30 finite metric XYZ points')
        if type(seed) is not int or not 0<=seed<2**31:
            raise ValueError('Grasp seed must be an integer in [0, 2**31)')
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        # sample returns raw 200 generated poses and discriminator scores. Keeping
        # that output preserves the candidates discarded by top-100 ranking.
        with torch.inference_mode():
            poses, scores, contacts = self.sampler.sample(
                torch.from_numpy(xyz).cuda(), threshold=-1., num_grasps=200, remove_outliers=True)
        poses = poses.detach().cpu().numpy() if hasattr(poses, 'detach') else np.asarray(poses)
        scores = scores.detach().cpu().numpy() if hasattr(scores, 'detach') else np.asarray(scores)
        if len(poses):
            poses[:, 3, 3] = 1
            order = np.argsort(-scores, kind='stable')[:100]
        else:
            poses = np.empty((0, 4, 4), float)
            scores = np.empty(0, float)
            order = np.empty(0, int)
        if contacts is not None and hasattr(contacts, 'detach'):
            contacts = contacts.detach().cpu().numpy()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'grasp_candidates.npz', poses_world_base=poses, scores=scores,
                            retained_indices=order, contacts=np.asarray(contacts) if contacts is not None else np.empty((0, 3)))
        record = {'model': 'GraspGen Panda diffusion + discriminator', 'seed': seed, 'requested_candidates': 200,
                  'generated_candidates': len(poses), 'retained_candidates': len(order), 'tries': 1,
                  'contacts_provided_by_model': contacts is not None,
                  'approach_axis': '+z', 'closing_axis': '+x', 'pose_origin': 'gripper_base_link',
                  'inference_seconds': time.monotonic() - before,
                  'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
                  'generator_checkpoint_sha256': _sha(self.cfg.eval.checkpoint),
                  'discriminator_checkpoint_sha256': _sha(self.cfg.discriminator.checkpoint)}
        (out / 'grasp_generation.json').write_text(json.dumps(record, indent=2) + '\n')
        return {'poses_world_base': poses, 'scores': scores, 'retained_indices': order, 'metadata': record}

    def prepare_grasps(self, perception, bundle, keypoint_id, out_dir, calibration, ik, sdf, *, articulated=False):
        from .planning import select_grasp

        cloud, mask, region = perception.selected_cloud(bundle, keypoint_id, articulated=articulated, out_dir=out_dir)
        generated = self.generate(cloud, out_dir)
        selected = select_grasp(generated, bundle['points_world'][keypoint_id], cloud, calibration, ik, sdf)
        selected['selected_region'] = region
        Path(out_dir, 'grasp_selection.json').write_text(json.dumps(selected, indent=2) + '\n')
        self.overlay(bundle, generated, calibration, Path(out_dir) / 'grasp_overlay.png', selected.get('selected_index'))
        return selected

    @staticmethod
    def overlay(bundle, generated, calibration, path, selected_index=None):
        camera = bundle['cameras'][bundle['main_camera']]
        image = Image.fromarray(camera['rgb']).copy()
        draw = ImageDraw.Draw(image)
        inverse = np.linalg.inv(pose_matrix(camera['camera_to_world']))
        k = np.asarray(camera['intrinsic'])
        for idx in generated['retained_indices']:
            pose = generated['poses_world_base'][idx]
            base = pose[:3, 3]
            tcp = graspgen_to_tcp(pose, calibration)[:3]
            points = transform_points([base, tcp], inverse)
            if (points[:, 2] <= 0).any():
                continue
            uv = points @ k.T
            uv = uv[:, :2] / uv[:, 2, None]
            color = 'lime' if int(idx) == selected_index else 'yellow'
            draw.line([tuple(uv[0]), tuple(uv[1])], fill=color, width=3 if int(idx) == selected_index else 1)
        image.save(path)
