"""Version 2 unknown-space admission without changing frozen version 1 sources.

Trilinear interpolation can be negative inside a voxel classified as unknown
when its neighboring voxels are free. This subclass retains the original SDF
and known-robot prior, but explicitly blocks every unknown-state query. The
subclass can be installed before importing the original runner by assigning
``geometry.ObservedSDF = StrictObservedSDF`` in a versioned entry point.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import ObservedSDF as _FrozenObservedSDF
from .geometry import graspgen_to_tcp, pose_matrix, transform_points


class StrictObservedSDF(_FrozenObservedSDF):
    """Unknown points stay positive independently of interpolated clearance."""

    def _unknown_penalty(self):
        return max(float(np.max(self.spacing)), float(np.finfo(np.float32).eps))

    def query(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        distances = super().query(points)
        unknown = self.states(points)['unknown']
        distances[unknown] = np.maximum(distances[unknown], self._unknown_penalty())
        return distances

    def collision_values(self, points, margin=.005, finger_flags=None, target_points=None, contact_radius=.012):
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('Collision clearance margin must be finite and nonnegative')
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        values, allowed = super().collision_values(points, margin, finger_flags, target_points, contact_radius)
        unknown = self.states(points)['unknown']
        values[unknown] = np.maximum(values[unknown], self._unknown_penalty() + margin)
        allowed[unknown] = False
        return values, allowed


def load_frozen_sdf(path, sdf_type=StrictObservedSDF):
    with np.load(path, allow_pickle=False) as cache:
        arguments = {key: cache[key] for key in
                     ('bounds_min', 'bounds_max', 'occupied', 'free', 'sdf_voxels', 'spacing')}
        if 'known_robot_prior' in cache:
            arguments['known_robot_prior'] = cache['known_robot_prior']
    return sdf_type(**arguments)


def audit_frozen_bowl_grasps(experiment_root, output_path=None):
    """Replay frozen tool checks and prove tightening changes no grasp outcomes.

    IK, membership and arm rejection diagnostics remain the recorded results;
    this audit does not claim to rerun those components. Unknown-space blocking
    only increases collision costs, so a previously rejected candidate cannot
    become accepted. No simulator, learned-model inference or API is invoked.
    """
    root = Path(experiment_root)
    folders = sorted(root.glob('libero_goal-08-state*-seed0'))
    if not folders:
        raise FileNotFoundError('No frozen bowl evaluation episodes found')
    rows, admitted_samples, added_blocks, tool_changes, original_accepted = [], 0, 0, 0, 0
    candidates = 0
    for folder in folders:
        selected = json.loads((folder / 'stage1-grasps/grasp_selection.json').read_text())
        original = load_frozen_sdf(folder / 'stage1-grasp-sdf.npz', _FrozenObservedSDF)
        strict = load_frozen_sdf(folder / 'stage1-grasp-sdf.npz')
        calibration = json.loads((folder / 'open-gripper/metadata.json').read_text())['robot_calibration']
        tool = np.asarray(calibration['tool_collision_points_tcp'], float)
        flags = np.asarray(calibration['tool_collision_finger_flags'], bool)
        target = np.load(folder / 'stage1-grasps/selected_cloud.npy', allow_pickle=False)
        with np.load(folder / 'stage1-grasps/grasp_candidates.npz', allow_pickle=False) as cache:
            poses = cache['poses_world_base']
            retained = cache['retained_indices']
        original_accepted += len(selected['feasible_candidates'])
        if selected['feasible_candidates']:
            raise ValueError('This rejection-preservation audit requires all frozen grasps to have been rejected')
        by_index = {item['index']: item for item in selected['rejected_candidates']}
        if len(by_index) != len(retained) or set(by_index) != set(map(int, retained)):
            raise ValueError('Every retained grasp must have an explicit frozen rejection')
        candidates += len(retained)
        case_aliases, case_blocks, case_tool_changes, affected = 0, 0, 0, []
        for index in retained:
            index = int(index)
            recorded = by_index[index]
            if 'collision' not in recorded:
                continue  # Earlier membership / IK gate rejected this candidate.
            tcp = graspgen_to_tcp(poses[index], calibration)
            world = transform_points(tool, pose_matrix(tcp))
            old_values, old_allowed = original.collision_values(world, .005, flags, target)
            new_values, new_allowed = strict.collision_values(world, .005, flags, target)
            unknown = original.states(world)['unknown']
            aliases = unknown & (old_values <= 0) & ~old_allowed
            additional = (old_values <= 0) & (new_values > 0)
            if np.any(new_values + 1e-12 < old_values):
                raise AssertionError('The strict gate must never reduce a collision violation')
            if new_allowed[unknown].any() or not (new_values[unknown] > 0).all():
                raise AssertionError('An unknown sample passed strict collision admission')
            old_accepted, new_accepted = bool((old_values <= 0).all()), bool((new_values <= 0).all())
            if old_accepted != recorded['collision']['accepted']:
                raise AssertionError('Frozen tool collision replay does not match the recorded check')
            changed = old_accepted != new_accepted
            case_aliases += int(aliases.sum())
            case_blocks += int(additional.sum())
            case_tool_changes += int(changed)
            if aliases.any():
                affected.append({'index': index, 'alias_samples': int(aliases.sum()),
                                 'frozen_reason': recorded['reason'],
                                 'old_tool_accepted': old_accepted, 'strict_tool_accepted': new_accepted,
                                 'old_blocked_samples': int((old_values > 0).sum()),
                                 'strict_blocked_samples': int((new_values > 0).sum()),
                                 'most_negative_unknown_collision_value_m': float(old_values[aliases].min())})
        rows.append({'episode': folder.name, 'retained_candidates': len(retained), 'frozen_accepted': 0,
                     'strict_accepted': 0, 'tool_admission_changes': case_tool_changes,
                     'unknown_samples_previously_admitted': case_aliases,
                     'additional_blocked_tool_samples': case_blocks, 'affected_candidates': affected})
        admitted_samples += case_aliases
        added_blocks += case_blocks
        tool_changes += case_tool_changes
    result = {
        'version': 'strict-unknown-audit-2.0', 'source_experiment': str(root.resolve()),
        'episodes': len(rows), 'retained_candidates': candidates,
        'frozen_accepted': original_accepted, 'strict_accepted': 0, 'candidate_outcome_changes': 0,
        'tool_admission_changes': tool_changes, 'unknown_samples_previously_admitted': admitted_samples,
        'additional_blocked_tool_samples': added_blocks,
        'new_api_calls': 0, 'new_model_inference_calls': 0, 'new_simulator_calls': 0,
        'scope': 'Replayed frozen tool checks; retained earlier IK, membership and arm rejections. '
                 'Strict unknown admission is monotonic, so no rejected grasp becomes accepted.',
        'strict_module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows': rows,
    }
    if output_path is not None:
        path = Path(output_path)
        if path.exists():
            raise FileExistsError('Preserve an existing strict audit')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + '\n')
    return result
