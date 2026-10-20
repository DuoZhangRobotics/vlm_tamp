"""Versioned grasp admission for cached GraspGen Panda candidates.

Raw detector poses are preserved alongside the Panda checkpoint's declared
antipodal half-turn and an explicitly measured embodiment alignment, when
configured. Scene collision comes from depth with strict unknown admission.
"""

from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial import cKDTree

from .geometry import graspgen_to_tcp, pose_matrix, transform_points
from .strict_geometry import StrictObservedSDF


COLLISION_MARGIN_M = .005


class AnalyticRobotSDF(StrictObservedSDF):
    """Recognize exact current robot hull boundaries without voxel inflation.

    Surface samples may round to a neighboring unknown voxel even when their
    exact coordinates lie on the known robot mesh. This analytic prior clears
    only configured current hull interiors and boundaries (0.1 micrometer
    numeric tolerance); it never clears nearby unobserved arm sweeps.
    """

    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self, '_current_robot_hulls'):
            self._current_robot_hulls = []
        if not hasattr(self, '_current_visual_material'):
            self._current_visual_material = None

    def exclude_known_robot_volumes(self, volumes):
        volumes = list(volumes)
        removed = super().exclude_known_robot_volumes(volumes)
        self._current_robot_hulls = volumes
        return removed

    def exact_robot_prior(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        inside = np.zeros(len(points), bool)
        valid = np.isfinite(points).all(axis=1)
        for volume in self._current_robot_hulls:
            planes = np.asarray(volume['planes_world'], float)
            vertices = np.asarray(volume['vertices_world'], float)
            near = (valid & ~inside & (points >= vertices.min(axis=0) - 1e-7).all(axis=1)
                    & (points <= vertices.max(axis=0) + 1e-7).all(axis=1))
            indices = np.flatnonzero(near)
            for offset in range(0, len(indices), 20000):
                batch = indices[offset:offset + 20000]
                inside[batch] = np.all(points[batch] @ planes[:, :3].T + planes[:, 3] <= 1e-7, axis=1)
        if self._current_visual_material is not None:
            candidates = valid & ~inside
            inside[candidates] = self._current_visual_material.contains(points[candidates])
        return inside

    def exclude_current_visual_material(self, material_prior):
        """Clear only actual current watertight robot material, preserving holes."""
        from .current_material_v3 import CurrentRobotMaterialPrior
        if not isinstance(material_prior, CurrentRobotMaterialPrior):
            raise ValueError('Robot visual material must use the exact current watertight mesh prior')
        self._current_visual_material = material_prior
        # Do not carve visual material into coarse voxel labels: a point 1mm
        # outside the mesh could round to a carved interior voxel and be wrongly
        # admitted. Exact states/query/collision_values apply this point prior;
        # all original outside-space evidence remains unchanged.
        return 0

    def states(self, points):
        states = super().states(points)
        known = self.exact_robot_prior(points)
        states['occupied'][known] = False
        states['unknown'][known] = False
        states['free'][known] = True
        return states

    def query(self, points):
        values = super().query(points)
        states = self.states(points)
        blocked = states['occupied'] | states['unknown']
        # Mixed-cell interpolation can hide an occupied sample among free
        # neighbors just as it can hide unknown cells. Both stay blocked.
        values[blocked] = np.maximum(values[blocked], self._unknown_penalty())
        # Current robot geometry is absent from the environment collision field;
        # this clearance says nothing about points outside the exact hull.
        known = self.exact_robot_prior(points)
        values[known] = -max(float(np.max(self.spacing)), COLLISION_MARGIN_M)
        return values

    def collision_values(self, points, margin=COLLISION_MARGIN_M, finger_flags=None,
                         target_points=None, contact_radius=.012):
        values, allowed = super().collision_values(points, margin, finger_flags, target_points, contact_radius)
        states = self.states(points)
        blocked = (states['occupied'] | states['unknown']) & ~allowed
        values[blocked] = np.maximum(values[blocked], self._unknown_penalty() + margin)
        known = self.exact_robot_prior(points)
        values[known] = -max(float(margin), float(np.finfo(np.float32).eps))
        allowed[known] = False  # This is robot prior, not an object-contact exception.
        return values, allowed

    def save(self, path):
        material = None if self._current_visual_material is None else self._current_visual_material.artifact
        np.savez_compressed(path, bounds_min=self.bounds_min, bounds_max=self.bounds_max,
                            occupied=self.occupied, free=self.free, sdf_voxels=self.sdf_voxels,
                            spacing=self.spacing, known_robot_prior=self.known_robot_prior,
                            analytic_robot_prior_json=np.asarray(json.dumps(self._current_robot_hulls)),
                            robot_material_artifact_json=np.asarray(json.dumps(material)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as cache:
            names = ['bounds_min', 'bounds_max', 'occupied', 'free', 'sdf_voxels', 'spacing', 'known_robot_prior']
            result = cls(**{name: cache[name].copy() for name in names})
            if 'analytic_robot_prior_json' in cache:
                result._current_robot_hulls = json.loads(str(cache['analytic_robot_prior_json'].item()))
            if 'robot_material_artifact_json' in cache:
                artifact = json.loads(str(cache['robot_material_artifact_json'].item()))
                if artifact is not None:
                    from .current_material_v3 import CurrentRobotMaterialPrior
                    result._current_visual_material = CurrentRobotMaterialPrior.from_artifact(artifact)
        return result


def _points(values, label):
    points = np.asarray(values, float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError(f'{label} must contain finite nonempty Nx3 points')
    return points


def contact_face_flags(calibration):
    """Only configured inward pad faces can touch the selected RGB region.

    Older saved observations label entire fingers. For those caches the fallback
    uses measured Panda inner-pad dimensions in the controller TCP frame. It is
    intentionally narrow and never promotes palm or outer-finger samples.
    """
    points = _points(calibration['tool_collision_points_tcp'], 'Tool geometry')
    fingers = np.asarray(calibration['tool_collision_finger_flags'], bool)
    if fingers.shape != (len(points),):
        raise ValueError('Tool finger labels must match configured collision samples')
    explicit = calibration.get('tool_collision_contact_face_flags', calibration.get('tool_contact_face_flags'))
    if explicit is not None:
        faces = np.asarray(explicit, bool)
        if faces.shape != fingers.shape or np.any(faces & ~fingers):
            raise ValueError('Contact faces must be a subset of configured finger geometry')
        return faces, 'configured inward Panda pad faces'
    faces = np.zeros(len(points), bool)
    for side in (-1, 1):
        group = fingers & (side * points[:, 0] > 0)
        if not group.any():
            continue
        inner = float(np.min(side * points[group, 0]))
        faces |= (group & (np.abs(side * points[:, 0] - inner) <= .001)
                  & (np.abs(points[:, 1]) <= .0085)
                  & (points[:, 2] >= -.012) & (points[:, 2] <= .007))
    return faces, 'legacy cache: measured inward pad plane, 1mm tolerance'


@dataclass
class LocalObservedField:
    """Use the fine field in its interior and coarse scene field elsewhere.

    Unknown local cells stay unknown even if a coarser neighbor was free. The
    interface is the native ReKep SDF interface, including voxel metadata.
    """

    coarse: object
    fine: object = None

    def __getattr__(self, name):
        return getattr(self.coarse, name)

    def _fine_mask(self, points):
        if self.fine is None:
            return np.zeros(len(points), bool)
        # Preserve interpolation support at the local field boundary.
        return (np.isfinite(points).all(axis=1)
                & (points >= self.fine.bounds_min + self.fine.spacing).all(axis=1)
                & (points <= self.fine.bounds_max - self.fine.spacing).all(axis=1))

    def query(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        values = self.coarse.query(points)
        local = self._fine_mask(points)
        if local.any():
            values[local] = self.fine.query(points[local])
        return values

    def states(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        result = self.coarse.states(points)
        local = self._fine_mask(points)
        if local.any():
            fine = self.fine.states(points[local])
            for name in result:
                result[name][local] = fine[name]
        return result

    def collision_values(self, points, margin=COLLISION_MARGIN_M, finger_flags=None,
                         target_points=None, contact_radius=.012):
        points = np.asarray(points, float).reshape(-1, 3)
        flags = None if finger_flags is None else np.asarray(finger_flags, bool)
        values, allowed = self.coarse.collision_values(points, margin, flags, target_points, contact_radius)
        local = self._fine_mask(points)
        if local.any():
            flags = None if flags is None else flags[local]
            values[local], allowed[local] = self.fine.collision_values(
                points[local], margin, flags, target_points, contact_radius)
        return values, allowed


def build_collision_fields(cameras, target_cloud, robot_volumes=(), *,
                           bounds_min=(-1.05, -.95, .65), bounds_max=(.8, .95, 1.9),
                           coarse_voxel=.01, fine_voxel=.003, local_padding=.16,
                           robot_visual_volumes=(), robot_material_artifact=None):
    """Build paired depth fields, excluding only exact current robot volumes.

    A bounded local grid avoids allocating a 3mm field across the full scene.
    Both fields retain unobserved cells outside configured robot interiors.
    """
    cloud = _points(target_cloud, 'Predicted target cloud')
    if not 0 < fine_voxel <= .003 or coarse_voxel < fine_voxel:
        raise ValueError('Fine collision voxels must be at most 3mm and no larger than coarse voxels')
    if not np.isfinite(local_padding) or local_padding < .13:
        raise ValueError('Local field must cover the physical Panda hand with at least 13cm padding')
    low, high = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
    local_low = np.maximum(low, cloud.min(axis=0) - local_padding)
    local_high = np.minimum(high, cloud.max(axis=0) + local_padding)
    cameras = list(cameras)
    robot_visual_volumes = list(robot_visual_volumes)
    self_filter = {'enabled': False, 'visual_hulls_clear_unknown': False}
    if robot_visual_volumes or any(camera.get('known_robot_depth_prior') is not None for camera in cameras):
        from .sensor_filter_v3 import robot_endpoint_cameras, build_sensor_field
        cameras, self_filter = robot_endpoint_cameras(cameras, robot_visual_volumes)
        coarse = build_sensor_field(AnalyticRobotSDF, cameras, low, high, coarse_voxel)
        fine = build_sensor_field(AnalyticRobotSDF, cameras, local_low, local_high, fine_voxel)
    else:
        coarse = AnalyticRobotSDF.from_rgbd(cameras, low, high, coarse_voxel)
        fine = AnalyticRobotSDF.from_rgbd(cameras, local_low, local_high, fine_voxel)
    robot_volumes = list(robot_volumes)
    removed = {'coarse': coarse.exclude_known_robot_volumes(robot_volumes),
               'fine': fine.exclude_known_robot_volumes(robot_volumes)}
    material_metadata = None
    if robot_material_artifact is not None:
        from .current_material_v3 import CurrentRobotMaterialPrior
        material = CurrentRobotMaterialPrior.from_artifact(robot_material_artifact)
        removed['coarse_exact_visual_material'] = coarse.exclude_current_visual_material(material)
        removed['fine_exact_visual_material'] = fine.exclude_current_visual_material(material)
        material_metadata = material.metadata()
    metadata = {'coarse_spacing_m': coarse.spacing.tolist(), 'fine_spacing_m': fine.spacing.tolist(),
                'local_bounds_min': local_low.tolist(), 'local_bounds_max': local_high.tolist(),
                'coarse_shape': list(coarse.free.shape), 'fine_shape': list(fine.free.shape),
                'known_robot_removed_cells': removed,
                'robot_sensor_self_filter': self_filter,
                'exact_current_robot_visual_material': material_metadata,
                'collision_margin_m': COLLISION_MARGIN_M,
                'unknown_space_policy': 'blocked in both fields; no arm sweep waiver',
                'occupancy_source': 'calibrated RGB-D only'}
    return coarse, fine, metadata


def _collision(sdf, points, contact_flags, target, margin=COLLISION_MARGIN_M):
    values, allowed = sdf.collision_values(points, margin, contact_flags, target)
    states = sdf.states(points)
    blocked = values > 0
    return {'accepted': bool(not blocked.any()),
            'maximum_violation_m': float(np.max(values, initial=-1)),
            'blocked_points': int(blocked.sum()), 'unknown_points': int(states['unknown'].sum()),
            'occupied_points': int(states['occupied'].sum()), 'contact_exemptions': int(allowed.sum()),
            'blocked_unknown_points': int((blocked & states['unknown']).sum()),
            'blocked_occupied_points': int((blocked & states['occupied']).sum())}


def _normalise_ik(result):
    if not isinstance(result, dict):
        result = vars(result)
    result = dict(result)
    joints = np.asarray(result['cspace_position'], float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError('IK must return all seven finite Panda joints')
    result['cspace_position'] = joints.tolist()
    return result


def _multi_ik(sim, tcp, seed=None, max_iterations=180):
    kwargs = {'seed_joints': None if seed is None else np.asarray(seed).tolist(),
              'max_iterations': max_iterations}
    if hasattr(sim, 'multi_ik'):
        result = sim.multi_ik(np.asarray(tcp).tolist(), **kwargs)
        if isinstance(result, dict):
            solutions = result.get('solutions', result.get('successful_solutions', []))
            attempts = result.get('attempts', [])
        else:
            solutions, attempts = result, []
    elif hasattr(sim, 'ik'):
        result = sim.ik(np.asarray(tcp).tolist(), **kwargs)
        solutions, attempts = ([result] if result['success'] else []), [result]
    else:
        result = sim(np.asarray(tcp), initial_joint_pos=seed, max_iterations=max_iterations)
        result = _normalise_ik(result)
        solutions, attempts = ([result] if result['success'] else []), [result]
    return [_normalise_ik(solution) for solution in solutions], attempts


def _arm_face_flags(solution, tcp, calibration):
    points = np.asarray(solution['robot_collision_points_world'], float).reshape(-1, 3)
    if solution.get('robot_collision_contact_face_flags') is not None:
        contact = np.asarray(solution['robot_collision_contact_face_flags'], bool)
        fingers = np.asarray(solution['robot_collision_finger_flags'], bool)
        if contact.shape != (len(points),) or fingers.shape != contact.shape or np.any(contact & ~fingers):
            raise ValueError('Robot contact faces must be a subset of configured finger geometry')
        return contact
    # Do not trust the old whole-finger flag to exempt finger backs or sides.
    faces, _ = contact_face_flags(calibration)
    local = transform_points(points, np.linalg.inv(pose_matrix(tcp)))
    tool = np.asarray(calibration['tool_collision_points_tcp'], float)
    distances, indices = cKDTree(tool).query(local)
    return faces[indices] & (distances <= .001)


def _solution_record(solution, index, nominal_tcp=None):
    record = {'solution_index': index, 'solution_joints': solution['cspace_position'],
            'ik_position_error_m': float(solution['position_error']),
            'ik_orientation_error_rad': float(solution.get('orientation_error', 0)),
            'ik_descents': int(solution.get('num_descents', 0)),
            'ik_seed_index': solution.get('seed_index')}
    fk = solution.get('eef_pose_world_xyzw')
    if fk is not None or nominal_tcp is not None:
        actual = np.asarray(nominal_tcp if fk is None else fk, float)
        if actual.shape != (7,) or not np.isfinite(actual).all():
            raise ValueError('Robot FK transport must return a finite world xyzw TCP pose')
        actual_matrix = pose_matrix(actual)
        record['fk_pose_world_xyzw'] = actual.tolist()
        record['fk_pose_source'] = ('nominal_fallback_missing_transport_fk' if fk is None
                                    else 'isolated_robot_forward_kinematics')
        if nominal_tcp is not None:
            nominal = pose_matrix(nominal_tcp)
            cosine = np.clip((np.trace(actual_matrix[:3, :3] @ nominal[:3, :3].T) - 1) / 2, -1, 1)
            record['fk_position_adjustment_m'] = float(np.linalg.norm(actual_matrix[:3, 3] - nominal[:3, 3]))
            record['fk_orientation_adjustment_rad'] = float(np.arccos(cosine))
    return record


def filter_candidates(generated, keypoint, target_cloud, calibration, sim, sdf, fine_sdf=None, *,
                      proximity_limit=.10, membership_limit=.035,
                      collision_margin=COLLISION_MARGIN_M, max_iterations=180,
                      initial_joints=None, asset_contact_alignment_m=None):
    """Return sorted feasible raw/symmetric candidates and every rejection.

    Each distinct collision-free IK solution is retained so path continuity can
    choose among them. Collision checks use the same unchanged 5mm clearance
    margin in the paired coarse/fine comparison. A configured asset alignment
    adds separate variants; it never overwrites the original detector pose.
    """
    if collision_margin != COLLISION_MARGIN_M:
        raise ValueError('Version 3 preserves the measured 5mm collision margin')
    target = _points(target_cloud, 'Predicted target cloud')
    keypoint = np.asarray(keypoint, float)
    if keypoint.shape != (3,) or not np.isfinite(keypoint).all():
        raise ValueError('Grasp keypoint must be a finite world point')
    tool = _points(calibration['tool_collision_points_tcp'], 'Configured tool')
    faces, label_source = contact_face_flags(calibration)
    field = LocalObservedField(sdf, fine_sdf)
    target_tree = cKDTree(target)
    feasible, rejected, attempted = [], [], 0
    symmetry = np.diag([-1., -1., 1., 1.])
    variants = [('raw', np.eye(4)), ('antipodal_half_turn', symmetry)]
    if asset_contact_alignment_m is None:
        learned_z = calibration.get('graspgen_finger_attach_z_m')
        physical_z = calibration.get('physical_finger_attach_z_m')
        asset_contact_alignment_m = 0. if learned_z is None or physical_z is None else float(learned_z) - float(physical_z)
    alignment = float(asset_contact_alignment_m)
    if not np.isfinite(alignment) or abs(alignment) > .01:
        raise ValueError('Asset contact alignment must be a measured axial difference of at most 1cm')
    if abs(alignment) > 1e-9:
        shift = np.eye(4)
        shift[2, 3] = alignment
        variants.extend([('asset_contact_aligned', shift), ('asset_contact_aligned_half_turn', shift @ symmetry)])
    limits = np.asarray(calibration.get('joint_limits', np.tile([-np.inf, np.inf], (7, 1))), float)
    if limits.shape != (7, 2):
        raise ValueError('Joint limits must contain all seven Panda joints')
    for raw_index in generated['retained_indices']:
        index = int(raw_index)
        for variant, rotation in variants:
            attempted += 1
            record = {'index': index, 'variant': variant, 'score': float(generated['scores'][index])}
            try:
                base = pose_matrix(generated['poses_world_base'][index]) @ rotation
                tcp = graspgen_to_tcp(base, calibration)
                record['tcp_pose_world_xyzw'] = tcp.tolist()
                record['keypoint_distance_m'] = float(np.linalg.norm(tcp[:3] - keypoint))
                record['region_distance_m'] = float(target_tree.query(tcp[:3])[0])
                if (record['keypoint_distance_m'] > proximity_limit
                        or record['region_distance_m'] > membership_limit):
                    record['reason'] = 'target_membership_or_keypoint_proximity'
                    rejected.append(record)
                    continue
                world_tool = transform_points(tool, pose_matrix(tcp))
                coarse_check = _collision(sdf, world_tool, faces, target, collision_margin)
                collision = _collision(field, world_tool, faces, target, collision_margin)
                record.update(collision=collision, coarse_tool_collision=coarse_check,
                              coarse_fine_tool_admission_change=coarse_check['accepted'] != collision['accepted'])
                if not collision['accepted']:
                    record['reason'] = 'tool_collision_or_unknown'
                    rejected.append(record)
                    continue
                solutions, attempts = _multi_ik(sim, tcp, initial_joints, max_iterations)
                record['ik_attempt_count'] = len(attempts)
                record['ik_solution_count'] = len(solutions)
                if not solutions:
                    record['reason'] = 'ik_unreachable'
                    record['ik_attempts'] = [{name: item.get(name) for name in
                        ['success', 'position_error', 'orientation_error', 'seed_index']}
                        for item in attempts if isinstance(item, dict)]
                    rejected.append(record)
                    continue
                accepted_solutions, solution_rejections = [], []
                for solution_index, solution in enumerate(solutions):
                    result = _solution_record(solution, solution_index, tcp)
                    fk_tcp = np.asarray(result['fk_pose_world_xyzw'], float)
                    joints = np.asarray(solution['cspace_position'])
                    if not solution.get('success', False) or (joints < limits[:, 0]).any() or (joints > limits[:, 1]).any():
                        result['reason'] = 'ik_or_joint_limits'
                    elif solution.get('robot_collision_points_world') is None:
                        # A detector pose alone cannot certify its entire arm.
                        result['reason'] = 'ik_missing_robot_collision_geometry'
                    else:
                        fk_tool = transform_points(tool, pose_matrix(fk_tcp))
                        result['fk_tool_collision'] = _collision(field, fk_tool, faces, target, collision_margin)
                        if not result['fk_tool_collision']['accepted']:
                            result['reason'] = 'fk_tool_collision_or_unknown'
                            solution_rejections.append(result)
                            continue
                        arm = np.asarray(solution['robot_collision_points_world'], float)
                        arm_faces = _arm_face_flags(solution, fk_tcp, calibration)
                        arm_check = _collision(field, arm, arm_faces, target, collision_margin)
                        result['arm_collision'] = arm_check
                        result['coarse_arm_collision'] = _collision(sdf, arm, arm_faces, target, collision_margin)
                        if not arm_check['accepted']:
                            result['reason'] = 'arm_collision_or_unknown'
                    if 'reason' in result:
                        solution_rejections.append(result)
                    else:
                        accepted_solutions.append(result)
                record['solution_rejections'] = solution_rejections
                if not accepted_solutions:
                    record['reason'] = 'arm_collision_or_unknown'
                    rejected.append(record)
                    continue
                for solution in accepted_solutions:
                    feasible.append({**record, **solution})
            except (ValueError, FloatingPointError) as error:
                record.update(reason='invalid_candidate', detail=str(error))
                rejected.append(record)
    seed = np.zeros(7) if initial_joints is None else np.asarray(initial_joints, float)
    feasible.sort(key=lambda item: (item['keypoint_distance_m'], -item['score'],
        float(np.linalg.norm(np.asarray(item['solution_joints']) - seed)),
        item['index'], item['variant'], item['solution_index']))
    chosen = feasible[0] if feasible else None
    return {'version': 'grasp-admission-3.0', 'accepted': chosen is not None,
            'selected_index': None if chosen is None else chosen['index'],
            'selected_variant': None if chosen is None else chosen['variant'],
            'selected_tcp_pose': None if chosen is None else chosen['tcp_pose_world_xyzw'],
            'selected_fk_tcp_pose': None if chosen is None else chosen['fk_pose_world_xyzw'],
            'selected_solution_joints': None if chosen is None else chosen['solution_joints'],
            'feasible_candidates': feasible, 'rejected_candidates': rejected,
            'attempted_pose_variants': attempted,
            'failure': None if chosen is not None else 'grasp_no_feasible_detector_candidates',
            'thresholds': {'proximity_m': proximity_limit, 'membership_m': membership_limit,
                           'collision_margin_m': collision_margin},
            'contact_face_labels': label_source, 'contact_face_points': int(faces.sum()),
            'asset_contact_alignment_m': alignment,
            'asset_alignment_policy': 'fixed robot asset difference; unchanged 97mm base-to-TCP calibration',
            'symmetry': 'Panda symmetric_antipodal: local approach-axis half-turn only',
            'field': 'strict RGB-D coarse and local fine; unknown blocked'}


def validate_pose_path(poses, calibration, sim, sdf, *, fine_sdf=None, initial_joints=None,
                       target_points=None, allow_target_contact=False, max_iterations=180):
    """Validate a dense path with continuous-seed IK and explicit face contacts.

    A lift without a verified attachment receives no target-contact exemptions;
    callers must provide an independently updated observed scene before lifting.
    """
    poses = np.asarray(poses, float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses):
        raise ValueError('A pose path must contain finite world xyzw poses')
    tool = _points(calibration['tool_collision_points_tcp'], 'Configured tool')
    faces, _ = contact_face_flags(calibration)
    faces = faces if allow_target_contact else np.zeros(len(tool), bool)
    target = target_points if allow_target_contact else None
    field = LocalObservedField(sdf, fine_sdf)
    checks, selected_joints, fk_poses = [], [], []
    seed = None if initial_joints is None else np.asarray(initial_joints, float)
    limits = np.asarray(calibration['joint_limits'], float)
    for sample_index, pose in enumerate(poses):
        tool_check = _collision(field, transform_points(tool, pose_matrix(pose)), faces, target)
        if not tool_check['accepted']:
            return {'accepted': False, 'failure': 'path_tool_collision_or_unknown',
                    'sample_index': sample_index, 'collision': tool_check, 'checks': checks}
        solutions, attempts = _multi_ik(sim, pose, seed, max_iterations)
        if seed is not None:
            solutions.sort(key=lambda item: np.linalg.norm(np.asarray(item['cspace_position']) - seed))
        rejected_solutions, chosen = [], None
        for index, solution in enumerate(solutions):
            joints = np.asarray(solution['cspace_position'], float)
            record = _solution_record(solution, index, pose)
            fk_tcp = np.asarray(record['fk_pose_world_xyzw'], float)
            if ((joints < limits[:, 0]).any() or (joints > limits[:, 1]).any()
                    or not solution.get('success', False)):
                record['reason'] = 'ik_or_joint_limits'
            elif seed is not None and np.max(np.abs(joints - seed)) > .35:
                record['reason'] = 'ik_branch_discontinuity'
            elif solution.get('robot_collision_points_world') is None:
                record['reason'] = 'ik_missing_robot_collision_geometry'
            else:
                fk_tool = transform_points(tool, pose_matrix(fk_tcp))
                record['fk_tool_collision'] = _collision(field, fk_tool, faces, target)
                if not record['fk_tool_collision']['accepted']:
                    record['reason'] = 'fk_tool_collision_or_unknown'
                    rejected_solutions.append(record)
                    continue
                arm_faces = (_arm_face_flags(solution, fk_tcp, calibration) if allow_target_contact
                             else np.zeros(len(solution['robot_collision_points_world']), bool))
                record['collision'] = _collision(field, solution['robot_collision_points_world'], arm_faces, target)
                if not record['collision']['accepted']:
                    record['reason'] = 'arm_collision_or_unknown'
            if 'reason' in record:
                rejected_solutions.append(record)
            else:
                chosen = record
                break
        if chosen is None:
            return {'accepted': False, 'failure': 'path_ik_or_arm_collision',
                    'sample_index': sample_index, 'checks': checks,
                    'ik_attempt_count': len(attempts), 'solution_rejections': rejected_solutions}
        seed = np.asarray(chosen['solution_joints'], float)
        selected_joints.append(seed.tolist())
        fk_poses.append(chosen['fk_pose_world_xyzw'])
        checks.append({'sample_index': sample_index, 'tool_collision': tool_check, **chosen})
    return {'accepted': True, 'failure': None, 'checks': checks, 'solution_joints': selected_joints,
            'fk_poses_world_xyzw': fk_poses,
            'command_pose_policy': 'checked robot FK pose paired with its checked seven-joint posture; nominal poses preserved separately'}
