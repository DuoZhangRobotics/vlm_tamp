"""Tagged RGB-D surfaces for coupled robot/hinge planning.

The baseline models observed surfaces. An optional, explicitly recorded planar
shape prior samples missing regions of the bounded selected door panel; these
inferred samples stay separate from observations and do not certify true hidden
geometry. Stationary observations are never cleared by hypothetical robot or
door poses. Contact permissions apply only to the selected moving geometry.
"""

import copy
import re

import numpy as np
from scipy.ndimage import binary_dilation, label
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .rekep_libero.geometry import pose_matrix, transform_points
from .rekep_libero.scene_state_v7 import bounded_points, observed_points, voxel_points


def _unit(vector):
    vector = np.asarray(vector, float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm < 1e-9:
        raise ValueError('Expected a finite nonzero three dimensional axis')
    return vector / norm


def _transform_volume(volume, transform):
    """Rigidly move a configured volume without rebuilding its shape."""
    rotation, translation = transform[:3, :3], transform[:3, 3]
    result = dict(volume)
    if volume['type'] == 'sphere':
        result['center_world'] = np.asarray(volume['center_world']) @ rotation.T + translation
    elif volume['type'] == 'convex':
        result['vertices_world'] = np.asarray(volume['vertices_world']) @ rotation.T + translation
        planes = np.asarray(volume['planes_world'], float)
        normals = planes[:, :3] @ rotation.T
        result['planes_world'] = np.c_[normals, planes[:, 3] - normals @ translation]
    else:
        raise ValueError('Unsupported configured robot volume: ' + volume['type'])
    return result


def _bounded_reference_points(points, tree, world_volume, reference_volume, transform, margin):
    """Match the world checker broad phase while querying a fixed point tree.

    Recomputing an inflated AABB in the rotated reference frame can change
    which positive-clearance corner samples are inspected. Transport the
    original world bounding sphere and retain its world AABB predicate so the
    optimization preserves the existing admission test, including its margin.
    """
    if world_volume['type'] == 'sphere':
        return bounded_points(points, tree, reference_volume, margin)
    vertices = np.asarray(world_volume['vertices_world'])
    low, high = vertices.min(0)-margin, vertices.max(0)+margin
    rotation, translation = transform[:3, :3], transform[:3, 3]
    centre_reference = (((low+high)/2)-translation)@rotation
    ids = np.asarray(tree.query_ball_point(centre_reference, np.linalg.norm(high-low)/2), int)
    world_near = points[ids]@rotation.T+translation
    ids = ids[(world_near >= low).all(1)&(world_near <= high).all(1)]
    planes = np.asarray(reference_volume['planes_world'])
    values = np.empty(len(ids))
    for offset in range(0, len(ids), 1024):
        batch = ids[offset:offset+1024]
        values[offset:offset+1024] = np.max(points[batch]@planes[:, :3].T+planes[:, 3], axis=1)
    inside = values <= margin
    return ids[inside], values[inside]


class RobotVolumes:
    """Configured Panda volumes moved by pure PoE kinematics, at fixed aperture.

    The reference volumes must be synchronized with ``reference_joints``. The
    joint axes/anchors themselves are calibrated at chain.reference_joints.
    Link N is downstream of N arm joints; the complete gripper is downstream
    of all seven. Unknown geometry names fail instead of guessing attachment.
    """

    def __init__(self, chain, reference_volumes, reference_joints=None):
        self.chain = chain
        self.reference_volumes = list(reference_volumes)
        reference = chain.reference_joints if reference_joints is None else reference_joints
        self.reference_joints = np.asarray(reference, float).copy()
        prefixes = self._prefixes(reference)
        self.entries = []
        for volume in self.reference_volumes:
            name = volume['robot_geom_name']
            match = re.search(r'(?:^|_)link([0-7])_collision$', name)
            if volume.get('fixed'):
                count = 0
            elif match:
                count = int(match.group(1))
            elif 'gripper' in name and ('hand' in name or 'finger' in name):
                count = 7
            else:
                raise ValueError('Unknown configured robot volume attachment: ' + name)
            if volume['type'] not in {'convex', 'sphere'}:
                raise ValueError('Unsupported configured robot volume: ' + volume['type'])
            self.entries.append((volume, count, np.linalg.inv(prefixes[count])))

    def _prefixes(self, joints):
        q = np.asarray(joints, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Expected seven finite robot joints')
        result = [np.eye(4)]
        product = np.eye(4)
        for axis, anchor, delta in zip(self.chain.axes, self.chain.anchors,
                                       q - self.chain.reference_joints):
            rotation = Rotation.from_rotvec(delta * axis).as_matrix()
            step = np.eye(4)
            step[:3, :3] = rotation
            step[:3, 3] = anchor - rotation @ anchor
            product = product @ step
            result.append(product.copy())
        return result

    def forward(self, joints):
        prefixes = self._prefixes(joints)
        volumes = []
        for original, count, reference_inverse in self.entries:
            transform = prefixes[count] @ reference_inverse
            volumes.append(_transform_volume(original, transform))
        return volumes


def observed_mount_mask(points, spec, handle_cloud, moving_seed, *, connectivity=.006):
    """Own supported observed mount samples in a finite upright-handle gap.

    This is a documented shape prior for an observed vertical bar fixed to an
    upright panel. It does not complete a solid mount or identify arbitrary
    objects: endpoints must be inside the finite bar-to-panel capsule AND
    within 6 mm of a measured endpoint already assigned to the moving part.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    handle = np.asarray(handle_cloud, float).reshape(-1, 3)
    result = np.zeros(len(points), bool)
    if 'source_anchor_world_m' not in spec or not len(handle):
        return result
    axis = _unit(spec['joint_axis_world'])
    normal = _unit(spec['panel_normal_world'])
    # The capsule assumes a vertical bar and a horizontal mounting direction.
    if abs(axis[2]) < 1.-1e-8 or abs(normal[2]) > .01 or np.ptp(handle[:, 2]) < .035:
        return result
    anchor = np.asarray(spec['source_anchor_world_m'], float)
    origin = np.asarray(spec['front_plane']['point_world_m'], float)
    radius = float(np.quantile(np.linalg.norm(handle[:, :2]-anchor[:2], axis=1), .98))
    if not .005 <= radius <= .025:
        return result
    projected = anchor-float((anchor-origin)@normal)*normal
    direction = projected[:2]-anchor[:2]
    squared_length = float(direction@direction)
    if not .001**2 <= squared_length <= .09**2:
        return result
    fraction = np.clip((points[:, :2]-anchor[:2])@direction/squared_length, 0., 1.)
    radial = np.linalg.norm(points[:, :2]-(anchor[:2]+fraction[:, None]*direction), axis=1)
    candidate = ((radial <= radius+.004)
                 & (points[:, 2] >= handle[:, 2].min()-.015)
                 & (points[:, 2] <= handle[:, 2].max()+.015))
    selected = np.flatnonzero(candidate)
    seeds = points[np.asarray(moving_seed, bool)]
    if len(selected) and len(seeds):
        distance = cKDTree(seeds).query(points[selected])[0]
        result[selected] = distance <= float(connectivity)
    return result


def bounded_panel_completion(observed, origin, normal, half_band, *, spacing=.005,
                             maximum_points=50000, footprint_support=None):
    """Sample an explicit finite planar shape prior across a selected panel.

    Only tagged measured panel support establishes the extent. Missing holes
    between those endpoints are conservatively filled, including transparent
    or occluded windows; these samples are inferred geometry, not observations.
    """
    observed = np.asarray(observed, float).reshape(-1, 3)
    origin, normal = np.asarray(origin, float), _unit(normal)
    if abs(normal[2]) > .01:
        raise ValueError('Planar completion requires a measured upright selected panel')
    if not .001 <= spacing <= .005 or not .004 <= half_band <= .05:
        raise ValueError('Unsupported planar completion spacing or measured panel thickness')
    side = _unit(np.cross(normal, [0., 0., 1.]))
    up = _unit(np.cross(side, normal))
    local = observed-origin
    # Protruding handles and mounts do not enlarge the panel's depth envelope.
    support = local[np.abs(local@normal) <= half_band+.004]
    if len(support) < 75:
        raise ValueError('Planar completion lacks at least 75 tagged measured panel samples')
    footprint = support
    if footprint_support is not None:
        supplied = np.asarray(footprint_support, float)
        if supplied.ndim != 2 or supplied.shape[1:] != (3,) or not np.isfinite(supplied).all():
            raise ValueError('Panel footprint needs finite measured Nx3 points')
        footprint = supplied-origin
        footprint = footprint[np.abs(footprint@normal) <= half_band+.004]
        if len(footprint)<75:
            raise ValueError('Panel footprint lacks at least 75 selected measured samples')
    coordinates = np.c_[footprint@side, footprint@up]
    lower, upper = np.quantile(coordinates, [.02, .98], axis=0)
    extent = upper-lower
    occupied_cells = np.unique(np.floor(coordinates/.005).astype(int), axis=0)
    if np.any(extent < .05) or len(occupied_cells) < 20:
        raise ValueError('Planar completion lacks finite two-dimensional panel support')
    band = float(half_band)+.004
    bounds = [(float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1])), (-band, band)]
    counts = [int(np.ceil((high-low)/spacing))+1 for low, high in bounds]
    count = int(np.prod(counts))
    if count > int(maximum_points):
        raise ValueError(f'Planar completion needs {count} samples, exceeding {maximum_points}; refusing coarser geometry')
    axes = [np.linspace(low, high, n) for (low, high), n in zip(bounds, counts)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    points = origin+grid[:, :1]*side+grid[:, 1:2]*up+grid[:, 2:3]*normal
    return points, {'enabled': True, 'source': 'bounded planar shape prior from tagged observed selected-panel support',
        'inferred_points': len(points), 'tagged_observed_support_points': len(support),
        'footprint_support_points':len(footprint),
        'footprint_source':'selected RGB-D plane pixels' if footprint_support is not None else 'all tagged moving support',
        'side_axis_world': side.tolist(), 'vertical_axis_world': up.tolist(),
        'normal_axis_world': normal.tolist(), 'origin_world_m': origin.tolist(),
        'side_bounds_m': list(bounds[0]), 'height_bounds_m': list(bounds[1]),
        'normal_bounds_m': list(bounds[2]), 'spacing_at_most_m': float(spacing),
        'robust_extent_quantiles': [.02, .98], 'maximum_points': int(maximum_points),
        'observations_relabelled_or_erased': False,
        'assumption': 'selected upright panel is a solid bounded slab, including unseen interior windows',
        'true_hidden_geometry_validated': False, 'continuous_solid_collision_guarantee': False}


def moving_mask(points, camera, spec, handle_cloud, panel_mask, choices=None, config=None):
    """Associate selected panel/handle observations; leave all others static.

    The image selection is intersected with measured plane thickness and its
    observed in-plane extents. Other views contribute only their measured
    endpoints, never completed/filled panel interiors.
    """
    config = config or {}
    points = np.asarray(points, float).reshape(-1, 3)
    handle = np.asarray(handle_cloud, float).reshape(-1, 3)
    origin = np.asarray(spec['front_plane']['point_world_m'], float)
    normal = _unit(spec['panel_normal_world'])
    up = np.array([0., 0., 1.])
    side = _unit(np.cross(normal, up))
    mask = np.asarray(panel_mask, bool)
    support = np.asarray(camera['world_points'])[mask]
    support = support[np.isfinite(support).all(1)]
    if len(support) < 50:
        raise ValueError('Selected moving panel lacks observed depth support')
    local = support - origin
    low, high = np.quantile(np.c_[local @ side, support[:, 2]], [.005, .995], axis=0)
    extent_pad = float(config.get('panel_extent_padding_m', .006))
    panel_band = float(spec.get('measured_panel_surface_half_band_m',
                                spec.get('panel_depth_association_m', .032))) + .004
    local = points - origin
    bounded = ((local @ side >= low[0] - extent_pad)
               & (local @ side <= high[0] + extent_pad)
               & (points[:, 2] >= low[1] - extent_pad)
               & (points[:, 2] <= high[1] + extent_pad)
               & (np.abs(local @ normal) <= panel_band))
    optical = transform_points(points, np.linalg.inv(pose_matrix(camera['camera_to_world'])))
    projected = optical @ np.asarray(camera['intrinsic'], float).T
    pixel = np.full((len(points), 2), -1., float)
    valid = optical[:, 2] > 1e-8
    pixel[valid] = projected[valid, :2] / projected[valid, 2:3]
    if choices is not None and 'moving_panel_box_xyxy' in choices:
        x0, y0, x1, y1 = choices['moving_panel_box_xyxy']
        selected = (valid & (pixel[:, 0] >= x0) & (pixel[:, 0] <= x1)
                    & (pixel[:, 1] >= y0) & (pixel[:, 1] <= y1))
    else:
        xy = np.rint(pixel).astype(int)
        valid &= ((xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1])
                  & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0]))
        selected = np.zeros(len(points), bool)
        selected[valid] = mask[xy[valid, 1], xy[valid, 0]]
    primary = bounded & selected
    # Cached RGB boxes can contain only a thin portion of the upright face.
    # Extend over connected *observed* planar support between the grounded
    # hinge and handle, with height bounded by the observed handle. This does
    # not fill image holes or classify a whole cabinet as moving.
    if config.get('extend_observed_panel', True):
        hinge = np.asarray(spec['joint_origin_world_m'], float)
        side_ends = np.r_[(hinge-origin)@side, (handle-origin)@side]
        side_padding = float(config.get('hinge_handle_panel_padding_m', .02))
        height_padding = float(config.get('handle_panel_height_padding_m', .035))
        side_coordinate = local @ side
        candidates = ((side_coordinate >= side_ends.min()-side_padding)
                      & (side_coordinate <= side_ends.max()+side_padding)
                      & (points[:, 2] >= handle[:, 2].min()-height_padding)
                      & (points[:, 2] <= handle[:, 2].max()+height_padding)
                      & (np.abs(local@normal) <= panel_band))
        indices = np.flatnonzero(candidates)
        if len(indices):
            planar = np.c_[side_coordinate[indices], points[indices, 2]]
            cell_size = float(config.get('panel_connectivity_spacing_m', .006))
            cells = np.floor((planar-planar.min(0))/cell_size).astype(int)+1
            shape = cells.max(0)+2
            occupancy = np.zeros(shape, bool)
            occupancy[tuple(cells.T)] = True
            groups, _ = label(binary_dilation(occupancy, structure=np.ones((3, 3), bool)))
            component_ids = groups[tuple(cells.T)]
            seeds = component_ids[primary[indices]]
            if len(seeds):
                connected = np.isin(component_ids, np.unique(seeds))
                primary[indices[connected]] = True
    near_handle = cKDTree(handle).query(points)[0] <= float(config.get('handle_association_m', .006))
    if 'source_anchor_world_m' in spec:
        # A partial observed cylinder must also own newly visible opposite
        # sides. NN distance to the old visible half alone leaves ghost bars
        # in the stationary map after a turn.
        centre=np.asarray(spec['source_anchor_world_m'],float)[:2]
        radius=float(np.quantile(np.linalg.norm(handle[:,:2]-centre,axis=1),.98))
        radial=np.linalg.norm(points[:,:2]-centre,axis=1)
        near_handle|=((radial<=radius+.004)&(points[:,2]>=handle[:,2].min()-.015)
                      &(points[:,2]<=handle[:,2].max()+.015))
    selected = primary | near_handle
    selected |= observed_mount_mask(points, spec, handle, selected,
        connectivity=float(config.get('mount_observed_connectivity_m', .006)))
    return selected


class Scene:
    """Robot validity at (q, theta), with optional recorded panel completion."""

    def __init__(self, cameras, metadata, spec, handle_cloud, panel_mask, choices=None,
                 config=None, robot=None, panel_completion_prior=None):
        config = config or {}
        if any(view.get('robot_depth_prior') and view.get('known_robot_depth_prior') is None
               for view in cameras.values()):
            raise ValueError('Synchronized configured robot depth prior is missing')
        points, filtering = observed_points(cameras, metadata,
            spacing=float(config.get('scene_voxel_spacing_m', .002)),
            robot_depth_tolerance=float(config.get('robot_self_depth_tolerance_m', .0001)))
        camera_name = choices['camera'] if choices is not None else spec['camera']
        selected = moving_mask(points, cameras[camera_name], spec, handle_cloud,
                               panel_mask, choices, config)
        if selected.sum() < 75:
            raise ValueError('Selected moving panel lacks multiview observed support')
        footprint = None
        if config.get('panel_completion_extent_source')=='selected_plane_pixels':
            footprint=np.asarray(cameras[camera_name]['world_points'])[np.asarray(panel_mask,bool)]
            footprint=footprint[np.isfinite(footprint).all(1)]
        self._initialize(points[~selected], points[selected], handle_cloud, spec, config,
                         robot, metadata['robot_collision_volumes_world'], panel_completion_prior, footprint)
        self.self_filter = filtering

    @classmethod
    def from_points(cls, static_points, moving_points, contact_points, spec, *,
                    config=None, robot=None, reference_volumes=(), panel_completion_prior=None, panel_support=None):
        """Construct from an explicitly tagged observed cloud (also useful in tests)."""
        result = cls.__new__(cls)
        result._initialize(static_points, moving_points, contact_points, spec,
                           config or {}, robot, reference_volumes, panel_completion_prior, panel_support)
        result.self_filter = {'source': 'caller supplied tagged observed points'}
        return result

    def _initialize(self, static, moving, contact, spec, config, robot, reference_volumes,
                    panel_completion_prior=None, panel_support=None):
        self.config, self.spec, self.robot = dict(config), dict(spec), robot
        self.static_points = np.asarray(static, float).reshape(-1, 3).copy()
        self.moving_reference = np.asarray(moving, float).reshape(-1, 3).copy()
        self.contact_reference = np.asarray(contact, float).reshape(-1, 3).copy()
        if not len(self.contact_reference) or not len(self.moving_reference):
            raise ValueError('Moving surface and selected contact observations are required')
        for cloud in (self.static_points, self.moving_reference, self.contact_reference):
            if not np.isfinite(cloud).all():
                raise ValueError('Tagged observed points must be finite')
            cloud.setflags(write=False)
        self.static_tree = cKDTree(self.static_points)
        self.moving_tree = cKDTree(self.moving_reference)
        self.contact_tree = cKDTree(self.contact_reference)
        self.origin = np.asarray(spec['joint_origin_world_m'], float)
        self.axis = _unit(spec['joint_axis_world'])
        self.panel_origin = np.asarray(spec['front_plane']['point_world_m'], float)
        self.panel_normal = _unit(spec['panel_normal_world'])
        self.panel_band = float(spec.get('measured_panel_surface_half_band_m', .032))
        self.inferred_panel_reference = np.empty((0, 3))
        self.panel_completion = {'enabled': False, 'inferred_points': 0}
        if panel_completion_prior is not None and not self.config.get('complete_planar_panel', False):
            raise ValueError('A panel completion prior requires complete_planar_panel=True')
        if self.config.get('complete_planar_panel', False):
            if abs(self.axis[2]) < 1.-1e-8:
                raise ValueError('Planar door completion requires an upright articulation axis')
            if panel_completion_prior is None:
                self.inferred_panel_reference, self.panel_completion = bounded_panel_completion(
                    self.moving_reference, self.panel_origin, self.panel_normal, self.panel_band,
                    spacing=float(self.config.get('panel_completion_spacing_m', .005)),
                    maximum_points=int(self.config.get('panel_completion_maximum_points', 50000)),
                    footprint_support=panel_support)
                self.panel_completion['rigid_prior_reused'] = False
            else:
                self.inferred_panel_reference, self.panel_completion = self._validated_panel_prior(
                    panel_completion_prior)
        self.inferred_panel_reference.setflags(write=False)
        self.collision_reference = np.vstack([self.moving_reference, self.inferred_panel_reference])
        self.collision_reference.setflags(write=False)
        self.collision_tree = cKDTree(self.collision_reference)
        self.required = frozenset(v['robot_geom_name'] for v in reference_volumes if not v.get('fixed'))
        self.tool_panel_radius = float(config.get('tool_panel_brace_radius_m', .10))
        pads = [np.asarray(v['vertices_world']).mean(0) for v in reference_volumes
                if v['robot_geom_name'].endswith('_pad_collision') and v['type'] == 'convex']
        hands = [np.asarray(v['vertices_world']) for v in reference_volumes
                 if v['robot_geom_name'].endswith('hand_collision') and v['type'] == 'convex']
        if len(pads) == 2 and len(hands) == 1:
            self.tool_panel_radius = float(np.linalg.norm(hands[0]-np.mean(pads, axis=0), axis=1).max()+.003)
        self._moving_cache = {}
        self._reference_contact_height = np.quantile(self.contact_reference[:, 2], [.01, .99])
        self._reference_contact_distance = self.contact_tree.query(self.collision_reference)[0]
        self._reference_panel_distance = np.abs((self.collision_reference-self.panel_origin)@self.panel_normal)

    def _validated_panel_prior(self, prior):
        if not isinstance(prior, dict) or not {'points_world', 'record'}.issubset(prior):
            raise ValueError('Panel completion prior needs points_world and record')
        points = np.asarray(prior['points_world'], float)
        limit = int(self.config.get('panel_completion_maximum_points', 50000))
        if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
                or not np.isfinite(points).all() or len(points) > limit):
            raise ValueError('Panel completion prior must contain finite Nx3 points within the count limit')
        record = copy.deepcopy(prior['record'])
        if not isinstance(record, dict) or record.get('inferred_points') != len(points):
            raise ValueError('Panel completion prior record has inconsistent point count')
        try:
            origin = np.asarray(record['origin_world_m'], float)
            basis = np.column_stack([record['side_axis_world'], record['vertical_axis_world'],
                                     record['normal_axis_world']]).astype(float)
            bounds = np.asarray([record['side_bounds_m'], record['height_bounds_m'],
                                 record['normal_bounds_m']], float)
            spacing = float(record['spacing_at_most_m'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Panel completion prior lacks finite primitive metadata') from error
        if (origin.shape != (3,) or basis.shape != (3, 3) or bounds.shape != (3, 2)
                or not np.isfinite(origin).all() or not np.isfinite(basis).all()
                or not np.isfinite(bounds).all() or np.any(bounds[:, 1] < bounds[:, 0])
                or not np.allclose(basis.T@basis, np.eye(3), atol=1e-7)
                or not np.isfinite(spacing) or not 0 < spacing <= .005):
            raise ValueError('Panel completion prior has invalid primitive frame, bounds or spacing')
        local = (points-origin)@basis
        if np.any(local < bounds[:, 0]-1e-7) or np.any(local > bounds[:, 1]+1e-7):
            raise ValueError('Panel completion prior points do not match their recorded finite bounds')
        record.update(enabled=True, inferred_points=len(points), rigid_prior_reused=True,
                      regenerated_from_fresh_view=False)
        return points.copy(), record

    def panel_prior_for(self, spec, handle_cloud):
        """Rigidly transport the existing inferred panel without changing shape.

        Fresh RGB-D remains responsible for observed/static surfaces. The old
        inferred grid is rotated with the measured upright panel normal and
        translated so the observed handle medians align, rather than rebuilding
        a different solid envelope from each partially occluded observation.
        """
        if not len(self.inferred_panel_reference):
            return None
        handle = np.asarray(handle_cloud, float)
        if handle.ndim != 2 or handle.shape[1:] != (3,) or not len(handle) or not np.isfinite(handle).all():
            raise ValueError('Rigid panel transport requires finite observed handle points')
        normal = _unit(spec['panel_normal_world'])
        axis = _unit(spec['joint_axis_world'])
        if (abs(self.panel_normal[2]) > 1e-8 or abs(normal[2]) > 1e-8
                or abs(axis[2]) < 1.-1e-8):
            raise ValueError('Rigid panel prior transport requires upright observed panel normals and axis')
        angle = float(np.arctan2(np.cross(self.panel_normal, normal)[2], self.panel_normal@normal))
        rotation = Rotation.from_rotvec([0., 0., angle]).as_matrix()
        previous_centre = np.median(self.contact_reference, axis=0)
        current_centre = np.median(handle, axis=0)
        translation = current_centre-rotation@previous_centre
        points = self.inferred_panel_reference@rotation.T+translation
        record = copy.deepcopy(self.panel_completion)
        record['origin_world_m'] = (rotation@np.asarray(record['origin_world_m'])+translation).tolist()
        for key in ['side_axis_world', 'vertical_axis_world', 'normal_axis_world']:
            record[key] = (rotation@np.asarray(record[key])).tolist()
        history = record.setdefault('rigid_transport_history', [])
        history.append({'rotation_about_world_z_rad': angle, 'rotation_world': rotation.tolist(),
            'translation_world_m': translation.tolist(),
            'previous_observed_handle_median_world_m': previous_centre.tolist(),
            'current_observed_handle_median_world_m': current_centre.tolist()})
        record.update(rigid_prior_reused=True, regenerated_from_fresh_view=False,
            rigid_shape_point_count_preserved=len(points)==len(self.inferred_panel_reference),
            rigid_shape_spacing_preserved=True,
            transport_source='observed upright panel normal rotation and observed handle median translation',
            current_observed_panel_origin_world_m=list(spec['front_plane']['point_world_m']))
        points.setflags(write=False)
        return {'points_world': points, 'record': record}

    def transform(self, theta):
        if not np.isfinite(theta):
            raise ValueError('Hinge progress must be finite')
        rotation = Rotation.from_rotvec(float(theta)*self.axis).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = self.origin-rotation@self.origin
        return transform

    def moved(self, theta):
        key = float(theta)
        if key not in self._moving_cache:
            transform = self.transform(theta)
            points = transform_points(self.moving_reference, transform)
            contacts = transform_points(self.contact_reference, transform)
            value = (points, cKDTree(points), contacts, cKDTree(contacts), transform)
            if len(self._moving_cache) >= 8:
                self._moving_cache.pop(next(iter(self._moving_cache)))
            self._moving_cache[key] = value
        return self._moving_cache[key]

    def check_volumes(self, volumes, theta, *, contact=True, execution=False):
        volumes = list(volumes)
        if not self.required.issubset({v['robot_geom_name'] for v in volumes}):
            raise ValueError('Full configured robot collision geometry is required')
        transform = self.transform(theta)
        inverse = np.eye(4)
        inverse[:3, :3] = transform[:3, :3].T
        inverse[:3, 3] = -inverse[:3, :3]@transform[:3, 3]
        # Plan in a fixed observed-door frame: moving robot volumes is cheaper
        # than transporting/reindexing tens of thousands of observed points
        # at every validity callback. Distances and permissions are invariant
        # under this common rigid transform. World height must still be used
        # for the local brace band if a non-upright hinge is ever supplied.
        upright = np.array_equal(self.axis[:2], [0., 0.])
        if upright:
            low, high = self._reference_contact_height
            moving_height = self.collision_reference[:, 2]
        else:
            moving_height = self.collision_reference@transform[2, :3]+transform[2, 3]
            low, high = np.quantile(self.contact_reference@transform[2, :3]+transform[2, 3], [.01, .99])
        local_height = (moving_height >= low-.015) & (moving_height <= high+.015)
        collisions, exemptions = [], 0
        margins = {}
        guard_margin = float(self.config.get('execution_clearance_margin_m', .001))
        if execution and (not np.isfinite(guard_margin) or guard_margin < 0):
            raise ValueError('Execution clearance margin must be finite and nonnegative')
        for volume in volumes:
            if volume.get('fixed'):
                continue
            name = volume['robot_geom_name']
            finger = 'finger' in name
            hand = name.endswith('hand_collision')
            margin = (float(self.config.get('noncontact_tool_clearance_m', 0.)) if finger or hand
                      else float(self.config.get('observed_clearance_margin_m', .001)))
            if execution:
                margin = min(margin, guard_margin)
            margins[name] = margin
            reference_volume = _transform_volume(volume, inverse)
            for tag, points, tree, query_volume in [('static', self.static_points, self.static_tree, volume),
                                      ('moving', self.collision_reference, self.collision_tree, reference_volume)]:
                if tag == 'moving':
                    ids, distance = _bounded_reference_points(points, tree, volume, query_volume,
                                                              transform, max(.001, margin))
                else:
                    ids, distance = bounded_points(points, tree, query_volume, max(.001, margin))
                if not len(ids):
                    continue
                hit = points[ids]
                allowed = np.zeros(len(hit), bool)
                if tag == 'moving' and contact and (finger or hand):
                    selected_distance = self._reference_contact_distance[ids]
                    if finger or self.config.get('allow_palm_handle_contact', True):
                        allowed |= selected_distance < float(self.config.get('selected_handle_contact_radius_m', .028))
                    brace_radius = .065 if finger else self.tool_panel_radius
                    if finger or (hand and self.spec.get('motion') == 'close'
                                  and self.config.get('allow_palm_panel_brace', True)):
                        allowed |= ((selected_distance < brace_radius)
                                    & (self._reference_panel_distance[ids] < self.panel_band)
                                    & local_height[ids])
                    maximum_depth = self.config.get('maximum_target_contact_penetration_m',
                        .003 if self.config.get('complete_planar_panel', False) else None)
                    if maximum_depth is not None:
                        if not np.isfinite(maximum_depth) or maximum_depth < 0:
                            raise ValueError('Target contact penetration limit must be finite and nonnegative')
                        maintained_finger_contact = np.zeros(len(hit), bool)
                        assembly_allowed = (self.config.get('allow_finger_handle_contact', False)
                                            and 'finger' in name)
                        pad_allowed = (self.config.get('allow_pad_handle_contact', False)
                                       and name.endswith('_pad_collision'))
                        if assembly_allowed or pad_allowed:
                            # A maintained grasp is an explicit permitted pair:
                            # configured finger assembly (or narrower pads-only
                            # legacy option) against the selected measured bar.
                            # Inferred panel, palm, other moving surfaces and
                            # stationary geometry never use this exemption.
                            maintained_finger_contact = ((ids < len(self.moving_reference))
                                                         & (selected_distance <= .006))
                        allowed &= (distance >= -float(maximum_depth)) | maintained_finger_contact
                blocked = (distance <= margin) & ~allowed
                exemptions += int(allowed.sum())
                if blocked.any():
                    examples = hit[blocked][:3]
                    if tag == 'moving':
                        examples = examples@transform[:3, :3].T+transform[:3, 3]
                    collisions.append({'robot_geom_name': name, 'scene_part': tag,
                        'blocked_points': int(blocked.sum()),
                        'blocked_inferred_panel_points': int((blocked&(ids >= len(self.moving_reference))).sum()) if tag == 'moving' else 0,
                        'minimum_signed_point_clearance_m': float(distance[blocked].min()),
                        'example_points_world_m': examples.tolist()})
        return {'accepted': not collisions, 'collisions': collisions,
                'checking_mode': 'execution_guard' if execution else 'planning',
                'clearance_margin_m_by_robot_part': margins,
                'local_contact_exemptions': exemptions, 'theta_rad': float(theta),
                'door_environment_collision_checked': False, 'unknown_space_validated': False,
                'sampled_observed_surfaces_only': not bool(len(self.inferred_panel_reference)),
                'inferred_panel_completion_used': bool(len(self.inferred_panel_reference))}

    def check(self, joints, theta, *, contact=True, execution=False):
        if self.robot is None:
            raise ValueError('Pure configured robot kinematics must be supplied for joint-state checks')
        return self.check_volumes(self.robot.forward(joints), theta, contact=contact, execution=execution)

    def retain_static(self,previous):
        """Occlusion during replanning must not erase a stationary obstacle."""
        self.static_points=voxel_points(np.vstack([previous.static_points,self.static_points]),
                                       float(self.config.get('scene_voxel_spacing_m',.003)))
        self.static_points.setflags(write=False)
        self.static_tree=cKDTree(self.static_points)
        self.static_prior_retained=True

    def record(self):
        return {'schema_version': 'coupled-observed-scene-1.0',
                'static_observed_points': len(self.static_points),
                'moving_observed_points': len(self.moving_reference),
                'inferred_panel_points': len(self.inferred_panel_reference),
                'moving_collision_samples_total': len(self.collision_reference),
                'planar_panel_completion': self.panel_completion,
                'contact_observed_points': len(self.contact_reference),
                'static_observations_immutable_during_planning': True,
                'prior_static_observations_retained':getattr(self,'static_prior_retained',False),
                'contact_permissions_restricted_to_tagged_moving_surface': True,
                'noncontact_tool_clearance_m': float(self.config.get('noncontact_tool_clearance_m', 0.)),
                'noncontact_arm_clearance_m': float(self.config.get('observed_clearance_margin_m', .001)),
                'execution_clearance_margin_m': float(self.config.get('execution_clearance_margin_m', .001)),
                'execution_guard_policy': 'minimum of planning margin and execution clearance; geometry, target pair permissions and penetration limits unchanged',
                'allow_palm_handle_contact': bool(self.config.get('allow_palm_handle_contact', True)),
                'allow_palm_panel_brace': bool(self.config.get('allow_palm_panel_brace', True)),
                'allow_pad_handle_contact': bool(self.config.get('allow_pad_handle_contact', False)),
                'allow_finger_handle_contact': bool(self.config.get('allow_finger_handle_contact', False)),
                'maintained_pad_contact_policy': 'only named pad against selected measured moving surface within 6mm; no inferred-panel or static exemption',
                'maintained_finger_contact_policy': 'when enabled, configured finger body and pad assembly against selected measured moving surface within 6mm; no palm, inferred-panel, nonhandle or static exemption',
                'maximum_target_contact_penetration_m': self.config.get('maximum_target_contact_penetration_m',
                    .003 if self.config.get('complete_planar_panel', False) else None),
                'robot_geometry': 'configured Panda collision volumes; fixed captured gripper aperture',
                'moving_geometry': 'RGB panel/slab seed, connected observed planar support within hinge/handle bounds, selected observed handle',
                'moving_association_ambiguous_coplanar_seams_validated': False,
                'upright_handle_mount_shape_prior': {
                    'source': 'observed vertical bar axis projected onto measured upright panel plane',
                    'ownership_region': 'finite horizontal capsule; measured bar radial q98 plus 4mm',
                    'height_bounds': 'observed bar endpoints plus or minus 15mm',
                    'requires_existing_moving_observed_support_within_m': float(self.config.get('mount_observed_connectivity_m', .006)),
                    'assumptions': 'upright revolute axis, vertical bar, upright face; radius 5-25mm and bar/face gap 1-90mm',
                    'hidden_mount_geometry_completed': False,
                    'unrelated_objects_inside_mount_gap_semantically_disambiguated': False},
                'observed_self_filter': self.self_filter,
                'door_environment_collision_checked': False,
                'hidden_door_volume_validated': False, 'unknown_space_validated': False,
                'continuous_collision_detection': False, 'simulator_scene_geometry_used': False}
