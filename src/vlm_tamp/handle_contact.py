"""Sensor-only handle geometry, arrival guards, and motion-based hinge fitting."""

import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import binary_closing, label
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .rekep_libero.geometry import pose_matrix, matrix_pose, transform_points
from .rekep_libero.mechanism_geometry import project, ray_plane, unit, Mechanism
from .rekep_libero.scene_state_v7 import voxel_points
from .rekep_libero.scene_state_v7 import observed_points, bounded_points
from .rekep_libero.object_motion_v3 import observed_object_cloud
from .rekep_libero.robot_chain_v4 import fixed_orientation_residual
from .robot_contracts import GROUND_FIELDS, RequestValidationError, observed_image_sizes, validate_ground_regions
from .rigid_hinge_tracking import robot_mask
from .part_perception import projected_selection


class HandleFitError(ValueError):
    """The selected contact does not fit a bar; door geometry may still be usable."""


def upright_plane(points, camera_position, seed=0):
    """Fit a vertical face, explicitly excluding horizontal table/top surfaces."""
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(1)]
    rng = np.random.default_rng(seed)
    sampled = points[rng.choice(len(points), min(7000, len(points)), replace=False)]
    best = None
    for _ in range(500):
        a, b = sampled[rng.choice(len(sampled), 2, replace=False)]
        delta = b[:2]-a[:2]
        if np.linalg.norm(delta) < .04:
            continue
        normal = unit(np.r_[-delta[1], delta[0], 0.])
        distances = np.abs((sampled-a)@normal)
        keep = distances < .003
        selected = sampled[keep]
        if len(selected) < 100:
            continue
        height = np.ptp(np.quantile(selected[:, 2], [.05, .95]))
        if height < .055:
            continue
        score = int(keep.sum())*min(height/.10, 1.)
        if best is None or score > best[0]:
            best = score, a, normal
    if best is None:
        raise ValueError('RGB panel box has no supported upright plane')
    _, centre, normal = best
    keep = np.abs((points-centre)@normal) < .004
    selected = points[keep]
    centre = selected.mean(0)
    _, _, vt = np.linalg.svd(selected[:, :2]-centre[:2], full_matrices=False)
    normal = unit(np.r_[vt[-1], 0.])
    if normal@(np.asarray(camera_position)-centre) < 0:
        normal *= -1
    keep = np.abs((points-centre)@normal) < .004
    return {'point': centre, 'normal': normal, 'inliers': keep,
            'rms': float(np.sqrt(np.mean(((points[keep]-centre)@normal)**2)))}


def image_motion_axis(anchor, pivot, camera, hint):
    """Resolve an upright axis sign only when the projected hint distinguishes it."""
    if hint is None:
        return None, []
    direction=np.asarray(hint,float)
    if not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError('A supplied image-motion hint must be finite and nonzero')
    # Scale before normalizing so any finite, nonzero magnitude has the same
    # meaning, including subnormal or near-maximum floating-point inputs.
    direction=direction/np.max(np.abs(direction))
    direction=direction/np.linalg.norm(direction)
    scores=[]
    for sign in [1.,-1.]:
        moved=pivot+Rotation.from_rotvec([0.,0.,.1*sign]).apply(anchor-pivot)
        delta=project([moved],camera)[0]-project([anchor],camera)[0]
        scores.append(float(delta@direction/max(np.linalg.norm(delta),1e-8)))
    if abs(scores[0]-scores[1])<=1e-8:
        return None,scores
    return np.array([0.,0.,1. if scores[0]>scores[1] else -1.]),scores


def direction_from_image_hint(anchor, pivot, camera, hint):
    """Use caller intent; an unsigned vertical axis is only a coordinate convention."""
    axis,scores=image_motion_axis(anchor,pivot,camera,hint)
    resolved=axis is not None
    if not resolved:axis=np.array([0.,0.,1.])
    return axis,{'source':'caller image-motion hint' if resolved else 'unsigned vertical-axis coordinate convention',
        'image_direction_xy':hint,'image_hint_status':'resolved' if resolved else 'unresolved',
        'image_projection_scores':scores,'vlm_image_axis':axis.tolist() if resolved else None,
        'selected_axis':axis.tolist(),'task_direction_status':'caller_specified' if resolved else 'unknown',
        'coordinate_convention_only':not resolved,'task_direction_verified':False,
        'positive_rotation_meaning':'caller image-motion direction' if resolved else
            'right-hand rotation about +Z; opening/closing is unknown'}


def selected_panel(camera, choices, panel_mask=None):
    """Fit only the caller-selected upright surface and retain its measured extent."""
    world = np.asarray(camera['world_points'])
    yy, xx = np.indices(world.shape[:2])
    x0, y0, x1, y1 = choices['moving_panel_box_xyxy']
    box = (xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)&np.isfinite(world).all(2)
    if panel_mask is not None:
        panel_mask = np.asarray(panel_mask, bool)
        if panel_mask.shape != box.shape:
            raise ValueError('Selected part mask must match its current camera image')
        box &= panel_mask
    excluded = int((box&robot_mask(camera)).sum())
    box &= ~robot_mask(camera)
    if box.sum()<100:
        raise ValueError(f'Panel fit has only {int(box.sum())} non-robot depth samples after excluding {excluded} robot samples; inspect another visible view')
    face = upright_plane(world[box], np.asarray(camera['camera_to_world'])[:3, 3])
    face['selected_mask'] = box
    mask = box & (np.abs((world-face['point'])@face['normal'])<.006)
    # Use the existing upright-face height quantiles, measured on its inliers.
    # A rectangle is a projection frustum: tabletop points below the door are
    # not handle observations even when they project inside that rectangle.
    face['height_bounds_m'] = np.quantile(world[box][face['inliers'], 2], [.05, .95])
    return face, mask, excluded


def ground_handle(cameras, choices, previous_spec, metadata=None,*,fixed_joint=None,
                  panel_mask=None, contact_mask=None):
    if fixed_joint is None:
        validate_ground_regions({key:choices[key] for key in GROUND_FIELDS if key in choices},
                                image_sizes=observed_image_sizes(cameras))
    else:
        from .robot_contracts import validate_operation
        validate_operation('refresh_contact',{key:choices[key] for key in
            ['camera','contact_box_xyxy','moving_panel_box_xyxy']},observed_image_sizes(cameras))
    hint = choices.get('initial_image_direction_xy')
    if hint is None and metadata is None and fixed_joint is None:
        raise RequestValidationError('ground','regions.initial_image_direction_xy','unresolved_direction',
            'The image direction is unresolved and adjacent stationary depth geometry is unavailable. Select another view or supply a supported image-motion hint.')
    camera = cameras[choices['camera']]
    face, mask, excluded_panel_robot_points = selected_panel(camera, choices, panel_mask)
    centre, normal = face['point'], face['normal']
    selection_mask = contact_mask if contact_mask is not None else panel_mask
    clouds, opposite = [], []
    for view in cameras.values():
        points = np.asarray(view['world_points'])[~robot_mask(view)].reshape(-1, 3)
        points = points[np.isfinite(points).all(1)]
        signed = (points-centre)@normal
        points = points[(signed>-.09)&(signed<.075)]
        keep = projected_selection(points, camera, choices['contact_box_xyxy'], selection_mask)
        selected = points[keep]
        distances = (selected-centre)@normal
        clouds.append(selected[(distances>.008)&(distances<.075)])
        opposite.append(selected[(distances<-.035)&(distances>-.09)])
    cloud = voxel_points(np.vstack(clouds), .0015)
    lo, hi = face['height_bounds_m']
    before_height = len(cloud)
    cloud = cloud[(cloud[:, 2]>=lo)&(cloud[:, 2]<=hi)]
    opposite_side = False
    if len(cloud)<100:
        cloud = voxel_points(np.vstack(opposite), .0015)
        before_height = len(cloud)
        cloud = cloud[(cloud[:, 2]>=lo)&(cloud[:, 2]<=hi)]
        normal = -normal
        opposite_side = True
    if len(cloud) < 100:
        raise HandleFitError('Selected handle region has insufficient protruding depth support within the observed door height')
    # Avoid the mounting brackets and lower panel/table boundary. A frontal
    # approach to the upper central handle also keeps the wrist above the table.
    low, high = np.quantile(cloud[:, 2], [.12, .92])
    middle = cloud[(cloud[:, 2]>=low+.25*(high-low))&(cloud[:, 2]<=high-.10*(high-low))]
    if len(middle)<3:
        raise HandleFitError('Selected contact has insufficient central vertical samples for a cylinder fit')
    centre_xy = np.median(middle[:, :2], axis=0)
    initial = np.r_[centre_xy, .01]
    fit = least_squares(lambda x: np.linalg.norm(middle[:, :2]-x[:2], axis=1)-x[2],
                        initial, bounds=(np.r_[centre_xy-.02, .005], np.r_[centre_xy+.02, .018]),
                        loss='soft_l1', f_scale=.0015, max_nfev=100)
    radius = float(fit.x[2])
    distances = np.abs(np.linalg.norm(cloud[:, :2]-fit.x[:2], axis=1)-radius)
    cylinder = cloud[distances<.004]
    if len(cylinder) < 75:
        raise HandleFitError('Handle cylinder has insufficient observed support')
    occupied=None
    initial_cylinder_points=len(cylinder)
    if metadata is not None:
        occupied,_=observed_points(cameras,metadata,spacing=.002,robot_depth_tolerance=.0001)
        radial=np.abs(np.linalg.norm(occupied[:,:2]-fit.x[:2],axis=1)-radius)
        extension=((radial<.004)&(occupied[:,2]>cylinder[:,2].min()-.025)
                   &(occupied[:,2]<cylinder[:,2].max()+.14)
                   &(occupied[:,2]>=lo)&(occupied[:,2]<=hi))
        if selection_mask is not None:
            extension &= projected_selection(occupied, camera, choices['contact_box_xyxy'], selection_mask)
        cylinder=voxel_points(np.vstack([cylinder,occupied[extension]]),.0015)
    bins = np.floor((cylinder[:, 2]-cylinder[:, 2].min())/.004).astype(int)
    histogram = np.bincount(bins)
    supported = binary_closing(histogram>=5, iterations=1)
    groups, number = label(supported)
    if number:
        longest = max(range(1, number+1), key=lambda k: int((groups==k).sum()))
        selected = groups[bins]==longest
        if selected.sum()>=75 and np.ptp(cylinder[selected, 2])>.035:
            cylinder = cylinder[selected]
    anchors = [np.r_[fit.x[:2], np.quantile(cylinder[:, 2], fraction)] for fraction in [.85, .70, .93]]
    residual = float(np.median(distances))
    if np.ptp(cylinder[:, 2])<=.065 or residual>=.0035:
        error=HandleFitError('Selected contact does not support the vertical-handle model')
        error.planning_diagnostic={'status':'contact_region_unresolved','module':'handle_contact.fit',
            'selected_region':choices['contact_box_xyxy'],'height_m':float(np.ptp(cylinder[:,2])),
            'radial_residual_m':residual,'allowed_radial_residual_m':.0035,'minimum_height_m':.065,
            'physical_motion_issued':False,'model_changed':False}
        raise error
    # Finish the selected bar fit before computing a joint direction from it.
    # An invalid contact must not become a misleading direction failure.
    zlo,zhi=np.quantile(cylinder[:,2],[.2,.8])
    central=cylinder[(cylinder[:,2]>=zlo)&(cylinder[:,2]<=zhi)]
    if len(central)<3:raise HandleFitError('Central handle surface has insufficient cylinder samples')
    old=fit.x[:2].copy()
    refined=least_squares(lambda x:np.linalg.norm(central[:,:2]-x[:2],axis=1)-x[2],
        np.r_[old,radius],bounds=(np.r_[old-.02,.005],np.r_[old+.02,.018]),
        loss='soft_l1',f_scale=.0005,max_nfev=150)
    central_error=float(np.median(np.abs(refined.fun)))
    if central_error>.0025:raise HandleFitError('Central handle surface is inconsistent with a cylinder')
    radius=float(refined.x[2]);anchors=[np.r_[refined.x[:2],a[2]] for a in anchors]
    mechanism, direction_record, panel_band, panel_band_measured = door_model(
        cameras, choices, previous_spec, metadata, camera, face, normal, mask,
        cylinder, anchors, fixed_joint=fixed_joint, occupied=occupied)
    return mechanism, cylinder, mask, anchors, {
        'upright_face_normal': normal.tolist(), 'upright_face_rms_m': face['rms'],
        'panel_robot_points_excluded':excluded_panel_robot_points,
        'robot_depth_excluded_from_handle_fit':True,
        'target_selection':{'panel_mask_used':panel_mask is not None,
            'contact_mask_used':contact_mask is not None,
            'observed_panel_height_bounds_m':[float(lo),float(hi)],
            'projected_contact_points_before_height':before_height,
            'selected_contact_points':len(cloud),
            'outside_observed_panel_height_removed':before_height-len(cloud),
            'source':'caller region/mask and measured upright-panel extent; no object geometry'},
        'cylinder_radius_m': radius, 'cylinder_points': len(cylinder),
        'initial_rgb_box_cylinder_points':initial_cylinder_points,
        'metric_cylinder_extended_from_observed_depth':metadata is not None,
        'continuous_cylinder_z_bounds_m':[float(cylinder[:,2].min()),float(cylinder[:,2].max())],
        'cylinder_robust_residual_m': residual,
        'central_bar_refinement':{'old_axis_xy':old.tolist(),'new_axis_xy':refined.x[:2].tolist(),
            'radial_residual_m':central_error,'excluded_height_quantiles':[.2,.8],'simulator_geometry_used':False},
        'handle_z_bounds_m': [float(low), float(high)], 'anchors_world_m': [a.tolist() for a in anchors],
        'handle_on_opposite_observed_face': opposite_side,
        'motion_direction_check': direction_record,
        'panel_surface_half_band_m':panel_band,'panel_surface_band_measured':panel_band_measured,
        'simulator_geometry_used': False}


def ground_panel(cameras, choices, previous_spec, metadata=None, *, fixed_joint=None, panel_mask=None):
    """Retain a door/joint hypothesis independently of the availability of a handle."""
    camera = cameras[choices['camera']]
    face, mask, excluded = selected_panel(camera, choices, panel_mask)
    if panel_mask is not None:
        # Keep nonplanar observed pieces of the explicitly selected part too.
        # The plane describes the joint geometry, not the entire moving shape.
        mask = face['selected_mask']
    cloud = voxel_points(np.asarray(camera['world_points'])[mask], .0015)
    anchor = np.median(cloud, axis=0)
    mechanism, direction, band, measured = door_model(
        cameras, choices, previous_spec, metadata, camera, face, face['normal'], mask,
        cloud, [anchor], fixed_joint=fixed_joint)
    mechanism.spec.update(generic_part_joint_model=True,
        geometry_source='selected observed upright door and annotated joint; no handle fit')
    mechanism.spec.pop('observed_handle_radius_m', None)
    return mechanism, cloud, mask, [anchor], {
        'upright_face_normal':face['normal'].tolist(),'upright_face_rms_m':face['rms'],
        'panel_robot_points_excluded':excluded,'motion_direction_check':direction,
        'target_selection':{'panel_mask_used':panel_mask is not None,
            'observed_panel_height_bounds_m':face['height_bounds_m'].tolist()},
        'panel_surface_half_band_m':band,'panel_surface_band_measured':measured,
        'handle_model_available':False,'simulator_geometry_used':False}


def door_model(cameras, choices, previous_spec, metadata, camera, face, normal, mask,
               cylinder, anchors, *, fixed_joint=None, occupied=None):
    """Common joint/direction calculation; the anchor may be on a door or handle."""
    centre = face['point']
    hint = choices.get('initial_image_direction_xy')
    pivot=(np.asarray(fixed_joint['joint_origin_world_m'],float) if fixed_joint is not None else
        np.asarray([ray_plane(p,camera,centre,normal) for p in choices['hinge_pixels_xy']]).mean(0))
    if fixed_joint is not None:
        axis=np.asarray(fixed_joint['joint_axis_world'],float)
        previous=fixed_joint.get('direction_source',{})
        previous=previous if isinstance(previous,dict) else {}
        direction_record={key:copy.deepcopy(previous[key]) for key in
            ['task_direction_status','coordinate_convention_only','positive_rotation_meaning'] if key in previous}
        direction_record.setdefault('task_direction_status','unknown')
        direction_record.setdefault('coordinate_convention_only',direction_record['task_direction_status']=='unknown')
        direction_record.update(source='retained fixed-world joint hypothesis; current contact geometry refreshed',
            selected_axis=axis.tolist(),image_direction_xy=hint,image_hint_status='not_reestimated',
            task_direction_verified=False,fixed_joint_preserved=True)
    else:
        axis,direction_record=direction_from_image_hint(anchors[0],pivot,camera,hint)
    panel_band=.032
    panel_band_measured=False
    if metadata is not None:
        if occupied is None:
            occupied,_=observed_points(cameras,metadata,spacing=.002,robot_depth_tolerance=.0001)
        height = np.quantile(cylinder[:, 2], [.05, .95])
        signed=(occupied-centre)@normal
        near=(np.linalg.norm(occupied-anchors[0],axis=1)<.12)&(np.abs(signed)<.04)
        near &= (occupied[:,2]>height[0]-.005)&(occupied[:,2]<height[1]+.005)
        near &= cKDTree(cylinder).query(occupied)[0]>.020
        if near.sum()>100:
            panel_band=float(np.clip(np.quantile(np.abs(signed[near]),.98)+.006,.012,.040))
            panel_band_measured=True
    spec = copy.deepcopy(previous_spec)
    spec.update(joint_origin_world_m=pivot.tolist(), joint_axis_world=axis.tolist(),
                panel_normal_world=normal.tolist(), pull_axis_world=normal.tolist(),
                source_anchor_world_m=anchors[0].tolist(),
                front_plane={'point_world_m': centre.tolist(), 'normal_world': normal.tolist(),
                             'rms_m': face['rms'], 'inliers': int(mask.sum())},
                geometry_source='RGB boxes and metric depth; upright face and vertical handle cylinder',
                direction_source=direction_record,
                panel_depth_association_m=.032,
                measured_panel_surface_half_band_m=panel_band,
                simulator_scene_state_used=False)
    return Mechanism(spec), direction_record, panel_band, panel_band_measured


def frontal_poses(normal, anchors, pad_local):
    """Observed-contact proposals spanning frontal and oblique approaches.

    These are search seeds, not a whitelist of admissible grasps. Every pose
    still needs IK, collision checking and a complete contact-motion path.
    """
    poses = []
    z = -unit(normal)
    y = np.array([0., 0., 1.])
    x = unit(np.cross(y, z))
    y = unit(np.cross(z, x))
    reference = np.column_stack([x, y, z])
    for anchor_index, anchor in enumerate(anchors):
        choices=[(0.,t) for t in [0.,-.20,.20,-.50,.50]]
        choices += [(yaw,tilt) for yaw in [-np.pi/6,np.pi/6,-np.pi/3,np.pi/3]
                    for tilt in [0.,-.20,.20]]
        for yaw,tilt in choices:
            rotation = Rotation.from_rotvec([0.,0.,yaw]).as_matrix()@Rotation.from_rotvec(tilt*x).as_matrix()@reference
            for flip in [False, True]:
                matrix = np.eye(4)
                matrix[:3, :3] = rotation@np.diag([-1., -1., 1.]) if flip else rotation
                matrix[:3, 3] = anchor-matrix[:3, :3]@pad_local
                poses.append({'pose': matrix_pose(matrix), 'source': 'RGB-D observed handle proposal',
                              'orientation_family':'frontal' if yaw==0 else 'oblique',
                              'anchor_index': anchor_index, 'tilt_rad': tilt,
                              'yaw_rad':yaw,'half_turn': flip})
    return poses


def arrival_error(requested, measured):
    a, b = pose_matrix(requested), pose_matrix(measured)
    return {'position_error_m': float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
            'orientation_error_rad': float(Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).magnitude())}


def require_arrival(requested, measured, position=.006, orientation=.08):
    error = arrival_error(requested, measured)
    if error['position_error_m']>position or error['orientation_error_rad']>orientation:
        raise ValueError(f'Arrival not verified; refusing closure: {error}')
    return error


def contact_pose_variants(target, measured, handle_axis):
    """Keep the contact target fixed while exploring bounded wrist rotations."""
    target, measured = np.asarray(target, float), np.asarray(measured, float)
    axis = unit(handle_axis)
    basis = pose_matrix(measured)[:3, :3]
    nominal = Rotation.from_quat(target[3:])
    offsets = [('handle_axis', value, axis) for value in [0., -.15, .15, -.30, .30]]
    offsets += [(name, value, basis[:, column])
                for name, column in [('tool_x', 0), ('tool_y', 1)]
                for value in [-.10, .10, -.20, .20]]
    variants = []
    for name, value, direction in offsets:
        pose = target.copy()
        pose[3:] = (Rotation.from_rotvec(value*direction)*nominal).as_quat()
        variants.append({'pose_world_xyzw': pose, 'rotation_axis': name, 'offset_rad': value})
    return variants


def fit_upright_hinge(points, minimum_span=.025):
    """Fit XY circle only after sufficient observed motion; report conditioning."""
    points = np.asarray(points, float)
    if len(points)<4 or np.max(np.linalg.norm(points[:, None]-points[None, :], axis=2))<minimum_span:
        return {'accepted': False, 'reason': 'insufficient_observed_motion'}
    xy = points[:, :2]
    a = 2*(xy[1:]-xy[0])
    b = (xy[1:]**2).sum(1)-(xy[0]**2).sum()
    singular = np.linalg.svd(a, compute_uv=False)
    if singular[-1]<1e-4 or singular[0]/singular[-1]>35:
        return {'accepted': False, 'reason': 'short_arc_or_nearly_straight_motion'}
    pivot = np.linalg.lstsq(a, b, rcond=None)[0]
    radii = np.linalg.norm(xy-pivot, axis=1)
    angles = np.unwrap(np.arctan2(xy[:, 1]-pivot[1], xy[:, 0]-pivot[0]))
    span = float(np.ptp(angles))
    rms = float(np.std(radii))
    accepted = .08<float(np.mean(radii))<.60 and span>.20 and rms<.004
    return {'accepted': accepted, 'hinge_world_m': [*pivot.tolist(), float(np.mean(points[:, 2]))],
            'radius_m': float(np.mean(radii)), 'angular_span_rad': span,
            'radial_rms_m': rms, 'condition_number': float(singular[0]/singular[-1]),
            'source': 'observed RGB-D handle centres; upright-axis prior',
            'simulator_geometry_used': False}


def local_panel_band(points, contact, panel, reach, fallback):
    """Measure current local panel support without absorbing distant surfaces."""
    points,contact=np.asarray(points),np.asarray(contact)
    origin,normal=panel
    signed=np.abs((points-origin)@normal)
    distance=cKDTree(contact).query(points)[0]
    low,high=np.quantile(contact[:,2],[.05,.95])
    selected=(distance>.020)&(distance<reach)&(signed<.040)
    selected&=(points[:,2]>low-.005)&(points[:,2]<high+.005)
    count=int(selected.sum())
    if count<100:
        return float(fallback),{'measured':False,'support_points':count}
    band=float(np.clip(np.quantile(signed[selected],.98)+.006,.012,.040))
    return band,{'measured':True,'support_points':count,'half_band_m':band,
                 'source':'current observed depth near tracked handle and predicted panel'}


class ObservedRobotScene:
    """Measured surfaces and local, phase-specific tool contacts; no door hull."""
    def __init__(self, cameras, metadata, contact_cloud, config, *, panel=None,
                 allow_tool_contact=False, panel_half_band_m=.010, tool_panel_contact=False):
        if any(view.get('robot_depth_prior') and view.get('known_robot_depth_prior') is None
               for view in cameras.values()):
            raise ValueError('Configured robot depth prior was not attached by the sensor loader')
        self.points, self.filter = observed_points(cameras, metadata,
            spacing=config['scene_voxel_spacing_m'], robot_depth_tolerance=config['robot_self_depth_tolerance_m'])
        self.tree = cKDTree(self.points)
        self.contact = np.asarray(contact_cloud)
        self.contact_tree = cKDTree(self.contact)
        self.required = {v['robot_geom_name'] for v in metadata['robot_collision_volumes_world'] if not v.get('fixed')}
        self.config, self.events, self.cache = config, [], {}
        self.panel, self.allow_tool_contact = panel, allow_tool_contact
        self.panel_half_band_m=panel_half_band_m
        self.tool_panel_contact=bool(tool_panel_contact)
        self.tool_panel_brace_radius=0.
        self.panel_band_evidence={'measured':False,'source':'supplied initial panel band'}
        if self.tool_panel_contact:
            volumes=metadata['robot_collision_volumes_world']
            pads=[np.asarray(v['vertices_world']).mean(0) for v in volumes
                  if v['robot_geom_name'].endswith('_pad_collision')]
            hands=[np.asarray(v['vertices_world']) for v in volumes
                   if v['robot_geom_name'].endswith('hand_collision')]
            if len(pads)!=2 or len(hands)!=1:
                raise ValueError('Panel bracing requires the configured hand and both finger pads')
            self.tool_panel_brace_radius=float(np.linalg.norm(hands[0]-np.mean(pads,axis=0),axis=1).max()+.003)
            if self.panel is not None:
                self.panel_half_band_m,self.panel_band_evidence=local_panel_band(
                    self.points,self.contact,self.panel,self.tool_panel_brace_radius,self.panel_half_band_m)

    def check(self, volumes, pose=None, *, contact=False):
        if not self.required.issubset({v['robot_geom_name'] for v in volumes}):
            raise ValueError('Full robot collision geometry is required')
        collisions, exemptions = [], 0
        for volume in volumes:
            if volume.get('fixed'):
                continue
            name = volume['robot_geom_name']
            finger = 'finger' in name
            captured_hand=self.allow_tool_contact and name.endswith('hand_collision')
            margin = 0. if finger or captured_hand else .001
            ids, distance = bounded_points(self.points, self.tree, volume, .001)
            if not len(ids):
                continue
            hit = self.points[ids]
            selected_distance = self.contact_tree.query(hit)[0]
            touching_tool = finger or (self.allow_tool_contact and name.endswith('hand_collision'))
            allowed = (selected_distance<.028) if contact and touching_tool else np.zeros(len(hit), bool)
            hand_brace=self.tool_panel_contact and name.endswith('hand_collision')
            suspected=np.zeros(len(hit),bool)
            if contact and (finger or hand_brace) and self.panel is not None:
                origin, normal = self.panel
                reach=.065 if finger else self.tool_panel_brace_radius
                plane_distance=np.abs((hit-origin)@normal)
                brace = (selected_distance<reach)&(plane_distance<self.panel_half_band_m)
                suspected=(selected_distance<reach+.010)&(plane_distance<self.panel_half_band_m+.008)
                low,high=np.quantile(self.contact[:,2],[.01,.99])
                local_height=(hit[:,2]>=low-.015)&(hit[:,2]<=high+.015)
                brace&=local_height
                suspected&=local_height
                allowed |= brace
            blocked = (distance<=margin)&~allowed
            exemptions += int(allowed.sum())
            if blocked.any():
                collisions.append({'robot_geom_name': name, 'blocked_points': int(blocked.sum()),
                    'minimum_signed_clearance_m':float(distance[blocked].min()),
                    'maximum_point_penetration_m': float(max(0., -distance[blocked].min())),
                    'suspected_moving_panel_overlap':bool(suspected[blocked].all()),
                    'example_points_world_m': hit[blocked][:3].tolist()})
        result = {'accepted': not collisions, 'collisions': collisions,
                  'local_contact_exemptions': exemptions, 'unknown_space_validated': False,
                  'tool_panel_contact':self.tool_panel_contact,
                  'panel_surface_half_band_m':self.panel_half_band_m,
                  'panel_band_evidence':getattr(self,'panel_band_evidence',None),
                  'predicted_door_hull_used': False}
        return result

    def joints(self, sim, q, aperture, contact=False):
        key = (*np.round(q, 6), *np.round(aperture, 5), contact)
        if key not in self.cache:
            state = sim.fk(q, include_volumes=True, gripper_qpos=aperture)
            self.cache[key] = self.check(state['robot_collision_volumes_world'], contact=contact)
        return self.cache[key]


def translated_surface(reference, cameras, volumes, expected):
    """Track an upright handle's position without inferring a pinch or weld."""
    reference = np.asarray(reference)
    expected = np.asarray(expected)
    predicted = np.eye(4)
    predicted[:3, 3] = expected
    observed = observed_object_cloud(cameras, reference, predicted, volumes)
    if len(observed)<75:
        return {'accepted': False, 'reason': 'insufficient_visible_handle_points'}
    tree = cKDTree(observed)
    translation = expected.copy()
    for _ in range(30):
        moved = reference+translation
        distances, indices = tree.query(moved)
        keep = distances<.014
        if keep.sum()<60:
            return {'accepted': False, 'reason': 'insufficient_handle_correspondences'}
        correction = np.median(observed[indices[keep]]-moved[keep], axis=0)
        translation += correction
        if np.linalg.norm(correction)<2e-5:
            break
    distances = tree.query(reference+translation)[0]
    coverage, p75 = float(np.mean(distances<.006)), float(np.quantile(distances, .75))
    return {'accepted': coverage>.6 and p75<.009, 'translation_world_m': translation.tolist(),
            'coverage_6mm': coverage, 'percentile75_m': p75, 'observed_points': len(observed),
            'simulator_geometry_used': False, 'source': 'RGB-D translation fit of the observed upright handle'}


def local_contact_ik(ik, target, seed):
    """Position priority, bounded orientation compliance, and joint centering."""
    target = pose_matrix(target)
    seed = np.asarray(seed, float)
    limits = ik.chain.limits
    low = np.maximum(limits[:, 0]+.08, seed-.40)
    high = np.minimum(limits[:, 1]-.08, seed+.40)
    if np.any(low>=high):
        return None
    middle = limits.mean(1)
    span = limits[:, 1]-limits[:, 0]
    cache, residual, jacobian = None, None, None
    def both(q):
        nonlocal cache, residual, jacobian
        if cache is None or not np.array_equal(q, cache):
            pose, jac = ik.chain.forward(q, jacobian=True)
            matrix = pose_matrix(pose)
            error, rotation_jac = fixed_orientation_residual(target[:3, :3], matrix[:3, :3], jac[3:])
            residual = np.r_[matrix[:3, 3]-target[:3, 3], .012*error,
                             .006*(q-middle)/span, .0002*(q-seed)]
            jacobian = np.vstack([jac[:3], .012*rotation_jac,
                                  np.diag(.006/span), np.eye(7)*.0002])
            cache=q.copy()
        return residual, jacobian
    result=least_squares(lambda q:both(q)[0],np.clip(seed,low+1e-6,high-1e-6),
        jac=lambda q:both(q)[1],bounds=(low,high),max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    pose=ik.chain.forward(result.x)
    error=arrival_error(target,pose)
    if error['position_error_m']>.0025 or error['orientation_error_rad']>.25 or not ik.self_valid(result.x):
        return None
    return {'joints':result.x.tolist(),'pose_world_xyzw':pose.tolist(),
            'joint_reserve_rad':float(np.minimum(result.x-limits[:,0],limits[:,1]-result.x).min()),
            'backend':'bounded position-priority contact IK','preferred_orientation_error_rad':error['orientation_error_rad']}
