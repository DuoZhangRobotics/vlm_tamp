"""Fit VLM-selected geometric hypotheses to measured RGB-D, without task IDs.

The fitting library supports arbitrary 3-D planes and straight bars. Executor
capabilities are declared separately; a successful fit does not imply execution.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .interaction_contract import InteractionError, backend_for
from .rekep_libero.mechanism_geometry import project
from .rekep_libero.scene_state_v7 import voxel_points


@dataclass
class GroundedInteraction:
    selection: dict
    cameras: dict
    bundle: dict
    spec: dict
    cloud: np.ndarray
    source_mask: np.ndarray
    panel_mask: np.ndarray
    target: object
    target_mask: object
    evidence: dict


def unit(a):
    a = np.asarray(a, float)
    if a.shape != (3,) or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-10:
        raise InteractionError('Finite nonzero spatial direction required')
    return a / np.linalg.norm(a)


def box_mask(camera, box):
    yy, xx = np.indices(camera['world_points'].shape[:2])
    a, b, c, d = box
    return (xx >= a) & (xx <= c) & (yy >= b) & (yy <= d) & np.isfinite(camera['world_points']).all(2)


def fit_plane(points, *, camera_position=None, resolution=.003, seed=0):
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    if len(points) < 30:
        raise InteractionError('Selected plane has fewer than 30 measured points')
    sample = points[np.linspace(0, len(points)-1, min(len(points), 4000), dtype=int)]
    rng = np.random.default_rng(seed); best = None
    for _ in range(128):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        good = np.abs((sample-a) @ normal) <= resolution
        count = int(good.sum())
        if best is None or count > best[0]:
            best = count, good
    if best is None or best[0] < max(30, len(sample)*.35):
        raise InteractionError('Selected region does not support a dominant plane')
    part = sample[best[1]]
    for _ in range(3):
        centre = np.mean(part, axis=0)
        _, s, vt = np.linalg.svd(part-centre, full_matrices=False)
        if s[1] < resolution:
            raise InteractionError('Plane evidence is degenerate')
        normal = vt[-1]
        keep = np.abs((points-centre) @ normal) <= resolution
        part = points[keep]
    if camera_position is not None and normal @ (np.asarray(camera_position)-centre) < 0:
        normal = -normal
    residual = (part-centre) @ normal
    sigma = float(max(np.sqrt(np.mean(residual**2)), resolution/np.sqrt(12)))
    return {'point': centre, 'normal': normal, 'inliers': keep, 'rms_m': float(np.sqrt(np.mean(residual**2))),
            'uncertainty_m': sigma, 'points': len(part)}


def pixel_depth(camera, pixels):
    """Sample actual pixels, with a bounded sensor-hole neighborhood only."""
    out = []
    for xy in pixels:
        x, y = np.rint(xy).astype(int)
        point = camera['world_points'][y, x]
        if not np.isfinite(point).all():
            patch = camera['world_points'][max(0,y-2):y+3, max(0,x-2):x+3].reshape(-1,3)
            patch = patch[np.isfinite(patch).all(1)]
            if not len(patch):
                raise InteractionError('Visual reference has no depth support')
            point = np.median(patch, axis=0)
        out.append(point)
    return np.asarray(out)


def ray_on_plane(camera, pixels, plane):
    transform = np.asarray(camera['camera_to_world']); origin = transform[:3, 3]
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(np.asarray(camera['intrinsic'])).T
    rays = rays @ transform[:3, :3].T
    denominator = rays @ plane['normal']
    if np.any(np.abs(denominator) < 1e-5):
        raise InteractionError('Visual axis is parallel to the fitted plane')
    distance = (plane['point']-origin) @ plane['normal'] / denominator
    if (distance <= 0).any():
        raise InteractionError('Visual axis projects behind the camera')
    return origin + distance[:, None]*rays


def fit_bar(points, endpoints=None, *, resolution=.003):
    """Robust cylinder fit, including its 3-D axis; no world-up or size prior."""
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) < 30 or not np.isfinite(points).all():
        raise InteractionError('Bar requires finite measured support')
    centre = np.median(points, axis=0)
    _, s, vt = np.linalg.svd(points-centre, full_matrices=False)
    axis = vt[0]
    if endpoints is not None:
        direction = unit(np.asarray(endpoints)[1]-np.asarray(endpoints)[0])
        if abs(axis @ direction) < .7:
            raise InteractionError('Selected bar endpoints disagree with measured long axis')
        if axis @ direction < 0: axis = -axis
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    b = unit(np.cross(axis, helper)); c = np.cross(axis, b)
    length = np.quantile((points-centre) @ axis, .95) - np.quantile((points-centre) @ axis, .05)
    radial = np.linalg.norm((points-centre) @ np.column_stack([b,c]), axis=1)
    scale = max(float(np.quantile(radial, .95)), resolution)
    # Remove the end regions, which may contain attachment mounts.
    low, high = np.quantile((points-centre) @ axis, [.15,.85])
    central = points[((points-centre) @ axis >= low) & ((points-centre) @ axis <= high)]
    def decode(x):
        a = unit(axis + x[2]*b + x[3]*c)
        origin = centre + x[0]*b + x[1]*c
        return origin, a, x[4]
    def residual(x):
        origin,a,radius = decode(x); delta = central-origin
        return np.linalg.norm(delta - np.outer(delta @ a,a), axis=1)-radius
    fitted = least_squares(residual, [0,0,0,0,scale],
        bounds=([-3*scale,-3*scale,-.5,-.5,resolution/10], [3*scale,3*scale,.5,.5,max(length,scale*4)]),
        loss='soft_l1', f_scale=resolution/3, max_nfev=150)
    origin, axis, radius = decode(fitted.x)
    error = float(np.median(np.abs(fitted.fun)))
    axial = (points-origin) @ axis; lo,hi = np.quantile(axial, [.02,.98])
    if error > resolution or hi-lo < 2*radius or radius >= (hi-lo)/2:
        raise InteractionError('Selected region does not support an elongated cylindrical bar')
    return {'origin_world_m': origin, 'axis_world': axis, 'radius_m': float(radius),
            'length_m': float(hi-lo), 'endpoints_world_m': np.array([origin+lo*axis,origin+hi*axis]),
            'residual_m': error, 'uncertainty_m': max(error,resolution/np.sqrt(12)),
            'world_up_prior': False, 'fixed_radius_bounds_used': False}


def fit_control(points, support, *, resolution=.003):
    """Select a protruding control relative to its VLM-selected support plane."""
    normal = support['normal']; points = np.asarray(points)
    heights = (points-support['point']) @ normal
    selected = points[heights > max(resolution, 3*support['uncertainty_m'])]
    if len(selected) < 30:
        raise InteractionError('Control has no measured protrusion above selected support')
    helper = np.eye(3)[np.argmin(np.abs(normal))]; b = unit(np.cross(normal,helper)); c = np.cross(normal,b)
    basis = np.column_stack([b,c]); uv = (selected-support['point']) @ basis
    lo,hi = np.quantile(uv,[.02,.98],axis=0)
    pivot = support['point'] + basis @ ((lo+hi)/2)
    return {'origin_world_m':pivot,'axis_world':normal,'footprint_extent_m':hi-lo,
            'uncertainty_m':support['uncertainty_m'],'measured_protrusion_used':True},selected


def align_planes(moving, fixed, axis, family):
    """Compute a goal from two explicitly selected observed planes."""
    axis=unit(axis); n=unit(moving['normal']); t=unit(fixed['normal'])
    if n @ t < 0: t=-t
    uncertainty=float(np.hypot(moving['uncertainty_m'], fixed['uncertainty_m']))
    if family=='prismatic':
        if abs(n@t)<np.cos(np.deg2rad(10)) or abs(t@axis)<.8:
            raise InteractionError('Closing reference is not parallel to the sliding front')
        progress=float((fixed['point']-moving['point'])@t/(axis@t))
    else:
        n=unit(n-axis*(n@axis));t=unit(t-axis*(t@axis))
        # Plane sign is unobservable; choose the forward angle closest to zero.
        options=[float(np.arctan2(axis@np.cross(n,s*t),n@(s*t))) for s in [1,-1]]
        positive=[x for x in options if x>=-1e-6]
        if not positive:raise InteractionError('Reference plane disagrees with selected rotation direction')
        progress=min(positive)
    if progress < -uncertainty:
        raise InteractionError('Goal plane lies opposite the chosen motion')
    return max(0.,progress),t,uncertainty


def robot_policy(metadata, resolution=.003):
    """Robot/sensor constants are shared by every task and never VLM supplied."""
    calibration=metadata['robot_calibration']
    points=np.asarray(calibration['tool_collision_points_tcp'],float)
    hand_span=float(np.max(np.ptp(points,axis=0)))
    open_width=float(np.sum(np.abs(metadata['gripper_qpos'])))
    return {'hand_clearance_m':max(hand_span,open_width), 'depth_resolution_m':resolution,
            'completion_uncertainty_multiplier':2.,'maximum_completion_m':open_width/8,
            'contact_association_m':2*resolution,'policy_source':'configured robot geometry and depth sampling'}


def fit_interaction(selection, cameras, metadata, bundle, perception, output):
    from pathlib import Path
    from .coupled_storage import write_json
    from .rekep_libero.task_geometry_v4 import surface_summary
    backend=backend_for(selection);output=Path(output);camera=cameras[selection['camera']]
    policy=robot_policy(metadata);resolution=policy['depth_resolution_m']
    contact=selection['contact'];moving=selection['moving'];mechanism=selection['mechanism'];goal=selection['goal']
    source_mask=box_mask(camera,moving['box_xyxy'] if backend=='placement' else contact['box_xyxy'])
    # SAM chooses the visible object containing the VLM point; no simulator masks.
    pixel=np.rint(contact['pixel_xy']).astype(int);point=pixel_depth(camera,[pixel])[0]
    extended={**bundle,'points_world':np.vstack([bundle['points_world'],point]),
              'pixels_yx':np.vstack([bundle['pixels_yx'],pixel[::-1]])}
    _,mask,_=perception.selected_cloud(extended,len(bundle['points_world']),articulated=backend!='placement',out_dir=output/'selected-mask')
    source_mask &= mask
    if source_mask.sum()<30:raise InteractionError('Selected contact has insufficient RGB-mask/depth support')
    cloud=camera['world_points'][source_mask]
    panel_mask=box_mask(camera,moving['box_xyxy']);target=None;target_mask=None
    evidence={'version':'interaction-fit-v1','backend':backend,'robot_policy':policy,
              'task_label_used':False,'simulator_geometry_used':False,'selected_shape':contact['shape']}
    if backend=='placement':
        target_mask=box_mask(camera,goal['reference_box_xyxy']) & ~source_mask
        a,b,c,d=goal['reference_box_xyxy'];target_pixel=np.rint([(a+c)/2,(b+d)/2]).astype(int)
        target_point=pixel_depth(camera,[target_pixel])[0]
        target_bundle={**bundle,'points_world':np.vstack([bundle['points_world'],target_point]),
                       'pixels_yx':np.vstack([bundle['pixels_yx'],target_pixel[::-1]])}
        _,target_sam,_=perception.selected_cloud(target_bundle,len(bundle['points_world']),articulated=False,out_dir=output/'support-mask')
        target_mask &= target_sam
        target=camera['world_points'][target_mask]
        plane=fit_plane(target,camera_position=np.asarray(camera['camera_to_world'])[:3,3],resolution=resolution)
        target=target[plane['inliers']]
        obj,plate=surface_summary(cloud),surface_summary(target)
        clearance=2*resolution
        spec={'schema_version':'interaction-grounded-1','task_type':'pick_place','instruction':metadata['instruction'],
              'source_anchor_world_m':point.tolist(),'object_surface':obj,'target_surface':plate,
              'object_support_point_initial_world_m':[*obj['center_xy_m'],obj['low_z_m']],
              'release_clearance_m':clearance,'goal':{'type':'point_target','point_local_m':[0.,0.,0.],
              'target_world_m':[*plate['center_xy_m'],plate['median_z_m']+clearance],
              'state_pose':'estimated_object_support_frame','frame':'world','equality_tolerance_m':4*resolution},
              'geometric_path_constraints':[],'attachment_binding_status':'unresolved_until_observed_lift',
              'simulator_object_state_used':False,'canonical_object_mesh_used':False,'geometry_source':'fresh VLM roles and current measured depth'}
        evidence['support_plane']={k:v for k,v in plane.items() if k!='inliers'}
    else:
        support_box=mechanism['support_box_xyxy']
        support_mask=box_mask(camera,support_box)
        face=fit_plane(camera['world_points'][support_mask],camera_position=np.asarray(camera['camera_to_world'])[:3,3],resolution=resolution)
        if np.median((cloud-face['point'])@face['normal']) < -3*face['uncertainty_m']:
            face['normal']=-face['normal']
        normal=face['normal'];origin=face['point']
        if moving['shape']=='planar_panel':
            panel_mask &= np.abs((camera['world_points']-origin)@normal)<=2*resolution
        else:
            panel_mask &= mask
        # Collect only measured points projecting into the selected contact box.
        fused=[cloud]
        for view in cameras.values():
            points=np.asarray(view['world_points']).reshape(-1,3);points=points[np.isfinite(points).all(1)]
            px=project(points,camera);a,b,c,d=contact['box_xyxy']
            keep=(px[:,0]>=a)&(px[:,0]<=c)&(px[:,1]>=b)&(px[:,1]<=d)
            points=points[keep]
            near=cKDTree(cloud).query(points)[0] < max(policy['hand_clearance_m']/4,2*resolution)
            fused.append(points[near])
        cloud=voxel_points(np.vstack(fused),resolution/2)
        if contact['shape']=='straight_bar':
            # Support-plane samples are excluded using measured uncertainty.
            heights=(cloud-origin)@normal
            raised=cloud[heights>max(resolution,3*face['uncertainty_m'])]
            if len(raised)<30:raise InteractionError('Selected bar is not separated from its support plane')
            bar=fit_bar(raised,pixel_depth(camera,contact['axis_pixels_xy']),resolution=resolution)
            evidence['contact_model']=bar;cloud=raised
            anchor=bar['origin_world_m']
        elif contact['shape']=='round_control':
            control,cloud=fit_control(cloud,face,resolution=resolution);evidence['contact_model']=control
            anchor=np.median(cloud,axis=0)
        else:
            anchor=np.median(cloud,axis=0);evidence['contact_model']={'kind':contact['shape'],'uncertainty_m':resolution}
        axis_source=mechanism['axis_source'];kind=mechanism['family'];direction=np.asarray(mechanism['initial_image_direction_xy'])
        if axis_source=='hinge_edge':
            endpoints=ray_on_plane(camera,np.asarray(mechanism['axis_pixels_xy']),face)
            axis=unit(endpoints[1]-endpoints[0]);pivot=endpoints.mean(0)
            evidence['hinge_endpoints_world_m']=endpoints
        elif axis_source=='control_face_normal':
            axis=normal.copy()
            axis_fit,_=fit_control(cloud,face,resolution=resolution)
            evidence['control_axis_model']=axis_fit;pivot=axis_fit['origin_world_m']
            axis*= -1 if mechanism['turn_direction_view']=='clockwise' else 1
        else:
            axis=normal.copy();pivot=anchor.copy()
        if axis_source!='control_face_normal':
            scores=[]
            for sign in [1.,-1.]:
                moved=anchor+sign*resolution*axis if kind=='prismatic' else pivot+Rotation.from_rotvec(sign*.05*axis).apply(anchor-pivot)
                delta=project([moved],camera)[0]-project([anchor],camera)[0]
                scores.append(float(delta@direction/max(np.linalg.norm(delta),1e-8)))
            if abs(scores[0]-scores[1])<.05:raise InteractionError('Image direction does not resolve spatial motion sign')
            axis*=1 if scores[0]>scores[1] else -1
        uncertainty=face['uncertainty_m'];terminal_normal=None
        if goal['relation']=='aligned_with_reference_plane':
            reference=fit_plane(camera['world_points'][box_mask(camera,goal['reference_box_xyxy'])],
                                camera_position=np.asarray(camera['camera_to_world'])[:3,3],resolution=resolution)
            progress,terminal_normal,uncertainty=align_planes(face,reference,axis,kind)
            evidence['goal_reference']={k:v for k,v in reference.items() if k!='inliers'}
        elif goal['relation']=='quarter_turn':progress=float(np.pi/2)
        elif goal['relation']=='hand_clearance':progress=policy['hand_clearance_m']
        else:raise InteractionError('Unsupported articulation goal relation')
        allowance=min(policy['maximum_completion_m'],policy['completion_uncertainty_multiplier']*uncertainty)
        radius=float(np.linalg.norm(np.cross(anchor-pivot,axis)))
        if kind=='revolute': allowance=allowance/max(radius,policy['hand_clearance_m'])
        if mechanism['motion']!='close':allowance=0.
        spec={'schema_version':'interaction-grounded-1','task_type':'articulated','instruction':metadata['instruction'],
              'joint_family':kind,'motion':mechanism['motion'],'joint_origin_world_m':pivot.tolist(),'joint_axis_world':axis.tolist(),
              'target_progress':progress+(allowance if kind=='prismatic' else 0.),'nominal_target_progress':progress,
              'progress_units':'meters' if kind=='prismatic' else 'radians','source_anchor_world_m':anchor.tolist(),
              'camera':selection['camera'],'front_plane':{'point_world_m':origin.tolist(),'normal_world':normal.tolist(),'rms_m':face['rms_m']},
              'panel_normal_world':normal.tolist(),'pull_axis_world':normal.tolist(),'panel_depth_association_m':2*resolution,
              'axis_prior':axis_source,'direction_source':'VLM visual references plus calibrated depth',
              'simulator_scene_state_used':False,'geometry_source':'shared interaction geometry fitters',
              'contact_policy':'selected role compiled to functional fingers; robot arm remains checked',
              'constraint':{'type':'point_on_line' if kind=='prismatic' else 'point_on_circle_with_coupled_orientation',
                            'position_tolerance_m':.015,'orientation_tolerance_rad':.10},
              'completion_allowance':allowance,'completion_trigger':max(allowance*2,1e-6),
              'terminal_normal_world':None if terminal_normal is None else terminal_normal.tolist()}
        evidence.update(goal_relation=goal['relation'],nominal_target_progress=progress,completion_allowance=allowance,
                        completion_source='two combined plane-fit standard deviations, capped by robot aperture / 8',
                        goal_uncertainty_m=uncertainty,front_plane={k:v for k,v in face.items() if k!='inliers'})
    spec['interaction_selection']=selection
    spec['selected_contact_roles']=selection['contact_roles']
    write_json(output/'fitted-spec.json',spec);write_json(output/'fit-evidence.json',evidence)
    np.savez_compressed(output/'fitted-surfaces.npz',contact=cloud,source_mask=source_mask,panel_mask=panel_mask,
                        target=np.empty((0,3)) if target is None else target,
                        target_mask=np.zeros_like(source_mask) if target_mask is None else target_mask)
    return GroundedInteraction(selection,cameras,bundle,spec,cloud,source_mask,panel_mask,target,target_mask,evidence)
