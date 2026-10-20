"""Sensor-grounded line and hinge manifolds for articulated manipulation."""

import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from vlm_tamp.drawer_local_plane import _ransac, local_plane_config
from .geometry import matrix_pose, pose_matrix, transform_points
from .object_motion_v3 import observed_object_cloud


def unit(vector):
    vector = np.asarray(vector, float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError('Mechanism axis must be a finite nonzero vector')
    return vector/length


def project(points, camera):
    local = transform_points(points, np.linalg.inv(np.asarray(camera['camera_to_world'])))
    image = local@np.asarray(camera['intrinsic']).T
    return image[:, :2]/image[:, 2:3]


def ray_plane(pixel, camera, origin, normal):
    transform = np.asarray(camera['camera_to_world'])
    ray = transform[:3, :3]@np.linalg.solve(np.asarray(camera['intrinsic']), np.r_[pixel, 1.])
    denominator = float(ray@normal)
    if abs(denominator)<1e-6:
        raise ValueError('Hinge ray is parallel to the observed panel')
    distance = float((origin-transform[:3, 3])@normal/denominator)
    if distance<=0:
        raise ValueError('Hinge projection is behind the camera')
    return transform[:3, 3]+distance*ray


class Mechanism:
    def __init__(self, spec):
        self.spec = spec
        self.kind = spec['joint_family']
        self.axis = unit(spec['joint_axis_world'])
        self.origin = np.asarray(spec['joint_origin_world_m'], float)
        self.goal = float(spec['target_progress'])
        if self.kind not in {'prismatic','revolute'} or not np.isfinite(self.goal) or self.goal<=0:
            raise ValueError('Invalid mechanism manifold')

    def transform(self, progress):
        if not np.isfinite(progress) or progress < -.025 or progress > self.goal+.10:
            raise ValueError('Mechanism progress exceeds the bounded task range')
        result = np.eye(4)
        if self.kind=='prismatic':
            result[:3, 3] = progress*self.axis
        else:
            rotation = Rotation.from_rotvec(progress*self.axis).as_matrix()
            result[:3, :3] = rotation
            result[:3, 3] = self.origin-rotation@self.origin
        return result

    def pose(self, grasp, progress):
        return matrix_pose(self.transform(progress)@pose_matrix(grasp))

    def progress_of_pose(self, grasp, current):
        a,b = pose_matrix(grasp), pose_matrix(current)
        if self.kind=='prismatic':
            return float((b[:3, 3]-a[:3, 3])@self.axis)
        return float(Rotation.from_matrix(b[:3, :3]@a[:3, :3].T).as_rotvec()@self.axis)

    def residual(self, grasp, pose):
        progress = self.progress_of_pose(grasp, pose)
        predicted = pose_matrix(self.pose(grasp, max(-.025, min(progress, self.goal+.10))))
        actual = pose_matrix(pose)
        return {'progress': progress,
            'position_residual_m': float(np.linalg.norm(predicted[:3, 3]-actual[:3, 3])),
            'orientation_residual_rad': float(Rotation.from_matrix(predicted[:3, :3].T@actual[:3, :3]).magnitude()),
            'family': 'point_on_line' if self.kind=='prismatic' else 'point_on_circle_with_coupled_orientation'}

    def path(self, grasp, start=0., end=None):
        end = self.goal if end is None else end
        radius = np.linalg.norm(pose_matrix(grasp)[:3, 3]-self.origin)
        count = max(1, int(np.ceil(abs(end-start)/(.015 if self.kind=='prismatic' else min(.08,.015/max(radius,.02))))))
        return [self.pose(grasp, value) for value in np.linspace(start,end,count+1)]


def ground_geometry(bundle, choices, contact_cloud, contact_mask, perception, output):
    name = choices['camera']
    camera = bundle['cameras'][name]
    rgb, world = camera['rgb'], np.asarray(camera['world_points'])
    box = np.asarray(choices['moving_panel_box_xyxy'], float)
    yy,xx = np.indices(contact_mask.shape)
    bounded = (xx>=box[0]) & (xx<=box[2]) & (yy>=box[1]) & (yy<=box[3])
    perception.point_predictor.set_image(rgb)
    masks, scores, _ = perception.point_predictor.predict(box=box, multimask_output=True)
    options = [(float(score), np.asarray(mask, bool)&bounded) for mask,score in zip(masks,scores)]
    options = [item for item in options if item[1].sum()>=60]
    if not options:
        raise ValueError('No RGB-predicted moving-panel region')
    _, mask = max(options, key=lambda item:item[0])
    valid = np.isfinite(world).all(2) & (np.asarray(camera['depth'])>0)
    mask &= valid
    cloud = world[mask & ~contact_mask]
    if len(cloud)<60:
        cloud = world[mask]
    if len(cloud)<60:
        raise ValueError('Panel region lacks metric support')
    # The dominant visible face supplies its plane, never an asset joint axis.
    model = _ransac(cloud, local_plane_config({}), 0)
    if model is None or model['inliers'].sum()<50 or model['rms']>.004:
        raise ValueError('Predicted panel has no reliable measured plane')
    centre, normal = model['center'], unit(model['normal'])
    toward = np.asarray(camera['camera_to_world'])[:3, 3]-centre
    normal *= 1 if normal@toward>0 else -1
    anchor = np.median(contact_cloud, axis=0)
    kind = choices['joint_family']
    if kind=='prismatic':
        axis = unit(np.r_[normal[:2],0.])
        axis *= -1 if choices['motion']=='close' else 1
        pivot, goal = anchor, float(choices['travel_m'])
        sign_source = 'open outward or close inward along observed upright panel normal'
    else:
        if choices['axis_source']=='vertical_hinge':
            hinge = np.array([ray_plane(pixel,camera,centre,normal) for pixel in choices['hinge_pixels_xy']])
            if np.linalg.norm(hinge[1]-hinge[0])<.03:
                raise ValueError('Visible hinge edge is too short to establish an upright axis')
            axis = np.array([0.,0.,1.])
            pivot = hinge.mean(0)
            pivot[2] = anchor[2]
            direction = np.asarray(choices['initial_image_direction_xy'],float)
            if direction.shape!=(2,) or np.linalg.norm(direction)<.5:
                raise ValueError('Hinge arc direction is visually unresolved')
            point = anchor-pivot
            projections=[]
            for sign in [1.,-1.]:
                moved = pivot+Rotation.from_rotvec(.12*sign*axis).apply(point)
                delta = project([moved],camera)[0]-project([anchor],camera)[0]
                projections.append(float(delta@direction/max(np.linalg.norm(delta),1e-6)))
            if abs(projections[0]-projections[1])<.05:
                raise ValueError('Chosen image cannot establish hinge motion direction')
            axis *= 1 if projections[0]>projections[1] else -1
            sign_source = 'VLM image-space motion choice projected through calibrated camera'
        else:
            axis, pivot = normal.copy(), anchor.copy()
            direction = choices['turn_direction_view']
            if direction not in {'clockwise','counterclockwise'}:
                raise ValueError('Control turn direction is unresolved')
            axis *= -1 if direction=='clockwise' else 1
            sign_source = 'VLM clockwise/counterclockwise choice about observed control-face normal'
        goal = float(np.deg2rad(choices['travel_deg']))
    support = cloud[model['inliers']]
    spec = {'schema_version':'articulated-spec-1.0','task_type':'articulated',
        'instruction':bundle['meta']['instruction'],'joint_family':kind,'motion':choices['motion'],
        'joint_origin_world_m':np.asarray(pivot).tolist(),'joint_axis_world':unit(axis).tolist(),
        'target_progress':goal,'progress_units':'meters' if kind=='prismatic' else 'radians',
        'source_anchor_world_m':anchor.tolist(),'camera':name,
        'front_plane':{'point_world_m':centre.tolist(),'normal_world':normal.tolist(),
                       'rms_m':float(model['rms']),'inliers':int(model['inliers'].sum())},
        'panel_normal_world':normal.tolist(),'pull_axis_world':normal.tolist(),
        'axis_prior':'upright prismatic/door geometry or control-face normal chosen from RGB',
        'direction_source':sign_source,'panel_depth_association_m':.020,
        'simulator_scene_state_used':False,'geometry_source':'predicted RGB regions and metric depth',
        'contact_policy':'functional fingers on selected handle/control and local panel; palm/arm/unrelated objects stay checked',
        'constraint':{'type':'point_on_line' if kind=='prismatic' else 'point_on_circle_with_coupled_orientation',
                      'position_tolerance_m':.015,'orientation_tolerance_rad':.10}}
    np.save(output/'panel-mask.npy',mask)
    np.save(output/'panel-plane-points.npy',support)
    (output/'task-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    return Mechanism(spec), mask


def refine_contact(bundle,choices,perception,output):
    """Honor the VLM contact pixel rather than snapping it to a nearby stem."""
    camera=bundle['cameras'][choices['camera']]
    x,y=np.rint(choices['grasp_pixel_xy']).astype(int)
    point=camera['world_points'][y,x]
    source=len(bundle['points_world'])
    selected={**bundle,'points_world':np.vstack([bundle['points_world'],point]),
              'pixels_yx':np.vstack([bundle['pixels_yx'],[y,x]])}
    cloud,mask,_=perception.selected_cloud(selected,source,articulated=True)
    yy,xx=np.indices(mask.shape)
    a,b,c,d=choices['contact_box_xyxy']
    mask &= (xx>=a-3)&(xx<=c+3)&(yy>=b-3)&(yy<=d+3)
    reference=camera['world_points'][mask]
    if len(reference)<30:
        raise ValueError('VLM contact pixel and RGB region have insufficient depth support')
    cloud=cloud[cKDTree(reference).query(cloud)[0]<=.008]
    np.save(output/'contact-pixel-mask.npy',mask)
    return cloud,mask


def expand_handle_surface(cameras,spec,choices,reference):
    """Fuse protruding surfaces in the selected RGB handle box, not the panel."""
    if choices['motion']=='turn_on':
        return reference
    camera=cameras[spec['camera']]
    lowx,lowy,highx,highy=choices['contact_box_xyxy']
    origin=np.asarray(spec['front_plane']['point_world_m'])
    normal=np.asarray(spec['panel_normal_world'])
    clouds=[reference]
    for other in cameras.values():
        points=np.asarray(other['world_points']).reshape(-1,3)
        points=points[np.isfinite(points).all(1)]
        distances=(points-origin)@normal
        keep=(distances>=.004)&(distances<=.06)
        points=points[keep]
        pixel=project(points,camera)
        keep=(pixel[:,0]>=lowx-3)&(pixel[:,0]<=highx+3)&(pixel[:,1]>=lowy-3)&(pixel[:,1]<=highy+3)
        clouds.append(points[keep])
    cloud=np.vstack(clouds)
    cells=np.floor(cloud/.002).astype(int)
    return cloud[np.unique(cells,axis=0,return_index=True)[1]]


def observe_progress(reference, mechanism, cameras, robot_volumes, expected, previous):
    predicted = mechanism.transform(expected)
    observed = observed_object_cloud(cameras,reference,predicted,robot_volumes)
    if len(observed)<50:
        return {'tracking_valid':False,'failure':'insufficient_observed_contact_region','observed_points':len(observed)}
    tree = cKDTree(observed)
    radius = np.linalg.norm(np.median(reference,axis=0)-mechanism.origin)
    width = .025 if mechanism.kind=='prismatic' else min(.3,.025/max(radius,.03))
    low,high = max(0.,min(previous,expected)-width), min(mechanism.goal+.08,max(previous,expected)+width)
    step = .0015 if mechanism.kind=='prismatic' else .01
    grid = np.linspace(low,high,max(5,int(np.ceil((high-low)/step))+1))
    costs=[]
    for value in grid:
        distances=tree.query(transform_points(reference,mechanism.transform(value)))[0]
        costs.append(float(np.mean(np.minimum(distances,.018)**2)))
    index=int(np.argmin(costs));progress=float(grid[index])
    distance=tree.query(transform_points(reference,mechanism.transform(progress)))[0]
    p75=float(np.quantile(distance,.75));coverage=float(np.mean(distance<.006))
    # Symmetric control geometry cannot establish its angle from depth alone.
    distant=np.abs(grid-progress)>(.008 if mechanism.kind=='prismatic' else .15)
    separation=(float(np.min(np.asarray(costs)[distant])-costs[index]) if distant.any() else float('inf'))
    ambiguous=bool(mechanism.kind=='revolute' and radius<.035 and separation<2e-7)
    return {'tracking_valid':bool(coverage>=.6 and p75<=.009 and not ambiguous),
        'observed_progress':progress,'expected_hand_progress':float(expected),
        'previous_observed_progress':float(previous),'hand_surface_error':float(expected-progress),
        'coverage_6mm':coverage,'percentile75_m':p75,'observed_points':len(observed),
        'angle_unobservable_from_depth':ambiguous,'cost_separation':None if not np.isfinite(separation) else separation,
        'simulator_scene_state_used':False,'source':'RGB-D template matched along the estimated mechanism manifold'}
