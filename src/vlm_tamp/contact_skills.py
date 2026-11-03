"""Numerical, observation-only push and hook contact skills.

No task identifiers, saved successful poses, native object geometry, or rewards
enter contact selection. Geometry predictions are explicitly distinct from
measured engagement and object movement.
"""
import copy
import time
from collections import Counter
from dataclasses import dataclass,field

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .coupled_scene import Scene,RobotVolumes
from .mcvamp_contact_numerical import NativeScene
from .rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from .rekep_libero.scene_state_v7 import observed_points
from .surface_contact_tools import contact_snapshot,selected_region_points
from .timed_execution import schedule
from .part_perception import projected_selection


class SkillStop(Exception):
    def __init__(self,reason,**evidence):
        super().__init__(reason);self.reason=reason;self.evidence=evidence


@dataclass
class Budget:
    seconds:float
    started:float=field(default_factory=time.monotonic)
    context:dict=field(default_factory=dict)
    @property
    def remaining(self):return max(0.,self.started+self.seconds-time.monotonic())
    def check(self):
        if self.remaining<=0:raise SkillStop('budget_exhausted',search_complete=False,**self.context)
    def child(self,seconds):return Budget(min(float(seconds),self.remaining))


def unit(vector):
    value=np.asarray(vector,float)
    if value.shape!=(3,) or not np.isfinite(value).all() or np.linalg.norm(value)<1e-8:
        raise ValueError('A finite nonzero world direction is required')
    return value/np.linalg.norm(value)


def region_points(cameras,metadata,camera,box,occupied=None,selected_mask=None):
    if camera not in cameras:raise ValueError('Choose a current observed camera')
    if occupied is None:occupied,_=observed_points(cameras,metadata,spacing=.003)
    points,_=selected_region_points(cameras[camera],box,occupied,selected_mask)
    if len(points)<12:raise SkillStop('insufficient_target_depth',supported_points=len(points))
    return points


def hook_feature(points,direction,cameras=None):
    """Require a resolved non-planar feature and its observed retaining surface.

    This v1 detects observed bars/lips; an arbitrary panel edge with an unseen
    reverse surface is unresolved, not an invented hookable opening.
    """
    points=np.asarray(points,float);center=np.median(points,axis=0)
    _,_,basis=np.linalg.svd(points-center,full_matrices=False)
    extents=np.ptp(np.quantile((points-center)@basis.T,[.05,.95],axis=0),axis=0)
    shape={'extent_m':extents.tolist(),'observed_points':len(points),
           'source':'selected non-robot RGB-D endpoints','catch_verified':False,
           'aspect_ratio':float(extents[0]/max(extents[1],1e-12)),
           'aspect_ratio_is_admission_gate':False}
    checks=[('observed_points',len(points),20,'minimum'),('long_extent_m',extents[0],.025,'minimum'),
        ('width_m',extents[1],.07,'maximum'),('thickness_m',extents[2],.004,'minimum')]
    shape['failed_checks']=[{'quantity':key,'actual':float(value),'limit':limit,'comparison':comparison}
        for key,value,limit,comparison in checks if (value<limit if comparison=='minimum' else value>limit)]
    if shape['failed_checks']:
        raise SkillStop('no_hookable_feature',feature=shape)
    d=unit(direction);projection=(points-center)@d
    rear=points[projection<=np.quantile(projection,.15)]
    if len(rear)<8 or np.ptp(projection)<.006:
        raise SkillStop('hook_back_surface_unobserved',feature=shape)
    # Keep contact away from observed endpoints/mounts; choose measured points.
    middle=np.abs((rear-center)@basis[0])<max(.012,.2*extents[0])
    if middle.sum()>=5:rear=rear[middle]
    point=rear[np.argmin(np.linalg.norm(rear-np.median(rear,axis=0),axis=1))]
    if cameras is not None:
        supported=[]
        for name,view in cameras.items():
            transform=np.asarray(view['camera_to_world']);camera_origin=transform[:3,3]
            # The retaining surface must be observed from its pulling-back
            # side, not merely inferred from the outline of a front view.
            if (camera_origin-point)@d>=-.01:continue
            camera_point=transform_points([point],np.linalg.inv(transform))[0]
            if camera_point[2]<=0:continue
            pixel=np.asarray(view['intrinsic'])@camera_point;pixel=pixel[:2]/pixel[2]
            x,y=np.rint(pixel).astype(int);xyz=np.asarray(view['world_points']);h,w=xyz.shape[:2]
            if not (0<=x<w and 0<=y<h):continue
            if np.isfinite(xyz[y,x]).all() and np.linalg.norm(xyz[y,x]-point)<.004:supported.append(name)
        if not supported:raise SkillStop('hook_back_surface_unobserved',feature=shape)
        shape['retaining_surface_supported_views']=supported
    return point,{**shape,'retaining_axis_world':basis[0].tolist(),'contact_world_m':point.tolist()}


def projected_surface_membership(points,view,region,origin,normal,selected_mask=None):
    """Associate observed views with the caller's selected local surface.

    A nearest-neighbor match to one camera's samples leaves holes at view/voxel
    seams. Project the whole observed cloud into the same selected image region,
    retaining the existing local-plane and extent bounds. No points are erased
    and no proposed robot pose participates in assigning these roles.
    """
    return (projected_selection(points,view,region,selected_mask,padding_pixels=0)
        &(np.abs((points-origin)@normal)<.015)&(np.linalg.norm(points-origin,axis=1)<.45))


def build_contact_scene(cameras,metadata,chain,camera,region,target_region,contact_surface,
                        direction=None,joint_degrees=None,joint_scene=None,*,skill_kind='push',observed=None,
                        contact_mask=None,target_mask=None,contact_pixel_xy=None,contact_name=None):
    if direction is None and joint_degrees is None:
        raise SkillStop('motion_direction_required',explanation='Supply a world direction or signed associated joint increment.')
    if observed is None:observed=observed_points(cameras,metadata,spacing=.003)
    occupied=observed[0]
    selected=region_points(cameras,metadata,camera,region,occupied,selected_mask=contact_mask)
    base,patch=contact_snapshot(cameras,metadata,camera,region,chain,radius_m=.035,observed=observed,
        selected_points=selected,selected_mask=contact_mask,contact_pixel_xy=contact_pixel_xy)
    origin=np.asarray(patch['point_world_m']);normal=np.asarray(patch['normal_toward_camera_world'])
    associated=joint_degrees is not None
    if associated and joint_scene is None:raise SkillStop('joint_model_required')
    if associated:
        target=np.asarray(joint_scene.moving_reference)
        direction=np.sign(joint_degrees)*np.cross(joint_scene.axis,origin-joint_scene.origin)
    else:
        target=(selected if target_region is None and target_mask is None else
            region_points(cameras,metadata,camera,target_region or region,occupied,selected_mask=target_mask))
    d=unit(direction)
    spec=copy.deepcopy(joint_scene.spec) if associated else {
        'joint_origin_world_m':origin.tolist(),'joint_axis_world':normal.tolist(),
        'front_plane':{'point_world_m':origin.tolist()},'panel_normal_world':normal.tolist(),
        'articulation_estimated':False}
    # Associate only measured selected endpoints at the observation grid's
    # resolution. Do not expand a rectangle into a guessed plane, bar or mount.
    owned=cKDTree(target).query(occupied,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035
    contact_owned=cKDTree(selected).query(occupied,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035
    if not associated:
        owned &= projected_selection(occupied,cameras[camera],target_region or region,
            target_mask if target_mask is not None else contact_mask,padding_pixels=0)
    contact_owned &= projected_selection(occupied,cameras[camera],region,contact_mask,padding_pixels=0)
    owned |= contact_owned
    scene=Scene.from_points(occupied[~owned],occupied[owned],occupied[contact_owned],spec,
        config=copy.deepcopy(joint_scene.config) if associated else {},
        robot=RobotVolumes(chain,metadata['robot_collision_volumes_world'],metadata['joint_positions']),
        reference_volumes=metadata['robot_collision_volumes_world'])
    scene.skill_contact_surface=contact_surface;scene.skill_kind=skill_kind
    scene.contact_policy_kind='contact_skill';scene.skill_observed_cameras=cameras
    scene.skill_joint_sign=float(np.sign(joint_degrees)) if associated else None
    return scene,{**patch,'direction_world':d.tolist(),'joint_associated':associated,
        'contact_name':contact_name,'contact_mask_used':contact_mask is not None,
        'target_mask_used':target_mask is not None,'contact_surface':contact_surface,
        'target_observed_points':int(owned.sum()),'feature_source':'caller-selected RGB-D pixels',
        'contact_permission_scope':'fingers versus all environment surfaces; selected palm only when explicitly requested',
        'target_association':'selected observed points at 3 mm scene resolution; no shape completion',
        'arm_environment_checked':True,'finger_environment_checked':False,
        'target_region_xyxy':target_region or region},selected


def contact_landmarks(metadata: dict, contact_surface: str, hook: bool = False) -> list[dict]:
    """Describe calibrated contact faces and their configured robot collision masks.

    Masks describe contact eligibility, not measured engagement. Pad distal faces
    use the actual box face; passive body-mesh references retain their old names.
    """
    inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']));result=[]
    for volume in metadata['robot_collision_volumes_world']:
        name=volume['robot_geom_name'];p=transform_points(volume['vertices_world'],inverse)
        masks=volume.get('physical_collision_masks')
        physical={'robot_collision_geom':name,'physical_collision_masks':masks,
            'initiates_simulator_contacts':bool(masks['contype']) if masks is not None else None}
        if name.endswith('_pad_collision'):
            # Facing axis of the pad is toward the other finger, along TCP x.
            sign=-np.sign(p[:,0].mean());normal=np.array([sign,0.,0.])
            point=p.mean(0);point[0]=p[:,0].max() if sign>0 else p[:,0].min()
            result.append({'name':name,'point':point,'normal':normal,
                'contact_role':'inner_jaw_face','role_description':'Faces the opposite finger across the open jaw gap.',**physical})
            if not hook:
                distal=p.mean(0);distal[2]=p[:,2].max()
                result.append({'name':name+'_distal_face','point':distal,
                    'normal':np.array([0.,0.,1.]),'contact_role':'distal_pad_face',
                    'role_description':'Pad end face along the finger length.',**physical})
        elif 'finger' in name and not hook:
            tip=p[p[:,2]>=p[:,2].max()-.003].mean(0)
            result.append({'name':name+'_distal_tip','point':tip,'normal':np.array([0.,0.,1.]),
                'contact_role':'body_mesh_tip','role_description':'Body-mesh reference; check its configured collision masks.',**physical})
        elif name=='gripper0_hand_collision' and contact_surface=='palm' and not hook:
            for axis in [1,2]:
                for sign in [-1,1]:
                    values=p[:,axis]*sign;point=p[values>=values.max()-.003].mean(0)
                    normal=np.eye(3)[axis]*sign
                    result.append({'name':name+f'_face_{axis}_{sign}','point':point,'normal':normal,
                        'contact_role':'palm_face','role_description':'Palm contact requires explicit selection.',**physical})
    # Stable labels, without recommending a contact strategy.
    return sorted(result,key=lambda x:x['name'])


def predicted_scene(scene,displacement,angle=0.):
    new=copy.copy(scene)
    if angle:
        r=Rotation.from_rotvec(np.asarray(scene.axis)*angle).as_matrix()
        t=scene.origin-r@scene.origin
    else:r=np.eye(3);t=np.asarray(displacement)
    new.moving_reference=scene.moving_reference@r.T+t
    new.contact_reference=scene.contact_reference@r.T+t
    return new


def skill_native_scene(scene,record,metadata,start,*,allow_contact=True):
    """Use configured collision surfaces with zero added clearance (user policy)."""
    context=NativeScene(scene,record,metadata,start,.05,articulated=False,
        allow_contact=allow_contact,gripper_padding=0.,collision_policy='veto')
    context.record['contact_skill_clearance']={'requested_m':0.,'used_m':0.,
        'source':'explicit user request: test without added collision padding',
        'zero_boundary_collisions_ignored':False}
    return context


def advance_contact_scene(scene,transform,cameras,control_step):
    """Move the retained rigid target from measured feedback; reuse the static map."""
    transform=np.asarray(transform,float);rotation=transform[:3,:3]
    new=copy.copy(scene)
    for name in ('moving_reference','contact_reference','inferred_panel_reference'):
        points=transform_points(getattr(scene,name,np.empty((0,3))),transform)
        points.setflags(write=False);setattr(new,name,points)
    new.panel_origin=transform_points([scene.panel_origin],transform)[0]
    new.panel_normal=rotation@scene.panel_normal
    if hasattr(scene,'spec'):
        new.spec=copy.deepcopy(scene.spec)
        new.spec['front_plane']['point_world_m']=new.panel_origin.tolist()
        new.spec['panel_normal_world']=new.panel_normal.tolist()
        if 'source_anchor_world_m' in new.spec:
            new.spec['source_anchor_world_m']=transform_points([new.spec['source_anchor_world_m']],transform)[0].tolist()
    new.moving_tree=cKDTree(new.moving_reference);new.contact_tree=cKDTree(new.contact_reference)
    new.collision_reference=np.vstack([new.moving_reference,new.inferred_panel_reference])
    new.collision_reference.setflags(write=False);new.collision_tree=cKDTree(new.collision_reference)
    new._moving_cache={}
    new._reference_contact_distance=new.contact_tree.query(new.collision_reference)[0]
    new._reference_panel_distance=np.abs((new.collision_reference-new.panel_origin)@new.panel_normal)
    if len(new.contact_reference):new._reference_contact_height=np.quantile(new.contact_reference[:,2],[.01,.99])
    new.skill_observed_cameras=cameras
    new.pose_update={'source':'rgbd_rigid_motion','reference_control_step':int(control_step),
        'transform_world':transform.tolist(),'stationary_points_changed':False,
        'fresh_depth_endpoints_added':False,'full_scene_rebuilt':False,'hinge_refined':False}
    return new,{'source':'accepted RGB-D rigid target motion','static_points_reused':len(new.static_points),
        'moving_points_transported':len(new.moving_reference),'full_scene_rebuilt':False,
        'new_depth_points_added':0,'simulator_object_state_used':False}


def refresh_collision_observations(scene,cameras,metadata):
    """Update occupancy from measured depth after a measured target transform.

    Keep occluded old obstacles. Remove an old endpoint only where a current
    calibrated ray measures free space in front of a farther surface, or where
    the endpoint belongs to the independently tracked target at its new pose.
    """
    old=np.asarray(scene.static_points);free=np.zeros(len(old),bool)
    for view in cameras.values():
        p=transform_points(old,np.linalg.inv(np.asarray(view['camera_to_world'])))
        good=np.isfinite(p).all(1)&(p[:,2]>.01);ids=np.flatnonzero(good)
        xy=p[good]@np.asarray(view['intrinsic']).T;xy=np.rint(xy[:,:2]/xy[:,2,None]).astype(int)
        depth=np.asarray(view['depth']).squeeze();h,w=depth.shape
        inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h);ids=ids[inside];xy=xy[inside]
        measured=depth[xy[:,1],xy[:,0]]
        free[ids]|=np.isfinite(measured)&(measured>0)&(p[ids,2]<measured-.006)
    current,_=observed_points(cameras,metadata,spacing=.003)
    target_tree=cKDTree(scene.moving_reference)
    target=target_tree.query(current,distance_upper_bound=.008)[0]<.008
    old_target=target_tree.query(old,distance_upper_bound=.006)[0]<.006
    new=copy.copy(scene)
    from .rekep_libero.scene_state_v7 import voxel_points
    new.static_points=voxel_points(np.vstack([old[~free&~old_target],current[~target]]),.003)
    if target.sum()>=12:
        new.moving_reference=current[target]
        contact=cKDTree(scene.contact_reference).query(new.moving_reference,distance_upper_bound=.008)[0]<.008
        if contact.sum()>=3:new.contact_reference=new.moving_reference[contact]
    new.skill_observed_cameras=cameras
    return new,{'source':'current RGB-D occupancy and tracked target association',
        'old_points_cleared_by_measured_free_space':int(free.sum()),
        'current_target_points':int(target.sum()),'retained_old_obstacle_points':int((~free&~old_target).sum()),
        'simulator_object_state_used':False}


def fresh_collision_snapshot(scene,cameras,metadata):
    """Rebuild from current views when a persistent map rejects the measured start.

    No currently measured obstacle is discarded. Older unobserved points are
    retired, not declared free. Unknown space remains explicitly unmodeled.
    """
    current,filtering=observed_points(cameras,metadata,spacing=.003)
    belongs=cKDTree(scene.moving_reference).query(current,distance_upper_bound=.008)[0]<.008
    if belongs.sum()<12:raise SkillStop('current_target_depth_unresolved',supported_points=int(belongs.sum()))
    new=copy.copy(scene);new.static_points=current[~belongs];new.moving_reference=current[belongs]
    contact=cKDTree(scene.contact_reference).query(new.moving_reference,distance_upper_bound=.008)[0]<.008
    if contact.sum()>=3:new.contact_reference=new.moving_reference[contact]
    new.skill_observed_cameras=cameras
    return new,{'source':'fresh calibrated RGB-D snapshot; current points retained',
        'camera_count':len(cameras),'current_observed_points':len(current),
        'current_static_points':int((~belongs).sum()),'current_target_points':int(belongs.sum()),
        'historical_unobserved_points_retired':True,'unknown_space_is_unmodeled':True,
        'simulator_object_state_used':False,'robot_filter':filtering}


def cartesian_path(
    ik: Any,
    scene: Any,
    record: dict[str, Any],
    metadata: dict[str, Any],
    start: ArrayLike,
    target_pose: ArrayLike,
    budget: Budget,
    *,
    displacement: ArrayLike | None = None,
    angle: float = 0.0,
    timing: Counter | None = None,
    contact_local: ArrayLike | None = None,
    allow_partial: bool = False,
    environment_collision: bool = True,
) -> np.ndarray:
    """Cartesian IK waypoints; no CPU pose adjustment or dense path recheck."""
    timing=timing if timing is not None else Counter()
    begin=pose_matrix(ik.chain.forward(start));target=pose_matrix(target_pose)
    local=np.zeros(3) if contact_local is None else np.asarray(contact_local)
    first=begin[:3,3]+begin[:3,:3]@local;last=target[:3,3]+target[:3,:3]@local
    count=max(1,int(np.ceil(np.linalg.norm(last-first)/.003)))
    path=[np.asarray(start)]
    if environment_collision:
        from .current_robot_scene import clear_current_robot_scene
        checked = clear_current_robot_scene(scene, metadata, metadata.get('control_steps', 0))
        began=time.perf_counter()
        ik.configure_observed_scene(checked)
        timing['ik_scene_setup_seconds']+=time.perf_counter()-began
    def incomplete(reason):
        scene.skill_partial_plan={'reason':reason,'completed_waypoints':len(path)-1,'requested_waypoints':count,
            'global_infeasibility_proved':False}
        if allow_partial and len(path)>1:return np.asarray(path)
        raise SkillStop(reason,completed_waypoints=len(path)-1,requested_waypoints=count)
    for fraction in np.linspace(0,1,count+1)[1:]:
        if budget.remaining<=0:return incomplete('stroke_planning_budget_exhausted')
        began=time.perf_counter()
        if environment_collision:
            transform=np.eye(4)
            if angle:
                rotation=Rotation.from_rotvec(np.asarray(scene.axis)*angle*fraction).as_matrix()
                transform[:3,:3]=rotation
                transform[:3,3]=scene.origin-rotation@scene.origin
            else:
                transform[:3,3]=fraction*(last-first)
            ik.move_scene_target(transform)
        normal_arguments={}
        if getattr(scene,'skill_contact_constraint','point')=='surface_normal':
            normal=np.asarray(scene.skill_surface_normal_world)
            if angle:
                normal=Rotation.from_rotvec(np.asarray(scene.axis)*angle*fraction).apply(normal)
            normal_arguments={'contact_normal_local':scene.skill_contact_normal_local,
                'surface_normal_world':normal}
        solution=ik.solve_contact(first+fraction*(last-first),local,path[-1],begin[:3,:3],
            environment_collision=environment_collision,**normal_arguments)
        timing['ik_seconds']+=time.perf_counter()-began
        if solution is None:return incomplete('contact_ik_failed')
        path.append(np.asarray(solution['joints']))
    return np.asarray(path)


def plan_contact_stroke(ik,scene,record,metadata,contact_local,delta,budget,*,angle=0.,timing=None,
                        physical_feedback=False,environment_collision=True):
    """Straight contact-point targets solved by cuRobo; no push path planner."""
    timing=timing if timing is not None else Counter()
    start=np.asarray(metadata['joint_positions']);tcp=pose_matrix(ik.chain.forward(start))
    goal=tcp.copy();goal[:3,3]+=delta;scene.skill_partial_plan=None
    path=cartesian_path(ik,scene,record,metadata,start,matrix_pose(goal),budget,
        timing=timing,contact_local=contact_local,angle=angle,allow_partial=physical_feedback,
        environment_collision=environment_collision)
    return path,{'motion_backend':'cartesian_ik','reachable_prefix':scene.skill_partial_plan,
        'maximum_adjacent_ik_joint_change_rad':float(np.abs(np.diff(path,axis=0)).max()) if len(path)>1 else 0.,
        'physical_feedback':physical_feedback,'orientation_is_soft_preference':False,
        'environment_planner_used':False,'ik_environment_collision_checked':environment_collision,
        'additional_dense_validation':False}


def plan_engagement(
    ik: Any,
    scene: Any,
    record: dict[str, Any],
    metadata: dict[str, Any],
    evidence: dict[str, Any],
    selected: Any,
    kind: str,
    budget: Budget,
    maximum_approach_controls: Any = None,
) -> dict[str, Any]:
    """Solve and plan exactly the contact Luna selected; no alternate pose menu."""
    from .surface_contact_tools import contact_solutions
    began=time.perf_counter();timing=Counter();attempts=[]
    point=np.asarray(evidence['point_world_m']);q=np.asarray(metadata['joint_positions'])
    name=evidence.get('contact_name')
    if 'hand_collision' in (name or '') and evidence['contact_surface']!='palm':
        raise ValueError('Palm contact requires contact_surface=palm')
    if kind=='hook_pull':
        cloud=np.asarray(selected,float)
        if len(cloud)<4 or np.linalg.matrix_rank(cloud-cloud.mean(0))<3:
            raise SkillStop('no_hookable_feature',reason_detail='Selected depth samples contain no retaining thickness')
        if name is None or not name.endswith('_pad_collision'):
            raise ValueError('Choose an inward-facing pad contact_name for a non-pinching hook')
    with skill_native_scene(scene,record,metadata,q) as context:
        ik.configure_scene(context)
        phase=time.perf_counter()
        constraint=evidence.get('contact_constraint','point')
        normal_arguments=({'contact_constraint':constraint,'surface_normal_world':evidence['surface_normal_world']}
            if constraint=='surface_normal' else {})
        candidate,solutions=contact_solutions(ik,metadata,point,name,**normal_arguments)
        timing['ik_seconds']+=time.perf_counter()-phase
        selection={}
        reference_normal=evidence.get('surface_normal_world') if constraint=='surface_normal' else evidence.get('normal_toward_camera_world')
        if reference_normal is not None:
            from .contact_orientation import selection_geometry
            current=pose_matrix(ik.chain.forward(q))
            current_contact=current[:3,3]+current[:3,:3]@candidate['contact_local_m']
            selection=selection_geometry(point,reference_normal,current_contact,evidence['direction_world'],
                getattr(scene,'skill_observed_cameras',{}))
            selection.update(normal_source='caller_selected_face' if constraint=='surface_normal' else 'observed_local_patch_toward_camera',
                contact_role=candidate.get('contact_role'))
        budget.check()
        for solution in solutions:
            budget.check();phase=time.perf_counter()
            path,report=context.plan(q,solution['joints'],constrained=False,seconds=min(8.,budget.remaining))
            timing['approach_planning_seconds']+=time.perf_counter()-phase
            attempts.append({'joints':solution['joints'],'planning':report})
            if not report['exact_solution']:continue
            refs=schedule(path,maximum_joint_speed=scene.skill_approach_speed_rad_s,endpoint_hold_steps=0)
            tcp=pose_matrix(ik.chain.forward(path[-1]))
            orientation={}
            if constraint=='surface_normal':
                from .contact_orientation import contact_orientation_report
                current=pose_matrix(metadata['eef_pose_world_xyzw'])
                current_contact=current[:3,3]+current[:3,:3]@candidate['contact_local_m']
                orientation=contact_orientation_report(tcp[:3,:3],candidate['contact_normal_local'],
                    evidence['surface_normal_world'],evidence['direction_world'],current_contact,point)
                orientation['pose_source']='planned IK endpoint'
                if evidence.get('joint_associated'):
                    from .contact_orientation import joint_push_prediction
                    orientation['joint_prediction']=joint_push_prediction(point,scene.origin,scene.axis,evidence['direction_world'])
            return {'path':path,'references':refs,'free_references':refs,
                'insertion_references':np.empty((0,7)),'direct_contact_approach':True,
                'contact_local':np.asarray(candidate['contact_local_m']),'contact_world':point,
                'contact_normal_local':np.asarray(candidate['contact_normal_local']) if 'contact_normal_local' in candidate else None,
                'contact_constraint':constraint,'contact_orientation':orientation,
                'selection_geometry':selection,
                'contact_tcp_pose_world_xyzw':matrix_pose(tcp),'rotation':tcp[:3,:3],
                'insertion_direction':unit(evidence['direction_world']),'contact_name':name,
                'contact_model':{key:candidate[key] for key in ('robot_collision_geom',
                    'physical_collision_masks','initiates_simulator_contacts')},
                'attempts':attempts,'candidate_index':0,'timing_seconds':dict(timing),
                'search_work':{'contact_points':1,'ik_queries':1,'ik_solutions':len(solutions),'path_queries':len(attempts)},
                'total_seconds':time.perf_counter()-began,'predicted_contact_only':True,
                'native_contact_policy':context.record,'automatic_pose_adjustment':False}
    raise SkillStop('contact_ik_failed' if not solutions else 'contact_path_not_found',
        ik_diagnostic=candidate.get('ik_diagnostic',{}),
        selection_geometry=selection,
        attempts=attempts,search_complete=True,global_infeasibility_proved=False,
        selected_pixel=evidence.get('contact_pixel_xy'),selected_contact=name,
        timing_seconds=dict(timing),search_work={'contact_points':1,'ik_queries':1,'ik_solutions':len(solutions)})


def fit_object_motion(first,second):
    """Robust 3D rigid fit to independently tracked, measured RGB-D pairs."""
    a=np.asarray(first,float).reshape(-1,3);b=np.asarray(second,float).reshape(-1,3)
    finite=np.isfinite(a).all(1)&np.isfinite(b).all(1);original_indices=np.flatnonzero(finite);a,b=a[finite],b[finite]
    result={'accepted':False,'pairs':len(a),'source':'RGB-D feature correspondences','simulator_object_state_used':False,
        'support_rule':'at least 8 spatially spread inliers and a strict majority of the measured correspondences',
        'inlier_residual_threshold_m':.004,'minimum_spatial_span_m':.025}
    if len(a)<8:return {**result,'reason':'insufficient_visible_matches'}
    def fit(x,y):
        u,_,vt=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)));r=vt.T@u.T
        if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
        return r,y.mean(0)-r@x.mean(0)
    rng=np.random.default_rng(0);best=np.zeros(len(a),bool)
    for _ in range(100):
        ids=rng.choice(len(a),3,replace=False)
        if np.linalg.norm(np.cross(a[ids[1]]-a[ids[0]],a[ids[2]]-a[ids[0]]))<1e-5:continue
        r,t=fit(a[ids],b[ids]);inliers=np.linalg.norm(a@r.T+t-b,axis=1)<.004
        if inliers.sum()>best.sum():best=inliers
    # A supported dominant motion need not explain 60% of a rectangular RGB-D
    # selection: occlusion and background can supply a minority of outliers.
    # Keep the geometric residual/span tests and reject tied motion groups.
    if best.sum()<8 or best.sum()*2<=len(a):
        return {**result,'reason':'inconsistent_matches','inliers':int(best.sum()),'inlier_fraction':float(best.mean())}
    r,t=fit(a[best],b[best]);error=np.linalg.norm(a[best]@r.T+t-b[best],axis=1)
    span=float(np.linalg.norm(np.ptp(a[best],axis=0)))
    if span<.025:return {**result,'reason':'insufficient_spatial_baseline','span_m':span}
    transform=np.eye(4);transform[:3,:3]=r;transform[:3,3]=t
    return {**result,'accepted':True,'reason':'measured_rigid_motion','inliers':int(best.sum()),
        'inlier_indices':original_indices[best].tolist(),
        'rms_m':float(np.sqrt(np.mean(error**2))),'span_m':span,'transform':transform.tolist()}


class ContactTracker:
    def __init__(self,cameras,target,step):
        from .rigid_hinge_tracking import RGBDHingeTracker
        self.target_reference=np.asarray(target).copy()
        self.tracker=RGBDHingeTracker(cameras,target,prior=np.mean(target,axis=0),reference_step=step,camera_names=list(cameras))
        self.last_transform=np.eye(4)
        self.last_view_counts={}
    def observe(self,cameras,step):
        # Keep the original multi-view references across strokes. Re-seeding
        # from a reduced camera capture made full-view recovery ineffective:
        # its newly captured cameras no longer had reference features.
        evidence=self.tracker.observe(cameras,control_step=step,estimate_joint=False)
        self.last_view_counts.update({v['camera']:v['tracked_pairs'] for v in evidence['camera_counts']})
        result=fit_object_motion(*self.tracker.last_pairs)
        result.update(reference_control_step=self.tracker.reference_step,
            reference_camera_count=len(self.tracker.references))
        if result['accepted']:
            absolute=np.asarray(result['transform'])
            result['transform']=(absolute@np.linalg.inv(self.last_transform)).tolist()
            self.last_transform=absolute
            # Advance optical-flow images after measured motion, retaining the
            # original world reference and cameras absent from this capture.
            # This shortens the next image displacement without rendering or
            # writing more observations, and does not relax motion-fit checks.
            from .rigid_hinge_tracking import RGBDHingeTracker
            target=self.target_reference@absolute[:3,:3].T+absolute[:3,3]
            fresh=RGBDHingeTracker(cameras,target,prior=np.mean(target,axis=0),
                reference_step=step,camera_names=list(cameras))
            for name,reference in fresh.references.items():
                reference['points']=(reference['points']-absolute[:3,3])@absolute[:3,:3]
                self.tracker.references[name]=reference
            result['tracking_images_advanced']=list(fresh.references)
        return result


def measured_contact_speed(events,chain,contact_local,start_q):
    qs=[np.asarray(start_q)]+[np.asarray(e['measured_joints']) for e in events]
    points=[]
    for q in qs:
        t=pose_matrix(chain.forward(q));points.append(t[:3,3]+t[:3,:3]@contact_local)
    speeds=np.linalg.norm(np.diff(points,axis=0),axis=1)*20 if len(points)>1 else np.array([])
    return {'mean_m_s':float(speeds.mean()) if len(speeds) else 0.,
        'peak_m_s':float(speeds.max()) if len(speeds) else 0.,'samples':len(speeds),
        'source':'measured robot joints and calibrated contact-landmark FK at 20 Hz'}
