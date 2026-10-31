"""Numerical, observation-only push and hook contact skills.

No task identifiers, saved successful poses, native object geometry, or rewards
enter contact selection. Geometry predictions are explicitly distinct from
measured engagement and object movement.
"""
import copy
import time
from contextlib import nullcontext
from collections import Counter
from dataclasses import dataclass,field

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .coupled_scene import Scene,RobotVolumes
from .mcvamp_contact_numerical import NativeScene
from .rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from .rekep_libero.scene_state_v7 import observed_points
from .surface_contact_tools import contact_snapshot
from .timed_execution import schedule,schedule_contact
from .contact_collision_evidence import current_collision_witness


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


def region_points(cameras,metadata,camera,box,occupied=None):
    if camera not in cameras:raise ValueError('Choose a current observed camera')
    xyz=np.asarray(cameras[camera]['world_points']);h,w=xyz.shape[:2]
    box=np.asarray(box,float)
    if box.shape!=(4,) or not np.isfinite(box).all() or not 0<=box[0]<box[2]<w or not 0<=box[1]<box[3]<h:
        raise ValueError('Region must use original image pixels inside the current image')
    y,x=np.indices((h,w));mask=(x>=box[0])&(x<=box[2])&(y>=box[1])&(y<=box[3])&np.isfinite(xyz).all(2)
    points=xyz[mask]
    if occupied is None:occupied,_=observed_points(cameras,metadata,spacing=.003)
    if not len(points) or not len(occupied):raise SkillStop('insufficient_target_depth')
    points=points[cKDTree(occupied).query(points,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035]
    if len(points)<12:raise SkillStop('insufficient_target_depth',supported_points=len(points))
    return points


def hook_feature(points,direction,cameras=None):
    """Require a resolved slender, non-planar feature and its rear surface.

    This v1 detects observed bars/lips; an arbitrary panel edge with an unseen
    reverse surface is unresolved, not an invented hookable opening.
    """
    points=np.asarray(points,float);center=np.median(points,axis=0)
    _,_,basis=np.linalg.svd(points-center,full_matrices=False)
    extents=np.ptp(np.quantile((points-center)@basis.T,[.05,.95],axis=0),axis=0)
    shape={'extent_m':extents.tolist(),'observed_points':len(points),
           'source':'selected non-robot RGB-D endpoints','catch_verified':False}
    if len(points)<20 or extents[0]<.025 or extents[1]>.07 or extents[2]<.004 or extents[0]<1.8*extents[1]:
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


def projected_surface_membership(points,view,region,origin,normal):
    """Associate observed views with the caller's selected local surface.

    A nearest-neighbor match to one camera's samples leaves holes at view/voxel
    seams. Project the whole observed cloud into the same selected image region,
    retaining the existing local-plane and extent bounds. No points are erased
    and no proposed robot pose participates in assigning these roles.
    """
    optical=transform_points(points,np.linalg.inv(pose_matrix(view['camera_to_world'])))
    projected=optical@np.asarray(view['intrinsic']).T
    forward=optical[:,2]>0
    uv=np.full((len(points),2),np.inf)
    uv[forward]=projected[forward,:2]/projected[forward,2,None]
    x1,y1,x2,y2=region
    return (forward&(uv[:,0]>=x1)&(uv[:,0]<=x2)&(uv[:,1]>=y1)&(uv[:,1]<=y2)
        &(np.abs((points-origin)@normal)<.015)&(np.linalg.norm(points-origin,axis=1)<.45))


def build_contact_scene(cameras,metadata,chain,camera,region,target_region,contact_surface,
                        direction=None,joint_degrees=None,joint_scene=None,*,skill_kind='push',observed=None):
    base,patch=contact_snapshot(cameras,metadata,camera,region,chain,radius_m=.035,observed=observed)
    occupied=np.vstack([base.static_points,base.contact_reference])
    selected=region_points(cameras,metadata,camera,region,occupied)
    target=region_points(cameras,metadata,camera,target_region or region,occupied)
    origin=np.asarray(patch['point_world_m']);normal=np.asarray(patch['normal_toward_camera_world'])
    # A broad image rectangle can include the table/background through a handle
    # opening. Keep the observed surface near the selected contact plane. This
    # is a disclosed local surface association, never an object-label oracle.
    nearby=(np.abs((target-origin)@normal)<.015)&(np.linalg.norm(target-origin,axis=1)<.45)
    if nearby.sum()>=12:target=target[nearby]
    associated=False
    feature_source='selected RGB-D region'
    contact_points=base.contact_reference
    if skill_kind=='push' and joint_degrees is not None and joint_scene is not None:
        # A region can include table/background pixels. An explicitly
        # associated current target provides a measured spatial filter.
        on_target=cKDTree(joint_scene.moving_reference).query(selected,distance_upper_bound=.004)[0]<.004
        selected=selected[on_target]
        if len(selected)<12:raise SkillStop('selected_region_lacks_joint_target_support',supported_points=len(selected))
        origin=selected[np.argmin(np.linalg.norm(selected-np.median(selected,axis=0),axis=1))]
        normal=np.asarray(joint_scene.panel_normal).copy()
        camera_origin=np.asarray(cameras[camera]['camera_to_world'])[:3,3]
        if normal@(camera_origin-origin)<0:normal=-normal
        patch.update(point_world_m=origin.tolist(),normal_toward_camera_world=normal.tolist())
        feature_source='current observed selected-region endpoints associated with the caller installed joint target'
    if skill_kind=='push':
        # The caller selects a region, not an exact robot contact point.
        # Permission covers measured samples in that region; it does not
        # inflate robot geometry or clear any unrelated observed obstacles.
        selected=selected[np.abs((selected-origin)@normal)<.015]
        if len(selected)<12:raise SkillStop('insufficient_selected_surface_support')
        contact_owned=cKDTree(selected).query(occupied,distance_upper_bound=.0035)[0]<.0035
        if joint_degrees is None:
            contact_owned|=projected_surface_membership(occupied,cameras[camera],region,origin,normal)
        contact_points=occupied[contact_owned]
        multiview_selected=contact_points
        patch.update(selected_observed_points=len(contact_points),
            contact_permission='distal fingertips/pads and explicitly chosen palm against measured selected-region samples',
            contact_permission_scope='caller-selected region on the observed surface',
            radius_m=float(np.linalg.norm(contact_points-origin,axis=1).max()))
    if skill_kind=='hook_pull':
        if joint_scene is not None and joint_degrees is not None:
            # Reuse an explicitly grounded selected handle, but retain only
            # current observed endpoints inside the caller's feature box.
            candidate=occupied[cKDTree(joint_scene.contact_reference).query(occupied,distance_upper_bound=.004)[0]<.004]
            camera_points=transform_points(candidate,np.linalg.inv(np.asarray(cameras[camera]['camera_to_world'])))
            projected=camera_points@np.asarray(cameras[camera]['intrinsic']).T
            uv=projected[:,:2]/projected[:,2,None]
            a,b,c,d=region;keep=(camera_points[:,2]>0)&(uv[:,0]>=a)&(uv[:,0]<=c)&(uv[:,1]>=b)&(uv[:,1]<=d)
            selected=candidate[keep]
            feature_source='current observed endpoints of the explicitly associated grounded handle, inside the requested region'
        else:
            camera_points=transform_points(selected,np.linalg.inv(np.asarray(cameras[camera]['camera_to_world'])))
            front=np.quantile(camera_points[:,2],.10)
            selected=selected[camera_points[:,2]<=front+.025]
            feature_source='nearest supported 25 mm depth band in the selected feature region; inspect preview for identity'
        if len(selected)<20:raise SkillStop('insufficient_hook_depth',feature_source=feature_source,observed_points=len(selected))
        multiview_selected=occupied[cKDTree(selected).query(occupied,distance_upper_bound=.012)[0]<.012]
        # When a grounded handle is available its spatial association also
        # rejects nearby panel/background points from other views.
        if joint_scene is not None and joint_degrees is not None:
            multiview_selected=multiview_selected[cKDTree(joint_scene.contact_reference).query(multiview_selected,distance_upper_bound=.004)[0]<.004]
        origin=selected[np.argmin(np.linalg.norm(selected-np.median(selected,axis=0),axis=1))]
        patch['point_world_m']=origin.tolist()
    if joint_degrees is not None:
        if joint_scene is None:raise SkillStop('joint_model_required')
        distance=float(cKDTree(joint_scene.moving_reference).query([origin])[0][0])
        if distance>.015:raise SkillStop('target_joint_association_failed',distance_m=distance,maximum_m=.015)
        associated=True;target=np.asarray(joint_scene.moving_reference)
        direction=np.sign(joint_degrees)*np.cross(joint_scene.axis,origin-joint_scene.origin)
    d=unit(direction if direction is not None else -normal)
    spec=copy.deepcopy(joint_scene.spec) if associated else {
        'joint_origin_world_m':origin.tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':origin.tolist()},'panel_normal_world':normal.tolist()}
    # Only target-associated measured endpoints move. Other observed surfaces
    # remain stationary obstacles even when inside a loose image selection.
    owned=(cKDTree(target).query(occupied,distance_upper_bound=np.nextafter(.004,np.inf))[0]<=.004)|(cKDTree(contact_points).query(occupied,distance_upper_bound=np.nextafter(.0035,np.inf))[0]<=.0035)
    if skill_kind=='push' and not associated:
        owned|=projected_surface_membership(occupied,cameras[camera],target_region or region,origin,normal)
    target=occupied[owned]
    if len(target)<12:raise SkillStop('insufficient_target_depth')
    scene=Scene.from_points(occupied[~owned],target,contact_points,spec,
        config=copy.deepcopy(joint_scene.config) if associated else {},
        robot=RobotVolumes(chain,metadata['robot_collision_volumes_world'],metadata['joint_positions']),
        reference_volumes=metadata['robot_collision_volumes_world'])
    scene.skill_contact_surface=contact_surface
    scene.skill_kind=skill_kind
    scene.contact_policy_kind='contact_skill'
    scene.skill_observed_cameras=cameras
    scene.skill_joint_sign=float(np.sign(joint_degrees)) if associated else None
    return scene,{**patch,'direction_world':d.tolist(),'joint_associated':associated,
        'contact_surface':contact_surface,'target_observed_points':len(target),'feature_source':feature_source,
        'contact_permission_scope':'selected moving part; sliding allowed' if skill_kind=='push' else 'selected retaining feature',
        'permitted_contact':'distal fingertips/pads'+(' and selected palm' if contact_surface=='palm' else ''),
        'target_association':('projected selected region across observed views, within the existing 15 mm local-plane band and 45 cm extent'
            if skill_kind=='push' and not associated else 'selected region, within 15 mm of the observed local contact plane and 45 cm of its centre'),
        'proximal_fingers_and_arm_checked':True,'target_region_xyxy':target_region or region},multiview_selected


def contact_landmarks(metadata,contact_surface,hook=False):
    inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']));result=[]
    for volume in metadata['robot_collision_volumes_world']:
        name=volume['robot_geom_name'];p=transform_points(volume['vertices_world'],inverse)
        if name.endswith('_pad_collision'):
            # Facing axis of the pad is toward the other finger, along TCP x.
            sign=-np.sign(p[:,0].mean());normal=np.array([sign,0.,0.])
            point=p.mean(0);point[0]=p[:,0].max() if sign>0 else p[:,0].min()
            result.append({'name':name,'point':point,'normal':normal})
        elif 'finger' in name and not hook:
            tip=p[p[:,2]>=p[:,2].max()-.003].mean(0)
            result.append({'name':name+'_distal_tip','point':tip,'normal':np.array([0.,0.,1.])})
        elif name=='gripper0_hand_collision' and contact_surface=='palm' and not hook:
            for axis in [1,2]:
                for sign in [-1,1]:
                    values=p[:,axis]*sign;point=p[values>=values.max()-.003].mean(0)
                    normal=np.eye(3)[axis]*sign
                    result.append({'name':name+f'_face_{axis}_{sign}','point':point,'normal':normal})
    # Point-first tips are the natural default; pads remain useful for hooks.
    return sorted(result,key=lambda x:('hand' not in x['name'] if contact_surface=='palm' else 'distal_tip' not in x['name']))


def orientation_menu(contact_normal,world_force,current_rotation):
    normal=unit(contact_normal);force=unit(world_force)
    cross=np.cross(normal,force);sine=np.linalg.norm(cross);cosine=float(np.clip(normal@force,-1,1))
    if sine>1e-8:base=Rotation.from_rotvec(cross/sine*np.arctan2(sine,cosine)).as_matrix()
    elif cosine>0:base=np.eye(3)
    else:
        axis=np.cross(normal,np.eye(3)[np.argmin(abs(normal))]);axis/=np.linalg.norm(axis)
        base=Rotation.from_rotvec(axis*np.pi).as_matrix()
    rows=[Rotation.from_rotvec(force*a).as_matrix()@base for a in np.arange(8)*np.pi/4]
    return sorted(rows,key=lambda r:Rotation.from_matrix(current_rotation.T@r).magnitude())


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


def robot_only_planning_scene(scene):
    """Separate advisory planning context; never changes observed or physical geometry."""
    planning=copy.copy(scene)
    planning.static_points=np.empty((0,3))
    planning.moving_reference=np.empty((0,3))
    # The measured contact reference remains metadata. NativeScene indexes
    # only moving endpoints, so this introduces no synthetic obstacle points.
    return planning


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


def cartesian_path(ik,scene,record,metadata,start,target_pose,budget,*,displacement=None,angle=0.,timing=None,contact_local=None,allow_partial=False):
    """Short Cartesian continuation with collision checks at dense joint samples.

    Never substitute an unconstrained RRT detour for a contact stroke.
    """
    timing=timing if timing is not None else Counter();begin=pose_matrix(ik.chain.forward(start))
    target=pose_matrix(target_pose);distance=np.linalg.norm(target[:3,3]-begin[:3,3])
    rotation=Rotation.from_matrix(target[:3,:3]@begin[:3,:3].T).as_rotvec()
    count=max(1,int(np.ceil(distance/.003)),int(np.ceil(np.linalg.norm(rotation)/.04)))
    path=[np.asarray(start)]
    def prefix(reason):
        scene.skill_partial_plan={'reason':reason,'completed_waypoints':len(path)-1,'requested_waypoints':count,
            'global_infeasibility_proved':False}
        return np.asarray(path)
    for fraction in np.linspace(0,1,count+1)[1:]:
        if allow_partial and len(path)>1 and budget.remaining<=0:return prefix('planning_time_limit')
        budget.check();pose=begin.copy();pose[:3,3]+=fraction*(target[:3,3]-begin[:3,3])
        pose[:3,:3]=Rotation.from_rotvec(fraction*rotation).as_matrix()@begin[:3,:3]
        if contact_local is not None:
            contact_local=np.asarray(contact_local)
            first=begin[:3,3]+begin[:3,:3]@contact_local
            last=target[:3,3]+target[:3,:3]@contact_local
            pose[:3,3]=first+fraction*(last-first)-pose[:3,:3]@contact_local
        began=time.perf_counter();solutions=ik.solve_many(matrix_pose(pose),path[-1],maximum=8)
        timing['ik_seconds']+=time.perf_counter()-began
        if allow_partial and len(path)>1 and budget.remaining<=0:return prefix('planning_time_limit')
        budget.check()
        valid=None;rejects=[]
        solutions=sorted(solutions,key=lambda s:np.linalg.norm(np.asarray(s['joints'])-path[-1]))
        if allow_partial and getattr(scene,'skill_collision_policy','veto')=='advisory' and solutions:
            # Physical push execution trusts the robot-valid IK references.
            # Do not add a second dense interpolation/admission pass; actual
            # contact and directional stalling are measured during execution.
            path.append(np.asarray(solutions[0]['joints']));continue
        for solution in solutions:
            q=np.asarray(solution['joints']);delta=q-path[-1]
            # Dense checks include every interpolation interval; no wrist-step
            # veto. Larger branch changes simply consume more checks/budget.
            checks=max(2,int(np.ceil(np.max(np.abs(delta))/.025)))
            advisory=getattr(scene,'skill_collision_policy','veto')=='advisory'
            predicted=None if advisory else predicted_scene(scene,np.asarray(displacement if displacement is not None else [0,0,0])*fraction,angle*fraction)
            began=time.perf_counter()
            with (nullcontext(None) if advisory else skill_native_scene(predicted,record,metadata,path[-1])) as context:
                accepted=True
                local=np.zeros(3) if contact_local is None else np.asarray(contact_local)
                first=pose_matrix(ik.chain.forward(path[-1]));last=pose_matrix(ik.chain.forward(q))
                first_point=first[:3,3]+first[:3,:3]@local
                last_point=last[:3,3]+last[:3,:3]@local
                for u in np.linspace(0,1,checks+1):
                    if allow_partial and len(path)>1 and budget.remaining<=0:return prefix('planning_time_limit')
                    budget.check();sample=path[-1]+delta*u
                    # Endpoint IK alone does not make joint interpolation a
                    # short Cartesian stroke. A different IK branch can swing
                    # the contact landmark far from the intended segment.
                    desired=(1-u)*first_point+u*last_point
                    measured=pose_matrix(ik.chain.forward(sample));error=float(np.linalg.norm(measured[:3,3]+measured[:3,:3]@local-desired))
                    if error>.006:
                        if advisory:
                            log=getattr(scene,'skill_path_deviation_log',None)
                            if log is not None:
                                log['maximum_m']=max(log['maximum_m'],error);log['samples_over_threshold']+=1
                        else:
                            rejects.append({'accepted':False,'reason':'contact_path_deviation','position_error_m':error,'maximum_m':.006})
                            accepted=False;break
                    e=({'accepted':bool(ik.self_valid(sample)),'reason':'robot_only_validity'}
                        if advisory else context.validate(sample))
                    if not e['accepted']:rejects.append(e);accepted=False;break
            timing['collision_seconds']+=time.perf_counter()-began
            if accepted:valid=q;break
        if valid is None:
            if allow_partial and len(path)>1:return prefix('no_next_kinematic_increment')
            raise SkillStop('contact_path_blocked',fraction=float(fraction),ik_solutions=len(solutions),rejections=rejects[-8:])
        path.append(valid)
    return np.asarray(path)


def plan_contact_stroke(ik,scene,record,metadata,contact_local,delta,budget,*,angle=0.,timing=None,physical_feedback=False):
    """Try local wrist alternatives while preserving the contact landmark path."""
    start=np.asarray(metadata['joint_positions']);tcp=pose_matrix(ik.chain.forward(start))
    if physical_feedback:
        scene=copy.copy(scene);scene.skill_collision_policy='advisory'
    point=tcp[:3,3]+tcp[:3,:3]@contact_local;direction=unit(delta)
    transverse=np.cross(direction,np.eye(3)[np.argmin(abs(direction))]);transverse=unit(transverse)
    rotations=[tcp[:3,:3]]
    for axis,degrees in [(direction,15),(transverse,10),(direction,30)]:
        for sign in [-1,1]:rotations.append(Rotation.from_rotvec(axis*np.radians(degrees)*sign).as_matrix()@tcp[:3,:3])
    attempts=[]
    for index,rotation in enumerate(rotations):
        budget.check();goal=tcp.copy();goal[:3,:3]=rotation;goal[:3,3]=point+delta-rotation@contact_local
        scene.skill_partial_plan=None
        try:
            path=cartesian_path(ik,scene,record,metadata,start,matrix_pose(goal),budget,
                displacement=delta,angle=angle,timing=timing,contact_local=contact_local,allow_partial=physical_feedback)
            return path,{'orientation_candidate':index,'earlier_attempts':attempts,
                'reachable_prefix':scene.skill_partial_plan,'physical_feedback':physical_feedback}
        except SkillStop as error:
            attempts.append({'orientation_candidate':index,'reason':error.reason,**error.evidence})
            if error.reason=='budget_exhausted':raise SkillStop('stroke_planning_budget_exhausted',orientation_attempts=attempts,search_complete=False)
    raise SkillStop('contact_path_blocked',orientation_attempts=attempts,global_infeasibility_proved=False)


def region_contact_candidates(points,seed,maximum=5):
    """Sample measured interior points across both surface dimensions."""
    points=np.asarray(points,float);center=np.median(points,axis=0)
    _,_,basis=np.linalg.svd(points-center,full_matrices=False)
    local=(points-center)@basis.T;tree=cKDTree(points)
    targets=[np.asarray(seed)]
    for axis in range(2):
        for fraction in [.2,.8]:
            targets.append(center+basis[axis]*np.quantile(local[:,axis],fraction))
    result=[]
    for target in targets:
        point=points[tree.query(target)[1]]
        if not any(np.linalg.norm(point-p)<.008 for p in result):result.append(point)
    return result[:maximum]


def current_contact_candidate(metadata,scene,direction,contact_surface,kind):
    """Find a supported current contact posture; engagement remains unverified."""
    tcp=pose_matrix(metadata['eef_pose_world_xyzw']);tree=cKDTree(scene.contact_reference)
    candidates=[]
    for contact in contact_landmarks(metadata,contact_surface,hook=kind=='hook_pull'):
        if (tcp[:3,:3]@contact['normal'])@unit(direction)<.1:continue
        point=tcp[:3,3]+tcp[:3,:3]@contact['point'];distance,index=tree.query(point)
        if distance<=.006:candidates.append((distance,contact,scene.contact_reference[index]))
    if not candidates:return None
    distance,contact,point=min(candidates,key=lambda row:row[0])
    return {'contact':contact,'point':point,'distance_m':float(distance)}


def plan_engagement(ik,scene,record,metadata,evidence,selected,kind,budget,maximum_approach_controls=None):
    if kind=='hook_pull':
        if getattr(scene,'skill_collision_policy','veto')=='advisory':
            # A hook needs a retaining posture, so search the observed geometry
            # across candidates before taking the first robot-valid posture.
            # Both passes share the original deadline. Failed scene predictions
            # still leave an advisory physical experiment available.
            started=time.perf_counter();preferred=copy.copy(scene);preferred.skill_collision_policy='veto'
            preferred_evidence=copy.deepcopy(evidence)
            try:
                result=_plan_contact_at_point(ik,preferred,record,metadata,preferred_evidence,selected,kind,
                    budget.child(budget.remaining/2),maximum_approach_controls)
            except SkillStop as error:
                preference={'selected':False,'seconds':time.perf_counter()-started,'reason':error.reason,
                    'environment_prediction_is_veto':False}
                result=_plan_contact_at_point(ik,scene,record,metadata,evidence,selected,kind,budget,maximum_approach_controls)
            else:
                evidence.update(preferred_evidence);scene.contact_reference=preferred.contact_reference.copy()
                preference={'selected':True,'seconds':time.perf_counter()-started,
                    'environment_prediction_is_veto':False}
            result['observed_hook_posture_preference']=preference
            result['total_seconds']=time.perf_counter()-started
            return result
        return _plan_contact_at_point(ik,scene,record,metadata,evidence,selected,kind,budget,maximum_approach_controls)
    started=time.perf_counter();points=region_contact_candidates(selected,evidence['point_world_m'])
    attempts=[];timing=Counter();work=Counter()
    for index,point in enumerate(points):
        if budget.remaining<=0:break
        local=copy.deepcopy(evidence);local['point_world_m']=point.tolist()
        if local['joint_associated']:
            local['direction_world']=unit(scene.skill_joint_sign*np.cross(scene.axis,point-scene.origin)).tolist()
        allowance=min(budget.remaining,max(6.,budget.remaining/(len(points)-index)))
        try:
            result=_plan_contact_at_point(ik,scene,record,metadata,local,selected,kind,budget.child(allowance),maximum_approach_controls)
            timing.update(result['timing_seconds']);evidence.update(local)
            work.update(result.get('search_work',{}))
            evidence['region_contact_candidates_world_m']=[p.tolist() for p in points]
            result.update(region_candidate_index=index,point_attempts=attempts,
                timing_seconds=dict(timing),search_work=dict(work),total_seconds=time.perf_counter()-started)
            return result
        except SkillStop as error:
            timing.update(error.evidence.get('timing_seconds',{}))
            work.update(error.evidence.get('search_work',{}))
            attempts.append({'point_world_m':point.tolist(),'reason':error.reason,**error.evidence})
            if error.reason in {'start_configuration_blocked','initial_target_contact_requires_release'}:raise
    raise SkillStop('contact_search_budget_exhausted' if budget.remaining<=0 else 'contact_search_exhausted',
        point_attempts=attempts,search_complete=len(attempts)==len(points) and all(a['reason']!='budget_exhausted' for a in attempts),
        global_infeasibility_proved=False,timing_seconds=dict(timing),search_work=dict(work))


def approach_connection(robot_context,observed_context,start,goal,budget,timing):
    """Prefer an observed clear approach; failed predictions never veto fallback."""
    selection={'environment_prediction_is_veto':observed_context is None}
    if observed_context is not None:
        began=time.perf_counter()
        before=observed_context.validate(start);after=observed_context.validate(goal)
        selection['observed_endpoints_accepted']=bool(before['accepted'] and after['accepted'])
        path=None;result={}
        if selection['observed_endpoints_accepted'] and budget.remaining>2.:
            path,result=observed_context.plan(start,goal,constrained=False,seconds=min(2.,budget.remaining))
        timing['observed_approach_preference_seconds']+=time.perf_counter()-began
        if result.get('exact_solution'):
            return path,result,{**selection,'selected':'observed_clear_approach'}
        selection['observed_preference_succeeded']=False
    budget.check();began=time.perf_counter()
    path,result=robot_context.plan(start,goal,constrained=False,seconds=min(2.,budget.remaining))
    elapsed=time.perf_counter()-began
    timing['connection_search_seconds']+=result.get('solve_seconds',0.)
    timing['connection_dense_validation_seconds']+=result.get('resampling_seconds',0.)
    timing['connection_other_seconds']+=max(0.,elapsed-result.get('solve_seconds',0.)-result.get('resampling_seconds',0.))
    return path,result,{**selection,'selected':'robot_valid_advisory_approach' if observed_context is not None else 'observed_scene_plan'}


def contact_pose_menu(metadata,evidence,point,d,kind,feature=None):
    current=pose_matrix(metadata['eef_pose_world_xyzw'])[:3,:3]
    menu=[]
    if kind=='push':
        # Preserve reachable wrist postures as candidates. A fingertip can push
        # with an edge/side; requiring its long axis to equal the push direction
        # unnecessarily recreates the old overconstrained TCP interface.
        inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
        rotations=[current]
        for axis in np.eye(3):
            for angle in [-np.pi/4,np.pi/4]:rotations.append(Rotation.from_rotvec(axis*angle).as_matrix()@current)
        for volume in metadata['robot_collision_volumes_world']:
            name=volume['robot_geom_name'];is_palm=name=='gripper0_hand_collision'
            if ('finger' not in name or name.endswith('_pad_collision')) and not (is_palm and evidence['contact_surface']=='palm'):continue
            vertices=transform_points(volume['vertices_world'],inverse)
            if not is_palm:vertices=vertices[vertices[:,2]>=vertices[:,2].max()-.012]
            for rotation in rotations:
                projection=vertices@(rotation.T@d)
                landmark=vertices[projection>=projection.max()-.0005].mean(0)
                contact={'name':name+('_palm_support' if is_palm else '_tip_support'),'point':landmark,'normal':rotation.T@d}
                target=np.eye(4);target[:3,:3]=rotation;target[:3,3]=point-rotation@landmark
                for gap in [.015,0.,.035]:
                    standoff=target.copy();standoff[:3,3]-=gap*d
                    menu.append((contact,target,standoff,d))
    for contact in contact_landmarks(metadata,evidence['contact_surface'],hook=kind=='hook_pull'):
        for rotation in orientation_menu(contact['normal'],d,current):
            # Hook approaches laterally, with the finger along the insertion
            # direction. Pulling begins only after this complete insertion.
            approach=rotation[:,2] if kind=='hook_pull' else d
            if kind=='hook_pull' and abs(approach@d)>.3:continue
            if kind=='hook_pull' and abs(approach@np.asarray(feature['retaining_axis_world']))>.4:
                # Insert across a retaining bar. Sliding a long finger along
                # the bar can put its proximal body in the subsequent pull.
                continue
            target=np.eye(4);target[:3,:3]=rotation
            target[:3,3]=point-rotation@contact['point']
            standoff=target.copy();standoff[:3,3]-=.035*approach
            menu.append((contact,target,standoff,approach))
    current_position=np.asarray(metadata['eef_pose_world_xyzw'])[:3]
    menu.sort(key=lambda row:np.linalg.norm(row[1][:3,3]-current_position)+.04*Rotation.from_matrix(current.T@row[1][:3,:3]).magnitude())
    return menu


def _plan_contact_at_point(ik,scene,record,metadata,evidence,selected,kind,budget,maximum_approach_controls=None):
    started=time.perf_counter();timing=Counter();work=Counter();attempts=[];q=np.asarray(metadata['joint_positions'])
    approach_speed=float(getattr(scene,'skill_approach_speed_rad_s',.6))
    approach_hold=int(getattr(scene,'skill_approach_hold_steps',0))
    budget.context={'attempts':attempts,'timing_seconds':timing,'search_work':work}
    point=np.asarray(evidence['point_world_m']);d=unit(evidence['direction_world']);normal=unit(evidence['normal_toward_camera_world'])
    if kind=='push' and d@(-normal)<.1:
        alignment={'force_alignment':float(d@(-normal)),'usual_compressive_alignment_minimum':.1,
            'interpretation':'Requested motion points away from or nearly tangent to the estimated visible face.'}
        if getattr(scene,'skill_collision_policy','veto')=='advisory':
            evidence['surface_alignment_prediction']=alignment
        else:raise SkillStop('wrong_side_contact',**alignment)
    if kind=='hook_pull':
        point,feature=hook_feature(selected,d,cameras=getattr(scene,'skill_observed_cameras',None));evidence['hook_feature']=feature
        scene.contact_reference=scene.moving_reference[cKDTree(np.asarray([point])).query(scene.moving_reference)[0]<.035]
        if len(scene.contact_reference)<5:raise SkillStop('hook_back_surface_unobserved')
    if callable(maximum_approach_controls):maximum_approach_controls=maximum_approach_controls(point)
    current=pose_matrix(metadata['eef_pose_world_xyzw'])[:3,:3]
    setup_started=time.perf_counter()
    advisory=getattr(scene,'skill_collision_policy','veto')=='advisory'
    planning_scene=robot_only_planning_scene(scene) if advisory else scene
    with skill_native_scene(planning_scene,record,metadata,q) as context, \
         skill_native_scene(planning_scene,record,metadata,q,allow_contact=False) as free_context, \
         (skill_native_scene(scene,record,metadata,q,allow_contact=False) if advisory else nullcontext(None)) as observed_context:
        context.record['planning_scope']='robot-only advisory simulation plan' if advisory else 'observed-scene collision checked'
        free_context.record['planning_scope']=context.record['planning_scope']
        timing['scene_context_setup_seconds']+=time.perf_counter()-setup_started
        initial=context.validate(q)
        if not initial['accepted']:raise SkillStop('start_configuration_blocked',diagnostic=initial)
        existing=current_contact_candidate(metadata,scene,d,evidence['contact_surface'],kind)
        if existing is not None:
            empty=np.empty((0,7));contact=existing['contact']
            evidence['point_world_m']=existing['point'].tolist()
            if evidence['joint_associated']:
                d=unit(scene.skill_joint_sign*np.cross(scene.axis,existing['point']-scene.origin))
                evidence['direction_world']=d.tolist()
            return {'path':np.vstack([q,q]),'references':empty,'free_references':empty,'insertion_references':empty,
                'contact_local':contact['point'],'contact_world':existing['point'],
                'rotation':current,'insertion_direction':d,'contact_name':contact['name'],
                'attempts':attempts,'candidate_index':None,'timing_seconds':dict(timing),
                'total_seconds':time.perf_counter()-started,'predicted_contact_only':True,
                'reuse_existing_contact':True,'measured_surface_distance_m':existing['distance_m'],
                'native_contact_policy':context.record}
        free_initial=free_context.validate(q)
        if not free_initial['accepted']:raise SkillStop('initial_target_contact_requires_release',diagnostic=free_initial)
        menu=contact_pose_menu(metadata,evidence,point,d,kind,feature if kind=='hook_pull' else None)
        target_cache={}
        for index,(contact,target,standoff,approach) in enumerate(menu):
            budget.check();work['pose_approach_combinations']+=1
            key=target.tobytes();reused=key in target_cache
            if reused:
                endpoints,endpoint_checks,first=target_cache[key];work['target_ik_reuses']+=1
            else:
                began=time.perf_counter();endpoints=ik.solve_many(matrix_pose(target),q,maximum=8)
                timing['ik_seconds']+=time.perf_counter()-began;work['target_ik_queries']+=1;budget.check()
                check_started=time.perf_counter();endpoint_checks=[context.validate(s['joints']) for s in endpoints]
                timing['goal_collision_checks_seconds']+=time.perf_counter()-check_started
                first=index;target_cache[key]=(endpoints,endpoint_checks,first)
            if not any(e['accepted'] for e in endpoint_checks):
                attempts.append({'candidate':index,'reason':'no_contact_ik' if not endpoints else 'contact_endpoint_collision',
                    'contact_name':contact['name'],**({'reused_endpoint_candidate':first} if reused else {'endpoint_checks':endpoint_checks})})
                continue
            began=time.perf_counter()
            solutions=ik.solve_many(matrix_pose(standoff),q,maximum=8)
            timing['ik_seconds']+=time.perf_counter()-began;work['standoff_ik_queries']+=1;budget.check()
            if not solutions:attempts.append({'candidate':index,'reason':'no_standoff_ik'});continue
            for solution in solutions:
                budget.check();goal=np.asarray(solution['joints'])
                lower_bound=(int(np.ceil(np.max(np.abs(goal-q))/approach_speed*20))+approach_hold
                    +int(np.ceil(np.linalg.norm(target[:3,3]-standoff[:3,3])/.03*20))+3)
                if maximum_approach_controls is not None and lower_bound>maximum_approach_controls:
                    attempts.append({'candidate':index,'reason':'approach_lower_bound_exceeds_control_budget',
                        'minimum_controls':lower_bound,'available_controls':maximum_approach_controls})
                    continue
                endpoint=free_context.validate(goal)
                if not endpoint['accepted']:
                    attempts.append({'candidate':index,'reason':'standoff_collision','diagnostic':endpoint});continue
                approach_path,result,approach_selection=approach_connection(free_context,observed_context,q,goal,budget,timing)
                if not result.get('exact_solution'):
                    attempts.append({'candidate':index,'reason':'approach_search_incomplete','diagnostic':result});continue
                try:
                    insert=cartesian_path(ik,scene,record,metadata,goal,matrix_pose(target),budget,timing=timing)
                except SkillStop as error:
                    if error.reason=='budget_exhausted':raise
                    attempts.append({'candidate':index,'reason':error.reason,**error.evidence});continue
                path=np.vstack([approach_path,insert[1:]])
                free_refs=schedule(approach_path,maximum_joint_speed=approach_speed,endpoint_hold_steps=approach_hold)
                insertion_refs=schedule_contact(insert,ik.chain,contact['point'],contact_speed_m_s=.03,endpoint_hold_steps=3)
                refs=np.vstack([free_refs,insertion_refs])
                if maximum_approach_controls is not None and len(refs)>maximum_approach_controls:
                    attempts.append({'candidate':index,'reason':'approach_exceeds_control_budget',
                        'required_controls':len(refs),'available_controls':maximum_approach_controls})
                    continue
                return {'path':path,'references':refs,'contact_local':contact['point'],
                    'free_references':free_refs,'insertion_references':insertion_refs,
                    'contact_tcp_pose_world_xyzw':matrix_pose(target),
                    'standoff_tcp_pose_world_xyzw':matrix_pose(standoff),
                    'contact_world':point,'rotation':target[:3,:3],'insertion_direction':approach,
                    'approach_selection':approach_selection,
                    'contact_name':contact['name'],'attempts':attempts,'candidate_index':index,
                    'timing_seconds':dict(timing),'search_work':dict(work),'total_seconds':time.perf_counter()-started,
                    'predicted_contact_only':True,'native_contact_policy':context.record}
    raise SkillStop('contact_search_exhausted',attempts=attempts,search_complete=True,
        candidates=len(menu),global_infeasibility_proved=False,timing_seconds=dict(timing),search_work=dict(work))


def fit_object_motion(first,second):
    """Robust 3D rigid fit to independently tracked, measured RGB-D pairs."""
    a=np.asarray(first,float).reshape(-1,3);b=np.asarray(second,float).reshape(-1,3)
    finite=np.isfinite(a).all(1)&np.isfinite(b).all(1);a,b=a[finite],b[finite]
    result={'accepted':False,'pairs':len(a),'source':'RGB-D feature correspondences','simulator_object_state_used':False}
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
    if best.sum()<8 or best.mean()<.6:return {**result,'reason':'inconsistent_matches','inliers':int(best.sum())}
    r,t=fit(a[best],b[best]);error=np.linalg.norm(a[best]@r.T+t-b[best],axis=1)
    span=float(np.linalg.norm(np.ptp(a[best],axis=0)))
    if span<.025:return {**result,'reason':'insufficient_spatial_baseline','span_m':span}
    transform=np.eye(4);transform[:3,:3]=r;transform[:3,3]=t
    return {**result,'accepted':True,'reason':'measured_rigid_motion','inliers':int(best.sum()),
        'rms_m':float(np.sqrt(np.mean(error**2))),'span_m':span,'transform':transform.tolist()}


class ContactTracker:
    def __init__(self,cameras,target,step):
        from .rigid_hinge_tracking import RGBDHingeTracker
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
