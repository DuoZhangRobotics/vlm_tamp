"""Planning-only MC-VAMP adapter for a perceived microwave and Panda gripper."""
import ctypes
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import weakref

import numpy as np
from scipy.spatial import cKDTree,HalfspaceIntersection
from scipy.spatial.transform import Rotation

from .rekep_libero.geometry import pose_matrix,transform_points
from .rekep_libero.robot_chain_v4 import RobotChain
from .coupled_storage import preserve_scene,write_json
from .rekep_libero.scene_state_v7 import voxel_points

ROOT=Path(__file__).resolve().parents[2]
LIBRARY=ROOT/'.runtime/mcvamp-contact-numerical/libmcvamp_contact_numerical.so'
SOURCE=ROOT/'src/vlm_tamp/native/mcvamp_contact_numerical.cc'
FLOAT=ctypes.POINTER(ctypes.c_float)
DOUBLE=ctypes.POINTER(ctypes.c_double)
INTEGER=ctypes.POINTER(ctypes.c_int)
from .native_scene_cache import NativeSceneCache
_TREE_CACHE=NativeSceneCache()
JOINT_ADMISSION_RESERVE_RAD=0.  # Actual robot bounds; no added joint-limit reserve.


def native_plan_status(count,stats):
    """Deadline expiry during validation is not evidence of a collision."""
    if count>1:return 'solved'
    if stats.get('timed_out'):
        return 'dense_validation_timeout' if count in {-2,-3,-4,-5} else 'search_timeout'
    return {0:'no_path',-2:'dense_path_capacity',-3:'dense_projection_failed',-4:'dense_scene_collision',
        -5:'dense_goal_collision',-6:'planner_path_capacity',-9:'native_exception'}.get(count,str(count))


def cleanup_worker_dumps(pid,started_at,*,folder=None,executable=None):
    """Remove only a crashed worker's own WSL dump, preserving other programs."""
    folder=Path(folder or '/mnt/c/Users/duo/AppData/Local/Temp/wsl-crashes')
    tag=str(Path(executable or sys.executable).resolve()).replace('/','_')
    removed=[]
    if not folder.is_dir():return removed
    for path in folder.glob(f'wsl-crash-*-{int(pid)}-{tag}-*.dmp'):
        if path.is_symlink() or not path.is_file() or path.stat().st_mtime<started_at-1.:continue
        record={'path':str(path),'bytes':path.stat().st_size,'worker_pid':int(pid)}
        try:path.unlink();record['removed']=True
        except OSError as error:record.update(removed=False,error=str(error))
        removed.append(record)
    return removed


def clear_native_cache():
    _TREE_CACHE.clear()


def _f(array):return np.ascontiguousarray(array,dtype=np.float32)
def _pointer(array):return array.ctypes.data_as(FLOAT)


def calibrated_frames(record):
    import vamp
    chain=RobotChain(record)
    world_robot=np.linalg.inv(np.asarray(record['calibration']['world_to_robot']))
    native=np.asarray(vamp.panda.eefk(chain.reference_joints),float)[0]
    native_tcp=np.linalg.inv(native)@np.linalg.inv(world_robot)@chain.reference_tcp
    errors=[]
    rng=np.random.default_rng(0)
    for q in np.vstack([chain.reference_joints,rng.uniform(chain.limits[:,0]+.1,chain.limits[:,1]-.1,(8,7))]):
        expected=pose_matrix(chain.forward(q))
        computed=world_robot@np.asarray(vamp.panda.eefk(q),float)[0]@native_tcp
        errors.append([float(np.linalg.norm(expected[:3,3]-computed[:3,3])),
            float(Rotation.from_matrix(expected[:3,:3].T@computed[:3,:3]).magnitude())])
    if np.max(errors,axis=0)[0]>.001 or np.max(errors,axis=0)[1]>.005:
        raise ValueError('MC-VAMP robot/TCP calibration disagrees with the configured Panda')
    return world_robot,native_tcp,{'samples':len(errors),'maximum_position_error_m':float(np.max(errors,axis=0)[0]),
        'maximum_orientation_error_rad':float(np.max(errors,axis=0)[1]),'robot_only_calibration':True}


def gripper_cover(metadata,world_robot,*,spacing=.012):
    """Cover configured convex gripper volumes with small enclosing cell balls.

    This replaces upstream's fixed open hand. A cell is retained only when its
    enclosing ball intersects the configured convex geometry. No scene mesh or
    simulator object identity is read. The approximation is deliberately logged.
    """
    import vamp
    eef=world_robot@np.asarray(vamp.panda.eefk(metadata['joint_positions']),float)[0]
    inverse=np.linalg.inv(eef)
    rows=[];records=[]
    for volume in metadata['robot_collision_volumes_world']:
        name=volume['robot_geom_name']
        if 'gripper' not in name:continue
        if volume['type']!='convex':raise ValueError('Configured gripper needs convex collision geometry')
        vertices=np.asarray(volume['vertices_world'],float)
        centre=vertices.mean(0)
        _,basis=np.linalg.eigh(np.cov((vertices-centre).T))
        coordinates=(vertices-centre)@basis
        low,high=coordinates.min(0),coordinates.max(0)
        counts=np.maximum(1,np.ceil((high-low)/spacing).astype(int))
        widths=(high-low)/counts
        radius=float(np.linalg.norm(widths)/2+.00005)
        local=np.asarray(list(itertools.product(*[low[j]+(np.arange(counts[j])+.5)*widths[j] for j in range(3)])))
        world=local@basis.T+centre
        planes=np.asarray(volume['planes_world'],float)
        keep=np.max(world@planes[:,:3].T+planes[:,3],axis=1)<=radius
        points=transform_points(world[keep],inverse)
        first=len(rows)
        rows.extend([*point,radius,float('finger' in name)] for point in points)
        records.append({'robot_geom_name':name,'sphere_range':[first,len(rows)],'sphere_radius_m':radius,
            'configured_vertices':len(vertices),'selected_handle_contact_allowed':'finger' in name})
    if not rows:raise ValueError('No configured gripper collision volumes')
    return np.asarray(rows,float),records


def gripper_convex(metadata,world_robot,parts,point_radius,contact_surface=None):
    """Known robot convex geometry in native EEF coordinates, without scene IDs."""
    import vamp
    eef=world_robot@np.asarray(vamp.panda.eefk(metadata['joint_positions']),float)[0]
    inverse=np.linalg.inv(eef)
    volumes={v['robot_geom_name']:v for v in metadata['robot_collision_volumes_world']}
    planes=[];offsets=[0];bounds=[]
    # Contact skills permit only distal finger geometry (including pad/body
    # overlap). Legacy grasp callers keep their original full-finger policy.
    tcp=pose_matrix(metadata['eef_pose_world_xyzw'])
    tcp_in_eef=inverse@tcp
    pad_vertices=[transform_points(v['vertices_world'],np.linalg.inv(tcp)) for v in volumes.values()
                  if v['robot_geom_name'].endswith('_pad_collision')]
    tip_start=min(p[:,2].min() for p in pad_vertices)-.001 if pad_vertices else 0.
    for part in parts:
        volume=volumes[part['robot_geom_name']]
        world=np.asarray(volume['planes_world'],float)
        local=np.column_stack([world[:,:3]@eef[:3,:3],world[:,3]+world[:,:3]@eef[:3,3]])
        local/=np.linalg.norm(local[:,:3],axis=1)[:,None]
        vertices=transform_points(volume['vertices_world'],inverse)
        allowed=part['selected_handle_contact_allowed']
        pieces=[(local,vertices,allowed)]
        name=part['robot_geom_name']
        if contact_surface is not None:
            if contact_surface not in {'fingertips','palm','grasped_fingers'}:raise ValueError('Unknown contact surface')
            if contact_surface=='grasped_fingers':
                # Closure establishes contact along the selected handle. Finger
                # bodies may share that contact; palm/arm and other surfaces do not.
                pieces=[(local,vertices,'finger' in name)]
            elif 'finger' in name and not name.endswith('_pad_collision'):
                # z_tcp = axis_eef dot (p_eef - tcp_origin_eef).
                axis=tcp_in_eef[:3,2];cut=tip_start+axis@tcp_in_eef[:3,3]
                pieces=[]
                from scipy.optimize import linprog
                for plane,permission in [(np.r_[axis,-cut],False),(np.r_[-axis,cut],True)]:
                    clipped=np.vstack([local,plane])
                    # Chebyshev centre supplies a strictly interior point even
                    # for asymmetric configured convex finger meshes.
                    fit=linprog([0,0,0,-1],A_ub=np.c_[clipped[:,:3],np.ones(len(clipped))],
                        b_ub=-clipped[:,3],bounds=[(None,None)]*3+[(0,None)],method='highs')
                    if not fit.success or fit.x[3]<1e-8:continue
                    points=HalfspaceIntersection(clipped,fit.x[:3]).intersections
                    pieces.append((clipped,points,permission))
            else:
                pieces=[(local,vertices,name.endswith('_pad_collision') or contact_surface=='palm')]
        for local_piece,points,permission in pieces:
            center=points.mean(0)
            expanded=local_piece.copy();expanded[:,3]-=point_radius
            padded_vertices=HalfspaceIntersection(expanded,center).intersections
            radius=float(np.max(np.linalg.norm(padded_vertices-center,axis=1))+.00001)
            planes.extend(local_piece);offsets.append(len(planes))
            bounds.append([*center,radius,float(permission),*part['sphere_range']])
    return _f(planes),np.ascontiguousarray(offsets,dtype=np.int32),_f(bounds)


def arm_convex(metadata,record,world_robot):
    from .coupled_scene import RobotVolumes
    import re
    chain=RobotChain(record)
    reference=RobotVolumes(chain,metadata['robot_collision_volumes_world'],metadata['joint_positions']).forward(chain.reference_joints)
    inverse=np.linalg.inv(world_robot);planes=[];offsets=[0];bounds=[]
    for volume in reference:
        match=re.fullmatch(r'robot0_link([1-7])_collision',volume['robot_geom_name'])
        if not match:continue
        if volume['type']!='convex':raise ValueError('Expected configured convex arm volume')
        world=np.asarray(volume['planes_world']);local=np.c_[world[:,:3]@world_robot[:3,:3],world[:,3]+world[:,:3]@world_robot[:3,3]]
        local/=np.linalg.norm(local[:,:3],axis=1)[:,None]
        vertices=transform_points(volume['vertices_world'],inverse);center=vertices.mean(0)
        planes.extend(local);offsets.append(len(planes));bounds.append([*center,np.linalg.norm(vertices-center,axis=1).max()+1e-5,int(match[1])])
    if len(bounds)<6:raise ValueError('Full configured moving arm geometry is required')
    axes=_f(chain.axes@inverse[:3,:3].T);anchors=_f(transform_points(chain.anchors,inverse))
    return _f(planes),np.asarray(offsets,np.int32),_f(bounds),axes,anchors,_f(chain.reference_joints)


class NativeScene:
    """A process-local CAPT context. It is never passed to the executor."""
    def __init__(self,scene,record,metadata,reference_q,goal_angle,*,articulated=True,aligned=True,
                 allow_contact=True,point_radius=.0025,gripper_padding=0.,output=None,grasp_constraint='point',collision_policy=None):
        if grasp_constraint not in {'point','rigid','jaw_axis'}:
            raise ValueError('Unknown grasp constraint')
        self.grasp_constraint=grasp_constraint
        self.collision_policy=collision_policy or scene.config.get('environment_collision_policy','veto')
        if self.collision_policy not in {'advisory','veto'}:raise ValueError('Unknown collision policy')
        self.position_tolerance_m=float(scene.config.get('arc_position_tolerance_m',.003))
        self.orientation_tolerance_rad=float(scene.config.get('arc_orientation_tolerance_rad',0.))
        if not np.isfinite(self.orientation_tolerance_rad) or not 0<=self.orientation_tolerance_rad<=np.pi:
            raise ValueError('Arc orientation tolerance must be finite and in [0, pi]')
        self.angular_weighted_tolerance=(.15*(self.orientation_tolerance_rad if grasp_constraint=='rigid'
            else 2*np.sin(self.orientation_tolerance_rad/2))
            if grasp_constraint!='point' and self.orientation_tolerance_rad else 1e-6)
        if not np.isfinite(self.position_tolerance_m) or self.position_tolerance_m<=0:
            raise ValueError('Arc position tolerance must be finite and positive')
        if not np.isfinite(gripper_padding) or gripper_padding<0:raise ValueError('Invalid gripper padding')
        build=json.loads((LIBRARY.parent/'build.json').read_text())
        if build['source_sha256']!=hashlib.sha256(SOURCE.read_bytes()).hexdigest():
            raise ValueError('Rebuild MC-VAMP adapter after a source change')
        self.lib=ctypes.CDLL(str(LIBRARY))
        self.lib.mcvamp_create.restype=ctypes.c_void_p
        self.lib.mcvamp_create.argtypes=[FLOAT,ctypes.c_int,FLOAT,ctypes.c_int,FLOAT,ctypes.c_int,FLOAT,FLOAT,ctypes.c_int,ctypes.c_float]
        self.lib.mcvamp_destroy.argtypes=[ctypes.c_void_p]
        self.lib.mcvamp_destroy.restype=None
        self.lib.mcvamp_clone.restype=ctypes.c_void_p
        self.lib.mcvamp_clone.argtypes=[ctypes.c_void_p,FLOAT,FLOAT,ctypes.c_int]
        self.lib.mcvamp_create_reusing_static.restype=ctypes.c_void_p
        self.lib.mcvamp_create_reusing_static.argtypes=[ctypes.c_void_p,FLOAT,ctypes.c_int,FLOAT,ctypes.c_int,FLOAT,FLOAT,ctypes.c_int]
        self.lib.mcvamp_set_gripper_convex.argtypes=[ctypes.c_void_p,FLOAT,INTEGER,FLOAT,ctypes.c_int,ctypes.c_float]
        self.lib.mcvamp_set_arm_convex.argtypes=[ctypes.c_void_p,FLOAT,INTEGER,FLOAT,ctypes.c_int,FLOAT,FLOAT,FLOAT]
        self.lib.mcvamp_set_existing_palm_tolerance.argtypes=[ctypes.c_void_p,ctypes.c_float]
        self.lib.mcvamp_set_position_tolerance.argtypes=[ctypes.c_void_p,ctypes.c_float]
        self.lib.mcvamp_set_orientation_tolerance.argtypes=[ctypes.c_void_p,ctypes.c_float]
        self.lib.mcvamp_constraint_error.argtypes=[ctypes.c_void_p,FLOAT,FLOAT,DOUBLE]
        self.lib.mcvamp_validate.argtypes=[ctypes.c_void_p,FLOAT,DOUBLE]
        self.lib.mcvamp_validate_hand_pose.argtypes=[ctypes.c_void_p,FLOAT,DOUBLE]
        self.lib.mcvamp_project.argtypes=[ctypes.c_void_p,FLOAT,FLOAT,FLOAT,DOUBLE]
        self.lib.mcvamp_plan.argtypes=[ctypes.c_void_p,FLOAT,FLOAT,FLOAT,ctypes.c_int,ctypes.c_int,ctypes.c_double,FLOAT,ctypes.c_int,DOUBLE]
        self.world_robot,self.native_tcp,frames=calibrated_frames(record)
        self.chain=RobotChain(record)
        self.scene=scene
        self.metadata=metadata
        self.reference_q=np.asarray(reference_q,float)
        self.reference_tcp=pose_matrix(self.chain.forward(reference_q))
        robot_world=np.linalg.inv(self.world_robot)
        inverse_tcp=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
        pads=[transform_points(v['vertices_world'],inverse_tcp).mean(0) for v in metadata['robot_collision_volumes_world']
            if v['robot_geom_name'].endswith('_pad_collision')]
        if len(pads)!=2:raise ValueError('Exactly two configured finger pads are required')
        self.pad_local=np.mean(pads,axis=0)
        self.pad_local[2]+=float(scene.config.get('grasp_contact_depth_offset_m',0.))
        self.anchor=self.reference_tcp[:3,3]+self.reference_tcp[:3,:3]@self.pad_local
        self.origin=np.asarray(scene.origin,float);self.axis=np.asarray(scene.axis,float)
        gripper,parts=gripper_cover(metadata,self.world_robot)
        contact_surface=getattr(scene,'skill_contact_surface',None) or 'fingertips'
        if self.collision_policy=='advisory':
            observed_counts={'stationary':len(scene.static_points),'moving':len(scene.moving_reference),'contact_reference':len(scene.contact_reference)}
            static,moving,contact=(_f(np.empty((0,3))) for _ in range(3))
        else:
            from .contact_collision_evidence import selected_contact_points
            selected=selected_contact_points(scene)
            static=_f(transform_points(voxel_points(scene.static_points,.0075),robot_world))
            moving=_f(transform_points(voxel_points(scene.moving_reference[~selected],.005),robot_world))
            # The transported handle reference locates the contact region; it is
            # not a fresh depth measurement. Adding it to the obstacle tree lets
            # submillimetre rigid-fit error invent palm penetration even when the
            # current measured moving surface is clear. Query current observed
            # moving endpoints, with the reference used only for association.
            contact=_f(transform_points(voxel_points(scene.moving_reference[selected],.003),robot_world))
            observed_counts={'stationary':len(static),'moving':len(moving),'contact':len(contact)}
        origin=transform_points([self.origin],robot_world)[0]
        axis=robot_world[:3,:3]@self.axis
        anchor=transform_points([self.anchor],robot_world)[0]
        reserve=float(scene.config.get('native_joint_reserve_rad',JOINT_ADMISSION_RESERVE_RAD))
        limits=self.chain.limits.copy();limits[:,0]+=reserve;limits[:,1]-=reserve
        self.joint_bounds=_f(limits)
        config=_f(np.r_[origin,axis,anchor,self.native_tcp.ravel(),self.pad_local,
            limits[:,0],limits[:,1],float(articulated),float(allow_contact),reference_q,goal_angle,float(aligned)])
        grip=_f(gripper)
        # Point roles are part of the key. Identical concatenated coordinates
        # must not let a stationary obstacle reuse a finger-contact-only tree.
        identity=hashlib.sha256()
        for name,array in [('stationary',static),('moving',moving),('contact',contact)]:
            identity.update(name.encode());identity.update(str((array.shape,array.dtype.str)).encode())
            identity.update(array.tobytes())
        identity.update(str(point_radius).encode())
        static_identity=hashlib.sha256()
        static_identity.update(str((static.shape,static.dtype.str,point_radius)).encode())
        static_identity.update(static.tobytes())
        static_key=static_identity.hexdigest();key=(static_key,identity.hexdigest())
        pinned=None
        if key not in _TREE_CACHE:
            previous=next((value for old,value in reversed(_TREE_CACHE.items()) if old[0]==static_key),None)
            if previous is not None:
                # Pin shared ownership across the cache's eviction-before-build.
                pinned=self.lib.mcvamp_clone(previous[1],_pointer(config),_pointer(grip),len(grip))
                if not pinned:raise RuntimeError('Could not retain stationary collision index')
        before=time.perf_counter()
        try:
            cached,reused=_TREE_CACHE.acquire(key,self.lib,lambda:
                self.lib.mcvamp_create_reusing_static(pinned,_pointer(moving),len(moving),_pointer(contact),len(contact),
                    _pointer(config),_pointer(grip),len(grip)) if pinned else
                self.lib.mcvamp_create(_pointer(static),len(static),_pointer(moving),len(moving),
                    _pointer(contact),len(contact),_pointer(config),_pointer(grip),len(grip),point_radius))
        finally:
            if pinned:self.lib.mcvamp_destroy(pinned)
        self.handle=self.lib.mcvamp_clone(cached[1],_pointer(config),_pointer(grip),len(grip))
        if not self.handle:raise RuntimeError('Native CAPT construction failed')
        self._release=weakref.finalize(self,self.lib.mcvamp_destroy,self.handle)
        if not self.lib.mcvamp_set_position_tolerance(self.handle,self.position_tolerance_m):
            self.close();raise RuntimeError('Could not configure arc position tolerance')
        if not self.lib.mcvamp_set_orientation_tolerance(self.handle,self.orientation_tolerance_rad):
            self.close();raise RuntimeError('Could not configure arc orientation tolerance')
        self.lib.mcvamp_set_grasp_constraint.argtypes=[ctypes.c_void_p,ctypes.c_int]
        if not self.lib.mcvamp_set_grasp_constraint(self.handle,{'point':0,'rigid':1,'jaw_axis':2}[grasp_constraint]):
            self.close();raise RuntimeError('Could not configure grasp constraint')
        planes,offsets,bounds=gripper_convex(metadata,self.world_robot,parts,gripper_padding,contact_surface)
        if not self.lib.mcvamp_set_gripper_convex(self.handle,_pointer(planes),offsets.ctypes.data_as(INTEGER),
                _pointer(bounds),len(bounds),gripper_padding):
            self.close();raise RuntimeError('Calibrated convex gripper installation failed')
        arm_planes,arm_offsets,arm_bounds,axes,anchors,reference=arm_convex(metadata,record,self.world_robot)
        if not self.lib.mcvamp_set_arm_convex(self.handle,_pointer(arm_planes),arm_offsets.ctypes.data_as(INTEGER),_pointer(arm_bounds),len(arm_bounds),_pointer(axes),_pointer(anchors),_pointer(reference)):
            self.close();raise RuntimeError('Configured arm geometry installation failed')
        self.record={'planner':'MC-VAMP CRRTC','upstream_revision':build['upstream_revision'],
            'stationary_index_reused':bool(reused or pinned),'maximum_cached_scene_templates':2,
            'environment_collision_policy':self.collision_policy,'observed_point_counts':observed_counts,
            'physical_simulator_contacts_enabled':True,'environment_collision_free_certificate':False,
            'native_adapter_sha256':build['source_sha256'],'frame_validation':frames,
            'static_points':len(static),'moving_noncontact_points':len(moving),'selected_contact_points':len(contact),
            'point_radius_m':point_radius,'capt_build_seconds':0. if reused else cached[2],
            'context_setup_seconds':time.perf_counter()-before,
            'shared_cloud_index_build_seconds':cached[2],
            'static_voxel_spacing_m':.0075,'moving_voxel_spacing_m':.005,'contact_voxel_spacing_m':.003,
            'capt_radius_bounds_m':[.006,.081],'capt_tree_reused':reused,
            'gripper_cover_spheres':len(grip),'gripper_cover':parts,'gripper_cell_spacing_m':.012,
            'gripper_collision_narrow_phase':'configured convex halfspaces; explicit contact-rich gripper padding',
            'gripper_spheres_are_broad_phase_only':True,'gripper_convex_point_padding_m':gripper_padding,
            'native_arm_spheres':[],'configured_convex_arm_links':len(arm_bounds),'arm_geometry_source':'calibrated robot convex volumes and seven-joint PoE kinematics','fixed_base_omitted':True,
            'moving_cloud_rotates_with_hand_implied_hinge_progress':bool(articulated),
            'inferred_solid_panel_used':False,
            'contact_exemption':'configured fingers versus observed selected bar and bounded local mount region; arm/palm still checked',
            'contact_geometry_source':'current observed moving endpoints; transported handle reference is association only',
            'transported_handle_reference_added_as_obstacle':False,
            'bar_contact_distance_m':.028,'local_mount_distance_m':.065,'local_mount_plane_association_extra_m':.006,
            'contact_pair_penetration_certificate':False,
            'unknown_space_model':'unobserved space is unmodeled',
            'door_environment_collision_checked':False,'simulator_object_geometry_used':False,
            'constraint':{
                'point':'grip point on perceived hinge circle; aligned or bounded-tilt jaws',
                'rigid':'grip point on perceived hinge circle; orientation transported with the articulation',
                'jaw_axis':'grip point on perceived hinge circle; closing-axis direction transported, axial roll free'
            }[grasp_constraint],
            'ambient_dimension':7,'hinge_progress_coordinate':'derived from grip point',
            'projection':'native MC-VAMP composition with adapter finite-difference damped least-squares step',
            'arc_position_tolerance_m':self.position_tolerance_m,
            'joint_bounds_rad':self.joint_bounds.tolist(),'joint_bound_buffer_rad':reserve,
            'arc_orientation_tolerance_rad':self.orientation_tolerance_rad,
            'angular_weighted_tolerance':self.angular_weighted_tolerance,
            'progress_bound_weighted_tolerance':1e-6,
            'constraint_metric':'1e-6 * max(position ratio, orientation ratio, progress-bound ratio)',
            'constraint_metric_units':'scaled_dimensionless_acceptance_ratio',
            'position_tolerance_affects_collision_checks':False,
            'native_simd_projection_fully_preserved':False,'aligned_jaws':bool(aligned)}
        self.record.update(grasp_constraint=grasp_constraint,
            shared_tool_stack='observed-hinged-contact-v1',
            grasp_orientation_reference='current measured TCP transported with the estimated articulation',
            independent_wrist_excursion_limit_used=None,
            wrist_policy_source='Resolved by the planning request, separately from the grasp constraint')
        if getattr(scene,'contact_policy_kind',None)=='observed_finger_patch':
            self.record.update(contact_exemption='configured fingers versus the selected observed patch only; arm/palm and other observed points checked',
                contact_geometry_source='fresh depth-selected bounded patch',
                object_motion_predicted=False,articulation_constraint_used=bool(articulated),
                bar_contact_distance_m=None,local_mount_distance_m=None,
                local_mount_plane_association_extra_m=None)
        if contact_surface is not None:
            self.record.update(contact_exemption='distal fingertips/pads and explicitly selected palm versus selected target patch only',
                skill_contact_surface=contact_surface,proximal_fingers_checked=contact_surface!='grasped_fingers',
                proximal_fingers_checked_against_other_surfaces=True,
                finger_body_target_contact_allowed=contact_surface=='grasped_fingers',
                palm_contact_allowed=contact_surface=='palm',contact_association_m=.0035,
                bar_contact_distance_m=None,local_mount_distance_m=None,
                local_mount_plane_association_extra_m=None)
            if contact_surface=='grasped_fingers':
                self.record['contact_exemption']='finger bodies/pads versus selected handle during grasped manipulation; arm, palm and unrelated surfaces checked'
        update=getattr(scene,'pose_update',None)
        if update is not None:
            self.record.update(contact_geometry_source='observed endpoints transported to the current query frame',
                moving_geometry_pose_source=update['source'],
                moving_geometry_reference_control_step=update['reference_control_step'],
                fresh_depth_endpoints_after_transport=False)
        self.record['current_robot_point_filter']=getattr(scene,'current_robot_filter',None)
        # Only explicit legacy comparisons can carry shallow palm/contact
        # overlap. Live tools check the palm without this tolerance.
        initial=self.validate(reference_q)
        legacy_palm_contact=bool(scene.config.get('allow_existing_palm_contact',False))
        self.record['existing_palm_contact']={'enabled':False,'legacy_opt_in':legacy_palm_contact,
            'maximum_allowed_depth_m':.0005 if legacy_palm_contact else 0.}
        if (legacy_palm_contact and not initial['accepted']
                and initial['last_rejected_part']==1000 and initial['last_rejected_surface']==2):
            from .rekep_libero.scene_state_v7 import bounded_points
            palm=next(v for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
            measured=transform_points(contact,self.world_robot)
            ids,dist=bounded_points(measured,cKDTree(measured),palm,0.)
            depth=float(-dist.min()) if len(ids) else float('inf')
            if 0.<=depth<=.00045:
                tolerance=min(.0005,depth+.00005)
                if not self.lib.mcvamp_set_existing_palm_tolerance(self.handle,tolerance):raise RuntimeError('Invalid contact tolerance')
                self.record['existing_palm_contact'].update(enabled=True,initial_depth_m=depth,
                    tolerance_m=tolerance,selected_contact_points=len(ids),new_penetration_allowance_m=.00005,
                    unrelated_surfaces_or_arm_exempt=False,source='existing configured palm versus current selected depth endpoints')
        if output:
            output=Path(output);output.mkdir(parents=True,exist_ok=True)
            preserve_scene(scene,output)
            np.savez_compressed(output/'native-geometry.npz',stationary=static,moving=moving,contact=contact,
                gripper=grip,configuration=config,gripper_planes=planes,gripper_plane_offsets=offsets,
                gripper_bounds=bounds)
            write_json(output/'native-scene.json',self.record)

    @staticmethod
    def q(q):
        q=np.asarray(q,float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Seven finite joints required')
        # Native configurations are padded to SIMD width.
        return _f(np.r_[q,0.])

    def validate(self,q):
        statistics=np.zeros(13,np.float64)
        result=self.lib.mcvamp_validate(self.handle,_pointer(self.q(q)),statistics.ctypes.data_as(DOUBLE))
        joints=np.asarray(q,np.float32)
        violated=np.where((joints<self.joint_bounds[:,0])|(joints>self.joint_bounds[:,1]))[0]
        return {**self._collision_report(result,self.statistics(statistics)),
            'joint_bound_violations':[{'joint':int(j+1),'value_rad':float(joints[j]),
                'allowed_range_rad':self.joint_bounds[j].tolist()} for j in violated]}

    def validate_hand_pose(self,pose_world_xyzw):
        """Configured open hand vs the full observed scene; no IK or arm claim."""
        if self.record['moving_cloud_rotates_with_hand_implied_hinge_progress']:
            raise ValueError('Pre-IK hand filtering requires a fixed current scene')
        native_eef=_f(np.linalg.inv(self.world_robot)@pose_matrix(pose_world_xyzw)@np.linalg.inv(self.native_tcp))
        statistics=np.zeros(13,np.float64)
        result=self.lib.mcvamp_validate_hand_pose(self.handle,_pointer(native_eef),statistics.ctypes.data_as(DOUBLE))
        if result<0:raise ValueError('Invalid hand-pose collision query')
        return {**self._collision_report(result,self.statistics(statistics)),
            'scope':'configured hand versus observed scene before IK','arm_posture_checked':False,
            'pose_world_xyzw':np.asarray(pose_world_xyzw).tolist()}

    def _collision_report(self,result,stats):
        code=int(stats['last_rejected_part'])
        part=(f'robot0_link{code-2000}_collision' if 2001<=code<=2007 else
            next((p['robot_geom_name'] for p in getattr(self,'record',{}).get('gripper_cover',[])
                  if 1000+p['sphere_range'][0]==code),None))
        return {'accepted':bool(result),**stats,
            'blocking_robot_part':part if not result else None,
            'blocking_surface':({0:'observed stationary geometry',1:'moving target outside contact patch',2:'selected contact patch'}
                .get(int(stats['last_rejected_surface'])) if not result else None)}

    @staticmethod
    def statistics(stats):
        return dict(zip(['state_checks','self_rejections','static_rejections','moving_rejections','bounds_rejections',
            'timed_out','last_rejected_part','last_rejected_surface','solve_seconds','resampling_seconds','iterations',
            'sparse_states','dense_states'],stats.tolist()))

    def constraint_errors(self,q):
        """Evaluate physical errors without projection or collision checking."""
        error=np.zeros(6,np.float32);metric=ctypes.c_double()
        code=self.lib.mcvamp_constraint_error(self.handle,_pointer(self.q(q)),_pointer(error),ctypes.byref(metric))
        return {'accepted':bool(code),'residual':metric.value,
            'residual_units':'scaled_dimensionless_acceptance_ratio',
            'constraint_acceptance_ratio':metric.value/1e-6,
            'position_error_m':float(np.linalg.norm(error[:2])),
            'radial_error_m':float(error[0]),'height_error_m':float(error[1]),
            'angular_weighted_error':float(np.linalg.norm(error[2:])),
            'position_tolerance_m':self.position_tolerance_m,
            'orientation_tolerance_rad':self.orientation_tolerance_rad,
            'angular_weighted_tolerance':self.angular_weighted_tolerance,
            'progress_bound_weighted_tolerance':1e-6}

    def project(self,q):
        result=np.zeros(8,np.float32);residual=ctypes.c_double()
        code=self.lib.mcvamp_project(self.handle,_pointer(_f(np.zeros(26))),_pointer(self.q(q)),
            _pointer(result),ctypes.byref(residual))
        return result[:7].astype(float),{**self.constraint_errors(result[:7]),
            'accepted':bool(code),'residual':residual.value}

    def plan(self,start,goal,*,constrained=True,seconds=10.,iterations=3000):
        start,goal=self.q(start),self.q(goal)
        before=time.perf_counter()
        endpoints={'start':self.validate(start[:7]),'goal':self.validate(goal[:7])}
        if not all(e['accepted'] for e in endpoints.values()):
            return np.empty((0,7)),{'exact_solution':False,'status':'invalid_endpoint','endpoints':endpoints,
                'total_seconds':time.perf_counter()-before}
        if constrained:
            residuals={}
            for name,q in [('start',start),('goal',goal)]:
                projected,info=self.project(q[:7]);residuals[name]=info
                if not info['accepted'] or np.max(np.abs(projected-q[:7]))>.01:
                    return np.empty((0,7)),{'exact_solution':False,'status':'endpoint_constraint_mismatch',
                        'endpoint_projection':residuals,'total_seconds':time.perf_counter()-before}
        # The child uses only already-built CPU geometry and the native library;
        # it never touches simulator clients, GPU models or their reader threads.
        # Files avoid pipe backpressure for dense paths and native debug output.
        with tempfile.TemporaryDirectory(prefix='mcvamp-plan-') as temporary:
            context=multiprocessing.get_context('fork')
            worker=context.Process(target=self._solve_native,
                args=(start,goal,constrained,seconds,iterations,temporary))
            worker_started=time.time();worker.start();worker.join(seconds+20.)
            if worker.is_alive():
                worker.kill();worker.join()
                return np.empty((0,7)),{'exact_solution':False,'status':'native_worker_timeout',
                    'total_seconds':time.perf_counter()-before}
            result_file=Path(temporary)/'result.json'
            if worker.exitcode or not result_file.exists():
                return np.empty((0,7)),{'exact_solution':False,'status':'native_worker_failed',
                    'worker_exit_code':worker.exitcode,'total_seconds':time.perf_counter()-before,
                    'owned_crash_dump_cleanup':cleanup_worker_dumps(worker.pid,worker_started)}
            result=json.loads(result_file.read_text());n=result['count'];stats=np.asarray(result['statistics'])
            output=np.load(Path(temporary)/'path.npy')
        statistics=self.statistics(stats)
        result={'exact_solution':n>1,'status':native_plan_status(n,statistics),
            'planner':'MC-VAMP CRRTC' if constrained else 'VAMP RRTC','endpoints':endpoints,
            'total_seconds':time.perf_counter()-before,**statistics,
            'path_validation':'upstream planner checks only',
            'additional_dense_validation':False,'post_plan_projection':False,
            'dynamic_domain_pruning':bool(constrained),
            'planner_motion_resolution':32,'continuous_collision_certificate':False,
            'maximum_solve_seconds':seconds,'maximum_planning_seconds':seconds,
            'deadline_scope':'upstream planner search','maximum_iterations':iterations}
        path=output[:max(0,n)].astype(float)
        # Native float32 bounds can round a few ulps outside the exact driver
        # bounds. Normalize the returned references to the same real limits.
        if len(path):path=np.clip(path,self.chain.limits[:,0],self.chain.limits[:,1])
        return path,result

    def _solve_native(self,start,goal,constrained,seconds,iterations,temporary):
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        with open(Path(temporary)/'native.log','wb') as stream:
            os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
            output=np.empty((20000,7),np.float32);stats=np.zeros(13,np.float64)
            n=self.lib.mcvamp_plan(self.handle,_pointer(start),_pointer(goal),_pointer(_f(np.zeros(26))),
                int(constrained),iterations,seconds,_pointer(output),len(output),stats.ctypes.data_as(DOUBLE))
            np.save(Path(temporary)/'path.npy',output[:max(0,n)])
            (Path(temporary)/'result.json').write_text(json.dumps({'count':n,'statistics':stats.tolist()}))

    def progress(self,q):
        matrix=pose_matrix(self.chain.forward(q))
        point=matrix[:3,3]+matrix[:3,:3]@self.pad_local
        first=self.anchor-self.origin;now=point-self.origin
        first-=self.axis*(first@self.axis);now-=self.axis*(now@self.axis)
        return float(np.arctan2(self.axis@np.cross(first,now),first@now))

    def close(self):
        release=getattr(self,'_release',None)
        if release is not None:release()
        self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

