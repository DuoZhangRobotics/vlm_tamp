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
_TREE_CACHE={}


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
    for library,handle,_ in _TREE_CACHE.values():library.mcvamp_destroy(handle)
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


def gripper_convex(metadata,world_robot,parts,point_radius):
    """Known robot convex geometry in native EEF coordinates, without scene IDs."""
    import vamp
    eef=world_robot@np.asarray(vamp.panda.eefk(metadata['joint_positions']),float)[0]
    inverse=np.linalg.inv(eef)
    volumes={v['robot_geom_name']:v for v in metadata['robot_collision_volumes_world']}
    planes=[];offsets=[0];bounds=[]
    for part in parts:
        volume=volumes[part['robot_geom_name']]
        world=np.asarray(volume['planes_world'],float)
        local=np.column_stack([world[:,:3]@eef[:3,:3],world[:,3]+world[:,:3]@eef[:3,3]])
        local/=np.linalg.norm(local[:,:3],axis=1)[:,None]
        vertices=transform_points(volume['vertices_world'],inverse)
        center=vertices.mean(0)
        expanded=local.copy();expanded[:,3]-=point_radius
        padded_vertices=HalfspaceIntersection(expanded,center).intersections
        radius=float(np.max(np.linalg.norm(padded_vertices-center,axis=1))+.00001)
        planes.extend(local);offsets.append(len(planes))
        bounds.append([*center,radius,float(part['selected_handle_contact_allowed']),*part['sphere_range']])
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
                 allow_contact=True,point_radius=.0025,gripper_padding=0.,output=None,grasp_constraint='point'):
        if grasp_constraint not in {'point','rigid','jaw_axis'}:
            raise ValueError('Unknown grasp constraint')
        self.grasp_constraint=grasp_constraint
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
        self.lib.mcvamp_set_gripper_convex.argtypes=[ctypes.c_void_p,FLOAT,INTEGER,FLOAT,ctypes.c_int,ctypes.c_float]
        self.lib.mcvamp_set_arm_convex.argtypes=[ctypes.c_void_p,FLOAT,INTEGER,FLOAT,ctypes.c_int,FLOAT,FLOAT,FLOAT]
        self.lib.mcvamp_set_existing_palm_tolerance.argtypes=[ctypes.c_void_p,ctypes.c_float]
        self.lib.mcvamp_validate.argtypes=[ctypes.c_void_p,FLOAT,DOUBLE]
        self.lib.mcvamp_project.argtypes=[ctypes.c_void_p,FLOAT,FLOAT,FLOAT,DOUBLE]
        self.lib.mcvamp_plan.argtypes=[ctypes.c_void_p,FLOAT,FLOAT,FLOAT,ctypes.c_int,ctypes.c_int,ctypes.c_double,FLOAT,ctypes.c_int,DOUBLE]
        self.world_robot,self.native_tcp,frames=calibrated_frames(record)
        self.chain=RobotChain(record)
        self.scene=scene
        self.reference_q=np.asarray(reference_q,float)
        self.reference_tcp=pose_matrix(self.chain.forward(reference_q))
        robot_world=np.linalg.inv(self.world_robot)
        inverse_tcp=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
        pads=[transform_points(v['vertices_world'],inverse_tcp).mean(0) for v in metadata['robot_collision_volumes_world']
            if v['robot_geom_name'].endswith('_pad_collision')]
        if len(pads)!=2:raise ValueError('Exactly two configured finger pads are required')
        self.pad_local=np.mean(pads,axis=0)
        self.anchor=self.reference_tcp[:3,3]+self.reference_tcp[:3,:3]@self.pad_local
        self.origin=np.asarray(scene.origin,float);self.axis=np.asarray(scene.axis,float)
        gripper,parts=gripper_cover(metadata,self.world_robot)
        distance=cKDTree(scene.contact_reference).query(scene.moving_reference)[0]
        low,high=np.quantile(scene.contact_reference[:,2],[.01,.99])
        panel_distance=np.abs((scene.moving_reference-scene.panel_origin)@scene.panel_normal)
        # Preserve the baseline's local finger/bar and finger/mount permissions.
        # This is not deletion of these points: arm and palm still query them.
        selected=(distance<.028)|((distance<.065)&(panel_distance<scene.panel_band+.006)
            &(scene.moving_reference[:,2]>=low-.015)&(scene.moving_reference[:,2]<=high+.015))
        static=_f(transform_points(voxel_points(scene.static_points,.0075),robot_world))
        moving=_f(transform_points(voxel_points(scene.moving_reference[~selected],.005),robot_world))
        # The transported handle reference locates the contact region; it is
        # not a fresh depth measurement. Adding it to the obstacle tree lets
        # submillimetre rigid-fit error invent palm penetration even when the
        # current measured moving surface is clear. Query current observed
        # moving endpoints, with the reference used only for association.
        contact=_f(transform_points(voxel_points(scene.moving_reference[selected],.003),robot_world))
        origin=transform_points([self.origin],robot_world)[0]
        axis=robot_world[:3,:3]@self.axis
        anchor=transform_points([self.anchor],robot_world)[0]
        limits=self.chain.limits.copy();limits[:,0]+=.005;limits[:,1]-=.005
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
        key=identity.hexdigest()
        before=time.perf_counter()
        reused=key in _TREE_CACHE
        if not reused:
            base=self.lib.mcvamp_create(_pointer(static),len(static),_pointer(moving),len(moving),
                _pointer(contact),len(contact),_pointer(config),_pointer(grip),len(grip),point_radius)
            if not base:raise RuntimeError('Native CAPT construction failed')
            _TREE_CACHE[key]=(self.lib,base,time.perf_counter()-before)
        self.handle=self.lib.mcvamp_clone(_TREE_CACHE[key][1],_pointer(config),_pointer(grip),len(grip))
        if not self.handle:raise RuntimeError('Native CAPT construction failed')
        self.lib.mcvamp_set_grasp_constraint.argtypes=[ctypes.c_void_p,ctypes.c_int]
        if not self.lib.mcvamp_set_grasp_constraint(self.handle,{'point':0,'rigid':1,'jaw_axis':2}[grasp_constraint]):
            self.close();raise RuntimeError('Could not configure grasp constraint')
        planes,offsets,bounds=gripper_convex(metadata,self.world_robot,parts,gripper_padding)
        if not self.lib.mcvamp_set_gripper_convex(self.handle,_pointer(planes),offsets.ctypes.data_as(INTEGER),
                _pointer(bounds),len(bounds),gripper_padding):
            self.close();raise RuntimeError('Calibrated convex gripper installation failed')
        arm_planes,arm_offsets,arm_bounds,axes,anchors,reference=arm_convex(metadata,record,self.world_robot)
        if not self.lib.mcvamp_set_arm_convex(self.handle,_pointer(arm_planes),arm_offsets.ctypes.data_as(INTEGER),_pointer(arm_bounds),len(arm_bounds),_pointer(axes),_pointer(anchors),_pointer(reference)):
            self.close();raise RuntimeError('Configured arm geometry installation failed')
        self.record={'planner':'MC-VAMP CRRTC','upstream_revision':build['upstream_revision'],
            'native_adapter_sha256':build['source_sha256'],'frame_validation':frames,
            'static_points':len(static),'moving_noncontact_points':len(moving),'selected_contact_points':len(contact),
            'point_radius_m':point_radius,'capt_build_seconds':0. if reused else _TREE_CACHE[key][2],
            'context_setup_seconds':time.perf_counter()-before,
            'shared_cloud_index_build_seconds':_TREE_CACHE[key][2],
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
            'native_simd_projection_fully_preserved':False,'aligned_jaws':bool(aligned)}
        self.record.update(grasp_constraint=grasp_constraint,
            shared_tool_stack='observed-hinged-contact-v1',
            grasp_orientation_reference='current measured TCP transported with the estimated articulation',
            independent_wrist_excursion_limit_used=None,
            wrist_policy_source='Resolved by the planning request, separately from the grasp constraint')
        update=getattr(scene,'pose_update',None)
        if update is not None:
            self.record.update(contact_geometry_source='observed endpoints transported to the current query frame',
                moving_geometry_pose_source=update['source'],
                moving_geometry_reference_control_step=update['reference_control_step'],
                fresh_depth_endpoints_after_transport=False)
        # This never clears the world or exempts the palm wholesale. Only an
        # existing submillimetre palm/contact overlap may be carried into a
        # contact-rich replan. Fresh separated starts retain the strict checker.
        initial=self.validate(reference_q)
        self.record['existing_palm_contact']={'enabled':False,'maximum_allowed_depth_m':.0005}
        if not initial['accepted'] and initial['last_rejected_part']==1000 and initial['last_rejected_surface']==2:
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
        return {'accepted':bool(result),**self.statistics(statistics)}

    @staticmethod
    def statistics(stats):
        return dict(zip(['state_checks','self_rejections','static_rejections','moving_rejections','bounds_rejections',
            'timed_out','last_rejected_part','last_rejected_surface','solve_seconds','resampling_seconds','iterations',
            'sparse_states','dense_states'],stats.tolist()))

    def project(self,q):
        result=np.zeros(8,np.float32);residual=ctypes.c_double()
        code=self.lib.mcvamp_project(self.handle,_pointer(_f(np.zeros(26))),_pointer(self.q(q)),
            _pointer(result),ctypes.byref(residual))
        return result[:7].astype(float),{'accepted':bool(code),'residual':residual.value}

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
        names={0:'no_path',-2:'dense_path_capacity',-3:'dense_projection_failed',-4:'dense_scene_collision',
            -5:'dense_goal_collision',-9:'native_exception'}
        result={'exact_solution':n>1,'status':'solved' if n>1 else names.get(n,str(n)),
            'planner':'MC-VAMP CRRTC' if constrained else 'VAMP RRTC','endpoints':endpoints,
            'total_seconds':time.perf_counter()-before,**self.statistics(stats),
            'sampled_motion_validation_step_rad':.006,'continuous_collision_certificate':False,
            'maximum_solve_seconds':seconds,'maximum_iterations':iterations}
        return output[:max(0,n)].astype(float),result

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
        if getattr(self,'handle',None):self.lib.mcvamp_destroy(self.handle);self.handle=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

