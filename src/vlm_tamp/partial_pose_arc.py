"""Continuous seven-joint IK with a handle-point target and free wrist rotation.

No end-effector quaternion menu is used. MC-VAMP performs scene checking and
constrained path connections. Optional continuation waypoints are assembled
into a complete path before any execution.
"""
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .pose_constraints import skew
from .rekep_libero.geometry import pose_matrix
from .rekep_libero.robot_chain_v4 import fixed_orientation_residual


class PartialPoseArc:
    def __init__(self,chain,origin,axis,reference_q,pad_local,*,jaw_tilt=np.pi/3,margin=0.,
                 grasp_constraint='point',position_tolerance_m=.003,orientation_tolerance_rad=0.):
        self.chain=chain;self.origin=np.asarray(origin,float)
        self.axis=np.asarray(axis,float);self.axis/=np.linalg.norm(self.axis)
        self.pad=np.asarray(pad_local,float)
        self.reference=np.asarray(reference_q,float)
        tcp=pose_matrix(chain.forward(self.reference))
        self.reference_rotation=tcp[:3,:3].copy()
        self.anchor=tcp[:3,3]+tcp[:3,:3]@self.pad
        # Project within the actual robot bounds, without another joint inset.
        self.bounds=np.asarray(chain.limits,float)+[margin,-margin]
        self.jaw_limit=float(np.sin(jaw_tilt))
        if grasp_constraint not in {'point','rigid','jaw_axis'}:
            raise ValueError('Unknown grasp constraint')
        self.grasp_constraint=grasp_constraint
        self.position_tolerance_m=float(position_tolerance_m)
        self.orientation_tolerance_rad=float(orientation_tolerance_rad)
        if not np.isfinite(self.orientation_tolerance_rad) or not 0<=self.orientation_tolerance_rad<=np.pi:
            raise ValueError('Orientation tolerance must be finite and in [0, pi]')
        self.angular_weighted_tolerance=(.15*(self.orientation_tolerance_rad if grasp_constraint=='rigid'
            else 2*np.sin(self.orientation_tolerance_rad/2))
            if grasp_constraint!='point' and self.orientation_tolerance_rad else 1e-6)
        if not np.isfinite(self.position_tolerance_m) or self.position_tolerance_m<=0:
            raise ValueError('Position tolerance must be finite and positive')

    def point(self,angle):
        return self.origin+Rotation.from_rotvec(angle*self.axis).apply(self.anchor-self.origin)

    def residual(self,q,angle,*,jacobian=False):
        pose,J=self.chain.forward(q,jacobian=True)
        tcp=pose_matrix(pose);R=tcp[:3,:3]
        point=tcp[:3,3]+R@self.pad
        point_J=J[:3]-R@skew(self.pad)@J[3:]
        expected=Rotation.from_rotvec(angle*self.axis).as_matrix()@self.reference_rotation
        if self.grasp_constraint!='point':
            if self.grasp_constraint=='rigid':
                orientation,orientation_J=fixed_orientation_residual(expected,R,J[3:])
            else:
                # Preserve the line joining the two contacts, leaving rotation
                # about that line free. This is weaker than a rigid attachment.
                orientation=R[:,0]-expected[:,0]
                orientation_J=-R@skew([1.,0.,0.])@J[3:]
            residual=np.r_[point-self.point(angle),.15*orientation]
            return (residual,np.vstack([point_J,.15*orientation_J])) if jacobian else residual
        value=float(self.axis@R[:,0])
        excess=max(0.,abs(value)-self.jaw_limit)
        residual=np.r_[point-self.point(angle),.15*excess]
        if not jacobian:return residual
        jaw_J=-self.axis@R@skew([1.,0.,0.])@J[3:]
        return residual,np.vstack([point_J,.15*np.sign(value)*jaw_J if excess>0 else np.zeros(7)])

    def project(self,seed,angle,*,evaluations=80,avoidance=None):
        low,high=self.bounds.T;seed=np.clip(np.asarray(seed,float),low+1e-8,high-1e-8)
        # A tiny joint regularizer selects a nearby member of the partial-pose
        # solution set without prescribing any wrist quaternion.
        def fun(q):
            residual=20*self.residual(q,angle)
            if avoidance is not None:residual=np.r_[residual,avoidance.evaluate(q,angle)[0]]
            if self.grasp_constraint=='point' and avoidance is not None and abs(angle)<np.radians(15):
                rotation=pose_matrix(self.chain.forward(q))[:3,:3]
                expected=Rotation.from_rotvec(angle*self.axis).as_matrix()@self.reference_rotation
                residual=np.r_[residual,.025*(rotation-expected).T.ravel()]
            return np.r_[residual,1e-6*(q-seed)]
        def jac(q):
            derivative=20*self.residual(q,angle,jacobian=True)[1]
            if avoidance is not None:derivative=np.vstack([derivative,avoidance.evaluate(q,angle)[1]])
            if self.grasp_constraint=='point' and avoidance is not None and abs(angle)<np.radians(15):
                pose,J=self.chain.forward(q,jacobian=True);rotation=pose_matrix(pose)[:3,:3]
                derivative=np.vstack([derivative,.025*np.vstack([-rotation@skew(e)@J[3:] for e in np.eye(3)])])
            return np.vstack([derivative,1e-6*np.eye(7)])
        solved=least_squares(fun,seed,jac=jac,bounds=(low,high),max_nfev=evaluations,
            ftol=1e-10,xtol=1e-10,gtol=1e-11)
        candidate=solved.x
        pre_polish_error=float(np.linalg.norm(self.residual(candidate,angle)[:3]))
        if avoidance is not None:
            candidate,polish=self.project(candidate,angle,evaluations=50)
        residual=self.residual(candidate,angle)
        orientation_norm=float(np.linalg.norm(residual[3:]))/.15
        orientation_error=(orientation_norm if self.grasp_constraint=='rigid' else
            2*np.arcsin(min(1.,orientation_norm/2))) if self.grasp_constraint!='point' else None
        return candidate,{'accepted':bool(np.linalg.norm(residual[:3])<=self.position_tolerance_m and np.linalg.norm(residual[3:])<=self.angular_weighted_tolerance),
            'position_error_m':float(np.linalg.norm(residual[:3])),
            'position_tolerance_m':self.position_tolerance_m,
            'orientation_tolerance_rad':self.orientation_tolerance_rad,
            'orientation_error_deg':float(np.degrees(orientation_error)) if orientation_error is not None else None,
            'orientation_tolerance_applied':self.grasp_constraint!='point',
            'angular_weighted_tolerance':self.angular_weighted_tolerance,
            'jaw_inequality_error':float(np.linalg.norm(residual[3:])), 'evaluations':solved.nfev,
            'grasp_constraint':self.grasp_constraint,
            'wrist_orientation_prescribed':self.grasp_constraint=='rigid','palm_clearance_objective_used':avoidance is not None,
            'pre_equality_polish_position_error_m':pre_polish_error}

    def nullspace(self,q,angle):
        J=self.residual(q,angle,jacobian=True)[1]
        return np.eye(7)-np.linalg.pinv(J,rcond=1e-6)@J


class PalmClearanceObjective:
    """A smooth local optimization aid; native MC-VAMP remains admission authority."""
    def __init__(self,context,*,margin=.002,samples=48,arm=True):
        self.context=context;self.margin=margin;self.samples=samples;self.arm=arm
        self.cached_q=None;self.cached_angle=None;self.cached_result=None
        if context.scene.robot is None:raise ValueError('Configured robot volumes required for palm objective')
        volumes=context.scene.robot.forward(context.reference_q)
        palm=next(v for v in volumes if v['robot_geom_name'].endswith('hand_collision'))
        tcp=pose_matrix(context.chain.forward(context.reference_q));world=np.asarray(palm['planes_world'])
        self.planes=np.c_[world[:,:3]@tcp[:3,:3],world[:,3]+world[:,:3]@tcp[:3,3]]
        self.planes/=np.linalg.norm(self.planes[:,:3],axis=1)[:,None]
        self.center=tcp[:3,:3].T@(np.mean(palm['vertices_world'],axis=0)-tcp[:3,3])
        self.clouds=[np.asarray(context.scene.static_points),
                     np.vstack([context.scene.moving_reference,context.scene.contact_reference])]
        self.angle=None;self.trees=None;self.rotated=None

    def evaluate(self,q,angle):
        if self.cached_angle==angle and self.cached_q is not None and np.array_equal(q,self.cached_q):
            return self.cached_result
        if self.angle!=angle:
            context=self.context
            self.rotated=[self.clouds[0],context.origin+Rotation.from_rotvec(angle*context.axis).apply(self.clouds[1]-context.origin)]
            self.trees=[cKDTree(cloud) for cloud in self.rotated];self.angle=angle
        pose,J=self.context.chain.forward(q,jacobian=True);tcp=pose_matrix(pose);R=tcp[:3,:3];t=tcp[:3,3]
        center=R@self.center+t;values=[];derivatives=[]
        for cloud,tree in zip(self.rotated,self.trees):
            if not len(cloud):
                values.extend(np.zeros(self.samples));derivatives.extend(np.zeros((self.samples,7)));continue
            _,indices=tree.query(center,k=min(self.samples,len(cloud)))
            indices=np.atleast_1d(indices);local=(cloud[indices]-t)@R
            signed=local@self.planes[:,:3].T+self.planes[:,3]
            active=np.argmax(signed,axis=1);distance=signed[np.arange(len(local)),active]
            for point,which,d in zip(local,active,distance):
                value=max(0.,self.margin-d);values.append(value)
                normal=self.planes[which,:3]
                derivative=normal@(-R.T@J[:3]+skew(point)@J[3:])
                derivatives.append(-derivative if value>0 else np.zeros(7))
            for _ in range(self.samples-len(local)):values.append(0.);derivatives.append(np.zeros(7))
        if self.arm:
            import vamp
            def spheres(joints):
                native=vamp.panda.fk(np.asarray(joints).tolist())[1:37]
                return np.asarray([[s.x,s.y,s.z,s.r] for s in native])
            native=spheres(q);world=self.context.world_robot
            centers=native[:,:3]@world[:3,:3].T+world[:3,3]
            center_J=np.empty((len(native),3,7))
            for j in range(7):
                step=np.zeros(7);step[j]=.001
                derivative=(spheres(q+step)[:,:3]-spheres(q-step)[:,:3])/.002
                center_J[:,:,j]=derivative@world[:3,:3].T
            for cloud,tree in zip(self.rotated,self.trees):
                if not len(cloud):
                    values.extend(np.zeros(len(native)));derivatives.extend(np.zeros((len(native),7)));continue
                distance,indices=tree.query(centers)
                unit=(centers-cloud[indices])/np.maximum(distance[:,None],1e-9)
                clearance=distance-native[:,3]-.0025
                residual=np.maximum(0.,self.margin-clearance)
                derivative=-np.einsum('ni,nij->nj',unit,center_J)
                derivative[residual<=0]=0.
                values.extend(residual);derivatives.extend(derivative)
        result=np.asarray(values),np.asarray(derivatives)
        self.cached_q=np.asarray(q).copy();self.cached_angle=angle;self.cached_result=result
        return result


def short_wrist_limit(context,goal):
    """Optional caller constraint, independent of horizon or hinge confidence.

    SO(3) rotation distance cannot exceed pi. Returning pi removes only this
    heuristic veto; joint limits, contact constraints and path checks still apply.
    """
    options=getattr(context,'motion_options',context.scene.config)
    limit=options.get('max_wrist_reorientation_deg')
    if limit is None:return np.pi
    if not np.isfinite(limit) or not 0<float(limit)<=180:
        raise ValueError('max_wrist_reorientation_deg must be null or in (0, 180]')
    return float(np.radians(limit))


def wrist_excursion(context,q):
    rotation=pose_matrix(context.chain.forward(q))[:3,:3]
    return float(Rotation.from_matrix(context.reference_tcp[:3,:3].T@rotation).magnitude())


def path_wrist_admitted(context,path,goal):
    limit=short_wrist_limit(context,goal)
    measured=max((wrist_excursion(context,q) for q in path),default=0.)
    return measured<=limit+1e-6,{'maximum_wrist_excursion_deg':float(np.degrees(measured)),
        'planning_wrist_limit_deg':float(np.degrees(limit)),'execution_veto':False}


def partial_goals(context,goal,*,proposals=160,maximum=8,seed=0,deadline=None):
    model=PartialPoseArc(context.chain,context.origin,context.axis,context.reference_q,context.pad_local,
                        grasp_constraint=getattr(context,'grasp_constraint','point'),
                        position_tolerance_m=context.scene.config.get('arc_position_tolerance_m',.003),
                        orientation_tolerance_rad=context.scene.config.get('arc_orientation_tolerance_rad',0.))
    rng=np.random.default_rng(seed);start=context.reference_q
    avoidance=PalmClearanceObjective(context)
    seeds=[start]
    # Cover the joint solution set rather than samples of a fixed wrist pose.
    seeds.extend(start+rng.normal(0,.65,7) for _ in range(proposals//3))
    seeds.extend(rng.uniform(model.bounds[:,0],model.bounds[:,1]) for _ in range(proposals-len(seeds)))
    valid=[];records=[]
    for index,initial in enumerate(seeds):
        if deadline is not None and time.perf_counter()>=deadline:break
        q,projection=model.project(initial,goal,avoidance=avoidance)
        row={'seed':index,'projection':projection}
        if projection['accepted']:
            q,native_projection=context.project(q)
            check=context.validate(q) if native_projection['accepted'] else None
            okay=bool(native_projection['accepted'] and check['accepted'] and abs(context.progress(q)-goal)<.002)
            excursion=wrist_excursion(context,q)
            if excursion>short_wrist_limit(context,goal):okay=False
            row.update(native_projection=native_projection,scene=check,joints=q.tolist(),accepted=okay)
            row['wrist_excursion_deg']=float(np.degrees(excursion))
            if okay and not any(np.max(np.abs(q-other))<.12 for other in valid):valid.append(q)
        else:row['accepted']=False
        records.append(row)
        if len(valid)>=maximum:break
    valid.sort(key=lambda q:np.linalg.norm(q-start))
    return model,valid,records


def full_arc_plan(context,goal,*,seconds=40.,proposals=160,beam_width=5,seed=0,goal_proposer=None):
    """Try global C-RRTC, then free-wrist arc waypoints with C-RRTC connectors."""
    if goal_proposer is not None and getattr(goal_proposer,'solver_mode',None)=='local_continuation':
        started=time.perf_counter()
        try:
            path,result=goal_proposer.plan(goal,seconds=seconds)
            remaining=seconds-(time.perf_counter()-started)
            if (result['status'] in {'continuation_no_admissible_goal','continuation_connection_failed'}
                    and getattr(goal_proposer,'global_fallback_enabled',False) and remaining>=3.):
                return goal_proposer.global_fallback(result,deadline=started+seconds,started=started)
            return path,result
        finally:
            goal_proposer.close()
    before=time.perf_counter();start=context.reference_q
    wrist_limit=short_wrist_limit(context,goal)
    if hasattr(context,'record'):
        context.record.update(independent_wrist_excursion_limit_used=bool(wrist_limit<np.pi-1e-8),
                              planning_wrist_limit_deg=float(np.degrees(wrist_limit)))
    initial=context.validate(start)
    report={'start':initial,'goal_angle_rad':goal,'wrist_orientation_menu_used':False,
        'end_effector_quaternion_prescribed':getattr(context,'grasp_constraint','point')=='rigid',
        'grasp_constraint':getattr(context,'grasp_constraint','point'),'complete_arc_required':True,
        'execution_during_search':False,'global_queries':[],'continuation_layers':[]}
    report['goal_backend']='curobo_v2' if goal_proposer is not None else 'scipy_partial_pose'
    if not initial['accepted']:return np.empty((0,7)),{**report,'status':'invalid_start'}
    if goal_proposer is None:
        model,goals,goal_records=partial_goals(context,goal,proposals=proposals,seed=seed,
                                            deadline=before+.65*seconds)
    else:
        model,goals,goal_records=goal_proposer.generate(goal,deadline=before+.65*seconds)
    report['goal_generation_seconds']=time.perf_counter()-before
    report.update(goal_proposals=goal_records,admissible_goals=len(goals))
    for number,target in enumerate(goals if goal_proposer is not None else goals[:3]):
        remaining=seconds-(time.perf_counter()-before)
        if remaining<=0:break
        path,result=context.plan(start,target,seconds=min(5.,remaining),iterations=5000)
        if result['exact_solution']:
            admitted,wrist=path_wrist_admitted(context,path,goal);result.update(wrist)
            if not admitted:result.update(exact_solution=False,status='planning_wrist_continuity_rejection')
        report['global_queries'].append(result)
        if result['exact_solution']:
            return path,{**report,'status':'solved','planner':'MC-VAMP C-RRTC with partial-pose joint goals',
                'terminal_progress_rad':context.progress(path[-1]),'total_seconds':time.perf_counter()-before}
    # This is a search over the entire arc before motion, not execution-time
    # segmentation/replanning. Every accepted edge is an upstream C-RRTC path.
    rng=np.random.default_rng(seed)
    avoidance=PalmClearanceObjective(context) if goal_proposer is None else None
    beam=[{'q':start,'path':start[None].copy(),'cost':0.}]
    count=max(1,int(np.ceil(goal/np.deg2rad(3.))))
    for angle in np.linspace(0,goal,count+1)[1:]:
        candidates=[];rejected={};attempted=0
        for parent in beam:
            N=model.nullspace(parent['q'],float(angle))
            seeds=[parent['q']]
            seeds.extend(parent['q']+N@rng.normal(0,.18,7) for _ in range(3))
            for target in goals[:2]:
                step=N@(target-parent['q']);scale=min(1.,.25/max(np.max(np.abs(step)),1e-9))
                seeds.append(parent['q']+scale*step)
            for initial_q in seeds:
                if time.perf_counter()-before>seconds:
                    return np.empty((0,7)),{**report,'status':'search_budget_exhausted','last_progress_rad':float(angle),
                        'total_seconds':time.perf_counter()-before}
                if goal_proposer is None:
                    q,projection=model.project(initial_q,float(angle),avoidance=avoidance)
                else:
                    q,projection=goal_proposer.project(initial_q,float(angle))
                attempted+=1
                if not projection['accepted']:continue
                q,native_projection=context.project(q)
                if not native_projection['accepted'] or abs(context.progress(q)-angle)>.002:continue
                if wrist_excursion(context,q)>short_wrist_limit(context,goal):continue
                check=context.validate(q)
                if not check['accepted']:
                    tag=f"{int(check['last_rejected_part'])}/{int(check['last_rejected_surface'])}"
                    rejected[tag]=rejected.get(tag,0)+1;continue
                if np.max(np.abs(q-parent['q']))>.6:continue
                if any(np.max(np.abs(q-other['q']))<.05 for other in candidates):continue
                edge,connection=context.plan(parent['q'],q,seconds=1.,iterations=1500)
                if not connection['exact_solution']:continue
                admitted,_=path_wrist_admitted(context,edge,goal)
                if not admitted:continue
                cost=parent['cost']+float(np.sum(np.linalg.norm(np.diff(edge,axis=0),axis=1)))
                candidates.append({'q':q,'path':np.vstack([parent['path'],edge[1:]]),'cost':cost})
        report['continuation_layers'].append({'angle_rad':float(angle),'attempted':attempted,
            'connected_candidates':len(candidates),'rejected_parts':rejected})
        if not candidates:
            return np.empty((0,7)),{**report,'status':'continuation_blocked','last_progress_rad':float(angle),
                'total_seconds':time.perf_counter()-before}
        candidates.sort(key=lambda row:row['cost']);beam=candidates[:beam_width]
    best=beam[0]
    return best['path'],{**report,'status':'solved','planner':'partial-pose full-arc search with MC-VAMP C-RRTC connectors',
        'terminal_progress_rad':context.progress(best['q']),'total_seconds':time.perf_counter()-before,
        'waypoints_are_planning_internal_only':True}
