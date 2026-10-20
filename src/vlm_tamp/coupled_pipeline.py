"""Full-motion grasp admission and planning helpers, using sensor geometry only."""

from collections import Counter
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .handle_contact import frontal_poses,upright_plane
from .coupled_manifold import CoupledHinge
from .coupled_local_projection import project_nearby
from .coupled_scene import Scene
from .rekep_libero.geometry import pose_matrix,matrix_pose,transform_points
from .rekep_libero.physical_tasks_v4 import candidate_variants
from .rekep_libero.scene_state_v7 import observed_points


ROOT=Path(__file__).resolve().parents[2]


def validated_grasp_reference(pad_world, observed_axis_anchor_world, maximum_distance=.008):
    """Keep the measured grasp offset after validating selected-bar proximity.

    The observed upright bar defines the independent horizontal axis location.
    Its point at pad height validates capture without translating a retained
    grasp onto an ideal cylinder centre. The returned reference is transported
    with the hinge. Finite bar height and continued retention remain checks
    performed by the caller using observed handle geometry and RGB-D motion.
    """
    pad=np.asarray(pad_world,float)
    observed=np.asarray(observed_axis_anchor_world,float)
    if (pad.shape!=(3,) or observed.shape!=(3,)
            or not np.isfinite(pad).all() or not np.isfinite(observed).all()):
        raise ValueError('Finite three-dimensional pad and observed bar points are required')
    maximum_distance=float(maximum_distance)
    if not np.isfinite(maximum_distance) or maximum_distance<=0:
        raise ValueError('A positive finite observed-bar proximity limit is required')
    axis_point=np.r_[observed[:2],pad[2]]
    offset=pad-axis_point
    distance=float(np.linalg.norm(offset))
    if distance>maximum_distance:
        raise ValueError('Measured pads are too far from the observed handle axis')
    return pad.copy(),{
        'reference_anchor_world_m':pad.tolist(),
        'observed_axis_point_world_m':axis_point.tolist(),
        'observed_pad_to_handle_distance_m':distance,
        'observed_axis_to_pad_offset_world_m':offset.tolist(),
        'maximum_binding_distance_m':maximum_distance,
        'measured_grasp_offset_preserved':True,
        'binding_semantics':'measured retained grasp point; its offset rotates with the articulated body',
        'finite_bar_height_and_retention_checked_by_caller':True,
        'simulator_scene_geometry_used':False,
    }


def project_ik_candidate(problem,joints,angle):
    """Repair a hypothetical IK proposal locally, preserving its posture."""
    raw=np.r_[np.asarray(joints,float),float(angle)]
    clipped=np.clip(raw,problem.bounds[:,0]+1e-10,problem.bounds[:,1]-1e-10)
    clipped[7]=angle
    result=project_nearby(problem,clipped,fixed_progress=angle)
    result['hypothetical_seed_clipping_rad']=float(np.max(np.abs(clipped[:7]-raw[:7])))
    return result


def closing_normal_residual(normal,terminal,axis):
    """Signed remaining closing rotation from two measured plane normals."""
    vectors=[np.asarray(v,float) for v in (normal,terminal,axis)]
    if any(v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-9 for v in vectors):
        raise ValueError('Finite nonzero three-dimensional plane normals and axis are required')
    n,t,a=[v/np.linalg.norm(v) for v in vectors]
    return float(np.arctan2(a@np.cross(n,t),n@t))


def terminal_angle(cameras,metadata,spec,cloud):
    if spec['motion']=='open':
        return float(np.pi/2),{'source':'configured 90 degree opening goal','simulator_scene_state_used':False}
    points,_=observed_points(cameras,metadata,spacing=.004,robot_depth_tolerance=.0001)
    hinge=np.asarray(spec['joint_origin_world_m']);normal=np.asarray(spec['panel_normal_world'])
    axis=np.asarray(spec['joint_axis_world']);origin=np.asarray(spec['front_plane']['point_world_m'])
    low,high=np.quantile(cloud[:,2],[.05,.95])
    keep=(np.linalg.norm(points[:,:2]-hinge[:2],axis=1)<.46)&(points[:,2]>low)&(points[:,2]<high)
    keep&=np.abs((points-origin)@normal)>.045
    remaining=points[keep];candidates=[]
    for index in range(5):
        if len(remaining)<200:break
        try:face=upright_plane(remaining,np.asarray(cameras[spec['camera']]['camera_to_world'])[:3,3],seed=index)
        except ValueError:break
        for sign in [1.,-1.]:
            direction=sign*face['normal']
            angle=float(np.arctan2(axis@np.cross(normal,direction),normal@direction))
            if np.deg2rad(45)<angle<np.deg2rad(110):
                candidates.append({'angle_rad':angle,'normal_world':direction.tolist(),
                    'support':int(face['inliers'].sum()),'plane_rms_m':face['rms']})
        remaining=remaining[~face['inliers']]
    if not candidates:raise ValueError('No observed stationary cabinet face resolves the closing endpoint')
    chosen=max(candidates,key=lambda row:row['support'])
    return chosen['angle_rad'],{'source':'relative normals of observed moving door and stationary cabinet face',
        'selected':chosen,'candidates':candidates,'simulator_scene_state_used':False}


def supported_handle_anchors(anchors,cloud,metadata,margin=.003):
    """Keep full pad support inside a robust observed vertical-bar interval.

    With horizontal jaw closure, local x contributes no vertical span. The
    remaining pad cross-section radius bounds its span at every allowed roll.
    Observed depth quantiles trim the uncertain rounded bar ends.
    """
    inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
    pads=[transform_points(v['vertices_world'],inverse)
          for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name'].endswith('_pad_collision')]
    if len(pads)!=2:raise ValueError('Two configured finger pads are required')
    midpoint=np.mean([p.mean(0) for p in pads],axis=0)
    half_span=float(np.max(np.linalg.norm(np.vstack(pads)[:,1:]-midpoint[1:],axis=1)))
    low,high=np.quantile(np.asarray(cloud)[:,2],[.05,.95])
    kept=[np.asarray(a).copy() for a in anchors if low+half_span+margin<=a[2]<=high-half_span-margin]
    record={'source':'observed depth quantiles and configured finite pad geometry',
        'observed_support_z_quantiles_m':[float(low),float(high)],'pad_vertical_span_bound_m':half_span,
        'support_margin_m':margin,'offered':len(anchors),'retained':len(kept),
        'simulator_object_geometry_used':False,'assumes_jaw_axis_perpendicular_to_bar':True}
    if not kept:raise ValueError('No observed handle interval supports the complete finger pads')
    return kept,record


def handle_anchor_menu(axis_xy, cloud, metadata):
    """Cover the full measured finite-pad support interval, including its top."""
    cloud=np.asarray(cloud,float)
    centre=np.r_[np.asarray(axis_xy,float),np.median(cloud[:,2])]
    _,record=supported_handle_anchors([centre],cloud,metadata)
    low,high=record['observed_support_z_quantiles_m']
    inset=record['pad_vertical_span_bound_m']+record['support_margin_m']+.001
    if high-low<=2*inset:raise ValueError('Observed bar lacks a buffered finite-pad grasp interval')
    heights=list(np.linspace(low+inset,high-inset,5))
    heights += list(np.quantile(cloud[:,2],[.20,.40,.60,.80,.85,.93]))
    anchors=[]
    for z in sorted(heights,reverse=True):
        if low+inset<=z<=high-inset and not any(abs(z-a[2])<.003 for a in anchors):
            anchors.append(np.r_[axis_xy,z])
    anchors,record=supported_handle_anchors(anchors,cloud,metadata)
    record['menu_source']='full safe observed height interval plus RGB-D quantiles'
    record['additional_endpoint_buffer_m']=.001
    return anchors,record


def grasp_poses(generated,anchors,normal,metadata):
    inverse=np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
    pads=[transform_points(v['vertices_world'],inverse).mean(0)
          for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name'].endswith('_pad_collision')]
    pad_local=np.mean(pads,axis=0)
    # Heights are alternatives, not a commitment before full-path checking.
    anchors=sorted(anchors,key=lambda p:-p[2])
    rows=frontal_poses(normal,anchors,pad_local)
    for index,variant,pose in candidate_variants(generated,metadata['robot_calibration']):
        rotation=pose_matrix(pose)[:3,:3]
        if -rotation[:,2]@normal<.75 or abs(rotation[:,0]@[0.,0.,1.])>.40:continue
        if min(np.linalg.norm(pose[:3]-a) for a in anchors)>.10:continue
        for number,anchor in enumerate(anchors):
            refined=np.asarray(pose).copy();refined[:3]=anchor-rotation@pad_local
            rows.append({'pose':refined,'source':'GraspGen rotation with observed handle translation',
                         'graspgen_index':index,'variant':variant,'anchor_index':number})
    unique=[]
    rows.sort(key=lambda r:r['anchor_index'])
    for row in rows:
        if not any(np.linalg.norm(row['pose'][:3]-v['pose'][:3])<.003 and
            (Rotation.from_quat(row['pose'][3:])*Rotation.from_quat(v['pose'][3:]).inv()).magnitude()<.05 for v in unique):
            unique.append(row)
    groups={number:[row for row in unique if row['anchor_index']==number] for number in range(len(anchors))}
    interleaved=[];used=set()
    # Diversify wrist orientation at the two highest supported heights before
    # exhausting the budget on one orientation across many nearby heights.
    for rank in range(3):
        for number,group in list(groups.items())[:2]:
            if rank<len(group):interleaved.append(group[rank]);used.add((number,rank))
    for rank in range(max(map(len,groups.values()),default=0)):
        for number,group in groups.items():
            if rank<len(group) and (number,rank) not in used:interleaved.append(group[rank])
    return interleaved,pad_local


def goal_pose_menu(problem, angle, reference, pad_local, axis, *, allow_roll=True):
    """Sample the two free rotations without moving the contact anchor."""
    base=problem.pose(angle)
    anchor=transform_points([reference[:3,3]+reference[:3,:3]@pad_local],problem.transform(angle))[0]
    choices=[(yaw,0.) for yaw in [0.,-.35,.35,-.70,.70]]
    if allow_roll:
        choices += [(yaw,roll) for roll in [-np.pi/4,np.pi/4,-np.pi/2,np.pi/2,np.pi]
                    for yaw in [0.,-.70,.70]]
    for yaw,roll in choices:
        target=base.copy()
        target[:3,:3]=(Rotation.from_rotvec(yaw*np.asarray(axis)).as_matrix()@base[:3,:3]
                      @Rotation.from_rotvec([roll,0.,0.]).as_matrix())
        target[:3,3]=anchor-target[:3,:3]@pad_local
        yield target,{'yaw_rad':yaw,'roll_about_jaw_rad':roll}


def partial_goal_candidates(problem, start_joints, angle, self_valid, scene, *, seed=0,
                            proposals=64, maximum=4):
    """Solve the actual partial task constraint, then filter hypothetical goals."""
    rng=np.random.default_rng(seed)
    initial=np.asarray(start_joints,float)
    lower,upper=problem.bounds[:7].T
    seeds=[initial]+[rng.uniform(lower+1e-8,upper-1e-8) for _ in range(proposals-1)]
    states=[];diagnostics=[]
    for number,q in enumerate(seeds):
        result=problem.project(np.r_[q,angle],fixed_progress=angle,max_iterations=150)
        x=result['state'];errors=problem.errors(x)
        accepted=(result['accepted'] and errors['weighted_residual_norm']<=1e-5
                  and problem.satisfied(x))
        robot_ok=bool(self_valid(x[:7])) if accepted else None
        check=scene.check(x[:7],angle) if accepted and robot_ok else None
        okay=bool(accepted and robot_ok and check['accepted'])
        diagnostics.append({'proposal':number,'accepted':okay,'state':x,
            'constraint_errors':errors,'self_valid':robot_ok,'scene_check':check})
        if okay and not any(np.max(np.abs(x[:7]-old[:7]))<.02 for old in states):
            states.append(x)
        if len(states)>=maximum:break
    states.sort(key=lambda x:float(np.linalg.norm(x[:7]-initial)))
    return states,diagnostics


def isolated_plan(problem,start,goal,scene,output,save,seconds=12.,strategy='atlas',seed=1,warm_path=None,
                  warm_maximum_joint_correction_rad=.04):
    if not np.isfinite(warm_maximum_joint_correction_rad) or not 0<warm_maximum_joint_correction_rad<=.2:
        raise ValueError('Warm reference proposal correction must be finite and in (0, .2] radians')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    bundle=output/'trusted-local-input.pkl'
    with bundle.open('wb') as stream:
        pickle.dump({'problem':problem,'start':start,'goal':goal,'scene':scene,'seconds':seconds,'strategy':strategy,'seed':seed,
                     'warm_path':warm_path,'warm_maximum_joint_correction_rad':float(warm_maximum_joint_correction_rad)},stream,
                    protocol=pickle.HIGHEST_PROTOCOL)
    from .coupled_storage import preserve_scene
    preserve_scene(scene, output)
    try:
        with (output/'planner.log').open('w') as stream:
            process=subprocess.run([sys.executable,str(ROOT/'scripts/solve_coupled_path.py'),
                '--input',str(bundle),'--output',str(output/'solution')],cwd=ROOT,stdout=stream,stderr=stream,timeout=seconds+90)
    finally:
        # subprocess.run kills and waits for the worker on timeout/interruption.
        bundle.unlink(missing_ok=True)
    result_path=output/'solution/result.json'
    import json
    result=json.loads(result_path.read_text()) if result_path.exists() else {'exact_solution':False,'worker_exit':process.returncode}
    path=np.load(output/'solution/path.npy') if (output/'solution/path.npy').exists() else np.empty((0,8))
    return path,result


def complete_motion_selection(generated,anchors,metadata,ik,open_scene,closed_scene,spec,goal_angle,
                              output,save,*,contact_model='point_contact',maximum_seconds=240,maximum_poses=36,require_valid_goal=False):
    from .coupled_manifold import CoupledHandlePoint
    rows,pad_local=grasp_poses(generated,anchors,np.asarray(spec['panel_normal_world']),metadata)
    start_time=time.monotonic();rejected=[];attempt=0
    gate_counts=Counter()

    def projection_record(projection):
        return {k:v for k,v in projection.items() if k!='state'}

    def finish_start(record,stage):
        record['terminal_gate']=stage
        gate_counts[stage]+=1

    def save_validation(ordinal,record):
        save(Path(output)/f'start-validation-{ordinal:03d}.json',record)

    initial=np.asarray(metadata['joint_positions'])
    for ordinal,row in enumerate(rows[:maximum_poses]):
        if time.monotonic()-start_time>maximum_seconds:break
        gate_counts['poses_examined']+=1
        validation={'pose_index':ordinal,'candidate':row,'goal_queries':[],
            'goal_fallback':None,'starts':[]}
        starts=ik.solve_many(row['pose'],initial,maximum=12)
        starts.sort(key=lambda s:-s['joint_reserve_rad'])
        validation['grasp_ik_postures_returned']=len(starts)
        gate_counts['grasp_ik_postures_returned']+=len(starts)
        if not starts:
            rejected.append({'pose':ordinal,'reason':'no_grasp_ik'})
            validation['terminal_gate']='no_grasp_ik';gate_counts['no_grasp_ik']+=1
            save_validation(ordinal,validation)
            continue
        reference=pose_matrix(starts[0]['pose_world_xyzw'])
        options=dict(progress_bounds=(-.03,goal_angle+.03),joint_margin=.025,
                     position_tolerance=.001,orientation_tolerance=.01)
        if contact_model=='rigid':
            problem=CoupledHinge(ik.chain,spec['joint_origin_world_m'],spec['joint_axis_world'],reference,**options)
        else:
            problem=CoupledHandlePoint(ik.chain,spec['joint_origin_world_m'],spec['joint_axis_world'],reference,
                                      pad_local=pad_local,orientation_cone=np.pi/2,
                                      jaw_axis_local=[1.,0.,0.] if contact_model=='jaw_contact' else None,
                                      align_axis=contact_model=='handle_point',**options)
        goals=[];goal_diagnostics=[]
        for target,angles in goal_pose_menu(problem,goal_angle,reference,pad_local,
                spec['joint_axis_world'],allow_roll=contact_model in {'point_contact','jaw_contact'}):
            goal_solutions=ik.solve_many(matrix_pose(target),starts[0]['joints'],maximum=12)
            query_record={**angles,'target_pose_world_xyzw':matrix_pose(target),
                'ik_candidates_returned':len(goal_solutions),'candidates':[]}
            validation['goal_queries'].append(query_record)
            gate_counts['goal_ik_queries']+=1
            gate_counts['goal_ik_candidates']+=len(goal_solutions)
            for solution in goal_solutions:
                projection=project_ik_candidate(problem,solution['joints'],goal_angle)
                x=projection['state']
                goal_diagnostics.append({k:v for k,v in projection.items() if k!='state'})
                goal_record={'raw_joints':solution['joints'],'projection':projection_record(projection),
                    'projected_state':x,'manifold_satisfied':None,'self_valid':None,'closed_scene':None}
                query_record['candidates'].append(goal_record)
                gate_counts['goal_projection_calls']+=1
                # Keep the original short-circuit order: diagnostics reuse
                # each result and never trigger an otherwise skipped check.
                if not projection['accepted']:
                    goal_record['terminal_gate']='goal_projection_failed'
                elif not projection['weighted_residual_norm']<=1e-5:
                    goal_record['terminal_gate']='goal_projection_residual'
                else:
                    goal_record['manifold_satisfied']=bool(problem.satisfied(x))
                    if not goal_record['manifold_satisfied']:
                        goal_record['terminal_gate']='goal_manifold_failed'
                    else:
                        goal_record['self_valid']=bool(ik.self_valid(x[:7]))
                        if not goal_record['self_valid']:
                            goal_record['terminal_gate']='goal_self_collision'
                        else:
                            goal_record['closed_scene']=closed_scene.check(x[:7],x[7])
                            goal_record['terminal_gate']=('goal_accepted' if goal_record['closed_scene']['accepted']
                                                          else 'goal_scene_collision')
                gate_counts[goal_record['terminal_gate']]+=1
                if goal_record['terminal_gate']=='goal_accepted':
                    goals.append(x)
            if goals:break
        partial_diagnostics=[]
        if not goals and contact_model!='rigid':
            goals,partial_diagnostics=partial_goal_candidates(problem,starts[0]['joints'],goal_angle,
                ik.self_valid,closed_scene)
            gate_counts['partial_goal_projection_calls']+=len(partial_diagnostics)
            gate_counts['partial_goal_valid_candidates']+=len(goals)
            validation['partial_goal_queries']=partial_diagnostics
        real_goals=bool(goals)
        save(Path(output)/f'goal-proposals-{ordinal:03d}.json',{'real_valid_goals':len(goals),
            'projection_diagnostics':goal_diagnostics,'partial_goal_diagnostics':partial_diagnostics,'goal_states':goals})
        if not goals:
            gate_counts['goal_pose_menu_empty']+=1
            if contact_model=='rigid' or require_valid_goal:
                rejected.append({'pose':ordinal,'reason':'no_collision_free_full_motion_goal_ik'})
                save(Path(output)/'selection-progress.json',{'poses':len(rows),'pose_index':ordinal,
                    'planner_attempts':attempt,'rejection_counts':dict(Counter(r['reason'] for r in rejected))})
                validation['terminal_gate']='no_collision_free_full_motion_goal_ik'
                save_validation(ordinal,validation)
                continue
            # Free axial twist is part of the manifold. A small menu of full
            # pose IK goals must not silently fix that degree of freedom.
            # The milestone planner searches its own final joint posture.
            goals=[np.r_[starts[0]['joints'],goal_angle]]
            validation['goal_fallback']='milestones_searches_free_terminal_joint_posture'
            gate_counts['goal_free_joint_fallback']+=1
        pose_attempts=0
        for rank,start in enumerate(starts[:3]):
            gate_counts['starts_considered']+=1
            start_record={'rank':rank,'raw_joints':start['joints'],
                'raw_joint_reserve_rad':start.get('joint_reserve_rad'),
                'projection':None,'projected_state':None,'self_valid':None,'closed_scene':None,
                'approach':None,'open_scene_collision_stop':None,'planner_results':[]}
            validation['starts'].append(start_record)
            projection=project_ik_candidate(problem,start['joints'],0.)
            xstart=projection['state']
            start_record['projection']=projection_record(projection)
            start_record['projected_state']=xstart
            start_record['maximum_projection_joint_displacement_rad']=float(np.max(np.abs(
                np.asarray(start['joints'])-xstart[:7])))
            start={**start,'joints':xstart[:7].tolist(),'pose_world_xyzw':ik.chain.forward(xstart[:7]).tolist()}
            if not projection['accepted']:
                finish_start(start_record,'start_projection_failed');continue
            if projection['weighted_residual_norm']>1e-5:
                finish_start(start_record,'start_projection_residual');continue
            start_record['self_valid']=bool(ik.self_valid(start['joints']))
            if not start_record['self_valid']:
                finish_start(start_record,'start_self_collision');continue
            start_record['closed_scene']=closed_scene.check(start['joints'],0.)
            if not start_record['closed_scene']['accepted']:
                finish_start(start_record,'start_scene_collision');continue
            gate_counts['start_accepted']+=1
            approach=ik.approach(initial,start)
            start_record['approach']=approach
            if not approach['accepted']:
                gate_counts['approach_failure_'+approach.get('reason','unspecified')]+=1
                finish_start(start_record,'approach_failed');continue
            valid=True
            for segment,(qa,qb) in enumerate(zip(approach['joint_postures'][:-1],approach['joint_postures'][1:])):
                count=max(1,int(np.ceil(np.max(np.abs(np.asarray(qb)-qa))/.03)))
                for w in np.linspace(0,1,count+1):
                    q=np.asarray(qa)+w*(np.asarray(qb)-qa)
                    contact=np.linalg.norm(ik.chain.forward(q)[:3]-start['pose_world_xyzw'][:3])<.02
                    checked=open_scene.check(q,0.,contact=contact)
                    if not checked['accepted']:
                        start_record['open_scene_collision_stop']={'segment':segment,
                            'fraction':float(w),'joints':q,'contact_enabled':bool(contact),'result':checked}
                        valid=False;break
                if not valid:break
            if not valid:
                rejected.append({'pose':ordinal,'reason':'approach_scene_collision'})
                finish_start(start_record,'approach_scene_collision');continue
            gate_counts['approach_accepted']+=1
            goals.sort(key=lambda x:np.linalg.norm(x[:7]-xstart[:7]))
            for xgoal in goals[:1]:
                folder=Path(output)/f'planner-attempt-{attempt:03d}';attempt+=1
                strategy='atlas' if real_goals else 'milestones'
                path,result=isolated_plan(problem,xstart,xgoal,closed_scene,folder,save,
                    seconds=60. if strategy=='milestones' else 20.,strategy=strategy)
                gate_counts['planner_calls']+=1
                start_record['planner_results'].append({'strategy':strategy,'folder':str(folder),'result':result})
                pose_attempts+=1
                if not result.get('exact_solution') and pose_attempts==1 and strategy=='atlas':
                    path,result=isolated_plan(problem,xstart,xgoal,closed_scene,
                        folder.parent/(folder.name+'-milestones'),save,seconds=60.,strategy='milestones')
                    gate_counts['planner_calls']+=1
                    start_record['planner_results'].append({'strategy':'milestones',
                        'folder':str(folder.parent/(folder.name+'-milestones')),'result':result})
                save(Path(output)/'selection-progress.json',{'poses':len(rows),'pose_index':ordinal,
                    'planner_attempts':attempt,'last_planner_result':result,
                    'rejection_counts':dict(Counter(r['reason'] for r in rejected))})
                if result.get('exact_solution') and len(path)>1:
                    chosen={**row,**start,'pose_index':ordinal,'approach':approach,
                        'contact_model':contact_model,'full_motion_goal_rad':goal_angle,
                        'planner_result':result,'planner_attempts':attempt}
                    finish_start(start_record,'planner_exact_solution')
                    validation['terminal_gate']='selected';save_validation(ordinal,validation)
                    save(Path(output)/'grasp-search.json',{'selected':{k:v for k,v in chosen.items() if k!='pose'},
                        'rejected':rejected,'gate_counts':dict(gate_counts),'search_was_dispatched':True})
                    return chosen,problem,path
                rejected.append({'pose':ordinal,'reason':'full_constrained_path_unsolved','planner':result})
                finish_start(start_record,'full_constrained_path_unsolved')
                if attempt>=12 or pose_attempts>=2:break
            if attempt>=12 or pose_attempts>=2:break
        validation['terminal_gate']='candidate_exhausted';save_validation(ordinal,validation)
        if attempt>=12:break
    if gate_counts['planner_calls']:
        classification='bounded_constrained_search_failure'
    elif gate_counts['start_accepted']:
        classification='approach_admission_exclusion'
    elif gate_counts['starts_considered']:
        classification='start_admission_exclusion'
    elif gate_counts['goal_pose_menu_empty']:
        classification='terminal_goal_proposal_exclusion'
    else:
        classification='grasp_ik_or_candidate_budget_exhausted'
    save(Path(output)/'grasp-search.json',{'selected':None,'rejected':rejected,'planner_attempts':attempt,
        'gate_counts':dict(gate_counts),'search_was_dispatched':bool(gate_counts['planner_calls']),
        'failure_classification':classification,'candidate_poses_offered':len(rows),'candidate_pose_limit':maximum_poses})
    raise ValueError('No tested grasp has both a clear approach and a complete constrained trajectory')
