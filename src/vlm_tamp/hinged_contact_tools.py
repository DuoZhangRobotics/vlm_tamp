"""Shared observed-contact geometry, IK, collision and planning tools.

Both the scripted baseline and the subscription agent import these functions.
The caller supplies its current RGB regions and requested motion. This module
contains no task names, initial-state indices, memorized poses or success oracle.
Existing upright-bar and local contact assumptions are retained, not learned.
"""
from pathlib import Path
import json
import time
import copy

import numpy as np

from .arc_contact_scene import ArcContactScene
from .coupled_manifold import CoupledHandlePoint
from .coupled_pipeline import handle_anchor_menu,grasp_poses,goal_pose_menu
from .coupled_storage import write_json,preserve_scene
from .mcvamp_contact_numerical import NativeScene,clear_native_cache
from .rekep_libero.task_ik_curobo import TaskIK
from .rekep_libero.geometry import pose_matrix,matrix_pose
from .reliable_handle import ground_handle as fit_observed_handle
from .timed_execution import schedule
from .execution_limits import control_limit

TOOL_STACK_VERSION='observed-hinged-contact-v2'
GRASP_APPROACH_HOLD_STEPS=6
# Existing baseline settings, shared by both callers. Horizons, gripper commands
# and when to reobserve remain explicit caller decisions.
COMMON_TOOL_DEFAULTS={
    'complete_planar_panel':False,'scene_voxel_spacing_m':.003,
    'maximum_grasp_poses':24,'maximum_grasp_search_seconds':30.,
    'maximum_contact_plan_seconds':8.,'planning_backend':'partial_pose_arc',
    'arc_search_seconds':20.,'arc_goal_proposals':32,
    'arc_position_tolerance_m':.006,
    'arc_orientation_tolerance_rad':float(np.radians(3.)),
    'max_wrist_reorientation_deg':None,'maximum_local_joint_step_rad':None,
    'maximum_grasp_ik_solutions':32,
    'curobo_v2_arc_solver_mode':'local_continuation',
    'curobo_v2_global_fallback':True,
    'arc_continuation_step_deg':3.,'arc_continuation_speed_deg_s':15.,
    'curobo_v2_moving_mesh_pitch_m':.002,
    'reconcile_handle_roles':True,
    'handle_association_m':.008,'curobo_deterministic_proposals':True,
    'minimum_manipulation_steps_reserved':0,'grasp_closure_steps_reserved':40,
    'grasp_budget_speed_rad_s':.6,'grasp_budget_hold_steps':6,
    'grasp_approach_speed_rad_s':.6,'grasp_standoff_distances_m':[0.,.03,.07],
    'environment_collision_policy':'advisory','command_joint_reserve_rad':0.,'native_joint_reserve_rad':0.,
    'ik_position_tolerance_m':.003,'ik_orientation_tolerance_rad':.01,
    'curobo_backend':'v2','arc_goal_backend':'curobo_v2',
}


def grasp_timing_options(speed=.6,hold=6,close_steps=40):
    """Use the executor's units and valid range when pricing a prospective plan."""
    if (not np.isfinite(speed) or not 0<float(speed)<=1.0 or
            type(hold) is not int or hold<0 or
            type(close_steps) is not int or close_steps<1):
        raise ValueError('Use speed (0,1.0] rad/s, nonnegative hold controls, and positive close_steps')
    return {'grasp_budget_speed_rad_s':float(speed),'grasp_budget_hold_steps':hold,
        'grasp_closure_steps_reserved':close_steps}


def grasp_budget_check(state, config):
    """Reject only impossible explicit reservations; price the actual path later."""
    used=int(state['control_steps'])
    closure=int(config.get('grasp_closure_steps_reserved',40))
    manipulation=int(config.get('minimum_manipulation_steps_reserved',0))
    hold=int(config.get('grasp_budget_hold_steps',GRASP_APPROACH_HOLD_STEPS))
    speed=float(config.get('grasp_budget_speed_rad_s',.6))
    grasp_timing_options(speed,hold,closure)
    limit=control_limit(state,config)
    available=limit-used-closure-manipulation
    return {'accepted':available>=hold+1,
        'control_steps':used,'remaining_control_steps':limit-used,'maximum_control_steps':limit,
        'grasp_closure_steps_reserved':closure,'manipulation_steps_reserved':manipulation,
        'available_approach_steps':available,'minimum_approach_steps':hold+1,
        'maximum_joint_speed':speed,'endpoint_hold_steps':hold,
        'physical_motion_issued':False}


def require_grasp_budget(state, config):
    check=grasp_budget_check(state,config)
    if not check['accepted']:
        error=ValueError(f"Insufficient grasp budget: {check['available_approach_steps']} approach controls "
            f"remain after reserving closure and manipulation; at least {check['minimum_approach_steps']} "
            "approach controls are required, including the requested endpoint hold. No grasp search was run.")
        error.planning_diagnostic={'status':'insufficient_grasp_budget',**check}
        raise error
    return check


def tool_configuration(base=None):
    config={**(base or {}),**COMMON_TOOL_DEFAULTS}
    for key in COMMON_TOOL_DEFAULTS:
        if base and key in base:config[key]=base[key]
    config.setdefault('motion_policy','short')
    config.setdefault('grasp_constraint','point')
    return config


def tool_contract(config):
    return {'version':TOOL_STACK_VERSION,'settings':{k:config[k] for k in COMMON_TOOL_DEFAULTS},
        'geometry':'refined observed bar plus finite configured finger-pad support',
        'arm_collision':'calibrated convex arm links',
        'gripper_collision':'calibrated convex gripper with existing local contact policy',
        'ik':f"cuRobo {config.get('curobo_backend','v1')} proposals and isolated seven-joint kinematics; up to {config.get('maximum_grasp_ik_solutions',32)} solutions per grasp pose",
        'planner':('cuRobo v2 contact-point goals with native MC-VAMP constrained connections'
            if config.get('arc_goal_backend')=='curobo_v2' else 'MC-VAMP continuous partial-pose arc'),
        'task_names_or_state_indices_used':False,'simulator_object_geometry_used':False}


def ground_handle(cameras,choices,template,metadata,*,allow_direction_conflict=False,fixed_joint=None):
    mechanism,cloud,mask,anchors,evidence=fit_observed_handle(cameras,choices,template,metadata,fixed_joint=fixed_joint)
    if not allow_direction_conflict:
        from .handle_contact import require_direction_agreement
        require_direction_agreement(evidence)
    anchors,support=handle_anchor_menu(anchors[0][:2],cloud,metadata)
    return mechanism,cloud,mask,anchors,{**evidence,'pad_support':support,
        'shared_tool_stack':TOOL_STACK_VERSION}


def save(path,data):
    write_json(path,data)


def virtual_metadata(state,robot,chain,q):
    result=dict(state['metadata']);result.update(joint_positions=np.asarray(q).tolist(),
        eef_pose_world_xyzw=chain.forward(q).tolist(),robot_collision_volumes_world=robot.forward(q))
    return result


def constraint_model(chain,spec,q,pad,goal,aligned=True):
    return CoupledHandlePoint(chain,spec['joint_origin_world_m'],spec['joint_axis_world'],pose_matrix(chain.forward(q)),
        pad_local=pad,align_axis=False,jaw_axis_local=[1.,0.,0.] if aligned else None,
        jaw_tilt_limit=None if aligned else np.pi/3,orientation_cone=np.pi/2,
        progress_bounds=(-.03,goal+.03),joint_margin=.025,position_tolerance=.001,orientation_tolerance=.01)


def goal_candidates(ik,context,spec,start,goal,*,maximum=4):
    """IK proposes endpoints; MC-VAMP checks their point-cloud occupancy."""
    model=constraint_model(ik.chain,spec,start,context.pad_local,goal,context.record['aligned_jaws'])
    candidates=[];records=[]
    def admit(q,source):
        q,projection=context.project(q)
        check=context.validate(q) if projection['accepted'] else None
        error=context.progress(q)-goal
        okay=bool(projection['accepted'] and abs(error)<.005 and check['accepted'])
        records.append({'source':'endpoint_admission','proposal_source':source,'joints':q,
            'projection':projection,'scene':check,'accepted':okay,'goal_error_rad':error})
        if okay and not any(np.max(np.abs(q-other))<.03 for other in candidates):candidates.append(q)
    previous=np.asarray(start,float)
    # Continuation proposes a terminal posture, not an executed controller path.
    # The actual connecting path is always searched and validated by MC-VAMP.
    for angle in np.linspace(0,goal,max(2,int(np.ceil(goal/.08))+1))[1:]:
        projected=model.project(np.r_[previous,angle],fixed_progress=float(angle),max_iterations=75)
        records.append({'source':'terminal_posture_continuation','angle_rad':float(angle),
            'accepted':projected['accepted'],'position_error_m':projected['position_error_m']})
        if not projected['accepted']:break
        previous=projected['state'][:7]
    else:
        admit(previous,'terminal_posture_continuation')
    for target,angles in goal_pose_menu(model,goal,pose_matrix(ik.chain.forward(start)),context.pad_local,
                                        spec['joint_axis_world'],allow_roll=True):
        if len(candidates)>=maximum:break
        solutions=ik.solve_many(matrix_pose(target),start,maximum=8)
        records.append({'source':'curobo_terminal_pose','angles':angles,'returned':len(solutions)})
        for solution in solutions:
            q=np.asarray(solution['joints'])
            admit(q,'curobo_terminal_pose')
            if len(candidates)>=maximum:break
    # Rejected endpoints do not consume the feasible-goal quota. Previously
    # four colliding proposals prevented searching the remaining wrist poses.
    candidates.sort(key=lambda q:np.linalg.norm(q-start))
    return candidates,records


def contact_plan(ik,scene,record,metadata,spec,start,goal,folder,*,seconds=8.,modes=(True,False),backend='endpoint_menu',search_options=None):
    if backend=='partial_pose_arc':
        from .arc_execution import contact_arc
        return contact_arc(scene,record,metadata,start,goal,folder,search_options or {},ik=ik)
    folder.mkdir(parents=True,exist_ok=False);attempts=[]
    preserve_scene(scene,folder)
    compact_metadata={key:metadata[key] for key in ['joint_positions','eef_pose_world_xyzw']}
    compact_metadata['robot_collision_volumes_world']=[v for v in metadata['robot_collision_volumes_world']
        if 'gripper' in v['robot_geom_name']]
    save(folder/'planning-input.json',{'task_spec':spec,'metadata':compact_metadata,'reference_joints':start,
        'goal_angle_rad':goal,'maximum_solve_seconds':seconds,'planning_only':True})
    for aligned in modes:
        with NativeScene(scene,record,metadata,start,goal,aligned=aligned) as context:
            begin=context.validate(start)
            if not begin['accepted']:
                attempts.append({'aligned_jaws':aligned,'status':'invalid_start','scene':begin});continue
            goals,proposals=goal_candidates(ik,context,spec,start,goal)
            save(folder/('goals-aligned.json' if aligned else 'goals-tilted.json'),proposals)
            for number,qgoal in enumerate(goals[:3]):
                path,result=context.plan(start,qgoal,seconds=seconds)
                result.update(aligned_jaws=aligned,goal_joints=qgoal.tolist())
                attempts.append(result)
                save(folder/'attempts.json',attempts)
                if result['exact_solution']:
                    progress=[context.progress(q) for q in path]
                    np.save(folder/'path.npy',path);np.save(folder/'door-progress.npy',progress)
                    save(folder/'native-scene.json',context.record)
                    preserve_scene(scene,folder)
                    save(folder/'result.json',result)
                    return path,result
    save(folder/'attempts.json',attempts)
    raise ValueError('MC-VAMP found no complete contact path to the full task endpoint')


def select_grasp(generated,anchors,state,ik,record,open_scene,closed_scene,spec,goal,folder,config):
    """Select the lowest-cost reachable approach found within the search budget.

    The manipulation request is retained as intent. Its executable path is
    generated after physical closure, using measured robot and part geometry.
    """
    budget=require_grasp_budget(state,config)
    folder.mkdir();base_rows,pad=grasp_poses(generated,anchors,np.asarray(spec['panel_normal_world']),state['metadata'])
    from .rekep_libero.geometry import transform_points
    inverse=np.linalg.inv(pose_matrix(state['metadata']['eef_pose_world_xyzw']))
    pad_points=np.vstack([transform_points(v['vertices_world'],inverse) for v in state['metadata']['robot_collision_volumes_world']
        if v['robot_geom_name'].endswith('_pad_collision')])
    zlo,zhi=float(pad_points[:,2].min()),float(pad_points[:,2].max())
    rows=[]
    for original in base_rows[:int(config['maximum_grasp_poses'])]:
        pose=np.asarray(original['pose']);rotation=pose_matrix(pose)[:3,:3]
        for fraction in config.get('grasp_pad_depth_fractions',[.5,.8,1.]):
            local=pad.copy();local[2]=zlo+float(fraction)*(zhi-zlo)
            shifted=pose.copy();shifted[:3]+=rotation@(pad-local)
            rows.append({**original,'pose':shifted,'contact_depth_offset_m':float(local[2]-pad[2]),
                'pad_depth_fraction':float(fraction),'contact_reference':'point on the calibrated pad face'})
    open_scene=copy.copy(open_scene);closed_scene=copy.copy(closed_scene)
    open_scene.skill_contact_surface=closed_scene.skill_contact_surface='fingertips'
    initial=np.asarray(state['metadata']['joint_positions']);attempts=[];before=time.perf_counter()
    deadline=before+float(config.get('maximum_grasp_search_seconds',30.))
    speed=budget['maximum_joint_speed'];hold=budget['endpoint_hold_steps'];maximum=int(config.get('maximum_grasp_ik_solutions',32))
    queue=[];best=None;best_cost=float('inf');fallback=None;fallback_cost=float('inf');examined=0;generation_complete=True
    for ordinal,row in enumerate(rows):
        if time.perf_counter()>=deadline:generation_complete=False;break
        solutions=ik.solve_many(row['pose'],initial,maximum=maximum)
        for rank,solution in enumerate(solutions):
            q=np.asarray(solution['joints']);lower=int(np.ceil(np.max(np.abs(q-initial))/speed*20))+hold
            queue.append((lower,float(np.linalg.norm(q-initial)),ordinal,rank,row,solution))
        if not solutions:attempts.append({'pose_index':ordinal,'source':row['source'],'status':'no_ik_solution'})
    queue.sort(key=lambda item:item[:4])
    ik_generation_seconds=time.perf_counter()-before
    approach_query_seconds=0.
    predicted_initial=virtual_metadata(state,closed_scene.robot,ik.chain,initial)
    policy=config.get('environment_collision_policy','advisory')
    with NativeScene(open_scene,record,state['metadata'],initial,goal,articulated=False,allow_contact=False,collision_policy=policy) as context, \
         NativeScene(closed_scene,record,predicted_initial,initial,goal,articulated=False,collision_policy=policy) as contact_context, \
         NativeScene(open_scene,record,state['metadata'],initial,goal,articulated=False,allow_contact=False,collision_policy='veto') as observed, \
         NativeScene(closed_scene,record,predicted_initial,initial,goal,articulated=False,collision_policy='veto') as observed_contact:
        start=context.validate(initial);observed_start=observed.validate(initial)
        save(folder/'approach-initial-scene.json',{'record':context.record,'start':start,'observed_start':observed_start})
        if not start['accepted']:
            error=ValueError('Measured start fails robot/selected-policy admission; inspect the detailed bounds or self/scene evidence')
            error.planning_diagnostic={'status':'invalid_start','diagnostic':start};raise error
        for lower,_,ordinal,rank,row,solution in queue:
            if time.perf_counter()>=deadline:break
            if lower>=best_cost:break
            examined+=1;q=np.asarray(solution['joints']);check=contact_context.validate(q)
            attempt={'pose_index':ordinal,'ik_rank':rank,'source':row['source'],'joints':q.tolist(),
                'approach_lower_bound_controls':lower,'start_scene':check,'status':'endpoint_admitted' if check['accepted'] else 'endpoint_rejected'}
            attempts.append(attempt)
            if not check['accepted']:continue
            observed_endpoint=observed.validate(q);observed_closed=observed_contact.validate(q)
            clear_goal=bool(observed_start['accepted'] and observed_endpoint['accepted'] and observed_closed['accepted'])
            attempt['observed_goal_admitted']=clear_goal
            if policy=='veto' and not clear_goal:continue
            if lower>budget['available_approach_steps']:
                attempt.update(status='approach_exceeds_episode_budget',approach_steps=lower);continue
            paths=[]
            for gap in config.get('grasp_standoff_distances_m',[0.,.03,.07]):
                if time.perf_counter()>=deadline:break
                if any(n<=lower and r['route']=='observed' for n,_,r in paths):break
                if gap==0:pregrasps=[{'joints':initial}]
                else:
                    target=np.asarray(solution['pose_world_xyzw']).copy();target[:3]-=float(gap)*pose_matrix(target)[:3,2]
                    pregrasps=ik.solve_many(target,q,maximum=8)
                    pregrasps.sort(key=lambda r:np.linalg.norm(np.asarray(r['joints'])-initial)+np.linalg.norm(np.asarray(r['joints'])-q))
                for pre in pregrasps:
                    if time.perf_counter()>=deadline:break
                    qpre=np.asarray(pre['joints']);ends=[(initial,q)] if gap==0 else [(initial,qpre),(qpre,q)]
                    path=None;route=None
                    # Prefer a route through observed free space for this
                    # posture; failed predictions do not veto the advisory pass.
                    contexts=([('observed',observed)] if clear_goal else [])
                    if policy=='advisory' and lower<fallback_cost:contexts.append(('advisory',context))
                    for route_name,planner in contexts:
                        pieces=[]
                        for a,b in ends:
                            remaining=deadline-time.perf_counter()
                            if remaining<=0:break
                            query_started=time.perf_counter()
                            piece,result=planner.plan(a,b,constrained=False,seconds=min(2.,remaining))
                            approach_query_seconds+=time.perf_counter()-query_started
                            with (folder/'approach-queries.jsonl').open('a') as stream:
                                stream.write(json.dumps({'pose_index':ordinal,'ik_rank':rank,'route':route_name,'standoff_m':gap,**result})+'\n')
                            if not result.get('exact_solution'):break
                            pieces.append(piece)
                        if len(pieces)==len(ends):
                            path=np.vstack([pieces[0],*(x[1:] for x in pieces[1:])]);route=route_name;break
                    if path is None:continue
                    count=len(schedule(path,maximum_joint_speed=speed,endpoint_hold_steps=hold))
                    paths.append((count,path,{'route':route,'standoff_m':float(gap)}))
                    break
            if not paths:attempt['status']='approach_search_incomplete';continue
            clear_paths=[p for p in paths if p[2]['route']=='observed']
            count,approach,route=min(clear_paths or paths,key=lambda x:x[0]);attempt.update(approach_steps=count,**route)
            if count>budget['available_approach_steps']:
                attempt['status']='approach_exceeds_episode_budget';continue
            is_clear=route['route']=='observed'
            if (is_clear and count<best_cost) or (not is_clear and count<fallback_cost):
                candidate={**row,'pose_world_xyzw':solution['pose_world_xyzw'],'joints':q.tolist(),
                    'pose_index':ordinal,'ik_rank':rank,'approach':approach,'planned_contact':None,
                    'manipulation_plan_requires_measured_closure':True,'requested_manipulation_radians':float(goal),
                    'approach_selection':route,
                    'budget_timing_assumptions':{'speed':speed,'hold':hold,'close_steps':budget['grasp_closure_steps_reserved'],
                        'approach_controls':count,'manipulation_controls':None,'complete_sequence_controls':None,
                        'approach_and_closure_controls':count+budget['grasp_closure_steps_reserved'],
                        'remaining_control_steps':budget['remaining_control_steps']}}
                if is_clear:best,best_cost=candidate,count
                else:fallback,fallback_cost=candidate,count
                attempt['status']='best_observed_route_so_far' if is_clear else 'best_advisory_fallback_so_far'
            else:attempt['status']='higher_control_cost'
            save(folder/'grasp-attempts.json',attempts)
    save(folder/'grasp-attempts.json',attempts)
    used_fallback=best is None
    if best is None:best=fallback
    if best is None:
        expired=time.perf_counter()>=deadline
        error=ValueError('Grasp search budget exhausted' if expired else 'No executable approach among the grasp candidates examined')
        error.planning_diagnostic={'status':'grasp_search_budget_exhausted' if expired else 'grasp_candidates_exhausted',
            'search_complete':bool(generation_complete and not expired),'physical_infeasibility_proved':False,
            'generated_ik_candidates':len(queue),'examined_candidates':examined,
            'elapsed_seconds':time.perf_counter()-before,'maximum_search_seconds':config.get('maximum_grasp_search_seconds',30.)}
        raise error
    best.update(search_seconds=time.perf_counter()-before,
        search_timing_seconds={'ik_generation_and_queue':ik_generation_seconds,
            'approach_queries':approach_query_seconds,
            'admission_scene_setup_and_bookkeeping':max(0.,time.perf_counter()-before-ik_generation_seconds-approach_query_seconds)},
        search_summary={'generated_ik_candidates':len(queue),'examined':examined,'ik_generation_complete':generation_complete,
            'deadline_reached':time.perf_counter()>=deadline,'selection':'lowest control cost among observed-clear candidates when found; otherwise robot-valid advisory fallback',
            'advisory_fallback_used':used_fallback,
            'global_optimality_claimed':False})
    save(folder/'selected.json',{k:v for k,v in best.items() if k not in {'approach','planned_contact'}})
    np.save(folder/'approach.npy',best['approach'])
    return best
