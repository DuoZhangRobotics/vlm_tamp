"""Shared observed-contact geometry, IK, collision and planning tools.

Both the scripted baseline and the subscription agent import these functions.
The caller supplies its current RGB regions and requested motion. This module
contains no task names, initial-state indices, memorized poses or success oracle.
Existing upright-bar and local contact assumptions are retained, not learned.
"""
from pathlib import Path
import time

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

TOOL_STACK_VERSION='observed-hinged-contact-v1'
# Existing baseline settings, shared by both callers. Horizons, gripper commands
# and when to reobserve remain explicit caller decisions.
COMMON_TOOL_DEFAULTS={
    'complete_planar_panel':False,'scene_voxel_spacing_m':.003,
    'maximum_grasp_poses':24,'maximum_grasp_search_seconds':600.,
    'maximum_contact_plan_seconds':8.,'planning_backend':'partial_pose_arc',
    'arc_search_seconds':20.,'arc_goal_proposals':32,
    'reconcile_handle_roles':True,
    'handle_association_m':.008,'curobo_deterministic_proposals':True,
    'minimum_manipulation_steps_reserved':200,'grasp_closure_steps_reserved':40,
}


def tool_configuration(base=None):
    config={**(base or {}),**COMMON_TOOL_DEFAULTS}
    config.setdefault('motion_policy','short')
    config.setdefault('grasp_constraint','point')
    return config


def tool_contract(config):
    return {'version':TOOL_STACK_VERSION,'settings':{k:config[k] for k in COMMON_TOOL_DEFAULTS},
        'geometry':'refined observed bar plus finite configured finger-pad support',
        'arm_collision':'calibrated convex arm links',
        'gripper_collision':'calibrated convex gripper with existing local contact policy',
        'ik':'cuRobo proposals and isolated seven-joint kinematics; eight proposals searched',
        'planner':'MC-VAMP continuous partial-pose arc',
        'task_names_or_state_indices_used':False,'simulator_object_geometry_used':False}


def ground_handle(cameras,choices,template,metadata):
    mechanism,cloud,mask,anchors,evidence=fit_observed_handle(cameras,choices,template,metadata)
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
        return contact_arc(scene,record,metadata,start,goal,folder,search_options or {})
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
    folder.mkdir();rows,pad=grasp_poses(generated,anchors,np.asarray(spec['panel_normal_world']),state['metadata'])
    initial=np.asarray(state['metadata']['joint_positions']);attempts=[]
    before=time.perf_counter()
    with NativeScene(open_scene,record,state['metadata'],initial,goal,articulated=False) as approach_scene:
        opening=approach_scene.validate(initial)
        save(folder/'approach-initial-scene.json',{'record':approach_scene.record,'start':opening})
        if not opening['accepted']:raise ValueError('Initial arm is in collision in the MC-VAMP observed point-cloud model')
        for ordinal,row in enumerate(rows[:config['maximum_grasp_poses']]):
            if time.perf_counter()-before>config['maximum_grasp_search_seconds']:break
            solutions=ik.solve_many(row['pose'],initial,maximum=8)
            solutions.sort(key=lambda s:-s['joint_reserve_rad'])
            for rank,solution in enumerate(solutions[:8]):
                q=np.asarray(solution['joints'])
                predicted=virtual_metadata(state,closed_scene.robot,ik.chain,q)
                with NativeScene(closed_scene,record,predicted,q,goal,aligned=False) as contact_scene:
                    check=contact_scene.validate(q)
                attempt={'pose_index':ordinal,'ik_rank':rank,'source':row['source'],'joints':q.tolist(),
                    'start_scene':check,'status':'start_admitted' if check['accepted'] else 'start_collision'}
                attempts.append(attempt)
                save(folder/'grasp-attempts.json',attempts)
                if not check['accepted']:continue
                # Plan to a measured 7 cm standoff before approaching the bar.
                target=np.asarray(solution['pose_world_xyzw']).copy()
                target[:3]-=.07*pose_matrix(target)[:3,2]
                pregrasps=ik.solve_many(target,q,maximum=8)
                pregrasps.sort(key=lambda s:np.linalg.norm(np.asarray(s['joints'])-q))
                approach=None
                for pre in pregrasps[:3]:
                    qpre=np.asarray(pre['joints'])
                    first,r1=approach_scene.plan(initial,qpre,constrained=False,seconds=3.)
                    if not r1['exact_solution']:continue
                    second,r2=approach_scene.plan(qpre,q,constrained=False,seconds=3.)
                    if r2['exact_solution']:
                        approach=np.vstack([first,second[1:]])
                        attempt['approach_plans']=[r1,r2];break
                if approach is None:attempt['status']='approach_planning_failed';continue
                approach_steps=len(schedule(approach,maximum_joint_speed=.30,endpoint_hold_steps=60))
                available=600-int(state['control_steps'])-int(config.get('grasp_closure_steps_reserved',40))-int(config.get('minimum_manipulation_steps_reserved',200))
                if approach_steps>available:
                    attempt.update(status='approach_exceeds_episode_budget',approach_steps=approach_steps,available_approach_steps=available);continue
                attempt['approach_steps']=approach_steps
                try:
                    path,result=contact_plan(ik,closed_scene,record,predicted,spec,q,goal,
                        folder/f'contact-pose{ordinal:02d}-ik{rank}',seconds=config['maximum_contact_plan_seconds'],
                        backend=config.get('planning_backend','endpoint_menu'),search_options=config)
                except ValueError as error:
                    attempt.update(status='contact_planning_failed',reason=str(error));continue
                attempt.update(status='selected',contact_plan=result)
                save(folder/'grasp-attempts.json',attempts)
                selected={**row,'pose_world_xyzw':solution['pose_world_xyzw'],'joints':q.tolist(),
                    'pose_index':ordinal,'ik_rank':rank,'approach':approach,'planned_contact':path,
                    'search_seconds':time.perf_counter()-before}
                save(folder/'selected.json',{k:v for k,v in selected.items() if k not in {'approach','planned_contact'}})
                np.save(folder/'approach.npy',approach);np.save(folder/'contact.npy',path)
                return selected
    save(folder/'grasp-attempts.json',attempts)
    raise ValueError('No candidate has a complete MC-VAMP approach and full-task contact path')
