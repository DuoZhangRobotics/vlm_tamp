"""Small, unmodified learned grasp proposals with one IK/route per proposal.

The sole pose conversion aligns the known Panda finger attachment and TCP
frames. No scene-dependent translations, angle menus or depth sweeps occur.
"""
import copy
import time

import numpy as np

from .coupled_storage import write_json
from .mcvamp_contact_numerical import NativeScene
from .rekep_libero.geometry import graspgen_to_tcp
from .timed_execution import schedule


def learned_poses(generated, calibration, maximum=8):
    """Convert the trained gripper frame once, preserving predicted contacts.

    Upstream config/grippers/franka_panda.py attaches fingers at z=.0584 m.
    Our robot calibration supplies its physical attachment and base->TCP.
    This fixed robot-asset correction is not a sampled contact-depth offset.
    """
    shift=np.eye(4)
    shift[2,3]=.0584-float(calibration['physical_finger_attach_z_m'])
    for index in generated['retained_indices'][:maximum]:
        index=int(index)
        yield {'pose':graspgen_to_tcp(generated['poses_world_base'][index]@shift,calibration),
            'source':'GraspGen full pose with fixed robot frame conversion',
            'graspgen_index':index,'graspgen_score':float(generated['scores'][index]),
            'variant':'calibrated_asset_to_tcp','asset_alignment_z_m':float(shift[2,3]),
            'contact_depth_offset_m':0.,'predicted_contact_pose_preserved':True}


def select_direct_grasp(generated,state,ik,record,open_scene,closed_scene,goal,folder,config,budget):
    from .hinged_contact_tools import virtual_metadata
    folder.mkdir()
    began=time.perf_counter();deadline=began+float(config.get('maximum_grasp_search_seconds',30.))
    initial=np.asarray(state['metadata']['joint_positions'])
    rows=list(learned_poses(generated,state['metadata']['robot_calibration'],
        maximum=int(config.get('maximum_grasp_poses',8))))
    open_scene=copy.copy(open_scene);closed_scene=copy.copy(closed_scene)
    open_scene.skill_contact_surface=closed_scene.skill_contact_surface='fingertips'
    predicted=virtual_metadata(state,closed_scene.robot,ik.chain,initial)
    policy=config.get('environment_collision_policy','advisory')
    attempts=[];queries=0;ik_seconds=path_seconds=0.;best=None
    speed=budget['maximum_joint_speed'];hold=budget['endpoint_hold_steps']
    with NativeScene(open_scene,record,state['metadata'],initial,goal,articulated=False,
                     allow_contact=False,collision_policy=policy) as approach_context, \
         NativeScene(closed_scene,record,predicted,initial,goal,articulated=False,
                     collision_policy=policy) as contact_context:
        start=approach_context.validate(initial)
        write_json(folder/'approach-initial-scene.json',{'record':approach_context.record,'start':start})
        if not start['accepted']:
            error=ValueError('Measured start fails the selected robot/collision admission')
            error.planning_diagnostic={'status':'invalid_start','diagnostic':start}
            raise error
        for ordinal,row in enumerate(rows):
            if time.perf_counter()>=deadline:break
            attempt={'pose_index':ordinal,'graspgen_index':row['graspgen_index'],
                'graspgen_score':row['graspgen_score'],'source':row['source']}
            attempts.append(attempt)
            t=time.perf_counter()
            # solve() chooses the closest robot-valid returned solution to the
            # current joints. Internal GPU seeds are not extra route candidates.
            solution=ik.solve(row['pose'],initial)
            ik_seconds+=time.perf_counter()-t
            if solution is None:
                attempt['status']='no_ik_solution';continue
            q=np.asarray(solution['joints'])
            check=contact_context.validate(q)
            attempt.update(joints=q.tolist(),start_scene=check)
            if not check['accepted']:
                attempt['status']='endpoint_rejected';continue
            lower=int(np.ceil(np.max(np.abs(q-initial))/speed*20))+hold
            attempt['approach_lower_bound_controls']=lower
            if lower>budget['available_approach_steps']:
                attempt.update(status='approach_exceeds_episode_budget',approach_steps=lower);continue
            remaining=deadline-time.perf_counter()
            if remaining<=0:attempt['status']='search_deadline';break
            t=time.perf_counter()
            path,result=approach_context.plan(initial,q,constrained=False,seconds=min(2.,remaining))
            path_seconds+=time.perf_counter()-t;queries+=1
            route={'route':policy,'standoff_m':0.,'one_route_per_learned_pose':True}
            with (folder/'approach-queries.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps({'pose_index':ordinal,'ik_rank':0,**route,**result})+'\n')
            attempt['planner_status']=result.get('status')
            if not result.get('exact_solution'):
                attempt['status']='approach_search_incomplete';continue
            count=len(schedule(path,maximum_joint_speed=speed,endpoint_hold_steps=hold))
            if count>budget['available_approach_steps']:
                attempt.update(status='approach_exceeds_episode_budget',approach_steps=count);continue
            attempt.update(status='selected',approach_steps=count)
            best={**row,'pose_world_xyzw':solution['pose_world_xyzw'],'joints':q.tolist(),
                'pose_index':ordinal,'ik_rank':0,'approach':path,'planned_contact':None,
                'manipulation_plan_requires_measured_closure':True,
                'requested_manipulation_radians':float(goal),'approach_selection':route,
                'budget_timing_assumptions':{'speed':speed,'hold':hold,
                    'close_steps':budget['grasp_closure_steps_reserved'],'approach_controls':count,
                    'manipulation_controls':None,'complete_sequence_controls':None,
                    'approach_and_closure_controls':count+budget['grasp_closure_steps_reserved'],
                    'remaining_control_steps':budget['remaining_control_steps']}}
            break
    write_json(folder/'grasp-attempts.json',attempts)
    elapsed=time.perf_counter()-began
    summary={'method':'graspgen_direct','learned_pose_count':len(rows),'examined':len(attempts),
        'generated_ik_candidates':sum('joints' in a for a in attempts),'approach_queries':queries,
        'deadline_reached':time.perf_counter()>=deadline,'global_optimality_claimed':False,
        'geometric_fallback_used':False,'selection':'first executable learned pose by discriminator rank; closest valid returned IK'}
    timing={'ik_generation_and_queue':ik_seconds,'approach_queries':path_seconds,
        'admission_scene_setup_and_bookkeeping':max(0.,elapsed-ik_seconds-path_seconds)}
    write_json(folder/'search-summary.json',{'search_summary':summary,'search_timing_seconds':timing})
    if best is None:
        error=ValueError('No executable approach in the small GraspGen batch')
        error.planning_diagnostic={'status':'grasp_search_budget_exhausted' if summary['deadline_reached'] else 'grasp_candidates_exhausted',
            'search_complete':len(attempts)==len(rows) and not summary['deadline_reached'],
            'physical_infeasibility_proved':False,'search_summary':summary,'search_timing_seconds':timing}
        raise error
    best.update(search_seconds=elapsed,search_summary=summary,search_timing_seconds=timing)
    write_json(folder/'selected.json',{k:v for k,v in best.items() if k not in {'approach','planned_contact'}})
    np.save(folder/'approach.npy',best['approach'])
    return best
