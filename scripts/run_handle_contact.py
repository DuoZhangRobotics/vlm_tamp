"""RGB-D handle approach with verified arrival and short feedback pulls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT/'.runtime/mc-vamp-deps')]

from try_rekep_bowl_v3 import save, read_observation_v3
from vlm_tamp.handle_contact import (ground_handle, frontal_poses, require_arrival, arrival_error,
    fit_upright_hinge, ObservedRobotScene, translated_surface)
from vlm_tamp.handle_contact import local_contact_ik
from vlm_tamp.handle_contact import contact_pose_variants
from vlm_tamp.rekep_libero.geometry import pose_matrix, matrix_pose, transform_points
from vlm_tamp.rekep_libero.mechanism_execution import cached_grounding
from vlm_tamp.rekep_libero.physical_tasks_v4 import candidate_variants
from vlm_tamp.rekep_libero.sim_client_articulated import ArticulatedClient
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edge(scene, sim, ik, first, last, aperture, target, step=.05):
    first, last = np.asarray(first), np.asarray(last)
    count = max(1, int(np.ceil(np.max(np.abs(last-first))/step)))
    for fraction in np.linspace(0., 1., count+1):
        q = first+fraction*(last-first)
        if fraction==0 and (np.any(q<ik.chain.limits[:,0]) or np.any(q>ik.chain.limits[:,1])):
            if np.any(q<ik.chain.limits[:,0]-.001) or np.any(q>ik.chain.limits[:,1]+.001):
                return {'accepted':False,'reason':'measured_start_outside_physical_softstop_tolerance'}
            # Actual geometry was checked from the measured state. Do not feed
            # a tiny physical soft-stop overshoot to hypothetical bounded FK.
            continue
        if not ik.self_valid(q):
            return {'accepted': False, 'reason': 'self_collision'}
        contact = np.linalg.norm(ik.chain.forward(q)[:3]-target[:3])<.025
        result = scene.joints(sim, q, aperture, contact)
        if not result['accepted']:
            return {**result, 'reason': 'robot_observed_surface_collision'}
    return {'accepted': True}


def checked_path(scene, sim, ik, path, aperture, target):
    for first, last in zip(path[:-1], path[1:]):
        result = edge(scene, sim, ik, first, last, aperture, target)
        if not result['accepted']:
            return result
    return {'accepted': True}


def rrt(scene, sim, ik, start, goal, aperture, target, seconds=45):
    """Bounded bidirectional joint RRT; all edges inspect measured surfaces."""
    rng = np.random.default_rng(0)
    trees = [[np.asarray(start)], [np.asarray(goal)]]
    parents = [[-1], [-1]]
    begin = time.monotonic()
    swapped = False
    def extend(nodes, links, destination):
        near = int(np.argmin([np.linalg.norm(q-destination) for q in nodes]))
        direction = destination-nodes[near]
        q = nodes[near]+direction*min(1., .22/max(np.linalg.norm(direction), 1e-9))
        result = edge(scene, sim, ik, nodes[near], q, aperture, target)
        if not result['accepted']:
            return None
        nodes.append(q); links.append(near)
        return len(nodes)-1
    def branch(nodes, links, index):
        path = []
        while index>=0:
            path.append(nodes[index]); index = links[index]
        return path[::-1]
    for iteration in range(400):
        if time.monotonic()-begin>seconds:
            break
        sample = trees[1][0] if rng.random()<.2 else rng.uniform(ik.chain.limits[:, 0]+.05, ik.chain.limits[:, 1]-.05)
        index = extend(trees[0], parents[0], sample)
        if index is not None:
            destination = trees[0][index]
            for _ in range(60):
                other = extend(trees[1], parents[1], destination)
                if other is None:
                    break
                if np.linalg.norm(trees[1][other]-destination)<1e-8:
                    path = branch(trees[0], parents[0], index)+branch(trees[1], parents[1], other)[-2::-1]
                    return path[::-1] if swapped else path
        trees.reverse(); parents.reverse(); swapped = not swapped
    return None


def select(generated, anchors, normal, metadata, ik, sim, scene, output):
    inverse = np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
    pads = [transform_points(v['vertices_world'], inverse).mean(0)
            for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name'].endswith('_pad_collision')]
    pad_local = np.mean(pads, axis=0)
    candidates = []
    for index, variant, pose in candidate_variants(generated, metadata['robot_calibration']):
        R = pose_matrix(pose)[:3, :3]
        if -R[:, 2]@normal<.85 or abs(R[:, 0]@[0., 0., 1.])>.35:
            continue
        if min(np.linalg.norm(pose[:3]-p) for p in anchors)>.08:
            continue
        for anchor_index, anchor in enumerate(anchors[:2]):
            refined = np.asarray(pose).copy()
            refined[:3] = anchor-R@pad_local
            candidates.append({'pose': refined, 'source': 'GraspGen pose with RGB-D handle-centre translation',
                               'graspgen_index': index, 'variant': variant, 'anchor_index': anchor_index})
    candidates += frontal_poses(normal, anchors, pad_local)
    unique = []
    # Favour a frontal approach, then use the detector's rotation among similar poses.
    candidates.sort(key=lambda row: (row['anchor_index'], 1+pose_matrix(row['pose'])[:3, 2]@normal))
    for row in candidates:
        if not any(np.linalg.norm(row['pose'][:3]-old['pose'][:3])<.003 and
                   Rotation.from_matrix(pose_matrix(row['pose'])[:3, :3].T@pose_matrix(old['pose'])[:3, :3]).magnitude()<.06
                   for old in unique):
            unique.append(row)
    rejected, deferred = [], []
    initial, aperture = metadata['joint_positions'], metadata['gripper_qpos']
    tested = 0
    for ordinal, row in enumerate(unique[:28]):
        solutions = ik.solve_many(row['pose'], initial, maximum=32)
        solutions.sort(key=lambda item:-item['joint_reserve_rad']+.01*item['posture_cost'])
        for alternative, solution in enumerate(solutions):
            tested += 1
            goal = solution['pose_world_xyzw']
            record = {'candidate': ordinal, 'ik_alternative': alternative, 'source': row['source']}
            if tested>240:
                break
            if solution['joint_reserve_rad']<.15:
                rejected.append({**record,'reason':'insufficient_grasp_joint_reserve'});continue
            result = scene.joints(sim, solution['joints'], aperture, True)
            if not result['accepted']:
                rejected.append({**record, 'reason': 'grasp_body_collision', 'check': result}); continue
            approach = ik.approach(np.asarray(initial), solution)
            if not approach['accepted']:
                rejected.append({**record, 'reason': approach['reason']}); continue
            result = checked_path(scene, sim, ik, approach['joint_postures'], aperture, goal)
            chosen = {**row, **solution, 'candidate': ordinal, 'ik_alternative': alternative,
                      'approach': approach, 'tested_postures': tested}
            save(output/'selection-progress.json', {'tested': tested, 'candidate': ordinal,
                 'rejection_counts': dict(Counter(r['reason'] for r in rejected))})
            if result['accepted']:
                save(output/'grasp-search.json', {'selected': {k:v for k,v in chosen.items() if k!='pose'},
                     'rejected': rejected, 'pose_candidates': len(unique), 'tested_postures': tested})
                return chosen
            rejected.append({**record, 'reason': 'approach_body_collision', 'check': result})
            if len(deferred)<2:
                deferred.append(chosen)
        if tested>240:
            break
    for chosen in deferred:
        target = np.asarray(chosen['pose_world_xyzw'])
        standoff = target[:3]-.07*pose_matrix(target)[:3, 2]
        path = np.asarray(chosen['approach']['joint_postures'])
        index = int(np.argmin([np.linalg.norm(ik.chain.forward(q)[:3]-standoff) for q in path]))
        free_path = rrt(scene, sim, ik, initial, path[index], aperture, target)
        if free_path is not None:
            combined = np.vstack([free_path, path[index+1:]])
            chosen['approach'] = {'joint_postures': combined.tolist(),
                'poses_world_xyzw': [ik.chain.forward(q).tolist() for q in combined],
                'method': 'sampled joint RRT to frontal standoff, Cartesian continuation to handle'}
            save(output/'grasp-search.json', {'selected': {k:v for k,v in chosen.items() if k!='pose'},
                 'rejected': rejected, 'pose_candidates': len(unique), 'tested_postures': tested})
            return chosen
    save(output/'grasp-search.json', {'selected': None, 'rejected': rejected,
         'pose_candidates': len(unique), 'tested_postures': tested})
    raise ValueError('No tested handle pose has a robot-clear approach')


def run(task, output, config, plan_only):
    output.mkdir(parents=True, exist_ok=False)
    sim, ik = ArticulatedClient(output/'simulator.log'), None
    stage, commands, tracking, closure_authorizations = 'reset', [], [], []
    clearance_refreshes=[]
    summary = {'task': task['label'], 'official_task_name': task['name'], 'initial_state': 1,
        'development': True, 'official_task_success': False, 'simulator_geometry_used_for_policy': False,
        'assisted_grasping': False, 'physical_contacts_enabled': True, 'new_api_calls': 0}
    try:
        state = sim.reset(task['name'], 1, output/'initial')
        state = sim.action(state['metadata']['eef_pose_world_xyzw'], gripper=-1, max_steps=60, output_dir=output/'opened')
        reference = ROOT/'results/articulated-physics-pilot'/f'{task["label"]}-state001'
        bundle, choices, _, _ = cached_grounding(reference/'grounding', output/'opened', state['metadata'], output/'grounding')
        # Grounding's image loader does not attach the robot-only render prior.
        # Collision scenes must use the synchronized sensor loader that does.
        bundle['cameras'], _ = read_observation_v3(output/'opened', state['metadata'])
        stage = 'handle_geometry'
        model, cloud, panel_mask, anchors, evidence = ground_handle(bundle['cameras'], choices,
            read(reference/'task-spec.json'), state['metadata'])
        save(output/'handle-geometry.json', evidence); save(output/'task-spec.json', model.spec)
        np.save(output/'handle-cloud.npy', cloud)
        with np.load(reference/'grasps/grasp_candidates.npz') as data:
            generated = {k:data[k].copy() for k in ['poses_world_base', 'scores', 'retained_indices']}
        save(output/'grasp-provenance.json', {'source': str((reference/'grasps/grasp_candidates.npz').relative_to(ROOT)),
            'sha256': sha(reference/'grasps/grasp_candidates.npz'), 'new_detector_inference': False,
            'refinement': 'detector rotations and frontal templates centred on newly measured handle'})
        record = sim.robot_chain();save(output/'robot-chain.json', record)
        import vamp
        empty = vamp.Environment()
        self_valid = lambda q: vamp.panda.validate(np.asarray(q).tolist(), empty, True)
        stage = 'ik_calibration'
        ik = TaskIK(record, config, self_valid, output=output, sim=sim)
        panel_origin=np.asarray(model.spec['front_plane']['point_world_m'])
        panel_normal=np.asarray(model.spec['panel_normal_world'])
        scene = ObservedRobotScene(bundle['cameras'], state['metadata'], cloud, config,
                                   panel=(panel_origin,panel_normal))
        stage = 'grasp_selection'
        chosen = select(generated, anchors, np.asarray(model.spec['panel_normal_world']),
                        state['metadata'], ik, sim, scene, output)
        save(output/'chosen-grasp.json', {k:v for k,v in chosen.items() if k!='pose'})
        summary.update(grasp_source=chosen['source'], tested_postures=chosen['tested_postures'])
        if plan_only:
            summary.update(status='planned_only', plan_only=True)
            return summary

        def move(pose, q, *, contact=False, gripper=0, maximum=32):
            nonlocal state,scene
            reference_cloud=np.asarray(cloud).copy()
            reference_hand=np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3].copy()
            reference_panel=scene.panel
            result = edge(scene, sim, ik, state['metadata']['joint_positions'], q,
                          state['metadata']['gripper_qpos'], pose)
            if not result['accepted']:
                raise ValueError('Requested edge lacks observed robot clearance')
            spent = 0
            while spent<maximum:
                remaining = 600-state['control_steps']
                if remaining<=0:
                    raise ValueError('600-control-step budget exhausted')
                batch = min(8, maximum-spent, remaining)
                state = sim.action(pose, gripper=gripper, max_steps=batch,
                    nullspace_joint_target=q, nullspace_kp=10 if stage=='feedback_pull' else 80)
                spent += state['executed_steps']
                error = arrival_error(pose, state['metadata']['eef_pose_world_xyzw'])
                joint_error = float(np.max(np.abs(np.asarray(state['metadata']['joint_positions'])-q)))
                commands.append({'stage': stage, 'step': state['control_steps'], 'requested_pose': list(pose),
                    'measured_pose': state['metadata']['eef_pose_world_xyzw'], **error,
                    'joint_error_rad': joint_error, 'gripper_command': gripper})
                save(output/'command-log.json', commands)
                save(output/'progress.json', {'stage': stage, 'control_steps': state['control_steps'], **error})
                checked = scene.check(state['metadata']['robot_collision_volumes_world'], contact=contact)
                if (not checked['accepted'] and stage=='feedback_pull'
                        and all(c.get('suspected_moving_panel_overlap',False) for c in checked['collisions'])):
                    # A contact motion moves the door as well as the hand.
                    # Recheck synchronized depth before classifying an overlap
                    # with the previous door position as a current collision.
                    refresh_folder=output/f'clearance-refresh-{len(clearance_refreshes):02d}'
                    state=sim.observe(refresh_folder)
                    fresh_cameras,_=read_observation_v3(refresh_folder,state['metadata'])
                    displacement=np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3]-reference_hand
                    tracked=translated_surface(reference_cloud,fresh_cameras,
                        state['metadata']['robot_collision_volumes_world'],displacement)
                    refresh={'control_step':state['control_steps'],'previous_scene_check':checked,
                             'handle_tracking':tracked,'observation':str(refresh_folder.relative_to(ROOT))}
                    if tracked['accepted']:
                        shift=np.asarray(tracked['translation_world_m'])
                        panel=(reference_panel[0]+shift,reference_panel[1]) if reference_panel is not None else None
                        fresh_scene=ObservedRobotScene(fresh_cameras,state['metadata'],reference_cloud+shift,
                            config,panel=panel,allow_tool_contact=True,
                            panel_half_band_m=model.spec['measured_panel_surface_half_band_m'],
                            tool_panel_contact=choices['motion']=='close')
                        checked=fresh_scene.check(state['metadata']['robot_collision_volumes_world'],contact=contact)
                        refresh['synchronized_scene_check']=checked
                        scene=fresh_scene
                    clearance_refreshes.append(refresh)
                    save(output/'clearance-refreshes.json',clearance_refreshes)
                if not checked['accepted']:
                    save(output/'actual-body-collision.json', checked)
                    raise ValueError('Measured robot is blocked by an observed non-contact surface')
                actual_q=np.asarray(state['metadata']['joint_positions'])
                if not ik.self_valid(actual_q) or np.any(actual_q<ik.chain.limits[:,0]-.001) or np.any(actual_q>ik.chain.limits[:,1]+.001):
                    raise ValueError('Measured robot violates its own joint/self geometry bounds')
                if error['position_error_m']<.006 and error['orientation_error_rad']<.08 and (joint_error<.08 or stage=='feedback_pull'):
                    return
            # A contact-constrained hand need not attain the whole requested
            # probe. Measure actual handle response before judging its effect.
            if stage=='feedback_pull' and error['position_error_m']<.022 and error['orientation_error_rad']<.20:
                return
            raise ValueError(f'Motion did not arrive within its bounded control budget: {error}')

        stage = 'verified_approach'
        path = chosen['approach']
        for pose, q in zip(path['poses_world_xyzw'][1:], path['joint_postures'][1:]):
            contact = np.linalg.norm(np.asarray(pose)[:3]-chosen['pose_world_xyzw'][:3])<.025
            move(pose, q, contact=contact, maximum=32)
        state = sim.observe(output/'at-grasp')
        # This check must precede every closure command, including retries.
        arrival = require_arrival(chosen['pose_world_xyzw'], state['metadata']['eef_pose_world_xyzw'])
        closure_authorizations.append({'control_step': state['control_steps'], **arrival})
        save(output/'closure-authorizations.json', closure_authorizations)
        stage = 'physical_closure'
        state = sim.action(chosen['pose_world_xyzw'], gripper=1, max_steps=min(40,600-state['control_steps']),
            nullspace_joint_target=chosen['joints'], nullspace_kp=80, output_dir=output/'closed')
        require_arrival(chosen['pose_world_xyzw'], state['metadata']['eef_pose_world_xyzw'], position=.012, orientation=.12)
        cameras, _ = read_observation_v3(output/'closed', state['metadata'])
        measured = translated_surface(cloud, cameras, state['metadata']['robot_collision_volumes_world'], np.zeros(3))
        if not measured['accepted']:
            raise ValueError('Handle cannot be reacquired after physical closure')
        cloud = cloud+np.asarray(measured['translation_world_m'])
        closure_translation=np.asarray(measured['translation_world_m'])
        scene = ObservedRobotScene(cameras, state['metadata'], cloud, config,
            panel=(panel_origin+closure_translation,panel_normal),allow_tool_contact=True,
            panel_half_band_m=model.spec['measured_panel_surface_half_band_m'],
            tool_panel_contact=choices['motion']=='close')
        summary['closing_palm_panel_brace_radius_m']=scene.tool_panel_brace_radius
        centres = [np.median(cloud, axis=0)]
        initial_centre=centres[0].copy()
        pivot, axis = model.origin.copy(), model.axis.copy()
        progress = 0.
        stage = 'feedback_pull'
        for index in range(32):
            hand = np.asarray(state['metadata']['eef_pose_world_xyzw'])
            radial = centres[-1]-pivot
            radius = max(.10, np.linalg.norm(np.cross(axis, radial)))
            tangent = np.cross(axis, radial); tangent /= np.linalg.norm(tangent)
            distance = .010 if index==0 else .018
            angle = distance/radius
            target = hand.copy(); target[:3] += distance*tangent
            target[3:] = (Rotation.from_rotvec(angle*axis)*Rotation.from_quat(hand[3:])).as_quat()
            seed=np.asarray(state['metadata']['joint_positions'])
            trials=[];solution=None
            # Explore small wrist turns/tilts while the commanded contact
            # point follows the same observed tangent on the upright bar.
            # Try these only when every nominal local posture is blocked;
            # retain exactly the same robot, obstacle and edge checks.
            for option in contact_pose_variants(target,hand,axis):
                variant=option['pose_world_xyzw']
                alternatives=ik.solve_many(variant,seed,maximum=32)
                alternatives=[row for row in alternatives if np.max(np.abs(np.asarray(row['joints'])-seed))<.45]
                alternatives.sort(key=lambda row:-row['joint_reserve_rad']+.03*np.linalg.norm(np.asarray(row['joints'])-seed))
                loose=local_contact_ik(ik,variant,seed)
                if loose is not None:
                    alternatives.insert(0,loose)
                for row in alternatives:
                    result=scene.joints(sim,row['joints'],state['metadata']['gripper_qpos'],True)
                    path_check=(edge(scene,sim,ik,seed,row['joints'],state['metadata']['gripper_qpos'],row['pose_world_xyzw'])
                                if result['accepted'] else {'accepted':False})
                    trials.append({'backend':row.get('backend','curobo'),'joint_reserve_rad':row['joint_reserve_rad'],
                                   'rotation_axis':option['rotation_axis'],'offset_rad':option['offset_rad'],
                                   'check':result,'path_check':path_check})
                    if result['accepted'] and path_check['accepted']:
                        solution={**row,'contact_orientation_axis':option['rotation_axis'],
                                  'contact_orientation_offset_rad':option['offset_rad']};break
                if solution is not None:
                    break
            save(output/f'continuation-{index:02d}.json',{'candidates':trials,'selected':solution})
            if solution is None:
                raise ValueError('No robot-clear local continuation to the next short pull')
            before = np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3]
            move(solution['pose_world_xyzw'], solution['joints'], contact=True, maximum=24)
            folder = output/f'feedback-{index:02d}'; state=sim.observe(folder)
            cameras, _=read_observation_v3(folder,state['metadata'])
            displacement=np.asarray(state['metadata']['eef_pose_world_xyzw'])[:3]-before
            observed=translated_surface(cloud,cameras,state['metadata']['robot_collision_volumes_world'],displacement)
            if not observed['accepted']:
                raise ValueError('RGB-D handle tracking was lost during the local pull')
            actual=np.asarray(observed['translation_world_m'])
            following=float(actual@tangent)
            observed.update(control_steps=state['control_steps'],commanded_distance_m=distance,
                observed_tangent_motion_m=following,hand_translation_world_m=displacement.tolist())
            cloud=cloud+actual; centres.append(np.median(cloud,axis=0))
            fit=fit_upright_hinge(centres); observed['online_hinge_fit']=fit
            if fit['accepted']:
                estimate=np.asarray(fit['hinge_world_m']); update=estimate-pivot
                if np.linalg.norm(update[:2])<.15:
                    pivot += .35*update
            progress += max(0.,following)/radius
            observed['observed_progress_rad']=progress
            tracking.append(observed);save(output/'feedback.json',tracking)
            rotation=Rotation.from_rotvec(progress*axis).as_matrix()
            updated_origin=(panel_origin+closure_translation-initial_centre)@rotation.T+centres[-1]
            scene=ObservedRobotScene(cameras,state['metadata'],cloud,config,
                panel=(updated_origin,rotation@panel_normal),allow_tool_contact=True,
                panel_half_band_m=model.spec['measured_panel_surface_half_band_m'],
                tool_panel_contact=choices['motion']=='close')
            if following<.002:
                raise ValueError('Observed handle did not follow the short pull, or the mechanism reached its stop')
            if progress>=model.goal-.04:
                break
        summary.update(status='completed_feedback',observed_progress_rad=progress)
        sim.observe(output/'final')
    except Exception as error:
        summary.update(status='failed',failure_stage=stage,failure=str(error))
        (output/'error-traceback.txt').write_text(traceback.format_exc())
    finally:
        if ik is not None:
            ik.close()
        closed=sim.close()
        actions=read(output/'initial/actions.json') if (output/'initial/actions.json').exists() else []
        summary.update(control_steps=len(actions),official_task_success=any(a['evaluation_task_success'] for a in actions),
                       final_task_success=bool(actions and actions[-1]['evaluation_task_success']),
                       closure_commands_authorized=len(closure_authorizations))
        save(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tasks',nargs='+',default=['open_microwave','close_microwave'])
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    config=read(ROOT/'configs/articulated_libero.json')
    tasks=[t for t in config['tasks'] if t['label'] in args.tasks]
    ledger=ROOT/'results/articulated-api-v1/ledger.json';before=sha(ledger)
    original=read(ROOT/'results/articulated-physics-pilot/protocol.json')['sources_sha256']
    if any(sha(ROOT/p)!=h for p,h in original.items()):
        raise ValueError('Original frozen experiment sources changed')
    sources={**original,'src/vlm_tamp/handle_contact.py':sha(ROOT/'src/vlm_tamp/handle_contact.py'),
             'scripts/run_handle_contact.py':sha(__file__)}
    for p in sources:
        destination=output/'source-snapshot'/p;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,destination)
    save(output/'protocol.json',{'source_sha256':sources,'development':True,'initial_state':1,
        'no_held_out_success_claim':True,'simulator_geometry_used_for_policy':False,
        'new_api_calls':0,'api_ledger_sha256_before':before,'physical_contacts_enabled':True,
        'assisted_grasping':False,'maximum_control_steps':600,'arrival_required_before_closure':True,
        'pinch_width_veto':False,'measured_robot_obstacles_enforced':True,
        'preferred_posture_is_not_a_pull_completion_veto':True,
        'initial_grasp_minimum_joint_reserve_rad':.15,
        'pull_orientation_compliance_limit_rad':.25,
        'blocked_pull_yaw_alternatives_rad':[0.,-.15,.15,-.30,.30],
        'blocked_pull_tool_axis_tilt_alternatives_rad':[-.10,.10,-.20,.20],
        'current_depth_recheck_before_rejecting_actual_contact_motion':True,
        'depth_recheck_requires_all_rejections_associated_with_selected_moving_panel':True,
        'postcapture_closing_palm_panel_brace':'known hand reach around tracked handle, within measured panel slab',
        'closing_panel_contact_band_reestimated_from_each_observation':True,
        'postcapture_gripper_surface_margin_m':0.,'arm_surface_margin_m':.001,
        'postcapture_local_tool_handle_contact_allowed':True,
        'finger_local_panel_brace_radius_m':.065,
        'palm_handle_contact_radius_m':.028,
        'direction_checked_against_adjacent_stationary_rgbd':True,
        'door_hull_enforced':False,'motion_strategy':'short tangent pulls and RGB-D co-motion feedback'})
    save(output/'manifest.json',{'planned_episodes':[{'task':t['label'],'name':t['name'],'state':1} for t in tasks],
         'all_planned_failures_in_denominator':True,'plan_only':args.plan_only})
    rows=[]
    for task in tasks:
        rows.append(run(task,output/f'{task["label"]}-state001',config,args.plan_only))
        save(output/'summary.json',{'planned':len(tasks),'completed':len(rows),'episodes':rows,
             'physical_successes':sum(r['official_task_success'] for r in rows),'new_api_calls':0,
             'api_ledger_unchanged':sha(ledger)==before})
        print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':
    main()
