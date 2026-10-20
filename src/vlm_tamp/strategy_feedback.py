"""Sensor/solver-only feedback. Never serialize simulator evaluation metadata."""
from collections import Counter
from pathlib import Path
import numpy as np

from .coupled_storage import read_json,write_json
from .rekep_libero.mechanism_geometry import project
from .rekep_libero.geometry import pose_matrix


DIAGNOSTIC_KEYS=('self_rejections','static_rejections','moving_rejections','bounds_rejections',
                 'timed_out','solve_seconds','iterations')
RESULT_KEYS={'status','phase','reason','manipulation_executed','completed_segments','object_progress_delta_m',
             'estimated_goal_reached','tracking_accepted','tracking_coverage','tracking_residual_m',
             'observed_target_translation_world_m','measured_hand_translation_world_m',
             'rejection_record_counts','solver_diagnostics','interpretation','last_proposed_pose_world_xyzw',
             'selected_strategy','contact_retention_inferred_from_motion_only','last_collision_prediction',
             'failure_loci','requested_step_mm','attempted_step_m','executed_plan_distances_m',
             'requested_posture_rank','requested_retreat_mm','release_executed','retreat_executed'}


def planner_feedback(folder):
    folder=Path(folder);reasons=Counter();checks=Counter();last_pose=None;last_candidate=None;collision=None
    def visit(value):
        nonlocal last_pose,last_candidate,collision
        if isinstance(value,list):
            for row in value:visit(row)
        elif isinstance(value,dict):
            if isinstance(value.get('reason'),str):reasons[value['reason']]+=1
            if type(value.get('candidate')) is int:last_candidate=value['candidate']
            if value.get('accepted') is False and type(value.get('last_rejected_part')) in (int,float):
                part=int(value['last_rejected_part']);surface=int(value.get('last_rejected_surface',-1))
                collision={'robot_part_class':'arm' if 1<=part<37 else 'gripper' if part>=1000 else 'unresolved',
                    'robot_part_index':part,'scene_region':{0:'observed_static_surface',1:'predicted_moving_surface',
                        2:'selected_contact_region'}.get(surface,'unresolved'),
                    'source':'planner geometry prediction, not a measured physical collision'}
            for key in DIAGNOSTIC_KEYS:
                if type(value.get(key)) in (int,float):checks[key]+=value[key]
            for key in ('pose','pose_world_xyzw','tcp_pose_world_xyzw'):
                if isinstance(value.get(key),list) and len(value[key])==7:last_pose=value[key]
            for key in ('attempts','rejected_candidates','diagnostic','planning','initial','native_planning','projection'):
                if key in value:visit(value[key])
    files=[p for name in ('contact-mode-admission.json','grasp-admission.json','planning.json')
           if (p:=folder/name).exists()]
    for path in files:visit(read_json(path))
    menu_path=folder/'contact-menu.json'
    if last_pose is None and last_candidate is not None and menu_path.exists():
        menu=read_json(menu_path)['candidates']
        if 0<=last_candidate<len(menu):last_pose=menu[last_candidate]['pose_world_xyzw']
    # The generic pinch failure is only an aggregate message. Detailed records
    # are retained separately; counts are diagnostic events, not physical tries.
    return {'rejection_record_counts':dict(reasons),'solver_diagnostics':dict(checks),
        'interpretation':'These are predictions/search outcomes in the observed scene, not proof of physical impossibility.',
        'last_proposed_pose_world_xyzw':last_pose,'last_collision_prediction':collision}


def feedback_packet(instruction,metadata,spec,progress,geometry,history,*,round_index,maximum_rounds,goal_reached,active_contact):
    """Explicit field selection prevents native success/poses/contacts leaking."""
    recent=progress.records[-1] if progress.records else {}
    return {'version':'strategy-feedback-v1','instruction':instruction,
        'observed_state':{'eef_pose_world_xyzw':list(metadata['eef_pose_world_xyzw']),
            'gripper_finger_positions_m':list(metadata['gripper_qpos']),
            'active_contact_strategy':active_contact,
            'contact_retention_verified':False,
            'estimated_goal_reached':bool(goal_reached),
            'estimated_object_progress_m':float(progress.observed_progress),
            'working_progress_m':float(progress.progress),
            'last_progress_source':recent.get('source','not observed yet')},
        'mechanism_estimate':{k:spec[k] for k in ('joint_family','motion','joint_axis_world',
            'panel_normal_world','source_anchor_world_m','target_progress','progress_units')},
        'geometry_measurements':{k:geometry[k] for k in ('measured_lower_clearance_m',
            'configured_palm_half_span_m','observed_extent_m','part_frame_world') if k in geometry},
        'capability_limits':['fixed contact templates','upright-panel geometry prior',
            'no task-specific strategy switch','unknown space is unmodeled','no measured force control'],
        'budget':{'decision_number':round_index+1,'decisions_remaining_including_this':maximum_rounds-round_index,
            'control_steps_remaining':max(0,600-int(metadata['control_steps']))},
        'history':[{'decision':{k:row['decision'][k] for k in ('version','operation','contact_mode',
            'approach_tilt_deg','motion_extent','evidence','expected_observation','step_mm','posture_rank','retreat_mm') if k in row['decision']},
            'result':{k:row['result'][k] for k in RESULT_KEYS if k in row['result']}}
            for row in history]}


def render_feedback(previous,current,cloud,spec,metadata,output,*,pose=None):
    """Scientific overlays of measured RGB-D and predicted poses; no synthetic scene."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.24,10.24),dpi=100)
    for row,(views,label) in enumerate(((previous,'Earlier observation'),(current,'Current observation'))):
        for column,name in enumerate(('agentview','frontview')):
            camera=views[name];ax=axes[row,column];ax.imshow(camera['rgb']);ax.set_title(f'{label}: {name}',fontsize=10)
            if row==1:
                points=np.asarray(cloud)[::max(1,len(cloud)//300)]
                pixels=project(points,camera);valid=np.isfinite(pixels).all(1)
                ax.scatter(pixels[valid,0],pixels[valid,1],s=1,c='cyan',alpha=.5)
                hand=project([metadata['eef_pose_world_xyzw'][:3]],camera)[0]
                if np.isfinite(hand).all():ax.plot(*hand,'o',color='royalblue',markersize=5)
                origin=np.median(cloud,axis=0);end=origin+.05*np.asarray(spec['joint_axis_world'])
                arrow=project([origin,end],camera)
                if np.isfinite(arrow).all():ax.annotate('',xy=arrow[1],xytext=arrow[0],arrowprops={'color':'yellow','width':1.5})
                if pose is not None:
                    transform=pose_matrix(pose)
                    points=np.vstack([transform[:3,3],transform[:3,3]+.035*transform[:3,2]])
                    line=project(points,camera)
                    if np.isfinite(line).all():ax.plot(line[:,0],line[:,1],'-x',color='magenta')
            height,width=camera['rgb'].shape[:2];ax.set_xlim(0,width);ax.set_ylim(height,0);ax.axis('off')
    fig.suptitle('Cyan: observed target cloud | Blue: measured hand | Yellow: estimated motion | Magenta: last proposed TCP',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(output);plt.close(fig)


def save_packet(packet,folder):
    write_json(Path(folder)/'feedback.json',packet)
