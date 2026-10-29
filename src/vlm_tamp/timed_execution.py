"""Send a complete timed joint reference, with no execution-time planning veto."""
import time

import numpy as np
from .execution_limits import control_limit


def gripper_arm_reference(measured,limits,*,planning_reserve_rad=0.,control_hz=20.,maximum_joint_speed=1.):
    """Hold the arm, or command one inward servo increment after limit overshoot.

    This changes a reference, never the measured/simulator configuration. The
    existing planner inset is supplied by the caller; no new margin is added.
    """
    q=np.asarray(measured,float);bounds=np.asarray(limits,float)
    if q.shape!=(7,) or bounds.shape!=(7,2) or not np.isfinite(q).all():
        raise ValueError('Finite measured joints and configured limits required')
    admitted=bounds+np.array([planning_reserve_rad,-planning_reserve_rad])
    below=q<admitted[:,0];above=q>admitted[:,1];affected=below|above
    target=q.copy();increment=maximum_joint_speed/control_hz
    target[below]+=increment;target[above]-=increment
    # The servo limits each actual increment even if measured overshoot is
    # larger than one tick. Always request a legal inward target.
    target=np.clip(target,bounds[:,0],bounds[:,1])
    return target,{'performed':bool(affected.any()),'method':'one inward joint reference increment; physical controller motion',
        'maximum_reference_speed_rad_s':maximum_joint_speed,'control_hz':control_hz,
        'controller_limits_actual_reference_increment':True,
        'joints':[{'joint':int(j+1),'measured_rad':float(q[j]),'reference_rad':float(target[j]),
            'admitted_range_rad':admitted[j].tolist()} for j in np.flatnonzero(affected)]}


def schedule(path,*,maximum_joint_speed=.45,control_hz=20.,minimum_steps=1,endpoint_hold_steps=15):
    path=np.asarray(path,float)
    if (path.ndim!=2 or path.shape[1]!=7 or len(path)<2 or not np.isfinite(path).all()
            or maximum_joint_speed<=0 or control_hz<=0 or minimum_steps<1 or endpoint_hold_steps<0):
        raise ValueError('A finite seven-joint path and positive timing settings are required')
    # Timing follows path length, not the number of dense planner samples.
    durations=np.max(np.abs(np.diff(path,axis=0)),axis=1)/maximum_joint_speed
    times=np.r_[0.,np.cumsum(durations)]
    keep=np.r_[True,np.diff(times)>1e-12]
    path,times=path[keep],times[keep]
    total=times[-1]
    count=max(minimum_steps,int(np.ceil(total*control_hz)))
    ticks=np.linspace(0.,total,count+1)[1:]
    references=np.column_stack([np.interp(ticks,times,path[:,j]) for j in range(7)])
    if endpoint_hold_steps:references=np.vstack([references,np.repeat(path[-1][None],endpoint_hold_steps,axis=0)])
    return references


def schedule_contact(path,chain,contact_local,*,contact_speed_m_s=.03,maximum_joint_speed=1.,control_hz=20.,endpoint_hold_steps=0):
    """Time a complete path by both contact-landmark and joint travel.

    The limit describes references, not a physical velocity/force guarantee.
    FK subdivision prices curved contact motion along joint interpolation.
    """
    from .rekep_libero.geometry import pose_matrix
    path=np.asarray(path,float);contact=np.asarray(contact_local,float)
    if (path.ndim!=2 or path.shape[1]!=7 or len(path)<2 or not np.isfinite(path).all()
            or contact.shape!=(3,) or not np.isfinite(contact).all()
            or not np.isfinite(contact_speed_m_s) or not 0<contact_speed_m_s
            or not np.isfinite(maximum_joint_speed) or not 0<maximum_joint_speed<=1.
            or control_hz<=0 or endpoint_hold_steps<0):
        raise ValueError('Finite path/contact, positive contact speed and joint speed (0,1] rad/s required')
    dense=[path[0]]
    for a,b in zip(path[:-1],path[1:]):
        count=max(1,int(np.ceil(np.max(np.abs(b-a))/.01)))
        dense.extend(a+(b-a)*t for t in np.linspace(0,1,count+1)[1:])
    dense=np.asarray(dense)
    transforms=[pose_matrix(chain.forward(q)) for q in dense]
    points=np.asarray([t[:3,3]+t[:3,:3]@contact for t in transforms])
    durations=np.maximum(np.max(np.abs(np.diff(dense,axis=0)),axis=1)/maximum_joint_speed,
                         np.linalg.norm(np.diff(points,axis=0),axis=1)/contact_speed_m_s)
    times=np.r_[0.,np.cumsum(durations)];keep=np.r_[True,np.diff(times)>1e-12]
    dense,times=dense[keep],times[keep]
    ticks=np.linspace(0,times[-1],max(1,int(np.ceil(times[-1]*control_hz)))+1)[1:]
    refs=np.column_stack([np.interp(ticks,times,dense[:,j]) for j in range(7)])
    if endpoint_hold_steps:refs=np.vstack([refs,np.repeat(dense[-1][None],endpoint_hold_steps,axis=0)])
    return refs


def execute(sim,state,references,*,gripper=1,maximum_control_steps=None,event_sink=None):
    """Advance by reference time, regardless of contacts, lag or task-success flags.

    Feedback is retained inside the official joint-position controller. The
    simulator's finite command/joint-limit handling remains in the driver.
    No scene, constraint, tracking-progress or collision checker is accepted.
    """
    references=np.asarray(references,float)
    if references.ndim!=2 or references.shape[1]!=7 or not np.isfinite(references).all():
        raise ValueError('Finite seven-joint references required')
    limit=control_limit(state) if maximum_control_steps is None else min(control_limit(state),maximum_control_steps)
    remaining=limit-int(state['control_steps'])
    if len(references)>remaining:
        raise ValueError('Complete trajectory exceeds the remaining episode control-step budget')
    before=time.perf_counter();events=[]
    for index,q in enumerate(references):
        state=sim.joint_action(q,gripper=gripper,max_steps=1)
        measured=np.asarray(state['metadata']['joint_positions'],float)
        event={'control_step':state['control_steps'],'reference_index':index,'reference_joints':q.tolist(),
            'measured_joints':measured.tolist(),'joint_reference_error_rad':float(np.max(np.abs(measured-q))),
            'command_issued':True,'execution_veto_checks':0}
        events.append(event)
        if event_sink:event_sink(event)
    return state,{'all_commands_issued':True,'planned_commands':len(references),'issued_commands':len(events),
        'wall_seconds':time.perf_counter()-before,'execution_veto_checks':0,
        'maximum_joint_reference_error_rad':max((e['joint_reference_error_rad'] for e in events),default=0.),
        'events':events}
