"""Send a complete timed joint reference, with no execution-time planning veto."""
import time

import numpy as np


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


def execute(sim,state,references,*,gripper=1,maximum_control_steps=600,event_sink=None):
    """Advance by reference time, regardless of contacts, lag or task-success flags.

    Feedback is retained inside the official joint-position controller. The
    simulator's finite command/joint-limit handling remains in the driver.
    No scene, constraint, tracking-progress or collision checker is accepted.
    """
    references=np.asarray(references,float)
    if references.ndim!=2 or references.shape[1]!=7 or not np.isfinite(references).all():
        raise ValueError('Finite seven-joint references required')
    remaining=maximum_control_steps-int(state['control_steps'])
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
