"""Execute a freshly admitted full manipulation sequence after physical closure.

This isolated composition leaves the concurrently frozen engines unchanged.
It uses no saved trajectory from another episode and no simulator object state.
"""
import hashlib
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
import run_success_first_family_regression as experiment
from vlm_tamp.coupled_storage import write_json
from vlm_tamp.timed_execution import schedule
from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points

original_mechanism_select=experiment.mechanisms.select_grasp
original_bowl_select=experiment.bowl.select_legacy
original_execute=experiment.execute_to_success


def retain(sim,path,*,release,source):
    path=np.asarray(path,float)
    if path.ndim!=2 or path.shape[1]!=7 or len(path)<2 or not np.isfinite(path).all():
        raise ValueError('Prepared manipulation must be a finite complete joint path')
    folder=sim.folder/'prepared-sequence';folder.mkdir(exist_ok=True)
    np.save(folder/'manipulation.npy',path)
    sim.prepared_sequence={'path':path,'release':release,'source':source}
    write_json(folder/'plan.json',{'source':source,'planned_before_physical_approach':True,
        'expected_closed_grasp_joints':path[0].tolist(),'release_at_end':release,
        'simulator_object_truth_used':False,'assisted_attachment':False,
        'planning_collision_checks_retained':True,'post_closure_replanning':False})


def mechanism_select(*args,**kwargs):
    chosen,approach=original_mechanism_select(*args,**kwargs)
    spec,sim,folder=args[2],args[7],Path(args[8])
    if spec['joint_family']=='revolute' and spec.get('axis_prior')!='hinge_edge':
        # Retain the engine's existing observed direction probe. A full plan
        # for an unconfirmed rotation sign cannot replace that information.
        write_json(sim.folder/'prepared-sequence-skipped.json',{
            'reason':'existing rotary-direction probe must run before committing the full motion',
            'direction_probe_preserved':True})
        return chosen,approach
    if chosen.get('horizon_support')=='full_task':
        path=folder/'full-task-probe'/f'candidate-{chosen["tested_ik_postures"]:03d}'/'probe/path.npy'
        retain(sim,np.load(path),release=False,source='current candidate full-task MC-VAMP probe')
    return chosen,approach


def bowl_select(*args,**kwargs):
    chosen=original_bowl_select(*args,**kwargs)
    spec,sim=args[2],args[6]
    if spec['task_type']=='pick_place':
        ik=args[4];solutions=chosen['task_preflight']['solutions']
        local=transform_points([spec['object_support_point_initial_world_m']],
            np.linalg.inv(pose_matrix(chosen['fk_pose_world_xyzw'])))[0]
        positions=[transform_points([local],pose_matrix(ik.chain.forward(s['joints'])))[0] for s in solutions]
        distance=np.linalg.norm(np.asarray(positions)-spec['goal']['target_world_m'],axis=1)
        goal=int(np.argmin(distance))
        if distance[goal]>.001:raise ValueError('Prepared placement path lacks its declared goal')
        # The old admission path also checked a hypothetical retreat. End at
        # placement and release; the user explicitly requested no retreat.
        path=np.vstack([chosen['solution_joints'],[s['joints'] for s in solutions[:goal+1]]])
        retain(sim,path,release=True,source='current collision-checked preflight through placement; retreat excluded')
        write_json(sim.folder/'prepared-sequence/placement-boundary.json',{
            'goal_solution_index':goal,'goal_position_error_m':float(distance[goal]),
            'unexecuted_retreat_states':len(solutions)-goal-1})
    return chosen


def execute_prepared(sim,state,references,*,gripper=1,event_sink=None,**kwargs):
    stage=sim.segments
    state,result=original_execute(sim,state,references,gripper=gripper,event_sink=event_sink,**kwargs)
    prepared=getattr(sim,'prepared_sequence',None)
    if stage!=1 or prepared is None:return state,result
    sim.prepared_sequence=None
    path=prepared['path'];folder=sim.folder/'prepared-sequence'
    difference=float(np.max(abs(np.asarray(state['metadata']['joint_positions'])-path[0])))
    hold=6;reserve=41 if prepared['release'] else 0
    available=600-int(state['control_steps'])-reserve
    refs=schedule(path,maximum_joint_speed=.35,endpoint_hold_steps=hold)
    speed=.35
    if len(refs)>available:
        if available<=hold:raise ValueError('Prepared sequence does not fit the remaining episode budget')
        speed*=((len(refs)-hold)/(available-hold))*1.001
        refs=schedule(path,maximum_joint_speed=speed,endpoint_hold_steps=hold)
    write_json(folder/'dispatch.json',{'maximum_start_joint_difference_rad':difference,
        'difference_is_logged_not_an_execution_veto':True,'reference_steps':len(refs),
        'speed_rad_s':speed,'release_steps_reserved':reserve,'started_at_control_step':state['control_steps']})
    began=time.perf_counter()
    try:
        state,_=original_execute(sim,state,refs,gripper=1,event_sink=event_sink)
        if prepared['release']:
            release=np.repeat(path[-1][None],41,axis=0)
            state,_=original_execute(sim,state,release,gripper=-1,event_sink=event_sink)
    except experiment.NativeSuccess:
        write_json(folder/'outcome.json',{'native_success':True,'execution_seconds':time.perf_counter()-began})
        raise
    sim.observe(sim.folder/'final')
    write_json(folder/'outcome.json',{'native_success':False,'execution_seconds':time.perf_counter()-began,
        'complete_prepared_sequence_issued':True})
    raise ValueError('Prepared full sequence completed without native success')


if __name__=='__main__':
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    experiment.mechanisms.select_grasp=mechanism_select
    experiment.bowl.select_legacy=bowl_select
    experiment.execute_to_success=execute_prepared
    experiment.main()
    output=(ROOT/sys.argv[sys.argv.index('--output')+1]).resolve()
    snapshot=output/'source-snapshot/scripts'/Path(__file__).name
    shutil.copyfile(__file__,snapshot)
    write_json(output/'prepared-sequence-adapter.json',{'source_sha256':source_hash,
        'source_snapshot':str(snapshot),'mode':'execute freshly admitted sequence; no post-closure replan',
        'core_engine_files_modified':False})
