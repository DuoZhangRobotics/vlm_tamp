"""Batched delivery, full-trajectory semantics, and fresh-request timing policy."""
from types import SimpleNamespace
import numpy as np
import pytest
from vlm_tamp.profiled_sim_client import execute_batch
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import fresh_request_arguments
from vlm_tamp.rekep_libero.api import request_arguments,reservation_usd


def test_batch_executor_sends_identical_references_in_one_request():
    refs=np.arange(28,dtype=float).reshape(4,7)/100;calls=[];events=[]
    class Client:
        def joint_trajectory(self,values,gripper):
            calls.append((values.copy(),gripper))
            rows=[{'reference_joints':q.tolist(),'measured_joints':q.tolist(),'execution_veto_checks':0} for q in values]
            return {'control_steps':64,'metadata':{'joint_positions':values[-1].tolist()},
                'trajectory':{'events':rows,'all_commands_issued':True,'issued_commands':len(rows)}}
    state,done=execute_batch(Client(),{'control_steps':60},refs,gripper=-1,event_sink=events.append)
    assert len(calls)==1 and np.array_equal(calls[0][0],refs) and calls[0][1]==-1
    assert done['issued_commands']==4 and len(events)==4 and state['control_steps']==64
    with pytest.raises(ValueError):execute_batch(Client(),{'control_steps':598},refs)
    assert len(calls)==1


def test_batch_stops_at_first_success_and_does_not_issue_later_references(monkeypatch):
    from vlm_tamp.profiled_sim_worker import ProfiledWorker
    from vlm_tamp.coupled_worker import CoupledWorker
    worker=ProfiledWorker()
    worker.ik_solver=SimpleNamespace(limits=np.tile([-3.,3.],(7,1)))
    worker.control_steps=60;worker.action_log=[]
    worker.env=SimpleNamespace(check_success=lambda:bool(worker.action_log and worker.action_log[-1]['evaluation_task_success']))
    worker._write_action_log=lambda:None
    worker._response_state=lambda:{'control_steps':worker.control_steps}
    issued=[]
    def step(self,q,gripper,max_steps):
        issued.append(q.copy());self.control_steps+=1
        self.action_log.append({'evaluation_task_success':len(issued)==2})
        return {'control_steps':self.control_steps,'metadata':{'joint_positions':q.tolist()}}
    monkeypatch.setattr(CoupledWorker,'joint_action',step)
    refs=np.arange(35).reshape(5,7)/100
    result=worker.joint_trajectory(refs,stop_on_success=True)['trajectory']
    assert len(issued)==2 and result['issued_commands']==2
    assert not result['all_commands_issued']
    assert result['terminated_on_simulator_success']
    assert result['execution_veto_checks']==0
    again=worker.joint_trajectory(refs,stop_on_success=True)['trajectory']
    assert again['issued_commands']==0 and len(issued)==2


def test_fresh_request_changes_only_non_prompt_metadata(tmp_path):
    image=tmp_path/'rgb.png';image.write_bytes(b'owned-test-image')
    original=request_arguments(image,'same task and images',12000)
    fresh=fresh_request_arguments(image,'same task and images','test-run-1')
    assert fresh.pop('metadata')=={'benchmark_id':'test-run-1'}
    assert fresh==original and reservation_usd(fresh)==reservation_usd(original)


def test_nested_timing_preserves_parent_and_task_context(tmp_path):
    t=Timeline();t.context={'task':'test','state':1}
    with t.span('outer') as outer:
        with t.span('inner') as inner:pass
    assert inner['parent']==outer['id'] and inner['task']=='test'
    assert 0<=inner['seconds']<=outer['seconds']
    t.save(tmp_path/'timing.json')


@pytest.mark.parametrize('spacing',[.002,.003,.0075])
def test_voxel_optimization_retains_exact_points_order_and_duplicates(spacing):
    from vlm_tamp.exact_voxels import voxel_points,baseline_voxel_points
    rng=np.random.default_rng(29);points=rng.uniform(-1,2,(20000,3))
    points=np.vstack([points,points[::7],np.zeros((5,3)),[[np.nan,0,0],[0,np.inf,0]],
        [[-spacing,0,spacing],[0,spacing,-spacing]]])
    assert np.array_equal(voxel_points(points,spacing),baseline_voxel_points(points,spacing))
    assert np.array_equal(voxel_points([],spacing),baseline_voxel_points([],spacing))


def test_voxel_large_key_range_falls_back_without_overflow():
    from vlm_tamp.exact_voxels import voxel_points,baseline_voxel_points
    points=np.array([[-1e12,-1e12,-1e12],[1e12,1e12,1e12],[0,0,0],[0,0,0]])
    assert np.array_equal(voxel_points(points,.001),baseline_voxel_points(points,.001))


def test_worker_batches_all_commands_without_error_or_success_veto(monkeypatch):
    from vlm_tamp.profiled_sim_worker import ProfiledWorker
    from vlm_tamp.coupled_worker import CoupledWorker
    worker=ProfiledWorker();worker.env=object();worker.ik_solver=SimpleNamespace(limits=np.tile([-3.,3.],(7,1)))
    commands=[];flushes=[]
    def one(self,target_joints,gripper=1,max_steps=1,output_dir=None):
        commands.append((np.asarray(target_joints).copy(),gripper));self.control_steps+=1
        return {'metadata':{'joint_positions':(np.asarray(target_joints)+.4).tolist()},'control_steps':self.control_steps,
                'evaluation':{'task_success':True}}
    monkeypatch.setattr(CoupledWorker,'joint_action',one)
    monkeypatch.setattr(worker,'_write_action_log',lambda:flushes.append(True))
    monkeypatch.setattr(worker,'_response_state',lambda:{'control_steps':worker.control_steps,'metadata':{}})
    refs=np.zeros((5,7));refs[:,0]=np.arange(5)*.01
    result=worker.joint_trajectory(refs,gripper=1)
    assert np.array_equal(np.array([c[0] for c in commands]),refs)
    assert result['trajectory']['issued_commands']==5 and result['trajectory']['execution_veto_checks']==0
    assert result['trajectory']['maximum_joint_reference_error_rad']==pytest.approx(.4)
    assert len(flushes)==1
    commands.clear();bad=refs.copy();bad[-1,0]=10
    with pytest.raises(ValueError):worker.joint_trajectory(bad)
    assert commands==[]


def test_instrumentation_restores_engine_and_reuses_only_identical_calibration(monkeypatch):
    from vlm_tamp.profiled_pipeline import instrument_engine
    from vlm_tamp import mcvamp_articulated as engine,mcvamp_mechanism as native
    before_execute=engine.execute;before_scene=engine.SurfaceScene;calls=[]
    def calibration(record):calls.append(record);return ('frames',record)
    monkeypatch.setattr(native,'calibrated_frames',calibration)
    with pytest.raises(RuntimeError):
        with instrument_engine(Timeline(),fast=True):
            a=native.calibrated_frames({'robot':[1,2]})
            b=native.calibrated_frames({'robot':[1,2]})
            native.calibrated_frames({'robot':[1,3]})
            assert a is b and len(calls)==2
            raise RuntimeError('test unwind')
    assert engine.execute is before_execute and engine.SurfaceScene is before_scene
    assert native.calibrated_frames is calibration
