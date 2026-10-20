from pathlib import Path
import numpy as np
from vlm_tamp.shared_feedback_directional import DirectionalFeedback
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.geometry import pose_matrix,matrix_pose
ROOT=Path(__file__).resolve().parents[1]


def setup(mode='press_drag'):
    chain=RobotChain(read_json(ROOT/'results/family-transfer-frozen-20261016/open_bottom_drawer-state005/robot-chain.json'))
    q=chain.reference_joints;pose=chain.forward(q)
    feedback=DirectionalFeedback(chain,np.repeat(q[None],40,axis=0),pose,variant='reactive',
        normal=[0,0,1],tangent=[0,1,0],phase='manipulate',mode=mode)
    return q,pose,feedback


def test_preload_is_retained_when_reference_starts_at_measured_contact():
    q,pose,f=setup()
    for i in range(40):_,info=f.command(i,q,pose)
    assert abs(info['offset_world_m'][2]+.002)<1e-8


def test_tracking_lag_cannot_cancel_planned_progress():
    q,pose,f=setup('pinch');measured=pose_matrix(pose);measured[:3,3]+=[.012,-.04,.015]
    for i in range(40):_,info=f.command(i,q,matrix_pose(measured))
    assert abs(info['tangential_offset_m'])<1e-12
    assert np.linalg.norm(np.array(info['offset_world_m'])[[0,2]])>.01


def test_large_deflection_remains_bounded():
    q,pose,f=setup();measured=pose_matrix(pose);measured[2,3]+=.2
    for i in range(40):_,info=f.command(i,q,matrix_pose(measured))
    assert np.linalg.norm(info['offset_world_m'])<=.035+1e-12


def test_next_plan_uses_object_motion_not_requested_motion():
    from vlm_tamp.shared_mechanism_pipeline import ObservedProgress
    progress=ObservedProgress([0,1,0])
    for _ in range(6):
        progress.update({'accepted':True,'translation_world_m':[0,.020,0]},[0,.025,0])
    assert abs(progress.progress-.12)<1e-12
    assert .15-progress.progress>.025


def test_missing_visual_update_never_stops_execution_and_can_recover():
    from vlm_tamp.shared_mechanism_pipeline import ObservedProgress
    progress=ObservedProgress([0,1,0])
    progress.update({'accepted':True,'translation_world_m':[0,.02,0]},[0,.025,0])
    progress.update({'accepted':False},[0,.025,0])
    assert abs(progress.progress-.045)<1e-12
    # New visual displacement covers the interval since the last valid image.
    progress.update({'accepted':True,'translation_world_m':[0,.04,0]},[0,.025,0])
    assert abs(progress.progress-.06)<1e-12
    assert not any(r['execution_veto'] for r in progress.records)
