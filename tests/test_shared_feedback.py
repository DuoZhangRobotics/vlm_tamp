from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.shared_feedback import FeedbackConfig,ReferenceFeedback,corrected_joints
from vlm_tamp.shared_contact import contact_frame,contact_menu
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.geometry import pose_matrix,matrix_pose

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/family-transfer-frozen-20261016/open_bottom_drawer-state005'


def test_joint_correction_reduces_cartesian_error_and_respects_robot_limits():
    chain=RobotChain(read_json(SOURCE/'robot-chain.json'));q=chain.reference_joints
    target=pose_matrix(chain.forward(q));target[2,3]+=.008
    corrected=corrected_joints(chain,q,target,FeedbackConfig())
    assert np.linalg.norm(pose_matrix(chain.forward(corrected))[:3,3]-target[:3,3])<.003
    assert np.max(abs(corrected-q))<=.12+1e-12
    assert np.all(corrected>chain.limits[:,0]) and np.all(corrected<chain.limits[:,1])


def test_normal_accommodation_does_not_cancel_tangential_progress():
    chain=RobotChain(read_json(SOURCE/'robot-chain.json'));q=chain.reference_joints;refs=np.repeat(q[None],25,axis=0)
    measured=pose_matrix(chain.forward(q));controller=ReferenceFeedback(chain,refs,matrix_pose(measured),
        variant='reactive',normal=[0,0,1],phase='contact')
    measured[:3,3]+=[.02,.01,.03]
    for i in range(25):command,info=controller.command(i,q,matrix_pose(measured))
    offset=np.array(info['offset_world_m'])
    assert .020<offset[2]<.035 and np.allclose(offset[:2],0)


def test_part_frame_rotates_with_scene_without_task_name():
    cloud=np.load(SOURCE/'grasp-cloud.npy');spec=read_json(SOURCE/'task-spec.json')
    f=contact_frame(cloud,spec);r=Rotation.from_euler('z',73,degrees=True).as_matrix();translation=np.array([.3,-.4,.1])
    moved={**spec,'joint_axis_world':(r@spec['joint_axis_world']).tolist(),'panel_normal_world':(r@spec['panel_normal_world']).tolist()}
    other=contact_frame(cloud@r.T+translation,moved)
    assert np.allclose(other['basis'],r@f['basis'],atol=1e-6)
    assert np.allclose(other['extent'],f['extent'])


def test_contact_menu_prioritizes_drag_when_gripper_has_insufficient_clearance():
    cloud=np.load(SOURCE/'grasp-cloud.npy');spec=read_json(SOURCE/'task-spec.json');meta=read_json(SOURCE/'opened/metadata.json')
    frame=contact_frame(cloud,spec)
    tabletop=frame['center'][None]+np.array([-.05,-.03,-.01,.01,.03,.05])[:,None]*frame['bar']-.04*frame['across']
    menu,evidence=contact_menu(cloud,spec,meta,tabletop)
    assert evidence['tight_clearance'] and menu[0]['mode']=='press_drag'
    menu,evidence=contact_menu(cloud,spec,meta,np.empty((0,3)))
    assert not evidence['tight_clearance'] and menu[0]['mode']=='pinch'


def test_feedback_worker_issues_every_sample_despite_tracking_error_and_success(monkeypatch):
    from vlm_tamp.shared_feedback_worker import FeedbackWorker
    from vlm_tamp.coupled_worker import CoupledWorker
    worker=FeedbackWorker();record=read_json(SOURCE/'robot-chain.json');chain=RobotChain(record)
    worker.env=object();worker.ik_solver=SimpleNamespace(limits=chain.limits)
    q=chain.reference_joints;refs=np.repeat(q[None],7,axis=0);commands=[]
    monkeypatch.setattr(worker,'robot_chain',lambda:record)
    monkeypatch.setattr(worker,'_eef_pose',lambda:chain.forward(q))
    monkeypatch.setattr(worker,'_write_action_log',lambda:None)
    monkeypatch.setattr(worker,'_response_state',lambda:{'control_steps':worker.control_steps})
    def physical(self,target_joints,**kwargs):
        commands.append(np.array(target_joints));self.control_steps+=1
        return {'control_steps':self.control_steps,'metadata':{'joint_positions':(q+.3).tolist()},'evaluation':{'task_success':True}}
    monkeypatch.setattr(CoupledWorker,'joint_action',physical)
    result=worker.feedback_trajectory(refs,1,{'variant':'reactive','normal_world':[0,0,1],'phase':'manipulate'})
    assert len(commands)==7 and result['trajectory']['issued_commands']==7
    assert result['trajectory']['execution_veto_checks']==0
    assert [e['reference_index'] for e in result['trajectory']['events']]==list(range(7))
