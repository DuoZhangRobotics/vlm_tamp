import copy
import numpy as np
from replay_astra_recovery_choices import compare_state,signature

def state():
    return {'observed_state':{'eef_pose_world_xyzw':[0,0,0,0,0,0,1],
        'gripper_finger_positions_m':[.02,-.02],'estimated_object_progress_m':.1,'working_progress_m':.1},
        'mechanism_estimate':{'joint_axis_world':[0,1,0],'panel_normal_world':[0,1,0],
            'source_anchor_world_m':[0,0,1],'target_progress':.15},
        'budget':{'control_steps_remaining':300}}

def test_replay_accepts_identical_sensor_state():
    image=np.zeros((16,16,3),np.uint8)
    assert compare_state(state(),state(),np.zeros(7),np.zeros(7),image,image)['accepted']

def test_replay_rejects_different_joint_posture_with_same_tcp():
    image=np.zeros((16,16,3),np.uint8);q=np.zeros(7);q[3]=.001
    assert not compare_state(state(),state(),q,np.zeros(7),image,image)['accepted']

def test_replay_rejects_changed_geometry_or_observed_scene():
    ref=state();changed=copy.deepcopy(ref);changed['mechanism_estimate']['source_anchor_world_m'][0]=.001
    image=np.zeros((16,16,3),np.uint8)
    assert not compare_state(changed,ref,np.zeros(7),np.zeros(7),image,image)['accepted']
    different=image.copy();different[0,0]=255
    assert not compare_state(ref,ref,np.zeros(7),np.zeros(7),image,different)['accepted']

def test_replay_requires_same_physical_step_and_progress():
    ref=state();changed=copy.deepcopy(ref);changed['budget']['control_steps_remaining']-=1
    image=np.zeros((16,16,3),np.uint8)
    assert not compare_state(changed,ref,np.zeros(7),np.zeros(7),image,image)['accepted']
    changed=copy.deepcopy(ref);changed['observed_state']['estimated_object_progress_m']+=.01
    assert not compare_state(changed,ref,np.zeros(7),np.zeros(7),image,image)['accepted']

def test_command_equivalence_ignores_wording_but_not_motion_parameters():
    a=dict(version='strategy-v2',operation='continue_motion',contact_mode=None,approach_tilt_deg=None,
        motion_extent='goal',step_mm=10,posture_rank=None,retreat_mm=None,evidence='One wording',expected_observation='Motion')
    b={**a,'evidence':'Different explanation'}
    assert signature(a)==signature(b)
    assert signature(a)!=signature({**b,'step_mm':5})
