import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vlm_tamp.strategy_decision import parse_decision,EnumeratedStrategy
from vlm_tamp.strategy_feedback import feedback_packet,planner_feedback
from vlm_tamp.shared_contact import contact_menu
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.rekep_libero.api import request_arguments,reservation_usd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/family-transfer-frozen-20261016/open_bottom_drawer-state005'


def decision(**changes):
    return {'version':'strategy-v1','operation':'try_contact','contact_mode':'press_drag',
        'approach_tilt_deg':45,'motion_extent':'probe','evidence':'Visible narrow contact region.',
        'expected_observation':'Target moves along the measured axis.',**changes}


def test_32k_requests_preserve_grounding_limits_and_reserve_reasoning(tmp_path):
    image=tmp_path/'image.png';image.write_bytes(b'RGB')
    old=request_arguments(image,'same instruction',12000);new=request_arguments(image,'same instruction',32000)
    assert new['reasoning']==old['reasoning']=={'effort':'high'}
    assert reservation_usd(new)-reservation_usd(old)==pytest.approx(.010)
    assert new['max_output_tokens']==32000 and old['max_output_tokens']==12000


@pytest.mark.parametrize('changes',[
    {'contact_mode':'bottom_drawer_special'}, {'approach_tilt_deg':90}, {'approach_tilt_deg':True},
    {'operation':'disable_collisions'}, {'extra':'code'}, {'motion_extent':'infinite'},
    {'operation':'finish'}, {'contact_mode':'pinch','approach_tilt_deg':45}, {'evidence':'x'*701},
])
def test_strategy_rejects_unsupported_choices_and_policy_changes(changes):
    with pytest.raises(ValueError):parse_decision(json.dumps(decision(**changes)))


def test_declared_mode_does_not_use_clearance_ordering():
    cloud=np.load(SOURCE/'grasp-cloud.npy');spec=read_json(SOURCE/'task-spec.json');meta=read_json(SOURCE/'opened/metadata.json')
    for mode in ('pinch','press_drag','push'):
        for motion in ('open','close'):
            menu,record=contact_menu(cloud,{**spec,'motion':motion},meta,np.empty((0,3)),mode_order=[mode])
            assert {r['mode'] for r in menu}=={mode} and record['mode_order_source']=='caller'


def test_rejected_pinch_never_silently_switches_mode(tmp_path,monkeypatch):
    from vlm_tamp import shared_contact_planning as module
    menu=Mock(return_value=([{'mode':'pinch','approach_tilt_deg':0}],{}))
    monkeypatch.setattr(module,'contact_menu',menu)
    monkeypatch.setattr(module,'rear_surface_contacts',lambda menu,*args:menu)
    pinch=Mock(side_effect=ValueError('No candidate'));monkeypatch.setattr(module,'select_pinch',pinch)
    scene=SimpleNamespace(static_points=np.zeros((1,3)),moving_reference=np.zeros((1,3)))
    with pytest.raises(ValueError,match='No contact mode'):
        module.select_contact({},np.zeros((1,3)),{},scene,{},None,
            {'metadata':{'joint_positions':[0]*7}},None,tmp_path,{'scene_voxel_spacing_m':.003},
            strategy={'contact_mode':'pinch'})
    assert menu.call_args.kwargs['mode_order']==['pinch'] and pinch.call_count==1


def test_feedback_omits_evaluation_fields_and_preserves_unknown_tracking():
    metadata={'eef_pose_world_xyzw':[0,0,0,0,0,0,1],'gripper_qpos':[.02,-.02],
        'control_steps':200,'evaluation_task_success':True,'object_ground_truth':{'secret':'oracle'}}
    spec={'joint_family':'prismatic','motion':'open','joint_axis_world':[0,1,0],
        'panel_normal_world':[0,1,0],'source_anchor_world_m':[0,0,0],'target_progress':.15,
        'progress_units':'meters','true_joint_axis':[1,0,0]}
    history=[{'decision':decision(),'result':{'status':'motion_completed','tracking_accepted':False,
        'observed_target_translation_world_m':None,'evaluation_task_success':True,'simulator_object_pose':[999]*3}}]
    progress=SimpleNamespace(records=[],observed_progress=0.,progress=.02)
    packet=feedback_packet('Open the selected part',metadata,spec,progress,{},history,
        round_index=1,maximum_rounds=5,goal_reached=False,active_contact=None)
    text=json.dumps(packet)
    assert 'evaluation_task_success' not in text and 'simulator_object_pose' not in text and 'true_joint_axis' not in text
    assert packet['history'][0]['result']['observed_target_translation_world_m'] is None
    assert packet['budget']['control_steps_remaining']==400


def test_feedback_counts_solver_events_without_claiming_actual_collision(tmp_path):
    write_json(tmp_path/'contact-mode-admission.json',{'attempts':[
        {'reason':'contact_pose_collision','diagnostic':{'static_rejections':1,'evaluation_success':False}},
        {'reason':'contact_probe_unplanned','planning':{'reason':'no_complete_constrained_path','attempts':[{'timed_out':1}]}}]})
    result=planner_feedback(tmp_path)
    assert result['rejection_record_counts']['contact_pose_collision']==1
    assert result['solver_diagnostics']['timed_out']==1
    assert 'evaluation_success' not in json.dumps(result)


def test_enumerated_control_uses_same_menu_and_waits_for_observed_progress(tmp_path):
    policy=EnumeratedStrategy();packet={'history':[],'observed_state':{'estimated_goal_reached':False}}
    first=policy.choose(packet,None,tmp_path,'x');assert first['contact_mode']=='pinch'
    packet['history']=[{'result':{'status':'motion_completed','tracking_accepted':False,'object_progress_delta_m':None}}]
    second=policy.choose(packet,None,tmp_path,'x');assert second['contact_mode']=='press_drag'
    packet['history']=[{'result':{'status':'motion_completed','tracking_accepted':True,'object_progress_delta_m':.02}}]
    third=policy.choose(packet,None,tmp_path,'x');assert third['operation']=='continue_motion'
