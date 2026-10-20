import json
from pathlib import Path

import numpy as np
import pytest

from run_all_prior_tasks import TASKS,manifest,configuration
from vlm_tamp.cohort_resources import CohortLedger
from vlm_tamp.cohort_execution import read_cohort_observation
from vlm_tamp.rekep_libero.api import BudgetError
from vlm_tamp.coupled_storage import compact_episode,read_json,write_json


def test_frozen_manifest_covers_every_prior_physical_task_and_state_once():
    planned=manifest(TASKS,[1,11,21,31,41])
    assert len(planned)==35
    assert len({(r['task'],r['state']) for r in planned})==35
    assert {t['adapter'] for t in TASKS}=={'point_goal','line','MC-VAMP hinge','control rotation'}
    with pytest.raises(ValueError):manifest(TASKS,[1,1])
    with pytest.raises(ValueError):manifest(TASKS,[0])


def test_cohort_keeps_physics_full_execution_and_budget_policy():
    config=configuration()
    assert config['physical_contacts_enabled'] and not config['assisted_grasping']
    assert not config['execution_vetoes'] and config['replay_grasp_source'] is None
    assert config['max_output_tokens']==12000 and config['maximum_control_steps']==600
    assert config['maximum_total_ik_postures']==96


def test_scene_limit_migration_preserves_costs_and_dollar_ceiling(tmp_path):
    # Exercise the exact ledger read/migration without credentials or network.
    ledger=object.__new__(CohortLedger);ledger.path=tmp_path/'ledger.json';ledger.max_calls=65;ledger.prior_spend=.181847
    original={'version':'rekep-libero-luna-1','model':'gpt-6-luna','budget_usd':2.,
        'prior_conservative_spend_usd':.181847,'max_scene_requests':30,
        'requests':{'old':{'cost_usd':.05,'reservation_usd':.06}}}
    ledger.path.write_text(json.dumps(original));updated=ledger._read()
    assert updated['requests']==original['requests'] and updated['budget_usd']==2.
    assert updated['max_scene_requests']==65 and updated['scene_limit_changes']
    original['budget_usd']=3.;ledger.path.write_text(json.dumps(original))
    with pytest.raises(BudgetError):ledger._read()


def test_cohort_observation_preserves_robot_depth_prior(tmp_path):
    arrays={'agentview_rgb':np.zeros((2,2,3),np.uint8),'agentview_depth':np.ones((2,2)),
        'agentview_world_points':np.ones((2,2,3)),'agentview_robot_depth_prior':np.full((2,2),.7)}
    np.savez(tmp_path/'obs.npz',**arrays)
    metadata={'primary_camera':'agentview','cameras':{'agentview':{}}}
    cameras,_=read_cohort_observation(tmp_path/'obs.npz',metadata)
    np.testing.assert_array_equal(cameras['agentview']['known_robot_depth_prior'],arrays['agentview_robot_depth_prior'])


def test_grasp_and_storage_retention_survive_repeated_cleanup(tmp_path):
    episode=tmp_path/'episode';episode.mkdir()
    write_json(episode/'summary.json',{'official_task_success':True})
    write_json(episode/'grasp-retention.json',{'verified':True,'source':'RGB-D observed transport'})
    compact_episode(episode,allowed_root=tmp_path)
    compact_episode(episode,allowed_root=tmp_path)
    assert read_json(episode/'grasp-retention.json')['verified']
    assert 'bytes_reclaimed_this_pass' in read_json(episode/'retention.json')
