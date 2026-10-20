import json
import numpy as np
import pytest
from vlm_tamp.family_transfer import TASKS,STATES,manifest,frozen_configuration,core_audit,TransferLedger
from vlm_tamp.cohort_resources import CohortLedger
from vlm_tamp.rekep_libero.api import BudgetError
from vlm_tamp.mechanism_surface import normalize_sheet_choices


def test_transfer_split_has_controls_and_disjoint_cases():
    rows=manifest()
    assert len(rows)==33 and sum(r['phase']=='reproduction_control' for r in rows)==3
    assert sum(r['phase']=='original_tasks' for r in rows)==15
    assert sum(r['phase']=='related_tasks' for r in rows)==15
    assert len({(r['task'],r['state']) for r in rows})==33
    assert not set(STATES)&{0,1,2,3,4,11,21,31,41}
    assert all('legacy_index' not in t for t in TASKS)


def test_numerical_geometry_and_planning_are_exact_baseline_sources():
    assert all(r['matches'] for r in core_audit())
    configuration=frozen_configuration()
    assert configuration['maximum_candidate_grasps']==100
    assert configuration['maximum_total_ik_postures']==256
    altered=frozen_configuration();altered['maximum_candidate_grasps']=1
    assert frozen_configuration()==configuration


def ledger(tmp_path,cap,maximum=84):
    service=TransferLedger(tmp_path,maximum_scenes=maximum)
    raw={'version':'rekep-libero-luna-1','model':'gpt-6-luna','budget_usd':2.,
         'prior_conservative_spend_usd':service.prior_spend,'max_scene_requests':cap,
         'requests':{'prior':{'reservation_usd':.007,'cost_usd':.002,'status':'completed'}}}
    service.path.write_text(json.dumps(raw));return service,raw


def test_count_extension_preserves_budget_and_charges(tmp_path):
    service,before=ledger(tmp_path,65);saved=service.snapshot()
    assert saved['max_scene_requests']==84 and saved['budget_usd']==2.
    assert saved['requests']==before['requests']
    assert saved['cumulative_conservative_spend_usd']==service.prior_spend+.002
    old=CohortLedger(tmp_path)
    assert old.snapshot()['max_scene_requests']==84


def test_count_extension_cannot_change_dollar_cap(tmp_path):
    service,raw=ledger(tmp_path,65);raw['budget_usd']=20.;service.path.write_text(json.dumps(raw))
    with pytest.raises(BudgetError):service.snapshot()


def test_existing_coordinate_repair_preserves_geometry_and_roles():
    raw={'camera':'frontview','joint_family':'prismatic','axis_source':'panel_normal','motion':'open',
        'contact_box_xyxy':[200,760,250,790],'moving_panel_box_xyxy':[180,710,270,820],
        'grasp_pixel_xy':[225,775],'hinge_pixels_xy':None,'travel_m':.15}
    repaired,record=normalize_sheet_choices(raw)
    assert record['offset_xy']==[0,560]
    assert repaired['grasp_pixel_xy']==[225,215] and repaired['motion']=='open'
    assert raw['grasp_pixel_xy']==[225,775]
