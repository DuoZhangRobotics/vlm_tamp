import inspect
import re
from pathlib import Path

import run_luna_pipeline_validation as study
import prepare_subscription_pilot as prepare
from serve_subscription_robot import RobotService


def test_three_fresh_regressions_each_begin_with_open37():
    from collections import Counter
    plan=study.plan();rows=plan['baseline']
    assert rows[0]['task']=='open_microwave' and rows[0]['state_index']==37
    assert len({r['run'] for r in rows})==12
    assert set(Counter((r['task'],r['state_index']) for r in rows).values())=={3}
    assert plan['control_steps']==600 and plan['episode_seconds']==1800


def test_holdouts_are_paired_and_excluded_from_this_study_training_and_regression():
    plan=study.plan();used={(r['task'],r['state_index']) for r in plan['baseline']+plan['practice_cases']}
    holdouts={(r['task'],r['state_index']) for r in plan['transfer']}
    assert len(holdouts)==4 and not used.intersection(holdouts)
    for case in holdouts:
        assert {r['memory_condition'] for r in plan['transfer'] if (r['task'],r['state_index'])==case}=={'none','frozen'}
    assert not plan['evaluation_notebooks_fed_forward']


def test_selected_canary_is_one_declared_case_and_is_not_repeated_in_the_batch():
    manifest={'plan':study.plan(),'completed_runs':[]}
    name='baseline-r1-close_microwave-state049-600'
    rows=study.pending_baselines(manifest,name,1)
    assert len(rows)==1 and rows[0]['run']==name
    manifest['completed_runs'].append({'run':name})
    remaining=study.pending_baselines(manifest)
    assert len(remaining)==11 and name not in {r['run'] for r in remaining}


def test_investigated_retry_preserves_original_directory_and_counts_scheduled_case(tmp_path):
    import json
    name='baseline-r1-close_microwave-state017-600';replacement=name+'-monitor-v2'
    original=tmp_path/'trials'/name;original.mkdir(parents=True)
    (original/'operator-interruption.json').write_text(json.dumps({'reason':'host-only pressure signal'}))
    manifest={'plan':study.plan(),'completed_runs':[],
              'retry_runs':{name:{'run':replacement,'reason':'Investigated monitoring interruption'}}}
    entry=next(r for r in manifest['plan']['baseline'] if r['run']==name)
    assert study.attempt_run_name(tmp_path,manifest,entry)==replacement
    assert original.exists()
    manifest['completed_runs'].append({'run':replacement,'scheduled_run':name})
    assert name not in {r['run'] for r in study.pending_baselines(manifest)}


def test_supplied_tool_help_matches_live_grasp_and_collision_contract():
    for method in ['plan_grasp','plan_sequence']:
        count=inspect.signature(getattr(RobotService,method)).parameters['max_candidates'].default
        assert re.search(rf'- {method}\([^)]*max_candidates={count}\b',prepare.DOC)
    assert 'collision_policy="advisory"' not in prepare.DOC
    assert 'different seed requests a' in prepare.DOC
    assert 'Closed-hand target contact is not a free-space admission test' in prepare.DOC
    practice=(Path(__file__).resolve().parents[1]/'prompts/contact_skills_practice.txt').read_text()
    assert 'collision predictions are advisory' not in practice
    assert 'successes on different cases do not' in practice
