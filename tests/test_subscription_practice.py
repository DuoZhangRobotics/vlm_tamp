"""Practice budgets cannot silently become evaluation retries."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_subscription_learning_practice import practice_prompt, build_command, summarize, efficiency_metrics, successful_method_payload, load_successful_method
from run_subscription_remembered_agent import command as evaluation_command
from vlm_tamp.subscription_agent import validate_attempt_budget
from vlm_tamp.practice_budget import PracticeBudget
from vlm_tamp.learning_phases import learning_phase


@pytest.mark.parametrize('attempts,practice,evaluation', [
    (0, True, False), (21, True, False), (True, True, False),
    (1, True, False), (5, False, False), (5, True, True),
])
def test_invalid_or_conflicting_budgets_are_rejected(attempts, practice, evaluation):
    with pytest.raises(ValueError):
        validate_attempt_budget(attempts, learning_practice=practice, evaluation_case=evaluation)


def test_practice_is_one_session_and_does_not_change_evaluation_command():
    args = SimpleNamespace(attempts=20, initial_attempts=5, state=47, scene_variant='nominal',
        agent_seconds=9000, output='results/practice-test-unused',
        memory_profile='configs/subscription_memory.json')
    command, _ = build_command(args, '/tmp/generated-practice-prompt.txt')
    assert command[command.index('--attempts')+1] == '20'
    assert command[command.index('--initial-attempts')+1] == '5'
    assert command.count('--learning-practice') == 1
    assert '--evaluation-case' not in command
    original, _ = evaluation_command(args)
    assert original[original.index('--attempts')+1] == '1'
    assert '--learning-practice' not in original
    profile = json.loads((ROOT/args.memory_profile).read_text())
    registered = (ROOT/profile['prompt']).read_text()
    prompt = practice_prompt(registered, 20, 5)
    assert 'There is ONE physical attempt' not in prompt
    assert 'reset for another attempt' not in prompt
    assert '600 recorded control steps per attempt' in prompt
    assert 'one continuous learning session' in prompt
    assert 'No external model APIs' in prompt
    assert 'Completed requires a fresh supported measured check' not in prompt
    assert 'Completion measurements are advisory' in prompt
    assert 'Current phase: SUCCESS FIRST' in prompt
    assert 'Until the phase changes, do not run efficiency experiments' in prompt
    assert 'Efficiency is a soft objective' not in prompt
    assert 'such as a longer useful planned segment' not in prompt
    assert '--stop-on-success' in command
    assert 'request_attempts' in prompt
    assert (ROOT/profile['prompt']).read_text() == registered


@pytest.mark.parametrize('attempts,practice', [(1, False), (20, True)])
def test_generated_tool_docs_match_actual_attempt_budget(tmp_path, attempts, practice):
    work, output = tmp_path/'work', tmp_path/'output'
    command = [sys.executable, str(ROOT/'scripts/prepare_subscription_pilot.py'),
        '--workspace', str(work), '--output', str(output), '--state', '47',
        '--attempts', str(attempts), '--abstract-experience', '--completion-checks', '--memory-store',
        str(ROOT/'data/agent_memory/luna-microwave-v2/library')]
    if practice:
        command.extend(['--learning-practice', '--initial-attempts', '5'])
    subprocess.run(command, check=True, capture_output=True, text=True)
    docs = (work/'tools.txt').read_text()
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['maximum_attempts'] == attempts
    assert protocol['learning_practice'] == practice
    assert protocol['maximum_control_steps_per_attempt'] == 600
    assert protocol['episodic_memory']['read_only']
    assert not protocol['episodic_memory']['automatic_cross_case_updates']
    if practice:
        assert 'at most 20 resets' in docs
        assert 'ONE physical attempt' not in docs
        assert 'current ONE' not in docs
        assert 'notes persist' in docs
        assert 'initially 5 attempts' in docs
        assert 'request_attempts' in docs
        assert 'completed requires a fresh supported check' not in docs
        assert protocol['initial_attempts_granted'] == 5
        assert protocol['completion_policy'] == 'advisory'
    else:
        assert 'exactly one reset' in docs
        assert 'ONE physical attempt' in docs
        assert 'completed requires a fresh supported check' in docs
        assert protocol['completion_policy'] == 'required_measurement'


def test_summary_keeps_failures_unused_budget_and_transient_success_separate(tmp_path):
    outcomes = [dict(attempt=1, official_task_success=False, final_task_success=False),
                dict(attempt=2, official_task_success=True, final_task_success=False),
                dict(attempt=3, official_task_success=True, final_task_success=True)]
    for row in outcomes:
        row.update(execution_veto_checks=0, all_issued_trajectories_completed=True)
    (tmp_path/'evaluation-private.json').write_text(json.dumps({'attempts': outcomes}))
    summary = summarize(tmp_path, 5, 0, 100)
    assert summary['attempts_used'] == 3
    assert summary['unused_attempt_budget'] == 2
    assert summary['native_goal_reached_attempts'] == [2, 3]
    assert summary['native_goal_retained_attempts'] == [3]
    assert summary['first_native_success_attempt'] == 2
    assert [r['status'] for r in summary['attempts']] == ['attempted']*3+['not_attempted']*2
    assert not summary['generalization_measured']
    assert not summary['automatic_memory_promotion']


def test_extensions_require_finished_block_and_rationale_and_respect_hard_cap():
    budget = PracticeBudget(5, 20)
    for used, active, learned, tests in [(4, False, 'lesson', ['test']),
                                        (5, True, 'lesson', ['test']),
                                        (5, False, '', ['test']),
                                        (5, False, 'lesson', []),
                                        (5, False, 'lesson', [''])]:
        with pytest.raises(ValueError):
            budget.extend(used, active, learned, tests)
        assert budget.granted == 5
    for used, expected in [(5, 10), (10, 15), (15, 20)]:
        result = budget.extend(used, False, 'Evidence remains ambiguous', ['Test a different observation'])
        assert result['granted_attempts'] == expected
        assert not result['extension']['native_success_used']
    with pytest.raises(ValueError, match='cap'):
        budget.extend(20, False, 'Another idea', ['Test it'])
    assert budget.granted == 20


def test_final_extension_is_clipped_to_user_cap():
    budget = PracticeBudget(5, 12)
    budget.extend(5, False, 'lesson', ['test'])
    result = budget.extend(10, False, 'lesson', ['test'])
    assert result['granted_attempts'] == 12
    assert result['remaining_granted_attempts'] == 2


def test_efficiency_counts_failed_planning_and_keeps_tool_time_separate():
    events=[{'request':{'operation':op},'response':{'ok':ok,'result':result},'wall_seconds':seconds}
            for op,ok,result,seconds in [
                ('reset',True,{'attempt':1},1.),('plan_arc',False,{},2.),
                ('plan_arc',True,{'planned':True},3.),('execute',True,{},4.),
                ('reset',True,{'attempt':2},1.),('plan_joint',True,{'planned':False},2.)]]
    rows=efficiency_metrics(events)
    assert rows[0]['planning_calls']==2 and rows[0]['failed_planning_calls']==1
    assert rows[0]['robot_tool_calls']==4 and rows[0]['executed_trajectories']==1
    assert rows[0]['tool_seconds']==10. and rows[0]['attempt_wall_seconds'] is None
    assert rows[1]['failed_planning_calls']==1


def test_efficiency_instructions_are_withheld_until_two_consecutive_successes():
    rows=[{'attempt':1,'official_task_success':True},{'attempt':2,'official_task_success':False},
          {'attempt':3,'official_task_success':True}]
    first=learning_phase(rows)
    assert first['phase']=='success_first' and not first['verified_baseline_attempts']
    assert 'Efficiency is now a soft objective' not in first['instructions']
    rows.append({'attempt':4,'official_task_success':True})
    ready=learning_phase(rows)
    assert ready['phase']=='efficiency' and ready['verified_baseline_attempts']==[3,4]
    assert 'Efficiency is now a soft objective' in ready['instructions']
    rows.append({'attempt':5,'official_task_success':False})
    assert learning_phase(rows)['verified_baseline_attempts']==[3,4]


def test_successful_method_is_agent_content_without_raw_history(tmp_path,monkeypatch):
    import run_subscription_learning_practice as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    run=tmp_path/'results/prior';(run/'agent-workspace').mkdir(parents=True)
    outcomes={'attempts':[{'attempt':n,'official_task_success':True,'final_task_success':True} for n in [1,2]]}
    (run/'evaluation-private.json').write_text(json.dumps(outcomes))
    source={'principles':[{'principle':'Use fresh geometry','relative_choice':'15 degree trial',
        'evidence':{'references':['replies/old.json'],'limitations':'One scene only'}}]}
    p=run/'agent-workspace/guidelines.json';p.write_text(json.dumps(source))
    result=successful_method_payload(run)
    expected=json.loads(json.dumps(source));del expected['principles'][0]['evidence']['references']
    assert result['method']==expected and result['observed_successful_attempts']==[1,2]
    source['principles'][0]['old_pose']='[0.1, 0.2, 0.3]';p.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='coordinate'):
        successful_method_payload(run)


def test_registered_method_is_loaded_by_default_and_tampering_cannot_fall_back(tmp_path,monkeypatch):
    import hashlib
    import run_subscription_learning_practice as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    path=tmp_path/'method.json';path.write_text(json.dumps({'method':{'principles':['Agent-authored lesson']}}))
    profile={'successful_method':'method.json','successful_method_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'successful_method_source_run':'results/verified-run'}
    value,source=load_successful_method(profile)
    assert source=='results/verified-run' and value['method']['principles']==['Agent-authored lesson']
    path.write_text('{}')
    with pytest.raises(RuntimeError,match='do not silently fall back'):
        load_successful_method(profile)


def test_service_blocks_next_reset_until_an_extension_is_granted(tmp_path):
    import importlib.util
    path = ROOT/'scripts/serve_subscription_robot.py'
    spec = importlib.util.spec_from_file_location('practice_robot_test', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot = module.RobotService.__new__(module.RobotService)
    robot.practice_budget = PracticeBudget(5, 20)
    robot.attempt = robot.limit = 5
    robot.sim = None
    robot.results = []
    robot.output = tmp_path
    with pytest.raises(ValueError, match='block exhausted'):
        robot.reset()
    result = robot.request_attempts('Compared failures; need another test', ['Check the current hypothesis'])
    assert robot.limit == result['granted_attempts'] == 10
    assert robot.attempt == 5
    record = json.loads((tmp_path/'practice-budget.json').read_text())
    assert len(record['extensions']) == 1
    assert not record['extensions'][0]['native_success_used']
