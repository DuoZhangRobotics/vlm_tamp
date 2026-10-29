import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

import run_subscription_agent_pilot as pilot
import run_varied_contact_practice as study
from vlm_tamp.execution_limits import LEARNING_CONTROL_STEPS


def test_clean_actor_has_no_inherited_memory_and_exact_control_budget(tmp_path):
    arguments=['--attempts','6',*study.memory_arguments(None)]
    command=study.actor_command(tmp_path,arguments,control_steps=600,episode_seconds=1800)
    assert '--clean-start' in command and '--no-example' in command
    assert '--guidelines-source' not in command
    assert '--memory-store' not in command
    assert '--successful-method-file' not in command
    assert command[command.index('--control-steps')+1]=='600'
    assert LEARNING_CONTROL_STEPS==600


def test_memory_evaluation_receives_only_the_selected_frozen_file(tmp_path):
    memory=tmp_path/'frozen/guidelines.json'
    command=study.actor_command(tmp_path,['--attempts','1',*study.memory_arguments(memory)],
        control_steps=600,episode_seconds=1800)
    assert command[command.index('--guidelines-source')+1]==str(memory)
    assert '--clean-start' not in command
    assert '--learning-practice' not in command
    assert command[command.index('--attempts')+1]=='1'


def test_explicit_extended_budget_remains_available(tmp_path):
    command=study.actor_command(tmp_path,[],control_steps=1200,episode_seconds=1800)
    assert command[command.index('--control-steps')+1]=='1200'


@pytest.mark.parametrize('field',['guidelines_source','successful_method_file','memory_store','abstract_experience'])
def test_clean_start_rejects_inherited_guidance(field):
    args=SimpleNamespace(clean_start=True,no_example=False,**{field:'some-prior'})
    with pytest.raises(ValueError,match='Clean start cannot inherit'):
        pilot.validate_clean_start(args)


def test_clean_start_omits_worked_examples():
    args=SimpleNamespace(clean_start=True,no_example=False)
    pilot.validate_clean_start(args)
    assert args.no_example


def test_evaluation_prompt_states_the_actual_single_episode_and_budget():
    prompt=pilot.default_prompt('close_microwave',1,600,no_example=True)
    assert 'at most 1\nphysical attempt on the provided case, 600 control steps each' in prompt
    assert 'Close the microwave' in prompt
    assert 'at most two' not in prompt
    assert 'approved example' not in prompt
    assert 'guidelines.json' in prompt  # Evaluators may keep their own lessons.


def write_protocol(out,expected=None,clean=True):
    (out/'protocol.json').write_text(json.dumps({'clean_start':clean,
        'frozen_agent_guidelines_sha256':expected,'example_supplied':False}))


def test_clean_audit_allows_new_notebooks(tmp_path):
    write_protocol(tmp_path)
    work=tmp_path/'agent-workspace';(work/'experience').mkdir(parents=True)
    (work/'guidelines.json').write_text('{"principles":["new lesson"]}')
    (work/'experience/guidelines.json').write_text('new notes written during this run')
    result=study.audit_memory_input(tmp_path,None)
    assert result['passed'] and result['delivered_memory_sha256'] is None
    assert result['experience_file_created_during_clean_run']


def test_clean_audit_rejects_an_inherited_protocol(tmp_path):
    write_protocol(tmp_path,'old-memory-hash',False)
    with pytest.raises(RuntimeError,match='Delivered memory'):
        study.audit_memory_input(tmp_path,None)


def test_frozen_memory_must_remain_exact(tmp_path):
    data=b'{"principles":["learned rule"]}'
    digest=hashlib.sha256(data).hexdigest();write_protocol(tmp_path,digest,False)
    p=tmp_path/'agent-workspace/experience/guidelines.json';p.parent.mkdir(parents=True);p.write_bytes(data)
    assert study.audit_memory_input(tmp_path,digest)['passed']
    p.write_bytes(data+b' ')
    with pytest.raises(RuntimeError,match='Delivered memory'):
        study.audit_memory_input(tmp_path,digest)


@pytest.fixture
def completed_practice():
    agent={'normal_terminal':True,'returncode':-15,'termination_reason':'attempt_budget_exhausted'}
    attempt={'reason':'control_budget_exhausted','official_task_success':False,
        'control_steps':600,'maximum_control_steps':600,'remaining_wall_seconds':10,
        'premature_end':False,'all_trajectories_completed_or_success_terminated':True}
    return agent,{'attempts':[copy.deepcopy(attempt) for _ in range(6)]}


def test_completed_unsuccessful_practice_is_valid_data(completed_practice):
    agent,evaluation=completed_practice
    rows=study.validate_run_exit(143,agent,evaluation,expected_attempts=6)
    assert len(rows)==6 and not any(r['official_task_success'] for r in rows)


@pytest.mark.parametrize('mode',['missing_episode','incomplete_motion','unexpected_exit','interrupted'])
def test_practice_does_not_hide_incomplete_or_interrupted_runs(completed_practice,mode):
    agent,evaluation=completed_practice
    if mode=='missing_episode':evaluation['attempts'].pop()
    if mode=='incomplete_motion':evaluation['attempts'][0]['all_trajectories_completed_or_success_terminated']=False
    with pytest.raises(RuntimeError):
        study.validate_run_exit(1 if mode=='unexpected_exit' else 143,agent,evaluation,
            expected_attempts=6,interrupted=mode=='interrupted')
