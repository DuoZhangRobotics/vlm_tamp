"""A CLI deadline must not discard a completed physical evaluation."""
import copy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / 'scripts/run_restriction_study.py'
spec = importlib.util.spec_from_file_location('restriction_study_under_test', path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture
def completed_control_budget():
    # Shape of the retained opening-state-13 episode that stopped the batch.
    agent = {'normal_terminal': True, 'returncode': -15,
             'termination_reason': 'control_budget_exhausted'}
    evaluation = {'attempts': [{
        'reason': 'control_budget_exhausted', 'official_task_success': False,
        'control_steps': 1200, 'maximum_control_steps': 1200,
        'remaining_wall_seconds': 31.1, 'premature_end': False,
        'all_trajectories_completed_or_success_terminated': True,
    }]}
    return agent, evaluation


def test_cli_deadline_preserves_completed_budget_failure(completed_control_budget):
    agent, evaluation = completed_control_budget
    before = copy.deepcopy(evaluation)
    attempt = study.validate_terminal(143, agent, evaluation)
    assert attempt['official_task_success'] is False
    assert attempt['control_steps'] == 1200
    assert evaluation == before


@pytest.mark.parametrize('reason,success,controls,wall', [
    ('simulator_success', True, 214, 1260.),
    ('wall_budget_exhausted', False, 1182, 0.),
])
@pytest.mark.parametrize('returncode', [0, 143])
def test_other_verified_terminals(completed_control_budget, reason, success, controls, wall, returncode):
    agent, evaluation = completed_control_budget
    agent.update(termination_reason=reason, returncode=-15 if returncode else 0)
    evaluation['attempts'][0].update(reason=reason, official_task_success=success,
                                    control_steps=controls, remaining_wall_seconds=wall)
    assert study.validate_terminal(returncode, agent, evaluation)['official_task_success'] is success


@pytest.mark.parametrize('change', [
    {'control_steps': 1199}, {'maximum_control_steps': 0},
    {'maximum_control_steps': None}, {'premature_end': True},
    {'all_trajectories_completed_or_success_terminated': False},
    {'reason': 'wall_budget_exhausted', 'remaining_wall_seconds': 1.},
    {'reason': 'simulator_success', 'official_task_success': False},
])
def test_incomplete_episode_is_not_accepted(completed_control_budget, change):
    agent, evaluation = completed_control_budget
    evaluation['attempts'][0].update(change)
    with pytest.raises(RuntimeError):
        study.validate_terminal(143, agent, evaluation)


@pytest.mark.parametrize('returncode,agent_change,interrupted', [
    (1, {}, False), (-15, {}, False), (143, {'returncode': 0}, False),
    (143, {'normal_terminal': False}, False),
    (143, {'termination_reason': 'operator_interruption'}, False),
    (143, {}, True),
])
def test_infrastructure_and_operator_interruptions_are_rejected(
        completed_control_budget, returncode, agent_change, interrupted):
    agent, evaluation = completed_control_budget
    agent.update(agent_change)
    with pytest.raises(RuntimeError):
        study.validate_terminal(returncode, agent, evaluation, interrupted)


@pytest.mark.parametrize('count', [0, 2])
def test_requires_exactly_one_recorded_episode(completed_control_budget, count):
    agent, evaluation = completed_control_budget
    evaluation['attempts'] *= count
    with pytest.raises(RuntimeError):
        study.validate_terminal(143, agent, evaluation)
