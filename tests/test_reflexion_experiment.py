"""Meaningful boundaries for counterfactual episode memory and time budgets."""
import copy

import pytest

from vlm_tamp.reflexion_experiment import (
    ACTOR_PROMPT, remaining_actor_seconds, validate_reflection,
)


ROWS = [{'sequence': 12, 'request': {'operation': 'set_model'},
         'response': {'ok': True, 'result': {'value': [1., 2., 3.], 'a/b': {'~key': False}}}}]
VALID = {'evidence': [{'event': 12, 'pointer': '/response/result/value', 'value': [1., 2., 3.]}],
         **{k: 'Uncertain candidate from public evidence.' for k in
            ['observation', 'hypothesis', 'next_change', 'test', 'uncertainty', 'scope']}}


def test_exact_public_citation_and_explicit_noncausal_validation():
    result = validate_reflection(VALID, ROWS)
    assert result['passed']
    assert 'not certified' in result['scope']


@pytest.mark.parametrize('change', [
    {'event': 13}, {'pointer': '/native_geometry'}, {'value': [1., 2., 3.0001]},
    {'pointer': ''}, {'event': True},
])
def test_rejects_unavailable_future_or_misquoted_evidence(change):
    value = copy.deepcopy(VALID)
    value['evidence'][0].update(change)
    with pytest.raises(ValueError):
        validate_reflection(value, ROWS)


def test_json_pointer_escape_and_boolean_type_preserved():
    value = copy.deepcopy(VALID)
    value['evidence'] = [{'event': 12, 'pointer': '/response/result/a~1b/~0key', 'value': False}]
    assert validate_reflection(value, ROWS)['passed']
    value['evidence'][0]['value'] = 0
    with pytest.raises(ValueError):
        validate_reflection(value, ROWS)


def test_replay_time_charged_and_no_reset_prompt():
    assert remaining_actor_seconds({'deadline_monotonic': 2800}, 1300) == 1500
    with pytest.raises(TimeoutError):
        remaining_actor_seconds({'deadline_monotonic': 2800}, 2801)
    assert 'Do not reset' in ACTOR_PROMPT
    assert 'previous attempt only' in ACTOR_PROMPT


def test_no_silent_overlong_memory():
    value = copy.deepcopy(VALID)
    value['hypothesis'] = 'Maybe ' * 351
    with pytest.raises(ValueError):
        validate_reflection(value, ROWS)
