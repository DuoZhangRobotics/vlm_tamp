"""A reviewer must know the retry start, separately from the preceding outcome."""
import json

import pytest

from vlm_tamp.reflexion_experiment import (
    prepare_review_packet, retry_start, tree_hashes,
)


def records():
    start = {'control_steps': 277, 'remaining_control_steps': 323, 'task_success': False,
             'images': {'frontview': 'robot_data/start.png'}}
    return [
        {'sequence': 1, 'request': {'operation': 'reset'}, 'response': {'ok': True,
            'result': {'instruction': 'Open the fixture', 'observation': start}}},
        {'sequence': 2, 'request': {'operation': 'ground'}, 'response': {'ok': True,
            'result': {'model': {'joint_origin_world_m': [.1, .2, .3]}}}},
        {'sequence': 3, 'request': {'operation': 'observe'}, 'response': {'ok': True, 'result': start}},
        {'sequence': 4, 'request': {'operation': 'set_model'}, 'response': {'ok': True,
            'result': {'model': {'joint_origin_world_m': [-.05, .2, .3]},
                       'point_clouds': 'robot_data/geometry004/points.npz'}}},
        {'sequence': 5, 'request': {'operation': 'ground'}, 'response': {'ok': True,
            'result': {'model': {'joint_origin_world_m': [9., 9., 9.]}}}},
        {'sequence': 6, 'request': {'operation': 'observe'}, 'response': {'ok': True,
            'result': {'control_steps': 600, 'remaining_control_steps': 0, 'task_success': True,
                       'images': {'frontview': 'robot_data/final.png'}}}},
    ]


def packet(tmp_path):
    source = tmp_path / 'actor-record'
    source.mkdir()
    (source / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records()))
    (source / 'checkpoint.json').write_text(json.dumps({'control_steps': 600,
        'current_observation': records()[-1]['response']['result'], 'scope': 'completed record'}))
    (source / 'outcome.json').write_text(json.dumps({'task_success': True}))
    return source, {'packet_sha256': tree_hashes(source)}


def test_restart_preserves_configured_bad_estimate_and_excludes_future():
    rows = records()
    before = json.dumps(rows, sort_keys=True)
    value = retry_start(rows, 4)
    assert value['control_steps'] == 277
    assert value['remaining_control_steps'] == 323
    assert value['current_observation']['task_success'] is False
    # Preserve the actual manual edit, rather than injecting an earlier/future fit.
    assert value['current_model']['joint_origin_world_m'] == [-.05, .2, .3]
    assert value['last_completed_event'] == 4
    assert value['another_reset_available'] is False
    assert json.dumps(rows, sort_keys=True) == before


@pytest.mark.parametrize('cutoff', [0, 3, 6, 99])
def test_missing_or_non_model_boundaries_are_rejected(cutoff):
    with pytest.raises(ValueError):
        retry_start(records(), cutoff)


def test_terminal_restart_is_rejected():
    rows = records()
    rows.append({'sequence': 7, 'request': {'operation': 'set_model'},
                 'response': {'ok': True, 'result': {'model': {'x': 1}}}})
    with pytest.raises(ValueError, match='nonterminal'):
        retry_start(rows, 7)


def test_only_reviewer_copy_gets_restart_context(tmp_path):
    source, manifest = packet(tmp_path)
    destination = tmp_path / 'review-inputs'
    result = prepare_review_packet(source, manifest, destination, 4)
    assert tree_hashes(source) == manifest['packet_sha256']
    assert not (source / 'retry-start.json').exists()
    assert result['actor_record_unchanged']
    assert result['packet_sha256'] == tree_hashes(destination)
    assert json.loads((destination / 'retry-start.json').read_text())['control_steps'] == 277
    terminal = json.loads((destination / 'checkpoint.json').read_text())
    assert terminal['control_steps'] == 600
    assert 'Terminal snapshot' in terminal['scope']
    assert 'retry-start.json' in terminal['scope']


def test_review_packet_does_not_follow_symlinks(tmp_path):
    source, manifest = packet(tmp_path)
    outside = tmp_path / 'private.txt'
    outside.write_text('not public evidence')
    (source / 'link.txt').symlink_to(outside)
    with pytest.raises(ValueError, match='symlinks'):
        prepare_review_packet(source, manifest, tmp_path / 'review', 4)
    assert not (tmp_path / 'review').exists()


@pytest.mark.parametrize('conditions,count', [(['reflection'], 2), (None, 6)])
def test_registration_limits_scheduled_work(tmp_path, monkeypatch, conditions, count):
    import run_reflexion_study as driver
    preflight = tmp_path / 'preflight002'
    preflight.mkdir()
    (preflight / 'checkpoint-replay-audit.json').write_text('{"passed": true}')
    memory = tmp_path / 'guidelines.json'
    memory.write_text('{}')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'manifest.json').write_text('{}')
    monkeypatch.setattr(driver, 'MEMORY', memory)
    monkeypatch.setattr(driver, 'MEMORY_SHA', driver.sha256(memory))
    monkeypatch.setattr(driver, 'DEFAULT_BUNDLE', bundle)
    monkeypatch.setattr(driver, 'load_manifest', lambda _: {'packages': {}, 'python_major_minor': [3, 12]})
    monkeypatch.setattr(driver, 'freeze_sources', lambda _: {})
    def completed_record(case, destination):
        destination.mkdir(parents=True)
        return {'packet_sha256': {}}
    monkeypatch.setattr(driver, 'completed_record', completed_record)
    out = tmp_path / 'study'
    driver.prepare(out, conditions=conditions)
    protocol = json.loads((out / 'protocol.json').read_text())
    schedule = json.loads((out / 'schedule.json').read_text())
    assert protocol['physical_attempts'] == protocol['review_calls'] == len(schedule) == count
    assert len({job['case_id'] for job in schedule}) == count
    if conditions == ['reflection']:
        assert all(job['condition'] == 'reflection' for job in schedule)
        assert 'nonconcurrent' in protocol['comparison_scope']


@pytest.mark.parametrize('conditions', [[], ['reflection', 'reflection'], ['unknown']])
def test_invalid_condition_sets_do_not_create_a_run(tmp_path, conditions):
    import run_reflexion_study as driver
    out = tmp_path / 'study'
    with pytest.raises(ValueError):
        driver.prepare(out, conditions=conditions)
    assert not out.exists()
