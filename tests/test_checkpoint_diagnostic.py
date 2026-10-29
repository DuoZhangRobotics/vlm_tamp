"""Causal cutoff, evidence isolation and immutability checks for saved decisions."""
import json

import pytest

from vlm_tamp.checkpoint_diagnostic import (
    build_packet, parse_decision, prefix_events, public_references, verify_packet,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_case(tmp_path):
    case = tmp_path / 'private-case'
    work = case / 'agent-workspace'
    for name in ['tools.txt', 'example/readme.txt', 'experience/guidelines.json']:
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    images = {}
    for camera in ['frontview', 'agentview', 'wrist', 'arm_side']:
        name = f'robot_data/attempt01/obs001/{camera}.png'
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'public-image')
        images[camera] = name
    sensor = case / 'failure-observations/attempt01/obs001/sensors.npz'
    sensor.parent.mkdir(parents=True)
    sensor.write_bytes(b'allowed-old-depth')
    events = [
        {'sequence': 1, 'request': {'operation': 'reset', 'arguments': {}},
         'response': {'ok': True, 'result': {'instruction': 'move the object',
             'control_steps': 60, 'remaining_control_steps': 540, 'images': images,
             'sensor_arrays': 'robot_data/attempt01/obs001/sensors.npz'}},
         'evaluator_secret': 'never include this'},
        {'sequence': 2, 'request': {'operation': 'execute', 'arguments': {}},
         'response': {'ok': True, 'result': {'task_success': True, 'future_secret': 'AFTER-CUTOFF'}}},
    ]
    (case / 'tool-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    (work / 'findings.txt').write_text('POST-EPISODE-ANSWER')
    (case / 'evaluation-private.json').write_text('NATIVE-GEOMETRY')
    return case


def test_only_prefix_and_referenced_assets_are_exposed(tmp_path):
    case = source_case(tmp_path)
    out = tmp_path / 'packet'
    manifest = build_packet(case, 1, out)
    assert verify_packet(out, manifest)['passed']
    text = '\n'.join(p.read_text() for p in out.rglob('*') if p.is_file())
    for forbidden in ['AFTER-CUTOFF', 'POST-EPISODE-ANSWER', 'NATIVE-GEOMETRY',
                      'evaluator_secret', str(case)]:
        assert forbidden not in text
    assert (out / 'robot_data/attempt01/obs001/sensors.npz').read_bytes() == b'allowed-old-depth'
    public = out / 'robot_data/attempt01/obs001/frontview.png'
    public.write_bytes(b'changed')
    assert (case / 'agent-workspace/robot_data/attempt01/obs001/frontview.png').read_bytes() == b'public-image'
    with pytest.raises(ValueError, match='registered hashes'):
        verify_packet(out, manifest)


@pytest.mark.parametrize('path', ['robot_data/../../private.json', 'robot_data//bad.json',
                                'robot_data/old\\private.json', 'robot_data/program.py'])
def test_references_cannot_grant_arbitrary_files(path):
    with pytest.raises(ValueError):
        public_references({'artifact': path})


def test_referenced_symlink_cannot_escape_source_case(tmp_path):
    case = source_case(tmp_path)
    external = tmp_path / 'private.png'
    external.write_bytes(b'private')
    image = case / 'agent-workspace/robot_data/attempt01/obs001/frontview.png'
    image.unlink()
    image.symlink_to(external)
    with pytest.raises(ValueError, match='outside'):
        build_packet(case, 1, tmp_path / 'packet')


def test_noncontiguous_prefix_and_future_artifact_fail_closed(tmp_path):
    case = source_case(tmp_path)
    with pytest.raises(ValueError, match='complete ordered'):
        prefix_events(case / 'tool-events.jsonl', 3)
    events = [json.loads(l) for l in (case / 'tool-events.jsonl').read_text().splitlines()]
    events[0]['response']['result']['secret'] = 'robot_data/attempt01/obs002/robot.json'
    (case / 'tool-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    with pytest.raises(ValueError, match='future call'):
        build_packet(case, 1, tmp_path / 'packet')


def test_missing_historical_asset_is_explicit(tmp_path):
    case = source_case(tmp_path)
    (case / 'failure-observations/attempt01/obs001/sensors.npz').unlink()
    manifest = build_packet(case, 1, tmp_path / 'packet')
    assert manifest['missing_assets'] == ['robot_data/attempt01/obs001/sensors.npz']


def test_malformed_decision_does_not_silently_pass(tmp_path):
    path = tmp_path / 'decision.json'
    dump(path, {'next_request': {'operation': 'observe', 'arguments': {}},
                'justification': 'Current state uncertain.',
                'expected_observation': 'A current image.', 'uncertainty': 'Contact unknown.'})
    assert parse_decision(path)['next_request']['operation'] == 'observe'
    value = json.loads(path.read_text())
    value['next_request']['arguments'] = 'not an object'
    dump(path, value)
    with pytest.raises(ValueError, match='next_request'):
        parse_decision(path)


def test_explicit_shared_memory_is_recorded_without_future_case_notes(tmp_path):
    case = source_case(tmp_path)
    (case / 'agent-workspace/experience/guidelines.json').unlink()
    memory = tmp_path / 'registered-memory.json'
    memory.write_text('{"principles":["A previously learned general lesson"]}')
    packet = tmp_path / 'packet'
    manifest = build_packet(case, 1, packet, shared_sources={'experience/guidelines.json': memory})
    assert (packet / 'experience/guidelines.json').read_bytes() == memory.read_bytes()
    provenance = manifest['asset_provenance']['experience/guidelines.json']
    assert provenance['explicitly_supplied_shared_input']
    assert not provenance['shared_original_input']
    assert not (packet / 'findings.txt').exists()


def test_shared_override_cannot_add_an_evaluator_file(tmp_path):
    case = source_case(tmp_path)
    with pytest.raises(ValueError, match='declared shared inputs'):
        build_packet(case, 1, tmp_path / 'packet',
                     shared_sources={'evaluation-private.json': case / 'evaluation-private.json'})


def test_multistage_rpc_can_explain_internal_artifact_id_gaps(tmp_path):
    case = source_case(tmp_path)
    path = case / 'tool-events.jsonl'
    first = json.loads(path.read_text().splitlines()[0])
    sequence = {'sequence': 4, 'request': {'operation': 'execute_sequence', 'arguments': {}},
                'response': {'ok': True, 'result': {'stages': [
                    {'stage': 'approach', 'artifact_id': 2},
                    {'stage': 'closure', 'artifact_id': 3},
                    {'stage': 'manipulation', 'artifact_id': 4}]}}}
    path.write_text(json.dumps(first) + '\n' + json.dumps(sequence) + '\n')
    assert [r['sequence'] for r in prefix_events(path, 4)] == [1, 4]
    # A cutoff inside an RPC cannot expose a later complete response.
    with pytest.raises(ValueError, match='complete ordered'):
        prefix_events(path, 3)


def test_unexplained_multistage_gap_is_still_rejected(tmp_path):
    case = source_case(tmp_path)
    path = case / 'tool-events.jsonl'
    first = json.loads(path.read_text().splitlines()[0])
    sequence = {'sequence': 4, 'request': {'operation': 'execute_sequence', 'arguments': {}},
                'response': {'ok': True, 'result': {'stages': [
                    {'stage': 'approach', 'artifact_id': 2},
                    {'stage': 'manipulation', 'artifact_id': 4}]}}}
    path.write_text(json.dumps(first) + '\n' + json.dumps(sequence) + '\n')
    with pytest.raises(ValueError, match='complete ordered'):
        prefix_events(path, 4)
