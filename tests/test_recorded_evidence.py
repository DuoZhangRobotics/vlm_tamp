"""Temporal isolation and diagnostic scoring tests, without model calls."""
import json

import pytest

from vlm_tamp.checkpoint_diagnostic import verify_packet
from vlm_tamp.recorded_evidence import (
    attached_images, build_stages, score_labels, validate_decision,
)
from test_checkpoint_diagnostic import source_case


def staged_case(tmp_path):
    source = source_case(tmp_path)
    work = source / 'agent-workspace'
    (work / 'tools').mkdir()
    (work / 'tools/execute.txt').write_text('Original public execute contract.')
    shared = {name: work / name for name in ['tools.txt', 'experience/guidelines.json', 'example/readme.txt']}
    return source, build_stages(source, 2, tmp_path / 'packets', shared_sources=shared)


def test_result_is_absent_from_before_packet_and_only_pending_request_is_added(tmp_path):
    source, manifests = staged_case(tmp_path)
    before = tmp_path / 'packets/before'
    after = tmp_path / 'packets/after'
    assert json.loads((before / 'pending-request.json').read_text()) == {'operation': 'execute', 'arguments': {}}
    before_text = '\n'.join(p.read_text() for p in before.rglob('*') if p.is_file())
    assert 'AFTER-CUTOFF' not in before_text
    assert 'evaluator_secret' not in before_text
    assert 'POST-EPISODE-ANSWER' not in before_text
    assert 'AFTER-CUTOFF' in (after / 'latest-result.json').read_text()
    for stage in ['before', 'after']:
        assert verify_packet(tmp_path / 'packets' / stage, manifests[stage])['passed']
    assert (before / 'tools/execute.txt').read_text() == 'Original public execute contract.'
    assert not (before / 'decision-schema.json').exists()


def test_attachment_order_preserves_source_bytes_and_stale_snapshot_identity(tmp_path):
    source, manifests = staged_case(tmp_path)
    images = attached_images(tmp_path / 'packets', manifests, 'after')
    assert len(images) == 8
    assert [p.name for p in images] == ['frontview.png'] * 2 + ['agentview.png'] * 2 + ['wrist.png'] * 2 + ['arm_side.png'] * 2
    assert all(p.read_bytes() == b'public-image' for p in images)
    # A result without an observation does not fabricate a new sensor capture.
    assert all('obs001' in str(p) for p in images)
    images[0].write_bytes(b'changed')
    assert (source / 'agent-workspace/robot_data/attempt01/obs001/frontview.png').read_bytes() == b'public-image'
    with pytest.raises(ValueError, match='registered hashes'):
        verify_packet(tmp_path / 'packets/before', manifests['before'])


def decision():
    return {'motion_assessment': 'no_visible_motion', 'pose_update_basis': 'hand_prediction',
            'contact_assessment': 'uncertain', 'task_success': False,
            'next_step': {'kind': 'inspect_existing', 'request': None, 'description': 'Compare original views.'},
            'observed_changes': 'No visible door displacement.', 'evidence': ['paired images'],
            'uncertainty': 'Contact not directly measured.', 'working_note': 'Object motion unverified.'}


def test_blanket_failure_is_not_correct_for_a_visible_motion_control():
    expected = {'motion_labels': ['moved'], 'pose_update_basis': 'hand_prediction',
                'task_success': False, 'next_step_kinds': ['inspect_existing']}
    scored = score_labels(decision(), expected)
    assert not scored['checks']['motion_discrimination']
    assert not scored['all_label_checks_passed']


def test_private_contact_loss_does_not_require_unjustified_contact_certainty():
    expected = {'motion_labels': ['no_visible_motion'], 'pose_update_basis': 'hand_prediction',
                'task_success': False, 'next_step_kinds': ['inspect_existing'], 'contact_loss_case': True}
    assert score_labels(decision(), expected)['all_label_checks_passed']
    value = decision()
    value['contact_assessment'] = 'retained'
    assert not score_labels(value, expected)['all_label_checks_passed']


def test_native_success_requires_no_further_request():
    value = decision()
    value.update(pose_update_basis='none', task_success=True)
    value['next_step'] = {'kind': 'stop_success', 'request': {'operation': 'execute', 'arguments': {}}, 'description': 'Done'}
    expected = {'motion_labels': ['no_visible_motion'], 'pose_update_basis': 'none',
                'task_success': True, 'next_step_kinds': ['stop_success']}
    assert not score_labels(value, expected)['all_label_checks_passed']


def test_malformed_answer_does_not_pass_schema(tmp_path):
    path = tmp_path / 'decision.json'
    value = decision()
    path.write_text(json.dumps(value))
    assert validate_decision(path) == value
    value['task_success'] = 'false'
    path.write_text(json.dumps(value))
    with pytest.raises(Exception):
        validate_decision(path)
