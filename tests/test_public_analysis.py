"""The public Python environment must not import private or mutable code."""
import json
from pathlib import Path

import pytest

from vlm_tamp.public_analysis import (
    command_environment, configuration_arguments, inventory, load_manifest,
    prepare_workspace,
)


def bundle_fixture(tmp_path):
    bundle = tmp_path / 'bundle'
    payload = bundle / 'payload/numpy'
    payload.mkdir(parents=True)
    (payload / '__init__.py').write_text('PUBLIC = True\n')
    manifest = {'python_major_minor': [3, 12], 'packages': {'numpy': 'test', 'scipy': 'test'},
                'files_sha256': inventory(bundle / 'payload')}
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    work = tmp_path / 'work'
    work.mkdir()
    return bundle, work


def test_private_pythonpath_is_replaced_not_appended(tmp_path):
    env = command_environment(tmp_path, {'PYTHONPATH': '/private/project/src', 'OTHER': 'kept'})
    assert env['PYTHONPATH'] == str(tmp_path / '.analysis_site')
    assert '/private' not in str(env)
    assert env['OTHER'] == 'kept'
    assert env['PYTHONNOUSERSITE'] == '1'
    assert 'shell_environment_policy.set=' in configuration_arguments(tmp_path)[1]


def test_workspace_copy_is_independent_and_tampering_is_rejected(tmp_path):
    bundle, work = bundle_fixture(tmp_path)
    prepare_workspace(work, bundle)
    target = work / '.analysis_site/numpy/__init__.py'
    target.write_text('CHANGED = True\n')
    assert (bundle / 'payload/numpy/__init__.py').read_text() == 'PUBLIC = True\n'
    with pytest.raises(ValueError, match='Workspace analysis libraries'):
        prepare_workspace(work, bundle)


def test_unregistered_dependency_file_is_rejected(tmp_path):
    bundle, _ = bundle_fixture(tmp_path)
    (bundle / 'payload/private.txt').write_text('must not be copied')
    with pytest.raises(ValueError, match='registered hashes'):
        load_manifest(bundle)


def test_bundle_symlink_cannot_expose_private_files(tmp_path):
    bundle, work = bundle_fixture(tmp_path)
    (bundle / 'payload/numpy/secret').symlink_to(Path('/etc/passwd'))
    with pytest.raises(ValueError, match='symlinks'):
        prepare_workspace(work, bundle)


def test_manifest_traversal_is_rejected(tmp_path):
    bundle, _ = bundle_fixture(tmp_path)
    path = bundle / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files_sha256']['../secret'] = 'bad'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsafe'):
        load_manifest(bundle)


def test_metadata_symlink_cannot_write_outside_workspace(tmp_path):
    bundle, work = bundle_fixture(tmp_path)
    outside = tmp_path / 'must-not-be-written.json'
    (work / 'analysis-environment.json').symlink_to(outside)
    with pytest.raises(ValueError, match='metadata must not be a symlink'):
        prepare_workspace(work, bundle)
    assert not outside.exists()
