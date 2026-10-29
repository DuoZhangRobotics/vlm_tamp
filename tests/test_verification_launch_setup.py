"""Regress the nested launcher failure without starting a model or simulator."""
import importlib.util
import json
from pathlib import Path

import pytest


def prepared_driver(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'scripts/run_verification_prompt_study.py'
    spec = importlib.util.spec_from_file_location('verification_launch_setup', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / 'study'
    out.mkdir()
    (out / 'results.json').write_text(json.dumps({'completed_records': 24}))
    (out / 'physical-schedule-private.json').write_text(json.dumps([
        {'case_id': 'state000-baseline', 'state': 0, 'condition': 'baseline'}]))
    monkeypatch.setattr(module, 'check_freeze', lambda out: None)
    monkeypatch.setattr(module, 'auth_check', lambda: None)
    monkeypatch.setattr(module, 'require_storage_headroom', lambda root: None)
    monkeypatch.setattr(module, 'load_retainer', lambda: (lambda case: None))
    return module, out


def test_child_nonrecursive_episode_mkdir_has_a_parent(tmp_path, monkeypatch):
    module, out = prepared_driver(tmp_path, monkeypatch)

    class ProbeComplete(Exception):
        pass

    def child(command, **kwargs):
        target = Path(command[command.index('--output') + 1])
        # This is the real launcher's operation that previously failed.
        target.mkdir(exist_ok=False)
        raise ProbeComplete

    monkeypatch.setattr(module.subprocess, 'Popen', child)
    with pytest.raises(ProbeComplete):
        module.run_physical(out)
    assert (out / 'physical/state000-baseline').is_dir()


def test_existing_partial_episode_is_never_restarted(tmp_path, monkeypatch):
    module, out = prepared_driver(tmp_path, monkeypatch)
    (out / 'physical/state000-baseline').mkdir(parents=True)
    launched = []
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: launched.append(a))
    with pytest.raises(RuntimeError, match='Partial episode retained'):
        module.run_physical(out)
    assert launched == []
