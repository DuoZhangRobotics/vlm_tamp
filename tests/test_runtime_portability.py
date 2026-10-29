from pathlib import Path

from vlm_tamp.runtime_paths import simulator_python
from vlm_tamp.subscription_agent import private_home_denials


def test_explicit_simulator_interpreter_takes_precedence(monkeypatch):
    monkeypatch.setenv('VLM_TAMP_SIM_PYTHON', '/configured/simulator/python')
    assert simulator_python() == '/configured/simulator/python'
    assert simulator_python('/explicit/python') == '/explicit/python'


def test_standalone_launcher_is_accessible_without_exposing_private_siblings(tmp_path):
    home = tmp_path / 'agent-home'
    runtime = home / 'packages/standalone/releases/current-version/bin'
    runtime.mkdir(parents=True)
    (runtime / 'codex').write_text('runtime')
    private = [home / 'auth.json', home / 'history.jsonl',
               home / 'packages/another-package/private.txt',
               runtime.parent / 'private.txt', home / 'tmp/other/private.txt']
    for path in private:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('canary')
    denied = private_home_denials(home, [runtime, home / 'tmp/arg0-linux'])
    assert not any((runtime / 'codex').is_relative_to(Path(path)) for path in denied)
    assert all(any(path.is_relative_to(Path(parent)) for parent in denied) for path in private)


def test_missing_optional_runtime_directories_do_not_break_profile(tmp_path):
    home = tmp_path / 'agent-home'
    home.mkdir()
    (home / 'auth.json').write_text('canary')
    assert private_home_denials(home, [home / 'bin', home / 'tmp/arg0-linux']) == {
        str(home / 'auth.json'): 'deny'}
