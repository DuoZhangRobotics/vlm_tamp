"""Verify the actor's documented numerical capabilities inside its OS sandbox."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.public_analysis import (
    DEFAULT_BUNDLE, PROBE_CODE, command_environment, configuration_arguments,
    prepare_workspace, readonly_paths,
)
from vlm_tamp.subscription_agent import clean_environment, permission_arguments


def run(output, bundle):
    output.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='vlm-public-analysis-probe-'))
    result = {'model_calls': 0, 'workspace': str(work), 'passed': False}
    try:
        result['bundle'] = prepare_workspace(work, bundle)
        private = ROOT / 'src/vlm_tamp/hinged_contact_tools.py'
        (work / 'private-code-link').symlink_to(private)
        code = PROBE_CODE + '''
denied = []
for filename in sys.argv[1:]:
    try:
        Path(filename).read_bytes()
        denied.append(False)
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        denied.append(True)
assert all(denied), 'Private project code was unexpectedly readable'
print(json.dumps({'analysis': analysis_probe, 'private_paths_denied': denied}))
'''
        shell_command = shlex.join(['python3', '-c', code, str(private), str(work / 'private-code-link')])
        command = ['codex', 'sandbox', '-P', 'research-agent',
                   *permission_arguments(readonly_paths=readonly_paths(work)),
                   *configuration_arguments(work), '-C', str(work),
                   '/usr/bin/bash', '-lc', shell_command]
        (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60,
            env=command_environment(work, clean_environment()))
        (output / 'stdout.txt').write_text(completed.stdout)
        (output / 'stderr.txt').write_text(completed.stderr)
        result['returncode'] = completed.returncode
        if completed.returncode:
            raise RuntimeError('Public analysis sandbox preflight failed; inspect stderr.txt')
        result['probe'] = json.loads(completed.stdout.strip().splitlines()[-1])
        result['passed'] = True
        shutil.rmtree(work)
        result['temporary_workspace_removed'] = True
    finally:
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    run(args.output.resolve(), args.bundle.resolve())
