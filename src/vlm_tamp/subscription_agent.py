"""Launch a fresh ChatGPT-authenticated Codex experiment with restricted reads.

The simulator lives outside this process. Only a public workspace and numerical
RPC interface are made available to the research agent.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess


def validate_attempt_budget(attempts, *, learning_practice=False, evaluation_case=False, initial_attempts=None):
    """Repeated development practice is explicit and cannot be a frozen evaluation."""
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError('Use an attempt budget between 1 and 20')
    if learning_practice and (evaluation_case or attempts < 2):
        raise ValueError('Learning practice requires multiple development attempts')
    if attempts > 2 and not learning_practice:
        raise ValueError('More than two attempts requires explicit learning practice')
    if initial_attempts is not None:
        if not learning_practice:
            raise ValueError('Adaptive attempt budgets require learning practice')
        from .practice_budget import PracticeBudget
        PracticeBudget(initial_attempts, attempts)


def clean_environment():
    env = os.environ.copy()
    for key in list(env):
        if key.endswith(('_API_KEY', '_TOKEN', '_SECRET')) or key in {
            'PYTHONPATH', 'PYTHONSTARTUP', 'BASH_ENV', 'ENV', 'LD_PRELOAD',
            'CODEX_THREAD_ID', 'CODEX_TURN_ID', 'CODEX_EXECUTION_ID',
        }:
            env.pop(key, None)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def permission_arguments(*, readonly_paths=()):
    # No :root read grant, no inherited workspace profile and no /tmp grant.
    # The session's one public working directory is the only writable subtree.
    binary = Path(shutil.which('codex') or '/usr/bin/codex').resolve()
    home = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))
    fs = {':minimal': 'read', ':workspace_roots': 'write',
          '/home/duo/.codex': 'deny',
          str(Path(__file__).resolve().parents[2]): 'deny',
          str(binary.parent): 'read', str(home/'tmp/arg0-linux'): 'read'}
    # Codex's platform-minimal reads can include its own home. Exclude each
    # history/credential/plugin entry while retaining only runtime binaries.
    for child in home.iterdir():
        if child.name not in {'bin', 'tmp'}:
            fs[str(child)] = 'deny'
    for child in (home/'tmp').iterdir():
        if child.name != 'arg0-linux':
            fs[str(child)] = 'deny'
    for path in readonly_paths:
        fs[str(Path(path).resolve())] = 'read'
    table = '{'+','.join(json.dumps(k)+'='+json.dumps(v) for k,v in fs.items())+'}'
    return [
        '-c', 'default_permissions="research-agent"',
        '-c', 'permissions.research-agent.filesystem='+table,
        '-c', 'permissions.research-agent.network.enabled=false',
    ]


def codex_arguments(workspace, output, *, model='gpt-6-luna', readonly_paths=()):
    return [shutil.which('codex') or 'codex', 'exec', '--ignore-user-config',
            '--ephemeral', '--skip-git-repo-check', '--json', '--color', 'never',
            '-C', str(Path(workspace).resolve()), '-m', model,
            '-c', 'model_reasoning_effort="high"', '-c', 'approval_policy="never"',
            '-c', 'forced_login_method="chatgpt"', '-c', 'project_doc_max_bytes=0',
            '-c', 'web_search="disabled"',
            '--disable', 'memories', '--disable', 'plugins', '--disable', 'apps',
            '--disable', 'multi_agent', '--disable', 'in_app_browser',
            '--disable', 'browser_use', '--disable', 'image_generation',
            '--disable', 'skill_search', '--enable', 'skip_host_skill_discovery',
            '--disable', 'shell_snapshot', *permission_arguments(readonly_paths=readonly_paths),
            '-o', str(Path(output).resolve()), '-']


def verify_isolation(workspace, forbidden_paths, *, readonly_paths=()):
    """Exercise the same OS profile with harmless reads before model access."""
    workspace = Path(workspace).resolve()
    canary = workspace/'allowed-canary.txt'
    canary.write_text('public workspace is readable\n')
    code = '''import json,pathlib,socket
paths=json.loads(__import__('sys').argv[1]); rows=[]
for p in paths:
 try:
  with open(p,'rb') as f: f.read(1)
  rows.append({'path':p,'readable':True})
 except (PermissionError,FileNotFoundError,NotADirectoryError):
  rows.append({'path':p,'readable':False})
print(json.dumps({'reads':rows,'canary':pathlib.Path('allowed-canary.txt').read_text().strip()}))
'''
    command = [shutil.which('codex') or 'codex', 'sandbox', '-P', 'research-agent',
               *permission_arguments(readonly_paths=readonly_paths), '-C', str(workspace),
               '/usr/bin/python3', '-c', code, json.dumps([str(p) for p in forbidden_paths])]
    run = subprocess.run(command, env=clean_environment(), text=True, capture_output=True, timeout=30)
    if run.returncode:
        raise RuntimeError('Isolation probe failed: '+run.stderr[-2000:])
    result = json.loads(run.stdout.strip().splitlines()[-1])
    if any(r['readable'] for r in result['reads']):
        raise RuntimeError('Restricted read unexpectedly succeeded: '+json.dumps(result))
    result['passed'] = True
    return result
