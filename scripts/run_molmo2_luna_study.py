"""Share one explicit local Molmo2 worker across fresh Luna evaluation actors.

Launch under the existing study resource supervisor and 32 GiB systemd cgroup.
The worker is owned by this batch; robot services only connect to its endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def preflight(study: Path, socket_path: Path) -> None:
    """Exercise real pointing with both camera indices and grasp/hook descriptions."""
    from vlm_tamp.perception_tools import PerceptionTools
    from vlm_tamp.pointing_worker import PointingClient

    source = ROOT / ('results/luna-communication-rerun-20261102/trials/'
                     'baseline-r1-close_microwave-state017-600/'
                     'agent-workspace/robot_data/attempt01/obs001')
    work = study / 'pointing-preflight'
    capture = work / 'capture'
    capture.mkdir(parents=True)
    for camera in ['agentview', 'wrist']:
        shutil.copyfile(source / (camera + '.png'), capture / (camera + '.png'))
    tools = PerceptionTools({1: {'sensors': capture / 'not-needed.npz',
                                 'cameras': ['agentview', 'wrist'], 'attempt': 1}},
                            work / 'perception', work, pointer=PointingClient(socket_path))
    requests = [
        ('agentview', 'Point to the visible microwave door.'),
        ('wrist', 'Point to a visible part of the microwave door that a gripper could grasp.'),
        ('wrist', 'Point to a visible retaining edge, lip or handle that a finger could hook to pull the door.'),
    ]
    rows = [tools.point(i + 2, 1, camera, query) for i, (camera, query) in enumerate(requests)]
    (work / 'summary.json').write_text(json.dumps({
        'scope': 'Operator-written interface smoke queries, outside all Luna workspaces. '
                 'No depth reconstruction, IK or physical contact execution.',
        'requests': rows, 'queries_fed_to_evaluation_agents': False,
    }, indent=2) + '\n')
    tools.close()


def main() -> None:
    """Start, monitor and shut down exactly one model for the declared study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--cases-file', type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    if 'VLM_TAMP_CAPACITY_STATE' not in os.environ:
        raise RuntimeError('Run this driver through the existing resource supervisor')
    from supervise_luna_dashboard_study import wait_for_capacity
    wait_for_capacity(study, 'molmo2-worker-load')
    folder = study / 'pointing-worker'
    folder.mkdir()
    local = Path(tempfile.mkdtemp(prefix='vlm-molmo2-'))
    socket_path = local / 'point.sock'
    ready = folder / 'ready.json'
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'),
                   'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
                   'HF_MODULES_CACHE': '/tmp/molmo2-reviewed-modules',
                   'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
                   'OPENBLAS_NUM_THREADS': '4'}
    # Molmo2 uses its packaged CUDA 13 runtime; the simulator/planner use CUDA 12.9.
    for key in ['CUDA_HOME', 'CPATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'TORCH_CUDA_ARCH_LIST']:
        environment.pop(key, None)
    command = [str(ROOT / '.runtime/envs/molmo2/bin/python'), '-u', '-m',
               'vlm_tamp.pointing_worker', '--socket', str(socket_path),
               '--ready-file', str(ready), '--checkpoint', str(ROOT / '.runtime/models/Molmo2-8B')]
    runner = None
    with (folder / 'worker.log').open('x') as stream:
        worker = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while not ready.exists():
                if worker.poll() is not None:
                    raise RuntimeError('Molmo2 failed to load; inspect pointing-worker/worker.log')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Molmo2 worker startup did not finish within 600 seconds')
                time.sleep(.2)
            preflight(study, socket_path)
            env = {**os.environ, 'VLM_TAMP_POINTING_SOCKET': str(socket_path)}
            runner = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_luna_pipeline_validation.py'),
                '--study', str(study), '--stage', 'baseline', '--cases-file', str(args.cases_file.resolve())],
                cwd=ROOT, env=env)
            while runner.poll() is None:
                if worker.poll() is not None:
                    raise RuntimeError('Shared Molmo2 worker exited during the study; no replacement is started')
                time.sleep(1)
            if runner.returncode:
                raise subprocess.CalledProcessError(runner.returncode, runner.args)
        finally:
            if runner is not None and runner.poll() is None:
                runner.terminate()
                runner.wait(timeout=20)
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=20)
            (folder / 'exit.json').write_text(json.dumps({'worker_pid': worker.pid,
                'worker_returncode': worker.returncode, 'finished_at': time.time()}) + '\n')
            if socket_path.exists():socket_path.unlink()
            local.rmdir()


if __name__ == '__main__':
    main()
