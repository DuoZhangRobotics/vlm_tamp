"""Fresh Luna/Molmo2 reference selection from recorded public observations only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from prepare_subscription_pilot import CLIENT
from report_subscription_pilot import archive_workspace
from serve_subscription_robot import handle_request
from vlm_tamp.goal_gap import landmark_sample
from vlm_tamp.perception_tools import PerceptionTools
from vlm_tamp.pointing_worker import PointingClient
from vlm_tamp.subscription_agent import clean_environment, codex_arguments, verify_isolation

PROMPT = '''The recorded robot task is to close a microwave door. This is a read-only goal-reference experiment, with no robot actions.
Choose a visible material point on the moving free edge of the door and the matching visible point on the fixed appliance opening where that edge should meet when closed. A hinge point alone cannot distinguish open from closed. A protruding handle need not meet the appliance frame; choose references whose relationship actually expresses closure.
Read tools.txt and inspect both original camera images for at least one capture. Other saved captures from these same two cameras are available through observe; you may select a different capture/camera for each reference if that gives visible evidence. This is an offline reference-selection test, not a claim that later views were available at the episode start.
Use the local point tool to request proposals for both semantic roles, then inspect its returned overlays. You may reject or adjust a proposal yourself, recording why. You may use sample_points to inspect recorded depth support. You receive no simulator object state, hinge/closing direction, expected coordinates, old notebooks, or measured gap trends.
Pick references from visible scene evidence, not a hypothetical hidden pixel or assumed world axis. Do not provide a push direction. If no reliable pair is visible, say so. You may retain a tentative visible pair while marking can_select=false and explaining the ambiguity; leave a reference null if it cannot be located.
Write references.json with this structure:
{"can_select":true|false,"moving":{"observation_id":1,"camera":"agentview|wrist","pixel_xy":[x,y],"description":"physical feature","selection_source":"Molmo2 proposal or your own selection","evidence":"why this belongs to the moving free edge"}|null,"fixed_closed":{"observation_id":1,"camera":"agentview|wrist","pixel_xy":[x,y],"description":"physical feature","selection_source":"Molmo2 proposal or your own selection","evidence":"why this belongs to the fixed matching closing reference"}|null,"why_the_pair_expresses_closure":"visible relationship","uncertainties":"limitations"}.
Coordinates must be integer ORIGINAL image pixels (x right, y down). Freeze your final choices in that file; numerical tracking and gap measurement happen after your session ends. Stay within this public workspace.
'''


def write(path: Path, value: object) -> None:
    """Save a small explicit experiment record."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class RecordedService:
    """Use the established mailbox transport, exposing only observation/point/depth reads."""

    def __init__(self, source: Path, work: Path, output: Path, endpoint: Path) -> None:
        self.workspace, self.output = work, output
        self.public = work / 'robot_data'
        self.public.mkdir()
        self.calls = 0
        self.records = {}
        self.catalog = []
        provenance = []
        for capture in sorted(source.glob('obs*')):
            identifier = int(capture.name[3:])
            folder = work / 'captures' / capture.name
            folder.mkdir(parents=True)
            for name in ['agentview.png', 'wrist.png', 'calibration.json', 'robot.json']:
                original = capture / name
                shutil.copyfile(original, folder / name)
                provenance.append({'source': str(original), 'file': str((folder/name).relative_to(work)),
                                   'sha256': hashlib.sha256(original.read_bytes()).hexdigest()})
            step = json.loads((folder / 'robot.json').read_text())['control_steps']
            self.records[identifier] = {'sensors': folder / 'sensors-not-exposed.npz',
                'source_sensors': capture / 'sensors.npz', 'calibration': folder / 'calibration.json',
                'cameras': ['agentview', 'wrist'], 'attempt': 1, 'control_steps': step}
            self.catalog.append({'observation_id': identifier, 'control_step': step})
        self.perception = PerceptionTools(self.records, self.public / 'pointing', work,
                                          pointer=PointingClient(endpoint))
        write(output / 'input-provenance.json', provenance)
        write(work / 'captures.json', self.catalog)
        write(self.public / 'episode-status.json', self.episode_status())

    def episode_status(self) -> dict:
        """Describe the read-only experiment without implying a robot episode."""
        return {'mode': 'recorded_reference_selection', 'robot_actions_available': False,
                'available_observations': self.catalog}

    def dispatch(self, operation: str, arguments: dict) -> dict:
        """Forward model requests unchanged and expose literal saved depth samples."""
        self.calls += 1
        if operation == 'observe':
            identifier = arguments['observation_id']
            record = self.perception._record(identifier)
            folder = Path(record['sensors']).parent
            return {'observation_id': identifier, 'control_step': record['control_steps'],
                    'images': {name: str((folder/(name+'.png')).relative_to(self.workspace))
                               for name in ['agentview', 'wrist']}}
        if operation == 'point':
            return self.perception.point(self.calls, **arguments)
        if operation == 'sample_points':
            record = self.perception._record(arguments['observation_id'])
            camera = arguments['camera']
            if camera not in record['cameras']:
                raise ValueError('Choose agentview or wrist from the recorded capture')
            view = None
            if record['source_sensors'].exists():
                with np.load(record['source_sensors'], allow_pickle=False) as arrays:
                    view = {key: arrays[camera+'_'+key] for key in
                            ['depth', 'world_points', 'known_robot_depth_prior']}
            return {'observation_id': arguments['observation_id'], 'camera': camera,
                    'samples': [landmark_sample(view, p) for p in arguments['points_xy']],
                    'scope': 'Saved depth and known-robot support only; no object identity or desired motion.'}
        raise ValueError('Only observe, point and sample_points are available. No robot actions or gap scoring.')


@contextmanager
def model_worker(study: Path) -> Iterator[Path]:
    """Keep one explicitly selected Molmo2 worker alive for reference selection."""
    folder = study / 'model-worker'
    folder.mkdir()
    local = Path(tempfile.mkdtemp(prefix='vlm-goal-pointing-'))
    endpoint = local / 'point.sock'
    ready = folder / 'ready.json'
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'HF_HUB_OFFLINE': '1',
           'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_MODULES_CACHE': '/tmp/molmo2-reviewed-modules',
           'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    for key in ['CUDA_HOME', 'CPATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'TORCH_CUDA_ARCH_LIST']:
        env.pop(key, None)
    with (folder / 'worker.log').open('x') as log:
        process = subprocess.Popen([str(ROOT / '.runtime/envs/molmo2/bin/python'), '-u', '-m',
            'vlm_tamp.pointing_worker', '--socket', str(endpoint), '--ready-file', str(ready),
            '--checkpoint', str(ROOT / '.runtime/models/Molmo2-8B')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while not ready.exists():
                if process.poll() is not None:
                    raise RuntimeError('Molmo2 startup failed; inspect model-worker/worker.log')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Molmo2 startup timed out')
                time.sleep(.2)
            yield endpoint
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
            write(folder / 'exit.json', {'pid': process.pid, 'returncode': process.returncode})
            if endpoint.exists():endpoint.unlink()
            local.rmdir()


def select_references(study: Path, state: int, endpoint: Path) -> None:
    """Give one fresh Luna agent the recordings and freeze its own final references."""
    source_run = ROOT / (f'results/luna-molmo2-integration-v3-20261103/trials/'
                         f'baseline-r1-close_microwave-state{state:03d}-600')
    output = study / f'state{state:03d}'
    output.mkdir()
    work = Path(tempfile.mkdtemp(prefix='vlm-goal-references-'))
    service = RecordedService(source_run / 'agent-workspace/robot_data/attempt01', work, output, endpoint)
    for name in ['requests', 'replies']:(work/name).mkdir()
    (work / 'robot.py').write_text(CLIENT)
    (work / 'tools.txt').write_text('''Recorded observations only; no robot motion is available.
Read captures.json for observation IDs and control timestamps. Both cameras are 512x512 original pixels.
python3 robot.py observe '{"observation_id":1}' returns saved image paths.
python3 robot.py point '{"observation_id":1,"camera":"agentview","query":"your description"}' calls local Molmo2. Inspect the returned annotated image. Proposals are not semantic guarantees.
python3 robot.py sample_points '{"observation_id":1,"camera":"agentview","points_xy":[[x,y]]}' reads literal recorded XYZ/depth support. It does not snap a point or identify a closed configuration.
Use the point tool for each reference role. You decide which proposal or explicitly chosen pixel to retain. Different cameras/captures are allowed for the two roles; camera calibration places samples in a common world frame.
''')
    isolation = verify_isolation(work, [source_run/'evaluation-private.json', source_run/'agent-workspace/findings.txt',
        source_run/'attempt01/initial/evaluation_contact_diagnostics.json', ROOT/'src/vlm_tamp/contact_skills.py',
        Path.home()/'.codex/sessions'], readonly_paths=[work/'captures'])
    write(output / 'isolation.json', isolation)
    (output / 'prompt.txt').write_text(PROMPT)
    command = codex_arguments(work, output/'agent-last.txt', model='gpt-6-luna', reasoning='high',
                              readonly_paths=[work/'captures'], ephemeral=False)
    write(output / 'agent-command.json', command)
    start = time.monotonic()
    with (output/'agent-events.jsonl').open('x') as log, (output/'agent-stderr.log').open('x') as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True, cwd=work,
                                   env=clean_environment(), stdout=log, stderr=errors)
        try:
            process.stdin.write(PROMPT)
            process.stdin.close()
            handled = set()
            while process.poll() is None:
                if time.monotonic() - start > 600:
                    raise TimeoutError('Reference-selection session exceeded its 600-second allowance')
                for request in sorted((work/'requests').glob('*.json'), key=lambda p: p.stat().st_mtime):
                    if request.name not in handled:
                        handle_request(service, request, work/'replies')
                        handled.add(request.name)
                time.sleep(.1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            references = json.loads((work/'references.json').read_text())
            write(output/'references.json', references)
            write(output/'reference-freeze.json', {'sha256': hashlib.sha256((work/'references.json').read_bytes()).hexdigest(),
                'frozen_before_gap_measurement': True, 'native_state_supplied': False,
                'gap_trends_supplied': False, 'robot_motion_executed': False})
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
            write(output/'agent-exit.json', {'returncode': process.returncode,
                'wall_seconds': time.monotonic()-start, 'robot_motion_executed': False})
            archive_workspace(work, output/'agent-workspace')
            service.perception.close()
            shutil.rmtree(work)
    print(json.dumps({'state': state, 'references': references}), flush=True)


def main() -> None:
    """Declare the protocol before independent reference selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    study.mkdir(parents=True, exist_ok=True)
    from supervise_luna_dashboard_study import wait_for_capacity
    wait_for_capacity(study, 'recorded-goal-reference-selection')
    write(study/'protocol.json', {
        'states': [17, 49], 'actor': 'gpt-6-luna', 'reasoning': 'high', 'fresh_agent_per_state': True,
        'pointing_backend': 'allenai/Molmo2-8B', 'pointing_revision': 'e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b',
        'selection_seconds_per_actor': 600, 'selection_views': 'All saved captures from agentview and wrist only',
        'references_frozen_before_measurement': True, 'expected_directions_supplied': False,
        'native_object_state_supplied': False, 'prior_notebooks_supplied': False,
        'tracking_backend': 'CoTracker3 scaled_offline', 'agentview_rgb_frames_maximum': 64,
        'wrist_tracking': 'Saved wrist RGB captures; no wrist video is recorded',
        'depth_policy': 'Literal nearest-pixel measured depth at capture timestamps; no invented/interpolated depth',
        'gap_definition': 'Euclidean distance between current tracked moving and fixed-reference samples',
        'scope': 'Offline recorded-data feasibility test. Later views and backward tracking are allowed; not an online-policy success claim.',
        'no_robot_motion': True, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
    })
    with model_worker(study) as endpoint:
        for state in [17, 49]:select_references(study, state, endpoint)
    write(study/'selection-complete.json', {'complete': True})


if __name__ == '__main__':main()
