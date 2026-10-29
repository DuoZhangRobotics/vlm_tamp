"""Build auditable, prefix-only robot decision packets without evaluator data.

This module never imports the simulator or sends a model request. Packet assets
are allowlisted from the public responses preceding the cutoff; later notes,
outcomes, case names and evaluator labels stay outside the packet.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil


PROMPT = """You are choosing the next operation in a recorded robot episode.
The task instruction, current observation and completed operation history are
in inputs/checkpoint.json and inputs/history.jsonl. Read inputs/tools.txt and
the supplied experience and example before deciding. The attached images are
the current front, agent, wrist and arm-side views, in that order. Other saved
views and public numerical artifacts are under inputs/robot_data/.

Choose one next operation from the documented robot interface to advance the
task from this point. This is an offline decision: the robot service is not
running, so do not execute robot requests or invent their results. You may
inspect files and write analysis code locally. Only this packet is available;
some historical sensor arrays were not retained, as listed in checkpoint.json.

Write decision.json containing next_request (an object with operation and
arguments), justification (a brief evidence-based explanation),
expected_observation (what you expect to learn or observe after that request),
and uncertainty (remaining uncertainty relevant to your choice). Cite event
numbers or public artifact names when useful. Keep the explanation concise.
Do not write a detailed reasoning transcript. Do not modify the input files.
"""

ATTACHED_CAMERAS = ('frontview', 'agentview', 'wrist', 'arm_side')
SHARED_FILES = ('tools.txt', 'experience/guidelines.json', 'example/readme.txt')
ASSET_SUFFIXES = {'.png', '.jpg', '.jpeg', '.npy', '.npz', '.json', '.txt'}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def public_references(value):
    """Only exact, normalized public filenames can grant a packet asset."""
    found = set()
    if isinstance(value, dict):
        for item in value.values():
            found.update(public_references(item))
    elif isinstance(value, list):
        for item in value:
            found.update(public_references(item))
    elif isinstance(value, str) and value.startswith('robot_data/'):
        path = PurePosixPath(value)
        if ('..' in path.parts or str(path) != value or '\\' in value
                or path.suffix not in ASSET_SUFFIXES):
            raise ValueError('Unsafe or unsupported public artifact reference')
        found.add(value)
    return found


def prefix_events(path, cutoff):
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError('Use a positive integer event cutoff')
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['sequence'] > cutoff:
                break
            # Measured call duration and all evaluator sidecar fields are absent.
            rows.append({key: row[key] for key in ('sequence', 'request', 'response')})
    previous = 0
    for row in rows:
        sequence = row['sequence']
        if sequence != previous + 1:
            # execute_sequence assigns one artifact ID per stage, but the RPC
            # is logged only once, with its final stage ID. Accept exactly the
            # gap evidenced by those stages, never an arbitrary missing event.
            stages = row['response'].get('result', {}).get('stages', [])
            stage_ids = [stage.get('artifact_id') for stage in stages]
            explained = (row['request']['operation'] == 'execute_sequence'
                and row['response'].get('ok') and 1 <= len(stages) <= 3
                and stage_ids == list(range(previous + 1, sequence + 1)))
            if not explained:
                raise ValueError('Checkpoint requires a complete ordered event prefix')
        previous = sequence
    if not rows or rows[-1]['sequence'] != cutoff:
        raise ValueError('Checkpoint requires a complete ordered event prefix')
    return rows


def latest_observation(rows):
    for row in reversed(rows):
        response = row['response']
        if not response.get('ok'):
            continue
        result = response.get('result', {})
        observation = result.get('observation', result)
        if 'images' in observation:
            return observation
    raise ValueError('No public observation in checkpoint prefix')


def permitted_asset(case, relative):
    """Restore only a referenced asset, never copy a whole retained directory."""
    case = Path(case).resolve()
    source = case / 'agent-workspace' / relative
    if source.is_file():
        if not source.resolve().is_relative_to(case):
            raise ValueError('Public asset resolves outside its source case')
        return source
    match = re.fullmatch(r'robot_data/(attempt\d+/obs\d+/(?:sensors\.npz|calibration\.json))', relative)
    if match:
        source = case / 'failure-observations' / match.group(1)
        if source.is_file():
            if not source.resolve().is_relative_to(case):
                raise ValueError('Retained asset resolves outside its source case')
            return source
    return None


def build_packet(case, cutoff, destination, *, shared_sources=None):
    """Return private provenance separately from the public packet."""
    case, destination = Path(case).resolve(), Path(destination).resolve()
    shared_sources = dict(shared_sources or {})
    if set(shared_sources) - set(SHARED_FILES):
        raise ValueError('Only declared shared inputs may be supplied explicitly')
    rows = prefix_events(case / 'tool-events.jsonl', cutoff)
    observation = latest_observation(rows)
    references = public_references(rows)
    destination.mkdir(parents=True, exist_ok=False)
    inventory, missing = {}, []
    queue = sorted(references)
    # Recursively include references from public JSON artifacts. A reference to
    # a future call is rejected even if a corrupt saved artifact names it.
    seen = set()
    while queue:
        relative = queue.pop(0)
        if relative in seen:
            continue
        seen.add(relative)
        for part in PurePosixPath(relative).parts:
            match = re.match(r'(?:obs|geometry|matches|scene|plan|grasp|p)(\d+)', part)
            if match and int(match.group(1)) > cutoff:
                raise ValueError('Public artifact references a future call')
        source = permitted_asset(case, relative)
        if source is None:
            missing.append(relative)
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)  # Never expose a writable evidence hardlink.
        inventory[relative] = {'sha256': sha256(target), 'bytes': target.stat().st_size,
                               'source': str(source), 'referenced_by_prefix': True}
        if target.suffix == '.json':
            queue.extend(sorted(public_references(read_json(target)) - seen))
    for relative in SHARED_FILES:
        supplied = relative in shared_sources
        source = Path(shared_sources[relative]).resolve() if supplied else case / 'agent-workspace' / relative
        if not source.is_file() or (not supplied and not source.resolve().is_relative_to(case)):
            raise ValueError('Shared input missing or outside source case: ' + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory[relative] = {'sha256': sha256(target), 'bytes': target.stat().st_size,
                               'source': str(source), 'shared_original_input': not supplied,
                               'explicitly_supplied_shared_input': supplied}
    with (destination / 'history.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    instruction = next(row['response']['result']['instruction'] for row in rows
                       if row['request']['operation'] == 'reset' and row['response'].get('ok'))
    current_images = [observation['images'][camera] for camera in ATTACHED_CAMERAS]
    if any(not (destination / path).is_file() for path in current_images):
        raise ValueError('Current image required by the registered packet is unavailable')
    write_json(destination / 'checkpoint.json', {
        'task_instruction': instruction, 'last_completed_event': cutoff,
        'control_steps': observation['control_steps'],
        'remaining_control_steps': observation['remaining_control_steps'],
        'current_observation': observation,
        'attached_images_in_order': dict(zip(ATTACHED_CAMERAS, current_images)),
        'missing_historical_assets': sorted(missing),
        'scope': 'Recorded public tool history and retained observations only; no live robot service.',
    })
    packet_hashes = {str(path.relative_to(destination)): sha256(path)
                     for path in sorted(destination.rglob('*')) if path.is_file()}
    return {'source_case': str(case), 'cutoff': cutoff,
            'source_events_sha256': sha256(case / 'tool-events.jsonl'),
            'packet_sha256': packet_hashes, 'asset_provenance': inventory,
            'missing_assets': sorted(missing), 'current_images': current_images,
            'physical_recovery_test': False, 'continuing_original_agent_context': False}


def verify_packet(packet, manifest):
    packet = Path(packet)
    actual = {str(p.relative_to(packet)): sha256(p) for p in packet.rglob('*') if p.is_file()}
    if actual != manifest['packet_sha256']:
        raise ValueError('Packet contents differ from their registered hashes')
    return {'passed': True, 'files': len(actual),
            'bytes': sum(p.stat().st_size for p in packet.rglob('*') if p.is_file())}


def parse_decision(path):
    value = read_json(path)
    required = {'next_request', 'justification', 'expected_observation', 'uncertainty'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('Decision must contain exactly the four requested fields')
    request = value['next_request']
    if (not isinstance(request, dict) or set(request) != {'operation', 'arguments'}
            or not isinstance(request['operation'], str)
            or not isinstance(request['arguments'], dict)):
        raise ValueError('Invalid next_request structure')
    if any(not isinstance(value[key], str) or not value[key].strip()
           for key in required - {'next_request'}):
        raise ValueError('Decision notes must be nonempty text')
    return value
