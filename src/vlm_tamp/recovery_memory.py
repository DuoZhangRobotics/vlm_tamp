"""Freeze agent-authored conditional lessons, excluding recorded geometry."""
import hashlib
import json
import re
from pathlib import Path


def abstract_lessons(document):
    if not isinstance(document,dict) or not document.get('principles'):
        raise ValueError('guidelines.json must contain a nonempty principles list')
    if not isinstance(document['principles'],list):
        raise ValueError('principles must be a list')
    def clean(value):
        if isinstance(value,dict):
            return {key:clean(item) for key,item in value.items()
                    if key not in {'references','evidence_paths','source_path'}}
        if isinstance(value,list):
            if value and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in value):
                raise ValueError('Historical numeric arrays are not transferable memory')
            return [clean(item) for item in value]
        if isinstance(value,str):
            if re.search(r'\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d',value):
                raise ValueError('Historical coordinate arrays must be removed from prose')
            if re.search(r'(?:robot_data/|replies/|requests/|/home/|/tmp/|\.np[yz]\b)',value):
                raise ValueError('Raw historical evidence links belong in findings, not frozen memory')
        return value
    return clean(document)


def freeze_lessons(run):
    run=Path(run)
    path=run/'agent-workspace/guidelines.json'
    source=json.loads(path.read_text())
    episodes=json.loads((run/'evaluation-private.json').read_text())['attempts']
    return {
        'origin':'Acting Luna agent; no human-written recovery strategy added',
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope':'Development on one initial state; transfer remains untested',
        'observed_attempts':[{'attempt':r['attempt'],'native_success':r['official_task_success'],
                             'control_steps':r['control_steps']} for r in episodes],
        'representation':'Conditional prose and tested relative choices; no recorded images, poses, pixels or trajectories',
        'method':abstract_lessons(source),
    }
