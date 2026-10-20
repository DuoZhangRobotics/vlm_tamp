"""Read malformed tool logs without promoting them to valid operations."""
import json
from pathlib import Path


def operation_name(event):
    request=event.get('request',{})
    value=request.get('operation') if isinstance(request,dict) else None
    return value if isinstance(value,str) else f'<invalid operation: {type(value).__name__}>'


def native_attempt_outcome(folder):
    path=Path(folder)/'evaluation-private.json'
    attempts=json.loads(path.read_text()).get('attempts',[]) if path.exists() else []
    return {'native_evidence_available':bool(attempts),
        'native_success':bool(attempts and attempts[0].get('official_task_success')),
        'native_attempts_recorded':len(attempts),
        'outcome_source':str(path)}
