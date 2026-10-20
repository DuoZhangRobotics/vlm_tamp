"""Versioned choices for displacement, posture and physical recontact."""
import json
from pathlib import Path
import re
from .strategy_decision import MODES, HeuristicStrategy
from .coupled_storage import write_json

PROMPT = Path(__file__).resolve().parents[2]/'configs/prompts/strategy_feedback_v2.txt'
KEYS = {'version','operation','contact_mode','approach_tilt_deg','motion_extent',
        'step_mm','posture_rank','retreat_mm','evidence','expected_observation'}

def parse_decision(text):
    value = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip()))
    if not isinstance(value,dict) or set(value)!=KEYS or value['version']!='strategy-v2':
        raise ValueError('Invalid recovery decision fields')
    op = value['operation']; mode = value['contact_mode']; tilt = value['approach_tilt_deg']
    extent = value['motion_extent']; step = value['step_mm']; rank = value['posture_rank']; retreat = value['retreat_mm']
    if not isinstance(op,str) or op not in {'try_contact','continue_motion','release_reposition','reobserve','finish'}:
        raise ValueError('Unsupported recovery operation')
    for key,maximum in [('evidence',700),('expected_observation',500)]:
        if not isinstance(value[key],str) or not value[key].strip() or len(value[key])>maximum:
            raise ValueError('Bounded evidence and expectation required')
    for key in ('approach_tilt_deg','step_mm','posture_rank','retreat_mm'):
        if value[key] is not None and type(value[key]) is not int: raise ValueError('Integer menu choices required')
    if rank is not None and rank not in range(6): raise ValueError('Posture rank must be 0..5 or null')
    if op in {'try_contact','continue_motion'}:
        if extent not in ('probe','goal') or step not in (5,10,25) or retreat is not None:
            raise ValueError('Motion needs a bounded distance and extent')
        if op=='try_contact':
            if not isinstance(mode,str) or mode not in MODES or tilt not in MODES[mode]:
                raise ValueError('Unknown contact mode or tilt')
        elif mode is not None or tilt is not None: raise ValueError('Continuation keeps the contact mode')
    elif op=='release_reposition':
        if retreat not in (20,40) or any(x is not None for x in (mode,tilt,extent,step)):
            raise ValueError('Release/reposition needs only retreat distance and optional posture rank')
    elif any(x is not None for x in (mode,tilt,extent,step,rank,retreat)):
        raise ValueError('Observation and finish have no motion parameters')
    return value

class RecoveryHeuristic(HeuristicStrategy):
    """Same existing mode ordering and default step, on the expanded tool API."""
    def choose(self,packet,image_path,folder,scene_id):
        result = super().choose(packet,image_path,folder,scene_id)
        result.update(version='strategy-v2', step_mm=25 if result['motion_extent'] else None,
                      posture_rank=None,retreat_mm=None)
        write_json(Path(folder)/'decision.json',result)
        return parse_decision(json.dumps(result))
