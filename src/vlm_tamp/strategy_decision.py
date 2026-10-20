"""Bounded declarative strategy decisions; model text never executes as code."""
import hashlib
import json
from pathlib import Path
import re
import time

from .coupled_storage import write_json
from .rekep_libero.api import request_arguments
from .rekep_libero.articulated_grounding import response_text

ROOT=Path(__file__).resolve().parents[2]
PROMPT=ROOT/'configs/prompts/strategy_feedback_v1.txt'
KEYS={'version','operation','contact_mode','approach_tilt_deg','motion_extent','evidence','expected_observation'}
MODES={'pinch':(None,), 'press_drag':(None,30,45,60),'push':(None,0,30,45)}


def parse_decision(text):
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip())
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!=KEYS or value['version']!='strategy-v1':
        raise ValueError('Invalid strategy decision fields')
    op=value['operation'];mode=value['contact_mode'];tilt=value['approach_tilt_deg'];extent=value['motion_extent']
    if not isinstance(op,str) or op not in {'try_contact','continue_motion','reobserve','finish'}:raise ValueError('Unsupported operation')
    if extent is not None and not isinstance(extent,str):raise ValueError('Motion extent must be a menu choice')
    for key,maximum in [('evidence',700),('expected_observation',500)]:
        if not isinstance(value[key],str) or not value[key].strip() or len(value[key])>maximum:
            raise ValueError('A bounded evidence and expectation record is required')
    if tilt is not None and type(tilt) is not int:raise ValueError('Tilt must be an integer menu choice or null')
    if op=='try_contact':
        if not isinstance(mode,str) or mode not in MODES or tilt not in MODES[mode] or extent not in {'probe','goal'}:
            raise ValueError('Contact mode, tilt and extent must be supported tool choices')
    elif op=='continue_motion':
        if mode is not None or tilt is not None or extent not in {'probe','goal'}:
            raise ValueError('Continuation changes only the motion extent')
    elif any(x is not None for x in [mode,tilt,extent]):
        raise ValueError('Observation and finish operations have no motion parameters')
    return value


class LunaStrategy:
    def __init__(self,ledger,*,allow_paid=False,prompt_path=PROMPT,parser=parse_decision):
        self.ledger=ledger;self.allow_paid=allow_paid;self.calls=0
        self.prompt_path=Path(prompt_path);self.parser=parser

    def choose(self,packet,image_path,folder,scene_id):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        prompt=self.prompt_path.read_text()+'\nCURRENT OBSERVATION AND TOOL HISTORY\n'+json.dumps(packet,sort_keys=True,allow_nan=False)
        (folder/'prompt.txt').write_text(prompt)
        arguments=request_arguments(image_path,prompt,32000)
        before=self.ledger.snapshot();started=time.perf_counter()
        try:
            raw=self.ledger.perform(arguments,scene_id,allow_paid=self.allow_paid)
        finally:
            after=self.ledger.snapshot();new=after['api_attempts']-before['api_attempts'];self.calls+=new
            write_json(folder/'api-accounting.json',{'api_run_id':self.ledger.run_id,
                'new_api_calls':new,'max_output_tokens':32000,'reasoning_effort':'high',
                'round_trip_seconds':time.perf_counter()-started,'automatic_api_retries':0,
                'run_spend_usd':after['run_conservative_spend_usd'],
                'remaining_run_budget_usd':after['remaining_run_budget_usd'],
                'request_sha256':hashlib.sha256(json.dumps(arguments,sort_keys=True).encode()).hexdigest()})
        write_json(folder/'response.json',raw);text=response_text(raw);(folder/'response.txt').write_text(text)
        if raw.get('status')!='completed':raise ValueError('Strategy response incomplete; no automatic retry')
        decision=self.parser(text);write_json(folder/'decision.json',decision);return decision


class EnumeratedStrategy:
    """Fixed order, same tools and decision budget, with no task-label lookup."""
    def __init__(self):self.index=0;self.calls=0

    def options(self,packet):
        return [('pinch',None),('press_drag',30),('press_drag',45),('press_drag',60),('push',0)]

    def choose(self,packet,image_path,folder,scene_id):
        options=self.options(packet)
        last=packet['history'][-1]['result'] if packet['history'] else {}
        if packet['observed_state']['estimated_goal_reached']:
            op='finish';mode=tilt=extent=None
        elif last.get('status')=='motion_completed' and last.get('tracking_accepted') and last.get('object_progress_delta_m',0)>.003:
            op='continue_motion';mode=tilt=None;extent='goal'
        elif self.index>=len(options):
            op='finish';mode=tilt=extent=None
        else:
            op='try_contact';mode,tilt=options[min(self.index,len(options)-1)];self.index+=1;extent='goal'
        result={'version':'strategy-v1','operation':op,'contact_mode':mode,'approach_tilt_deg':tilt,
            'motion_extent':extent,'evidence':'Fixed shared search order and measured progress.',
            'expected_observation':'Movement toward the sensor-estimated goal.'}
        write_json(Path(folder)/'decision.json',result)
        return parse_decision(json.dumps(result))


class HeuristicStrategy(EnumeratedStrategy):
    """Retain the existing geometry-based mode ordering as a stronger control."""
    def options(self,packet):
        geometry=packet['geometry_measurements']
        if packet['mechanism_estimate']['motion']=='close':modes=['push','pinch']
        elif geometry['measured_lower_clearance_m']<geometry['configured_palm_half_span_m']+.008:
            modes=['press_drag','pinch','push']
        else:modes=['pinch','press_drag','push']
        return [(mode,None) for mode in modes]
