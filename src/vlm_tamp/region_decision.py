"""Typed RGB-region revision layered onto the existing contact tool menu."""
import json
from pathlib import Path
import re
import math
from .recovery_decision import KEYS as OLD_KEYS,parse_decision as parse_old

PROMPT=Path(__file__).resolve().parents[2]/'configs/prompts/strategy_feedback_v3.txt'
REGION_KEYS={'camera','contact_uv','contact_box_uv','moving_panel_box_uv','contact_kind'}

def validate_region(value):
    if not isinstance(value,dict) or set(value)!=REGION_KEYS:raise ValueError('Region fields do not match the declared schema')
    if value['camera'] not in ('agentview','frontview') or value['contact_kind'] not in ('bar','surface'):
        raise ValueError('Unknown camera or contact kind')
    for key,length in (('contact_uv',2),('contact_box_uv',4),('moving_panel_box_uv',4)):
        x=value[key]
        if not isinstance(x,list) or len(x)!=length or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in x):
            raise ValueError('Use finite camera-local normalized coordinates in [0,1]')
        if length==4 and (x[2]-x[0]<.01 or x[3]-x[1]<.01):raise ValueError('Region box has insufficient area')
    u,v=value['contact_uv'];a,b,c,d=value['contact_box_uv']
    if not a<=u<=c or not b<=v<=d:raise ValueError('Contact point must lie inside its contact box')
    return value

def parse_decision(text):
    value=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()))
    if not isinstance(value,dict) or set(value)!=OLD_KEYS|{'region'} or value['version']!='strategy-v3':
        raise ValueError('Invalid region-strategy fields')
    old={k:v for k,v in value.items() if k!='region'};old['version']='strategy-v2'
    if value['operation']=='revise_region':
        validate_region(value['region']);old['operation']='reobserve'
    elif value['region'] is not None:raise ValueError('Only revise_region accepts region coordinates')
    parse_old(json.dumps(old))
    return value

def pixels(region,width,height):
    """Camera-local normalization, never whole-sheet coordinate correction."""
    validate_region(region)
    point=[int(round(region['contact_uv'][0]*(width-1))),int(round(region['contact_uv'][1]*(height-1)))]
    def box(key):return [int(round(v*([width-1,height-1]*2)[i])) for i,v in enumerate(region[key])]
    return point,box('contact_box_uv'),box('moving_panel_box_uv')
