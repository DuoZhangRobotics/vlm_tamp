"""Two-point Luna-high grounding with frozen local-depth geometry on new views."""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .dataset import read_records
from .drawer_local_plane import compile_local_drawer, validate_local_blanks
from .drawer_plane_trial import check_protocol as check_parent, grade_local_drawer as grade_local_drawer, settings, support_mask, support_quality
from .experiment import BudgetExceeded, Ledger
from .libero import ROOT
from .providers import ProviderResponse
from .rgbd_landmark_trial import file_sha
from .rgbd_numericals import json_text
from .task_grounding_frontview import request_cost_reservation as parent_reservation


FIELDS=('source_uv','handle_uv','joint_type','motion','path_constraint')
CASE_IDS=('drawer_geometry_camera_left','drawer_geometry_camera_right',
          'drawer_geometry_yaw_plus20_open','drawer_geometry_yaw_minus20_open')


def load_config(path):
    config=json.loads(Path(path).read_text())
    if (config['parent_config']!='configs/drawer_plane_ablation.json'
            or config['dataset']!='data/rgbd-drawer-geometry-stress'
            or config['model']!='gpt-6-luna' or config['reasoning_effort']!='high'
            or config['service_tier']!='default' or config['image_detail']!='high'
            or type(config['max_requests']) is not int or config['max_requests']!=4
            or type(config['max_retries']) is not int or config['max_retries']!=0
            or type(config['max_output_tokens']) is not int or config['max_output_tokens']!=6000):
        raise ValueError('Use the approved two-point Luna-high stress test')
    parent=json.loads((ROOT/config['parent_config']).read_text())
    for key in ('geometry','local_plane','evaluation'):
        if json_text(config[key])!=json_text(parent[key]):raise ValueError('Frozen local-plane policy changed: '+key)
    budget=config['overall_budget_usd']; prior=config['prior_usage_cost_usd']; available=config['max_budget_usd']
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (budget,prior,available))
            or budget!=2. or prior!=parent['prior_usage_cost_usd'] or abs(available+prior-budget)>1e-12
            or available<=0 or config['requested_extension_m'] is not None):
        raise ValueError('Include all prior API spending under the same two-dollar ceiling')
    expected={'input_per_million':.1,'cached_input_per_million':.01,'cache_write_per_million':.125,'output_per_million':.5}
    if config['pricing']!=expected:raise ValueError('Use the verified standard Luna prices')
    if ([c['id'] for c in config['cases']]!=list(CASE_IDS)
            or any(c['example_id']!=c['id'] or c['task_type']!='open_middle_drawer' for c in config['cases'])):
        raise ValueError('Use the four predetermined camera/state interventions')
    return config


def blank_schema():
    point={'anyOf':[{'type':'array','items':{'type':'number','minimum':0,'maximum':1},'minItems':2,'maxItems':2},{'type':'null'}]}
    properties={key:copy.deepcopy(point) for key in ('source_uv','handle_uv')}
    for key,choices in {'joint_type':['prismatic','revolute','unresolved'],'motion':['pull','push','unresolved'],
                        'path_constraint':['prismatic_pull','none','unresolved']}.items():
        properties[key]={'type':'string','enum':choices}
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def inputs(config,case,record):
    if set(record['model_inputs'])!={'instruction','image'}:raise ValueError('Only instruction and image are model inputs')
    folder=ROOT/config['dataset']/case['example_id']
    original=(folder/'rgb.png').read_bytes()
    payload={'instruction':record['model_inputs']['instruction'],'task_type':'open_middle_drawer','blanks':list(FIELDS)}
    return original,payload


def request_arguments(config,case,record):
    original,payload=inputs(config,case,record)
    return {'model':config['model'],'instructions':(ROOT/config['prompt']).read_text(),
            'input':[{'role':'user','content':[{'type':'input_text','text':json_text(payload)},
                     {'type':'input_image','detail':config['image_detail'],
                      'image_url':'data:image/png;base64,'+base64.b64encode(original).decode()}]}],
            'text':{'format':{'type':'json_schema','name':'DrawerSemanticSeeds','schema':blank_schema(),'strict':True}},
            'reasoning':{'effort':config['reasoning_effort']},'service_tier':'default',
            'max_output_tokens':config['max_output_tokens'],'store':False}


def request_cost_reservation(config,args):
    return parent_reservation(config,args)


def response_text(raw):
    return ''.join(item['text'] for block in raw.get('output',[]) if block['type']=='message'
                   for item in block['content'] if item['type']=='output_text')


def observed_support_audit(config,case,numeric):
    folder=ROOT/config['dataset']/case['example_id']
    reference=json.loads((folder/'evaluation/reference.json').read_text())
    masks={k:np.load(folder/v['npy'],allow_pickle=False) for k,v in reference['masks'].items()}
    observed=numeric['local_plane_support']
    return {name:support_quality(support_mask(observed[key],masks['middle_panel'].shape),masks)
            for name,key in (('local_depth','inlier_pixel_xy'),('local_candidates','candidate_pixel_xy'),
                            ('local_rejected','rejected_candidate_pixel_xy'))}


def compile_response(config,case,record,raw):
    if raw.get('status')!='completed':
        truncated=raw.get('status')=='incomplete' and (raw.get('incomplete_details') or {}).get('reason')=='max_output_tokens'
        return {'status':'response_truncated' if truncated else 'invalid_output','choices':None,'blanks':None,
                'numericals':None,'metrics':None,'support_audit':None,'error_type':'IncompleteResponse'}
    try:
        choices=json.loads(response_text(raw))
        blanks=validate_local_blanks(copy.deepcopy(choices))
        folder=ROOT/config['dataset']/case['example_id']
        numeric=compile_local_drawer(blanks,{'instruction':record['model_inputs']['instruction']},
            np.load(folder/'depth_meters.npy',allow_pickle=False),json.loads((folder/'calibration.json').read_text()),settings(config))
        row={'status':'valid','choices':choices,'blanks':blanks,'numericals':numeric}
        # References are read after the sampler/fit has completed, without rescue.
        row['metrics']=grade_local_drawer(record,ROOT,case,row,config)
        row['support_audit']=observed_support_audit(config,case,numeric)
        return row
    except (ValueError,TypeError,KeyError) as error:
        return {'status':'invalid_output','choices':None,'blanks':None,'numericals':None,
                'metrics':None,'support_audit':None,'error_type':type(error).__name__}


def freeze(config_path):
    config=load_config(config_path)
    _,parent,_=check_parent(ROOT/config['parent_config'])
    destination=ROOT/config['frozen_protocol']
    if destination.exists():raise FileExistsError('Preserve existing stress protocol')
    records,_=read_records(ROOT/config['dataset']);records={r['example_id']:r for r in records}
    implementation=copy.deepcopy(parent['implementation_hashes'])
    for name in ('src/vlm_tamp/drawer_geometry_stress.py','scripts/prepare_drawer_geometry_stress.py',
                 'scripts/run_drawer_geometry_stress.py','scripts/report_drawer_geometry_stress.py',
                 'scripts/verify_drawer_geometry_stress.py'):
        implementation[name]=file_sha(ROOT/name)
    inputs_hashes={str(p.relative_to(ROOT)):file_sha(p) for p in sorted((ROOT/config['dataset']).rglob('*')) if p.is_file()}
    preserved=copy.deepcopy(parent['preserved_artifacts'])
    for folder in ('results/drawer-plane-ablation','results/drawer-plane-ablation-report'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file():preserved[str(p.relative_to(ROOT))]=file_sha(p)
    requests=[{'case_id':c['id'],'request_sha256':hashlib.sha256(json_text(request_arguments(config,c,records[c['example_id']])).encode()).hexdigest()}
              for c in config['cases']]
    protocol={'version':'drawer-geometry-stress-1.0','config':config,'requests':requests,
              'parent_protocol':parent['config']['frozen_protocol'],
              'parent_protocol_sha256':file_sha(ROOT/parent['config']['frozen_protocol']),
              'implementation_hashes':implementation,'input_hashes':inputs_hashes,'preserved_artifacts':preserved,
              'prompt_sha256':file_sha(ROOT/config['prompt']),'local_policy_unchanged':True,
              'reference_inputs_sent':False,'provided_inventory':False,'depth_sent_to_vlm':False,
              'real_capture_manifest_sha256':file_sha(ROOT/config['dataset']/'manifest.json'),
              'scope':config['scope']}
    destination.write_text(json_text(protocol)+'\n');return protocol


def check_protocol(config_path):
    config=load_config(config_path)
    check_parent(ROOT/config['parent_config'])
    protocol=json.loads((ROOT/config['frozen_protocol']).read_text())
    if json_text(config)!=json_text(protocol['config']):raise ValueError('Frozen stress controls changed')
    if file_sha(ROOT/protocol['parent_protocol'])!=protocol['parent_protocol_sha256']:raise ValueError('Parent protocol changed')
    for path,sha in {**protocol['implementation_hashes'],**protocol['input_hashes'],**protocol['preserved_artifacts'],
                     config['prompt']:protocol['prompt_sha256']}.items():
        if file_sha(ROOT/path)!=sha:raise ValueError('Frozen or prior artifact changed: '+path)
    records,_=read_records(ROOT/config['dataset'])
    return config,protocol,{r['example_id']:r for r in records}


def run(config_path,output,*,allow_paid=False,dry_run=False):
    config,protocol,records=check_protocol(config_path)
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new stress trial output directory')
    if not dry_run and not allow_paid:raise PermissionError('Actual Luna requests require --allow-paid')
    output.mkdir(parents=True)
    ledger=Ledger(config)
    manifest={'config':config,'provider':'openai','model':config['model'],'reasoning_effort':'high',
              'prompt_text':(ROOT/config['prompt']).read_text(),'frozen_protocol_sha256':file_sha(ROOT/config['frozen_protocol']),
              'dry_run':dry_run,'offline_mock':False,'model_input_inventory':False,'reference_answers_sent':False,
              'depth_sent_to_vlm':False,'crops_used':False,'segmentation_used_in_inference':False,
              'robot_actions':0,'planned_case_count':4,'ledger':ledger.as_dict()}
    rows=[{'case_id':c['id'],'example_id':c['example_id'],'condition':c['condition'],'model':config['model'],
           'status':'not_run','choices':None,'blanks':None,'numericals':None,'metrics':None,'support_audit':None,
           'attempts':0,'cost_usd':0.} for c in config['cases']]
    def save():
        manifest['ledger']=ledger.as_dict()
        manifest['cumulative_estimated_usage_cost_usd']=config['prior_usage_cost_usd']+ledger.actual_cost
        (output/'manifest.json').write_text(json_text(manifest)+'\n')
        (output/'predictions.jsonl').write_text(''.join(json_text(row)+'\n' for row in rows))
    prepared=[]
    for case,frozen in zip(config['cases'],protocol['requests']):
        args=request_arguments(config,case,records[case['example_id']])
        if frozen['case_id']!=case['id'] or frozen['request_sha256']!=hashlib.sha256(json_text(args).encode()).hexdigest():
            raise ValueError('Frozen stress request changed')
        folder=output/case['id'];folder.mkdir()
        original,payload=inputs(config,case,records[case['example_id']])
        (folder/'original.png').write_bytes(original)
        (folder/'payload.json').write_text(json_text(payload)+'\n')
        (folder/'request.json').write_text(json_text(args)+'\n')
        prepared.append((case,args,folder))
    save()
    if dry_run:return manifest
    from openai import OpenAI
    client=OpenAI(base_url='https://api.openai.com/v1',timeout=config['timeout_seconds'],max_retries=0)
    for index,(case,args,folder) in enumerate(prepared):
        row=rows[index];reservation=request_cost_reservation(config,args)
        try:ledger.admit(reservation)
        except BudgetExceeded as error:
            manifest['stopped']={'reason':'budget_admission','detail':str(error)};save();break
        row.update(status='api_failure',attempts=1)
        try:response=client.responses.create(**args)
        except Exception as error:
            ledger.unknown_cost_reservation+=reservation;row['error_type']=type(error).__name__
            manifest['stopped']={'reason':'api_failure','detail':'No retry or substitute inference'};save();break
        raw=response.model_dump(mode='json');(folder/'response.json').write_text(json_text(raw)+'\n')
        row.update(response_model=raw.get('model'),response_status=raw.get('status'))
        usage=response.usage
        row['cost_usd']=ledger.account(ProviderResponse(response.output_text,raw,
            usage.input_tokens if usage else 0,usage.output_tokens if usage else 0),reservation)
        row.update(compile_response(config,case,records[case['example_id']],raw))
        if row['numericals'] is not None:(folder/'numericals.json').write_text(json_text(row['numericals'])+'\n')
        save()
        print(case['id'],row['status'],'plane',row['numericals']['drawer_front_plane']['status'] if row['numericals'] else None,
              'estimated $'+format(row['cost_usd'],'.6f'),flush=True)
    return manifest
