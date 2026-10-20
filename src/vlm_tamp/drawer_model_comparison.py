"""Matched high-reasoning models over frozen RGB drawer inputs and geometry."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .drawer_constraints import compile_drawer
from .experiment import BudgetExceeded, Ledger
from .libero import ROOT
from .providers import ProviderResponse
from .rgbd_numericals import json_text
from .task_grounding_frontview import (
    check_protocol as check_parent, file_sha, grade_drawer as grade_drawer, inputs, resolve_choices,
    request_cost_reservation as parent_reservation, sdk_arguments,
)

MODEL_IDS = {'luna':'gpt-6-luna','astra':'gpt-6-astra'}
PRICES = {'luna':{'input_per_million':.1,'cached_input_per_million':.01,
                  'cache_write_per_million':.125,'output_per_million':.5},
          'astra':{'input_per_million':10.,'cached_input_per_million':1.,
                   'cache_write_per_million':12.5,'output_per_million':50.}}
PRIOR_TRIALS = ['results/rgbd-dino-luna-comparison','results/task-grounding-scope-luna',
                'results/task-grounding-frontview-luna']


def load_config(path):
    config=json.loads(Path(path).read_text())
    if (config['models']!=MODEL_IDS or config['pricing_by_model']!=PRICES
            or config['reasoning_effort']!='high' or config['service_tier']!='default'
            or type(config['max_retries']) is not int or config['max_retries']!=0
            or type(config['max_requests']) is not int or config['max_requests']!=8
            or type(config['max_output_tokens']) is not int or config['max_output_tokens']!=6000):
        raise ValueError('Use the approved matched high-reasoning models and fixed request controls')
    parent=json.loads((ROOT/config['parent_config']).read_text())
    for key in ('dataset','depth_dataset','candidate_dataset','prompt','image_detail','geometry',
                'image_token_allowance','requested_extension_m','evaluation'):
        if json_text(config[key])!=json_text(parent[key]):
            raise ValueError('Frozen model-comparison input changed: '+key)
    if config['prior_usage_trials']!=PRIOR_TRIALS:
        raise ValueError('Include all three prior trial ledgers')
    prior,unknown=0.,0.
    for folder in PRIOR_TRIALS:
        ledger=json.loads((ROOT/folder/'manifest.json').read_text())['ledger']
        prior+=ledger['estimated_usage_cost_usd']
        unknown+=ledger['unknown_usage_reserved_usd']
    if unknown:
        raise ValueError('Prior unknown charges must be resolved before a new comparison')
    budget=config['overall_budget_usd']; available=config['max_budget_usd']; declared=config['prior_usage_cost_usd']
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (budget,available,declared))
            or not 0<budget<=2 or available<=0 or abs(prior-declared)>1e-12
            or abs(budget-prior-available)>1e-12):
        raise ValueError('All model usage must share the authorized two-dollar ceiling')
    expected=[(key,f'libero_goal-00-state{state:03d}-seed0')
              for state in (0,10,20,30) for key in MODEL_IDS]
    if (len(config['cases'])!=8
            or [(c['model_key'],c['example_id']) for c in config['cases']]!=expected
            or any(c['model']!=MODEL_IDS[c['model_key']] or c['condition']!='direct_rgb'
                   or c['task_type']!='open_middle_drawer' for c in config['cases'])):
        raise ValueError('Use four matched direct-RGB scenes per model')
    identifiers=[c['id'] for c in config['cases']]
    if (len(set(identifiers))!=8 or any(not isinstance(v,str) or not v or v in {'.','..'}
                                     or Path(v).name!=v for v in identifiers)):
        raise ValueError('Case IDs must be unique safe directory names')
    return config


def model_config(config,case):
    key=case['model_key']
    if key not in config['models']:
        raise ValueError('Undeclared comparison model')
    return {**config,'model':config['models'][key],'pricing':config['pricing_by_model'][key],'max_requests':4}


def request_arguments(config,case,record):
    settings=model_config(config,case)
    candidates,original,numbered,_,payload=inputs(config,case,record)
    return sdk_arguments(settings,case,payload,original,numbered,(ROOT/config['prompt']).read_text(),candidates)


def request_cost_reservation(config,args):
    keys=[key for key,value in config['models'].items() if value==args['model']]
    if len(keys)!=1:
        raise ValueError('Request must use exactly one declared priced model')
    return parent_reservation(model_config(config,{'model_key':keys[0]}),args)


class SharedLedger:
    """One admission budget across independently priced model usage ledgers."""
    def __init__(self,config):
        self.config=config
        self.ledgers={key:Ledger(model_config(config,{'model_key':key})) for key in config['models']}

    def admit(self,model_key,reservation):
        if isinstance(reservation,bool) or not isinstance(reservation,(int,float)) or not math.isfinite(reservation) or reservation<0:
            raise ValueError('Invalid cost reservation')
        if model_key not in self.ledgers:
            raise ValueError('Undeclared model')
        if sum(v.attempts for v in self.ledgers.values())>=self.config['max_requests']:
            raise BudgetExceeded('Maximum shared API attempts reached')
        used=sum(v.actual_cost+v.unknown_cost_reservation for v in self.ledgers.values())
        if self.config['prior_usage_cost_usd']+used+reservation>self.config['overall_budget_usd']+1e-12:
            raise BudgetExceeded('Next attempt exceeds the shared cumulative budget')
        self.ledgers[model_key].admit(reservation)

    def account(self,model_key,response,reservation):
        return self.ledgers[model_key].account(response,reservation)

    def reserve_unknown(self,model_key,reservation):
        if not math.isfinite(reservation) or reservation<0:
            raise ValueError('Invalid unknown-charge reservation')
        self.ledgers[model_key].unknown_cost_reservation+=reservation

    def as_dict(self):
        arms={key:value.as_dict() for key,value in self.ledgers.items()}
        fields=('api_attempts','estimated_usage_cost_usd','unknown_usage_reserved_usd','input_tokens',
                'output_tokens','reasoning_tokens','cached_input_tokens','cache_write_tokens')
        result={field:sum(value.get(field,0) for value in arms.values()) for field in fields}
        result.update(per_model=arms,budget_usd=self.config['max_budget_usd'],
                      overall_budget_usd=self.config['overall_budget_usd'],
                      prior_usage_cost_usd=self.config['prior_usage_cost_usd'],
                      cumulative_estimated_usage_cost_usd=self.config['prior_usage_cost_usd']+result['estimated_usage_cost_usd'],
                      cumulative_usage_plus_unknown_reserved_usd=self.config['prior_usage_cost_usd']+
                          result['estimated_usage_cost_usd']+result['unknown_usage_reserved_usd'],
                      budget_semantics='One shared conservative estimate across both models; provider invoice is authoritative')
        return result


def response_text(raw):
    return ''.join(item['text'] for block in raw.get('output',[]) if block['type']=='message'
                   for item in block['content'] if item['type']=='output_text')


def compile_response(config,case,record,raw):
    result={'choices':None,'blanks':None,'numericals':None}
    if raw.get('status')!='completed':
        truncated=(raw.get('status')=='incomplete' and (raw.get('incomplete_details') or {}).get('reason')=='max_output_tokens')
        return {**result,'status':'response_truncated' if truncated else 'invalid_output',
                'error_type':'OutputTokenLimit' if truncated else 'IncompleteResponse'}
    try:
        choices=json.loads(response_text(raw))
        blanks=resolve_choices(choices,'direct_rgb',None,(512,512))
        capture=ROOT/config['dataset']/case['example_id']
        declared={'instruction':record['model_inputs']['instruction'],'task_type':'open_middle_drawer'}
        numeric=compile_drawer(blanks,declared,np.load(capture/'depth_meters.npy',allow_pickle=False),
            json.loads((capture/'calibration.json').read_text()),config['geometry'])
        return {'choices':choices,'blanks':blanks,'numericals':numeric,'status':'valid'}
    except (ValueError,TypeError,KeyError) as error:
        return {**result,'status':'invalid_output','error_type':type(error).__name__}


def freeze(config_path):
    config=load_config(config_path)
    _,parent,records=check_parent(ROOT/config['parent_config'])
    destination=ROOT/config['frozen_protocol']
    if destination.exists():
        raise FileExistsError('Preserve the existing comparison protocol')
    implementation=copy.deepcopy(parent['implementation_hashes'])
    for path in ('src/vlm_tamp/drawer_model_comparison.py','scripts/run_drawer_models_high.py',
                 'scripts/report_drawer_models_high.py','scripts/verify_drawer_models_high.py'):
        implementation[path]=file_sha(ROOT/path)
    preserved=copy.deepcopy(parent['preserved_artifacts'])
    for folder in ('results/task-grounding-frontview-luna','results/task-grounding-frontview-report'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file():
                preserved[str(path.relative_to(ROOT))]=file_sha(path)
    requests=[]
    for case in config['cases']:
        args=request_arguments(config,case,records[case['example_id']])
        requests.append({'case_id':case['id'],'model_key':case['model_key'],
                         'request_sha256':hashlib.sha256(json_text(args).encode()).hexdigest()})
    protocol={'version':'drawer-models-high-1.0','config':config,'geometry':config['geometry'],'requests':requests,
              'prompt_sha256':file_sha(ROOT/config['prompt']),'implementation_hashes':implementation,
              'input_hashes':copy.deepcopy(parent['input_hashes']),'preserved_artifacts':preserved,
              'parent_protocol':parent['config']['frozen_protocol'],
              'parent_protocol_sha256':file_sha(ROOT/parent['config']['frozen_protocol']),
              'same_rgb_prompt_schema_geometry_and_grader':True,'reasoning_effort':'high',
              'reference_inputs_sent':False,'provided_inventory':False,'segmentation_used_in_inference':False,
              'crops_used':False,'depth_sent_to_vlm':False,
              'scope':'Four correlated frontal observations, same cabinet pose; development diagnostic, not population ranking'}
    destination.write_text(json_text(protocol)+'\n')
    return protocol


def check_protocol(config_path):
    config=load_config(config_path)
    _,parent,records=check_parent(ROOT/config['parent_config'])
    protocol=json.loads((ROOT/config['frozen_protocol']).read_text())
    if json_text(config)!=json_text(protocol['config']) or protocol['input_hashes']!=parent['input_hashes']:
        raise ValueError('Frozen comparison settings or scenes changed')
    if file_sha(ROOT/protocol['parent_protocol'])!=protocol['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    for path,sha in {**protocol['implementation_hashes'],**protocol['input_hashes'],
                     **protocol['preserved_artifacts'],config['prompt']:protocol['prompt_sha256']}.items():
        if file_sha(ROOT/path)!=sha:
            raise ValueError('Frozen or previous artifact changed: '+path)
    return config,protocol,records


def run(config_path,output,*,allow_paid=False,dry_run=False):
    config,protocol,records=check_protocol(config_path)
    output=Path(output)
    if output.exists():
        raise FileExistsError('Use a new comparison output directory')
    if not allow_paid and not dry_run:
        raise PermissionError('Actual model calls require --allow-paid')
    ledger=SharedLedger(config)
    manifest={'config':config,'provider':'openai','models':config['models'],'reasoning_effort':'high',
              'prompt_text':(ROOT/config['prompt']).read_text(),'frozen_protocol_sha256':file_sha(ROOT/config['frozen_protocol']),
              'dry_run':dry_run,'offline_mock':False,'model_input_inventory':False,
              'reference_answers_sent':False,'depth_sent_to_vlm':False,'input_images':1,
              'segmentation_used':False,'crops_used':False,'planned_case_count':8,'ledger':ledger.as_dict()}
    rows=[{'case_id':c['id'],'base_case_id':c['base_case_id'],'example_id':c['example_id'],
           'condition':'direct_rgb','model_key':c['model_key'],'model':c['model'],
           'status':'not_run','attempts':0,'cost_usd':0.,'choices':None,'blanks':None,'numericals':None}
          for c in config['cases']]
    output.mkdir(parents=True)
    def save():
        manifest['ledger']=ledger.as_dict()
        (output/'manifest.json').write_text(json_text(manifest)+'\n')
        (output/'predictions.jsonl').write_text(''.join(json_text(row)+'\n' for row in rows))
    prepared=[]
    for case,frozen in zip(config['cases'],protocol['requests']):
        _,original,_,_,payload=inputs(config,case,records[case['example_id']])
        args=request_arguments(config,case,records[case['example_id']])
        if frozen['case_id']!=case['id'] or frozen['request_sha256']!=hashlib.sha256(json_text(args).encode()).hexdigest():
            raise ValueError('Frozen request changed')
        folder=output/case['id'];folder.mkdir()
        (folder/'request.json').write_text(json_text(args)+'\n')
        (folder/'payload.json').write_text(json_text(payload)+'\n')
        (folder/'original.png').write_bytes(original)
        prepared.append((case,args,folder))
    save()
    if dry_run:
        return manifest
    from openai import OpenAI
    client=OpenAI(base_url='https://api.openai.com/v1',timeout=config['timeout_seconds'],max_retries=0)
    for index,(case,args,folder) in enumerate(prepared):
        row=rows[index];reservation=request_cost_reservation(config,args)
        try:
            ledger.admit(case['model_key'],reservation)
        except BudgetExceeded as error:
            manifest['stopped']={'reason':'budget_admission','detail':str(error),'next_case_id':case['id']}
            for pending in rows[index:]:
                if pending['status']=='not_run':pending['reason']=str(error)
            save();break
        row.update(status='api_failure',attempts=1,reservation_usd=reservation)
        try:
            response=client.responses.create(**args)
        except Exception as error:
            ledger.reserve_unknown(case['model_key'],reservation)
            row['error_type']=type(error).__name__
            manifest['stopped']={'reason':'api_failure','detail':'No retry or substitute prediction','case_id':case['id']}
            save();break
        raw=response.model_dump(mode='json')
        (folder/'response.json').write_text(json_text(raw)+'\n')
        row.update(response_model=raw.get('model'),response_status=raw.get('status'),
                   response_incomplete_details=raw.get('incomplete_details'))
        usage=response.usage
        row['cost_usd']=ledger.account(case['model_key'],ProviderResponse(response.output_text,raw,
            usage.input_tokens if usage else 0,usage.output_tokens if usage else 0),reservation)
        row.update(compile_response(config,case,records[case['example_id']],raw))
        if row['numericals'] is not None:
            (folder/'numericals.json').write_text(json_text(row['numericals'])+'\n')
        save()
        print(case['id'],row['status'],'estimated $'+format(row['cost_usd'],'.6f'),
              'reasoning tokens',((raw.get('usage') or {}).get('output_tokens_details') or {}).get('reasoning_tokens'),flush=True)
    return manifest
