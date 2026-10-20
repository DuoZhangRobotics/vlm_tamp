"""Task-specific blanks: no pick-place path constraint; drawer pull candidates.

Only RGB instructions and optional RGB-derived candidate IDs enter the API.
Depth supplies numerical estimates locally. References are read only by grading.
"""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import read_records
from .drawer_constraints import DRAWER_POINT_FIELDS, compile_drawer, validate_drawer_blanks
from .experiment import BudgetExceeded, Ledger, input_price_ceiling
from .libero import ROOT
from .providers import ProviderResponse
from .rgbd_landmark_choices import validate_candidates
from .rgbd_luna_landmark_trial import check_protocol as check_parent
from .rgbd_numericals import json_text, pixel_index

METHOD_FIELDS = ('joint_type', 'motion', 'path_constraint')
ID_FIELDS = tuple(key.removesuffix('_uv') + '_id' for key in DRAWER_POINT_FIELDS)
METHOD_CHOICES = {'joint_type': ['prismatic', 'revolute', 'unresolved'],
                  'motion': ['pull', 'push', 'unresolved'],
                  'path_constraint': ['prismatic_pull', 'none', 'unresolved']}
CONDITIONS = ('direct_rgb', 'dinov3_candidates')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text())
    if (config['model'] != 'gpt-6-luna' or config['reasoning_effort'] != 'none'
            or config['service_tier'] != 'default' or config['image_detail'] != 'high'
            or type(config['max_retries']) is not int or config['max_retries'] != 0
            or type(config['max_requests']) is not int or config['max_requests'] != 8
            or type(config['max_output_tokens']) is not int or config['max_output_tokens'] != 1000
            or config['conditions'] != list(CONDITIONS)):
        raise ValueError('Use the approved Luna controls and eight matched drawer requests')
    overall, prior, available = (config[key] for key in ('overall_budget_usd', 'prior_usage_cost_usd', 'max_budget_usd'))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (overall, prior, available)):
        raise ValueError('Invalid budget accounting')
    if not 0 < overall <= 2 or prior < 0 or available <= 0 or abs(overall-prior-available) > 1e-12:
        raise ValueError('Cumulative Luna experiment usage must fit the authorized two-dollar ceiling')
    ledger = json.loads((ROOT / config['pick_place_source_trial'] / 'manifest.json').read_text())['ledger']
    if abs(prior - ledger['estimated_usage_cost_usd']) > 1e-12:
        raise ValueError('Previous Luna usage must be included in the new budget')
    expected_prices = {'input_per_million': .10, 'cached_input_per_million': .01,
                       'cache_write_per_million': .125, 'output_per_million': .50}
    if config['pricing'] != expected_prices or config['requested_extension_m'] is not None:
        raise ValueError('Use verified prices; no unrequested numerical drawer extension')
    expected = [(condition, f'libero_goal-00-state{state:03d}-seed0')
                for condition in CONDITIONS for state in (0, 10, 20, 30)]
    if (len(config['cases']) != 8 or [(c['condition'], c['example_id']) for c in config['cases']] != expected
            or any(c['task_type'] != 'open_middle_drawer' for c in config['cases'])):
        raise ValueError('Use four predetermined middle-drawer observations per input method')
    ids = [case['id'] for case in config['cases']]
    if len(set(ids)) != 8 or any(not isinstance(v, str) or not v or Path(v).name != v or v in {'.', '..'} for v in ids):
        raise ValueError('Case IDs must be distinct safe directory names')
    return config


def blank_schema(condition, candidates=None):
    if condition not in CONDITIONS:
        raise ValueError('Unknown drawer input method')
    if condition == 'direct_rgb':
        field_names = DRAWER_POINT_FIELDS
        point = {'anyOf': [{'type':'array','items':{'type':'number','minimum':0,'maximum':1},
                            'minItems':2,'maxItems':2}, {'type':'null'}]}
    else:
        ids = validate_candidates(candidates)
        field_names = ID_FIELDS
        point = ({'anyOf':[{'type':'integer','minimum':0,'maximum':len(ids)-1},{'type':'null'}]}
                 if ids else {'type':'null'})
    properties = {field:copy.deepcopy(point) for field in field_names}
    properties.update({key:{'type':'string','enum':values} for key, values in METHOD_CHOICES.items()})
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def resolve_choices(choices, condition, candidates, shape):
    schema = blank_schema(condition,candidates)
    if not isinstance(choices,dict) or set(choices) != set(schema['properties']):
        raise ValueError('Fill exactly the task-specific drawer blanks')
    if condition == 'direct_rgb':
        return validate_drawer_blanks(copy.deepcopy(choices))
    ids = validate_candidates(candidates,shape)
    by_id = {point['id']:point for point in candidates}
    height,width = shape
    blanks = {key:choices[key] for key in METHOD_FIELDS}
    for field,uv_field in zip(ID_FIELDS,DRAWER_POINT_FIELDS):
        selected = choices[field]
        if selected is not None and (type(selected) is not int or selected not in ids):
            raise ValueError('Choose an available integer candidate ID or null')
        if selected is None:
            blanks[uv_field] = None
        else:
            x,y = by_id[selected]['pixel_xy']
            blanks[uv_field] = [(x+.5)/width,(y+.5)/height]
    return validate_drawer_blanks(blanks)


def inputs(config,case,record):
    if set(record['model_inputs']) != {'instruction','image'}:
        raise ValueError('Drawer model inputs may contain only instruction and image')
    capture = ROOT / config['dataset'] / case['example_id']
    original = (capture / 'rgb.png').read_bytes()
    with Image.open(capture / 'rgb.png') as image:
        shape = (image.height,image.width)
    if shape != (512,512):
        raise ValueError('Use the original 512 RGB-D captures')
    candidates,numbered = None,None
    fields = DRAWER_POINT_FIELDS
    if case['condition']=='dinov3_candidates':
        folder = ROOT / config['candidate_dataset'] / case['example_id']
        candidates = json.loads((folder / 'candidates.json').read_text())
        validate_candidates(candidates,shape)
        if (folder / 'original.png').read_bytes() != original:
            raise ValueError('Candidate RGB must equal the unchanged capture')
        numbered = (folder / 'numbered.png').read_bytes()
        fields = ID_FIELDS
    payload = {'instruction':record['model_inputs']['instruction'],'task_type':'open_middle_drawer',
               'blanks':list(fields+METHOD_FIELDS)}
    if candidates is not None:
        payload['available_candidate_ids'] = list(range(len(candidates)))
    return candidates,original,numbered,shape,payload


def sdk_arguments(config,case,payload,original,numbered,prompt,candidates=None):
    expected = {'instruction','task_type','blanks'} | ({'available_candidate_ids'} if numbered is not None else set())
    if set(payload) != expected or payload['task_type'] != 'open_middle_drawer':
        raise ValueError('Only task-specific instruction and candidate IDs may enter inference')
    names = DRAWER_POINT_FIELDS if case['condition']=='direct_rgb' else ID_FIELDS
    if payload['blanks'] != list(names+METHOD_FIELDS) or not payload['instruction'].strip():
        raise ValueError('Unexpected drawer blank declaration')
    if (case['condition']=='direct_rgb') != (numbered is None):
        raise ValueError('Input method and numbered image must agree')
    if numbered is not None and payload['available_candidate_ids'] != validate_candidates(candidates):
        raise ValueError('Candidate ID declaration changed')
    images = [original] + ([] if numbered is None else [numbered])
    if any(not isinstance(v,bytes) or not v for v in images):
        raise ValueError('Images must be nonempty PNG bytes')
    return {'model':config['model'],'instructions':prompt,
            'input':[{'role':'user','content':[{'type':'input_text','text':json_text(payload)}]+
                     [{'type':'input_image','detail':config['image_detail'],
                       'image_url':'data:image/png;base64,'+base64.b64encode(v).decode()} for v in images]}],
            'text':{'format':{'type':'json_schema','name':'DrawerGroundingBlanks',
                              'schema':blank_schema(case['condition'],candidates),'strict':True}},
            'reasoning':{'effort':config['reasoning_effort']},'service_tier':'default',
            'max_output_tokens':config['max_output_tokens'],'store':False}


def request_arguments(config,case,record):
    candidates,original,numbered,_,payload = inputs(config,case,record)
    return sdk_arguments(config,case,payload,original,numbered,(ROOT/config['prompt']).read_text(),candidates)


def request_cost_reservation(config,args):
    payload = args['input'][0]['content'][0]['text']
    count = len((args['instructions']+payload+json_text(args['text']['format']['schema'])).encode())
    return ((count+config['image_token_allowance']+512)*input_price_ceiling(config)
            +config['max_output_tokens']*config['pricing']['output_per_million'])/1e6


def derive_pick_place():
    """Reuse saved ordinary cases; no new inference or upright variants."""
    source = ROOT/'results/rgbd-dino-luna-report/summary.json'
    summary = json.loads(source.read_text())
    derived = []
    for task in summary['tasks']:
        case = next(c for c in summary['config']['cases'] if c['id']==task['id'])
        if case['upright']:
            continue
        row = copy.deepcopy(task['raw_row'])
        row.update(instruction=task['instruction'],source_case_id=task['id'],
                   task_type='pick_place',reused_prediction=True,new_api_calls=0,
                   source_summary=str(source.relative_to(ROOT)),source_summary_sha256=file_sha(source))
        row['choices'] = {k:v for k,v in row['choices'].items() if not k.startswith('axis_') and k!='path_constraint'}
        row['blanks'] = {k:v for k,v in row['blanks'].items() if not k.startswith('axis_') and k!='path_constraint'}
        numeric = row['numericals']
        goals = [r for r in numeric['relations'] if r['scope']=='goal']
        numeric.pop('object_axis',None)
        numeric['landmarks'] = {k:v for k,v in numeric['landmarks'].items() if not k.startswith('axis_')}
        numeric.update(schema_version='pick-place-scope-1.0',task_type='pick_place',
                       relations=goals,goals=copy.deepcopy(goals),path_constraints=[],
                       stages=[{'id':stage,'path_constraint_ids':[]} for stage in ('approach','grasp','transport','place','release')])
        row['metrics'] = {'semantic_complete_goal':task['metrics']['semantic_complete_goal'],
                          'anchors':[a for a in task['metrics']['anchors'] if not a['field'].startswith('axis_')],
                          'path_constraints_required':False,'path_constraints_emitted':0,
                          'robot_execution_evaluated':False}
        derived.append(row)
    if len(derived)!=8:
        raise ValueError('Reuse exactly four ordinary cases from each DINO candidate arm')
    return derived


def freeze(config_path):
    config = load_config(config_path)
    _,parent,_ = check_parent(ROOT/config['parent_config'])
    destination = ROOT/config['frozen_protocol']
    if destination.exists():
        raise FileExistsError('Preserve the existing frozen task-scope protocol')
    records,_ = read_records(ROOT/config['dataset'])
    records = {r['example_id']:r for r in records}
    files, requests, coverage = {},[],{}
    for folder in (ROOT/config['dataset'], ROOT/config['candidate_dataset']):
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                files[str(path.relative_to(ROOT))] = file_sha(path)
    for case in config['cases']:
        args = request_arguments(config,case,records[case['example_id']])
        requests.append({'case_id':case['id'],'request_sha256':hashlib.sha256(json_text(args).encode()).hexdigest()})
    # This development audit schedules calls; it never changes proposals or predictions.
    for identifier in records:
        folder = ROOT/config['dataset']/identifier
        candidates = json.loads((ROOT/config['candidate_dataset']/identifier/'candidates.json').read_text())
        counts = {}
        for role in ('middle_drawer','middle_handle','middle_panel'):
            mask = np.load(folder/('evaluation/'+role+'_mask.npy'),allow_pickle=False)
            counts[role] = sum(bool(mask[p['pixel_xy'][1],p['pixel_xy'][0]]) for p in candidates)
        coverage[identifier] = {'available_points_by_reference_role':counts,
                                'required_handle_covered':counts['middle_handle']>0,
                                'scope':'Offline development setup audit only; not a deployable semantic gate',
                                'reference_points_added':False}
    implementation = dict(parent['implementation_hashes'])
    for path in ('src/vlm_tamp/task_grounding_scope.py','src/vlm_tamp/drawer_constraints.py',
                 'scripts/prepare_drawer_grounding.py','scripts/prepare_drawer_candidates.py',
                 'scripts/run_task_grounding_scope.py','scripts/report_task_grounding_scope.py',
                 'scripts/verify_task_grounding_scope.py'):
        implementation[path] = file_sha(ROOT/path)
    preserved = dict(parent['preserved_artifacts'])
    for folder in ('results/rgbd-dino-luna-comparison','results/rgbd-dino-luna-report'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file():
                preserved[str(path.relative_to(ROOT))] = file_sha(path)
    protocol = {'version':'task-grounding-scope-1.0','config':config,'geometry':config['geometry'],'requests':requests,
                'prompt_sha256':file_sha(ROOT/config['prompt']),'implementation_hashes':implementation,
                'input_hashes':files,'preserved_artifacts':preserved,
                'parent_protocol':parent['config']['frozen_protocol'],
                'parent_protocol_sha256':file_sha(ROOT/parent['config']['frozen_protocol']),
                'pick_place_path_constraints':False,'drawer_constraint_stage':'pull',
                'offline_proposal_coverage':coverage,
                'reference_inputs_sent':False,'provided_inventory':False,
                'reference_files_used_only_for_evaluation':True,
                'scope':'four correlated drawer development observations; no population ranking'}
    destination.write_text(json_text(protocol)+'\n')
    return protocol


def check_protocol(config_path):
    config = load_config(config_path)
    check_parent(ROOT/config['parent_config'])
    protocol = json.loads((ROOT/config['frozen_protocol']).read_text())
    if json_text(config) != json_text(protocol['config']):
        raise ValueError('Frozen task-scope settings changed')
    if file_sha(ROOT/protocol['parent_protocol']) != protocol['parent_protocol_sha256']:
        raise ValueError('Parent Luna protocol changed')
    for path,sha in {**protocol['implementation_hashes'],**protocol['input_hashes'],
                     **protocol['preserved_artifacts'],config['prompt']:protocol['prompt_sha256']}.items():
        if file_sha(ROOT/path)!=sha:
            raise ValueError('Frozen or previous artifact changed: '+path)
    records,_ = read_records(ROOT/config['dataset'])
    return config,protocol,{r['example_id']:r for r in records}


def grade_drawer(record,root,case,row):
    """Independent held-back instance/joint scoring; no identity rescue."""
    root = Path(root)
    folder = root/'data/rgbd-drawer-pilot'/case['example_id']
    reference = json.loads((folder/'evaluation/reference.json').read_text())
    masks = {key:np.load(folder/value['npy'],allow_pickle=False) for key,value in reference['masks'].items()}
    blanks,numeric = row.get('blanks'),row.get('numericals')
    expected = {'source_uv':'middle_drawer','handle_uv':'middle_handle',
                'face_a_uv':'middle_panel','face_b_uv':'middle_panel','face_c_uv':'middle_panel'}
    anchors = []
    for field,role in expected.items():
        uv = blanks.get(field) if blanks else None
        correct = False
        if uv is not None:
            x,y = pixel_index(uv,masks[role].shape)
            correct = bool(masks[role][y,x])
        anchors.append({'field':field,'uv':uv,'expected_role_offline_only':role,
                        'correct':correct,'diagnostic':'abstained' if uv is None else 'correct_instance' if correct else 'wrong_instance_or_background',
                        'depth_status':numeric['landmarks'][field]['status'] if numeric else 'unresolved'})
    by_field = {a['field']:a for a in anchors}
    axis_error,handle_distance = None,None
    direction_correct,axis_within_tolerance = False,False
    if numeric and numeric['drawer_axis']['status']=='estimated':
        axis = np.asarray(numeric['drawer_axis']['axis_world'])
        target = np.asarray(reference['middle_drawer']['axisopeningworld'])
        cosine = float(axis@target/(np.linalg.norm(axis)*np.linalg.norm(target)))
        axis_error = float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
        direction_correct,axis_within_tolerance = cosine>0,axis_error<=5.0
    if numeric and numeric['landmarks']['handle_uv']['status']=='measured' and masks['middle_handle'].any():
        cloud = np.load(folder/'world_points.npy',allow_pickle=False)
        point = np.asarray(numeric['landmarks']['handle_uv']['position_m'])
        handle_distance = float(np.min(np.linalg.norm(cloud[masks['middle_handle']]-point,axis=1)))
    type_correct = bool(blanks and blanks['joint_type']==reference['middle_drawer']['joint_type'])
    motion_correct = bool(blanks and blanks['motion']=='pull')
    path_correct = bool(blanks and blanks['path_constraint']=='prismatic_pull')
    return {'valid_form':row['status']=='valid','anchors':anchors,
            'middle_drawer_correct':by_field['source_uv']['correct'],
            'middle_handle_correct':by_field['handle_uv']['correct'],
            'role_binding_correct':by_field['source_uv']['correct'] and by_field['handle_uv']['correct'],
            'correct_anchors':sum(a['correct'] for a in anchors),'anchor_denominator':5,
            'front_body_landmarks_correct':all(by_field[k]['correct'] for k in ('face_a_uv','face_b_uv','face_c_uv')),
            'joint_type_correct':type_correct,'motion_correct':motion_correct,'path_constraint_correct':path_correct,
            'constraint_template_correct':type_correct and motion_correct and path_correct,
            'axis_error_degrees':axis_error,'axis_direction_correct':direction_correct,
            'axis_within_tolerance':axis_within_tolerance,'axis_tolerance_degrees':5.0,
            'handle_surface_distance_m':handle_distance,
            'handle_surface_distance_scope':'Distance to visible held-back handle surface; not a certified grasp center',
            'line_constraint_compiled':bool(numeric and numeric['path_constraints']),
            'numerical_candidate_correct':bool(numeric and numeric['path_constraints'] and by_field['handle_uv']['correct'] and axis_within_tolerance),
            'front_face_semantics_verified':False,'joint_limits_supplied_to_inference':False,
            'opening_goal_resolved':bool(numeric and numeric['opening_goal']['status']!='unresolved'),
            'planner_ready':False,'grasp_success_evaluated':False,'robot_execution_evaluated':False}


def run(config_path,output,*,allow_paid=False,dry_run=False):
    config,protocol,records = check_protocol(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new task-scope output directory')
    if not dry_run and not allow_paid:
        raise PermissionError('Actual Luna requests require --allow-paid')
    prompt = (ROOT/config['prompt']).read_text()
    ledger = Ledger(config)
    manifest = {'config':config,'provider':'openai','model':config['model'],'prompt_text':prompt,
                'frozen_protocol_sha256':file_sha(ROOT/config['frozen_protocol']),
                'dry_run':dry_run,'offline_mock':False,'model_input_inventory':False,
                'reference_answers_sent':False,'depth_sent_to_vlm':False,
                'ledger':ledger.as_dict(),'overall_budget_usd':config['overall_budget_usd'],
                'prior_usage_cost_usd':config['prior_usage_cost_usd'],
                'pick_place_reused_cases':8,'pick_place_new_api_calls':0,'planned_case_count':8,
                'offline_development_proposal_gate_used':True}
    output.mkdir(parents=True)
    (output/'pick_place.json').write_text(json_text(derive_pick_place())+'\n')
    arguments=[]
    rows=[{'case_id':c['id'],'example_id':c['example_id'],'condition':c['condition'],'status':'not_run',
           'choices':None,'blanks':None,'numericals':None,'attempts':0,'cost_usd':0.} for c in config['cases']]
    def save():
        manifest['ledger']=ledger.as_dict()
        manifest['cumulative_luna_estimated_cost_usd']=config['prior_usage_cost_usd']+ledger.actual_cost
        (output/'manifest.json').write_text(json_text(manifest)+'\n')
        (output/'predictions.jsonl').write_text(''.join(json_text(r)+'\n' for r in rows))
    for case,frozen in zip(config['cases'],protocol['requests']):
        candidates,original,numbered,shape,payload=inputs(config,case,records[case['example_id']])
        args=sdk_arguments(config,case,payload,original,numbered,prompt,candidates)
        if frozen['case_id']!=case['id'] or frozen['request_sha256']!=hashlib.sha256(json_text(args).encode()).hexdigest():
            raise ValueError('Frozen request changed')
        folder=output/case['id']; folder.mkdir()
        (folder/'request.json').write_text(json_text(args)+'\n')
        (folder/'payload.json').write_text(json_text(payload)+'\n')
        (folder/'original.png').write_bytes(original)
        if numbered is not None:
            (folder/'numbered.png').write_bytes(numbered)
        arguments.append((case,candidates,shape,payload,args,folder))
    save()
    if dry_run:
        return manifest
    from openai import OpenAI
    client=OpenAI(base_url='https://api.openai.com/v1',timeout=config['timeout_seconds'],max_retries=0)
    for index,(case,candidates,shape,payload,args,folder) in enumerate(arguments):
        row=rows[index]
        if (case['condition']=='dinov3_candidates'
                and not protocol['offline_proposal_coverage'][case['example_id']]['required_handle_covered']):
            row.update(status='proposal_unavailable',
                       reason='Offline development proposal audit: no candidate on the required middle handle; no inference performed')
            save()
            print(case['id'],row['status'],'no API call',flush=True)
            continue
        reservation=request_cost_reservation(config,args)
        try:
            ledger.admit(reservation)
        except BudgetExceeded as error:
            manifest['stopped']={'reason':'budget_admission','detail':str(error)}; save(); break
        row.update(status='api_failure',attempts=1)
        try:
            response=client.responses.create(**args)
        except Exception as error:
            ledger.unknown_cost_reservation+=reservation
            row['error_type']=type(error).__name__
            manifest['stopped']={'reason':'api_failure','detail':'No retry or substitute inference'}; save(); break
        raw=response.model_dump(mode='json')
        (folder/'response.json').write_text(json_text(raw)+'\n')
        row.update(response_model=raw.get('model'),response_status=raw.get('status'))
        row['cost_usd']=ledger.account(ProviderResponse(response.output_text,raw,
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0),reservation)
        try:
            if raw.get('status')!='completed':
                raise ValueError('A complete drawer form is required')
            choices=json.loads(response.output_text)
            blanks=resolve_choices(choices,case['condition'],candidates,shape)
            capture=ROOT/config['dataset']/case['example_id']
            numeric=compile_drawer(blanks,payload,np.load(capture/'depth_meters.npy',allow_pickle=False),
                json.loads((capture/'calibration.json').read_text()),config['geometry'])
            row.update(choices=choices,blanks=blanks,numericals=numeric,status='valid')
            (folder/'numericals.json').write_text(json_text(numeric)+'\n')
        except (ValueError,TypeError,KeyError) as error:
            row.update(status='invalid_output',error_type=type(error).__name__)
        save()
        print(case['id'],row['status'],'estimated $'+format(row['cost_usd'],'.6f'),flush=True)
    return manifest
