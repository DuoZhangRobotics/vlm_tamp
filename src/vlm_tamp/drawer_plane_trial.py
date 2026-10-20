"""Zero-call paired geometry replay with separate held-back support evaluation."""
import copy
import json
from pathlib import Path

import numpy as np

from .drawer_local_plane import LOCAL_PLANE_DEFAULTS, compile_local_drawer
from .drawer_model_comparison import check_protocol as check_source, file_sha
from .libero import ROOT
from .rgbd_numericals import json_text, pixel_index
from .task_grounding_frontview import grade_drawer


LOCAL_FIELDS=('source_uv','handle_uv','joint_type','motion','path_constraint')


def load_config(path):
    config=json.loads(Path(path).read_text())
    parent=json.loads((ROOT/config['source_config']).read_text())
    if (config['source_config']!='configs/drawer_models_high.json'
            or config['source_trial']!='results/drawer-models-high'
            or config['dataset']!=parent['dataset'] or config['geometry']!=parent['geometry']
            or config['model']!='gpt-6-luna' or config['reasoning_effort']!='high'
            or type(config['api_calls']) is not int or config['api_calls']!=0):
        raise ValueError('Reuse the approved cached Luna-high predictions without new inference')
    if json_text(config['local_plane'])!=json_text(LOCAL_PLANE_DEFAULTS):
        raise ValueError('Use the predeclared blind local-plane policy')
    expected=[f'luna_high_state{s:03d}' for s in (0,10,20,30)]
    if ([c['id'] for c in config['cases']]!=expected
            or any(c['source_case_id']!=c['id'] or c['example_id']!=f"libero_goal-00-state{int(c['id'][-3:]):03d}-seed0"
                   for c in config['cases'])):
        raise ValueError('Use the same four cached Luna scenes')
    ledger=json.loads((ROOT/config['source_trial']/'manifest.json').read_text())['ledger']
    if config['prior_usage_cost_usd']!=ledger['cumulative_estimated_usage_cost_usd'] or config['overall_budget_usd']!=2:
        raise ValueError('Keep cumulative usage and the authorized budget explicit')
    return config


def settings(config):
    return {**config['geometry'],**config['local_plane']}


def source_rows(config):
    return {r['case_id']:r for r in map(json.loads,(ROOT/config['source_trial']/'predictions.jsonl').read_text().splitlines())}


def support_mask(pixels,shape):
    mask=np.zeros(shape,dtype=bool)
    for pixel in pixels:
        if (not isinstance(pixel,list) or len(pixel)!=2 or any(type(v) is not int for v in pixel)
                or not 0<=pixel[0]<shape[1] or not 0<=pixel[1]<shape[0]):
            raise ValueError('Support must contain unchanged original sensor pixels')
        mask[pixel[1],pixel[0]]=True
    return mask


def old_support_mask(numeric,depth,geometry):
    selected=np.zeros(depth.shape,dtype=bool)
    radius=geometry['plane_patch_radius_pixels']
    keys=('face_a_uv','face_b_uv','face_c_uv')
    if not all(numeric['landmarks'][key]['status']=='measured' for key in keys):
        return selected
    for key in keys:
        x,y=numeric['landmarks'][key]['pixel_xy']
        ys=slice(max(0,y-radius),min(depth.shape[0],y+radius+1))
        xs=slice(max(0,x-radius),min(depth.shape[1],x+radius+1))
        selected[ys,xs]|=np.abs(depth[ys,xs]-depth[y,x])<=geometry['plane_depth_band_m']
    return selected


def support_quality(mask,reference_masks):
    count=int(mask.sum())
    middle=int(np.count_nonzero(mask & reference_masks['middle_panel']))
    handle=int(np.count_nonzero(mask & reference_masks['middle_handle']))
    cabinet=int(np.count_nonzero(mask & reference_masks['cabinet']))
    return {'support_sample_count':count,'middle_panel_samples':middle,
            'middle_panel_purity':middle/count if count else None,
            'middle_handle_samples':handle,'middle_handle_fraction':handle/count if count else None,
            'other_cabinet_samples':cabinet-middle-handle,'non_cabinet_samples':count-cabinet,
            'part_labels_used_in_sampling':False,
            'reference_scope':'Visible middle drawer body mesh excluding handle; not certified front-face-only segmentation'}


def support_audit(record,root,case,old_numeric,numeric,config):
    """Read held-back labels only after both supports have been determined."""
    folder=Path(root)/config['dataset']/case['example_id']
    reference=json.loads((folder/'evaluation/reference.json').read_text())
    masks={k:np.load(folder/v['npy'],allow_pickle=False) for k,v in reference['masks'].items()}
    depth=np.load(folder/'depth_meters.npy',allow_pickle=False)
    old=old_support_mask(old_numeric,depth,config['geometry'])
    observed=numeric['local_plane_support']
    local=support_mask(observed['inlier_pixel_xy'],depth.shape)
    candidates=support_mask(observed['candidate_pixel_xy'],depth.shape)
    rejected=support_mask(observed['rejected_candidate_pixel_xy'],depth.shape)
    return {'three_point':support_quality(old,masks),'local_depth':support_quality(local,masks),
            'local_candidates':support_quality(candidates,masks),'local_rejected':support_quality(rejected,masks),
            'support_mask_is_semantic_segmentation':False,
            'coplanar_neighbor_drawers_resolved':False,'evaluation_only':True}


def grade_local_drawer(record,root,case,row,config=None):
    """Score semantic anchors and geometry independently under the reduced contract."""
    if config is None:
        config=json.loads((ROOT/'configs/drawer_plane_ablation.json').read_text())
    folder=Path(root)/config['dataset']/case['example_id']
    reference=json.loads((folder/'evaluation/reference.json').read_text())
    masks={k:np.load(folder/v['npy'],allow_pickle=False) for k,v in reference['masks'].items()}
    blanks,numeric=row['blanks'],row['numericals']
    bindings={}
    for field,part in (('source_uv','middle_drawer'),('handle_uv','middle_handle')):
        uv=blanks[field]
        hit=False
        if uv is not None:
            x,y=pixel_index(uv,masks[part].shape)
            hit=bool(masks[part][y,x])
        bindings[field]=hit
    joint_ok=blanks['joint_type']==reference['middle_drawer']['joint_type']
    template_ok=joint_ok and blanks['motion']=='pull' and blanks['path_constraint']=='prismatic_pull'
    axis_error=None
    if numeric['drawer_axis']['status']=='estimated':
        axis=np.asarray(numeric['drawer_axis']['axis_world'])
        target=np.asarray(reference['middle_drawer']['axisopeningworld'])
        axis_error=float(np.rad2deg(np.arccos(np.clip(axis@target/(np.linalg.norm(axis)*np.linalg.norm(target)),-1,1))))
    tolerance=config['evaluation']['axis_tolerance_degrees']
    axis_ok=axis_error is not None and axis_error<=tolerance
    observed=support_mask(numeric['local_plane_support']['inlier_pixel_xy'],masks['middle_panel'].shape)
    quality=support_quality(observed,masks)
    purity=quality['middle_panel_purity']
    purity_ok=purity is not None and purity>=config['evaluation']['minimum_support_purity_diagnostic']
    anchor_ok=all(bindings.values())
    line_ok=bool(numeric['path_constraints'])
    criterion=anchor_ok and template_ok and line_ok and axis_ok
    return {'middle_drawer_correct':bindings['source_uv'],'middle_handle_correct':bindings['handle_uv'],
            'role_binding_correct':anchor_ok,'constraint_template_correct':template_ok,
            'line_constraint_compiled':line_ok,'front_plane_estimated':numeric['drawer_front_plane']['status']=='estimated',
            'axis_error_degrees':axis_error,'axis_within_tolerance':axis_ok,'axis_tolerance_degrees':tolerance,
            'two_point_geometry_criterion':criterion,'middle_panel_support_purity':purity,
            'support_purity_above_diagnostic_threshold':purity_ok,
            'two_point_geometry_and_reference_pure_support':criterion and purity_ok,
            'support_purity_threshold':config['evaluation']['minimum_support_purity_diagnostic'],
            'support_purity_is_fit_gate':False,'coplanar_drawer_boundaries_resolved':False,
            'semantic_front_face_verified':False,'planner_ready':False,'robot_execution_evaluated':False,
            'comparison_scope':'Reduced two-anchor geometry contract, not correction of the old five-anchor answer'}


def replay_case(config,case,record,source_row):
    if source_row['status']!='valid' or source_row['model_key']!='luna':
        raise ValueError('Reuse only completed real Luna-high predictions')
    blanks={key:copy.deepcopy(source_row['blanks'][key]) for key in LOCAL_FIELDS}
    folder=ROOT/config['dataset']/case['example_id']
    depth=np.load(folder/'depth_meters.npy',allow_pickle=False)
    calibration=json.loads((folder/'calibration.json').read_text())
    numeric=compile_local_drawer(blanks,{'instruction':record['model_inputs']['instruction']},depth,calibration,settings(config))
    row={'case_id':case['id'],'source_case_id':case['source_case_id'],'example_id':case['example_id'],
         'condition':'cached_local_depth','raw_source_model':source_row['model'],
         'source_prediction_status':source_row['status'],'source_blanks':copy.deepcopy(source_row['blanks']),
         'blanks':blanks,'old_numericals':copy.deepcopy(source_row['numericals']),'numericals':numeric,
         'status':'derived' if numeric['drawer_front_plane']['status']=='estimated' else 'geometry_unresolved',
         'new_api_calls':0,'cost_usd':0.,'semantic_pixels_changed':False,
         'old_metrics':grade_drawer(record,ROOT,case,source_row)}
    row['metrics']=grade_local_drawer(record,ROOT,case,row,config)
    row['support_audit']=support_audit(record,ROOT,case,row['old_numericals'],numeric,config)
    row['old_metrics']['old_five_anchor_criterion']=bool(row['old_metrics']['role_binding_correct'] and
        row['old_metrics']['front_body_landmarks_correct'] and row['old_metrics']['constraint_template_correct'] and
        row['old_metrics']['line_constraint_compiled'] and row['old_metrics']['axis_within_tolerance'])
    return row


def freeze(config_path):
    config=load_config(config_path)
    _,parent,_=check_source(ROOT/config['source_config'])
    destination=ROOT/config['frozen_protocol']
    if destination.exists():raise FileExistsError('Preserve the existing offline comparison protocol')
    implementation=copy.deepcopy(parent['implementation_hashes'])
    for path in ('src/vlm_tamp/drawer_local_plane.py','src/vlm_tamp/drawer_plane_trial.py',
                 'scripts/run_drawer_plane_ablation.py','scripts/report_drawer_plane_ablation.py',
                 'scripts/verify_drawer_plane_ablation.py'):
        implementation[path]=file_sha(ROOT/path)
    preserved=copy.deepcopy(parent['preserved_artifacts'])
    for folder in ('results/drawer-models-high','results/drawer-models-high-report'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file():preserved[str(path.relative_to(ROOT))]=file_sha(path)
    protocol={'version':'drawer-plane-ablation-1.0','config':config,
              'parent_protocol':parent['config']['frozen_protocol'],
              'parent_protocol_sha256':file_sha(ROOT/parent['config']['frozen_protocol']),
              'implementation_hashes':implementation,'input_hashes':copy.deepcopy(parent['input_hashes']),
              'preserved_artifacts':preserved,'api_calls':0,'semantic_pixels_unchanged':True,
              'local_policy_chosen_before_case_scoring':True,'reference_mask_rescue':False,
              'changed_contract':'Three model-selected face landmarks replaced by locally observed geometric support'}
    destination.write_text(json_text(protocol)+'\n')
    return protocol


def check_protocol(config_path):
    config=load_config(config_path)
    _,parent,records=check_source(ROOT/config['source_config'])
    protocol=json.loads((ROOT/config['frozen_protocol']).read_text())
    if json_text(config)!=json_text(protocol['config']):raise ValueError('Frozen offline configuration changed')
    if (file_sha(ROOT/protocol['parent_protocol'])!=protocol['parent_protocol_sha256']
            or protocol['input_hashes']!=parent['input_hashes']):raise ValueError('Source protocol or RGB-D inputs changed')
    for path,sha in {**protocol['implementation_hashes'],**protocol['input_hashes'],**protocol['preserved_artifacts']}.items():
        if file_sha(ROOT/path)!=sha:raise ValueError('Frozen or prior artifact changed: '+path)
    return config,protocol,records


def run(config_path,output):
    config,protocol,records=check_protocol(config_path)
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new offline output directory')
    output.mkdir(parents=True)
    sources=source_rows(config)
    rows=[]
    for case in config['cases']:
        row=replay_case(config,case,records[case['example_id']],sources[case['source_case_id']])
        folder=output/case['id'];folder.mkdir()
        (folder/'numericals.json').write_text(json_text(row['numericals'])+'\n')
        (folder/'result.json').write_text(json_text(row)+'\n')
        rows.append(row)
        print(case['id'],row['status'],row['numericals']['drawer_front_plane'].get('reason'),
              'axis error',row['metrics']['axis_error_degrees'],'support purity',row['metrics']['middle_panel_support_purity'],flush=True)
    manifest={'config':config,'offline_replay':True,'offline_mock':False,'api_calls':0,'new_usage_cost_usd':0.,
              'source_model':'gpt-6-luna','source_reasoning_effort':'high','semantic_pixels_changed':False,
              'frozen_protocol_sha256':file_sha(ROOT/config['frozen_protocol']),
              'prior_usage_cost_usd':config['prior_usage_cost_usd'],'overall_budget_usd':config['overall_budget_usd']}
    (output/'manifest.json').write_text(json_text(manifest)+'\n')
    (output/'predictions.jsonl').write_text(''.join(json_text(row)+'\n' for row in rows))
    return manifest
