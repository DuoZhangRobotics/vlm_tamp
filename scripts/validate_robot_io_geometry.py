"""Observer-only regression and geometry checks on retained sensor snapshots."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from try_rekep_bowl_v3 import read_observation_v3
from vlm_tamp.handle_contact import ground_handle
from vlm_tamp.hinged_contact_tools import ground_handle as shared_ground_handle
from vlm_tamp.local_agent import write_json
from vlm_tamp.robot_contracts import observed_image_sizes,validate_operation


def load_case(name, *, trial_root=None):
    source=Path(trial_root) if trial_root is not None else ROOT/'results/astra-luna-comparison-20261023/trials'/name
    opened=source/'attempt01/opened';public=source/'agent-workspace/robot_data/attempt01/obs001'
    meta=json.loads((opened/'metadata.json').read_text())
    cameras,meta=read_observation_v3(opened,meta)
    calibration=json.loads((public/'calibration.json').read_text())
    checks={}
    with np.load(public/'sensors.npz') as arrays:
        for name,view in cameras.items():
            for key in ['depth','world_points','known_robot_depth_prior']:
                checks[name+'_'+key]=bool(np.array_equal(view[key],arrays[name+'_'+key],equal_nan=True))
            for key in ['intrinsic','camera_to_world']:
                checks[name+'_'+key]=bool(np.array_equal(view[key],calibration[name][key]))
            # RGB exports can differ by a rounding unit. Use the exact public
            # image array supplied to the model, without changing metric data.
            view['rgb']=arrays[name+'_rgb'].copy()
    if not all(checks.values()):
        raise ValueError('Retained geometry/calibration mismatch; cannot reuse robot-prior provenance')
    provenance={'source':str(source.relative_to(ROOT)),
        'public_sensor_sha256':hashlib.sha256((public/'sensors.npz').read_bytes()).hexdigest(),
        'original_metadata_sha256':hashlib.sha256((opened/'metadata.json').read_bytes()).hexdigest(),
        'metric_arrays_and_calibration_match':all(checks.values()),
        'robot_prior_provenance':'Restored by the normal loader from original metadata after exact depth/prior equality checks.',
        'rgb':'Exact public sensor RGB arrays.'}
    return cameras,meta,provenance


def choices(regions):
    return {**copy.deepcopy(regions),'motion':'open','joint_family':'revolute','axis_source':'vertical_hinge'}


def template(camera):
    return {'schema_version':'articulated-spec-1.0','task_type':'articulated','camera':camera,
        'instruction':'open the microwave','joint_family':'revolute','motion':'open',
        'target_progress':float(np.pi/2),'progress_units':'radians','axis_prior':'upright door',
        'contact_policy':'selected functional finger contact',
        'constraint':{'type':'point_on_circle_with_coupled_orientation'}}


def regression(root):
    case='state039-r1-luna';cameras,meta,provenance=load_case(case)
    source=ROOT/provenance['source']
    events=[json.loads(s) for s in (source/'tool-events.jsonl').read_text().splitlines()]
    selected=choices(next(x['request']['arguments']['regions'] for x in events if x['request']['operation']=='ground'))
    spec=importlib.util.spec_from_file_location('vlm_tamp._io_previous_handle_contact',root/'before/src/vlm_tamp/handle_contact.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    times={};outputs={}
    for label,method,args in [('before',old.ground_handle,selected),('after',ground_handle,selected),
                              ('unknown',ground_handle,{**selected,'initial_image_direction_xy':None})]:
        started=time.monotonic()
        outputs[label]=method(cameras,copy.deepcopy(args),template(selected['camera']),meta)
        times[label]=time.monotonic()-started
    before,after,unknown=[outputs[k] for k in ['before','after','unknown']]
    differences={key:float(np.max(np.abs(np.asarray(before[0].spec[key])-np.asarray(after[0].spec[key]))))
                 for key in ['joint_origin_world_m','joint_axis_world','source_anchor_world_m','panel_normal_world']}
    differences['contact_cloud']=float(np.max(np.abs(before[1]-after[1])))
    differences['anchors']=float(np.max(np.abs(np.asarray(before[3])-np.asarray(after[3]))))
    mask_equal=bool(np.array_equal(before[2],after[2]));unknown_axis_equal=bool(np.array_equal(after[0].spec['joint_axis_world'],unknown[0].spec['joint_axis_world']))
    result={'provenance':provenance,'seconds':times,'maximum_absolute_differences':differences,
        'mask_identical':mask_equal,'unknown_preserves_depth_selected_axis':unknown_axis_equal,
        'direction_evidence':after[4]['motion_direction_check'],'unknown_direction_evidence':unknown[4]['motion_direction_check'],
        'fixture_corrections':['Use retained original robot-prior provenance through the normal loader.','Supply the normal revolute mechanism template, including target_progress.'],
        'passed':mask_equal and unknown_axis_equal and all(x<1e-10 for x in differences.values())}
    write_json(root/'saved-geometry-regression.json',result)
    print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)


def evaluate(root):
    rows=json.loads((root/'qwen-retest/results.json').read_text());results=[];cache={}
    native=json.loads((ROOT/'results/astra-luna-comparison-20261023/grounding-diagnostics-private.json').read_text())
    native_by_trial={x['trial']:x for x in native}
    for row in rows:
        if row['kind']!='vision':continue
        result={'id':row['id'],'contract_valid':row['final_contract_valid'],'geometry_accepted':False,
                'private_observer_only':True,'supplied_to_model':False}
        if not row['final_contract_valid']:
            result['skip_reason']='No valid request; no geometry work attempted.'
        else:
            case=row['source_private']['case']
            if case not in cache:cache[case]=load_case(case)
            cameras,meta,provenance=cache[case];regions=row['attempts'][-1]['arguments']['regions']
            result['provenance']=provenance
            started=time.monotonic()
            try:
                validate_operation('ground',{'regions':regions},observed_image_sizes(cameras))
                mechanism,cloud,mask,anchors,evidence=shared_ground_handle(cameras,choices(regions),template(regions['camera']),meta)
                delta=np.asarray(mechanism.spec['joint_origin_world_m'])-native_by_trial[case]['native_hinge_anchor_world_m']
                result.update(geometry_accepted=True,model=mechanism.spec,fit_diagnostics=evidence,
                    native_hinge_line_error_mm=float(np.linalg.norm(delta[:2])*1000),
                    interpretation='An admitted fit is not proof that the selected hinge or physical contact is correct.')
            except Exception as error:
                result['error']=f'{type(error).__name__}: {error}'
            result['geometry_seconds']=time.monotonic()-started
        results.append(result);write_json(root/'qwen-geometry-private.json',results)
        print(json.dumps({k:v for k,v in result.items() if k not in {'model','fit_diagnostics','provenance'}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--evaluate-qwen',action='store_true');args=parser.parse_args()
    evaluate(args.run.resolve()) if args.evaluate_qwen else regression(args.run.resolve())
