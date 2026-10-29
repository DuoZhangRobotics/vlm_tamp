"""Registered offline Qwen3.5 size comparison using retained public observations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.local_agent import LocalChatClient,parse_call,write_json
from vlm_tamp.robot_contracts import validate_operation
from validate_robot_io_geometry import load_case
from evaluate_qwen_sampling_geometry import evaluate
from probe_qwen_tool_assisted_grounding import run_case
from diagnose_qwen_sampling import repetition

BASE=ROOT/'results/qwen-sampling-diagnostic-20261025'
SOURCES=['scripts/compare_qwen_sizes.py','scripts/serve_local_qwen.py',
 'scripts/probe_qwen_tool_assisted_grounding.py','scripts/evaluate_qwen_sampling_geometry.py',
 'scripts/validate_robot_io_geometry.py','src/vlm_tamp/local_agent.py',
 'src/vlm_tamp/local_context_budget.py','src/vlm_tamp/robot_contracts.py',
 'src/vlm_tamp/handle_contact.py','src/vlm_tamp/hinged_contact_tools.py']

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(out,phase):
    dest=out/'source-snapshots'/phase;dest.mkdir(parents=True,exist_ok=False)
    hashes={p:sha(ROOT/p) for p in SOURCES}
    for name in hashes:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_json(dest/'hashes.json',hashes)
    return hashes


def audit_sources(out,phase,hashes):
    result={'passed':all(sha(ROOT/p)==h for p,h in hashes.items())}
    write_json(out/'source-snapshots'/phase/'completion-audit.json',result)
    if not result['passed']:raise RuntimeError('Sources changed during a comparison phase')


def raw_hinge_pixels(rows,native,loader):
    cache={};scores=[]
    for row in rows:
        if not row['contract_valid']:continue
        case=row['source_private']['case'];regions=row['arguments']['regions']
        if case not in cache:cache[case]=loader(case)
        cameras,_,_=cache[case];camera=cameras[regions['camera']]
        origin=np.asarray(native[case]['native_hinge_anchor_world_m']);axis=np.asarray(native[case].get('native_axis_world',[0,0,1]))
        world=np.vstack([origin-.1*axis,origin+.1*axis]);c2w=np.asarray(camera['camera_to_world']);xyz=(world-c2w[:3,3])@c2w[:3,:3]
        uvh=xyz@np.asarray(camera['intrinsic']).T;uv=uvh[:,:2]/uvh[:,2:]
        a,b=uv;direction=b-a;points=np.asarray(regions['hinge_pixels_xy']);delta=points-a
        distances=np.abs(direction[0]*delta[:,1]-direction[1]*delta[:,0])/np.linalg.norm(direction)
        proposed=points[1]-points[0];cos=abs(np.dot(proposed,direction))/(np.linalg.norm(proposed)*np.linalg.norm(direction))
        scores.append({'id':row['id'],'mean_distance_to_projected_native_hinge_line_px':float(distances.mean()),
            'image_line_angle_error_deg':float(np.degrees(np.arccos(np.clip(cos,0,1)))),
            'observer_only':True,'supplied_to_model':False})
    return scores


def replay(out,endpoint):
    phase='replay27b';hashes=freeze(out,phase);protocol=read(out/'protocol.json')
    study=out/'study';study.mkdir(exist_ok=False);shutil.copytree(BASE/'study/public',study/'public')
    info=LocalChatClient(endpoint,'qwen27b',study/'preflight').server_info();write_json(study/'server-info.json',info)
    baseline={r['id']:r for r in read(BASE/'study/results.json')};rows=[]
    for ident in protocol['initial_cases']:
        old=BASE/'study/responses'/ident/'call001';request=read(old/'request.json')
        folder=study/'responses'/ident;client=LocalChatClient(endpoint,'qwen27b',folder,seed=request['seed'])
        row={'id':ident,'source_private':baseline[ident]['source_private'],'generation_completed':False,
             'contract_valid':False,'baseline':baseline[ident]}
        write_json(study/'status.json',{'phase':'running','case':ident,'current_call':f'study/responses/{ident}/call001','completed':len(rows)})
        try:
            response,timing=client.complete(request['messages'],request['tools'],study/'public')
            row['timing']=timing;row['generation_completed']=timing['finish_reason'] in {'stop','tool_calls'}
            row['repetition']=repetition(response.get('reasoning',response.get('reasoning_content','')))
            if not row['generation_completed']:row['failure']='output_token_limit' if timing['finish_reason']=='length' else 'unexpected_finish_reason'
            elif len(response.get('tool_calls',[]))!=1:row['failure']='missing_or_multiple_ground_calls'
            else:
                name,args=parse_call(response['tool_calls'][0],request['tools']);validate_operation(name,args,{'frontview':{'width':512,'height':512},'agentview':{'width':512,'height':512}})
                row.update(contract_valid=True,operation=name,arguments=args)
        except Exception as e:row.update(error=f'{type(e).__name__}: {e}',timing=client.calls[-1] if client.calls else {})
        actual=read(folder/'call001/request.json');expected={**request,'model':'qwen27b'}
        row['input_equivalence']={'only_model_changed':actual==expected,'changed_fields':sorted(k for k in request.keys()|actual.keys() if request.get(k)!=actual.get(k))}
        write_json(folder/'result.json',row)
        if actual!=expected:raise RuntimeError('Unexpected input difference')
        rows.append(row);write_json(study/'results.json',rows)
        print(json.dumps({'id':ident,'valid':row['contract_valid'],'seconds':row.get('timing',{}).get('wall_seconds'),'usage':row.get('timing',{}).get('usage'),'error':row.get('error'),'failure':row.get('failure')}),flush=True)
    geometry=evaluate(rows,out/'geometry27b-private.json')
    native={r['trial']:r for r in read(ROOT/'results/astra-luna-comparison-20261023/grounding-diagnostics-private.json')}
    write_json(out/'raw-hinge27b-private.json',raw_hinge_pixels(rows,native,load_case))
    write_json(out/'raw-hinge9b-private.json',raw_hinge_pixels(list(baseline.values()),native,load_case))
    result={'requests':len(rows),'valid':sum(r['contract_valid'] for r in rows),'completed':sum(r['generation_completed'] for r in rows),
        'tool_gate_passed':sum(r['contract_valid'] for r in rows)>=4,'fit_accepted':sum(r['geometry_accepted'] for r in geometry),
        'model_seconds':sum(r['timing']['wall_seconds'] for r in rows)}
    write_json(study/'summary.json',result);write_json(study/'status.json',{'phase':'complete',**result});audit_sources(out,phase,hashes)
    print(json.dumps(result),flush=True)


def matched(out,endpoint):
    if not read(out/'study/summary.json')['tool_gate_passed']:raise ValueError('Registered tool gate did not pass')
    phase='matched27b';hashes=freeze(out,phase);dest=out/phase;dest.mkdir();rows=[]
    for ident,case in [('fresh01','state020-r1-luna'),('fresh02','state035-r1-luna')]:
        result=run_case(dest,ident,case,endpoint,max_turns=16,model_name='qwen27b')
        before=read(BASE/'tool-assisted-retest'/ident/'model-calls/call001/request.json')
        after=read(dest/ident/'model-calls/call001/request.json')
        result['initial_input_only_model_changed']=after=={**before,'model':'qwen27b'}
        if not result['initial_input_only_model_changed']:raise RuntimeError('Matched tool initial prompt or schemas changed')
        result['baseline']=read(BASE/'tool-assisted-retest'/ident/'summary.json');rows.append(result)
        write_json(dest/'summary.json',{'cases':rows});write_json(dest/ident/'summary.json',result)
    write_json(dest/'status.json',{'phase':'complete','cases':len(rows)});audit_sources(out,phase,hashes)


def fresh(out,endpoint27,endpoint9):
    if not read(out/'study/summary.json')['tool_gate_passed']:raise ValueError('Registered tool gate did not pass')
    phase='fresh';hashes=freeze(out,phase);fixtures=read(out/'fresh-fixtures-private.json');rows=[]
    def loader(case):return load_case(case,trial_root=ROOT/fixtures[case]['trial_root'])
    def observer(records,path):return evaluate(records,path,case_loader=loader,native_records=fixtures)
    for model in ['qwen9b','qwen27b']:
        (out/('fresh-'+model)).mkdir()
    for case,model in read(out/'protocol.json')['fresh_order']:
        ident=case.split('-')[0];dest=out/('fresh-'+model)
        row=run_case(dest,ident,case,endpoint27 if model=='qwen27b' else endpoint9,max_turns=16,
            model_name=model,trial_root=ROOT/fixtures[case]['trial_root'],case_loader=loader,observer=observer)
        row.update(model=model,case=case);rows.append(row);write_json(out/'fresh-summary.json',{'cases':rows})
        write_json(dest/'summary.json',{'cases':[r for r in rows if r['model']==model]})
    for model in ['qwen9b','qwen27b']:write_json(out/('fresh-'+model)/'status.json',{'phase':'complete'})
    paired=[]
    for case in fixtures:
        ident=case.split('-')[0];a=read(out/'fresh-qwen9b'/ident/'model-calls/call001/request.json');b=read(out/'fresh-qwen27b'/ident/'model-calls/call001/request.json')
        paired.append({'case':case,'only_model_differs':{**a,'model':'qwen27b'}==b})
    write_json(out/'fresh-input-pair-audit.json',{'cases':paired,'passed':all(r['only_model_differs'] for r in paired)})
    audit_sources(out,phase,hashes)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=['replay','matched','fresh'],required=True)
    p.add_argument('--endpoint27',default='http://127.0.0.1:8128/v1');p.add_argument('--endpoint9',default='http://127.0.0.1:8127/v1');a=p.parse_args()
    if a.phase=='replay':replay(a.run.resolve(),a.endpoint27)
    elif a.phase=='matched':matched(a.run.resolve(),a.endpoint27)
    else:fresh(a.run.resolve(),a.endpoint27,a.endpoint9)
