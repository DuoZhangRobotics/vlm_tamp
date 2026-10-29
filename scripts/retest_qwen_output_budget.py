"""Replay three previously truncated primary requests with 32k output allowance.

No correction turns or new task hints: saved messages, tool schemas, images,
sampling settings and seeds are preserved. Context and time headroom are recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.local_agent import LocalChatClient,parse_call,write_json
from vlm_tamp.robot_contracts import validate_operation

CASES=('vision1-r1','vision2-r2','vision3-r2')


def common_prefix(a,b):
    i=0
    for x,y in zip(a,b):
        if x!=y:break
        i+=1
    return i


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--endpoint',default='http://127.0.0.1:8127/v1')
    parser.add_argument('--max-output-tokens',type=int,default=32768)
    parser.add_argument('--request-seconds',type=float,default=900)
    args=parser.parse_args();out=args.output.resolve()
    if args.max_output_tokens<32768:raise ValueError('This study requires at least 32,768 output tokens')
    source=ROOT/'results/robot-io-contract-20261024/qwen-retest'
    rows={row['id']:row for row in json.loads((source/'results.json').read_text())}
    preflight=LocalChatClient(args.endpoint,'qwen9b',out/'preflight',max_tokens=args.max_output_tokens,
                              request_timeout_seconds=args.request_seconds,temperature=.6,presence_penalty=None)
    info=preflight.server_info()
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'public',out/'public')
    protocol={'model':'Qwen/Qwen3.5-9B','checkpoint_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'max_output_tokens':args.max_output_tokens,'request_seconds':args.request_seconds,
        'server_info':info,'old_context_tokens':32768,'actual_context_tokens':info['max_model_len'],
        'source_study':str(source.relative_to(ROOT)),'cases':list(CASES),'requests_per_case':1,
        'selection':'Exactly the three primary requests that ended at 4096 reasoning tokens without a tool call.',
        'changed_request_fields':['max_tokens'],'unchanged':'Messages, images, tools, model weights, seed, sampling and thinking mode.',
        'context_time_changes':'Server context and request timeout enlarged to support the output budget.',
        'luna_comparison':'Luna CLI launches did not set a token cap; its effective per-response subscription limit is not recorded. This is not verified token-budget parity.',
        'physical_motion':False,'memory_updates':False,'automatic_retries':0,
        'image_sha256':{str(p.relative_to(out/'public')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/'public').rglob('*.png'))},
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'src/vlm_tamp/local_agent.py',ROOT/'src/vlm_tamp/robot_contracts.py',ROOT/'scripts/serve_local_qwen.py']}}
    write_json(out/'protocol.json',protocol)
    for name in protocol['source_sha256']:
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    results=[]
    sizes={'frontview':{'width':512,'height':512},'agentview':{'width':512,'height':512}}
    for ident in CASES:
        old_folder=source/'responses'/ident/'call001'
        old_request=json.loads((old_folder/'request.json').read_text())
        old_response=json.loads((old_folder/'response.json').read_text())
        old_timing=json.loads((old_folder/'timing.json').read_text())
        if old_timing.get('finish_reason')!='length' or old_timing['usage']['completion_tokens']!=4096:
            raise ValueError('Registered source case is no longer the expected truncated request')
        folder=out/'responses'/ident
        client=LocalChatClient(args.endpoint,'qwen9b',folder,max_tokens=args.max_output_tokens,
                              request_timeout_seconds=args.request_seconds,seed=old_request['seed'],
                              thinking=old_request['chat_template_kwargs']['enable_thinking'],
                              temperature=old_request['temperature'],presence_penalty=old_request.get('presence_penalty'))
        row={'id':ident,'source_private':rows[ident]['source_private'], 'contract_valid':False,
             'generation_completed':False,'old_usage':old_timing['usage'],'old_seconds':old_timing['wall_seconds']}
        write_json(out/'status.json',{'phase':'running','current_case':ident,'completed':len(results),'planned':len(CASES)})
        try:
            response,timing=client.complete(old_request['messages'],old_request['tools'],out/'public')
            row['timing']=timing
            row['generation_completed']=timing['finish_reason']!='length'
            old_reason=old_response.get('reasoning',old_response.get('reasoning_content',''))
            new_reason=response.get('reasoning',response.get('reasoning_content',''))
            prefix=common_prefix(old_reason,new_reason)
            row['reasoning_prefix_comparison']={'old_characters':len(old_reason),'common_prefix_characters':prefix,
                'old_text_is_exact_prefix':new_reason.startswith(old_reason),
                'scope':'Text comparison, not a token-level determinism guarantee.'}
            calls=response.get('tool_calls',[])
            if not row['generation_completed']:
                row['failure']='output_token_limit'
            elif len(calls)!=1:
                row['failure']='No single complete ground call was returned'
            else:
                name,payload=parse_call(calls[0],old_request['tools'])
                validate_operation(name,payload,sizes)
                row.update(contract_valid=True,operation=name,arguments=payload)
        except Exception as error:
            row.update(error=f'{type(error).__name__}: {error}',timing=client.calls[-1] if client.calls else None)
        actual=json.loads((folder/'call001/request.json').read_text())
        differences=sorted(k for k in old_request.keys()|actual.keys() if old_request.get(k)!=actual.get(k))
        row['input_equivalence']={'changed_fields':differences,'only_output_budget_changed':differences==['max_tokens']}
        if differences!=['max_tokens']:raise ValueError('Unexpected model-input change; preserve evidence and stop')
        results.append(row);write_json(folder/'result.json',row);write_json(out/'results.json',results)
        print(json.dumps({'id':ident,'completed':row['generation_completed'],'contract_valid':row['contract_valid'],
                          'usage':(row.get('timing') or {}).get('usage'),'seconds':(row.get('timing') or {}).get('wall_seconds'),
                          'error':row.get('error'),'failure':row.get('failure')}),flush=True)
    write_json(out/'summary.json',{'requests':len(results),'generation_completed':sum(r['generation_completed'] for r in results),
        'contract_valid':sum(r['contract_valid'] for r in results),
        'output_limit_hits':sum(r.get('failure')=='output_token_limit' for r in results),
        'errors':sum('error' in r for r in results),'model_seconds':sum((r.get('timing') or {}).get('wall_seconds',0) for r in results),
        'scope':'Selected truncated cases only. No physical task-success or matched Luna comparison claimed.'})
    write_json(out/'status.json',{'phase':'complete','completed':len(results),'planned':len(CASES)})


if __name__=='__main__':main()
