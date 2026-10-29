"""Reproduce the historical 4k shared-contract/named-schema retest.

Only schema rejections may trigger one correction turn. Geometry assessments
remain evaluator-only and never become hints or learning memories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.local_agent import LocalChatClient,parse_call,write_json
from vlm_tamp.robot_contracts import GROUND_HELP,RequestValidationError,operation_schemas,validate_operation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--endpoint',default='http://127.0.0.1:8127/v1')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'results/qwen9b-pilot-20261024/capability'
    cases=[case for case in json.loads((old/'cases-private.json').read_text()) if case['kind'] in {'contract','vision'}]
    public=out/'public';shutil.copytree(old/'public',public)
    sizes={'frontview':{'width':512,'height':512},'agentview':{'width':512,'height':512}}
    schemas=operation_schemas(sizes)
    old_results={row['id']:row for row in json.loads((old/'results.json').read_text())}
    protocol={'model':'Qwen/Qwen3.5-9B','revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'dtype':'bfloat16','vllm':'0.32.0','temperature':.6,'thinking':True,'seeds':[17,18],
        'max_output_tokens':4096,'tool_choice':'auto','constrained_output_decoding':False,
        'primary_responses':8,'maximum_schema_correction_turns_per_case':1,
        'changes':['Shared clarified ground help','Exact named operation schemas','Explicit null for unresolved image direction','Shared validation error feedback after an invalid first request'],
        'same_image_bytes':True,'same_model_settings':True,'new_memory_or_target_coordinates_supplied':False,
        'geometry_or_physical_feedback_supplied':False,'physical_motion':False,
        'comparability':'Primary calls compare interface packages on the same development images; correction turns are additional evidence, not replacements. Null is newly documented, so contract compliance is not proof of correct direction or localization.',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'src/vlm_tamp/robot_contracts.py',ROOT/'src/vlm_tamp/local_agent.py',ROOT/'scripts/prepare_subscription_pilot.py',ROOT/'scripts/serve_subscription_robot.py',ROOT/'src/vlm_tamp/handle_contact.py',ROOT/'src/vlm_tamp/compact_robot_interface.py']},
        'image_sha256':{str(p.relative_to(public)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(public.rglob('*.png'))}}
    write_json(out/'protocol.json',protocol)
    for name in protocol['source_sha256']:
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    results=[]
    for case in cases:
        operation='execute' if case['kind']=='contract' else 'ground'
        if operation=='ground':
            instruction=('Task: open the microwave. The two images are current frontview and agentview, each 512 by 512. '
                'Propose a ground request from your interpretation of the images by calling the provided ground function. '
                'Use one supplied camera and its original pixel coordinates. This is an offline proposal; no physical action or geometry fit will run.\nContract:\n'+GROUND_HELP)
        else:
            instruction=case['text'].replace('Use submit_request; no request will actually execute.','Call the provided execute function; no request will actually execute.')
        for rep in [1,2]:
            ident=f'{case["id"]}-r{rep}';folder=out/'responses'/ident
            content=[{'type':'text','text':instruction}]
            for picture in case['images']:
                content.extend([{'type':'text','text':picture['camera']},{'type':'local_image','path':picture['path']}])
            messages=[{'role':'system','content':'Use the provided function to submit your answer. Base it only on the supplied public evidence. Keep explanations brief.'},
                      {'role':'user','content':content}]
            # Preserve this historical diagnostic's registered 4k budget.
            client=LocalChatClient(args.endpoint,'qwen9b',folder,seed=16+rep,max_tokens=4096,
                                   temperature=.6,presence_penalty=None)
            attempts=[]
            for attempt in [1,2]:
                record={'attempt':attempt,'contract_valid':False}
                response=None
                try:
                    response,timing=client.complete(messages,[schemas[operation]],public,timeout=240)
                    calls=response.get('tool_calls',[])
                    if len(calls)!=1:
                        raise RequestValidationError(operation,'tool_call','incomplete_submission',
                            'Submit exactly one complete named function call. '+('The output token budget was exhausted.' if timing.get('finish_reason')=='length' else ''))
                    name,payload=parse_call(calls[0],[schemas[operation]])
                    validate_operation(name,payload,sizes)
                    record.update(contract_valid=True,operation=name,arguments=payload,timing=timing)
                    if operation=='execute':record['open_approach_correct']=payload.get('plan_id')=='p003' and payload.get('gripper')==-1
                except Exception as error:
                    feedback={'ok':False,'error_type':type(error).__name__,'error':str(error)[:1500]}
                    if hasattr(error,'request_diagnostic'):feedback['request_diagnostic']=error.request_diagnostic
                    record.update(feedback=feedback,timing=client.calls[-1] if client.calls else None)
                attempts.append(record)
                write_json(folder/'attempts.json',attempts)
                if record['contract_valid'] or response is None:break
                if attempt==1:
                    messages.append(response)
                    calls=response.get('tool_calls',[])
                    if calls and all(call.get('id') for call in calls):
                        for call in calls:
                            messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(record['feedback'])})
                    messages.append({'role':'user','content':'The submitted request was rejected before any geometry work or motion. Correct only the request using the documented contract. '+json.dumps(record['feedback'])})
            result={'id':ident,'case':case['id'],'kind':case['kind'],'replicate':rep,
                    'primary_contract_valid':attempts[0]['contract_valid'],
                    'final_contract_valid':attempts[-1]['contract_valid'],
                    'attempts':attempts,'old_gate_passed':old_results[ident]['passed'],
                    'source_private':case.get('source_private'),
                    'total_model_seconds':sum(x['timing']['wall_seconds'] for x in attempts if x.get('timing'))}
            results.append(result);write_json(folder/'result.json',result);write_json(out/'results.json',results)
            print(json.dumps({'id':ident,'primary_valid':result['primary_contract_valid'],'final_valid':result['final_contract_valid'],
                              'model_calls':len(attempts),'seconds':result['total_model_seconds']}),flush=True)
    summary={'primary_responses':len(results),'primary_contract_valid':sum(r['primary_contract_valid'] for r in results),
        'final_contract_valid':sum(r['final_contract_valid'] for r in results),
        'model_calls':sum(len(r['attempts']) for r in results),
        'vision_primary_valid':sum(r['primary_contract_valid'] for r in results if r['kind']=='vision'),
        'vision_final_valid':sum(r['final_contract_valid'] for r in results if r['kind']=='vision'),
        'open_approach_correct':sum(r['attempts'][-1].get('open_approach_correct',False) for r in results if r['kind']=='contract'),
        'scope':'Contract compliance only. No geometric or physical success claim.'}
    write_json(out/'summary.json',summary)


if __name__=='__main__':main()
