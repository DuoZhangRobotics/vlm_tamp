"""Match displayed feedback against the actual retained API request payloads."""
import base64
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.rekep_libero.api import request_arguments,_canonical
from vlm_tamp.coupled_storage import read_json,write_json

def main():
    run=ROOT/'results/vlm-recovery-frozen-20261018'
    ledger=ROOT/'results/api-runs/vlm-recovery-tools-20261018'
    verified=[]
    for p in sorted(run.glob('*-vlm/decisions/round-*/api-accounting.json')):
        account=read_json(p);folder=p.parent
        arguments=request_arguments(folder/'feedback-sheet.png',(folder/'prompt.txt').read_text(),32000)
        # The accounting record and durable ledger use different JSON spacing;
        # validate both hashes against their original serialization contracts.
        assert hashlib.sha256(json.dumps(arguments,sort_keys=True).encode()).hexdigest()==account['request_sha256']
        digest=hashlib.sha256(_canonical(arguments).encode()).hexdigest()
        request=read_json(ledger/f'request-{digest}.json')
        assert request==arguments
        content=request['input'][0]['content'];pictures=[x for x in content if x['type']=='input_image']
        assert len(pictures)==1 and pictures[0]['detail']=='high'
        assert base64.b64decode(pictures[0]['image_url'].split(',',1)[1])==(folder/'feedback-sheet.png').read_bytes()
        assert [x['text'] for x in content if x['type']=='input_text']==[(folder/'prompt.txt').read_text()]
        assert request['store'] is False and request['reasoning']['effort']=='high'
        raw=read_json(folder/'response.json')
        verified.append({'round':str(folder),'image_sha256':hashlib.sha256((folder/'feedback-sheet.png').read_bytes()).hexdigest(),
            'status':raw['status'],'usage':raw['usage'],'max_output_tokens':request['max_output_tokens']})
    output=ROOT/'results/vlm-recovery-tools-report';output.mkdir(exist_ok=True)
    write_json(output/'communication-audit.json',{'verified_requests':len(verified),
        'image_bytes_matched_actual_API_payload':True,'prompt_matched_actual_API_payload':True,'requests':verified})
    print({'actual_text_image_requests_verified':len(verified),'all_completed':all(r['status']=='completed' for r in verified),
           'maximum_output_tokens_used':max(r['usage']['output_tokens'] for r in verified)})

if __name__=='__main__':main()
