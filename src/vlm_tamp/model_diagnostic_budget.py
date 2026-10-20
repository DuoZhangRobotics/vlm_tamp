"""Separate, model-priced $2 ledger for the approved Astra/Luna diagnostic.

Existing Luna-only ledgers and defaults are untouched. Reservations survive
timeouts and process interruption; no failed or incomplete call is retried.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
from .rekep_libero.api import _atomic,_canonical,ApiError,BudgetError,UncertainRequest

PRICES={'gpt-6-luna':{'input_ceiling_per_million':.125,'output_per_million':.5},
        'gpt-6-astra':{'input_ceiling_per_million':12.5,'output_per_million':50.}}
SOURCE='https://developers.openai.com/api/docs/models/gpt-6-astra'

def validate(arguments):
    if (arguments.get('model') not in PRICES or arguments.get('reasoning')!={'effort':'high'}
        or arguments.get('max_output_tokens')!=12000 or arguments.get('store') is not False
        or arguments.get('service_tier')!='default'):
        raise ApiError('Only the frozen model/high/12000/default diagnostic is supported')
    if set(arguments)!={'model','reasoning','max_output_tokens','store','service_tier','input'}:
        raise ApiError('Unsupported request fields')
    messages=arguments['input']
    if len(messages)!=1 or messages[0]['role']!='user':raise ApiError('Expected one saved observation request')
    content=messages[0]['content']
    if (len(content)!=2 or content[0]['type']!='input_text' or not isinstance(content[0]['text'],str)
        or content[1]['type']!='input_image' or content[1]['detail']!='high'
        or not content[1]['image_url'].startswith('data:image/png;base64,')):
        raise ApiError('Expected saved text and one high-detail PNG')
    import base64,io
    from PIL import Image
    with Image.open(io.BytesIO(base64.b64decode(content[1]['image_url'].split(',',1)[1],validate=True))) as image:
        if image.format!='PNG' or image.size!=(1024,1024):raise ApiError('Frozen feedback image must be 1024 square')

def reservation(arguments):
    validate(arguments);rates=PRICES[arguments['model']]
    tokens=len(arguments['input'][0]['content'][0]['text'].encode())+512+8192
    if tokens>100000:raise BudgetError('Diagnostic input exceeds its short-context bound')
    return (tokens*rates['input_ceiling_per_million']+12000*rates['output_per_million'])/1e6

def usage_cost(raw,model,reserved):
    u=raw.get('usage') or {};i=u.get('input_tokens');o=u.get('output_tokens')
    if type(i) is not int or type(o) is not int or i<0 or o<0:return reserved,False
    rates=PRICES[model]
    return (i*rates['input_ceiling_per_million']+o*rates['output_per_million'])/1e6,True

class DiagnosticLedger:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'ledger.json'

    def _read(self):
        expected={'version':'astra-recovery-diagnostic-1','budget_usd':2.,'budget_scope':'experiment_run',
                  'model_prices':PRICES,'pricing_source':SOURCE,'pricing_verified_date':'2026-10-18'}
        if not self.path.exists():return {**expected,'requests':{}}
        saved=json.loads(self.path.read_text())
        if any(saved.get(k)!=v for k,v in expected.items()):raise BudgetError('Frozen ledger policy changed')
        return saved

    def _save(self,data):
        amounts=[x.get('cost_usd',x['reservation_usd']) for x in data['requests'].values()]
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in amounts):
            raise BudgetError('Invalid saved accounting')
        data.update(conservative_spend_usd=sum(amounts),remaining_usd=max(0.,2.-sum(amounts)),
                    paid_attempts=len(data['requests']))
        _atomic(self.path,data)

    def snapshot(self):
        with (self.directory/'ledger.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            data=self._read();self._save(data);return data

    def perform(self,arguments,request_id,*,allow_paid=False,client=None):
        if not isinstance(request_id,str) or not request_id:raise ApiError('Stable request identity required')
        reserved=reservation(arguments);digest=hashlib.sha256(_canonical(arguments).encode()).hexdigest()
        with (self.directory/'ledger.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            data=self._read();self._save(data)
            prior=data['requests'].get(request_id)
            if prior is None:
                prior=next((x for x in data['requests'].values() if x['request_sha256']==digest),None)
            response=self.directory/f'response-{digest}.json'
            if prior:
                if prior['request_sha256']!=digest:raise ApiError('Changed inputs under an existing request identity')
                if response.exists():
                    raw=json.loads(response.read_text())
                    cost,known=usage_cost(raw,arguments['model'],prior['reservation_usd'])
                    prior.update(status='completed',response_status=raw.get('status'),cost_usd=cost,usage_known=known)
                    self._save(data);return raw
                raise UncertainRequest('Earlier attempt has no durable response; no automatic retry')
            if not allow_paid:raise ApiError('Paid diagnostic calls were not enabled')
            if data['conservative_spend_usd']+reserved>2.+1e-12:
                raise BudgetError('Shared two-dollar experiment cap cannot admit this request')
            entry={'model':arguments['model'],'request_sha256':digest,'reservation_usd':reserved,'status':'pending'}
            data['requests'][request_id]=entry
            _atomic(self.directory/f'request-{digest}.json',arguments);self._save(data)
            try:
                if client is None:
                    from openai import OpenAI
                    client=OpenAI(base_url='https://api.openai.com/v1',timeout=240,max_retries=0)
                raw=client.responses.create(**arguments).model_dump(mode='json')
                _atomic(response,raw)
            except Exception as error:
                entry.update(status='uncertain',error_type=type(error).__name__);self._save(data)
                raise UncertainRequest('Diagnostic attempt failed; full reservation retained and no automatic retry') from error
            cost,known=usage_cost(raw,arguments['model'],reserved)
            entry.update(status='completed',response_status=raw.get('status'),cost_usd=cost,usage_known=known)
            self._save(data);return raw
