from types import SimpleNamespace
from unittest.mock import Mock
from PIL import Image
import pytest
from vlm_tamp.rekep_libero.api import request_arguments,ApiError,BudgetError,UncertainRequest
from vlm_tamp.model_diagnostic_budget import DiagnosticLedger,reservation,usage_cost

def args(tmp_path,model='gpt-6-astra',text='Saved observation'):
    image=tmp_path/'image.png';Image.new('RGB',(1024,1024)).save(image)
    request=request_arguments(image,text,12000);request['model']=model;return request

def client(raw):
    create=Mock(return_value=SimpleNamespace(model_dump=lambda **kw:raw))
    return SimpleNamespace(responses=SimpleNamespace(create=create))

def test_model_specific_reservations_and_usage(tmp_path):
    a=args(tmp_path);l=args(tmp_path,'gpt-6-luna')
    assert reservation(a)==pytest.approx(100*reservation(l))
    cost,known=usage_cost({'usage':{'input_tokens':4000,'output_tokens':2000}},'gpt-6-astra',1.)
    assert known and cost==pytest.approx(.15)

@pytest.mark.parametrize('key,value',[('model','unknown'),('reasoning',{'effort':'low'}),
    ('max_output_tokens',32000),('service_tier','priority'),('store',True)])
def test_reject_changed_request_policy(tmp_path,key,value):
    request=args(tmp_path);request[key]=value
    with pytest.raises(ApiError):reservation(request)

def test_shared_cap_uses_astra_rates_and_no_unknown_usage_discount(tmp_path):
    ledger=DiagnosticLedger(tmp_path/'ledger');c=client({'status':'completed'})
    ledger.perform(args(tmp_path,text='one'),'one',allow_paid=True,client=c)
    ledger.perform(args(tmp_path,text='two'),'two',allow_paid=True,client=c)
    with pytest.raises(BudgetError):ledger.perform(args(tmp_path,text='three'),'three',allow_paid=True,client=c)
    assert c.responses.create.call_count==2
    assert ledger.snapshot()['conservative_spend_usd']<=2

def test_uncertain_call_never_retried_and_reservation_survives(tmp_path):
    ledger=DiagnosticLedger(tmp_path/'ledger');c=client({});c.responses.create.side_effect=TimeoutError()
    request=args(tmp_path)
    with pytest.raises(UncertainRequest):ledger.perform(request,'one',allow_paid=True,client=c)
    with pytest.raises(UncertainRequest):DiagnosticLedger(tmp_path/'ledger').perform(request,'one',allow_paid=True,client=c)
    assert c.responses.create.call_count==1
    assert ledger.snapshot()['conservative_spend_usd']==reservation(request)

def test_cache_preserves_incomplete_response_without_new_attempt(tmp_path):
    ledger=DiagnosticLedger(tmp_path/'ledger');raw={'status':'incomplete','usage':{'input_tokens':4000,'output_tokens':12000}}
    c=client(raw);request=args(tmp_path)
    assert ledger.perform(request,'one',allow_paid=True,client=c)==raw
    assert ledger.perform(request,'alias',client=c)==raw
    assert c.responses.create.call_count==1
    assert ledger.snapshot()['requests']['one']['response_status']=='incomplete'
    with pytest.raises(ApiError):ledger.perform(args(tmp_path,text='changed'),'one',allow_paid=True,client=c)

def test_paid_inference_requires_runner_opt_in(tmp_path):
    c=client({})
    with pytest.raises(ApiError):DiagnosticLedger(tmp_path/'ledger').perform(args(tmp_path),'one',client=c)
    c.responses.create.assert_not_called()
