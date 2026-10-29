"""Response limits must reach the request and survive the surrounding runner."""
import io
import json

import pytest

from vlm_tamp.local_agent import LocalChatClient,run_actor
from vlm_tamp.local_context_budget import ContextBudgetExceeded


class Stream:
    def __init__(self,broken=False):self.broken=broken
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"reasoning":"unfinished diagnostic text"},"finish_reason":null}]}\n'
        if self.broken:raise TimeoutError('simulated stalled stream')
        yield b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"completion_tokens":8}}\n'
        yield b'data: [DONE]\n'


class Opener:
    def __init__(self,broken=False,capacity=65536,prompt_tokens=1658):
        self.broken=broken;self.capacity=capacity;self.prompt_tokens=prompt_tokens;self.calls=[]
    def open(self,request,timeout):
        self.calls.append((request,timeout))
        if isinstance(request,str):
            return io.BytesIO(json.dumps({'data':[{'id':'qwen9b','max_model_len':self.capacity}]}).encode())
        if request.full_url.endswith('/tokenize'):
            return io.BytesIO(json.dumps({'count':self.prompt_tokens,'max_model_len':self.capacity}).encode())
        return Stream(self.broken)


def test_32k_output_and_900_second_default_reach_request_and_record(tmp_path):
    client=LocalChatClient('http://127.0.0.1:8127/v1','qwen9b',tmp_path/'calls')
    client.opener=Opener()
    response,timing=client.complete([{'role':'user','content':'test'}],[],tmp_path)
    request,timeout=next(c for c in client.opener.calls if not isinstance(c[0],str) and c[0].full_url.endswith('/chat/completions'))
    assert json.loads(request.data)['max_tokens']==32768
    assert json.loads(request.data)['temperature']==1.0
    assert json.loads(request.data)['presence_penalty']==1.5
    assert 899<timeout<=900
    assert timing['requested_max_output_tokens']==32768 and timing['request_timeout_seconds']==900
    assert response['content']=='done'


def test_explicit_historical_budget_remains_4096(tmp_path):
    client=LocalChatClient('http://localhost/v1','qwen9b',tmp_path/'calls',max_tokens=4096,
                           temperature=.6,presence_penalty=None)
    client.opener=Opener()
    _,timing=client.complete([{'role':'user','content':'test'}],[],tmp_path,timeout=240)
    request=next(c[0] for c in client.opener.calls if not isinstance(c[0],str) and c[0].full_url.endswith('/chat/completions'))
    assert json.loads(request.data)['max_tokens']==4096
    assert timing['request_timeout_seconds']==240


def test_sampling_override_reaches_generation_without_changing_prompt_admission(tmp_path):
    from vlm_tamp.local_context_budget import tokenize_payload
    client=LocalChatClient('http://localhost/v1','qwen9b',tmp_path/'calls',
                           temperature=1.,presence_penalty=1.5)
    client.opener=Opener()
    client.complete([{'role':'user','content':'test'}],[],tmp_path)
    request=next(c[0] for c in client.opener.calls if c[0].full_url.endswith('/chat/completions'))
    body=json.loads(request.data)
    assert body['temperature']==1. and body['presence_penalty']==1.5
    assert body['max_tokens']==32768 and body['chat_template_kwargs']['enable_thinking']
    historical={**body,'temperature':.6};historical.pop('presence_penalty')
    assert tokenize_payload(body)==tokenize_payload(historical)
    actual=json.loads((tmp_path/'calls/call001/request.json').read_text())
    assert actual['presence_penalty']==1.5


def test_explicit_legacy_sampling_preserves_historical_request_body(tmp_path):
    client=LocalChatClient('http://localhost/v1','qwen9b',tmp_path/'calls',
                           temperature=.6,presence_penalty=None)
    client.opener=Opener()
    client.complete([{'role':'user','content':'test'}],[],tmp_path)
    body=json.loads((tmp_path/'calls/call001/request.json').read_text())
    assert body['temperature']==.6 and 'presence_penalty' not in body


@pytest.mark.parametrize('settings',[
    {'temperature':True},{'temperature':-1},{'temperature':float('nan')},
    {'temperature':3},{'presence_penalty':False},{'presence_penalty':float('inf')},
    {'presence_penalty':2.1},
])
def test_bad_sampling_configuration_is_rejected(settings,tmp_path):
    with pytest.raises(ValueError,match='must be finite'):
        LocalChatClient('http://localhost/v1','qwen9b',tmp_path,**settings)


def test_output_budget_requires_input_headroom_on_real_server_capacity(tmp_path):
    client=LocalChatClient('http://localhost/v1','qwen9b',tmp_path)
    client.opener=Opener(capacity=32768)
    with pytest.raises(ValueError,match='leave room for input'):client.server_info()
    client.opener=Opener(capacity=65536)
    assert client.server_info()['max_model_len']==65536


def test_partial_stream_is_retained_but_not_returned_as_completed(tmp_path):
    client=LocalChatClient('http://localhost/v1','qwen9b',tmp_path/'calls')
    client.opener=Opener(broken=True)
    with pytest.raises(TimeoutError):client.complete([{'role':'user','content':'test'}],[],tmp_path)
    folder=tmp_path/'calls/call001'
    partial=json.loads((folder/'partial-response.json').read_text())
    assert partial['complete'] is False
    assert partial['message']['reasoning']=='unfinished diagnostic text'
    assert not (folder/'response.json').exists()
    assert json.loads((folder/'timing.json').read_text())['usage'] is None


def test_context_overflow_blocks_generation_without_reducing_32k(tmp_path):
    client=LocalChatClient('http://localhost/v1','qwen9b',tmp_path/'calls')
    client.opener=Opener(prompt_tokens=40000)
    with pytest.raises(ContextBudgetExceeded):
        client.complete([{'role':'user','content':'test'}],[],tmp_path)
    assert len(client.opener.calls)==1
    assert client.opener.calls[0][0].full_url.endswith('/tokenize')
    folder=tmp_path/'calls/call001';budget=json.loads((folder/'context-budget.json').read_text())
    assert budget['available_output_tokens']==25536 and budget['requested_output_tokens']==32768
    assert not budget['output_budget_reduced']
    assert json.loads((folder/'timing.json').read_text())['generation_request_sent'] is False


def test_truncation_never_executes_even_a_valid_looking_tool_call(tmp_path):
    from vlm_tamp.robot_contracts import operation_schemas
    class Client:
        calls=[]
        request_timeout_seconds=900
        def complete(self,*args,**kwargs):
            return {'role':'assistant','tool_calls':[{'id':'x','function':{'name':'execute','arguments':'{"plan_id":"p003"}'}}]}, {'finish_reason':'length'}
    class Tools:
        workspace=tmp_path
        def schemas(self):return [operation_schemas()['execute']]
        def dispatch(self,*args):pytest.fail('A truncated reply must never execute a tool')
    result=run_actor(Client(),Tools(),'test',tmp_path,seconds=1800)
    assert result['termination_reason']=='model_output_limit' and result['tool_calls']==0


@pytest.mark.parametrize('seconds,minimum',[(1800,899),(100,99)])
def test_actor_uses_configured_request_budget_bounded_by_episode(tmp_path,seconds,minimum):
    (tmp_path/'robot_data').mkdir()
    class Client:
        calls=[]
        request_timeout_seconds=900
        observed=None
        def complete(self,*args,timeout):
            self.observed=timeout
            (tmp_path/'robot_data/episode-status.json').write_text('{"task_success":true}')
            return {'role':'assistant','content':'done'},{'finish_reason':'stop'}
    class Tools:
        workspace=tmp_path
        def schemas(self):return []
    client=Client()
    result=run_actor(client,Tools(),'test',tmp_path,seconds=seconds)
    assert minimum<client.observed<=min(900,seconds)
    assert result['termination_reason']=='simulator_success'


@pytest.mark.parametrize('budget',[0,-1,True,32768.5])
def test_invalid_output_budget_rejected(budget,tmp_path):
    with pytest.raises(ValueError,match='positive integer'):
        LocalChatClient('http://localhost/v1','qwen9b',tmp_path,max_tokens=budget)
