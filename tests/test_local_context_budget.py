"""Exact request-context admission, including image/template serialization."""
import copy
import io
import json
import urllib.error

import pytest

from vlm_tamp.local_context_budget import (
    ContextBudgetExceeded, admit_context,
    tokenize_payload, tokenizer_endpoint,
)


def generation_request():
    return {'model': 'qwen9b', 'messages': [
        {'role': 'system', 'content': 'Use the tool.\nKeep the evidence.'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Locate café ☕.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,TEST_IMAGE_BYTES'}}]}],
        'tools': [{'type': 'function', 'function': {'name': 'ground', 'parameters': {'type': 'object'}}}],
        'tool_choice': 'auto', 'parallel_tool_calls': False, 'max_tokens': 32768,
        'temperature': .6, 'top_p': .95, 'top_k': 20, 'seed': 17,
        'stream': True, 'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': True}}


class FakeOpener:
    def __init__(self, prompt_tokens=1658, context_tokens=65536, response=None):
        self.response = response if response is not None else {'count': prompt_tokens, 'max_model_len': context_tokens}
        self.calls = []

    def open(self, request, *, timeout):
        self.calls.append((request, timeout))
        return io.BytesIO(json.dumps(self.response).encode())


def test_serialized_tokenizer_request_matches_generation_content():
    body = generation_request()
    body.update(mm_processor_kwargs={'max_pixels': 262144}, media_io_kwargs={'image': {'image_mode': 'RGB'}})
    original = copy.deepcopy(body)
    opener = FakeOpener()
    budget = admit_context(opener, 'http://127.0.0.1:8127/v1', body)
    assert body == original
    assert len(opener.calls) == 1
    request, timeout = opener.calls[0]
    assert request.full_url == 'http://127.0.0.1:8127/tokenize'
    assert request.get_method() == 'POST' and timeout == 30
    actual = json.loads(request.data)
    for key in ['messages', 'tools', 'chat_template_kwargs', 'mm_processor_kwargs', 'media_io_kwargs']:
        assert actual[key] == body[key]
    assert actual['add_generation_prompt'] is True
    assert actual['continue_final_message'] is False
    assert actual['add_special_tokens'] is False
    assert 'max_tokens' not in actual and 'stream' not in actual
    assert budget['prompt_tokens'] == 1658 and budget['requested_output_tokens'] == 32768
    assert budget['headroom_after_full_output_tokens'] == 31110
    assert budget['accepted'] and not budget['output_budget_reduced']
    assert not any(key in budget for key in ['messages', 'images', 'tokens', 'tools'])


@pytest.mark.parametrize('prompt,context,accepted', [(32768,65536,True), (32769,65536,False), (65537,65536,False)])
def test_exact_boundary_and_overflow_never_shrink_output(prompt,context,accepted):
    body = generation_request();before = copy.deepcopy(body);opener = FakeOpener(prompt, context)
    if accepted:
        result = admit_context(opener, 'http://localhost:8127/v1/', body)
        assert result['headroom_after_full_output_tokens'] == 0
    else:
        with pytest.raises(ContextBudgetExceeded) as exc:
            admit_context(opener, 'http://localhost:8127/v1/', body)
        assert exc.value.context_budget['requested_output_tokens'] == 32768
        assert exc.value.context_budget['shortfall_tokens'] == prompt+32768-context
        assert not exc.value.context_budget['output_budget_reduced']
    assert body == before and len(opener.calls) == 1


def test_generation_and_prefill_flags_match_request():
    body = generation_request()
    body.update(add_generation_prompt=False, continue_final_message=True, add_special_tokens=True)
    payload = tokenize_payload(body)
    assert not payload['add_generation_prompt'] and payload['continue_final_message']
    assert payload['add_special_tokens']
    body['add_generation_prompt'] = True
    with pytest.raises(ValueError, match='cannot both'):
        tokenize_payload(body)


def test_deprecated_reasoning_alias_matches_chat_normalization_without_mutation():
    body = generation_request()
    message = {'role': 'assistant', 'content': '', 'reasoning_content': 'retained reasoning'}
    body['messages'].append(message)
    assert tokenize_payload(body)['messages'][-1] == {'role':'assistant','content':'','reasoning':'retained reasoning'}
    assert message['reasoning_content'] == 'retained reasoning'
    message['reasoning'] = 'canonical wins'
    assert tokenize_payload(body)['messages'][-1]['reasoning'] == 'canonical wins'


@pytest.mark.parametrize('response', [None, [], {'count':True,'max_model_len':65536},
    {'count':1,'max_model_len':False}, {'count':-1,'max_model_len':65536},
    {'count':2,'max_model_len':65536,'tokens':[1]}])
def test_bad_tokenizer_response_fails_closed(response):
    opener = FakeOpener();opener.response = response
    with pytest.raises(RuntimeError):
        admit_context(opener,'http://127.0.0.1:8127/v1',generation_request())


def test_tokenizer_http_errors_do_not_echo_image_data():
    class Broken:
        def open(self, request, **kwargs):
            raise urllib.error.HTTPError(request.full_url,422,'invalid',{},io.BytesIO(b'SECRET_IMAGE_CONTENT'))
    with pytest.raises(RuntimeError) as exc:
        admit_context(Broken(),'http://localhost:8127/v1',generation_request())
    assert 'HTTP 422' in str(exc.value) and 'SECRET' not in str(exc.value)


@pytest.mark.parametrize('mode', ['required','none',{'type':'function','function':{'name':'ground'}}])
def test_unverified_tool_rendering_modes_fail_before_http(mode):
    body = generation_request();body['tool_choice'] = mode;opener = FakeOpener()
    with pytest.raises(ValueError,match='automatic tool choice'):
        admit_context(opener,'http://localhost:8127/v1',body)
    assert not opener.calls


def test_unencoded_images_and_truncation_are_not_approximated():
    body = generation_request()
    body['messages'][1]['content'][1] = {'type':'local_image','path':'frame.png'}
    with pytest.raises(ValueError,match='Encode local images'):
        tokenize_payload(body)
    body = generation_request();body['truncate_prompt_tokens'] = 100
    with pytest.raises(ValueError,match='does not support'):
        tokenize_payload(body)


@pytest.mark.parametrize('endpoint', ['https://example.com/v1','http://localhost.evil/v1',
    'http://user:password@localhost/v1','http://localhost/v1?key=secret','http://localhost/v1/chat/completions'])
def test_tokenizer_endpoint_stays_on_the_local_preprocessing_route(endpoint):
    with pytest.raises(ValueError):tokenizer_endpoint(endpoint)
