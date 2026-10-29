import json
from pathlib import Path

import pytest

from vlm_tamp.local_agent import LocalChatClient, PublicTools, encode_images, merge_chunk, parse_call, public_path, run_actor
from vlm_tamp.robot_contracts import operation_schemas, RequestValidationError


def test_public_paths_reject_escape_and_symlink(tmp_path):
    work = tmp_path / 'public'
    work.mkdir()
    secret = tmp_path / 'private.txt'
    secret.write_text('private')
    (work / 'link').symlink_to(secret)
    for path in ['../private.txt', str(secret), 'link']:
        with pytest.raises(ValueError):
            public_path(work, path)
    assert public_path(work, 'findings.txt') == work / 'findings.txt'


def test_tools_validate_before_execution():
    schemas=[operation_schemas()['execute']]
    call = {'function': {'name': 'execute', 'arguments': json.dumps({'arguments': '{"plan_id":"p003"}'})}}
    with pytest.raises(RequestValidationError):
        parse_call(call,schemas)
    call['function']['arguments'] = json.dumps({'plan_id': 'p003', 'gripper': -1})
    assert parse_call(call,schemas)[1]['gripper'] == -1
    call['function']['arguments']='{"plan_id":'
    with pytest.raises(RequestValidationError) as exc:parse_call(call,schemas)
    assert exc.value.request_diagnostic['reason']=='invalid_json'
    assert exc.value.request_diagnostic['physical_motion_issued'] is False


def test_stream_fragments_preserve_tool_arguments():
    message, calls = {}, {}
    merge_chunk(message, calls, {'reasoning': 'brief '})
    merge_chunk(message, calls, {'tool_calls': [{'index': 0, 'id': 'call-x', 'function': {'name': 'execute', 'arguments': '{"plan_id":'}}]})
    merge_chunk(message, calls, {'tool_calls': [{'index': 0, 'function': {'arguments': '"p003","gripper":-1}'}}]})
    assert parse_call(calls[0],[operation_schemas()['execute']]) == ('execute', {'plan_id':'p003','gripper':-1})
    assert message['reasoning'] == 'brief '


def test_local_endpoint_required(tmp_path):
    for endpoint in ['https://example.com/v1', 'http://127.0.0.1.evil/v1', 'http://user:pass@localhost/v1']:
        with pytest.raises(ValueError):
            LocalChatClient(endpoint, 'qwen9b', tmp_path)


def test_original_image_bytes_and_messages_preserved(tmp_path):
    from PIL import Image
    image = tmp_path / 'image.png'
    Image.new('RGB', (8, 8), (1, 2, 3)).save(image)
    messages = [{'role': 'user', 'content': [{'type': 'local_image', 'path': 'image.png'}]}]
    encoded, inventory = encode_images(messages, tmp_path)
    assert messages[0]['content'][0]['type'] == 'local_image'
    import base64
    assert base64.b64decode(encoded[0]['content'][0]['image_url']['url'].split(',')[1]) == image.read_bytes()
    assert inventory[0]['bytes'] == image.stat().st_size


def test_supplied_memory_cannot_be_overwritten(tmp_path):
    guide = tmp_path / 'guide.json'
    guide.write_text('{}')
    tools = PublicTools(tmp_path, extra_readonly=[guide])
    with pytest.raises(ValueError):
        tools.dispatch('write_file', {'path': 'guide.json', 'text': 'changed'})
    assert guide.read_text() == '{}'


def test_success_stops_additional_same_turn_actions(tmp_path):
    (tmp_path / 'robot_data').mkdir()
    class Client:
        calls = []
        request_timeout_seconds=900
        def complete(self, *args, **kwargs):
            call = lambda n: {'id': str(n), 'function': {'name': 'execute', 'arguments': '{"plan_id":"p003"}'}}
            return {'role': 'assistant', 'tool_calls': [call(1), call(2)]}, {'finish_reason': 'tool_calls'}
    class Tools:
        workspace = tmp_path
        count = 0
        def schemas(self):return [operation_schemas()['execute']]
        def dispatch(self, *args):
            self.count += 1
            (tmp_path / 'robot_data/episode-status.json').write_text('{"task_success":true}')
            return {'ok': True}, []
    tools = Tools()
    result = run_actor(Client(), tools, 'test', tmp_path, seconds=10)
    assert tools.count == 1
    assert result['termination_reason'] == 'simulator_success'


def test_named_robot_schemas_load_only_after_reading_help(tmp_path,monkeypatch):
    from vlm_tamp.compact_robot_interface import install
    from vlm_tamp.robot_contracts import GROUND_HELP
    install(tmp_path,'- reset(): fresh state.\n'+GROUND_HELP,attempts=1,initial_attempts=1,practice_mode=False)
    tools=PublicTools(tmp_path)
    names=lambda:{s['function']['name'] for s in tools.schemas()}
    assert 'reset' in names() and 'ground' not in names() and 'robot' not in names()
    monkeypatch.setattr(tools,'python',lambda *a,**k:pytest.fail('Help must not call the robot'))
    reply,_=tools.dispatch('robot_help',{'operation':'ground'})
    assert reply['named_function_enabled'] and 'ground' in names()
    schema=next(s for s in tools.schemas() if s['function']['name']=='ground')
    assert schema['function']['parameters']['required']==['regions']
    assert not schema['function']['parameters']['additionalProperties']


def test_full_reply_preserves_access_to_additional_camera_schemas(tmp_path):
    tools=PublicTools(tmp_path)
    full=tmp_path/'full.json'
    full.write_text(json.dumps({'ok':True,'result':{'image_sizes':{'frontview':{'width':512,'height':512},'side':{'width':640,'height':480}}}}))
    tools.record_image_sizes({'full_response':'full.json','result':{'image_sizes':{'frontview':{'width':512,'height':512}}}})
    assert tools.image_sizes['side']=={'width':640,'height':480}
