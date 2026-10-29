"""Local chat-completions actor with explicit, audited public-workspace tools.

The model server has no agent tools. This controller mediates every call;
model-written Python runs in the existing OS sandbox, outside the model server.
"""
import base64
import copy
import hashlib
import json
import mimetypes
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from .subscription_agent import clean_environment, permission_arguments
from .public_analysis import command_environment, readonly_paths
from .robot_contracts import RequestValidationError, operation_schemas, validate_operation
from .local_context_budget import admit_context

DEFAULT_MAX_OUTPUT_TOKENS = 32768
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900
DEFAULT_TEMPERATURE = 1.0
DEFAULT_PRESENCE_PENALTY = 1.5


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def public_path(workspace, relative):
    root = Path(workspace).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Use a relative path inside the public workspace')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path leaves the public workspace')
    return path


def encode_images(messages, workspace):
    """Log filenames/hashes separately; only the HTTP body contains image bytes."""
    result, images = copy.deepcopy(messages), []
    for message in result:
        if not isinstance(message.get('content'), list):
            continue
        for item in message['content']:
            if item.get('type') != 'local_image':
                continue
            relative = item['path']
            path = public_path(workspace, relative)
            data = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in {'image/png', 'image/jpeg', 'image/webp'}:
                raise ValueError('Unsupported image type')
            images.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            item.clear()
            item.update(type='image_url', image_url={'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode()})
    return result, images


def merge_chunk(message, calls, delta):
    for field in ('content', 'reasoning_content', 'reasoning'):
        if delta.get(field):
            message[field] = (message.get(field) or '') + delta[field]
    for item in delta.get('tool_calls', []):
        index = item['index']
        call = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
        if item.get('id'):
            call['id'] = item['id']
        for key in ('name', 'arguments'):
            if item.get('function', {}).get(key):
                call['function'][key] += item['function'][key]


class LocalChatClient:
    def __init__(self, endpoint, model, output, *, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                 request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS, seed=17, thinking=True,
                 temperature=DEFAULT_TEMPERATURE, presence_penalty=DEFAULT_PRESENCE_PENALTY):
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('This pilot only accepts a local HTTP model endpoint')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Unexpected endpoint credentials/query')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        if isinstance(request_timeout_seconds,bool) or not isinstance(request_timeout_seconds,(int,float)) or not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError('request_timeout_seconds must be positive and finite')
        for name, value, lower, upper in [('temperature', temperature, 0, 2),
                                          ('presence_penalty', presence_penalty, -2, 2)]:
            if name == 'presence_penalty' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{name} must be finite and between {lower} and {upper}')
        self.endpoint, self.model, self.output = endpoint.rstrip('/'), model, Path(output)
        self.max_tokens, self.seed, self.thinking = max_tokens, seed, thinking
        # New runs use the checkpoint's general-thinking recipe. Historical
        # reproduction callers explicitly pass .6/None to preserve old bodies.
        self.temperature, self.presence_penalty = temperature, presence_penalty
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.calls = []

    def server_info(self):
        """Record the actual serving capacity rather than a hard-coded claim."""
        with self.opener.open(self.endpoint+'/models',timeout=30) as response:
            models=json.load(response)
        info=next((x for x in models.get('data',[]) if x.get('id')==self.model),None)
        if info is None:
            raise ValueError('Requested local model is not advertised by the server')
        capacity=info.get('max_model_len')
        if type(capacity) is not int or capacity <= self.max_tokens:
            raise ValueError('Server context must exceed the output budget and leave room for input; check --context-tokens')
        return info

    def complete(self, messages, tools, workspace, *, timeout=None):
        timeout=self.request_timeout_seconds if timeout is None else timeout
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('Use a positive finite request timeout')
        number = len(self.calls) + 1
        folder = self.output / f'call{number:03d}'
        folder.mkdir(parents=True, exist_ok=False)
        encoded, images = encode_images(messages, workspace)
        body = {'model': self.model, 'messages': encoded, 'tools': tools,
                'tool_choice': 'auto', 'parallel_tool_calls': False,
                'temperature': self.temperature, 'top_p': .95, 'top_k': 20, 'seed': self.seed,
                'max_tokens': self.max_tokens, 'stream': True,
                'stream_options': {'include_usage': True},
                'chat_template_kwargs': {'enable_thinking': self.thinking}}
        if self.presence_penalty is not None:
            body['presence_penalty'] = self.presence_penalty
        raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        logged = {**body, 'messages': messages}
        write_json(folder / 'request.json', logged)
        started = time.monotonic()
        meta = {'call': number, 'request_sha256': hashlib.sha256(raw).hexdigest(),
                'request_bytes': len(raw), 'images': images, 'started_monotonic': started,
                'requested_max_output_tokens':self.max_tokens,'request_timeout_seconds':timeout,
                'generation_request_sent':False,
                'timeout_scope':'Wall check on stream progress; the socket timeout bounds inactivity.'}
        message, calls, finish, usage = {'role': 'assistant', 'content': ''}, {}, None, None
        first = None
        last_progress=started
        received_events=0
        request = urllib.request.Request(self.endpoint + '/chat/completions', raw,
                                        {'Content-Type': 'application/json'})
        try:
            budget=admit_context(self.opener,self.endpoint,body,timeout=min(30,timeout))
            meta['context_budget']=budget
            write_json(folder/'context-budget.json',budget)
            remaining=timeout-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('Request budget expired during context admission')
            meta['generation_request_sent']=True
            with self.opener.open(request, timeout=remaining) as response:
                for line in response:
                    if time.monotonic() - started > timeout:
                        raise TimeoutError('Model request exceeded its wall budget')
                    if not line.startswith(b'data: '):
                        continue
                    payload = line[6:].strip()
                    if payload == b'[DONE]':
                        break
                    chunk = json.loads(payload)
                    received_events+=1
                    if chunk.get('error'):
                        raise RuntimeError(json.dumps(chunk['error']))
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    for choice in chunk.get('choices', []):
                        delta = choice.get('delta', {})
                        if first is None and any(delta.get(k) for k in ('content', 'reasoning_content', 'reasoning', 'tool_calls')):
                            first = time.monotonic()
                        merge_chunk(message, calls, delta)
                        finish = choice.get('finish_reason') or finish
                    if time.monotonic()-last_progress>=5:
                        last_progress=time.monotonic()
                        write_json(folder/'progress.json',{'complete':False,'elapsed_seconds':last_progress-started,
                            'received_stream_events':received_events,'tool_calls_seen':len(calls),
                            'reasoning_characters':sum(len(message.get(k) or '') for k in ('reasoning','reasoning_content')),
                            'content_characters':len(message.get('content') or ''),
                            'note':'Character/event counts are live progress, not measured token counts.'})
            if calls:
                message['tool_calls'] = [calls[i] for i in sorted(calls)]
            if finish is None:
                raise RuntimeError('Model stream ended without a finish reason')
            meta.update(finish_reason=finish, usage=usage,
                        first_output_seconds=first-started if first is not None else None)
            write_json(folder / 'response.json', message)
            return message, meta
        except urllib.error.HTTPError as error:
            meta['error'] = f'HTTP {error.code}: {error.read().decode(errors="replace")[:8000]}'
            raise RuntimeError(meta['error']) from error
        except Exception as error:
            meta['error'] = f'{type(error).__name__}: {error}'
            if hasattr(error,'context_budget'):
                meta['context_budget']=error.context_budget
                write_json(folder/'context-budget.json',error.context_budget)
            raise
        finally:
            if meta.get('error') and (calls or any(message.get(k) for k in ('content','reasoning','reasoning_content'))):
                partial=copy.deepcopy(message)
                if calls:partial['tool_calls']=[calls[i] for i in sorted(calls)]
                write_json(folder/'partial-response.json',{'complete':False,'message':partial,
                    'interpretation':'Partial diagnostic output only. It was not admitted for tool execution.'})
            meta['wall_seconds'] = time.monotonic() - started
            meta.setdefault('first_output_seconds',first-started if first is not None else None)
            meta.setdefault('usage',usage)
            write_json(folder / 'timing.json', meta)
            write_json(folder/'progress.json',{'complete':not bool(meta.get('error')),
                'elapsed_seconds':meta['wall_seconds'],'finish_reason':meta.get('finish_reason'),
                'usage':usage,'error':meta.get('error')})
            self.calls.append(meta)


def function(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False}}}


TOOLS = [
    function('read_file', 'Read a UTF-8 file in the public workspace. Use tools/OPERATION.txt for a robot contract.',
             {'path': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1},
              'max_lines': {'type': 'integer', 'minimum': 1, 'maximum': 300}}, ['path']),
    function('view_image', 'Inspect a public PNG/JPEG image; the next message supplies its original pixels. Use a workspace-relative path such as scratch/crop.png. Absolute paths and files in isolated /tmp are unavailable.',
             {'path': {'type': 'string', 'description': 'Relative path inside the public workspace.'}}, ['path']),
    function('run_python', 'Run Python in the isolated public workspace with NumPy, SciPy and Pillow. Save generated files under scratch/ (already exists), using relative paths; supplied inputs may be read-only. /tmp is isolated and not shared with view_image or later calls. Inspect array shapes before indexing. Print concise results. Network and private project data are unavailable.',
             {'code': {'type': 'string'}}, ['code']),
    function('robot_help', 'Read a robot operation contract and enable that named function for subsequent calls. No robot request is issued.',
             {'operation': {'type': 'string'}}, ['operation']),
    function('write_file', 'Save findings or analysis as a UTF-8 file in the public workspace.',
             {'path': {'type': 'string'}, 'text': {'type': 'string'}}, ['path', 'text']),
]


def parse_call(call, schemas=TOOLS):
    import jsonschema
    name = call['function']['name']
    schema = next((x['function']['parameters'] for x in schemas if x['function']['name'] == name), None)
    if schema is None:
        raise ValueError('Unknown tool: ' + name)
    try:
        args = json.loads(call['function']['arguments'])
    except (json.JSONDecodeError,TypeError) as error:
        raise RequestValidationError(name,'arguments','invalid_json',
            'Supply one complete JSON argument object. No robot request was issued.') from error
    if name in operation_schemas():
        validate_operation(name,args)
    try:
        jsonschema.Draft202012Validator(schema).validate(args)
    except jsonschema.ValidationError as error:
        field='arguments'+''.join(f'[{p}]' if isinstance(p,int) else '.'+p for p in error.absolute_path)
        raise RequestValidationError(name,field,'schema_mismatch',error.message[:1000]) from error
    return name, args


class PublicTools:
    def __init__(self, workspace, *, extra_readonly=(), live=True):
        self.workspace = Path(workspace).resolve()
        scratch = public_path(self.workspace, 'scratch')
        scratch.mkdir(exist_ok=True)
        self.readonly = [*readonly_paths(self.workspace), *map(Path, extra_readonly)]
        self.live = live
        index=self.workspace/'tools/schema-index.json'
        self.available=set(json.loads(index.read_text())['operations']) if index.is_file() else set()
        self.active=self.available & {'reset','observe','episode_status'}
        self.image_sizes={}

    def schemas(self):
        base=copy.deepcopy(TOOLS)
        helper=next(x for x in base if x['function']['name']=='robot_help')
        helper['function']['parameters']['properties']['operation']['enum']=sorted(self.available)
        named=operation_schemas(self.image_sizes)
        return base+[named[name] for name in sorted(self.active)]

    def enable_operation(self,name):
        if name not in self.available:
            raise ValueError('Operation is not enabled in this public workspace; see tools.txt')
        self.active.add(name)

    def record_image_sizes(self,response):
        if response.get('full_response'):
            path=public_path(self.workspace,response['full_response'])
            if path.is_file():
                response=json.loads(path.read_text())
        result=response.get('result',{})
        observation=result.get('observation',result)
        if isinstance(observation.get('image_sizes'),dict):
            self.image_sizes.update(observation['image_sizes'])

    def python(self, code, *, timeout=30, arguments=()):
        cmd = ['codex', 'sandbox', '-P', 'research-agent',
               *permission_arguments(readonly_paths=self.readonly), '-C', str(self.workspace),
               '--', '/usr/bin/python3', '-c', code, *arguments]
        proc = subprocess.Popen(cmd, cwd=self.workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=command_environment(self.workspace, clean_environment()),
                                start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
            return {'ok': False, 'error': 'Python time limit exceeded; a submitted robot request may still be pending.',
                    'stdout': stdout[-16000:], 'stderr': stderr[-4000:]}
        return {'ok': proc.returncode == 0, 'returncode': proc.returncode,
                'stdout': stdout[:20000], 'stderr': stderr[:4000],
                'output_truncated': len(stdout) > 20000 or len(stderr) > 4000}

    def dispatch(self, name, args):
        if name=='robot_help':
            self.enable_operation(args['operation'])
            path=public_path(self.workspace,'tools/'+args['operation']+'.txt')
            return {'operation':args['operation'],'contract':path.read_text(),
                    'schema_file':'tools/schemas/'+args['operation']+'.json','named_function_enabled':True},[]
        if name == 'read_file':
            path = public_path(self.workspace, args['path'])
            lines = path.read_text().splitlines()
            if path.parent==self.workspace/'tools' and path.suffix=='.txt' and path.stem in self.available:
                self.enable_operation(path.stem)
            start, count = args.get('start_line', 1)-1, args.get('max_lines', 150)
            selected = '\n'.join(lines[start:start+count])
            return {'path': args['path'], 'start_line': start+1, 'total_lines': len(lines),
                    'text': selected[:20000], 'truncated': start+count < len(lines) or len(selected)>20000}, []
        if name == 'view_image':
            from PIL import Image
            path = public_path(self.workspace, args['path'])
            with Image.open(path) as picture:
                width, height = picture.size
            return {'path': args['path'], 'width': width, 'height': height}, [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'Result of view_image: ' + args['path']},
                                           {'type': 'local_image', 'path': args['path']}]}]
        if name == 'run_python':
            return self.python(args['code']), []
        if name == 'write_file':
            path = public_path(self.workspace, args['path'])
            if any(path == p.resolve() or path.is_relative_to(p.resolve()) for p in self.readonly):
                raise ValueError('This supplied input is read-only')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args['text'])
            return {'ok': True, 'path': args['path']}, []
        if name in self.available:
            if not self.live:
                raise ValueError('Offline diagnostic: no robot service is running')
            validate_operation(name,args,self.image_sizes or None)
            result = self.python('import json,sys;from robot import brief;print(json.dumps(brief(sys.argv[1],**json.loads(sys.argv[2]))))',
                                 timeout=930, arguments=[name, json.dumps(args, allow_nan=False)])
            if not result['ok']:
                return result, []
            reply=json.loads(result['stdout'])
            self.record_image_sizes(reply)
            return reply, []
        raise ValueError('Unsupported tool')


def run_actor(client, public_tools, prompt, output, *, seconds=1800, max_turns=80):
    out = Path(output)
    messages = [{'role': 'system', 'content': 'You control a simulated robot through documented public tools. Use the supplied tool schemas. Inspect evidence, act, and revise your decisions. Keep user-facing explanations concise.'},
                {'role': 'user', 'content': prompt}]
    started, tool_seconds, malformed, tool_errors = time.monotonic(), 0., 0, 0
    events = []
    reason = 'turn_budget_exhausted'
    for turn in range(1, max_turns+1):
        elapsed = time.monotonic() - started
        status_file = public_tools.workspace / 'robot_data/episode-status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if status.get('task_success') or status.get('episode_done'):
            reason = 'simulator_success' if status.get('task_success') else status.get('termination_reason', 'episode_done')
            break
        if elapsed >= seconds:
            reason = 'wall_budget_exhausted'
            break
        schemas=public_tools.schemas()
        response, timing = client.complete(messages, schemas, public_tools.workspace,
            timeout=min(client.request_timeout_seconds, seconds-elapsed))
        messages.append(response)
        if timing.get('finish_reason')=='length':
            reason='model_output_limit'
            write_json(out/'conversation.json',messages)
            break
        calls = response.get('tool_calls', [])
        if not calls:
            messages.append({'role': 'user', 'content': 'The episode remains incomplete. Continue using the public tools while time and control budget remain. Check episode_status. Do not reset for another attempt.'})
            continue
        extra_messages = []
        for call in calls:
            begin = time.monotonic()
            event = {'turn': turn, 'call': call, 'started_monotonic': begin}
            try:
                name, args = parse_call(call,schemas)
                event['schema_valid'] = True
                status = json.loads(status_file.read_text()) if status_file.exists() else {}
                if status.get('task_success') or status.get('episode_done') or time.monotonic()-started >= seconds:
                    raise RuntimeError('Episode is terminal; this additional tool call was not executed')
                result, images = public_tools.dispatch(name, args)
                extra_messages.extend(images)
            except Exception as error:
                result = {'ok': False, 'error_type': type(error).__name__, 'error': str(error)[:4000]}
                if hasattr(error,'request_diagnostic'):
                    result['request_diagnostic']=error.request_diagnostic
                if not event.get('schema_valid'):
                    event['schema_valid'] = False
                    malformed += 1
                tool_errors += 1
            event.update(result=result, wall_seconds=time.monotonic()-begin)
            tool_seconds += event['wall_seconds']
            events.append(event)
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result, ensure_ascii=False)})
            with (out / 'local-tool-events.jsonl').open('a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        messages.extend(extra_messages)
        write_json(out / 'conversation.json', messages)
        write_json(out / 'actor-progress.json', {'turn': turn, 'elapsed_seconds': time.monotonic()-started,
                   'model_seconds': sum(x['wall_seconds'] for x in client.calls), 'tool_seconds': tool_seconds,
                   'tool_calls': len(events), 'tool_errors': tool_errors, 'malformed_calls': malformed})
    summary = {'termination_reason': reason, 'wall_seconds': time.monotonic()-started,
               'model_seconds': sum(x['wall_seconds'] for x in client.calls), 'tool_seconds': tool_seconds,
               'model_calls': len(client.calls), 'tool_calls': len(events), 'tool_errors': tool_errors,
               'malformed_calls': malformed}
    write_json(out / 'actor-summary.json', summary)
    return summary
