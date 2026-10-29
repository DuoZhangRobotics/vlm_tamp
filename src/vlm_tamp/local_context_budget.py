"""Exact context admission for the pinned vLLM automatic-tool chat interface.

Call with the already image-encoded generation body. vLLM's /tokenize endpoint
uses the model's real chat template and multimodal preprocessing, but performs
no generation. The helper returns count metadata only and never shrinks the
requested output allowance or rewrites the caller's generation request.
"""
import copy
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request


class ContextBudgetExceeded(ValueError):
    """The complete requested output allowance does not fit after the prompt."""
    def __init__(self, budget):
        self.context_budget = budget
        super().__init__(
            f"Prompt uses {budget['prompt_tokens']} tokens and requested output uses "
            f"{budget['requested_output_tokens']}; server context is "
            f"{budget['context_tokens']} (shortfall {budget['shortfall_tokens']}). "
            "Generation was not admitted; the output budget was not reduced.")


# Only options whose rendering behavior has been checked against the pinned
# vLLM endpoint are accepted. New prompt-affecting modes require a matching
# tokenizer implementation rather than an approximate admission decision.
_ALLOWED_FIELDS = {
    'model', 'messages', 'tools', 'tool_choice', 'parallel_tool_calls',
    'temperature', 'top_p', 'top_k', 'seed', 'presence_penalty',
    'max_tokens', 'stream', 'stream_options',
    'add_generation_prompt', 'continue_final_message', 'add_special_tokens',
    'chat_template', 'chat_template_kwargs', 'media_io_kwargs',
    'mm_processor_kwargs',
}


def tokenizer_endpoint(endpoint):
    """Derive the root /tokenize route from this local server's /v1 base."""
    parsed = urllib.parse.urlsplit(endpoint)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Context measurement requires an unauthenticated loopback HTTP endpoint')
    if parsed.path.rstrip('/') not in {'', '/v1', '/tokenize'}:
        raise ValueError('Use the local vLLM base URL ending in /v1 or its /tokenize endpoint')
    return urllib.parse.urlunsplit(parsed._replace(path='/tokenize'))


def requested_output_tokens(request):
    """Our local chat client sends its explicit allowance as max_tokens."""
    value = request.get('max_tokens')
    if type(value) is not int or value < 1:
        raise ValueError('An explicit positive integer output-token allowance is required')
    return value


def tokenize_payload(generation_request):
    """Mirror ChatCompletionRequest rendering inputs in TokenizeChatRequest.

    Chat completion normalizes reasoning_content to reasoning before rendering;
    the tokenize endpoint does not. Other messages, image data, tools and
    template kwargs are copied without modification to their meaning.
    """
    if not isinstance(generation_request, dict):
        raise ValueError('Use the JSON generation request object')
    unknown = set(generation_request) - _ALLOWED_FIELDS
    if unknown:
        raise ValueError('Exact context measurement does not support request fields: ' + ', '.join(sorted(unknown)))
    requested_output_tokens(generation_request)
    if generation_request.get('tool_choice') not in (None, 'auto'):
        raise ValueError('Exact context admission currently supports automatic tool choice only')
    model, messages = generation_request.get('model'), generation_request.get('messages')
    if not isinstance(model, str) or not model or not isinstance(messages, list):
        raise ValueError('A model name and message list are required')
    payload = {'model': model, 'messages': copy.deepcopy(messages), 'return_token_strs': False}
    for message in payload['messages']:
        if not isinstance(message, dict):
            raise ValueError('Messages must be JSON objects')
        old_reasoning = message.pop('reasoning_content', None)
        if old_reasoning is not None and message.get('reasoning') is None:
            message['reasoning'] = old_reasoning
        content = message.get('content')
        if isinstance(content, list) and any(isinstance(part, dict) and part.get('type') == 'local_image' for part in content):
            raise ValueError('Encode local images exactly as for generation before measuring context')
    for name in ('tools', 'chat_template', 'media_io_kwargs', 'mm_processor_kwargs'):
        if name in generation_request:
            payload[name] = copy.deepcopy(generation_request[name])
    for name, default in [('add_generation_prompt', True), ('continue_final_message', False), ('add_special_tokens', False)]:
        value = generation_request.get(name, default)
        if type(value) is not bool:
            raise ValueError(name + ' must be boolean')
        payload[name] = value
    if payload['add_generation_prompt'] and payload['continue_final_message']:
        raise ValueError('A generation prompt and an open final message cannot both be requested')
    kwargs = generation_request.get('chat_template_kwargs')
    if kwargs is not None and not isinstance(kwargs, dict):
        raise ValueError('chat_template_kwargs must be an object or null')
    kwargs = copy.deepcopy(kwargs or {})
    if kwargs or 'chat_template_kwargs' in generation_request:
        payload['chat_template_kwargs'] = kwargs
    return payload


def admit_context(opener, endpoint, generation_request, *, timeout=30):
    """Measure the prompt and admit the full allowance, or fail without generation.

    ``opener`` is the caller's urllib-compatible opener. ``endpoint`` is its
    local /v1 base (or /tokenize URL). Only safe count/hash/timing metadata is
    returned: raw images, prompt text and token IDs are deliberately omitted.
    This checks context capacity; other serving limits remain caller policy.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Tokenizer timeout must be positive and finite')
    started = time.monotonic()
    url = tokenizer_endpoint(endpoint)
    payload = tokenize_payload(generation_request)
    output_tokens = requested_output_tokens(generation_request)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    request = urllib.request.Request(url, raw, {'Content-Type': 'application/json'})
    try:
        with opener.open(request, timeout=timeout) as response:
            measured = json.load(response)
    except urllib.error.HTTPError as error:
        # Error bodies may echo supplied image data; do not log or expose them.
        raise RuntimeError(f'Tokenizer returned HTTP {error.code}; exact context capacity is unavailable and generation was not admitted') from error
    if not isinstance(measured, dict):
        raise RuntimeError('Tokenizer returned an invalid count response; generation was not admitted')
    prompt_tokens, context_tokens = measured.get('count'), measured.get('max_model_len')
    if type(prompt_tokens) is not int or prompt_tokens < 0 or type(context_tokens) is not int or context_tokens < 1:
        raise RuntimeError('Tokenizer response requires nonnegative prompt count and positive context size; generation was not admitted')
    if 'tokens' in measured and (not isinstance(measured['tokens'], list) or len(measured['tokens']) != prompt_tokens):
        raise RuntimeError('Tokenizer token list disagrees with its count; generation was not admitted')
    remaining = context_tokens - prompt_tokens
    budget = {'accepted': prompt_tokens + output_tokens <= context_tokens,
        'prompt_tokens': prompt_tokens, 'requested_output_tokens': output_tokens,
        'context_tokens': context_tokens, 'available_output_tokens': max(0, remaining),
        'headroom_after_full_output_tokens': remaining-output_tokens,
        'shortfall_tokens': max(0, output_tokens-remaining),
        'output_budget_reduced': False, 'measurement': 'vllm_tokenize_endpoint',
        'tokenization_seconds': time.monotonic()-started,
        'tokenize_request_sha256': hashlib.sha256(raw).hexdigest(),
        'tokenize_request_bytes': len(raw)}
    if not budget['accepted']:
        raise ContextBudgetExceeded(budget)
    return budget
