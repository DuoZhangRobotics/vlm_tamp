"""Cross-check one vision/tool prompt in vLLM and Transformers with the same weights."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vlm_tamp.local_agent import encode_images, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'results/qwen9b-pilot-20261024/coordinate-format'
    request = json.loads((source / 'responses/1view-normalized_1000-r1/call001/request.json').read_text())
    protocol = {'scope': 'One deterministic backend cross-check; not a performance benchmark.',
        'model': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'dtype': 'bfloat16', 'thinking': False, 'sampling': 'greedy', 'max_output_tokens': 256,
        'same_public_messages_and_tool_schema': True, 'transformers_attention': 'sdpa',
        'note': 'Tokenization, output and tensor shapes are retained; kernel implementations differ.'}
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'input.json', {'messages': request['messages'], 'tools': request['tools']})
    encoded, image_inventory = encode_images(request['messages'], source / 'public')
    body = {'model': 'qwen9b', 'messages': encoded, 'tools': request['tools'], 'tool_choice': 'auto',
            'temperature': 0., 'seed': 123, 'max_tokens': 256, 'stream': False,
            'parallel_tool_calls': False, 'chat_template_kwargs': {'enable_thinking': False}}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    with opener.open(urllib.request.Request('http://127.0.0.1:8127/v1/chat/completions', json.dumps(body).encode(),
                                          {'Content-Type': 'application/json'}), timeout=180) as response:
        vllm_result = json.load(response)
    write_json(out / 'vllm.json', {'seconds': time.monotonic()-started, 'response': vllm_result, 'images': image_inventory})
    print(json.dumps({'vllm_complete': True, 'response': vllm_result['choices'][0]['message']}), flush=True)
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    torch.manual_seed(123)
    model_path = ROOT / '.runtime/models/Qwen3.5-9B'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    messages = copy.deepcopy(request['messages'])
    for message in messages:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part['type'] == 'local_image':
                    picture = Image.open(source / 'public' / part['path']).convert('RGB')
                    part.clear()
                    part.update(type='image', image=picture)
    inputs = processor.apply_chat_template(messages, tools=request['tools'], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors='pt', enable_thinking=False)
    tensor_shapes = {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}
    rendered = processor.apply_chat_template(messages, tools=request['tools'], add_generation_prompt=True,
        tokenize=False, enable_thinking=False)
    (out / 'transformers-prompt.txt').write_text(rendered)
    write_json(out / 'transformers-input-metadata.json', {'shapes': tensor_shapes,
        'input_ids': inputs['input_ids'][0].tolist(), 'image_grid_thw': inputs.get('image_grid_thw').tolist(),
        'prompt_tokens': int(inputs['input_ids'].shape[-1]), 'vllm_prompt_tokens': vllm_result.get('usage', {}).get('prompt_tokens'),
        'torch': torch.__version__, 'transformers': transformers.__version__})
    begin = time.monotonic()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
    load_seconds = time.monotonic()-begin
    print(json.dumps({'reference_model_loaded': True, 'seconds': load_seconds, 'shapes': tensor_shapes}), flush=True)
    inputs = inputs.to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    torch.cuda.synchronize()
    seconds = time.monotonic()-begin
    new_tokens = generated[0, inputs['input_ids'].shape[-1]:]
    text = processor.decode(new_tokens, skip_special_tokens=False)
    result = {'output': text, 'output_token_ids': new_tokens.tolist(), 'output_tokens': len(new_tokens),
              'inference_seconds': seconds, 'load_seconds': load_seconds,
              'torch_peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
              'torch_peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(out / 'transformers.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
