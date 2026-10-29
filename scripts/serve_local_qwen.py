"""Start the pinned local Qwen pilot server and record its own GPU memory."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_TOKENS = 65536
DEFAULT_GPU_MEMORY_UTILIZATION = .34


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--port', type=int, default=8127)
    parser.add_argument('--model-path', type=Path, default=ROOT / '.runtime/models/Qwen3.5-9B')
    parser.add_argument('--served-model-name', default='qwen9b')
    parser.add_argument('--context-tokens', type=int, default=DEFAULT_CONTEXT_TOKENS,
                        help='Total prompt/history plus generated-token window.')
    parser.add_argument('--gpu-memory-utilization', type=float,
                        default=DEFAULT_GPU_MEMORY_UTILIZATION,
                        help='vLLM GPU memory budget fraction, in (0, 1].')
    return parser


def server_command(args):
    """Validate budgets and construct the command without importing GPU libraries."""
    if type(args.context_tokens) is not int or args.context_tokens <= 0:
        raise ValueError('context-tokens must be a positive integer')
    fraction = args.gpu_memory_utilization
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('gpu-memory-utilization must be finite and in (0, 1]')
    if not args.served_model_name or any(c.isspace() for c in args.served_model_name):
        raise ValueError('Use a nonempty served model name without whitespace')
    return [str(ROOT / '.runtime/envs/qwen-vllm/bin/vllm'), 'serve',
        str(args.model_path.resolve()), '--served-model-name', args.served_model_name,
        '--host', '127.0.0.1', '--port', str(args.port), '--dtype', 'bfloat16',
        '--max-model-len', str(args.context_tokens), '--max-num-seqs', '1', '--max-num-batched-tokens', '2048',
        '--gpu-memory-utilization', str(fraction), '--reasoning-parser', 'qwen3',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
        '--limit-mm-per-prompt', '{"image":16,"video":0}',
        '--mm-processor-kwargs', '{"max_pixels":262144}', '--generation-config', 'vllm']


def main():
    parser = argument_parser()
    args = parser.parse_args()
    try:
        command = server_command(args)
    except ValueError as error:
        parser.error(str(error))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu),
           'PATH': str(ROOT / '.runtime/envs/qwen-vllm/bin') + ':/usr/local/cuda-13.0/bin:' + os.environ.get('PATH', ''),
           'CUDA_HOME': '/usr/local/cuda-13.0',
           'FLASHINFER_WORKSPACE_BASE': str(ROOT / '.runtime/cache/qwen-flashinfer'),
           'XDG_CACHE_HOME': str(ROOT / '.runtime/cache/qwen-xdg'),
           'HF_HOME': str(ROOT / '.runtime/cache/qwen-hf'), 'HF_HUB_OFFLINE': '1',
           'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
           'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1', 'OMP_NUM_THREADS': '1',
           'VLLM_CACHE_ROOT': str(ROOT / '.runtime/cache/qwen-vllm'),
           'TORCHINDUCTOR_CACHE_DIR': str(ROOT / '.runtime/cache/qwen-inductor'),
           'CUDA_CACHE_PATH': str(ROOT / '.runtime/cache/qwen-cuda')}
    (out / 'server-command.json').write_text(json.dumps({'command': command,
        'configured_vllm_settings': {'max_model_len': args.context_tokens,
                                    'model_path': str(args.model_path.resolve()),
                                    'served_model_name': args.served_model_name,
                                    'gpu_memory_utilization': args.gpu_memory_utilization,
                                    'max_num_seqs': 1},
        'environment': {k: env[k] for k in env if k.startswith(('CUDA_', 'HF_', 'VLLM_', 'TORCHINDUCTOR_'))},
        'gpu_memory_measurement': 'NVML per-process usage of this server and its descendants; other jobs excluded.'}, indent=2)+'\n')
    import psutil
    import pynvml
    pynvml.nvmlInit()
    device = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    start, peak, code = time.monotonic(), 0, None
    with (out / 'server.log').open('w') as log, (out / 'gpu-memory.jsonl').open('w') as memory:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        (out / 'server-process.json').write_text(json.dumps({'pid': process.pid, 'launcher_pid': os.getpid(),
                                                            'gpu': args.gpu, 'port': args.port})+'\n')
        try:
            while process.poll() is None:
                pids = {process.pid}
                try:
                    pids.update(p.pid for p in psutil.Process(process.pid).children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
                records = [{'pid': p.pid, 'bytes': p.usedGpuMemory} for p in pynvml.nvmlDeviceGetComputeRunningProcesses(device)
                           if p.pid in pids and isinstance(p.usedGpuMemory, int) and p.usedGpuMemory < 2**60]
                used = sum(p['bytes'] for p in records)
                peak = max(peak, used)
                memory.write(json.dumps({'time_unix': time.time(), 'elapsed_seconds': time.monotonic()-start,
                                         'server_gpu_bytes': used, 'processes': records})+'\n')
                memory.flush()
                if (out / 'stop-server').exists() or time.monotonic()-start > 10800:
                    process.terminate()
                    break
                time.sleep(2)
            code = process.wait(timeout=90)
        finally:
            if process.poll() is None:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (out / 'server-exit.json').write_text(json.dumps({'returncode': process.returncode,
                'peak_server_gpu_bytes': peak, 'wall_seconds': time.monotonic()-start}, indent=2)+'\n')
            pynvml.nvmlShutdown()
    if code:
        sys.exit(code if code > 0 else 128-code)


if __name__ == '__main__':
    main()
