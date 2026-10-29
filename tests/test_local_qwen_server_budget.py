"""Server budget configuration can be tested without launching a GPU worker."""
from types import SimpleNamespace

import pytest

from serve_local_qwen import argument_parser, server_command


def option(command, name):
    return command[command.index(name)+1]


def test_default_context_has_room_for_32k_output_and_input():
    args = argument_parser().parse_args(['--output', '/tmp/qwen-server-test'])
    command = server_command(args)
    context = int(option(command, '--max-model-len'))
    assert context == 65536
    assert context >= 32768 + 1658
    assert float(option(command, '--gpu-memory-utilization')) == .34
    assert option(command, '--max-num-seqs') == '1'
    assert option(command, '--host') == '127.0.0.1'
    assert option(command, '--dtype') == 'bfloat16'


def test_explicit_budgets_reach_the_server_command():
    args = argument_parser().parse_args(['--output', '/tmp/qwen-server-test',
        '--context-tokens', '98304', '--gpu-memory-utilization', '0.5'])
    command = server_command(args)
    assert option(command, '--max-model-len') == '98304'
    assert option(command, '--gpu-memory-utilization') == '0.5'


def test_explicit_checkpoint_and_name_do_not_change_default_precision_or_parsers(tmp_path):
    model=tmp_path/'Qwen3.5-27B'
    args=argument_parser().parse_args(['--output',str(tmp_path/'server'),
        '--model-path',str(model),'--served-model-name','qwen27b',
        '--gpu-memory-utilization','0.78'])
    command=server_command(args)
    assert command[2]==str(model.resolve())
    assert option(command,'--served-model-name')=='qwen27b'
    assert option(command,'--dtype')=='bfloat16'
    assert option(command,'--reasoning-parser')=='qwen3'
    assert option(command,'--tool-call-parser')=='qwen3_coder'
    assert option(command,'--max-model-len')=='65536'


@pytest.mark.parametrize('context', [0, -1, 32768.5, True])
def test_invalid_context_budget_is_rejected_before_startup(context):
    args = SimpleNamespace(context_tokens=context, gpu_memory_utilization=.34, port=8127)
    with pytest.raises(ValueError, match='positive integer'):
        server_command(args)


@pytest.mark.parametrize('fraction', [0, -.1, 1.01, float('nan'), float('inf'), True])
def test_invalid_memory_fraction_is_rejected_before_startup(fraction):
    args = SimpleNamespace(context_tokens=65536, gpu_memory_utilization=fraction, port=8127)
    with pytest.raises(ValueError, match='finite and in'):
        server_command(args)
