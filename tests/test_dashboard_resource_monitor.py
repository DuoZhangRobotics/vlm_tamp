from supervise_luna_dashboard_study import pressure_stop
import threading
import time
import pytest


def row(io=0,memory=0):return {'io_full_avg10':io,'memory_full_avg10':memory}


def test_transient_pressure_does_not_interrupt_a_trial():
    assert pressure_stop([row(),row(30),row()]) is None
    assert pressure_stop([row(30),row(30)]) is None


def test_recurring_server_strain_is_an_infrastructure_stop():
    assert pressure_stop([row(11),row(12),row(13)])=='sustained_host_io_pressure'
    assert pressure_stop([row(0,6)]*3)=='sustained_host_memory_pressure'


def test_blocked_process_inspection_cannot_block_health_loop(monkeypatch):
    import supervise_luna_dashboard_study as monitor
    release=threading.Event();started=threading.Event()
    def blocked(_):started.set();release.wait(5);return {'time':time.time()}
    monkeypatch.setattr(monitor,'process_details',blocked)
    sampler=monitor.ProcessSampler(None)
    try:
        assert started.wait(1)
        before=time.monotonic();assert sampler.read()=={};sampler.close()
        assert time.monotonic()-before<.2
        assert pressure_stop([row(20)]*3)=='sustained_host_io_pressure'
    finally:release.set()


def test_launch_requires_kernel_memory_and_swap_limits(tmp_path):
    from supervise_luna_dashboard_study import require_memory_limit
    (tmp_path/'memory.max').write_text('max');(tmp_path/'memory.swap.max').write_text('0')
    (tmp_path/'memory.oom.group').write_text('1')
    with pytest.raises(RuntimeError,match='MemoryMax'):require_memory_limit(tmp_path)
    (tmp_path/'memory.max').write_text(str(32*1024**3))
    assert require_memory_limit(tmp_path)==32*1024**3
    (tmp_path/'memory.swap.max').write_text('max')
    with pytest.raises(RuntimeError):require_memory_limit(tmp_path)
    (tmp_path/'memory.swap.max').write_text('0');(tmp_path/'memory.oom.group').write_text('0')
    with pytest.raises(RuntimeError,match='OOMPolicy'):require_memory_limit(tmp_path)
