from supervise_luna_dashboard_study import pressure_stop


def row(io=0,memory=0):return {'io_full_avg10':io,'memory_full_avg10':memory}


def test_transient_pressure_does_not_interrupt_a_trial():
    assert pressure_stop([row(),row(30),row()]) is None
    assert pressure_stop([row(30),row(30)]) is None


def test_recurring_server_strain_is_an_infrastructure_stop():
    assert pressure_stop([row(11),row(12),row(13)])=='sustained_host_io_pressure'
    assert pressure_stop([row(0,6)]*3)=='sustained_host_memory_pressure'
