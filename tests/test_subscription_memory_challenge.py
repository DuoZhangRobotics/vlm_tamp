import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('memory_challenge',ROOT/'scripts/run_subscription_memory_challenge.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_known_failures_remain_separate_from_reserved_transfer():
    rows=module.scenarios([6,17,25])
    known=[r for r in rows if r['group']=='known_challenge']
    fresh=[r for r in rows if r['group']=='fresh_transfer']
    assert {(r['state'],r['scene_variant']) for r in known}=={
        (47,'nominal'),(37,'yaw_negative_shifted'),(47,'clutter_shifted')}
    assert [r['state'] for r in fresh]==[6,17,25]
    assert {r['scene_variant'] for r in fresh}=={'yaw_negative_shifted','clutter_shifted','frontview_shifted'}
    assert len({r['scenario_id'] for r in rows})==6


@pytest.mark.parametrize('states',[[6,6,25],[37,17,25],[6,17,47],[6,17,50]])
def test_reserved_transfer_rejects_duplicate_or_known_states(states):
    with pytest.raises(ValueError):module.scenarios(states)
