import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_disabled_dino_has_no_inference_side_effect(monkeypatch):
    module=load('serve_subscription_robot');service=module.RobotService.__new__(module.RobotService)
    service.dino_tools=False
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('Disabled tool launched inference'))
    with pytest.raises(ValueError,match='unavailable'):service.dino_landmarks()


def test_runner_failure_before_motion_stays_in_planned_denominator(tmp_path):
    module=load('run_subscription_dino_study')
    record=module.collect({'case_id':'state008-no_dino','state':8,'condition':'no_dino'},tmp_path/'absent',1,2.)
    assert record['status']=='runner_failure'
    assert record['physical_attempts_used']==0
    assert not record['within_two_attempts_success']
    assert (tmp_path/'absent/study-case-result.json').exists()


def test_dino_reference_and_custom_analysis_raw_inputs_are_retained(tmp_path):
    module=load('run_subscription_dino_study');run=tmp_path/'run';run.mkdir()
    archive=run/'agent-workspace';archive.mkdir()
    for n in [1,2,3,4,5]:
        d=archive/'robot_data/attempt01'/f'obs{n:03d}';d.mkdir(parents=True)
        (d/'sensors.npz').write_bytes(b'raw-'+str(n).encode())
    events=[{'request':{'operation':'dino_match','arguments':{'reference_observation':2}},
             'response':{'result':{'observation_id':3}}}]
    (run/'tool-events.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
    (run/'agent-events.jsonl').write_text('')
    result=module.retained_inputs(run)
    assert not (archive/'robot_data/attempt01/obs004/sensors.npz').exists()
    assert all((archive/f'robot_data/attempt01/obs{n:03d}/sensors.npz').exists() for n in [1,2,3,5])
    assert len(result['removed'])==1
