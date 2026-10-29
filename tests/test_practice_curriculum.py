import json
from types import SimpleNamespace
import pytest
import serve_practice_curriculum as curriculum


def test_curriculum_changes_case_only_after_finishing_previous_episode(monkeypatch):
    calls=[]
    def original(service):
        calls.append(('reset',service.task,service.state_index))
        return 'reset'
    monkeypatch.setattr(curriculum.backend.RobotService,'reset',original)
    curriculum.install_curriculum([{'task':'close_microwave','state_index':17},
                                   {'task':'open_microwave','state_index':13}])
    service=SimpleNamespace(learning_practice=True,attempt=1,limit=6,sim=object(),
        task='close_microwave',state_index=17,episode_status=lambda:{'episode_done':False})
    curriculum.backend.RobotService.reset(service)
    assert calls==[('reset','close_microwave',17)]
    service.episode_status=lambda:{'episode_done':True,'termination_reason':'simulator_success'}
    def finish(reason):
        calls.append(('finished',service.task,service.state_index));service.sim=None
    service.finish_attempt=finish
    curriculum.backend.RobotService.reset(service)
    assert calls[-2:]==[('finished','close_microwave',17),('reset','open_microwave',13)]
    assert service.task_spec['motion']=='open'


@pytest.mark.parametrize('cases',[[],[{'task':'close_microwave','state_index':50}],
    [{'task':'close_microwave','state_index':True}],
    [{'task':'close_microwave','state_index':17,'successful_pixels':[1,2]}]])
def test_curriculum_rejects_invalid_cases_and_extra_hints(tmp_path,cases):
    path=tmp_path/'cases.json';path.write_text(json.dumps(cases))
    with pytest.raises(ValueError):curriculum.checked_cases(path)
