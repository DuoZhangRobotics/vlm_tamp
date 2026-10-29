import json
from pathlib import Path
import numpy as np
import pytest

import serve_contact_checkpoint as checkpoint


def test_prefix_publishes_only_current_observation_and_replays_whole_commands(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    (source/'protocol.json').write_text(json.dumps({'case':{'state_index':17,'task':'close_microwave'}}))
    observation=source/'agent-workspace/robot_data/attempt01/obs027';observation.mkdir(parents=True)
    expected=np.full(7,.1)
    (observation/'robot.json').write_text(json.dumps({'control_steps':64,'joint_positions':expected.tolist()}))
    commands=[{'event':8,'references':np.repeat(expected[None],4,axis=0),'gripper':0}]
    monkeypatch.setattr(checkpoint,'source_commands',lambda *args:commands)
    captures=[]
    class Fake:
        def reset(self):
            self.state={'control_steps':60,'metadata':{'joint_positions':[0.]*7}}
            return self.capture()
        def capture(self):
            captures.append(self.state['control_steps'])
            return {'control_steps':self.state['control_steps'],'observation_id':self.calls}
    monkeypatch.setattr(checkpoint.backend,'RobotService',Fake)
    def execute(sim,state,refs,**kwargs):
        assert len(refs)==4 and kwargs['gripper']==0
        return {'control_steps':64,'metadata':{'joint_positions':expected.tolist()}}, {
            'all_commands_issued':True,'issued_commands':4,'events':[]}
    monkeypatch.setattr(checkpoint,'execute_batch',execute)
    checkpoint.install_prefix(source,64)
    service=Fake();service.state_index=17;service.task='close_microwave'
    service.task_spec={'instruction':'close the microwave'};service.sim=object();service.stop_on_success=True
    service.calls=1;service.events=[];service.executions=[];service.attempt=1
    service.output=tmp_path/'run';service.output.mkdir();service.episode=service.output/'attempt01';service.episode.mkdir()
    result=service.reset()
    assert captures==[64]
    assert result['control_steps']==64 and result['instruction']=='close the microwave'
    assert 'source' not in result
    assert np.array_equal(np.load(next(service.episode.glob('*-issued-*'))),commands[0]['references'])
    audit=json.loads((service.output/'initial-prefix.json').read_text())
    assert audit['maximum_joint_error_rad']==0 and not audit['simulator_state_assignment']


def test_checkpoint_cannot_truncate_an_original_issued_trajectory(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'case':{}}))
    monkeypatch.setattr(checkpoint,'source_commands',lambda *args:[{'references':np.zeros((5,7))}])
    with pytest.raises(ValueError,match='complete original issued trajectory'):
        checkpoint.install_prefix(tmp_path,62)
