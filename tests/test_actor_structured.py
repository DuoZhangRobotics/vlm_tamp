"""The bounded actor dispatches data requests and preserves actual transition evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import vlm_tamp.actor_structured as actor


def test_tool_argument_text_is_never_a_shell_command(tmp_path: Path,monkeypatch) -> None:
    calls=[]
    monkeypatch.setattr(actor.subprocess,'run',lambda command,**kw:
        (calls.append((command,kw)) or SimpleNamespace(returncode=0,stdout='{"ok":true}',stderr='')))
    literal='{"query":"$(never_execute_this)"}'
    assert actor.tool_call(tmp_path,'point',literal,{})['ok']
    assert calls[0][0]==['/usr/bin/python3',str(tmp_path/'robot.py'),'--full','point',literal]
    assert 'shell' not in calls[0][1]


def test_image_selection_excludes_private_paths_and_duplicate_source_crops(tmp_path: Path) -> None:
    Image.new('RGB',(4,4)).save(tmp_path/'shown.png')
    Image.new('RGB',(4,4)).save(tmp_path/'source.png')
    value={'images':['shown.png'],'metadata':{'source_image':'source.png'},
           'private':{'image':'../hidden.png'}}
    assert actor.image_paths(value,tmp_path)==['shown.png']


def test_episode_uses_one_structured_action_and_complete_before_after_memory(tmp_path: Path,monkeypatch) -> None:
    work=tmp_path/'public';out=tmp_path/'run';work.mkdir();out.mkdir();(work/'robot_data').mkdir()
    (work/'tools').mkdir()
    (work/'tools/schema-index.json').write_text(json.dumps({'operations':{'push_stroke':{},'observe':{},'finish':{}}}))
    state={'observation_id':1,'attempt':1,'control_steps':60,'maximum_control_steps':600}
    choices=iter([
        {'operation':'help','arguments_json':'{"operation":"push_stroke"}','expected_effect':'read contract','note':'unverified'},
        {'operation':'push_stroke','arguments_json':'{"direction_world":[1,0,0],"distance_m":0.01}',
         'expected_effect':'test a direction','note':'await outcome'},
        {'operation':'finish','arguments_json':'{}','expected_effect':'record completion','note':'observed native success'}])
    dispatched=[];inputs=[]
    def tool(_work: Path,operation: str,args: str,environment: dict) -> dict:
        dispatched.append(operation)
        if operation=='reset':return {'ok':True,'result':{'instruction':'fixture task'}}
        if operation=='push_stroke':state.update(control_steps=70,observation_id=2)
        return {'ok':True,'result':{'observation':dict(state),'task_success':state['control_steps']==70}}
    def pair(_work: Path,observation: dict) -> Path:
        path=work/f"obs{observation['observation_id']}.png"
        if not path.exists():Image.new('RGB',(10,10)).save(path)
        return path
    def decision(prompt: str,images: list,folder: Path,*args) -> dict:
        inputs.append(json.loads(prompt.rsplit('\n',1)[1]));folder.mkdir()
        (folder/'events.jsonl').write_text('')
        return next(choices)
    monkeypatch.setattr(actor,'tool_call',tool)
    monkeypatch.setattr(actor,'latest_observation',lambda _:dict(state))
    monkeypatch.setattr(actor,'observation_pair',pair)
    monkeypatch.setattr(actor,'run_decision',decision)
    monkeypatch.setattr(actor,'episode_status',lambda _:{'episode_done':state['control_steps']==70,
        'task_success':state['control_steps']==70,'termination_reason':'simulator_success'})
    assert actor.run_structured_actor(work,out,{},'fixture',60,model='gpt-6-luna',reasoning='high',
        memory_mode='anchored',memory_slots=8,visual_interface=False)==0
    assert dispatched==['reset','push_stroke','finish']
    assert inputs[1]['memory']['retained_executed_transitions']==0
    transition=inputs[2]['memory']['events'][0]
    assert transition['before_control']==60 and transition['after_control']==70
    assert transition['before_image']=='obs1.png' and transition['after_image']=='obs2.png'
    assert transition['preceding_nonmotion'][0]['operation']=='help'
    assert inputs[2]['terminal_summary_requested']
    assert len(inputs[2]['images'])==2
    assert 'scene_views' not in inputs[0]['tools']
    assert json.loads((out/'agent-exit.json').read_text())['generated_policy_execution'] is False
