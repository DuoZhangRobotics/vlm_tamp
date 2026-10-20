import json
from pathlib import Path

import pytest

from vlm_tamp.episode_persistence import continuation_decision,continuation_prompt
from vlm_tamp.subscription_agent import codex_resume_arguments


def robot(tmp_path,steps=338):
    from serve_subscription_robot import RobotService
    service=RobotService.__new__(RobotService)
    service.calls=0;service.output=tmp_path;service.state={'control_steps':steps}
    service.persistent_episode=True;service.episode_success=False;service.attempt=1
    service.stop_on_success=True;service.latest_observation={}
    service.finish_attempt=lambda reason:None
    return service


@pytest.mark.parametrize('arguments',[{}, {'claim':'incomplete'}, {'claim':'uncertain'}, {'claim':'completed'}])
def test_no_finish_claim_can_end_an_incomplete_episode_with_remaining_budget(tmp_path,arguments):
    service=robot(tmp_path)
    service.finish_attempt=lambda reason:pytest.fail('An early finish must not close the simulator')
    result=service.dispatch('finish',arguments)
    assert not result['finished'] and result['must_continue']
    assert result['remaining_control_steps']==262
    assert not result['episode_done']


@pytest.mark.parametrize('condition,reason',[
    ('native','simulator_success'),('steps','control_budget_exhausted'),('time','wall_budget_exhausted')])
def test_only_native_success_or_real_budget_exhaustion_allows_finish(tmp_path,condition,reason):
    service=robot(tmp_path,600 if condition=='steps' else 100);ended=[]
    service.episode_success=condition=='native'
    if condition=='time':
        (tmp_path/'episode-deadline.json').write_text(json.dumps({'deadline_monotonic':0}))
    service.finish_attempt=ended.append
    result=service.dispatch('finish',{})
    assert result['finished'] and result['termination_reason']==reason and ended==[reason]


def test_final_response_is_not_a_terminal_event():
    status={'task_success':False,'remaining_control_steps':108}
    assert continuation_decision(status,0,100)=='continue_same_episode'
    assert continuation_decision(status,0,0)=='wall_budget_exhausted'
    assert continuation_decision(status,1,100)=='agent_runtime_error'
    assert continuation_decision({'task_success':True},0,100)=='simulator_success'
    assert continuation_decision({'remaining_control_steps':0},0,100)=='control_budget_exhausted'
    message=continuation_prompt(status,100)
    assert '108' in message and 'Do not reset' in message and 'hinge' not in message and 'inside' not in message


def test_resume_uses_exact_thread_and_keeps_research_isolation(tmp_path):
    identity='12345678-1234-4321-8123-123456789abc'
    args=codex_resume_arguments(tmp_path,tmp_path/'last.txt',identity)
    assert args[1:3]==['exec','resume'] and args[-2:]==[identity,'-']
    assert '--ephemeral' not in args and '-C' not in args
    assert 'forced_login_method="chatgpt"' in args
    assert 'permissions.research-agent.network.enabled=false' in args
    assert 'model_reasoning_effort="high"' in args
    assert 'gpt-6-luna' in args and 'approval_policy="never"' in args


def test_busy_transport_does_not_queue_duplicates_and_status_does_not_wait(tmp_path):
    import time
    from prepare_subscription_pilot import CLIENT
    for name in ['requests','replies','robot_data']:(tmp_path/name).mkdir()
    namespace={'__file__':str(tmp_path/'robot.py'),'__name__':'transport_test'}
    exec(compile(CLIENT,'robot.py','exec'),namespace)
    token='a'*32
    (tmp_path/'requests'/f'{token}.json').write_text(json.dumps({'operation':'plan_sequence','arguments':{'degrees':85}}))
    (tmp_path/'robot_data/episode-status.json').write_text(json.dumps({
        'task_success':False,'remaining_control_steps':540,'control_steps':60}))
    (tmp_path/'robot_data/episode-deadline.json').write_text(json.dumps({
        'deadline_monotonic':time.monotonic()+100}))
    rejected=namespace['call']('plan_sequence',degrees=85)
    assert rejected['error_type']=='RequestInFlight'
    assert len(list((tmp_path/'requests').glob('*.json')))==1
    status=namespace['call']('episode_status')['result']
    assert status['remaining_control_steps']==540 and 0<status['remaining_wall_seconds']<=100
    assert status['in_flight_requests'][0]['request_id']==token
    assert len(list((tmp_path/'requests').glob('*.json')))==1
    original={'ok':True,'result':{'plan_id':'original'}}
    (tmp_path/'replies'/f'{token}.json').write_text(json.dumps(original))
    assert namespace['call']('await_request',request_id=token)==original


def test_supervisor_resumes_an_early_final_in_the_same_episode(tmp_path,monkeypatch):
    from vlm_tamp import episode_persistence as supervisor
    work=tmp_path/'work';out=tmp_path/'out';out.mkdir();work.mkdir()
    for name in ['robot_data','requests','replies']:(work/name).mkdir()
    identity='12345678-1234-4321-8123-123456789abc';commands=[];prompts=[]
    class Input:
        def write(self,text):prompts.append(text)
        def close(self):pass
    class Process:
        pid=123
        def __init__(self,command,**kwargs):
            commands.append(command);self.stdin=Input();self.turn=len(commands)
            kwargs['stdout'].write(json.dumps({'type':'thread.started','thread_id':identity})+'\n')
        def wait(self,**kwargs):
            (out/'live-episode-status.json').write_text(json.dumps({
                'task_success':self.turn==2,'remaining_control_steps':262 if self.turn==1 else 240}))
            return 0
        def poll(self):return 0
    monkeypatch.setattr(supervisor.subprocess,'Popen',Process)
    code=supervisor.run_persistent_agent(work,out,{},[],'Initial instructions.',20)
    record=json.loads((out/'continuation.json').read_text())
    assert code==0 and len(commands)==2
    assert commands[1][1:3]==['exec','resume'] and identity in commands[1]
    assert '262' in prompts[1] and 'Do not reset' in prompts[1]
    assert record['turns'][0]['decision']=='continue_same_episode'
    assert record['termination_reason']=='simulator_success'
