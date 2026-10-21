import json
import time

import pytest

from vlm_tamp.episode_persistence import practice_continuation_decision
from vlm_tamp.recovery_memory import abstract_lessons,freeze_lessons
from test_persistent_episode import robot


def test_practice_does_not_stop_at_first_success_or_early_final():
    budget={'attempts_started':1,'initial_attempts':5,'maximum_attempts':20,
            'learning_phase':{'verified_baseline_attempts':[]}}
    summary={'best_supported_method':'conditional procedure','stop_reason':'repeatable'}
    assert practice_continuation_decision({},budget,summary,0,100)=='continue_same_episode'
    terminal={'episode_done':True,'task_success':True}
    assert practice_continuation_decision(terminal,budget,summary,0,100)=='continue_practice'
    budget.update(attempts_started=5,learning_phase={'verified_baseline_attempts':[3,4]})
    assert practice_continuation_decision(terminal,budget,summary,0,100)=='practice_complete'
    assert practice_continuation_decision(terminal,budget,{},0,100)=='continue_practice'
    budget['attempts_started']=20
    assert practice_continuation_decision(terminal,budget,{},0,100)=='practice_attempt_budget_exhausted'
    assert practice_continuation_decision(terminal,budget,{},1,100)=='agent_runtime_error'
    assert practice_continuation_decision({},budget,{},0,0)=='practice_wall_budget_exhausted'


def test_terminal_practice_allows_new_attempt_but_not_further_motion(tmp_path):
    service=robot(tmp_path,600)
    service.learning_practice=True;service.sim=None
    service.dino_tools=False;service.completion_checks=False;service.practice_budget=object()
    service.reset=lambda:{'new_attempt':True}
    assert service.dispatch('reset',{})=={'new_attempt':True}
    with pytest.raises(ValueError,match='episode ended'):
        service.dispatch('execute',{'plan_id':'stale'})


def test_active_practice_cannot_reset_away_from_a_failed_strategy(tmp_path):
    from serve_subscription_robot import RobotService
    service=robot(tmp_path)
    service.learning_practice=True;service.sim=object()
    with pytest.raises(ValueError,match='Continue the active episode'):
        RobotService.reset(service)


def test_memory_rejects_historical_geometry_and_preserves_uncertain_lessons(tmp_path):
    lesson={'principles':[{'when':'observed motion is small',
        'action':'reobserve before choosing another contact',
        'evidence':{'attempt':'attempt 2 failed','references':['robot_data/private.png']},
        'limits':'unconfirmed'}]}
    clean=abstract_lessons(lesson)
    assert 'references' not in clean['principles'][0]['evidence']
    assert clean['principles'][0]['limits']=='unconfirmed'
    for value in ([1.,2.,3.], '[0.1, 0.2, 0.3]', 'robot_data/old/obs.npz'):
        with pytest.raises(ValueError):
            abstract_lessons({'principles':[{'action':value}]})
    (tmp_path/'agent-workspace').mkdir()
    (tmp_path/'agent-workspace/guidelines.json').write_text(json.dumps(lesson))
    (tmp_path/'evaluation-private.json').write_text(json.dumps({'attempts':[
        {'attempt':1,'official_task_success':False,'control_steps':600}]}))
    frozen=freeze_lessons(tmp_path)
    assert frozen['observed_attempts'][0]['native_success'] is False
    assert frozen['method']==clean


def test_practice_reset_renews_deadline_without_changing_session_limit(tmp_path,monkeypatch):
    # Stop before simulator construction: the reset must first set a bounded new
    # deadline, and an unfinished prior episode must never be silently discarded.
    import serve_subscription_robot as module
    service=robot(tmp_path,600)
    service.learning_practice=True;service.sim=None;service.limit=20;service.practice_budget=None
    service.public=tmp_path/'public';service.public.mkdir()
    deadline=time.monotonic()+3600
    (tmp_path/'practice-deadline.json').write_text(json.dumps({
        'session_deadline_monotonic':deadline,'episode_seconds':1800}))
    monkeypatch.setattr(module,'ProfiledClient',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('test stop')))
    with pytest.raises(RuntimeError,match='test stop'):
        module.RobotService.reset(service)
    updated=json.loads((tmp_path/'episode-deadline.json').read_text())
    assert time.monotonic()+1790<updated['deadline_monotonic']<=deadline
    assert updated['attempt']==2


def test_practice_supervisor_resumes_same_thread_after_one_success(tmp_path,monkeypatch):
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
            (out/'live-episode-status.json').write_text(json.dumps({'task_success':True,'episode_done':True}))
            (out/'practice-budget.json').write_text(json.dumps({
                'attempts_started':1 if self.turn==1 else 5,'initial_attempts':5,'maximum_attempts':20,
                'learning_phase':{'verified_baseline_attempts':[] if self.turn==1 else [4,5]}}))
            (work/'practice-summary.json').write_text(json.dumps({
                'best_supported_method':'fresh-geometry conditional method','stop_reason':'repeatable baseline'}))
            (work/'guidelines.json').write_text(json.dumps({'principles':['Measure fresh geometry']}))
            return 0
        def poll(self):return 0
    monkeypatch.setattr(supervisor.subprocess,'Popen',Process)
    code=supervisor.run_persistent_agent(work,out,{},[],'Practice.',20,learning_practice=True)
    assert code==0 and len(commands)==2 and commands[1][1:3]==['exec','resume']
    assert identity in commands[1]
    assert 'fresh reset' in prompts[1] and 'hinge' not in prompts[1]
    assert json.loads((out/'agent-exit.json').read_text())['termination_reason']=='practice_complete'
