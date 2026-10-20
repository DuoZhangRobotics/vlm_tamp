import importlib.util
import json
from pathlib import Path

from vlm_tamp.tool_event_audit import native_attempt_outcome,operation_name


def test_malformed_operation_is_preserved_as_invalid_not_inferred_as_nested_tool():
    event={'request':{'operation':{'tool':'ground','arguments':{}}}}
    assert operation_name(event)=='<invalid operation: dict>'
    assert not operation_name(event).startswith('ground')
    assert event['request']['operation']['tool']=='ground'


def test_native_success_survives_missing_or_failed_aggregate_report(tmp_path):
    (tmp_path/'evaluation-private.json').write_text(json.dumps({'attempts':[
        {'official_task_success':True,'control_steps':335}]}))
    result=native_attempt_outcome(tmp_path)
    assert result['native_success'] and result['native_evidence_available']
    assert result['native_attempts_recorded']==1


def test_absent_native_evidence_is_not_inferred_from_agent_claim(tmp_path):
    (tmp_path/'agent-last.txt').write_text('I succeeded.')
    result=native_attempt_outcome(tmp_path)
    assert not result['native_success'] and not result['native_evidence_available']


def test_collection_handles_invalid_requests_and_preserves_native_outcome(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/run_subscription_dino_study.py'
    spec=importlib.util.spec_from_file_location('malformed_collection_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'agent-workspace/robot_data').mkdir(parents=True)
    (tmp_path/'tool-events.jsonl').write_text(json.dumps({'request':{'operation':{}},
        'response':{'ok':False,'error':'invalid operation'}})+'\n')
    (tmp_path/'evaluation-private.json').write_text(json.dumps({'attempts':[
        {'official_task_success':True,'final_task_success':True}]}))
    result=module.collect({'case_id':'test'},tmp_path,1,2.)
    assert result['status']=='runner_failure'
    assert result['first_attempt_success'] and result['physical_attempts_used']==1


def test_malformed_calls_do_not_break_coordinate_free_method_export(tmp_path):
    from run_subscription_learning_practice import relative_tool_choices
    events=[
        {'request':{'operation':'reset'},'response':{'ok':True,'result':{'attempt':1}}},
        {'request':{'operation':{'tool':'ground'}},'response':{'ok':False}},
        {'request':{'operation':'plan_arc','arguments':[]},'response':{'ok':False}},
        {'request':{'operation':'plan_arc','arguments':{'degrees':10}},
         'response':{'ok':True,'result':{'plan_id':'p1','kind':'arc'}}},
    ]
    (tmp_path/'tool-events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    choices=relative_tool_choices(tmp_path,[{'attempt':1,'official_task_success':True}])
    assert len(choices)==1 and choices[0]['provided_parameters']=={'degrees':10}


def test_report_renders_rejected_nonstring_operation(tmp_path,monkeypatch):
    import report_subscription_pilot as report
    run=tmp_path/'run';work=tmp_path/'workspace';run.mkdir();work.mkdir()
    (run/'agent-exit.json').write_text('{"returncode":0}')
    (run/'agent-events.jsonl').write_text('')
    (run/'tool-events.jsonl').write_text(json.dumps({'sequence':1,
        'request':{'operation':{'tool':'ground'},'arguments':{}},
        'response':{'ok':False,'error':'invalid operation'},'wall_seconds':.001})+'\n')
    monkeypatch.setattr('sys.argv',['report','--run',str(run),'--workspace',str(work)])
    report.main()
    assert (run/'audit.json').exists()
    assert '&lt;invalid operation: dict&gt;' in (run/'report/index.html').read_text()
