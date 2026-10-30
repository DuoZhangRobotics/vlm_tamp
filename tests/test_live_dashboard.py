import json
from pathlib import Path

import live_study_dashboard as dashboard


def fixture(tmp_path,monkeypatch):
    study=tmp_path/'study';run=study/'trials/case';run.mkdir(parents=True)
    manifest={'phase':'baseline','plan':{'baseline':[{}],'transfer':[]},'completed_runs':[]}
    (study/'study-manifest.json').write_text(json.dumps(manifest))
    (study/'active-process.json').write_text('{"run":"case","status":"running","pid":1}')
    (run/'protocol.json').write_text('{"case":{"task":"open_microwave","state_index":37}}')
    (run/'live-episode-status.json').write_text('{"control_steps":60,"maximum_control_steps":600}')
    (run/'tool-events.jsonl').write_text('')
    monkeypatch.setattr(dashboard,'_live_process',lambda _:True)
    return study,run,manifest


def test_frequent_tool_updates_do_not_rewrite_html_or_publish_too_often(tmp_path,monkeypatch):
    study,run,_=fixture(tmp_path,monkeypatch)
    first=dashboard.publish(study,now=0)
    page=study/'watch.html';stamp=page.stat().st_mtime_ns
    assert first['html_bytes_written']<20000
    (run/'live-episode-status.json').write_text('{"control_steps":200}')
    result=dashboard.publish(study,now=15)
    assert result['bytes_written']==0
    assert json.loads((run/'watch-data.json').read_text())['control_steps']==60
    result=dashboard.publish(study,now=60)
    assert result['html_bytes_written']==0
    assert page.stat().st_mtime_ns==stamp
    assert json.loads((run/'watch-data.json').read_text())['control_steps']==200


def test_terminal_boundary_publishes_immediately(tmp_path,monkeypatch):
    study,run,manifest=fixture(tmp_path,monkeypatch)
    dashboard.publish(study,now=0)
    manifest.update(experiments_stopped_by_user=True,stop_reason='User stop')
    (study/'study-manifest.json').write_text(json.dumps(manifest))
    assert dashboard.publish(study,now=5)['published']
    status=json.loads((study/'watch-status.json').read_text())
    assert not status['running'] and status['status_text'].startswith('Experiments stopped')


def test_huge_tool_payload_never_enters_main_status(tmp_path,monkeypatch):
    study,run,_=fixture(tmp_path,monkeypatch)
    e={'sequence':1,'request':{'operation':'push','arguments':{}},'wall_seconds':1,
       'response':{'ok':True,'result':{'stop_reason':'blocked','feedback_count':3,
       'diagnostic':{'positions':[[1,2,3]]*10000},'stages':[]}}}
    (run/'tool-events.jsonl').write_text(json.dumps(e)+'\n')
    dashboard.publish(study,now=0)
    assert (study/'watch-status.json').stat().st_size<5000
    assert (run/'watch-data.json').stat().st_size<10000
    assert json.loads((run/'watch-data.json').read_text())['skill']['Reported feedback count across tools']==3
    assert 'positions' not in (study/'watch.html').read_text()


def test_inactive_publisher_does_not_keep_writing_heartbeats(tmp_path,monkeypatch):
    study,run,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(dashboard,'_live_process',lambda _:False)
    dashboard.publish(study,now=0);stamp=(study/'watch-status.json').stat().st_mtime_ns
    assert dashboard.publish(study,now=120)['bytes_written']==0
    assert (study/'watch-status.json').stat().st_mtime_ns==stamp
    assert not json.loads((study/'watch-status.json').read_text())['running']


def test_static_page_uses_lazy_media_and_data_polling():
    assert 'location.reload' not in dashboard.CLIENT_JS
    assert "video.preload='none'" in dashboard.CLIENT_JS
    assert '60000' in dashboard.CLIENT_JS and '!document.hidden' in dashboard.CLIENT_JS
    assert 'watch-status.json' in dashboard.CLIENT_JS
    assert 'id="live" type="checkbox" checked' not in dashboard.HTML


def test_video_index_contains_every_saved_episode(tmp_path,monkeypatch):
    study,run,_=fixture(tmp_path,monkeypatch)
    for i in [1,2,3]:
        p=run/f'attempt{i:02d}/initial/rollout.mp4';p.parent.mkdir(parents=True);p.write_bytes(b'recorded')
    (run/'evaluation-private.json').write_text(json.dumps({'attempts':[{'attempt':1,'control_steps':600},{'attempt':2,'control_steps':400}]}))
    dashboard.publish(study,now=0)
    videos=json.loads((study/'watch-status.json').read_text())['runs'][0]['videos']
    assert len(videos)==3
    assert [v['partial'] for v in videos]==[False,False,True]


def test_failure_cause_and_evaluator_recheck_stay_separate_from_actor_log(tmp_path,monkeypatch):
    study,run,_=fixture(tmp_path,monkeypatch)
    event={'sequence':50,'request':{'operation':'plan_arc','arguments':{}},'wall_seconds':1,
        'response':{'ok':False,'error':'invalid_start','planning_diagnostic':{
            'start_rejection':{'blocking_robot_part':'finger','blocking_surface':'moving target',
                'overlap_evidence':{'overlap_depth_mm_range':[.2,.9]}}}}}
    original=json.dumps(event)+'\n';(run/'tool-events.jsonl').write_text(original)
    (run/'evaluator-diagnostics.json').write_text(json.dumps({'events':{'50':{
        'summary':'The repaired query produced a complete plan. Physical execution untested.'}}}))
    dashboard.publish(study,force=True)
    issue=json.loads((run/'watch-data.json').read_text())['issues'][0]
    assert 'finger / moving target' in issue['explanation'] and '0.200' in issue['explanation']
    assert 'Physical execution untested' in issue['evaluator_addendum']['summary']
    assert (run/'tool-events.jsonl').read_text()==original


def test_timeout_and_wrong_direction_explanations_name_the_failed_stage():
    text=dashboard.issue_explanation({'planning_stage':{'stage':'contact_insertion',
        'elapsed_seconds':1.9,'effective_seconds':1.8,'remaining_skill_seconds':20.}})
    assert 'contact insertion' in text and '20.00s left' in text
    text=dashboard.issue_explanation({'point_attempts':[{'reason':'wrong_side_contact'}]*4})
    assert 'all 4' in text and 'requested push direction' in text


def test_rerun_shows_scored_success_and_small_resource_snapshot(tmp_path,monkeypatch):
    study,run,manifest=fixture(tmp_path,monkeypatch)
    manifest.update(scheduled_stages=['baseline'],baseline_completed=True,
        completed_runs=[{'run':'case','stage':'baseline','attempts':[
            {'official_task_success':True},{'official_task_success':False}]}])
    (study/'study-manifest.json').write_text(json.dumps(manifest))
    sample=tmp_path/'sample.json';sample.write_text(json.dumps({'time':1,'job_memory_gib':8.,
        'job_memory_peak_gib':11.,'io_full_avg10':.1,'processes':[{'large':'x'*10000}]}))
    (study/'resource-monitor.json').write_text(json.dumps({'latest_sample':str(sample),'policy':'test watchdog'}))
    monkeypatch.setattr(dashboard,'_live_process',lambda _:False)
    dashboard.publish(study,force=True)
    status=json.loads((study/'watch-status.json').read_text())
    assert status['task_outcomes']=={'successes':1,'completed':2}
    assert list(status['totals'])==['baseline'] and status['status_text'].startswith('Complete')
    assert 'peak 11.0 GiB' in status['health_text']
    resource=json.loads((study/'resource-summary.json').read_text())
    assert resource['sample_interval_seconds']==2 and 'processes' not in resource
    assert (study/'resource-summary.json').stat().st_size<2500
