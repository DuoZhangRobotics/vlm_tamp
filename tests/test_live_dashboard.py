import json
from pathlib import Path
import pytest

import live_study_dashboard as dashboard


def test_varied_practice_uses_compact_dashboard_and_preserves_episode_scores(tmp_path: Path) -> None:
    """The four-episode runner publishes small static HTML and counts each outcome."""
    from run_varied_contact_practice import save_study_manifest, render

    study = tmp_path / 'study'
    practice = study / 'trials' / 'clean-practice'
    practice.mkdir(parents=True)
    (practice / 'agent-exit.json').write_text('{"wall_seconds": 50}')
    attempts = [{'official_task_success': value, 'control_steps': 600} for value in [True, False, True, False]]
    manifest = {'plan': {'control_steps': 600, 'later_evaluation_cases': [{'task': 'close_microwave', 'state_index': 8}],
        'evaluation_memory_conditions': ['none', 'frozen']}, 'source_sha256': {},
        'practice_completed': True, 'practice_run': str(practice), 'practice_attempts': attempts,
        'evaluations': [], 'evaluations_pending': True}
    save_study_manifest(study, study / 'test-practice-manifest.json', manifest)
    render(study, force=True)
    status = json.loads((study / 'watch-status.json').read_text())
    assert status['task_outcomes'] == {'completed': 4, 'successes': 2}
    assert status['totals']['transfer']['planned'] == 2
    page = study / 'watch.html'
    assert page.stat().st_size < 20000
    assert 'watch.js' in page.read_text()
    stamp = page.stat().st_mtime_ns
    render(study)
    assert page.stat().st_mtime_ns == stamp


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


def test_io_admission_wait_is_not_labelled_as_memory_shortage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page identifies the actual resource preventing actor admission."""
    study, _, _ = fixture(tmp_path, monkeypatch)
    (study/'capacity-wait.json').write_text(json.dumps({'waiting': True,
        'wait_reasons': ['io_wait_without_responsive_login'], 'io_full_avg10': 30,
        'additional_headroom_needed_gib': 18, 'shared_headroom_gib': 140}))
    dashboard.publish(study, force=True)
    status = json.loads((study/'watch-status.json').read_text())
    assert status['status_text'].startswith('Waiting for host responsiveness')
    assert '30%' in status['message'] and 'login check' in status['message']


def test_missed_endpoint_is_visible_even_when_all_commands_were_issued(tmp_path,monkeypatch):
    study,run,_=fixture(tmp_path,monkeypatch)
    event={'sequence':5,'request':{'operation':'execute','arguments':{}},'wall_seconds':2,
        'response':{'ok':True,'result':{'execution':{'all_commands_issued':True,'endpoint_tracking':{
            'position_error_m':.07314,'orientation_error_deg':21.33,'within_controller_reporting_tolerance':False}}}}}
    (run/'tool-events.jsonl').write_text(json.dumps(event)+'\n')
    dashboard.publish(study,force=True)
    issue=json.loads((run/'watch-data.json').read_text())['issues'][0]
    assert issue['reason']=='reference_endpoint_not_reached'
    assert '73.1 mm' in issue['explanation'] and '21.3°' in issue['explanation']


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


def test_continuation_outcome_and_per_call_timings_are_visible(tmp_path,monkeypatch):
    study,run,_=fixture(tmp_path,monkeypatch)
    result={'skill':'push','stop_reason':'requested_stroke_complete','contact_reused':True,
        'scene_reused':True,'full_scene_builds':0,'contact_searches':0,
        'timing_seconds':{'stroke_planning_seconds':1.3,'execution_and_capture_seconds':3.2}}
    event={'sequence':12,'request':{'operation':'continue_push','arguments':{}},'wall_seconds':5.2,
        'response':{'ok':True,'result':result}}
    (run/'tool-events.jsonl').write_text(json.dumps(event)+'\n')
    dashboard.publish(study,force=True)
    data=json.loads((run/'watch-data.json').read_text())
    assert not data['issues']
    call=data['contact_calls'][0]
    assert call['wall_seconds']==5.2 and call['contact_searches']==0 and call['scene_reused']
    assert call['timing_seconds']['stroke_planning_seconds']==1.3
    assert data['timing']['nested_contact_skill_seconds']['execution_and_capture_seconds']==3.2


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


def test_evaluation_groups_keep_new_cases_and_repeats_separate(tmp_path,monkeypatch):
    study,run,manifest=fixture(tmp_path,monkeypatch)
    manifest['plan']['baseline']=[{'evaluation_group':name} for name in
        ['additional_closing','additional_opening','repeat_closing','repeat_closing']]
    manifest['completed_runs']=[{'run':'case','stage':'baseline','evaluation_group':'repeat_closing',
        'official_task_success':True,'attempts':[{'official_task_success':True}]}]
    (study/'study-manifest.json').write_text(json.dumps(manifest))
    dashboard.publish(study,force=True)
    groups=json.loads((study/'watch-status.json').read_text())['evaluation_groups']
    assert groups['repeat_closing']=={'planned':2,'finished':1,'successes':1}
    assert groups['additional_closing']==groups['additional_opening']=={'planned':1,'finished':0,'successes':0}


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
    assert 'all 4' in text and 'before IK/path search' in text


def test_physical_stall_explanation_distinguishes_motion_from_prediction():
    text=dashboard.issue_explanation({'feedback':{'stalled':True,'samples':22,'stall_window':{
        'simulation_seconds':1.,'commanded_travel_m':.03,'measured_directional_progress_m':.0008}}})
    assert '22 controls' in text and '30.0 mm' in text and '0.80 mm' in text
    assert 'not a point-cloud collision veto' in text


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
