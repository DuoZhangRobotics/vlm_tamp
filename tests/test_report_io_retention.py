import json
from pathlib import Path

import pytest

from report_contact_skills import read_events, skill_card
from report_luna_pipeline_validation import render
from vlm_tamp.coupled_storage import compact_public_observations


def test_large_search_arrays_stay_out_of_dashboard(tmp_path):
    result={'skill':'push','mode':'controlled','stop_reason':'start_configuration_blocked',
            'requested_contact_speed_m_s':.03,'feedback_count':2,'stages':[],
            'diagnostic':{'collision_error_mm':6.3,'part':'palm'},
            'search':{'attempts':[{'point':[1,2,3]} for _ in range(10000)]}}
    page=skill_card('probe',result,tmp_path/'tool-events.jsonl',tmp_path,{})
    assert len(page)<10000
    assert '6.3' in page and 'start_configuration_blocked' in page
    assert '10000' in page and 'Complete evidence' in page


def test_event_cache_invalidates_when_log_changes(tmp_path):
    path=tmp_path/'events.jsonl';path.write_text('{"sequence":1}\n')
    first=read_events(path)
    assert read_events(path) is first
    with path.open('a') as stream:stream.write('{"sequence":2}\n')
    assert [r['sequence'] for r in read_events(path)]==[1,2]


def test_unchanged_stopped_dashboard_does_not_rewrite_files(tmp_path):
    study=tmp_path/'study';study.mkdir();(study/'trials').mkdir()
    (study/'study-manifest.json').write_text(json.dumps({'phase':'stopped',
        'completed_runs':[],'experiments_stopped_by_user':True,'stop_reason':'I/O investigation'}))
    (study/'active-process.json').write_text('{"status":"stopped_by_user"}')
    render(study);page=study/'watch.html';before=page.stat().st_mtime_ns
    render(study)
    assert page.stat().st_mtime_ns==before
    assert 'Experiments stopped' in page.read_text()
    assert 'setTimeout' not in page.read_text()


def test_public_retention_preserves_endpoints_images_and_notes(tmp_path):
    for i in range(1,5):
        folder=tmp_path/'robot_data/attempt01'/f'obs{i:03d}'
        folder.mkdir(parents=True);(folder/'sensors.npz').write_bytes(bytes(i))
        (folder/'frontview.png').write_bytes(b'image')
    (tmp_path/'guidelines.json').write_text('{"principles":["learner note"]}')
    outside=tmp_path/'outside.npz';outside.write_bytes(b'keep')
    link=tmp_path/'robot_data/attempt01/obs005';link.mkdir()
    (link/'sensors.npz').symlink_to(outside)
    with pytest.raises(ValueError,match='finished'):
        compact_public_observations(tmp_path)
    result=compact_public_observations(tmp_path,finished=True)
    assert len(result['removed'])==2 and result['bytes_reclaimed_this_pass']==5
    assert (tmp_path/'robot_data/attempt01/obs001/sensors.npz').exists()
    assert (tmp_path/'robot_data/attempt01/obs004/sensors.npz').exists()
    assert len(list(tmp_path.glob('robot_data/attempt01/*/frontview.png')))==4
    assert outside.read_bytes()==b'keep'
    assert json.loads((tmp_path/'guidelines.json').read_text())['principles']==['learner note']


def test_public_retention_keeps_measurement_inputs(tmp_path):
    work=tmp_path/'workspace';work.mkdir()
    for i in range(1,5):
        folder=work/'robot_data/attempt01'/f'obs{i:03d}'
        folder.mkdir(parents=True);(folder/'sensors.npz').write_bytes(b'depth')
    event={'request':{'operation':'measure_surface','arguments':{'observation_id':2}},
           'response':{'result':{}}}
    (tmp_path/'tool-events.jsonl').write_text(json.dumps(event)+'\n')
    result=compact_public_observations(work,finished=True,run=tmp_path)
    assert [r['path'] for r in result['removed']]==['robot_data/attempt01/obs003/sensors.npz']


def test_unknown_custom_depth_dependencies_are_not_deleted(tmp_path):
    for i in range(1,4):
        folder=tmp_path/'robot_data/attempt01'/f'obs{i:03d}'
        folder.mkdir(parents=True);(folder/'sensors.npz').write_bytes(b'depth')
    (tmp_path/'analysis.py').write_text('data = np.load(selected_path)\n')
    result=compact_public_observations(tmp_path,finished=True)
    assert result['raw_input_dependencies_uncertain']
    assert result['removed']==[]


def test_supplied_image_viewer_does_not_pin_every_depth_export(tmp_path):
    from vlm_tamp.visual_inspection import install
    install(tmp_path)
    for i in range(1,4):
        folder=tmp_path/'robot_data/attempt01'/f'obs{i:03d}'
        folder.mkdir(parents=True);(folder/'sensors.npz').write_bytes(b'depth')
    result=compact_public_observations(tmp_path,finished=True)
    assert not result['raw_input_dependencies_uncertain']
    assert len(result['removed'])==1
    (tmp_path/'visual.py').write_text('values = np.load(dynamic_filename)\n')
    result=compact_public_observations(tmp_path,finished=True)
    assert result['raw_input_dependencies_uncertain']


def test_stopped_study_cannot_launch_an_actor(tmp_path,monkeypatch):
    import run_luna_pipeline_validation as runner
    study=tmp_path/'results/stopped';study.mkdir(parents=True)
    (study/'study-manifest.json').write_text('{"experiments_stopped_by_user":true}')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID','3')
    monkeypatch.setattr('sys.argv',['runner','--study',str(study),'--stage','baseline'])
    def unexpected(*args,**kwargs):raise AssertionError('Stopped study reached launch/freeze processing')
    monkeypatch.setattr(runner,'require_frozen',unexpected)
    monkeypatch.setattr(runner.runner,'run_actor',unexpected)
    with pytest.raises(RuntimeError,match='stopped by the user'):runner.main()
