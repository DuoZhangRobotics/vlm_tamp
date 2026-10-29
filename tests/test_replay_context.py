import gzip
import json
from replay_subscription_episode import source_case,source_reference_mode


def test_replay_uses_episode_case_in_a_curriculum_and_preserves_budget(tmp_path):
    (tmp_path/'protocol.json').write_text(json.dumps({'case':{'task':'close_microwave','state_index':17},
        'maximum_control_steps_per_attempt':1200}))
    episode=tmp_path/'attempt02';episode.mkdir()
    (episode/'summary.json').write_text(json.dumps({'task':'open_microwave','state_index':13,'maximum_control_steps':1200}))
    case=source_case(tmp_path,2)
    assert case['task']=='open_microwave' and case['state_index']==13
    assert case['task_name']=='KITCHEN_SCENE7_open_the_microwave' and case['maximum_control_steps']==1200


def test_compressed_controller_metadata_preserves_historical_mode(tmp_path):
    folder=tmp_path/'attempt01/initial';folder.mkdir(parents=True)
    with gzip.open(folder/'controller-changes.json.gz','wt') as stream:
        json.dump([{'reference_mode':'slew_limited'}],stream)
    assert source_reference_mode(tmp_path)=='slew_limited'
