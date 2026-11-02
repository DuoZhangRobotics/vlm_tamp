"""Recorded RGB lookup must not read hidden state or invent intermediate depth."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from vlm_tamp.observed_video import inspect_recording
from vlm_tamp.robot_contracts import validate_operation


def recording(root: Path) -> Path:
    episode=root/'private';frames=episode/'initial/frames';frames.mkdir(parents=True)
    for step in (0,10,20,30,40,50):
        Image.new('RGB',(64,48),(step,100,120)).save(frames/f'step{step:06d}.jpg')
    (episode/'evaluation-private.json').write_text('not valid JSON: must not be read')
    return episode


def test_video_lookup_is_bounded_chronological_and_read_only(tmp_path):
    episode=recording(tmp_path);workspace=tmp_path/'public';folder=workspace/'strip'
    result=inspect_recording(episode,folder,workspace,10,50,4)
    steps=[f['control_step'] for f in result['frames']]
    assert len(steps)==4 and steps==sorted(steps) and steps[0]==10 and steps[-1]==50
    assert result['recorded_frames_in_interval']==5
    assert not result['depth_available'] and not result['new_sensor_capture'] and not result['robot_motion_issued']
    assert (workspace/result['comparison_image']).exists()
    assert all(f['simulated_seconds']==f['control_step']/20 for f in result['frames'])
    assert not (workspace/'requests').exists()


def test_specific_frame_and_crop_preserve_recorded_pixels(tmp_path):
    episode=recording(tmp_path);workspace=tmp_path/'public'
    result=inspect_recording(episode,workspace/'strip',workspace,20,20,8,[4,6,24,26])
    assert result['selected_frame_count']==1
    with Image.open(episode/'initial/frames/step000020.jpg') as source:
        expected=np.asarray(source.convert('RGB').crop((4,6,24,26)))
    np.testing.assert_array_equal(np.asarray(Image.open(workspace/result['frames'][0]['image'])),expected)
    missing=inspect_recording(episode,workspace/'missing',workspace,21,21)
    assert not missing['available'] and not (workspace/'missing').exists()


def test_completed_recording_uses_control_timestamps(tmp_path):
    from vlm_tamp.control_video import encode_control_video
    import shutil
    if not shutil.which('ffmpeg'):pytest.skip('Existing ffmpeg unavailable')
    episode=recording(tmp_path);paths=list((episode/'initial/frames').glob('*.jpg'))
    encode_control_video(paths,episode/'initial/rollout.mp4')
    for p in paths:p.unlink()
    workspace=tmp_path/'public'
    result=inspect_recording(episode,workspace/'strip',workspace,10,40,3)
    assert result['source']=='completed recorded MP4'
    assert [r['control_step'] for r in result['frames']]==[10,20,40]
    assert result['selected_frame_count']==3


def test_video_schema_requires_explicit_executed_interval():
    validate_operation('inspect_video',{'start_control_step':60,'end_control_step':90})
    with pytest.raises(ValueError):validate_operation('inspect_video',{'start_control_step':60})
    with pytest.raises(ValueError):validate_operation('inspect_video',{'start_control_step':60,'end_control_step':90,'max_frames':1000})
