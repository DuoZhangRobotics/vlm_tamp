from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from vlm_tamp.storage_budget import InsufficientStorage,require_storage_headroom
from report_subscription_pilot import archive_workspace


def test_full_host_is_detected_despite_free_linux_space(monkeypatch,tmp_path):
    monkeypatch.setattr(Path,'is_mount',lambda p:str(p)=='/mnt/c')
    monkeypatch.setattr('vlm_tamp.storage_budget.shutil.disk_usage',
        lambda p:SimpleNamespace(free=(1 if str(p)=='/mnt/c' else 100)*1024**3))
    with pytest.raises(RuntimeError,match='Windows C:'):
        require_storage_headroom(tmp_path)


def test_finished_archive_moves_without_copying_and_preserves_notes(tmp_path):
    work=tmp_path/'work';archive=tmp_path/'archive';work.mkdir()
    note=work/'findings.txt';note.write_text('Measured outcome');inode=note.stat().st_ino
    (work/'robot_data').mkdir();(work/'robot_data/frame.txt').write_text('evidence')
    (work/'episodic_library').mkdir();(work/'episodic_library/prior.json').write_text('{}')
    (work/'external').symlink_to(tmp_path/'outside.txt')
    retained=archive_workspace(work,archive)
    assert (archive/'findings.txt').read_text()=='Measured outcome'
    assert (archive/'findings.txt').stat().st_ino==inode
    assert not note.exists()
    assert (archive/'robot_data/frame.txt').read_text()=='evidence'
    assert (work/'episodic_library/prior.json').exists()
    assert not (archive/'episodic_library').exists() and not (archive/'external').exists()
    assert retained==['findings.txt']
    assert archive_workspace(archive,archive)==['findings.txt']


def test_archive_conflict_keeps_both_versions(tmp_path):
    work=tmp_path/'work';archive=tmp_path/'archive';work.mkdir();archive.mkdir()
    (work/'findings.txt').write_text('new');(archive/'findings.txt').write_text('old')
    with pytest.raises(RuntimeError,match='preserving both'):
        archive_workspace(work,archive)
    assert (work/'findings.txt').read_text()=='new'
    assert (archive/'findings.txt').read_text()=='old'


def test_storage_block_stops_before_launch_and_retains_completed_count(tmp_path,monkeypatch):
    import run_recovery_learning_study as study
    (tmp_path/'status.json').write_text(json.dumps({'active':True,'completed':2,'planned':8}))
    def reject(*args,**kwargs):
        raise InsufficientStorage('host drive low')
    monkeypatch.setattr(study,'require_storage_headroom',reject)
    monkeypatch.setattr(study.subprocess,'run',lambda *a,**k:pytest.fail('No process should launch'))
    with pytest.raises(InsufficientStorage):
        study.run_logged(['unused'],tmp_path/'next.log')
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['phase']=='storage_blocked' and status['active'] is False
    assert status['completed']==2 and status['planned']==8
