"""Prevent new experiments from filling the host drive behind WSL."""
from pathlib import Path
import shutil


class InsufficientStorage(RuntimeError):
    pass


def require_storage_headroom(workspace, minimum_free_bytes=20*1024**3):
    paths=[('experiment filesystem',Path(workspace))]
    host=Path('/mnt/c')
    if host.is_mount():
        paths.append(('Windows C:',host))
    rows=[]
    for label,path in paths:
        free=shutil.disk_usage(path).free
        rows.append({'label':label,'path':str(path),'free_bytes':free,
                     'required_free_bytes':minimum_free_bytes})
    short=[r for r in rows if r['free_bytes']<minimum_free_bytes]
    if short:
        descriptions='; '.join(f"{r['label']}: {r['free_bytes']/1024**3:.1f} GiB free" for r in short)
        raise InsufficientStorage(f"Storage reserve not met ({descriptions}); need {minimum_free_bytes/1024**3:.0f} GiB before starting another experiment. Clean retained outputs and reclaim host space first.")
    return rows
