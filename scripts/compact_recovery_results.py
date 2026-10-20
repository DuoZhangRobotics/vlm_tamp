"""Lossless deduplication only within this completed recovery experiment."""
import hashlib
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json

RUNS=[ROOT/'results/vlm-recovery-ik-only-20261018',ROOT/'results/vlm-recovery-frozen-20261018']
OUT=ROOT/'results/vlm-recovery-tools-report'

def main():
    smoke=ROOT/'results/vlm-recovery-tool-smoke-20261018'
    if smoke.exists():RUNS.append(smoke)
    for run in RUNS:
        manifest=read_json(run/'manifest.json')['episodes']
        if not all((run/f"{r['task']}-state{r['state']:03d}-{r['policy']}"/'summary.json').exists() for r in manifest):
            raise ValueError('Never deduplicate a running experiment')
    groups={};dedup=[];storage=[]
    for run in RUNS:
        for summary in sorted(run.glob('*/summary.json')):
            folder=summary.parent;storage.append({'episode':str(folder),**read_json(summary)['storage']})
            path=folder/'opened/obs.npz'
            if not path.is_file() or path.is_symlink():raise ValueError('Expected an ordinary retained observation file')
            digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()
            key=(path.stat().st_size,digest)
            if key not in groups:groups[key]=path;continue
            source=groups[key]
            if path.stat().st_ino==source.stat().st_ino:continue
            allocated=path.stat().st_blocks*512;temp=path.with_name('obs-dedup.tmp')
            if temp.exists():raise ValueError('Unexpected unfinished deduplication')
            os.link(source,temp);os.replace(temp,path)
            dedup.append({'path':str(path),'source':str(source),'sha256':digest,'allocated_bytes_reclaimed':allocated})
    files=[p for root in RUNS for p in root.rglob('*') if p.is_file()]
    unique={}
    for path in files:
        stat=path.stat();unique[(stat.st_dev,stat.st_ino)]=stat.st_blocks*512
    OUT.mkdir(exist_ok=True)
    previous=read_json(OUT/'storage-summary.json') if (OUT/'storage-summary.json').exists() else {}
    dedup=previous.get('deduplicated',[])+dedup
    report={'episode_cleanup':storage,'deduplicated':dedup,
        'episode_bytes_reclaimed':sum(s['bytes_reclaimed_this_pass'] for s in storage),
        'dedup_allocated_bytes_reclaimed':sum(r['allocated_bytes_reclaimed'] for r in dedup),
        'retained_allocated_bytes':sum(unique.values()),'old_results_modified':False,
        'evidence_removed':False,'dedup_scope':'byte-identical initial RGB-D within completed new experiment only'}
    write_json(OUT/'storage-summary.json',report)
    print({k:v for k,v in report.items() if k not in ('episode_cleanup','deduplicated')})

if __name__=='__main__':main()
