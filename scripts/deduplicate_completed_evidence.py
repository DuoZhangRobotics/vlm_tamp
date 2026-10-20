"""Share byte-identical immutable evidence files in completed overnight runs.

No measured content or path is removed. Active runs, symlinks and old project
results are excluded. The default is a read-only inventory; --apply hard-links
verified identical files and records every replacement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from report_long_horizon_experiments import ROOT,OUT,RUNS,read


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def roots():
    result=[]
    for _,name in RUNS:
        folder=ROOT/'results'/name;summary=read(folder/'summary.json',{})
        if summary.get('planned',0)>0 and summary.get('completed')==summary['planned']:
            result.append(folder)
    for name in ['subscription-scene-sync-practice-20261020','subscription-prepared-sequence-practice-20261020']:
        folder=ROOT/'results'/name
        if (folder/'learning-summary.json').exists():result.append(folder)
    folder=ROOT/'results/robot-sequence-api-smoke-20261020'
    if (folder/'summary.json').exists() and (folder/'cleanup.json').exists():result.append(folder)
    folder=ROOT/'results/frozen-prepared-method-transfer-20261020'
    summary=read(folder/'summary.json',{})
    if summary.get('planned',0)>0 and summary.get('completed')==summary['planned']:result.append(folder)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();folders=roots();by_size={}
    for folder in folders:
        for path in folder.rglob('*'):
            if (not path.is_file() or path.is_symlink() or path.suffix not in {'.npz','.npy','.png','.mp4'}
                    or any(part.startswith('temporary-') for part in path.parts)):continue
            stat=path.stat()
            if stat.st_size>=1024*1024:by_size.setdefault(stat.st_size,[]).append(path)
    groups={};records=[];shared=0
    for size,paths in by_size.items():
        if len(paths)<2:continue
        for path in paths:
            stat=path.stat();sha=digest(path);key=(size,sha)
            if key not in groups:groups[key]=path;continue
            canonical=groups[key];base=canonical.stat()
            if (base.st_dev,base.st_ino)==(stat.st_dev,stat.st_ino):continue
            record={'path':str(path),'canonical':str(canonical),'sha256':sha,'bytes':size,'applied':False}
            if args.apply:
                # Recheck both participants immediately before atomic replacement.
                assert digest(canonical)==sha and digest(path)==sha
                now=path.stat()
                assert (now.st_ino,now.st_size,now.st_mtime_ns)==(stat.st_ino,stat.st_size,stat.st_mtime_ns)
                temporary=path.with_name('.'+path.name+'.dedup-'+uuid.uuid4().hex)
                try:
                    os.link(canonical,temporary)
                    assert digest(temporary)==sha
                    os.replace(temporary,path)
                    assert digest(path)==sha
                finally:
                    temporary.unlink(missing_ok=True)
                record['applied']=True
            records.append(record);shared+=size
    result={'mode':'apply' if args.apply else 'inventory','roots':[str(p) for p in folders],
        'byte_identical_files':len(records),'logical_duplicate_bytes':shared,
        'content_and_paths_preserved':True,'active_runs_excluded':True,'replacements':records}
    OUT.mkdir(exist_ok=True)
    destination=OUT/('evidence-deduplication.json' if args.apply else 'evidence-deduplication-inventory.json')
    if destination.exists():
        destination=destination.with_name(destination.stem+'-'+uuid.uuid4().hex[:8]+'.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='replacements'}))


if __name__=='__main__':main()
