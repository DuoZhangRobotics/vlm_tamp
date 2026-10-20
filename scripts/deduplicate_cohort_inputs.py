"""Deduplicate byte-identical immutable inputs/videos from completed trials.

JSON, policy snapshots, paths and planning-timing artifacts are never touched.
Every original pathname and byte remains available through a local hard link.
"""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.coupled_storage import read_json,write_json


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):result.update(data)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();primary=ROOT/'results/all-prior-tasks-frozen-v3-20261015'
    roots=[ROOT/'results'/name for name in ['all-prior-tasks-frozen-20261015','all-prior-tasks-frozen-v2-20261015','all-prior-tasks-frozen-v3-20261015']]
    by_size=defaultdict(list)
    for root in roots:
        for episode in sorted(root.glob('*-state???')):
            summary=episode/'summary.json'
            if not summary.exists() or 'storage' not in read_json(summary):continue
            for path in episode.rglob('*'):
                if path.is_symlink() or not path.is_file():continue
                eligible=path.name in {'obs.npz','robot_visual_meshes.npz','grasp_candidates.npz','input-cloud.npy','rollout.mp4'} or path.suffix=='.png'
                if eligible and path.stat().st_size>=65536:by_size[path.stat().st_size].append(path)
    groups=defaultdict(list)
    for size,paths in by_size.items():
        if len(paths)<2:continue
        for path in paths:groups[(size,digest(path))].append(path)
    links=[];saved=0
    for (size,sha),paths in groups.items():
        if len(paths)<2:continue
        canonical=paths[0]
        for path in paths[1:]:
            old=path.stat();first=canonical.stat()
            if (old.st_dev,old.st_ino)==(first.st_dev,first.st_ino):continue
            if old.st_dev!=first.st_dev:continue
            release=size if old.st_nlink==1 else 0
            if args.apply:
                temporary=path.with_name(path.name+f'.dedup-{os.getpid()}')
                try:
                    os.link(canonical,temporary);os.replace(temporary,path)
                finally:temporary.unlink(missing_ok=True)
                assert path.stat().st_ino==canonical.stat().st_ino
            saved+=release
            links.append({'path':str(path.relative_to(ROOT)),'canonical':str(canonical.relative_to(ROOT)),
                'sha256_unchanged':sha,'bytes':size,'unique_inode_bytes_reclaimed':release})
    result={'applied':args.apply,'identical_binary_links':links,'unique_inode_bytes_reclaimed':saved,
        'no_content_removed':True,'all_original_paths_preserved':True,'planning_timestamps_preserved':True}
    if args.apply:write_json(primary/'binary-deduplication.json',result)
    print({'applied':args.apply,'identical_binary_links':len(links),'unique_inode_bytes_reclaimed':saved})


if __name__=='__main__':main()
