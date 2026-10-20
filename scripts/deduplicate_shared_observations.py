"""Deduplicate byte-identical completed-run RGB-D archives without data loss."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    seen={};rows=[]
    for name in a.runs:
        root=(ROOT/name).resolve()
        if root.parent!=ROOT/'results' or not root.name.startswith('shared-feedback-'):
            raise ValueError('This cleanup is restricted to the new shared-feedback experiments')
        for path in sorted(root.glob('*state*/opened/obs.npz')):
            if path.is_symlink() or path.parent.is_symlink():continue
            summary=path.parent.parent/'summary.json'
            if not summary.exists() or 'controller_variant' not in read_json(summary):continue
            digest=sha(path);prior=seen.get(digest)
            if prior is None:seen[digest]=path;continue
            if path.stat().st_ino==prior.stat().st_ino and path.stat().st_dev==prior.stat().st_dev:continue
            before=path.stat();temporary=path.with_name(f'.obs-dedup-{os.getpid()}.tmp')
            if temporary.exists():raise ValueError('Unexpected cleanup temporary file')
            try:
                os.link(prior,temporary)
                if sha(temporary)!=digest:raise ValueError('Deduplication checksum changed')
                os.replace(temporary,path)
            finally:temporary.unlink(missing_ok=True)
            rows.append({'path':str(path),'same_bytes_as':str(prior),'sha256':digest,
                         'bytes':before.st_size,'allocated_bytes_reclaimed':before.st_blocks*512 if before.st_nlink==1 else 0})
    output=ROOT/a.output;previous=read_json(output) if output.exists() else {'deduplicated':[]}
    previous['deduplicated'].extend(rows)
    previous.update(byte_identical=True,observations_removed=False,hard_links_retain_data_if_another_path_is_removed=True,
                    allocated_bytes_reclaimed=sum(r['allocated_bytes_reclaimed'] for r in previous['deduplicated']))
    write_json(output,previous)
    print({'new_duplicate_archives':len(rows),'total_reclaimed_GiB':previous['allocated_bytes_reclaimed']/2**30})

if __name__=='__main__':main()
