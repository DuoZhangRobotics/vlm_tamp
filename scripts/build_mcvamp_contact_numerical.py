"""Build the small planning adapter using the already installed MC-VAMP headers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/vlm_tamp/native/mcvamp_contact_numerical.cc'
BUILD=ROOT/'.runtime/mcvamp-contact-numerical'
UPSTREAM=ROOT/'third_party/mc-vamp'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eigen-include',default=os.environ.get('EIGEN3_INCLUDE_DIR'),
                        help='Directory containing Eigen/Core; defaults to the system or project-local headers')
    args=parser.parse_args()
    candidates=([Path(args.eigen_include)] if args.eigen_include else
                [Path('/usr/include/eigen3'),ROOT/'.runtime/envs/rekep-libero/include/eigen3'])
    eigen=next((p.resolve() for p in candidates if (p/'Eigen/Core').is_file()),None)
    if eigen is None:raise FileNotFoundError('Install Eigen or pass --eigen-include containing Eigen/Core')
    BUILD.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    original=UPSTREAM/'build/cp310-cp310-linux_x86_64'
    deps=original/'_deps'
    command=['g++','-std=c++17','-O3','-DNDEBUG','-march=native','-mavx2','-fPIC','-shared',
        '-Wno-ignored-attributes','-Wno-write-strings','-I'+str(UPSTREAM/'src/impl'),
        '-I'+str(eigen),'-I'+str(deps/'nigh-src/src'),'-I'+str(deps/'pdqsort-src'),
        '-I'+str(deps/'simdxorshift-src/include'),str(SOURCE),'-o',str(BUILD/'libmcvamp_contact_numerical.so')]
    with (BUILD/'build.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=log)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=source_hash:
        raise RuntimeError('Adapter source changed during compilation; rebuild it')
    revision=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    record={'upstream_revision':revision,'source_sha256':source_hash,
        'library_sha256':hashlib.sha256((BUILD/'libmcvamp_contact_numerical.so').read_bytes()).hexdigest(),
        'command':command,'upstream_sources_unchanged':True}
    (BUILD/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
