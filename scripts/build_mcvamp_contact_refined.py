"""Build the small planning adapter using the already installed MC-VAMP headers."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/vlm_tamp/native/mcvamp_contact_refined.cc'
BUILD=ROOT/'.runtime/mcvamp-contact-refined'
UPSTREAM=ROOT/'third_party/mc-vamp'

def main():
    BUILD.mkdir(exist_ok=True)
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    original=UPSTREAM/'build/cp310-cp310-linux_x86_64'
    deps=original/'_deps'
    command=['g++','-std=c++17','-O3','-DNDEBUG','-march=native','-mavx2','-fPIC','-shared',
        '-Wno-ignored-attributes','-Wno-write-strings','-I'+str(UPSTREAM/'src/impl'),
        '-I/usr/include/eigen3','-I'+str(deps/'nigh-src/src'),'-I'+str(deps/'pdqsort-src'),
        '-I'+str(deps/'simdxorshift-src/include'),str(SOURCE),'-o',str(BUILD/'libmcvamp_contact_refined.so')]
    with (BUILD/'build.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=log)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=source_hash:
        raise RuntimeError('Adapter source changed during compilation; rebuild it')
    revision=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    record={'upstream_revision':revision,'source_sha256':source_hash,
        'library_sha256':hashlib.sha256((BUILD/'libmcvamp_contact_refined.so').read_bytes()).hexdigest(),
        'command':command,'upstream_sources_unchanged':True}
    (BUILD/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
