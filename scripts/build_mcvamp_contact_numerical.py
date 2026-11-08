"""Build and test the pinned VAMP runtime with the recorded collision-tree repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/vlm_tamp/native/mcvamp_contact_numerical.cc'
BUILD = ROOT / '.runtime/mcvamp-contact-numerical'
UPSTREAM = ROOT / 'third_party/mc-vamp'
PATCH_MANIFEST = ROOT / 'patches/mcvamp-capt-neighbors.json'


def digest(path: Path) -> str:
    """Identify the exact file used by a build."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_reviewed_patch() -> dict:
    """Apply only the recorded patch to its known input; reject unknown sources."""
    manifest = json.loads(PATCH_MANIFEST.read_text())
    patch = PATCH_MANIFEST.parent / manifest['patch']
    header = UPSTREAM / manifest['file']
    revision = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != manifest['upstream_revision'] or digest(patch) != manifest['patch_sha256']:
        raise ValueError('VAMP revision or reviewed patch differs from the build manifest')
    if digest(header) == manifest['before_sha256']:
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=UPSTREAM, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=UPSTREAM, check=True)
    if digest(header) != manifest['after_sha256']:
        raise ValueError('VAMP collision header differs from the reviewed patch')
    return manifest


def publish(source: Path, destination: Path) -> None:
    """Replace a completed binary atomically; no running study may use this build."""
    temporary = destination.with_name(destination.name + '.building')
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> None:
    """Compile with parallel jobs and record all modified dependency identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eigen-include', default=os.environ.get('EIGEN3_INCLUDE_DIR'))
    args = parser.parse_args()
    candidates = ([Path(args.eigen_include)] if args.eigen_include else
                  [Path('/usr/include/eigen3'), ROOT / '.runtime/envs/rekep-libero/include/eigen3'])
    eigen = next((path.resolve() for path in candidates if (path / 'Eigen/Core').is_file()), None)
    if eigen is None:
        raise FileNotFoundError('Install Eigen or pass --eigen-include containing Eigen/Core')
    manifest = apply_reviewed_patch()
    BUILD.mkdir(parents=True, exist_ok=True)
    dependency_paths = [UPSTREAM / manifest['file'], PATCH_MANIFEST,
                        PATCH_MANIFEST.parent / manifest['patch']]
    dependencies = {str(path.relative_to(ROOT)): digest(path) for path in dependency_paths}
    source_hash = digest(SOURCE)
    build_directory = BUILD / 'cmake-build'
    upstream_build = UPSTREAM / 'build/cp310-cp310-linux_x86_64'
    commands = [
        ['cmake', '-S', str(SOURCE.parent), '-B', str(build_directory), '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DEIGEN_INCLUDE=' + str(eigen)],
        ['cmake', '--build', str(build_directory), '-j32'],
        ['ctest', '--test-dir', str(build_directory), '--output-on-failure'],
        ['cmake', '-S', str(UPSTREAM), '-B', str(upstream_build),
         '-DFETCHCONTENT_FULLY_DISCONNECTED=ON'],
        ['cmake', '--build', str(upstream_build), '--target', '_core_ext', '-j32'],
    ]
    with (BUILD / 'build.log').open('w') as log:
        for command in commands:
            log.write(' '.join(command) + '\n')
            log.flush()
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=log)
    if digest(SOURCE) != source_hash or any(digest(ROOT / name) != sha for name, sha in dependencies.items()):
        raise RuntimeError('Native sources changed during compilation; rebuild them')
    bindings = list(upstream_build.rglob('_core_ext.cpython-310-*.so'))
    if len(bindings) != 1:
        raise RuntimeError(f'Expected one compiled VAMP Python binding, found {len(bindings)}')
    binding = ROOT / '.runtime/mc-vamp-deps/vamp/_core' / bindings[0].name
    publish(bindings[0], binding)
    library = BUILD / 'libmcvamp_contact_numerical.so'
    publish(build_directory / library.name, library)
    dirty = subprocess.check_output(['git', '-C', str(UPSTREAM), 'diff', '--name-only'], text=True).splitlines()
    record = {'upstream_revision': manifest['upstream_revision'], 'source_sha256': source_hash,
              'library_sha256': digest(library), 'commands': commands,
              'upstream_sources_unchanged': not dirty, 'upstream_modified_files': dirty,
              'reviewed_patch': manifest, 'dependency_sha256': dependencies,
              'python_binding': str(binding.relative_to(ROOT)), 'python_binding_sha256': digest(binding),
              'native_tests_passed': True}
    temporary = BUILD / 'build.json.building'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(BUILD / 'build.json')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
