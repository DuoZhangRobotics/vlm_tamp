"""A pinned, public-only numerical Python bundle for isolated research actors.

This supplies ordinary array/geometry computation, never simulator code, robot
policies, project paths, model credentials or private experiment observations.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / '.runtime/public-analysis-cp312-v1'
DIRECTORIES = ('numpy', 'numpy.libs', 'numpy-1.26.4.dist-info',
               'scipy', 'scipy.libs', 'scipy-1.14.1.dist-info')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inventory(folder):
    result = {}
    for path in sorted(Path(folder).rglob('*')):
        if path.is_symlink():
            raise ValueError('Public analysis bundles must not contain symlinks')
        if path.is_file():
            result[str(path.relative_to(folder))] = digest(path)
    return result


def register_bundle(bundle, install_report):
    """Freeze installed, pinned wheels; this function performs no downloads."""
    bundle = Path(bundle).resolve()
    report = json.loads(Path(install_report).read_text())
    packages = {entry['metadata']['name'].lower(): entry['metadata']['version'] for entry in report['install']}
    if packages != {'numpy': '1.26.4', 'scipy': '1.14.1'}:
        raise ValueError('Unexpected public analysis package versions')
    target = bundle / 'payload'
    target.mkdir(exist_ok=False)
    for name in DIRECTORIES:
        shutil.copytree(bundle / 'packages' / name, target / name)
    files = inventory(target)
    value = {'schema_version': 1, 'python_major_minor': [3, 12], 'packages': packages,
             'files_sha256': files, 'install_report_sha256': digest(install_report)}
    (bundle / 'manifest.json').write_text(json.dumps(value, indent=2) + '\n')
    shutil.copyfile(install_report, bundle / 'install-report.json')
    return value


def load_manifest(bundle):
    bundle = Path(bundle).resolve()
    path = bundle / 'manifest.json'
    if not path.is_file():
        raise RuntimeError('Public numerical analysis bundle is not prepared: ' + str(path))
    manifest = json.loads(path.read_text())
    for relative in manifest['files_sha256']:
        name = PurePosixPath(relative)
        if (name.is_absolute() or '..' in name.parts or str(name) != relative
                or name.parts[0] not in DIRECTORIES):
            raise ValueError('Unsafe or unexpected public analysis file')
    if inventory(bundle / 'payload') != manifest['files_sha256']:
        raise ValueError('Public analysis bundle differs from its registered hashes')
    return manifest


def prepare_workspace(workspace, bundle=DEFAULT_BUNDLE):
    workspace, bundle = Path(workspace).resolve(), Path(bundle).resolve()
    manifest = load_manifest(bundle)
    site = workspace / '.analysis_site'
    if site.is_symlink():
        raise ValueError('Analysis library destination must not be a symlink')
    if not site.exists():
        shutil.copytree(bundle / 'payload', site)
    if inventory(site) != manifest['files_sha256']:
        raise ValueError('Workspace analysis libraries differ from the frozen bundle')
    public = {'python_major_minor': manifest['python_major_minor'], 'packages': manifest['packages'],
              'bundle_manifest_sha256': digest(bundle / 'manifest.json'),
              'files': len(manifest['files_sha256']),
              'scope': 'Public numerical libraries only; no robot implementation, hidden data or task strategy.'}
    path = workspace / 'analysis-environment.json'
    if path.is_symlink():
        raise ValueError('Analysis environment metadata must not be a symlink')
    if path.exists() and json.loads(path.read_text()) != public:
        raise ValueError('Existing analysis environment metadata differs')
    path.write_text(json.dumps(public, indent=2) + '\n')
    return public


def readonly_paths(workspace):
    work = Path(workspace).resolve()
    return [work / '.analysis_site', work / 'analysis-environment.json']


def command_environment(workspace, environment):
    """For the local sandbox preflight; never reuse this for the robot backend."""
    return {**environment, 'PYTHONPATH': str(Path(workspace).resolve() / '.analysis_site'),
            'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}


def configuration_arguments(workspace):
    """Apply the same public import path to every shell command and resume."""
    return ['-c', 'shell_environment_policy.set=' + '{' + ','.join(
        json.dumps(k) + '=' + json.dumps(v) for k, v in command_environment(workspace, {}).items()) + '}']


# Real numerical operations catch missing binary dependencies as well as names.
# The preflight also checks that model commands cannot rewrite the library.
PROBE_CODE = '''
import errno
import io
import json
from pathlib import Path
import sys
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from PIL import Image
expected = json.loads(Path('analysis-environment.json').read_text())
assert list(sys.version_info[:2]) == expected['python_major_minor']
assert np.__version__ == expected['packages']['numpy']
assert scipy.__version__ == expected['packages']['scipy']
a = np.array([[2., 1.], [1., 3.]])
assert np.allclose(a @ np.linalg.solve(a, np.array([1., 2.])), [1., 2.])
assert np.allclose(Rotation.from_euler('z', 90, degrees=True).apply([1., 0., 0.]), [0., 1., 0.])
buffer = io.BytesIO()
np.savez_compressed(buffer, points=np.eye(3))
buffer.seek(0)
with np.load(buffer) as values:
    assert np.array_equal(values['points'], np.eye(3))
assert Image.new('RGB', (2, 2), (3, 4, 5)).getpixel((0, 0)) == (3, 4, 5)
try:
    with Path(np.__file__).open('a'):
        pass
    library_readonly = False
except OSError as error:
    if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
        raise
    library_readonly = True
assert library_readonly, 'Numerical dependency files must be read-only to the actor'
analysis_probe = {'passed': True, 'python': sys.version.split()[0],
    'numpy': np.__version__, 'scipy': scipy.__version__, 'Pillow': Image.__version__,
    'linear_algebra': True, 'rigid_rotation': True, 'npz_roundtrip': True,
    'image_pixels': True, 'library_readonly': library_readonly}
'''
