"""Versioned strict unknown-space correction; reuse the saved Luna answers only."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from vlm_tamp.rekep_libero import geometry, report
    from vlm_tamp.rekep_libero.strict_geometry import StrictObservedSDF
    geometry.ObservedSDF = StrictObservedSDF
    spec = importlib.util.spec_from_file_location('_rekep_libero_v1_runner', ROOT / 'scripts/run_rekep_libero.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    native_freeze = runner.freeze

    def strict_freeze(config, destination):
        destination = Path(destination)
        base = destination.with_name('frozen-base-protocol.json')
        native_freeze(config, base)
        manifest = json.loads(base.read_text())
        extras = [Path(__file__).resolve(), ROOT / 'configs/rekep_libero_v2.json',
                  ROOT / 'src/vlm_tamp/rekep_libero/strict_geometry.py']
        for path in extras:
            manifest['files_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['correction'] = 'Unknown-state points explicitly blocked independently of SDF interpolation'
        manifest['model_requests'] = 'Existing scene IDs and exact saved Luna request hashes; no additional API calls'
        if destination.exists() and json.loads(destination.read_text()) != manifest:
            raise RuntimeError('Frozen strict implementation changed')
        if not destination.exists():
            runner.dump(destination, manifest)

    runner.freeze = strict_freeze
    native_report = report.build_report

    def strict_report(episode_dirs, _output_dir, **kwargs):
        return native_report(episode_dirs, ROOT / 'results/rekep-libero-strict-report', **kwargs)

    report.build_report = strict_report
    if '--allow-paid' in sys.argv:
        raise ValueError('This correction reuses cached Luna answers; omit --allow-paid')
    if '--config' not in sys.argv:
        sys.argv += ['--config', 'configs/rekep_libero_v2.json']
    if '--phase' not in sys.argv:
        sys.argv += ['--phase', 'evaluation']
    runner.main()


if __name__ == '__main__':
    main()
