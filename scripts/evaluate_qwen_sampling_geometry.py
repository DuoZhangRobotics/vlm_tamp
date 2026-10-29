"""Private, observer-only scoring of saved Qwen grounding proposals."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from validate_robot_io_geometry import load_case, choices, template
from vlm_tamp.hinged_contact_tools import ground_handle
from vlm_tamp.local_agent import write_json
from vlm_tamp.robot_contracts import observed_image_sizes, validate_operation


def evaluate(rows, output, *, case_loader=load_case, native_records=None):
    if native_records is None:
        native = json.loads((ROOT / 'results/astra-luna-comparison-20261023/grounding-diagnostics-private.json').read_text())
        native = {row['trial']: row for row in native}
    else:
        native = native_records
    cache, results = {}, []
    for row in rows:
        result = {'id': row['id'], 'contract_valid': row['contract_valid'],
                  'geometry_accepted': False, 'supplied_to_model': False}
        if not row['contract_valid']:
            result['skip_reason'] = 'No complete valid request; no geometry attempted.'
        else:
            case = row['source_private']['case']
            if case not in cache:
                cache[case] = case_loader(case)
            cameras, metadata, provenance = cache[case]
            regions = row['arguments']['regions']
            result['provenance'] = provenance
            started = time.monotonic()
            try:
                validate_operation('ground', row['arguments'], observed_image_sizes(cameras))
                mechanism, cloud, mask, anchors, evidence = ground_handle(
                    cameras, choices(regions), template(regions['camera']), metadata)
                delta = np.asarray(mechanism.spec['joint_origin_world_m']) - native[case]['native_hinge_anchor_world_m']
                result.update(geometry_accepted=True, model=mechanism.spec, fit_diagnostics=evidence,
                    native_hinge_line_error_mm=float(np.linalg.norm(delta[:2]) * 1000),
                    interpretation='Fit acceptance is a sensor-model check, not proof of correct mechanism or physical task success.')
            except Exception as error:
                result['error'] = f'{type(error).__name__}: {error}'
            result['geometry_seconds'] = time.monotonic() - started
        results.append(result)
        write_json(output, results)
        print(json.dumps({k: v for k, v in result.items() if k not in {'model', 'fit_diagnostics', 'provenance'}}), flush=True)
    return results


def evaluate_new(rows, output):
    """Append scores for newly completed immutable one-step requests."""
    previous = json.loads(output.read_text()) if output.is_file() else []
    done = {r['id'] for r in previous}
    fresh = [r for r in rows if r['id'] not in done]
    if not fresh:
        return previous
    temporary = output.with_name(output.stem + '-pending.json')
    additions = evaluate(fresh, temporary)
    combined = {r['id']: r for r in previous + additions}
    ordered = [combined[r['id']] for r in rows]
    write_json(output, ordered)
    temporary.unlink()
    return ordered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    evaluate_new(json.loads((root / 'study/results.json').read_text()), root / 'geometry-private.json')
