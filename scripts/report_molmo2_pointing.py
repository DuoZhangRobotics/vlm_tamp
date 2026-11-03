"""Render the recorded Molmo2 outputs with unchanged pixels and timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics


ROOT = Path(__file__).resolve().parents[1]


def report(study: Path) -> None:
    """Combine model outputs and separate operator assessments without scoring depth as pointing."""
    protocol = json.loads((study / 'protocol.json').read_text())
    assessments = json.loads((study / 'assessments.json').read_text())
    rows = []
    for case in protocol['cases']:
        result = json.loads((study / 'queries' / case['id'] / 'result.json').read_text())
        rows.append({**case, 'result': result, **assessments['cases'][case['id']]})
    resources = json.loads((study / 'resource-summary.json').read_text())
    load = json.loads((study / 'load.json').read_text())
    data = {'cases': rows, 'load': load, 'conclusion': assessments['conclusion'],
            'warm_median_seconds': statistics.median(r['result']['total_seconds'] for r in rows[1:]),
            'total_generation_seconds': sum(r['result']['generation_seconds'] for r in rows),
            'resources': resources, 'resource_description': resources['description']}
    (study / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
    template = (ROOT / 'templates/molmo2_pointing.html').read_text()
    payload = json.dumps(data).replace('<', '\\u003c')
    (study / 'watch.html').write_text(template.replace('DATA_PLACEHOLDER', payload))
    shutil.copyfile(ROOT / 'templates/luna_documents.js', study / 'document-viewer.js')


def main() -> None:
    """Render the requested frozen study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    args = parser.parse_args()
    report(args.study)


if __name__ == '__main__':
    main()
