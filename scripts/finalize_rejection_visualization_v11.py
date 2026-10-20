"""Finalize and validate the diagnostic report, preserving benchmark sources."""

from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for name, value in attrs if name in {'href', 'src'} and value)


def main():
    output = ROOT / 'results/rejected-grasp-visualization-v11'
    page = output / 'index.html'
    document = page.read_text()
    # Show solid geometry first. The signed-gap panel remains available directly
    # below it, with the detailed rotatable figures further down the page.
    previous = '<h1>Inspecting rejected grasp and motion poses</h1><img class="static" src="comparison.png">'
    replacement = ('<h1>Inspecting rejected grasp and motion poses</h1><img class="static" src="solid-comparison.png">'
        '<details><summary>Compare the observed-model signed gaps</summary><img class="static" src="comparison.png"></details>')
    document = document.replace(previous, replacement)
    page.write_text(document)
    links = Links()
    links.feed(document)
    local = [unquote(urlparse(s).path) for s in links.links if not urlparse(s).scheme and not s.startswith('#')]
    missing = [name for name in local if not (output / name).is_file()]
    protocol = json.loads((ROOT / 'results/physical-tasks-v11-eval/protocol.json').read_text())
    unchanged = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                    for name, digest in protocol['sources_sha256'].items())
    validation = json.loads((output / 'validation.json').read_text())
    validation.update(local_links_checked=len(local), missing_links=missing,
                      frozen_benchmark_source_hashes_match=unchanged,
                      no_external_figure_data_fetch_required='await fetch(el.dataset.figure)' not in document)
    validation['diagnostic_sources_sha256'] = {name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
        for name in ['visualize_rejected_grasps_v11.py','supplement_rejection_visualization_v11.py',
                     'render_rejected_pose_evaluation.py','finalize_rejection_visualization_v11.py']}
    (output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps({k:v for k,v in validation.items() if k != 'diagnostic_sources_sha256'}))
    if missing or not unchanged:
        raise ValueError('Diagnostic report validation failed')


if __name__ == '__main__':
    main()
