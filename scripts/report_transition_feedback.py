"""Render the declared offline feedback comparison with exact judgments and sources."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

from vlm_tamp.coupled_storage import read_json, write_json


def render(folder: Path) -> None:
    """Publish small text tables and image links; include every declared repeat."""
    protocol = read_json(folder / 'protocol.json')
    summary = read_json(folder / 'summary.json')
    rows = summary['comparisons']
    totals = {}
    for condition in protocol['conditions']:
        chosen = [row for row in rows if row['condition'] == condition]
        scores = [score for row in chosen for score in row['scores'].values()]
        totals[condition] = {'judgments': len(chosen), 'scored_claims': len(scores),
            'correct': sum(score['correct'] for score in scores),
            'uncertain': sum(score['abstained'] for score in scores),
            'false_assertions': sum(score['false_assertion'] for score in scores),
            'model_and_client_seconds': sum(row['timing']['model_and_client_seconds'] for row in chosen)}
    write_json(folder / 'metrics.json', totals)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Compact feedback: recorded judgments</title><style>'
            'body{font:16px/1.5 system-ui;max-width:1100px;margin:24px auto;padding:0 16px;overflow-wrap:anywhere}'
            'table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}'
            'img{max-width:100%;width:420px}pre{white-space:pre-wrap}.scroll{overflow:auto}</style>'
            '<h1>Compact feedback: recorded judgments</h1>'
            '<p>Fresh Luna/high decisions on identical public images and measurements. '
            'No returned action was executed. These selected development cases measure interpretation, '
            'not task success or generalization. Private labels are used for scoring only.</p>'
            '<p><a href="metrics.json">Aggregate metrics</a> · <a href="protocol.json">Declared protocol</a> · '
            '<a href="../watch.html">Live paired trials</a></p>']
    page.append('<div class="scroll"><table><tr><th>Format</th><th>Correct</th><th>Uncertain</th>'
                '<th>False assertions</th><th>Total model/client seconds</th></tr>')
    for condition, row in totals.items():
        page.append(f'<tr><td>{condition}</td><td>{row["correct"]}/{row["scored_claims"]}</td>'
                    f'<td>{row["uncertain"]}</td><td>{row["false_assertions"]}</td>'
                    f'<td>{row["model_and_client_seconds"]:.2f}</td></tr>')
    page.append('</table></div>')
    for index, case in enumerate(protocol['cases'], 1):
        source = Path(case['source'])
        context = read_json(source / 'input.json')
        page.append(f'<h2>{html.escape(case["name"])}</h2><p>Original text {case["original_bytes"]:,} bytes; '
                    f'compact {case["compact_bytes"]:,} bytes. Same {len(case["images"])} image attachments.</p>')
        for row in [item for item in rows if item['context'] == index]:
            assessment = row['decision']['assessment']
            page.append(f'<h3>{row["condition"]}, repeat {row["repeat"]} — '
                        f'{row["timing"]["model_and_client_seconds"]:.2f} s</h3>'
                        f'<p>Goal change: <strong>{html.escape(assessment["object_goal_change"])}</strong>; '
                        f'contact: <strong>{html.escape(assessment["contact_retention"])}</strong>.</p>'
                        f'<p>{html.escape(assessment["evidence"])}</p>'
                        f'<p><a href="{row["folder"]}/input.json">Exact input</a> · '
                        f'<a href="{row["folder"]}/prompt.txt">Prompt</a> · '
                        f'<a href="{row["folder"]}/decision.json">Full decision</a></p>')
        page.append('<details><summary>Identical image sequence supplied to both formats</summary>')
        for item in context['images']:
            path = source.parents[1] / 'live-images' / item['path']
            url = html.escape(os.path.relpath(path, folder), quote=True)
            caption = html.escape(f'Image {item["image_number"]}: {item["path"]}')
            page.append(f'<figure><img loading="lazy" src="{url}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
        page.append('</details>')
    page.append('</html>')
    (folder / 'watch.html').write_text('\n'.join(page))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    render(parser.parse_args().folder.resolve())
