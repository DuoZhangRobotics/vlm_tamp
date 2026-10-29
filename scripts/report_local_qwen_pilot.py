"""Render local Qwen inputs, outputs, image proposals, timings and failures."""
import argparse
import html
import json
import os
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def pretty(value):
    return html.escape(json.dumps(value, indent=2, ensure_ascii=False))


def link(path, root, text):
    return f'<a href="{html.escape(os.path.relpath(path, root))}">{html.escape(text)}</a>' if path.exists() else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    cards = []
    offline = root / 'capability'
    cases = {x['id']: x for x in read(offline / 'cases-private.json', [])}
    rows = read(offline / 'results.json', [])
    for row in rows:
        folder = offline / 'responses' / row['id']
        timing = row.get('timing') or {}
        figure = ''
        case = cases[row['case']]
        answer = row.get('answer', {})
        regions = answer.get('arguments', {}).get('regions', {})
        for picture in case.get('images', []):
            img = offline / 'public' / picture['path']
            overlay = ''
            if picture['camera'] == regions.get('camera'):
                for key, color in [('contact_box_xyxy', '#ffb84d'), ('moving_panel_box_xyxy', '#65d4ff')]:
                    b = regions.get(key, [])
                    if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                        overlay += f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" fill="none" stroke="{color}" stroke-width="3"/>'
                points = regions.get('hinge_pixels_xy', [])
                if isinstance(points, list) and len(points) == 2 and all(isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in points):
                    overlay += f'<line x1="{points[0][0]}" y1="{points[0][1]}" x2="{points[1][0]}" y2="{points[1][1]}" stroke="#ff4d79" stroke-width="4"/>'
            figure += f'<figure><figcaption>{html.escape(picture["camera"])}</figcaption><svg viewBox="0 0 512 512"><image href="{html.escape(os.path.relpath(img, root))}" width="512" height="512"/>{overlay}</svg></figure>'
        cards.append(f'<article><h3>{html.escape(row["id"])} · {"pass" if row["passed"] else "fail"}</h3>'
            f'<p>{timing.get("wall_seconds", 0):.2f} s model request · first output {timing.get("first_output_seconds")}</p>'
            f'<p>{link(folder/"call001/request.json",root,"Input JSON")} · {link(folder/"call001/response.json",root,"Full model output")} · {link(folder/"result.json",root,"Grading and timing")}</p>'
            f'<div class="images">{figure}</div><details><summary>Submitted answer</summary><pre>{pretty(answer or row.get("error"))}</pre></details>'
            f'<details><summary>Input text</summary><pre>{html.escape(case["text"])}</pre></details></article>')
    coordinate = root / 'coordinate-format'
    coordinate_rows = read(coordinate / 'results.json', [])
    coordinate_protocol = read(coordinate / 'protocol-private.json', {})
    coordinate_cards = []
    for row in coordinate_rows:
        overlay = ''
        for box, color, dash in [(coordinate_protocol.get('reference_box_private'), '#7aff9c', '6,4'),
                                  (row.get('box_pixels'), '#ffb84d', 'none')]:
            if isinstance(box, list) and len(box) == 4:
                l,t,r,b = box
                overlay += f'<rect x="{l}" y="{t}" width="{r-l}" height="{b-t}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>'
        image = os.path.relpath(coordinate / 'public/frontview.png', root)
        folder = coordinate / 'responses' / row['id']
        coordinate_cards.append(f'<article><h3>{html.escape(row["id"])} · {"pass" if row["passed"] else "fail"}</h3>'
            f'<figure><svg viewBox="0 0 512 512"><image href="{html.escape(image)}" width="512" height="512"/>{overlay}</svg></figure>'
            f'<p>{link(folder/"call001/request.json",root,"Input")} · {link(folder/"call001/response.json",root,"Full output")} · {link(folder/"result.json",root,"Result and timing")}</p>'
            f'<pre>{pretty({k:v for k,v in row.items() if k not in {"timing"}})}</pre></article>')
    episodes = []
    for folder in sorted((root / 'episodes').glob('*')) if (root / 'episodes').exists() else []:
        result = read(folder / 'summary.json')
        progress = read(folder / 'actor-progress.json', {})
        data = result or progress
        turns = []
        for call in sorted((folder / 'model-calls').glob('call*')):
            response = read(call / 'response.json', {})
            timing = read(call / 'timing.json', {})
            tools = response.get('tool_calls', [])
            turns.append(f'<details><summary>{call.name} · {timing.get("wall_seconds",0):.2f} s · '
                         f'{html.escape(", ".join(x["function"]["name"] for x in tools))}</summary>'
                         f'<p>{link(call/"request.json",root,"Exact input text and image references")} · '
                         f'{link(call/"response.json",root,"Full model output")} · {link(call/"timing.json",root,"Timing and token usage")}</p>'
                         f'<pre>{pretty({k:v for k,v in response.items() if k not in {"reasoning", "reasoning_content"}})}</pre></details>')
        episodes.append(f'<article><h3>{folder.name} · {"success" if result and result["native_success"] else "finished" if result else "running"}</h3>'
                        f'<p>{link(folder/"local-tool-events.jsonl",root,"Tool requests and replies")} · '
                        f'{link(folder/"conversation.json",root,"Conversation")} · {link(folder/"summary.json",root,"Outcome and time breakdown")}</p>'
                        f'<pre>{pretty(data)}</pre>{"".join(turns)}</article>')
    memory = []
    if (root / 'server/gpu-memory.jsonl').exists():
        memory = [json.loads(x) for x in (root / 'server/gpu-memory.jsonl').read_text().splitlines()]
    peak = max((x['server_gpu_bytes'] for x in memory), default=0) / 2**30
    protocol = read(offline / 'protocol.json', {})
    summary = read(offline / 'summary.json', {'completed_responses': len(rows), 'planned_responses': 12})
    nonthinking = read(root / 'nonthinking/results.json', [])
    reference = read(root / 'reference-backend/transformers.json', {})
    reference_short = {k:v for k,v in reference.items() if k != 'output_token_ids'}
    findings = read(root / 'analysis.json', {})
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>Local Qwen 9B pilot</title>
<style>body{{font:16px/1.5 system-ui;background:#111823;color:#e8eef6;max-width:1200px;margin:auto;padding:28px}}a{{color:#8ed3ff}}article{{background:#1c2737;padding:20px;border-radius:12px;margin:20px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}}details{{margin:14px 0}}summary{{cursor:pointer}}.images{{display:flex;flex-wrap:wrap}}figure{{margin:8px;flex:1;min-width:240px;max-width:512px}}svg{{width:100%}}.note{{color:#b9c8dd}}</style></head>
<body><h1>Local Qwen3.5-9B pilot</h1><p>BF16 · local vLLM · 32,768-token context · one active sequence. Measured server-process peak: <strong>{peak:.2f} GiB</strong>, including allocated cache. Other users’ GPU allocations are excluded.</p>
<p class="note">These are development capability and integration checks. The local agent wrapper differs from Codex; outcomes are not a matched Luna comparison. Weights and supplied memory remain fixed. Visual proposal structure passing does not establish correct geometry.</p>
<p>{link(root/"checkpoint-verification.json",root,"Checkpoint verification")} · {link(root/"runtime-freeze.txt",root,"Runtime versions")} · {link(root/"server/server.log",root,"Server log")} · {link(root/"server/gpu-memory.jsonl",root,"GPU memory samples")}</p>
<h2>Findings</h2><pre>{pretty(findings)}</pre>
<h2>Capability checks</h2><pre>{pretty(summary)}</pre><p>Image overlays: orange = contact box, blue = moving panel, pink = proposed hinge. Original images are unchanged.</p>{''.join(cards)}
<h2>Coordinate-format follow-up</h2><p>One scene, fresh responses, two image-count conditions and two coordinate conventions. Orange = model box converted to pixels; dashed green = approximate human reference, never provided to the model.</p><pre>{pretty(read(coordinate/'summary.json', {'completed': len(coordinate_rows), 'planned': 8}))}</pre>{''.join(coordinate_cards)}
<h2>Thinking-mode follow-up</h2><p>Only enable_thinking changed for these two paired requests. {link(root/"nonthinking/paired-input-audit.json",root,"Paired input audit")}</p><pre>{pretty(nonthinking)}</pre>
<h2>Independent runtime check</h2><p>The same checkpoint and public image were also tested through Transformers. {link(root/"reference-backend/input.json",root,"Shared input")} · {link(root/"reference-backend/vllm.json",root,"vLLM result")} · {link(root/"reference-backend/transformers-input-metadata.json",root,"Reference tokenization and image grid")}</p><pre>{pretty(reference_short)}</pre>
<h2>Simulation episodes</h2>{''.join(episodes) or '<p>Not started.</p>'}
<details><summary>Registered capability protocol</summary><pre>{pretty(protocol)}</pre></details></body></html>'''
    (root / 'watch.html').write_text(body)
    print(json.dumps({'html': str(root / 'watch.html'), 'capability_responses': len(rows), 'episodes': len(episodes), 'peak_server_gpu_gib': peak}))


if __name__ == '__main__':
    main()
