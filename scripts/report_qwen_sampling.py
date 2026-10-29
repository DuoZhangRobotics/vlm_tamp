"""Inspectable sampling/grounding diagnostic with local live progress."""
import argparse
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path, fallback=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return fallback


def render(root):
    root = Path(root).resolve()
    rows = read(root / 'study/results.json', [])
    geometry = {r['id']: r for r in read(root / 'geometry-private.json', [])}
    status = read(root / 'study/status.json', {'phase': 'preparing'})
    assisted = {'initial_eight_turn_cases': read(root / 'tool-assisted/summary.json', {}),
                'documented_sixteen_turn_retest': read(root / 'tool-assisted-retest/summary.json', {})}
    esc = lambda x: html.escape(str(x), quote=True)
    num = lambda x: f'{x:,.2f}' if isinstance(x, (int, float)) else '—'
    link = lambda path, label: f'<a href="{esc(path)}">{esc(label)}</a>'
    cards, table = [], []
    for row in rows:
        ident = row['id']; prefix = f'study/responses/{ident}/call001'
        timing = row.get('timing') or {}; usage = timing.get('usage') or {}
        baseline = row.get('baseline_timing') or {}; old_usage = baseline.get('usage') or {}
        old = Path(os.path.relpath(ROOT / row['baseline_call'], root)).as_posix()
        g = geometry.get(ident, {})
        fit = ('Admitted; handle recovered by depth' if (g.get('fit_diagnostics') or {}).get('handle_recovery') else 'Admitted') if g.get('geometry_accepted') else g.get('error', g.get('skip_reason', 'Pending'))
        condition = 'Valid call' if row['contract_valid'] else row.get('failure', row.get('error', 'Invalid'))
        table.append(f'<tr><td>{esc(ident)}<small>{esc(row["phase"])}</small></td><td>{esc(condition)}</td>'
            f'<td>{num(baseline.get("wall_seconds"))} → {num(timing.get("wall_seconds"))}</td>'
            f'<td>{num(old_usage.get("completion_tokens"))} → {num(usage.get("completion_tokens"))}</td>'
            f'<td>{esc(fit)}</td><td>{num(g.get("native_hinge_line_error_mm"))}</td></tr>')
        req = read(root / prefix / 'request.json', {})
        resp = read(root / prefix / 'response.json', {})
        action = {k: v for k, v in resp.items() if k not in {'reasoning', 'reasoning_content'}}
        images = []
        regions = (row.get('arguments') or {}).get('regions', {})
        for message in req.get('messages', []):
            for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
                if part.get('type') != 'local_image':
                    continue
                path = part['path']; camera = Path(path).stem
                marks = ''
                if regions.get('camera') == camera:
                    for field, color in [('contact_box_xyxy', '#ffba59'), ('moving_panel_box_xyxy', '#73caff')]:
                        box = regions[field]
                        marks += f'<rect x="{box[0]}" y="{box[1]}" width="{box[2]-box[0]}" height="{box[3]-box[1]}" fill="none" stroke="{color}" stroke-width="3"/>'
                    a, b = regions['hinge_pixels_xy']
                    marks += f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#ff668e" stroke-width="3"/>'
                images.append(f'<figure><figcaption>{esc(camera)}</figcaption><svg viewBox="0 0 512 512"><image href="study/public/{esc(path)}" width="512" height="512"/>{marks}</svg></figure>')
        links = ' · '.join([link(prefix + '/request.json', 'Exact input'), link(prefix + '/response.json', 'Full output'),
            link(prefix + '/timing.json', 'Timing and tokens'), link(old + '/request.json', 'Baseline input'),
            link(old + '/response.json', 'Baseline output'), link(f'study/responses/{ident}/result.json', 'Validation')])
        cards.append(f'<article><h2>{esc(ident)}</h2><p>{links}</p><div class="images">{"".join(images)}</div>'
            '<p class="muted">Orange: handle; blue: panel; pink: proposed hinge. These are model proposals, not correct-answer annotations.</p>'
            f'<details><summary>Submitted function / final output</summary><pre>{esc(json.dumps(action, indent=2))}</pre></details>'
            f'<details><summary>Prompt and schema</summary><pre>{esc(json.dumps(req, indent=2))}</pre></details>'
            f'<details><summary>Geometry and repetition assessment</summary><pre>{esc(json.dumps({"geometry":g,"repetition":row.get("repetition")}, indent=2))}</pre></details></article>')
    primary = [r for r in rows if r['phase'] == 'primary']
    model_seconds = sum((r.get('timing') or {}).get('wall_seconds', 0) for r in rows)
    tokenization = sum(((r.get('timing') or {}).get('context_budget') or {}).get('tokenization_seconds', 0) for r in rows)
    geometric_seconds = sum(g.get('geometry_seconds', 0) for g in geometry.values())
    actor_cards = []
    for folder in [*sorted((root / 'tool-assisted').glob('fresh*')),
                   *sorted((root / 'tool-assisted-retest').glob('fresh*'))]:
        if not folder.is_dir():
            continue
        prefix = folder.parent.name + '/' + folder.name
        summary = read(folder / 'summary.json', {})
        calls = []
        for call in sorted((folder / 'model-calls').glob('call*')):
            timing = read(call / 'timing.json', {})
            usage = timing.get('usage') or {}
            response = read(call / 'response.json', {})
            names = ', '.join(c.get('function', {}).get('name', '') for c in response.get('tool_calls', []))
            path = prefix + '/model-calls/' + call.name
            output_link = link(path + '/response.json', 'Output') if (call / 'response.json').is_file() else link(path + '/partial-response.json', 'Partial output') if (call / 'partial-response.json').is_file() else 'Generating'
            calls.append(f'<tr><td>{esc(call.name)}</td><td>{esc(names or timing.get("finish_reason", "running"))}</td>'
                f'<td>{num(timing.get("wall_seconds"))}</td><td>{num(usage.get("completion_tokens"))}</td>'
                f'<td>{link(path+"/request.json","Input")} · {output_link} · {link(path+"/timing.json","Timing")}</td></tr>')
        private_link = link(prefix + '/geometry-private.json', 'Private final scoring') if (folder / 'geometry-private.json').is_file() else 'No private fit score yet'
        pictures = [p for p in sorted((folder / 'agent-workspace').glob('**/*')) if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}]
        gallery = ''.join(f'<figure><figcaption>{link(p.relative_to(root).as_posix(),p.relative_to(folder / "agent-workspace").as_posix())}</figcaption><img loading="lazy" src="{esc(p.relative_to(root).as_posix())}" style="max-width:100%;max-height:300px" alt="Retained actor image artifact"></figure>' for p in pictures)
        actor_cards.append(f'<article><h3>{esc(folder.parent.name)} / {esc(folder.name)}</h3><p>{link(prefix+"/conversation.json","Full conversation")} · '
            f'{link(prefix+"/tool-events.jsonl","Tool inputs, feedback and timings")} · '
            f'{link(prefix+"/ground-results.json","Grounding attempts")} · {private_link}</p>'
            '<table><thead><tr><th>Turn</th><th>Action</th><th>Client seconds</th><th>Output tokens</th><th>Logs</th></tr></thead>'
            f'<tbody>{"".join(calls)}</tbody></table><details><summary>Original observations and model-created image artifacts ({len(pictures)})</summary><div class="images">{gallery}</div></details><details><summary>Case result and time breakdown</summary><pre>{esc(json.dumps(summary,indent=2))}</pre></details></article>')
    title = 'Qwen sampling and grounding diagnostic'
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{font:16px/1.5 system-ui;background:#111a27;color:#ecf2fa;margin:28px auto;max-width:1250px;padding:0 24px}}a{{color:#91d4ff}}article,.box{{background:#1d2a3c;padding:20px;border:1px solid #405370;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #405370;text-align:left;vertical-align:top}}small{{display:block;color:#bdcbe0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:550px;overflow:auto;font-size:13px}}.muted{{color:#bdcbe0}}.scroll{{overflow:auto}}.images{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0;max-width:512px;min-width:260px;flex:1}}svg{{width:100%}}summary{{cursor:pointer}}button{{padding:8px 12px}}</style>
<h1>{title}</h1><p>Same saved inputs and 32,768-token output allowance. Primary comparison changes temperature 0.6 → 1.0 and presence penalty 0 → 1.5. No physical motion or learner-memory updates.</p>
<p>{link('protocol.json','Registered protocol')} · {link('study/summary.json','One-step summary')} · {link('geometry-private.json','Observer-only geometry')} · {link('summary.json','Overall findings')}</p>
<div class="box"><strong>Primary valid calls: {sum(r['contract_valid'] for r in primary)} / {len(primary)} completed requests</strong><p id="live">{esc(json.dumps(status))}</p><pre id="progress"></pre><button onclick="refresh()">Refresh progress</button><p class="muted" id="mode">Open with VSCode Live Preview for live progress. Re-run the report generator to refresh the full snapshot.</p></div>
<h2>Completion and geometry are separate</h2><div class="scroll"><table><thead><tr><th>Case</th><th>Output</th><th>Baseline → new seconds</th><th>Baseline → new output tokens</th><th>Numerical fit</th><th>Hinge error (mm)</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p class="muted">The three primary baselines used 32k. Confirmation baselines used 4k, so confirmation is a regression check rather than a sampling-only paired comparison. GPU timing is descriptive on shared hardware. Native hinge error is private evaluation and was never supplied in one-step requests.</p>
<h2>Time breakdown</h2><p>One-step client wall time: <strong>{num(model_seconds)} s</strong>, including exact prompt admission: <strong>{num(tokenization)} s</strong>. Observer-only geometry: <strong>{num(geometric_seconds)} s</strong>. Grasp planning, IK, arc planning and execution: <strong>0 s</strong> (offline diagnostic).</p>
{''.join(cards)}<h2>Tool-assisted observations and documented retest</h2><p>These observations use the same microwave asset, previously evaluated by Luna. The initial eight-turn cases were new to Qwen’s probes. Both ended before submitting ground. One registered retest reuses them with clearer writable paths/array layouts and a 16-turn cap. No correct-answer coordinates or previous model traces are supplied. Neither phase establishes transfer to new objects.</p><p>{link('tool-assisted/summary.json','Initial results')} · {link('tool-assisted-retest-protocol.json','Registered retest changes')} · {link('tool-assisted-retest/summary.json','Retest results')}</p>{''.join(actor_cards)}<details><summary>All tool-assisted results</summary><pre>{esc(json.dumps(assisted, indent=2))}</pre></details>
<script>
async function get(p){{try{{let r=await fetch(p,{{cache:'no-store'}});return r.ok?await r.json():null}}catch{{return null}}}}
async function refresh(){{if(location.protocol==='file:')return;let s=await get('study/status.json');let a=await get('tool-assisted/status.json');let b=await get('tool-assisted-retest/status.json');if(a&&a.phase!=='complete')s=a;if(b)s=b;if(s){{document.getElementById('live').textContent=JSON.stringify(s);if(s.current_call){{let p=await get(s.current_call+'/progress.json');document.getElementById('progress').textContent=p?JSON.stringify(p,null,2):'Waiting for first progress record';}}}}document.getElementById('mode').textContent='Progress refreshed '+new Date().toLocaleTimeString()+'; full case tables are the generated snapshot.'}}
if(location.protocol!=='file:'){{refresh();setInterval(refresh,5000)}}
</script></html>'''
    (root / 'watch.html').write_text(body)
    return root / 'watch.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(render(parser.parse_args().run))
