"""Inspect the shared I/O repair and the fixed-image Qwen retest."""
import argparse
import html
import json
import os
from pathlib import Path


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def pretty(value):
    return html.escape(json.dumps(value,indent=2,ensure_ascii=False))


def link(root,path,label):
    return f'<a href="{html.escape(os.path.relpath(path,root))}">{html.escape(label)}</a>' if path.exists() else ''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args()
    root=args.run.resolve();trial=root/'qwen-retest';rows=read(trial/'results.json',[])
    geometric={r['id']:r for r in read(root/'qwen-geometry-private.json',[])}
    cards=[];table=[]
    for row in rows:
        folder=trial/'responses'/row['id'];detail=[];last=row['attempts'][-1]
        fit=geometric.get(row['id'],{})
        if row['kind']=='contract':outcome='Open approach correct' if last.get('open_approach_correct') else 'Incorrect approach'
        elif not row['final_contract_valid']:outcome='Incomplete generation: output limit'
        elif fit.get('geometry_accepted'):outcome=f'Fit admitted; hinge error {fit["native_hinge_line_error_mm"]:.1f} mm (private audit)'
        else:outcome=fit.get('error','Geometry assessment pending')
        table.append(f'<tr><td>{row["id"]}</td><td>{row["primary_contract_valid"]}</td><td>{len(row["attempts"])}</td><td>{row["total_model_seconds"]:.2f}</td><td>{fit.get("geometry_seconds",0):.3f}</td><td>{html.escape(outcome)}</td></tr>')
        for attempt in row['attempts']:
            call=folder/f'call{attempt["attempt"]:03d}'
            detail.append(f'<details><summary>Call {attempt["attempt"]}: {"valid" if attempt["contract_valid"] else "rejected/incomplete"}</summary>'
                f'<p>{link(root,call/"request.json","Input and image references")} · {link(root,call/"response.json","Model output")} · {link(root,call/"timing.json","Timing and tokens")}</p><pre>{pretty(attempt)}</pre></details>')
        figure=''
        if row['kind']=='vision':
            regions=last.get('arguments',{}).get('regions',{})
            for camera in ['frontview','agentview']:
                image=trial/'public'/row['case']/(camera+'.png');overlay=''
                if regions.get('camera')==camera:
                    for field,color in [('contact_box_xyxy','#ffb84d'),('moving_panel_box_xyxy','#65d4ff')]:
                        b=regions.get(field)
                        if b:overlay+=f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" stroke="{color}" stroke-width="3" fill="none"/>'
                    p=regions.get('hinge_pixels_xy')
                    if p:overlay+=f'<line x1="{p[0][0]}" y1="{p[0][1]}" x2="{p[1][0]}" y2="{p[1][1]}" stroke="#ff4d79" stroke-width="4"/>'
                figure+=f'<figure><figcaption>{camera}</figcaption><svg viewBox="0 0 512 512"><image href="{html.escape(os.path.relpath(image,root))}" width="512" height="512"/>{overlay}</svg></figure>'
        cards.append(f'<article><h3>{row["id"]}</h3><p>{html.escape(outcome)}</p><div class="images">{figure}</div>{"".join(detail)}'
            f'<details><summary>Observer geometry assessment (never provided to model)</summary><pre>{pretty(fit)}</pre></details></article>')
    summary=read(root/'summary.json',{})
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Robot I/O contract repair</title><style>
body{{font:16px/1.5 system-ui;background:#111823;color:#e8eef6;max-width:1200px;margin:auto;padding:28px}}a{{color:#8ed3ff}}article{{background:#1c2737;padding:20px;border-radius:12px;margin:20px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}}details{{margin:14px 0}}summary{{cursor:pointer}}.images{{display:flex;flex-wrap:wrap}}figure{{margin:8px;flex:1;min-width:240px;max-width:512px}}svg{{width:100%}}td,th{{padding:9px;border-bottom:1px solid #42526a;text-align:left}}table{{border-collapse:collapse;width:100%}}
</style></head><body><h1>Shared robot I/O contract repair</h1>
<p>The shared service validates Luna and Qwen requests before numerical work. Qwen receives exact named schemas on demand. This study tests contract compliance separately from image localization; no simulator episode was run.</p>
<p>{link(root,root/'public-preflight/public-example/tools/ground.txt','New grounding help')} · {link(root,root/'public-preflight/public-example/tools/schemas/ground.json','Exact grounding schema')} · {link(root,root/'historical-luna-all-requests-audit.json','Historical Luna request audit')} · {link(root,root/'saved-geometry-regression.json','Numerical regression')} · {link(root,root/'final-audit.json','Final audit')}</p>
<h2>Findings</h2><pre>{pretty(summary)}</pre>
<h2>Qwen timing and outcomes</h2><p>Model time includes all retained calls, including a permitted correction turn. Geometry fit times exclude loading saved sensor arrays and were measured after model responses. Grasp planning, IK, and physical execution were not run. Orange = contact box; blue = door box; pink = hinge proposal.</p>
<table><thead><tr><th>Case</th><th>Valid first request</th><th>Model calls</th><th>Model seconds</th><th>Fit seconds</th><th>Outcome</th></tr></thead><tbody>{''.join(table)}</tbody></table>
{''.join(cards)}<details><summary>Registered model test</summary><pre>{pretty(read(trial/'protocol.json',{}))}</pre></details>
</body></html>'''
    (root/'watch.html').write_text(body)
    print(json.dumps({'html':str(root/'watch.html'),'cases':len(rows)}))


if __name__=='__main__':main()
