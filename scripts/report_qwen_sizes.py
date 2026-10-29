"""Render the inspectable 9B/27B comparison and collect exact request timings."""
import argparse
import csv
import html
import json
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/qwen-sampling-diagnostic-20261025'

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def main(root):
    esc=lambda x:html.escape(str(x),quote=True)
    num=lambda x:f'{x:,.2f}' if isinstance(x,(int,float)) else '—'
    link=lambda p,t:f'<a href="{esc(p)}">{esc(t)}</a>'
    rel=lambda p:Path(os.path.relpath(p,root)).as_posix()
    old={r['id']:r for r in read(BASE/'study/results.json',[])}
    rows={r['id']:r for r in read(root/'study/results.json',[])}
    before={r['id']:r for r in read(BASE/'geometry-private.json',[])}
    after={r['id']:r for r in read(root/'geometry27b-private.json',[])}
    raw27={r['id']:r for r in read(root/'raw-hinge27b-private.json',[])}
    raw9={r['id']:r for r in read(root/'raw-hinge9b-private.json',[])}
    protocol=read(root/'protocol.json',{});table=[];cards=[];timings=[]
    def collect_timings(folder,phase,case):
        calls=[]
        for call in sorted(folder.glob('call*')):
            request=read(call/'request.json',{});response=read(call/'response.json',{});timing=read(call/'timing.json',{});usage=timing.get('usage') or {}
            names=', '.join(c['function']['name'] for c in response.get('tool_calls',[]))
            if timing:timings.append({'phase':phase,'case':case,'call':call.name,'model':request.get('model'),
                'wall_seconds':timing.get('wall_seconds'),'context_admission_seconds':(timing.get('context_budget') or {}).get('tokenization_seconds'),
                'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),
                'reasoning_tokens':(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                'finish_reason':timing.get('finish_reason'),'tools':names,'error':timing.get('error')})
            action=names or timing.get('finish_reason','generating')
            links=[link(rel(call/'request.json'),'Input')]
            for name,title in [('response','Output'),('partial-response','Partial output'),('timing','Timing')]:
                if (call/(name+'.json')).is_file():links.append(link(rel(call/(name+'.json')),title))
            calls.append(f'<tr><td>{call.name}</td><td>{esc(action)}</td><td>{num(timing.get("wall_seconds"))}</td><td>{num(usage.get("completion_tokens"))}</td><td>{" · ".join(links)}</td></tr>')
        return calls
    for ident in protocol.get('initial_cases',[]):
        row=rows.get(ident,{});g=after.get(ident,{});b=before.get(ident,{});timing=row.get('timing') or {}
        table.append(f'<tr><td>{esc(ident)}</td><td>{num(old[ident]["timing"]["wall_seconds"])} → {num(timing.get("wall_seconds"))}</td><td>{esc(row.get("contract_valid","pending"))}</td><td>{esc(g.get("geometry_accepted","pending"))}</td><td>{num(b.get("native_hinge_line_error_mm"))} → {num(g.get("native_hinge_line_error_mm"))}</td><td>{num(raw9.get(ident,{}).get("mean_distance_to_projected_native_hinge_line_px"))} → {num(raw27.get(ident,{}).get("mean_distance_to_projected_native_hinge_line_px"))}</td><td>{esc(bool((g.get("fit_diagnostics") or {}).get("handle_recovery"))) if g else "—"}</td></tr>')
        folder=root/'study/responses'/ident;call=folder/'call001';request=read(call/'request.json',{});response=read(call/'response.json',{});pictures=[];regions=(row.get('arguments') or {}).get('regions',{})
        for m in request.get('messages',[]):
            for p in m.get('content',[]) if isinstance(m.get('content'),list) else []:
                if p.get('type')!='local_image':continue
                image_path=root/'study/public'/p['path'];camera=image_path.stem;marks=''
                if regions.get('camera')==camera:
                    for key,color in [('contact_box_xyxy','#ffbc65'),('moving_panel_box_xyxy','#71d9ff')]:
                        l,t,r,b=regions[key];marks+=f'<rect x="{l}" y="{t}" width="{r-l}" height="{b-t}" stroke="{color}" fill="none" stroke-width="3"/>'
                    a,b=regions['hinge_pixels_xy'];marks+=f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#ff5e8a" stroke-width="3"/>'
                pictures.append(f'<figure><figcaption>{esc(camera)}</figcaption><svg viewBox="0 0 512 512"><image href="{esc(rel(image_path))}" width="512" height="512"/>{marks}</svg></figure>')
        action={k:v for k,v in response.items() if k not in {'reasoning','reasoning_content'}}
        calls=collect_timings(folder,'one-step27b',ident)
        cards.append(f'<article><h3>{esc(ident)}</h3><p>{link(rel(BASE/"study/responses"/ident/"call001/response.json"),"9B baseline output")} · {link(rel(BASE/"study/responses"/ident/"call001/request.json"),"9B baseline input")}</p><table>{"".join(calls)}</table><div class="images">{"".join(pictures)}</div><p class="muted">Orange: proposed handle; blue: panel; pink: hinge. These are model proposals, not correct-answer annotations.</p><details><summary>27B submitted action</summary><pre>{esc(json.dumps(action,indent=2))}</pre></details><details><summary>Geometry / request diagnosis</summary><pre>{esc(json.dumps({"geometry":g,"error":row.get("error"),"failure":row.get("failure")},indent=2))}</pre></details></article>')
    actors=[]
    for phase in ['matched27b','fresh-qwen9b','fresh-qwen27b']:
        for folder in sorted((root/phase).glob('*')):
            if not folder.is_dir() or folder.name=='source-snapshot':continue
            summary=read(folder/'summary.json',{});calls=collect_timings(folder/'model-calls',phase,folder.name)
            links=[]
            for name,title in [('conversation.json','Conversation'),('tool-events.jsonl','Tool code and feedback'),('ground-results.json','Grounding attempts'),('geometry-private.json','Private scoring')]:
                if (folder/name).exists():links.append(link(rel(folder/name),title))
            photos=[]
            for p in sorted((folder/'agent-workspace').glob('**/*')):
                if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'}:photos.append(f'<figure><figcaption>{link(rel(p),p.name)}</figcaption><img loading="lazy" src="{esc(rel(p))}" style="max-width:100%;max-height:280px" alt="Model observation or generated analysis image"></figure>')
            actors.append(f'<article><h3>{esc(phase)} / {esc(folder.name)}</h3><p>{" · ".join(links)}</p><table><thead><tr><th>Turn</th><th>Action</th><th>Seconds</th><th>Output tokens</th><th>Logs</th></tr></thead><tbody>{"".join(calls)}</tbody></table><details><summary>Observations and model-generated images</summary><div class="images">{"".join(photos)}</div></details><details><summary>Outcome and time breakdown</summary><pre>{esc(json.dumps(summary,indent=2))}</pre></details></article>')
    if timings:
        with (root/'timing.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(timings[0]));w.writeheader();w.writerows(timings)
    headline=read(root/'summary.json',{})
    fresh_rows=''.join(f'<tr><td>{esc(r["id"])}</td><td>{esc(r["model"])}</td><td>{esc(r["termination_reason"])}</td><td>{num(r["selected_hinge_error_mm"])}</td><td>{r["model_calls"]}</td><td>{num(r["model_seconds"])}</td></tr>' for r in headline.get('fresh_tool_comparison',[]))
    totals=headline.get('model_totals',{});tool_totals=headline.get('tool_totals',{});scope=headline.get('offline_timing_scope',{})
    time_rows=[(model+' client',v['seconds'],str(v['requests'])+' new requests') for model,v in totals.items()]
    time_rows += [('Python analysis',tool_totals.get('run_python',{}).get('seconds',0),'Sandboxed numerical/image code'),('Grounding tool',tool_totals.get('ground',{}).get('seconds',0),'Observed-depth fitting'),('Private scoring',scope.get('one_step_observer_geometry_seconds',0)+scope.get('tool_case_private_scoring_seconds',0),'Observer-only; excluded from model feedback'),('Grasp / IK / arc / execution',0,'Not exercised in this offline comparison')]
    time_table=''.join(f'<tr><td>{esc(name)}</td><td>{num(seconds)}</td><td>{esc(detail)}</td></tr>' for name,seconds,detail in time_rows)
    overview=f'<h2>Fresh-observation controls</h2><table><thead><tr><th>State</th><th>Model</th><th>Workflow outcome</th><th>Selected hinge error mm</th><th>Calls</th><th>Client seconds</th></tr></thead><tbody>{fresh_rows}</tbody></table><h2>New-work timing breakdown</h2><table><thead><tr><th>Stage</th><th>Seconds</th><th>Scope</th></tr></thead><tbody>{time_table}</tbody></table><p class="muted">The totals cover different numbers of new requests; use matched cases to compare speed. Model-client time includes context admission and HTTP streaming. Private scoring is separate from actor tool time.</p>'
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen 9B versus 27B</title><style>body{{font:16px/1.5 system-ui;background:#101b2b;color:#e8eff9;max-width:1280px;margin:28px auto;padding:0 24px}}a{{color:#9bdcff}}article,.box{{padding:20px;border:1px solid #3e526e;background:#1c2a40;margin:18px 0;border-radius:12px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #3e526e;text-align:left;vertical-align:top}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:550px;overflow:auto;font-size:13px}}.scroll{{overflow:auto}}.images{{display:flex;flex-wrap:wrap;gap:12px}}figure{{margin:0;flex:1;min-width:250px;max-width:512px}}svg{{width:100%}}.muted{{color:#b7c9de}}summary{{cursor:pointer}}</style><h1>Qwen3.5: 9B versus 27B</h1><p>Corrected interface, BF16, 32,768 output allowance, temperature 1.0 and presence penalty 1.5. Same saved one-step requests, followed by bounded tool-assisted cases. Luna remains primary.</p><p>{link('protocol.json','Registered protocol')} · {link('checkpoint-verification.json','27B checkpoint verification')} · {link('timing.csv','Per-request timings')} · {link('summary.json','Overall results')}</p><div class="box"><strong>Live phase</strong><pre id="status">{esc(json.dumps(read(root/'work-status.json',{}),indent=2))}</pre><pre id="progress"></pre><p class="muted">Open with VSCode Live Preview for live progress. Tables and galleries reflect the most recently generated snapshot.</p></div><h2>Matched one-step results</h2><div class="scroll"><table><thead><tr><th>Case</th><th>9B → 27B seconds</th><th>27B valid call</th><th>27B fit admitted</th><th>9B → 27B hinge error mm</th><th>9B → 27B raw hinge-line error px</th><th>27B handle recovery</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p class="muted">Native scores are private observer measurements and were never supplied to either model. An admitted fit is not proof of a correct mechanism. Archived baselines and shared GPUs limit timing comparisons.</p>{''.join(cards)}<h2>Tool-assisted processes</h2><p>Matched cases reuse states020/035. Fresh states000/041 were absent from prior Qwen inputs but come from earlier Luna development observations of the same asset; no cross-asset transfer claim.</p>{''.join(actors)}<h2>Overall findings and timing</h2><pre>{esc(json.dumps(headline,indent=2))}</pre><script>async function get(p){{try{{let r=await fetch(p,{{cache:'no-store'}});return r.ok?await r.json():null}}catch{{return null}}}}async function refresh(){{if(location.protocol==='file:')return;let global=await get('work-status.json');let statuses=await Promise.all(['study','matched27b','fresh-qwen9b','fresh-qwen27b'].map(p=>get(p+'/status.json')));let active=statuses.filter(s=>s&&s.phase==='running').at(-1);document.getElementById('status').textContent=JSON.stringify(active??global,null,2);if(active?.current_call)document.getElementById('progress').textContent=JSON.stringify(await get(active.current_call+'/progress.json'),null,2);}}refresh();if(location.protocol!=='file:')setInterval(refresh,5000);</script></html>'''
    document=document.replace('<h2>Matched one-step results</h2>',overview+'<h2>Matched one-step results</h2>')
    (root/'watch.html').write_text(document)
    print(json.dumps({'html':str(root/'watch.html'),'completed_one_step':len(rows),'model_request_records':len(timings)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
