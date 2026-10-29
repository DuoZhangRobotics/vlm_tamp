"""Render the 32k Qwen output-budget study, including unfinished live calls.

The report is a standalone snapshot when opened from disk. Under VSCode Live
Preview or another local HTTP server it refreshes status/progress every 5 s.
Only public artifact files are read; this script never calls a model or robot.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'results/robot-io-contract-20261024/qwen-retest'
CASES = ('vision1-r1', 'vision2-r2', 'vision3-r2')


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        # Writers update progress while the report is reading it. A partial
        # JSON write must not prevent inspecting the remaining experiment.
        return default


def relative(path, root):
    return Path(os.path.relpath(path, root)).as_posix()


def snapshot(root):
    study = root / 'study'
    protocol = read(study / 'protocol.json', {})
    source = ROOT / protocol.get('source_study', str(BASELINE.relative_to(ROOT)))
    rows = read(study / 'results.json', [])
    by_id = {row['id']: row for row in rows}
    cases = []
    for ident in protocol.get('cases', CASES):
        folder = study / 'responses' / ident
        call = folder / 'call001'
        old_call = source / 'responses' / ident / 'call001'
        paths = {name: relative(call / (name + '.json'), root) for name in
                 ['request', 'response', 'timing', 'progress', 'partial-response']}
        paths['result'] = relative(folder / 'result.json', root)
        paths['old_request'] = relative(old_call / 'request.json', root)
        paths['old_response'] = relative(old_call / 'response.json', root)
        paths['old_timing'] = relative(old_call / 'timing.json', root)
        data = {'id': ident, 'paths': paths,
                'row': by_id.get(ident) or read(folder / 'result.json'),
                'old_timing': read(old_call / 'timing.json', {}),
                'old_request': read(old_call / 'request.json'),
                'request': read(call / 'request.json'),
                'response': read(call / 'response.json'),
                'partial': read(call / 'partial-response.json'),
                'timing': read(call / 'timing.json'),
                'progress': read(call / 'progress.json')}
        cases.append(data)
    return {'status': read(study / 'status.json', {'phase': 'not_started', 'completed': 0, 'planned': len(cases)}),
            'protocol': protocol, 'rows': rows, 'cases': cases,
            'study_summary': read(study / 'summary.json'),
            'root_summary': read(root / 'summary.json'),
            'geometry': read(root / 'geometry-private.json') or read(study / 'geometry-private.json'),
            'luna_audit': read(root / 'luna-budget-audit.json'),
            'final_audit': read(root / 'final-audit.json')}


DOCUMENT = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen output-budget test</title>
<style>
:root{color-scheme:dark}body{font:16px/1.5 system-ui;background:#111823;color:#e8eef6;max-width:1280px;margin:auto;padding:26px}a{color:#8ed3ff}h1{line-height:1.2}.muted{color:#b9c8dd}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.card,article{background:#1c2737;border:1px solid #34445b;border-radius:12px;padding:18px;margin:15px 0}.card strong{display:block;font-size:25px}.warn{border-left:4px solid #ffb84d;padding:12px 16px;background:#262839}button{padding:8px 14px;cursor:pointer;background:#29435e;border:1px solid #7698b6;border-radius:6px;color:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace;max-height:650px;overflow:auto}details{margin:14px 0}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #40516c;padding:9px;text-align:left;vertical-align:top}.scroll{overflow:auto}.images{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;flex:1;min-width:240px;max-width:512px}svg{width:100%;background:#18212e}meter{width:100%;height:16px}.good{color:#90e3b0}.bad{color:#ffb2b2}.pill{display:inline-block;background:#293b52;border-radius:5px;padding:2px 8px}#liveNote{font-size:14px}
</style></head><body>
<h1>Qwen output-budget test</h1>
<p>The same three previously truncated requests are replayed with a larger output allowance. Completion, valid tool calls, and geometric accuracy are separate outcomes. No physical robot episode or learner-memory update is part of this study.</p>
<div class="warn"><strong>Model-limit caveat:</strong> Luna’s recorded subscription launches request high reasoning but do not set an explicit output-token or context limit. Qwen’s 32k allowance is an explicit experimental budget, not verified parity with Luna. <a href="luna-budget-audit.json">Luna configuration audit</a>. The <a href="https://developers.openai.com/api/docs/models/gpt-6-luna">official Luna API card</a> advertises 128,000 output tokens and 1,050,000 context tokens; those API figures do not establish effective Codex/ChatGPT-subscription limits.</div>
<p><button id="refresh" type="button">Refresh results</button> <span id="liveNote" class="muted"></span></p>
<div id="headline"></div><div id="comparison"></div><div id="cases"></div>
<section id="later"></section>
<details id="protocol"><summary>Registered protocol</summary><pre id="protocolText"></pre></details>
<details id="luna"><summary>What the Luna audit establishes</summary><pre id="lunaText"></pre></details>
<script type="application/json" id="initial">__INITIAL__</script>
<script>
let state=JSON.parse(document.getElementById('initial').textContent);
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const num=x=>Number.isFinite(x)?x.toLocaleString(undefined,{maximumFractionDigits:2}):'—';
const pretty=x=>esc(JSON.stringify(x,null,2));
const clip=(x,n=6000)=>{const t=typeof x==='string'?x:JSON.stringify(x,null,2);return esc((t??'').slice(0,n))+((t??'').length>n?'\n[…display clipped; the linked JSON contains the complete record.]':'');};
const link=(url,label)=>`<a href="${esc(url)}">${esc(label)}</a>`;
const timing=c=>c.row?.timing??c.timing??{};
const usage=c=>timing(c).usage??c.progress?.usage??{};
function condition(c){
 const t=timing(c),r=c.row;
 if(r?.error||t.error)return 'Request error';
 if(r?.failure==='output_token_limit'||t.finish_reason==='length'||c.progress?.finish_reason==='length')return 'Output limit reached';
 if(r?.contract_valid)return 'Complete, valid tool call';
 if(r?.generation_completed)return 'Generation ended; invalid/missing tool call';
 if(t.finish_reason)return 'Generation ended; validation pending';
 if(c.progress||c.request)return 'Generating';
 return 'Waiting';
}
function images(c){
 const list=[];
 for(const m of (c.request??c.old_request)?.messages??[])for(const p of Array.isArray(m.content)?m.content:[]){
  if(p.type==='local_image'&&typeof p.path==='string'&&!p.path.startsWith('/')&&!p.path.includes('\\')&&!p.path.split('/').includes('..'))list.push(p.path);
 }
 const regions=c.row?.contract_valid?c.row.arguments?.regions:null;
 return list.map(path=>{
  const camera=path.split('/').at(-1).split('.')[0];const url='study/public/'+path.split('/').map(encodeURIComponent).join('/');let marks='';
  const finite=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(Number.isFinite);
  if(regions?.camera===camera){
   for(const [key,color] of [['contact_box_xyxy','#ffb84d'],['moving_panel_box_xyxy','#65d4ff']]){
    const b=regions[key];if(finite(b,4))marks+=`<rect x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" stroke="${color}" stroke-width="3" fill="none"/>`;
   }
   const p=regions.hinge_pixels_xy;if(Array.isArray(p)&&p.length===2&&p.every(x=>finite(x,2)))marks+=`<line x1="${p[0][0]}" y1="${p[0][1]}" x2="${p[1][0]}" y2="${p[1][1]}" stroke="#ff4d79" stroke-width="4"/>`;
  }
  return `<figure><figcaption>${link(url,camera+' — original image')}</figcaption><svg viewBox="0 0 512 512"><image href="${esc(url)}" width="512" height="512"/>${marks}</svg></figure>`;
 }).join('');
}
function geometric(id){
 const g=state.geometry;if(!g)return null;
 const rows=Array.isArray(g)?g:(g.results??g.cases??g.records);
 if(Array.isArray(rows))return rows.find(r=>r.id===id||r.case===id||r.case_id===id)??null;
 return g[id]??null;
}
function render(){
 const opened=[...document.querySelectorAll('details[open]')].map(x=>x.id);
 const p=state.protocol??{},s=state.status??{},cases=state.cases??[];
 const ended=cases.filter(c=>c.row).length,valid=cases.filter(c=>c.row?.contract_valid).length;
 const oldOutput=cases.reduce((a,c)=>a+(c.old_timing?.usage?.completion_tokens??c.row?.old_usage?.completion_tokens??0),0);
 const oldTime=cases.reduce((a,c)=>a+(c.old_timing?.wall_seconds??c.row?.old_seconds??0),0);
 const newOutput=cases.reduce((a,c)=>a+(usage(c).completion_tokens??0),0);
 const newTime=cases.reduce((a,c)=>a+(timing(c).wall_seconds??0),0);
 document.getElementById('headline').innerHTML=`<p><span class="pill">${esc(s.phase??'not started')}</span> ${s.current_case?'Current case: '+esc(s.current_case):''}</p><div class="cards"><div class="card">Processed requests<strong>${ended} / ${cases.length}</strong></div><div class="card">Valid tool calls<strong>${valid} / ${cases.length}</strong></div><div class="card">Output allowance<strong>${num(p.max_output_tokens??32768)}</strong>tokens per request</div><div class="card">Configured context<strong>${num(p.actual_context_tokens??65536)}</strong>input + generation</div></div><p class="muted">Request wall guard: ${num(p.request_seconds??900)} seconds. Context and time headroom were enlarged alongside the output allowance; only max_tokens changes in the saved model request.</p>`;
 document.getElementById('comparison').innerHTML=`<h2>4k versus 32k</h2><p class="muted">The 4k records are the three selected primary calls, not their correction turns. New usage totals cover only calls with reported usage; live character counts are not token counts. A maximum budget is not mandatory consumption.</p><p>Old totals: <strong>${num(oldOutput)}</strong> output tokens, <strong>${num(oldTime)} s</strong>. New reported totals: <strong>${num(newOutput)}</strong> output tokens, <strong>${num(newTime)} s</strong>.</p><div class="scroll"><table><thead><tr><th>Case</th><th>4k output / reasoning</th><th>4k seconds</th><th>32k output / reasoning</th><th>32k seconds</th><th>Outcome</th></tr></thead><tbody>${cases.map(c=>{const old=c.old_timing?.usage??c.row?.old_usage??{},u=usage(c);return `<tr><td>${esc(c.id)}</td><td>${num(old.completion_tokens)} / ${num(old.completion_tokens_details?.reasoning_tokens)}</td><td>${num(c.old_timing?.wall_seconds??c.row?.old_seconds)}</td><td>${num(u.completion_tokens)} / ${num(u.completion_tokens_details?.reasoning_tokens)}</td><td>${num(timing(c).wall_seconds??c.progress?.elapsed_seconds)}</td><td>${esc(condition(c))}</td></tr>`;}).join('')}</tbody></table></div>`;
 document.getElementById('cases').innerHTML=cases.map(c=>{
  const id=c.id,t=timing(c),progress=c.progress??{},active=!c.row&&!t.finish_reason&&!t.error&&(c.progress||c.request),reason=c.response?.reasoning??c.response?.reasoning_content??'',output=c.response?Object.fromEntries(Object.entries(c.response).filter(([k])=>!['reasoning','reasoning_content'].includes(k))):null;
  const links=[];if(c.request)links.push(link(c.paths.request,'Exact input'));
  if(c.response)links.push(link(c.paths.response,'Full output JSON'));
  if(c.timing)links.push(link(c.paths.timing,'Timing and token usage'));
  if(c.progress)links.push(link(c.paths.progress,'Progress JSON'));
  if(c.partial)links.push(link(c.paths['partial-response'],'Partial output JSON'));
  if(c.row)links.push(link(c.paths.result,'Validation result'));
  if(c.old_timing){links.push(link(c.paths.old_request,'4k input'));links.push(link(c.paths.old_timing,'4k timing'));links.push(link(c.paths.old_response,'4k output'));}
  let live='';if(active)live=`<div class="warn"><strong>Generation in progress</strong><p>${num(progress.elapsed_seconds)} s elapsed · ${num(progress.reasoning_characters)} reasoning characters · ${num(progress.content_characters)} answer characters · ${num(progress.tool_calls_seen)} tool calls seen.</p><meter min="0" max="${Number(p.request_seconds??900)}" value="${Number(progress.elapsed_seconds??0)}"></meter><small>Elapsed-time budget: ${num(p.request_seconds??900)} s. Stream events and character counts are progress indicators, not measured token usage.</small></div>`;
  const error=c.row?.error??c.row?.failure??t.error;
  return `<article><h2>${esc(id)}</h2><p><strong>${esc(condition(c))}</strong>${error?' · '+esc(error):''}</p>${live}<p>${links.join(' · ')}</p><div class="images">${images(c)}</div><p class="muted">When a valid proposal exists: orange = handle box, blue = moving panel, pink = hinge proposal. Original images are unchanged.</p>${output?`<details id="${id}-output"><summary>Submitted action / answer</summary><pre>${clip(output,12000)}</pre></details>`:''}${reason?`<details id="${id}-reason"><summary>Reasoning output — ${num(reason.length)} characters (collapsed)</summary><pre>${clip(reason)}</pre></details>`:''}${c.partial?`<details id="${id}-partial"><summary>Partial diagnostic output — never admitted for execution</summary><pre>${clip(c.partial)}</pre></details>`:''}${c.request?`<details id="${id}-input"><summary>Input text and schemas (collapsed)</summary><pre>${clip(c.request,14000)}</pre></details>`:''}${c.row?`<details id="${id}-result"><summary>Validation, input-equivalence and prefix comparison</summary><pre>${pretty(c.row)}</pre></details>`:''}${geometric(id)?`<details id="${id}-geometry"><summary>Private geometry assessment — never provided to model</summary><pre>${clip(geometric(id),14000)}</pre></details>`:''}</article>`;
 }).join('');
 document.getElementById('later').innerHTML=`${state.study_summary?'<h2>Study summary</h2><pre>'+pretty(state.study_summary)+'</pre>':''}${state.root_summary?'<details id="final-summary"><summary>Final findings</summary><pre>'+pretty(state.root_summary)+'</pre></details>':''}${state.geometry?'<details id="all-geometry"><summary>Geometry assessment records</summary><pre>'+clip(state.geometry,16000)+'</pre></details>':''}${state.final_audit?'<details id="final-audit"><summary>Final integrity audit</summary><pre>'+pretty(state.final_audit)+'</pre></details>':''}`;
 document.getElementById('protocolText').textContent=JSON.stringify(p,null,2);
 const audit=state.luna_audit??{};document.getElementById('lunaText').textContent=JSON.stringify({trials_checked:audit.trials_checked,commands_checked:audit.initial_and_resume_commands_checked,explicit_settings:audit.explicit_experiment_settings,token_limits:audit.runner_token_limit_settings,usage_observability:audit.usage_observability,official_api_card_reference:audit.official_api_card_reference,conclusion:audit.conclusion},null,2);
 for(const id of opened){const element=document.getElementById(id);if(element)element.open=true;}
}
async function get(path){try{const r=await fetch(path,{cache:'no-store'});return r.ok?await r.json():null;}catch{return null;}}
let busy=false;
async function refresh(){
 if(busy)return;if(location.protocol==='file:'){document.getElementById('liveNote').textContent='Snapshot view. Open with VSCode Live Preview for automatic updates, or rerun the report generator.';return;}
 busy=true;
 try{
  const [status,rows,protocol,summary,rootSummary,geometry,geometryStudy,audit]=await Promise.all(['study/status.json','study/results.json','study/protocol.json','study/summary.json','summary.json','geometry-private.json','study/geometry-private.json','final-audit.json'].map(get));
  if(status)state.status=status;if(rows)state.rows=rows;if(protocol)state.protocol=protocol;if(summary)state.study_summary=summary;if(rootSummary)state.root_summary=rootSummary;if(geometry||geometryStudy)state.geometry=geometry??geometryStudy;if(audit)state.final_audit=audit;
  const byId=Object.fromEntries((state.rows??[]).map(r=>[r.id,r]));
  await Promise.all(state.cases.map(async c=>{
   c.row=byId[c.id]??c.row;
   const [request,response,timing,progress,partial,result]=await Promise.all([c.paths.request,c.paths.response,c.paths.timing,c.paths.progress,c.paths['partial-response'],c.paths.result].map(get));
   if(request)c.request=request;if(response)c.response=response;if(timing)c.timing=timing;if(progress)c.progress=progress;if(partial)c.partial=partial;if(result)c.row=result;
  }));render();document.getElementById('liveNote').textContent='Updated '+new Date().toLocaleTimeString()+' · refreshing every 5 seconds';
 }finally{busy=false;}
}
render();document.getElementById('refresh').onclick=refresh;
if(location.protocol==='file:')refresh();else{refresh();setInterval(refresh,5000);}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = snapshot(root)
    initial = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    document = DOCUMENT.replace('__INITIAL__', initial)
    temporary = root / 'watch.html.pending'
    temporary.write_text(document)
    temporary.replace(root / 'watch.html')
    print(json.dumps({'html': str(root / 'watch.html'), 'cases': len(data['cases']),
                      'processed_requests': len(data['rows']), 'phase': data['status'].get('phase')}))


if __name__ == '__main__':
    main()
