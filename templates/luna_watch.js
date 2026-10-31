'use strict';
const el=id=>document.getElementById(id);
const key='luna-inspection:'+location.pathname;
let saved={};try{saved=JSON.parse(sessionStorage.getItem(key)||'{}')}catch{}
const initialScroll=saved.scrollY||0;
let state=null,chosen=new URLSearchParams(location.search).get('run')||saved.chosen||null;
let selectedVersion=null,overviewRequest=null,detailBusy=false,restoring=false;
const cards=new Map(),rows=new Map(),detailsCache=new Map();
const num=x=>typeof x==='number'?x.toFixed(1):'—';
function text(id,value){el(id).textContent=value??''}
function node(tag,value){const n=document.createElement(tag);if(value!==undefined)n.textContent=value;return n}
function link(parent,label,path){if(!path)return;const a=node('a',label);a.href=path;a.target='_blank';a.rel='noopener';parent.append(a)}
function table(parent,values){const t=node('table');for(const row of values){const tr=t.insertRow();for(const value of row)tr.insertCell().textContent=value??'—'}parent.append(t)}
function save(){
 saved.chosen=chosen;saved.scrollY=scrollY;saved.open=saved.open||{};saved.video=saved.video||{};
 if(chosen&&selectedVersion!==null&&!el('detailPanel').hidden){const open={};for(const d of el('detailPanel').querySelectorAll('details[data-open-key]'))open[d.dataset.openKey]=d.open;saved.open[chosen]=open}
 for(const [path,c] of cards)if(c.video.getAttribute('src'))saved.video[path]=c.video.currentTime;
 try{sessionStorage.setItem(key,JSON.stringify(saved))}catch{}
}
function pauseDisplay(reason='Manual display updates. The experiment continues independently.'){
 el('live').checked=false;text('displayMode',reason);save();
}
function image(id,path){const n=el(id);n.hidden=!path;if(path&&n.getAttribute('src')!==path)n.src=path;if(!path)n.removeAttribute('src')}
function showImageWhenOpen(detailsId,imageId){const d=el(detailsId),n=el(imageId);if(d.open)image(imageId,n.dataset.path||null)}
function titleFor(row){const match=row.id.match(/baseline-r(\d+)-/);return row.label+(match?' · repetition '+match[1]:'')}
function addVideo(row,item){
 if(cards.has(item.path)){const c=cards.get(item.path);c.outcome.textContent=row.outcome;return}
 const box=node('article');box.dataset.videoPath=item.path;
 const h=node('h3',titleFor(row)),label=node('p',item.label),outcome=node('p',row.outcome);
 outcome.className='muted';const play=node('button','Play'),inspect=node('button','Inspect this run');
 play.dataset.play=item.path;inspect.dataset.inspect=row.id;
 const video=node('video');video.controls=true;video.preload='none';video.hidden=true;
 const note=node('p','Video loads only when requested.');note.className='muted';
 const remembered=saved.video?.[item.path]||0;
 if(remembered>0)play.textContent='Resume at '+remembered.toFixed(1)+' s';
 play.onclick=async()=>{
  pauseDisplay('Display updates paused for video inspection. Experiments continue independently.');
  video.hidden=false;
  if(!video.getAttribute('src')){
   video.src=item.path;
   video.addEventListener('loadedmetadata',()=>{if(remembered>0)video.currentTime=Math.min(remembered,Math.max(0,video.duration-.1))},{once:true});
   video.load();
  }
  try{await video.play()}catch(error){note.textContent='Use the video controls to play. '+error.message}
 };
 inspect.onclick=()=>inspectRun(row.id,true);
 video.addEventListener('play',()=>pauseDisplay('Display updates paused for video inspection. Experiments continue independently.'));
 video.addEventListener('pause',()=>{play.textContent=video.ended?'Replay':'Resume at '+video.currentTime.toFixed(1)+' s';save()});video.addEventListener('seeked',save);
 video.addEventListener('timeupdate',()=>{saved.video=saved.video||{};saved.video[item.path]=video.currentTime});
 const buttons=node('p');buttons.append(play,document.createTextNode(' '),inspect);
 box.append(h,label,outcome,buttons,video,note);el('videos').append(box);
 cards.set(item.path,{box,video,outcome});
}
function updateOverview(next){
 state=next;text('status',state.status_text);text('message',state.message);
 const age=Date.now()/1000-state.checked_at;
 text('stamp','Last published check: '+new Date(state.checked_at*1000).toLocaleString()+(state.running&&age>150?' · Saved status is stale.':''));
 el('totals').replaceChildren();for(const [name,v] of Object.entries(state.totals||{})){
  const d=node('div');d.append(node('b',v.finished+' / '+v.planned),document.createTextNode(name));el('totals').append(d);
 }
 if(state.task_outcomes){const d=node('div');d.append(node('b',state.task_outcomes.successes+' / '+state.task_outcomes.completed),document.createTextNode('task successes / completed episodes'));el('totals').append(d)}
 el('studyLinks').replaceChildren();for(const [name,path] of Object.entries(state.links||{}))link(el('studyLinks'),name,path);
 for(const row of state.runs||[]){
  let entry=rows.get(row.id);
  if(!entry){const tr=node('tr');tr.dataset.run=row.id;const cells=Array.from({length:5},()=>tr.appendChild(node('td')));
   const button=node('button','Inspect');button.dataset.inspect=row.id;button.onclick=()=>inspectRun(row.id,true);
   const action=node('td');action.append(button);tr.append(action);el('runs').append(tr);entry={cells};rows.set(row.id,entry);
  }
  [titleFor(row),row.memory,row.outcome,row.controls,num(row.actor_seconds)+' s'].forEach((v,i)=>{entry.cells[i].textContent=v??'—'});
  for(const video of row.videos||[])addVideo(row,video);
 }
 text('videoCount','('+cards.size+')');text('videoEmpty',cards.size?'':'No encoded recordings yet. New episode videos appear when they are saved.');
 text('health',state.health_text||'');
 const selected=(state.runs||[]).find(r=>r.id===chosen);
 text('updateNote',selected&&selectedVersion!==null&&selected.detail_version!==selectedVersion?'New data is available. Press Refresh this trial to load it.':'');
}
async function fetchDetail(row,force=false){
 const cached=detailsCache.get(row.id);if(!force&&cached?.version===row.detail_version)return cached.data;
 const response=await fetch(row.detail+'?v='+row.detail_version+(force?'&refresh='+Date.now():''),{cache:force?'no-store':'no-cache'});
 if(!response.ok)throw Error('Run details are not available yet');const data=await response.json();
 detailsCache.set(row.id,{version:row.detail_version,data});return data;
}
function renderDetail(data,row){
 text('runTitle',titleFor(row));text('runStatus',data.status_text);el('detailPanel').hidden=false;
 el('controls').max=data.maximum_control_steps||600;el('controls').value=data.control_steps||0;
 el('runLinks').replaceChildren();for(const [name,path] of Object.entries(data.links||{}))link(el('runLinks'),name,path);
 el('frame').dataset.path=data.frame||'';el('overlay').dataset.path=data.contact_overlay||'';
 image('frame',null);image('overlay',null);
 el('skill').replaceChildren();table(el('skill'),Object.entries(data.skill||{}));
 el('contactHistory').replaceChildren();table(el('contactHistory'),[
  ['Event / tool','Outcome','Total','Scene prep','Contact search','Stroke plan','Execute + capture','Feedback','Reused','Builds / searches'],
  ...(data.contact_calls||[]).map(c=>{const t=c.timing_seconds||{};return [c.event+' / '+c.operation,
   c.stop_reason,num(c.wall_seconds),num(t.perception_seconds||0),num(t.contact_search_seconds||0),
   num(t.stroke_planning_seconds||0),num(t.execution_and_capture_seconds||0),
   num((t.tracker_initialization_seconds||0)+(t.motion_tracking_seconds||0)+(t.moving_geometry_update_seconds||0)),
   [c.contact_reused?'contact':null,c.scene_reused?'scene':null,c.preview_reused?'preview':null].filter(Boolean).join(', ')||'none',
   (c.full_scene_builds??'—')+' / '+(c.contact_searches??'—')];})]);
 const timing=data.timing||{};text('timeScope',timing.outside_service_scope||'Only completed requests have timings.');
 el('times').replaceChildren();table(el('times'),[['Actor total (s)',num(timing.actor_wall_seconds)],['Robot tools, inclusive (s)',num(timing.service_wall_seconds)],['Outside tools: Luna + transport + local analysis (s)',num(timing.outside_service_seconds)]]);
 table(el('times'),Object.entries(timing.service_seconds_by_operation||{}).map(([name,value])=>[name,num(value)+' s']));
 el('nested').replaceChildren();for(const [name,values] of Object.entries(timing)){
  if(!name.startsWith('nested_')||!values||typeof values!=='object')continue;
  el('nested').append(node('h4',name.replaceAll('_',' ')));table(el('nested'),Object.entries(values).map(([k,v])=>[k.replaceAll('_',' '),num(v)+' s']));
 }
 el('issues').replaceChildren();for(const issue of data.issues||[]){
  const d=node('details');d.dataset.openKey='issue:'+issue.event+':'+issue.operation;
  d.append(node('summary','Event '+issue.event+' · '+issue.operation+' · '+issue.reason));
  if(issue.explanation)d.append(node('p',issue.explanation));
  if(issue.evaluator_addendum){d.append(node('p','Evaluator diagnosis: '+issue.evaluator_addendum.summary));
   if(issue.evaluator_addendum.report)link(d,'Repair test details',issue.evaluator_addendum.report);}
  if(issue.diagnostic)d.append(node('pre',JSON.stringify(issue.diagnostic,null,2)));el('issues').append(d);
 }
 restoring=true;for(const d of el('detailPanel').querySelectorAll('details[data-open-key]'))d.open=!!saved.open?.[chosen]?.[d.dataset.openKey];restoring=false;
 showImageWhenOpen('frameDetails','frame');showImageWhenOpen('contactDetails','overlay');
 selectedVersion=row.detail_version;text('updateNote','');
}
async function inspectRun(id,move=false,force=false){
 if(detailBusy)return;detailBusy=true;save();pauseDisplay();
 const row=(state?.runs||[]).find(r=>r.id===id);if(!row){detailBusy=false;return}
 const y=scrollY;
 try{const data=await fetchDetail(row,force);chosen=id;renderDetail(data,row);save();
  if(move)el('detailPanel').scrollIntoView({behavior:'instant',block:'start'});else scrollTo(0,y);
 }catch(error){text('updateNote',error.message)}finally{detailBusy=false}
}
function refreshResults(background=false){
 if(background&&!el('live').checked)return Promise.resolve();
 if(overviewRequest)return overviewRequest;
 overviewRequest=refreshOverview().finally(()=>{overviewRequest=null});
 return overviewRequest;
}
async function refreshOverview(){
 const y=scrollY,anchor=document.elementFromPoint(24,80)?.closest('section,article');
 const anchorTop=anchor?.getBoundingClientRect().top;
 try{const r=await fetch('watch-status.json',{cache:'no-cache'});if(!r.ok)throw Error('No saved status yet');updateOverview(await r.json());
  // Compatibility for a report published before video lists were added to the index.
  for(const row of state.runs||[])if(!Array.isArray(row.videos)){const data=await fetchDetail(row);for(const video of data.videos||[])addVideo(row,video)}
  text('videoCount','('+cards.size+')');if(cards.size)text('videoEmpty','');
  // Never replace an inspection or a player during a status refresh.
  if(chosen&&selectedVersion===null)await inspectRun(chosen,false);
  if(anchor&&document.contains(anchor))scrollBy(0,anchor.getBoundingClientRect().top-anchorTop);else scrollTo(0,y);
 }catch(error){text('status','Open this file with VS Code Live Preview. '+error.message)}
}
el('refresh').onclick=()=>refreshResults(false);
el('updateRun').onclick=async()=>{
 const id=chosen;if(!id||detailBusy)return;
 el('updateRun').disabled=true;
 try{await refreshResults(false);if(chosen===id)await inspectRun(id,false,true)}
 finally{el('updateRun').disabled=false}
};
el('live').onchange=()=>{text('displayMode',el('live').checked?'Live overview is on. Use Refresh this trial to update its details.':'Manual display updates. The experiment continues independently.');if(el('live').checked)refreshResults(true)};
el('detailPanel').addEventListener('toggle',event=>{
 if(restoring)return;if(event.target.open)pauseDisplay('Display updates paused for inspection. Experiments continue independently.');
 showImageWhenOpen('frameDetails','frame');showImageWhenOpen('contactDetails','overlay');save();
},true);
window.addEventListener('pagehide',save);
let scrollSave;window.addEventListener('scroll',()=>{clearTimeout(scrollSave);scrollSave=setTimeout(save,200)},{passive:true});
setInterval(()=>{if(!document.hidden&&el('live').checked)refreshResults(true)},60000);
refreshResults(false).then(()=>{if(initialScroll)scrollTo(0,initialScroll)});
