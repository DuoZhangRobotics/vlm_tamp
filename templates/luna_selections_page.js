'use strict';
const trial=document.getElementById('trial'),status=document.getElementById('pageStatus');
const viewer=new LunaSelectionViewer(document.getElementById('viewer'));
let runs=[],chosen=new URLSearchParams(location.search).get('run');
async function selectRun(){
 const row=runs.find(r=>r.id===trial.value);if(!row)return;
 try{const response=await fetch(row.detail,{cache:'no-cache'});if(!response.ok)throw Error('Trial data unavailable');
  const data=await response.json();chosen=row.id;viewer.setData(data.selection_history||[],row.id);viewer.setVisible(true);
  document.getElementById('backToWatch').href='watch.html?run='+encodeURIComponent(row.id);
  status.textContent=row.label+' · '+(data.selection_history||[]).length+' recorded selections/direction requests.';
  const event=new URLSearchParams(location.search).get('event');
  if(event){const item=viewer.records.find(r=>String(r.event)===event);if(item)viewer.select(item.id)}
  history.replaceState(null,'','?run='+encodeURIComponent(row.id));
 }catch(error){status.textContent='Open this page with VS Code Live Preview. '+error.message}
}
async function refresh(){
 try{const response=await fetch('watch-status.json',{cache:'no-cache'});if(!response.ok)throw Error('Study data unavailable');
  const data=await response.json();runs=data.runs||[];trial.replaceChildren();
  for(const row of runs){const option=document.createElement('option');option.value=row.id;option.textContent=row.label;trial.append(option)}
  if(runs.some(r=>r.id===chosen))trial.value=chosen;await selectRun();
 }catch(error){status.textContent='Open this page with VS Code Live Preview. '+error.message}
}
trial.onchange=selectRun;document.getElementById('reload').onclick=refresh;refresh();
