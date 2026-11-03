'use strict';
/* Exact numerical overlays on unchanged saved images; no image inference. */
(()=>{
let serial=0;
const ns='http://www.w3.org/2000/svg';
function make(tag,attrs={},value){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(value!==undefined)n.textContent=value;return n}
function lineInImage(pair,w,h){
 if(!pair||pair.length!==2)return null;
 const [a,b]=pair,dx=b[0]-a[0],dy=b[1]-a[1],hits=[];
 for(const x of [0,w])if(Math.abs(dx)>1e-9){const t=(x-a[0])/dx,y=a[1]+t*dy;if(y>=0&&y<=h)hits.push([x,y,t])}
 for(const y of [0,h])if(Math.abs(dy)>1e-9){const t=(y-a[1])/dy,x=a[0]+t*dx;if(x>=0&&x<=w)hits.push([x,y,t])}
 hits.sort((a,b)=>a[2]-b[2]);return hits.length>1?[hits[0].slice(0,2),hits.at(-1).slice(0,2)]:null;
}
class Viewer{
 constructor(root,{onInteract=()=>{}}={}){
  this.root=root;this.onInteract=onInteract;this.records=[];this.visible=false;this.uid='selection-'+(++serial);
  root.innerHTML=`<p class="selection-controls"><button data-role="prev">Previous</button> <select data-role="event" aria-label="Luna selection event"></select> <button data-role="next">Next</button></p>
   <p class="selection-controls"><label>Zoom <select data-role="zoom"><option value="1">100%</option><option value="1.5">150%</option><option value="2">200%</option><option value="3">300%</option></select></label>
   <label><input type="checkbox" data-role="annotations" checked> Luna annotations</label>
   <label><input type="checkbox" data-role="direction" checked> Direction</label>
   <label><input type="checkbox" data-role="model" checked> Fitted hinge</label></p>
   <p class="selection-controls"><label><input type="checkbox" data-role="mask"> SAM mask</label> <select data-role="maskChoice" aria-label="SAM mask"></select>
   <label><input type="checkbox" data-role="reference"> Evaluator hinge (operator only)</label></p>
   <p data-role="caption"></p><p data-role="sourceCoords"></p><p data-role="outcome" class="muted"></p>
   <p data-role="legend" class="muted">Yellow cross: exact Luna contact pixel. Purple H1/H2: Luna hinge annotation. Cyan box: Luna region. Green box: tool mask bounds. S+/S−: Luna’s SAM clicks. Blue cross: projected 3D reference, not a click.</p>
   <div class="selection-viewport"><div class="selection-canvas"><img data-role="image" alt="Original observation"><svg data-role="svg" aria-label="Recorded pixel annotations"></svg></div></div>
   <p data-role="coordinates" class="muted"></p><p data-role="directionText"></p><p data-role="modelSource"></p><p data-role="notes" class="muted"></p>
   <p><a data-role="original" target="_blank" rel="noopener">Open original image</a> · <a data-role="maskLink" target="_blank" rel="noopener" hidden>Open mask image</a> · <a data-role="resultImage" target="_blank" rel="noopener" hidden>Open unannotated result image</a></p>`;
  this.get=role=>root.querySelector(`[data-role="${role}"]`);
  this.get('event').onchange=()=>{this.onInteract();this.select(this.get('event').value)};
  for(const [role,delta] of [['prev',-1],['next',1]])this.get(role).onclick=()=>{
   const index=this.records.findIndex(r=>r.id===this.selected);const row=this.records[index+delta];
   if(row){this.onInteract();this.select(row.id)}
  };
  for(const role of ['zoom','annotations','direction','model','mask','maskChoice','reference'])this.get(role).onchange=()=>{this.onInteract();this.persist();this.draw()};
  this.get('image').onerror=()=>{this.get('notes').textContent='The exact saved image could not be loaded. No substitute capture is displayed.'};
  this.get('image').onload=()=>this.draw();
  this.get('svg').parentElement.onmousemove=event=>{
   const row=this.current();if(!row?.image_size)return;
   const rect=this.get('image').getBoundingClientRect();
   if(!rect.width||!rect.height)return;
   const x=(event.clientX-rect.left)*row.image_size[0]/rect.width,y=(event.clientY-rect.top)*row.image_size[1]/rect.height;
   if(x>=0&&y>=0&&x<row.image_size[0]&&y<row.image_size[1])this.get('coordinates').textContent=`Cursor in original image: (${x.toFixed(1)}, ${y.toFixed(1)}). Origin top-left; x right, y down.`;
  };
 }
 current(){return this.records.find(r=>r.id===this.selected)}
 setData(records,run){
  const changed=this.run!==run;this.run=run;this.records=records||[];
  this.storage='luna-selection-view:'+location.pathname+':'+run;
  if(changed){
   let saved={};try{saved=JSON.parse(sessionStorage.getItem(this.storage)||'{}')}catch{}
   this.selected=saved.id;
   for(const role of ['annotations','direction','model','mask','reference'])this.get(role).checked=saved[role]??['annotations','direction','model'].includes(role);
   this.get('zoom').value=saved.zoom||'1';
  }
  if(!this.current())this.selected=(this.records.find(r=>r.contact_pixel_xy&&['push','hook_pull'].includes(r.operation)&&!r.preview)||this.records.find(r=>r.contact_pixel_xy)||this.records[0])?.id;
  const select=this.get('event');select.replaceChildren();
  for(const r of this.records){const option=document.createElement('option');option.value=r.id;option.textContent=r.summary;select.append(option)}
  select.value=this.selected||'';
  this.refreshMasks();if(this.visible)this.draw();
 }
 setVisible(visible){this.visible=visible;if(visible)this.draw()}
 select(id){this.selected=id;this.get('event').value=id;this.refreshMasks();this.persist();this.draw()}
 refreshMasks(){
  const select=this.get('maskChoice'),previous=select.value;select.replaceChildren();
  for(const m of this.current()?.masks||[]){
   const option=document.createElement('option');option.value=m.part_id;option.textContent=m.role+' · '+m.part_id+(m.same_capture?'':' · different capture');
   option.disabled=!m.same_capture||!m.image;select.append(option);
  }
  const valid=[...select.options].find(o=>o.value===previous&&!o.disabled)||[...select.options].find(o=>!o.disabled);
  if(valid)select.value=valid.value;
  this.get('mask').disabled=!valid;select.disabled=!valid;
  this.get('reference').disabled=!this.current()?.evaluator_axis_line;
 }
 persist(){
  if(!this.storage)return;const data={id:this.selected,zoom:this.get('zoom').value};
  for(const role of ['annotations','direction','model','mask','reference'])data[role]=this.get(role).checked;
  try{sessionStorage.setItem(this.storage,JSON.stringify(data))}catch{}
 }
 draw(){
  if(!this.visible)return;const r=this.current(),svg=this.get('svg'),picture=this.get('image');svg.replaceChildren();
  if(!r){this.get('caption').textContent='No pixel selections have been recorded.';picture.removeAttribute('src');return}
  const [w,h]=r.image_size||[512,512],mask=(r.masks||[]).find(m=>m.part_id===this.get('maskChoice').value);
  const path=this.get('mask').checked&&!this.get('mask').disabled&&mask?.same_capture?mask.image:r.image;
  picture.hidden=!path;if(path&&picture.getAttribute('src')!==path)picture.src=path;if(!path)picture.removeAttribute('src');
  picture.parentElement.style.width=(w*Number(this.get('zoom').value))+'px';
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.hidden=!path||!picture.complete||!picture.naturalWidth;
  const index=this.records.indexOf(r);this.get('prev').disabled=index<=0;this.get('next').disabled=index>=this.records.length-1;
  this.get('caption').textContent=`Event ${r.event} · ${r.operation}${r.preview?' · preview only':''} · ${r.camera||'camera unavailable'} · capture ${r.observation_id??'?'} · control ${r.control_step??'?'} · ${r.image_size?w+' × '+h+' original pixels':'source image unavailable'}`;
  const outcomes={preview_only:'Preview only; no robot motion',approach_not_reached:'Approach attempted; goal not reached; push was not issued',push_reference_not_reached:'Physical stroke attempted; requested robot goal was not reached',stroke_executed:'Hand stroke executed; object/task progress is separate',control_budget_exhausted:'Episode control budget exhausted'};
  let outcome=outcomes[r.outcome]||r.outcome;
  if(typeof r.arrival?.goal_error_m==='number')outcome+=' · final robot goal error '+(r.arrival.goal_error_m*1000).toFixed(1)+' mm';
  if(r.result_control_step!==undefined&&r.result_control_step!==null)outcome+=' · result at control '+r.result_control_step;
  this.get('outcome').textContent=(r.ok===false?'Request rejected: ':'Outcome: ')+outcome+(r.contact_name?' · robot contact: '+r.contact_name:'');
  const selected=[];if(r.contact_pixel_xy)selected.push('Contact P: ('+r.contact_pixel_xy.join(', ')+')');
  if(r.hinge_pixels_xy)r.hinge_pixels_xy.forEach((p,i)=>selected.push('H'+(i+1)+': ('+p.join(', ')+')'));
  for(const [i,p] of (r.segmentation_points||[]).entries())selected.push('S'+(i+1)+(p.include===0?'−':p.include===1?'+':'?')+': ('+p.xy.join(', ')+')');
  this.get('sourceCoords').textContent=selected.length?'Luna’s exact input pixels — '+selected.join(' · '):'No contact/hinge pixel coordinates were supplied in this call.';
  this.get('coordinates').textContent=r.contact_pixel_xy?`Luna contact pixel: (${r.contact_pixel_xy.join(', ')}). Coordinates are shown exactly as supplied.`:'No contact pixel was supplied in this call.';
  const source=this.get('modelSource');source.replaceChildren();
  if(r.model_reference){
   source.append(document.createTextNode(`Requested joint model epoch ${r.joint_model_epoch}: `));
   const button=document.createElement('button');button.textContent=`Grounding event ${r.model_reference.event} · ${r.model_reference.camera}`;
   button.onclick=()=>{this.onInteract();this.select(r.model_reference.id)};source.append(button);
  }else if(r.geometry_epoch!==undefined){
   source.textContent='This request produced model epoch '+r.geometry_epoch+'. It remains a hypothesis. ';
   for(const used of r.referenced_by||[]){const button=document.createElement('button');button.textContent='Requested by event '+used.event+(used.ok===false?' (rejected)':used.preview?' (preview)':'');button.onclick=()=>{this.onInteract();this.select(used.id)};source.append(button)}
  }
  else if(r.direction_world)source.textContent='World-direction request; no hinge model is required by this operation.';
  const notes=[...(r.notes||[])];
  if(this.get('reference').checked&&r.evaluator_axis_line)notes.push(r.evaluator_reference_scope);
  if(this.get('mask').checked&&!this.get('mask').disabled&&mask)notes.push('Showing the saved SAM output '+mask.part_id+'. Luna selected the part ID; SAM generated the mask.');
  this.get('notes').textContent=notes.join(' ');
  const original=this.get('original');original.hidden=!r.image;if(r.image)original.href=r.image;
  const maskLink=this.get('maskLink');maskLink.hidden=!mask?.image;if(mask?.image)maskLink.href=mask.image;
  const after=this.get('resultImage');after.hidden=!r.result_image;if(r.result_image)after.href=r.result_image;
  const defs=make('defs'),marker=make('marker',{id:this.uid+'-arrow',viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});
  marker.append(make('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#ef4444'}));defs.append(marker);svg.append(defs);
  const line=(a,b,color,width=2,dash=null)=>{const n=make('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:color,'stroke-width':width});if(dash)n.setAttribute('stroke-dasharray',dash);svg.append(n);return n};
  const label=(p,value,color)=>svg.append(make('text',{x:p[0]+(p[0]>.72*w?-10:10),y:Math.max(18,p[1]-10),'text-anchor':p[0]>.72*w?'end':'start',fill:color,stroke:'#172b43','stroke-width':3,'paint-order':'stroke','font-size':14,'font-family':'system-ui'},value));
  const cross=(p,color,name)=>{svg.append(make('circle',{cx:p[0],cy:p[1],r:6,fill:'none',stroke:color,'stroke-width':2}));line([p[0]-10,p[1]],[p[0]+10,p[1]],color);line([p[0],p[1]-10],[p[0],p[1]+10],color);label(p,name,color)};
  if(this.get('annotations').checked){
   for(const box of r.regions||[]){const [a,b,c,d]=box.box_xyxy;svg.append(make('rect',{x:a,y:b,width:Math.max(0,c-a),height:Math.max(0,d-b),fill:'none',stroke:box.source==='Luna request'?'#22d3ee':'#4ade80','stroke-width':2,'stroke-dasharray':box.source==='Luna request'?'':'4 3'}))}
   if(r.contact_pixel_xy)cross(r.contact_pixel_xy,'#fde047',`P (${r.contact_pixel_xy.join(',')})`);
   else if(r.projected_world_point_xy)cross(r.projected_world_point_xy,'#60a5fa','Projected reference');
   if(r.hinge_pixels_xy){line(...r.hinge_pixels_xy,'#e879f9',3);r.hinge_pixels_xy.forEach((p,i)=>cross(p,'#e879f9','H'+(i+1)))}
   for(const [i,p] of (r.segmentation_points||[]).entries())cross(p.xy,p.include===0?'#fb7185':p.include===1?'#4ade80':'#cbd5e1','S'+(i+1)+(p.include===0?'−':p.include===1?'+':'?'));
  }
  if(this.get('model').checked&&r.model_axis_line){const pair=lineInImage(r.model_axis_line,w,h);if(pair)line(...pair,'#a855f7',2,'8 5')}
  if(this.get('reference').checked&&r.evaluator_axis_line){const pair=lineInImage(r.evaluator_axis_line,w,h);if(pair){line(...pair,'#111827',4,'8 5');line(...pair,'#ffffff',2,'8 5')}}
  let direction='';
  if(r.direction_world)direction=`${r.direction_source}: [${r.direction_world.map(x=>Number(x.toFixed(4))).join(', ')}]. `+(r.direction_arrow?'Red arrow: projection of a 5 cm direction guide, not measured motion.':'No world-to-image direction projection is available.');;
  if(typeof r.requested_distance_m==='number')direction+=' Requested hand travel: '+(1000*r.requested_distance_m).toFixed(1)+' mm.';
  if(r.requested_joint_degrees!==undefined&&r.requested_joint_degrees!==null)direction+=' Requested joint increment: '+r.requested_joint_degrees+'°.';
  if(r.requested_rotation_direction)direction+=' Luna requested '+r.requested_rotation_direction.toUpperCase()+'. '+(r.rotation_view_convention||'');
  if(r.contact_effect)direction+=' '+r.contact_effect.explanation;
  if(this.get('direction').checked){
   for(const arrow of [r.direction_arrow,r.image_direction_hint])if(arrow){const n=line(arrow.start,arrow.end,'#ef4444',3);n.setAttribute('marker-end',`url(#${this.uid}-arrow)`)}
  }
  if(r.image_direction_hint)direction+=' Red image arrow: Luna’s supplied 2D direction hint (schematic length).';
  this.get('directionText').textContent=direction;
 }
}
window.LunaSelectionViewer=Viewer;
})();
