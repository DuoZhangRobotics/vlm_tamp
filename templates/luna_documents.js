'use strict';
class LunaDocumentReader {
 constructor(root,{onOpen=()=>{}}={}){
  this.root=root;this.onOpen=onOpen;this.request=0;this.records=[];
  this.title=root.querySelector('[data-document-title]');this.status=root.querySelector('[data-document-status]');
  this.content=root.querySelector('[data-document-content]');this.select=root.querySelector('[data-document-record]');
  this.navigation=root.querySelector('[data-document-navigation]');
  root.querySelector('[data-document-close]').onclick=()=>this.close();
  root.querySelector('[data-document-refresh]').onclick=()=>this.open(this.label,this.path,this.trigger);
  root.querySelector('[data-document-previous]').onclick=()=>this.show(this.select.selectedIndex-1);
  root.querySelector('[data-document-next]').onclick=()=>this.show(this.select.selectedIndex+1);
  this.select.onchange=()=>this.show(this.select.selectedIndex);
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!root.hidden)this.close()});
 }
 static parse(path,raw){
  if(/\.jsonl(?:[?#]|$)/i.test(path)){
   const records=raw.split(/\r?\n/).filter(line=>line.trim()).map((line,i)=>{
    try{return JSON.parse(line)}catch(error){throw Error('Invalid JSON on log line '+(i+1)+': '+error.message)}
   }).filter(row=>row.item?.type!=='reasoning');
   return {records,text:null};
  }
  return {records:null,text:/\.json(?:[?#]|$)/i.test(path)?JSON.stringify(JSON.parse(raw),null,2):raw};
 }
 static label(row,index){
  if(row.request?.operation)return 'Event '+(row.sequence??index+1)+' · '+row.request.operation;
  const item=row.item||{};
  return [index+1,row.type,item.type,item.command?.slice(0,90)].filter(Boolean).join(' · ');
 }
 async open(label,path,trigger=null){
  this.label=label;this.path=path;this.trigger=trigger;const request=++this.request;
  this.root.hidden=false;this.title.textContent=label;this.status.textContent='Loading '+path+'…';
  this.status.setAttribute('role','status');this.content.textContent='';this.navigation.hidden=true;this.onOpen();
  this.root.querySelector('[data-document-close]').focus({preventScroll:true});
  try{
   const url=new URL(path,document.baseURI);
   if(url.origin!==location.origin)throw Error('This reader only opens files from this report server.');
   const response=await fetch(url,{cache:'no-store'});
   if(!response.ok)throw Error('HTTP '+response.status+' while reading '+path);
   const raw=await response.text();if(request!==this.request)return;
   const parsed=LunaDocumentReader.parse(path,raw);
   if(parsed.records!==null){
    this.records=parsed.records;this.select.replaceChildren();
    this.records.forEach((row,i)=>{const option=document.createElement('option');option.textContent=LunaDocumentReader.label(row,i);this.select.append(option)});
    this.navigation.hidden=!this.records.length;
    this.status.textContent=this.records.length+' records · public messages and tool I/O · select a record to inspect';
    const first=this.records.findIndex(r=>r.request||(r.type==='item.completed'&&r.item?.type==='command_execution'));this.show(Math.max(0,first));
    if(!this.records.length)this.content.textContent='No records have been saved yet.';
   }else{
    this.records=[];this.content.textContent=parsed.text;this.status.textContent=path+' · '+raw.length.toLocaleString()+' characters';
   }
  }catch(error){
   if(request!==this.request)return;
   this.status.setAttribute('role','alert');this.status.textContent='Could not open this document: '+error.message;
   this.content.textContent='Use Reload document to retry. The report and its videos remain open.';
  }
 }
 show(index){
  if(!this.records.length)return;index=Math.max(0,Math.min(index,this.records.length-1));
  this.select.selectedIndex=index;this.content.textContent=JSON.stringify(this.records[index],null,2);
  this.content.scrollTop=0;
 }
 close(){this.request++;this.root.hidden=true;this.trigger?.focus({preventScroll:true})}
}
if(typeof module!=='undefined')module.exports=LunaDocumentReader;
