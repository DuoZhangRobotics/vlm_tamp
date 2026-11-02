"""The report reader keeps prompts and structured public logs inspectable."""
import json
from pathlib import Path
import subprocess


def test_document_reader_preserves_prompt_and_public_records():
    root=Path(__file__).resolve().parents[1]
    script="""
const Reader=require('./templates/luna_documents.js');
const prompt='Close the microwave.\\nRead the observations.\\n';
if(Reader.parse('agent-prompt.txt',prompt).text!==prompt)throw Error('Prompt changed');
const rows=[{sequence:1,request:{operation:'reset'},response:{ok:true}},
 {type:'item.completed',item:{type:'reasoning',text:'private'}},
 {type:'item.completed',item:{type:'command_execution',command:'python3 robot.py push',aggregated_output:'<literal text>'}}];
const parsed=Reader.parse('tool-events.jsonl',rows.map(JSON.stringify).join('\\n')+'\\n');
if(parsed.records.length!==2)throw Error('Public record selection failed');
if(parsed.records[1].item.aggregated_output!=='<literal text>')throw Error('Literal output changed');
let visibleError=false;
try{Reader.parse('broken.jsonl','{}\\ninvalid');}catch(error){visibleError=error.message.includes('line 2');}
if(!visibleError)throw Error('Parse failure was hidden');
console.log(JSON.stringify({passed:true,records:parsed.records.length}));
"""
    result=subprocess.run(['node','-e',script],cwd=root,text=True,capture_output=True,check=True)
    assert json.loads(result.stdout)=={'passed':True,'records':2}
