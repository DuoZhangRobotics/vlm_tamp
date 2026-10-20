"""Choose previously unused microwave states from retained experiment metadata."""
from pathlib import Path
import json
import random
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def reserve(output):
    paths=subprocess.check_output(['rg','--files','results','data','-g','summary.json','-g','protocol.json',
        '-g','manifest.json','-g','metadata.json','-g','!**/source-snapshot/**','-g','!**/agent-workspace/**'],cwd=ROOT,text=True).splitlines()
    evidence={}
    def add(i,path):
        if isinstance(i,int) and not isinstance(i,bool) and 0<=i<50:evidence.setdefault(i,set()).add(path)
    def ids(text,path):
        for n in re.findall(r'state[-_]?0*(\d+)',text):add(int(n),path)
    def walk(value,path,context=False):
        if isinstance(value,dict):
            tag=' '.join(str(value.get(k,'')) for k in ['task','task_name','label','name','instruction','tasks'])
            own='open_microwave' in tag or 'open_the_microwave' in tag or 'open the microwave' in tag.lower()
            ctx=context or own
            if ctx:
                ids(path,path)
                for k in ['id','example_id','case_id']:ids(str(value.get(k,'')),path)
                for k in ['state','state_index','initial_state','initial_state_index','development_state']:add(value.get(k),path)
                for k in ['states','initial_states','state_indices']:
                    v=value.get(k,[])
                    if isinstance(v,list):
                        for i in v:add(i,path)
            for k,v in value.items():
                if k not in ['sources_sha256','files_sha256','protected_artifacts','source_hashes','requests']:
                    walk(v,path,ctx)
        elif isinstance(value,list):
            for item in value:walk(item,path,context)
    for name in paths:
        path=ROOT/name
        if path.stat().st_size>8_000_000:continue
        try:data=json.loads(path.read_text())
        except (ValueError,OSError):continue
        walk(data,name,'open_microwave' in name or 'open_the_microwave' in name)
    unused=[i for i in range(50) if i not in evidence]
    if len(unused)<5:raise ValueError('Fewer than five unused states in retained metadata')
    selected=sorted(random.Random(0).sample(unused,5))
    record={'task':'KITCHEN_SCENE7_open_the_microwave','searched_metadata_files':len(paths),
        'excluded_used_or_previously_planned_states':sorted(evidence),'unused_candidates':unused,
        'selection_seed':0,'selected_states':selected,
        'scope':'No prior use found in retained metadata. Historical deleted or unrecorded runs cannot be ruled out.',
        'evidence':{str(i):sorted(v)[:20] for i,v in evidence.items()}}
    Path(output).write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    record=reserve(args.output)
    print(json.dumps({k:v for k,v in record.items() if k!='evidence'},indent=2))
