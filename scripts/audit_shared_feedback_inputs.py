"""Check whether paired variants actually shared visual and grasp inputs."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=[]
    for name in a.runs:
        root=ROOT/name;manifest=read_json(root/'manifest.json')['episodes']
        for key in dict.fromkeys((r['task'],r['state']) for r in manifest):
            entries=[r for r in manifest if (r['task'],r['state'])==key]
            folders=[root/f'{r["task"]}-state{r["state"]:03d}-{r["variant"]}' for r in entries]
            if len(folders)<2 or not all((f/'summary.json').exists() and 'controller_variant' in read_json(f/'summary.json') for f in folders):continue
            checks={}
            for filename in ['opened/agentview.png','opened/frontview.png','task-spec.json','grasps/input-cloud.npy']:
                present=[(f/filename).exists() for f in folders]
                checks[filename]={'present_in_all':all(present),'identical':len({sha(f/filename) for f in folders if (f/filename).exists()})<=1}
            grasp_files=[f/'grasps/grasp_candidates.npz' for f in folders]
            if all(f.exists() for f in grasp_files):
                arrays=[dict(np.load(f)) for f in grasp_files]
                checks['grasp_candidates_arrays']={'present_in_all':True,
                    'identical':all(set(a)==set(arrays[0]) and all(np.array_equal(a[k],arrays[0][k]) for k in a) for a in arrays)}
            refs=[list((f/'execution-segments').glob('*-approach/nominal.npy')) for f in folders]
            if all(len(x)==1 for x in refs):
                arrays=[np.load(x[0]) for x in refs]
                checks['planned_initial_approach']={'present_in_all':True,'identical':all(np.array_equal(x,arrays[0]) for x in arrays)}
            rows.append({'run':name,'task':key[0],'state':key[1],'variants':[e['variant'] for e in entries],
                         'checks':checks,'all_available_inputs_identical':all(c['identical'] for c in checks.values())})
    write_json(ROOT/a.output,{'paired_cases_checked':len(rows),'cases':rows,
        'all_available_paired_inputs_identical':all(r['all_available_inputs_identical'] for r in rows)})
    print({'paired_cases_checked':len(rows),'all_available_paired_inputs_identical':all(r['all_available_inputs_identical'] for r in rows)})

if __name__=='__main__':main()
