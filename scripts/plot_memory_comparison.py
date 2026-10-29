"""Offline task progress for the declared clean-practice memory comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.coupled_storage import read_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--study',default=str(ROOT/'results/clean-start-600-20261029'))
parser.add_argument('--tag',default='clean-v1')
args=parser.parse_args()
STUDY=Path(args.study).resolve()
manifest=read_json(STUDY/(args.tag+'-practice-manifest.json'))
cases=manifest['plan']['later_evaluation_cases']
fig,axes=plt.subplots(1,len(cases),figsize=(4.6*len(cases),4.1),layout='constrained',squeeze=False)
sources=[]
for ax,case in zip(axes[0],cases):
    for condition,label,color in [('none','No inherited memory','#858c97'),
                                  ('frozen','Frozen practice notes','#16846d')]:
        rows=[row for row in manifest.get('evaluations',[]) if row['task']==case['task']
              and row['state_index']==case['state_index'] and row['memory_condition']==condition]
        if not rows:continue
        row=rows[0];run=STUDY/'trials'/row['run']
        path=run/'attempt01/initial/evaluation_contact_diagnostics.json'
        actual=path if path.exists() else path.with_suffix('.json.gz')
        values=[]
        for frame in read_json(path)['steps']:
            if frame['step']<60:continue
            joint=next(j for j in frame['evaluation_only_articulation_state']
                if j['joint_name']=='microwave_1_microjoint')
            values.append((frame['step'],max(0.,-math.degrees(joint['position']))))
        ax.plot([v[0] for v in values],[v[1] for v in values],label=label,color=color)
        first=row.get('first_native_success_control_step')
        if first is not None and first in dict(values):
            ax.scatter(first,dict(values)[first],color=color,marker='*',s=120,zorder=5)
        sources.append({'run':row['run'],'condition':condition,'native_success':row['official_task_success'],
            'path':str(actual),'sha256':hashlib.sha256(actual.read_bytes()).hexdigest()})
    threshold=math.degrees(1.3 if case['task']=='open_microwave' else .005)
    ax.axhline(threshold,linestyle=':',color='#48535f',linewidth=1,label=f'Native threshold {threshold:.2f}°')
    ax.set(title=f"{case['task'].replace('_',' ')} / {case['state_index']}",xlabel='Control step',
           ylabel='Door opening (degrees)',xlim=(50,610))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Fresh agents at 600 controls: frozen practice notes versus no memory\nOffline grading; stars mark first native success',fontsize=13)
fig.savefig(STUDY/'task-progress-comparison.png',dpi=180)
fig.savefig(STUDY/'task-progress-comparison.pdf')
(STUDY/'task-progress-comparison-sources.json').write_text(json.dumps({
    'offline_grading_only':True,'native_values_sent_to_actor_or_planner':False,
    'one_independent_actor_per_case_and_condition':True,'sources':sources},indent=2)+'\n')
print('Plotted',len(sources),'completed evaluations.')
