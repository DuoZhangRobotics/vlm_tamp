"""Offline native-angle plot; no values from this script enter actor/planner input."""
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

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--study',default=str(ROOT/'results/restriction-fixes-20261029'))
p.add_argument('--final-tag');args=p.parse_args()
STUDY=Path(args.study).resolve()
sources=[]
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,(task,state) in zip(axes,[('close_microwave',17),('close_microwave',49),('open_microwave',13)]):
    runs=[(ROOT/'results/contact-skills-20261028/trials'/f'advisory-v4-{task}-state{state:03d}',
        'Earlier policy / 600','#8c939d')]
    runs += [(STUDY/'trials'/f'cleaned-v1-{task}-state{state:03d}-{budget}',
        f'Before follow-up / {budget}',color) for budget,color in [(600,'#b85835'),(1200,'#176aa7')]]
    if args.final_tag:runs.append((STUDY/'trials'/f'{args.final_tag}-{task}-state{state:03d}-1200','Repaired + practiced / 1200','#268351'))
    for run,label,color in runs:
        if not (run/'evaluation-private.json').exists():continue
        file=run/'attempt01/initial/evaluation_contact_diagnostics.json'
        actual=file if file.exists() else file.with_suffix('.json.gz')
        if not actual.exists():continue
        records=read_json(file)['steps']
        q=[(r['step'],max(0.,-math.degrees(next(j['position'] for j in r['evaluation_only_articulation_state']
            if j['joint_name']=='microwave_1_microjoint')))) for r in records if r['step']>=60]
        ax.plot([x[0] for x in q],[x[1] for x in q],label=label,color=color)
        actions=read_json(run/'attempt01/initial/actions.json')
        first=next((a['step'] for a in actions if a['evaluation_task_success']),None)
        if first is not None and first in dict(q):ax.scatter(first,dict(q)[first],color=color,s=110,marker='*',zorder=5)
        sources.append({'run':str(run),'label':label,'file':str(actual),'sha256':hashlib.sha256(actual.read_bytes()).hexdigest()})
    threshold=math.degrees(1.3 if task=='open_microwave' else .005)
    ax.axhline(threshold,color='#48535f',linestyle=':',linewidth=1,label=f'Native threshold: {threshold:.2f}°')
    ax.set(title=f'{task.replace("_"," ")} / state {state}',xlabel='Control step',ylabel='Door opening (degrees)')
    ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Offline task progress • stars mark first native success',fontsize=14)
fig.savefig(STUDY/'task-progress-comparison.png',dpi=180)
fig.savefig(STUDY/'task-progress-comparison.pdf')
(STUDY/'task-progress-comparison-sources.json').write_text(json.dumps({'offline_grading_only':True,
    'no_native_values_sent_to_actor_or_planner':True,'separate_rollouts_not_fixed_action_counterfactuals':True,
    'threshold_source':'third_party/LIBERO/libero/libero/envs/objects/articulated_objects.py:Microwave.is_open/is_close',
    'opening_angle_definition':'max(0, -degrees(native microwave joint position))',
    'sources':sources},indent=2)+'\n')
print('Plotted',len(sources),'completed rollouts.')
