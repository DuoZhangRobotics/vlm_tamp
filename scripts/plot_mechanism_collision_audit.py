"""Scientific illustration of the old partial-cloud convex-envelope rejection."""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/mcvamp-drawer-stove-report';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
records=[]
for ax,label,title in zip(axes,['open_top_drawer','turn_on_stove'],['Top drawer: initial reconstruction','Stove: initial reconstruction']):
    folder=ROOT/'results/all-prior-tasks-frozen-v3-20261015'/f'{label}-state001'
    with np.load(folder/'initial-scene.npz') as a:
        moving=a['moving_reference'];stationary=a['static_points'];contact=a['contact_reference']
    hull=ConvexHull(moving);low=moving.min(0)-.008;high=moving.max(0)+.008
    nearby=stationary[((stationary>=low)&(stationary<=high)).all(1)]
    inside=np.max(nearby@hull.equations[:,:3].T+hull.equations[:,3],axis=1)<=0
    normal=np.asarray(json.loads((folder/'task-spec.json').read_text())['panel_normal_world'])
    center=moving.mean(0)
    if label=='turn_on_stove':u=np.array([1.,0,0]);v=np.array([0,1.,0])
    else:u=normal;v=np.array([0.,0,1.])
    project=lambda points:np.column_stack([(points-center)@u,(points-center)@v])*1000
    xy=project(moving);boundary=ConvexHull(xy)
    ax.fill(xy[boundary.vertices,0],xy[boundary.vertices,1],color='#cae5fa',alpha=.5,label='Projected convex completion')
    stride=max(1,len(xy)//1800);ax.scatter(xy[::stride,0],xy[::stride,1],s=3,color='#247ba0',label='Observed moving surface')
    points=project(nearby[inside]);ax.scatter(points[:,0],points[:,1],s=12,color='#d64045',label='Stationary points inside 3-D hull',zorder=4)
    ax.set_title(title);ax.set_xlabel('Plane coordinate (mm)' if label=='turn_on_stove' else 'Depth along face normal (mm)')
    ax.set_ylabel('Plane coordinate (mm)' if label=='turn_on_stove' else 'Height (mm)')
    ax.set_aspect('equal');ax.grid(alpha=.18)
    records.append({'task':label,'state':1,'strictly_inside_stationary_points':int(inside.sum()),
        'interpretation':'Reconstruction/classification overlap; not proof of physical collision',
        'initial_cloud_source':str(folder/'initial-scene.npz')})
axes[0].legend(loc='best',fontsize=8)
fig.suptitle('Old collision model rejected an overlapping reconstruction before any manipulation',fontsize=13)
fig.savefig(out/'collision-model-audit.png',dpi=170);plt.close(fig)
(out/'collision-model-audit.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records))
