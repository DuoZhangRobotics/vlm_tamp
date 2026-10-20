"""Endpoint collisions must not exhaust the feasible IK proposal quota."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vlm_tamp import hinged_contact_tools as runner


def test_rejected_endpoints_do_not_hide_later_feasible_ik(monkeypatch):
    model=SimpleNamespace(project=lambda *a,**k:{'accepted':False,'position_error_m':1.})
    monkeypatch.setattr(runner,'constraint_model',lambda *a,**k:model)
    def poses(*args,**kwargs):
        for index in range(6):
            target=np.eye(4);target[0,3]=index
            yield target,{'index':index}
    monkeypatch.setattr(runner,'goal_pose_menu',poses)
    chain=SimpleNamespace(forward=lambda q:np.r_[np.zeros(3),[0,0,0,1]])
    ik=SimpleNamespace(chain=chain,solve_many=lambda pose,*a,**k:[{'joints':np.full(7,pose[0])}])
    context=SimpleNamespace(pad_local=np.zeros(3),record={'aligned_jaws':False},
        project=lambda q:(q,{'accepted':True}),progress=lambda q:np.pi/2,
        validate=lambda q:{'accepted':bool(q[0]>=4)})
    goals,records=runner.goal_candidates(ik,context,{'joint_axis_world':[0,0,1]},np.zeros(7),np.pi/2,maximum=1)
    assert len(goals)==1 and goals[0][0]==4
    admissions=[r for r in records if r['source']=='endpoint_admission']
    assert len(admissions)==5 and sum(r['accepted'] for r in admissions)==1
