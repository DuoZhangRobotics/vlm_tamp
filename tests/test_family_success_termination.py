from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_success_first_family_regression as experiment
from vlm_tamp.reliable_mechanisms import contact_probe_progress


def test_family_success_preserves_full_requested_and_shorter_issued_segment(tmp_path,monkeypatch):
    calls=[]
    sim=SimpleNamespace(folder=tmp_path,segments=0,observe=lambda path:calls.append(str(path)))
    refs=np.arange(35,dtype=float).reshape(5,7)
    def execute(sim,state,references,**kwargs):
        assert kwargs['stop_on_success']
        return {'control_steps':12},{'issued_commands':2,'terminated_on_simulator_success':True,
            'events':[{'control_step':11},{'control_step':12}]}
    monkeypatch.setattr(experiment,'execute_batch',execute)
    with pytest.raises(experiment.NativeSuccess,match='simulator_success'):
        experiment.execute_to_success(sim,{'control_steps':10},refs)
    np.testing.assert_array_equal(np.load(tmp_path/'execution-segments/000/requested.npy'),refs)
    np.testing.assert_array_equal(np.load(tmp_path/'execution-segments/000/issued.npy'),refs[:2])
    assert calls==[str(tmp_path/'final')]
    assert (tmp_path/'success-termination.json').exists()


def test_full_task_probe_uses_the_grounded_extent_for_both_mechanism_families():
    for kind,extent in [('prismatic',.18),('revolute',1.2)]:
        spec={'joint_family':kind,'target_progress':extent}
        assert contact_probe_progress(spec,{'grasp_probe_full_task':True})==extent
    assert contact_probe_progress({'joint_family':'prismatic'}, {})==.025
    assert contact_probe_progress({'joint_family':'prismatic'}, {}, {'step_mm':40})==.04


def test_adaptive_grasp_search_preserves_short_fallback_without_motion(tmp_path,monkeypatch):
    import vlm_tamp.reliable_mechanisms as module
    calls=[]
    def select(*args,**kwargs):
        config=args[-1];calls.append(config['grasp_probe_full_task'])
        if config['grasp_probe_full_task']:raise module.NoFeasibleGrasp('No full goal')
        return {'index':2},np.zeros((3,7))
    monkeypatch.setattr(module,'_select_grasp_once',select)
    chosen,path=module.select_grasp(None,None,{'joint_family':'prismatic'},None,None,None,None,None,
        tmp_path,{'adaptive_grasp_horizon':True,'maximum_grasp_search_seconds':10.})
    assert calls==[True,False]
    assert chosen['horizon_support']=='short_probe_only'
    assert path.shape==(3,7)
