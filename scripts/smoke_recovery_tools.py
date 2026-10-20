"""Separate scripted physical interface check; excluded from policy success rates."""
import hashlib
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from run_strategy_feedback import Resources,TASKS
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.strategy_episode import run_episode
from vlm_tamp.recovery_decision import parse_decision
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.run_api_budget import api_run_scope,ledger_for_output
import json
import random
import numpy as np

OUT=ROOT/'results/vlm-recovery-tool-smoke-20261018'

class ScriptedInterfaceCheck:
    calls=0
    def choose(self,packet,image,folder,scene_id):
        n=len(packet['history'])
        values=[('try_contact','press_drag',None,'probe',5,None,None),
                ('release_reposition',None,None,None,None,None,20),
                ('try_contact','press_drag',None,'probe',5,1,None),
                ('finish',None,None,None,None,None,None)]
        op,mode,tilt,extent,step,rank,retreat=values[min(n,3)]
        d=dict(version='strategy-v2',operation=op,contact_mode=mode,approach_tilt_deg=tilt,
            motion_extent=extent,step_mm=step,posture_rank=rank,retreat_mm=retreat,
            evidence='Predeclared interface smoke sequence; not a VLM decision.',
            expected_observation='The requested tool returns accurate execution and planning evidence.')
        write_json(Path(folder)/'decision.json',d);return parse_decision(json.dumps(d))

def main():
    OUT.mkdir(exist_ok=False)
    config=read_json(ROOT/'results/vlm-recovery-frozen-20261018/protocol.json')['configuration']['open_bottom_drawer']
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name in hashes:
        dest=OUT/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write_json(OUT/'protocol.json',{'sources_sha256':hashes,'scripted_interface_check':True,
        'policy_success_rate_denominator':False,'maximum_decision_rounds':4,'configuration':config})
    write_json(OUT/'manifest.json',{'episodes':[{'task':'open_bottom_drawer','state':15,'policy':'scripted'}]})
    ledger=ledger_for_output(OUT,'vlm-recovery-tools-20261018')
    resources=Resources(OUT,Timeline());resources.allow_paid=False
    with api_run_scope(ledger.run_id):
        resources.warm_models()
        import torch
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
        folder=OUT/'open_bottom_drawer-state015-scripted'
        result=run_episode(TASKS['open_bottom_drawer'],15,folder,config,resources,ScriptedInterfaceCheck(),maximum_rounds=4)
        compress_retained_observations(folder);result=read_json(folder/'summary.json')
        result.update(strategy_policy='scripted',scripted_interface_check=True)
        write_json(folder/'summary.json',result)
        resources.cleanup_case();resources.case_cache.rmdir()
    print(json.dumps(result))

if __name__=='__main__':main()
