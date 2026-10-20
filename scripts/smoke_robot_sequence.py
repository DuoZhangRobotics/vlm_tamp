"""Physical smoke test of the public prepared-sequence RPC on matched RGB input."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from serve_subscription_robot import RobotService
from replay_subscription_episode import compress_observations
from vlm_tamp.coupled_storage import read_json,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--degrees',type=float,default=90.)
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'results'):raise ValueError('Use a new result directory')
    out.mkdir();work=out/'temporary-workspace';work.mkdir()
    source=ROOT/'results/subscription-success-first-continuation-20261019'
    events=[json.loads(x) for x in (source/'tool-events.jsonl').read_text().splitlines()]
    regions=next(e['request']['arguments']['regions'] for e in events if e['request']['operation']=='ground')
    service=RobotService(out,work,47,1,stop_on_success=True);error=None
    def call(op,**arguments):
        try:
            result=service.dispatch(op,arguments);response={'ok':True,'result':result}
            return result
        except Exception as exc:
            response={'ok':False,'error':str(exc)};raise
        finally:
            with (out/'tool-events.jsonl').open('a') as f:
                f.write(json.dumps({'sequence':service.calls,'request':{'operation':op,'arguments':arguments},
                    'response':response},default=lambda x:x.tolist())+'\n')
    try:
        call('reset')
        with np.load(out/'attempt01/opened/obs.npz') as a,np.load(source/'attempt01/opened/obs.npz') as b:
            same=all(k in b and np.array_equal(a[k],b[k],equal_nan=True) for k in a.files)
        if not same:raise ValueError('Current observations differ; cached RGB regions cannot be reused')
        write_json(out/'protocol.json',{'initial_arrays_identical':True,'source_regions':str(source),
            'source_grasp_or_trajectory_replayed':False,'source_coordinates_given_to_agent':False,
            'planning_rpc':'plan_sequence','execution_rpc':'execute_sequence',
            'degrees':args.degrees,'max_control_steps':600,'api_calls':0})
        call('ground',regions=regions)
        plan=call('plan_sequence',degrees=args.degrees)
        result=call('execute_sequence',plan_id=plan['plan_id'])
        write_json(out/'sequence-result.json',result)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        (out/'error.txt').write_text(traceback.format_exc())
    finally:
        service.finish_attempt('sequence_smoke_failure' if error else 'sequence_smoke_complete')
        shutil.rmtree(work)
        compression=compress_observations(out)
        write_json(out/'cleanup.json',{'temporary_workspace_removed':True,'compression':compression})
    summary=read_json(out/'attempt01/summary.json')
    write_json(out/'summary.json',{'native_success':summary['official_task_success'],
        'steps':summary['control_steps'],'error':error,'episode':summary})
    print(json.dumps({'native_success':summary['official_task_success'],'steps':summary['control_steps'],'error':error}),flush=True)


if __name__=='__main__':main()
