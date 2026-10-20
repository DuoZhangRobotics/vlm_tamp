"""Replay saved baseline planning queries using the new MC-VAMP scene adapter.

This is a component benchmark with different collision approximations, not a
controlled reproduction of MC-VAMP's published SIMD projection speedups.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_scene import Scene
from vlm_tamp.mcvamp_microwave import NativeScene,clear_native_cache
from run_mcvamp_microwave import baseline_folder


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for task in ['open_microwave','close_microwave']:
        source=baseline_folder(task,1);record=read_json(source/'robot-chain.json')
        for folder in sorted(source.glob('plan-*')):
            if not (folder/'path.npy').exists():continue
            before=time.perf_counter()
            inputs=read_json(folder/'inputs.json');path=np.load(folder/'path.npy')
            reference=next(folder.rglob('scene-reference.json'))
            arrays=np.load(read_json(reference)['arrays_path'])
            scene=Scene.from_points(arrays['static_points'],arrays['moving_reference'],arrays['contact_reference'],inputs['task_spec'])
            with NativeScene(scene,record,inputs['metadata'],path[0,:7],path[-1,7],
                    aligned=inputs['contact_mode']=='aligned') as context:
                solved,result=context.plan(path[0,:7],path[-1,:7],seconds=3.,iterations=1000)
                row={'task':task,'source_plan':str(folder.relative_to(ROOT)),
                    'source_planner_result':read_json(folder/'result.json'),'mc_vamp':result,
                    'capt_build_seconds':context.record['capt_build_seconds'],
                    'total_wall_seconds_including_inputs':time.perf_counter()-before}
                if result['exact_solution']:
                    np.save(output/f'{task}-{folder.name}-path.npy',solved)
                rows.append(row);write_json(output/'queries.json',rows)
                print(task,folder.name,result['status'],result.get('total_seconds'),flush=True)
            clear_native_cache()
    valid=[r['mc_vamp'] for r in rows if r['mc_vamp']['exact_solution']]
    write_json(output/'summary.json',{'queries':len(rows),'solved':len(valid),'physical_execution':False,
        'median_native_search_seconds':float(np.median([r['solve_seconds'] for r in valid])) if valid else None,
        'median_planner_with_resampling_and_worker_seconds':float(np.median([r['total_seconds'] for r in valid])) if valid else None,
        'median_capt_build_seconds':float(np.median([r['capt_build_seconds'] for r in rows])),
        'different_collision_model_and_projection':True,'new_api_calls':0,
        'retained':'small numeric paths, per-query outcomes and timing; no duplicate sensor archives'})

if __name__=='__main__':main()
