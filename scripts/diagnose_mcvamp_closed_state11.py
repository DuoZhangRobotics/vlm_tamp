"""Replay the exact rejected closed-gripper state, directly targeting 90 degrees."""
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_scene import Scene
from vlm_tamp.mcvamp_microwave import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.task_ik_curobo import TaskIK
from run_mcvamp_microwave import contact_plan


def main():
    output=ROOT/'results/mcvamp-convex-closed-state11-full90-20261015'
    output.mkdir(exist_ok=False)
    source=ROOT/'results/mcvamp-microwave-frozen-20261015/open_microwave-state011'
    folder=source/'plan-00/horizon-0'
    inputs=read_json(folder/'planning-input.json');reference=read_json(folder/'scene-reference.json')
    arrays=np.load(reference['arrays_path']);spec=inputs['task_spec'];record=read_json(source/'robot-chain.json')
    scene=Scene.from_points(arrays['static_points'],arrays['moving_reference'],arrays['contact_reference'],spec)
    q=np.asarray(inputs['reference_joints']);metadata=inputs['metadata'];goal=np.pi/2
    config=read_json(ROOT/'configs/articulated_libero.json')
    import vamp
    environment=vamp.Environment();ik=None;start=time.perf_counter()
    result={'source':str(source),'goal_angle_deg':90.,'physical_execution':False,'bounded_fallback':False}
    try:
        with NativeScene(scene,record,metadata,q,goal) as context:result['start_check']=context.validate(q)
        ik=TaskIK(record,config,lambda x:vamp.panda.validate(np.asarray(x).tolist(),environment,True),output=output)
        path,info=contact_plan(ik,scene,record,metadata,spec,q,goal,output/'full90',seconds=8.)
        angles=np.load(output/'full90/door-progress.npy')
        result.update(status='full90_path_found',path_states=len(path),terminal_progress_deg=float(np.degrees(angles[-1])),planning=info)
    except Exception as error:
        result.update(status='no_full90_path_found',failure=str(error))
        (output/'error.txt').write_text(traceback.format_exc())
    finally:
        if ik:ik.close()
        clear_native_cache()
        result['wall_seconds']=time.perf_counter()-start
        write_json(output/'summary.json',result)
        print(result,flush=True)


if __name__=='__main__':main()
