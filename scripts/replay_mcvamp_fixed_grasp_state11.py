"""Paired diagnostic: replay the frozen RGB-D-derived grasp, then plan afresh.

Only the initial approach/grasp is replayed, after image, proprioception and
planning-scene checks. All postclosure paths come from the corrected MC-VAMP
adapter. This is a development diagnostic, not another independent benchmark.
"""
import argparse
from pathlib import Path
import hashlib
import shutil

import numpy as np
from PIL import Image

import run_mcvamp_microwave as runner
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.mcvamp_microwave import NativeScene

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/mcvamp-microwave-frozen-20261015/open_microwave-state011'


def replay_grasp(generated,anchors,state,ik,record,open_scene,closed_scene,spec,goal,folder,config):
    folder.mkdir()
    original=read_json(SOURCE/'summary.json')
    horizon=original['selected_initial_horizon_rad']
    if abs(goal-horizon)>1e-7:
        raise ValueError('Paired replay retains the original 20-degree segment; it does not claim a 90-degree plan')
    before=read_json(SOURCE/'opened/metadata.json')
    difference=float(np.max(np.abs(np.asarray(before['joint_positions'])-state['metadata']['joint_positions'])))
    if difference>1e-7:raise ValueError('Initial proprioception differs from the frozen grasp episode')
    images={}
    cameras,_=runner.read_observation_v3(folder.parent/'opened',state['metadata'])
    for name in ['agentview','frontview']:
        expected=np.asarray(Image.open(SOURCE/'opened'/f'{name}.png'))
        current=np.asarray(cameras[name]['rgb'])
        images[name]=float(np.abs(expected.astype(float)-current).max())
    if any(images.values()):raise ValueError('Initial RGB differs from the frozen grasp episode')
    cloud=np.load(SOURCE/'handle-cloud.npy')
    if not np.array_equal(cloud,np.load(folder.parent/'handle-cloud.npy')):
        raise ValueError('Current sensor-derived handle cloud differs from the frozen grasp')
    selected=read_json(SOURCE/'chosen-grasp.json')
    approach=np.load(SOURCE/'selection-bounded-segment/approach.npy')
    contact=np.load(SOURCE/'pregrasp-complete-path.npy')
    q=np.asarray(selected['joints'])
    with NativeScene(open_scene,record,state['metadata'],approach[0],goal,articulated=False) as context:
        approach_checks=[context.validate(row)['accepted'] for row in approach]
    predicted=runner.virtual_metadata(state,closed_scene.robot,ik.chain,q)
    with NativeScene(closed_scene,record,predicted,q,goal,aligned=False) as context:
        contact_checks=[context.validate(row)['accepted'] for row in contact]
    evidence={'diagnostic_replay':True,'source':str(SOURCE),'same_rgb':images,
        'maximum_initial_joint_difference_rad':difference,'same_predicted_handle_cloud':True,
        'approach_states':len(approach),'contact_states':len(contact),
        'all_approach_states_admitted_before_execution':all(approach_checks),
        'all_contact_states_admitted_before_execution':all(contact_checks),
        'physical_grasp_only':True,'postclosure_plans_recomputed':True,
        'execution_vetoes':False,'source_approach_sha256':hashlib.sha256((SOURCE/'selection-bounded-segment/approach.npy').read_bytes()).hexdigest()}
    write_json(folder/'replay-evidence.json',evidence)
    if not all(approach_checks+contact_checks):raise ValueError('Frozen initial path no longer passes current planning scene')
    selected.update(approach=approach,planned_contact=contact,search_seconds=0.)
    np.save(folder/'approach.npy',approach);np.save(folder/'contact.npy',contact)
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/mcvamp-convex-frozen-grasp-state11-v2-20261015')
    args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(exist_ok=False)
    source_names=['scripts/replay_mcvamp_fixed_grasp_state11.py','scripts/run_mcvamp_microwave.py',
        'src/vlm_tamp/mcvamp_microwave.py','src/vlm_tamp/native/mcvamp_microwave.cc',
        'src/vlm_tamp/timed_execution.py','scripts/build_mcvamp_microwave.py']
    sources={name:runner.digest(ROOT/name) for name in source_names}
    for name in source_names:
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    baseline=read_json(runner.BASE/'frozen-evaluation/protocol.json')['sources_sha256']
    config=read_json(ROOT/'configs/articulated_libero.json')
    config.update(complete_planar_panel=False,scene_voxel_spacing_m=.003,
        maximum_grasp_poses=24,maximum_grasp_search_seconds=240.,maximum_contact_plan_seconds=8.,full_task_only=False)
    protocol={'sources_sha256':sources,'baseline_frozen_sources_sha256':baseline,'config':config,
        'seed':0,'initial_states':[11],'maximum_control_steps':600,'frozen_before_first_episode':True,
        'diagnostic_replay':True,'frozen_grasp_source':str(SOURCE),'new_api_calls_allowed':False,
        'postclosure_plans_recomputed':True,'execution_vetoes':False,
        'native_goal_is_evaluation_only':True,'physical_grasp_only':True,'not_independent_benchmark':True}
    write_json(output/'protocol.json',protocol)
    ledger=ROOT/'results/articulated-api-v1/ledger.json';api_hash=runner.digest(ledger)
    runner.select_grasp=replay_grasp
    task=next(t for t in config['tasks'] if t['label']=='open_microwave')
    result=runner.episode(task,11,output/'open_microwave-state011',config,runner.CoupledResources())
    write_json(output/'summary.json',{'planned':1,'completed':1,'episodes':[result],
        'api_ledger_unchanged':runner.digest(ledger)==api_hash,'diagnostic_replay':True,
        'baseline_sources_unchanged':all(runner.digest(ROOT/n)==h for n,h in baseline.items()),
        'sources_unchanged':all(runner.digest(ROOT/n)==h for n,h in sources.items())})
    print(result,flush=True)


if __name__=='__main__':main()
