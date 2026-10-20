"""Freeze original versus agent-consolidated guidance on reserved scene cases."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.subscription_scene_variants import VARIANTS
from run_subscription_dino_study import read,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=(ROOT/a.output).resolve();out.mkdir(exist_ok=False)
    prior=ROOT/'results/subscription-dino-study-20261019'
    experience=ROOT/'results/subscription-strategy-consolidation-v2-20261019'
    assert read(experience/'completion.json')['completed']
    assert read(experience/'audit-validation.json')['evidence_references_exist']
    reservation=ROOT/'results/strategy-transfer-state-reservation-20261019.json'
    states=read(reservation)['selected_states'];assert len(states)==5
    validation=ROOT/'results/subscription-scene-validation-v2-20261019/summary.json'
    assert len(read(validation)['records'])==6
    shutil.copyfile(reservation,out/'state-reservation.json');shutil.copyfile(validation,out/'scene-validation.json')
    shutil.copyfile(prior/'guidelines.json',out/'original-guidelines.json')
    shutil.copyfile(experience/'agent-workspace/guidelines.json',out/'consolidated-guidelines.json')
    shutil.copyfile(experience/'agent-workspace/audit-review.json',out/'consolidation-audit.json')
    shutil.copyfile(prior/'prompt.txt',out/'prompt.txt')
    shutil.copyfile(validation.parent/'scene-sheet.png',out/'development-scene-sheet.png')
    scenarios=[]
    for i,state in enumerate(states):scenarios.append({'scenario_id':f'nominal-state{state:03d}','group':'nominal','state':state,'scene_variant':'nominal'})
    for state,variant in zip(states,[x for x in VARIANTS if x!='nominal']):
        scenarios.append({'scenario_id':f'{variant}-state{state:03d}','group':'changed_scene','state':state,'scene_variant':variant})
    cases=[]
    for i,s in enumerate(scenarios):
        for condition in (['original','consolidated'] if i%2==0 else ['consolidated','original']):
            cases.append({**s,'condition':condition,'case_id':s['scenario_id']+'-'+condition})
    manifest={'task':'KITCHEN_SCENE7_open_the_microwave','scenarios':scenarios,'cases':cases,
        'planned_cases':20,'cases_per_condition':10,'maximum_physical_attempts':40,'all_planned_cases_in_denominator':True}
    write(out/'manifest.json',manifest)
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))+list((ROOT/'scripts').glob('*.py'))+[ROOT/'configs/articulated_libero.json']
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files};write(out/'frozen-source-hashes.json',hashes)
    old=read(prior/'frozen-source-hashes.json')
    changed=[f for f,h in old.items() if hashes.get(f)!=h]
    allowed={'src/vlm_tamp/subscription_agent.py','src/vlm_tamp/rekep_libero/sim_worker.py',
        'src/vlm_tamp/rekep_libero/sim_worker_articulated.py','scripts/serve_subscription_robot.py',
        'scripts/run_subscription_agent_pilot.py','scripts/report_subscription_pilot.py'}
    if set(changed)-allowed:raise ValueError('Unexpected prior-source changes: '+str(set(changed)-allowed))
    inherited=read(prior/'protocol.json')
    protocol={k:copy.deepcopy(inherited[k]) for k in ['model','reasoning_effort','authentication',
        'separate_api_calls_allowed','simulator_seed','settling_actions','maximum_control_steps_per_attempt',
        'maximum_attempts_per_case','agent_seconds_per_case','initial_gripper_open_steps_included_in_budget',
        'controller','execution_vetoes','current_conversation_inherited','source_code_access',
        'native_object_truth_sent_to_agent','dino','dino_runtime','runtime_artifacts_sha256']}
    protocol.update(task=manifest['task'],states=states,scene_variants=VARIANTS,
        paired_conditions=['original','consolidated'],optional_dino_tools_in_both_conditions=True,
        shared_experience_frozen=True,prompt_sha256=sha(out/'prompt.txt'),
        guidelines_sha256={c:sha(out/f'{c}-guidelines.json') for c in ['original','consolidated']},
        guide_sources={'original':str(prior/'guidelines.json'),'consolidated':str(experience/'agent-workspace/guidelines.json')},
        consolidation_source=str(experience),prior_source_changes=changed,
        planning_grasp_ik_control_implementations_unchanged=True,
        nominal_rgb_identical_to_prior_study=True,
        primary_metrics=['first-attempt native success / 10','native success within two attempts / 10'],
        strata=['nominal: 5 scenarios per condition','changed_scene: 5 scenarios per condition'],
        diagnostic_metrics=['planning failures','tracking rejection counts','model updates','re-groundings','reset count','initial hinge XY error, private evaluator only','DINO calls','case and nested tool wall time'],
        limits='Same task and microwave asset; five new initial states and five predeclared variants. Variants are custom LIBERO scene perturbations, not official benchmark episodes. No new mechanism/asset generalization claim. Model and numerical proposals can vary. Single run per matched scenario/condition; descriptive comparison only.')
    write(out/'protocol.json',protocol)
    print(json.dumps({'study':str(out),'planned_cases':20,'states':states,'changed_prior_sources':changed},indent=2))
if __name__=='__main__':main()
