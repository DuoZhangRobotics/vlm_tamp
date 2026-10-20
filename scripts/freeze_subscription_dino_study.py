"""Freeze the optional-DINO availability study before reserved-state exposure."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    reservation=ROOT/'results/dino-agent-state-reservation-v2-20261019.json'
    record=json.loads(reservation.read_text());states=record['selected_states']
    shutil.copyfile(reservation,out/'state-reservation.json')
    source=ROOT/'results/subscription-agent-fit-diagnostics-20261019/agent-workspace/guidelines.json'
    original=json.loads(source.read_text());reviews=[
        'Supplied method; not an independent discovery.',
        'Observed in one development case; timing of reference selection was not isolated in a controlled ablation.',
        'Applying measured updates was supplied in the example; the quoted fit statistics are observed case evidence.',
        'Planning outcomes are supported by the trace; this does not establish a general guarantee.',
        'Tracking failed and the agent used a separate image/depth analysis; applicable evidence is limited to one case.',
        'Approximate plane/width analysis was performed; the stated uncertainty was self-assigned, not calibrated.'
    ]
    public={'source':'Agent-written guidelines from one earlier development case',
        'use':'Treat these as conditional hypotheses. Case-specific numerical values are observations, not mandatory settings. Evaluate them against current evidence.',
        'guidelines':[{'id':i+1,'origin_reported_by_agent':g['origin'],'condition':g['condition'],
            'guideline':g['guideline'],'evidence_review':reviews[i]} for i,g in enumerate(original['guidelines'])]}
    (out/'guidelines.json').write_text(json.dumps(public,indent=2)+'\n')
    (out/'guideline-provenance-private.json').write_text(json.dumps({'source':str(source),
        'source_sha256':sha(source),'original_guidelines':original,
        'public_guideline_text_is_verbatim':True,'removed_fields':'Old case file paths only; no old trajectory, pixel coordinates or analysis implementation is exposed.'},indent=2))
    prompt=(ROOT/'configs/prompts/subscription_agent_goal_v1.txt').read_text()
    prompt=prompt.replace('Your only supplied recovery guidance is the explicit example.',
        'Your supplied recovery guidance is the explicit example and the frozen agent-written hypotheses in experience/guidelines.json. Read both; evaluate the hypotheses against current evidence.')
    assert 'experience/guidelines.json' in prompt
    prompt+='\nUse available tools when useful. Optional tools do not have to be invoked. Do not access other cases or modify shared experience during this evaluation.\n'
    (out/'prompt.txt').write_text(prompt)
    cases=[]
    for number,state in enumerate(states):
        order=['no_dino','optional_dino'] if number%2==0 else ['optional_dino','no_dino']
        for condition in order:cases.append({'case_id':f'state{state:03d}-{condition}','state':state,'condition':condition})
    (out/'manifest.json').write_text(json.dumps({'cases':cases,'planned_cases':10,
        'planned_cases_per_condition':5,'maximum_physical_attempts':20,'all_planned_cases_in_denominator':True},indent=2))
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))+list((ROOT/'scripts').glob('*.py'))+[ROOT/'configs/articulated_libero.json']
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
    (out/'frozen-source-hashes.json').write_text(json.dumps(hashes,indent=2))
    checkpoint=ROOT/'.runtime/hf-cache/models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32/model.safetensors'
    pins={'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'model':'facebook/dinov3-vits16-pretrain-lvd1689m',
          'revision':'114c1379950215c8b35dfcd4e90a5c251dde0d32'}
    runtime_files=[ROOT/'.runtime/mcvamp-microwave/libmcvamp_microwave.so',
        ROOT/'.runtime/mcvamp-microwave/build.json',
        ROOT/'.runtime/graspgen-models/checkpoints/graspgen_franka_panda.yml',
        ROOT/'.runtime/graspgen-models/checkpoints/graspgen_franka_panda_gen.pth',
        ROOT/'.runtime/graspgen-models/checkpoints/graspgen_franka_panda_dis.pth']
    runtime_pins={str(p):sha(p) for p in runtime_files}
    versions=subprocess.check_output([str(ROOT/'.runtime/envs/dinov3/bin/python'),'-c',
        'import json,torch,numpy,scipy,transformers;print(json.dumps({"torch":torch.__version__,"numpy":numpy.__version__,"scipy":scipy.__version__,"transformers":transformers.__version__}))'],text=True)
    protocol={'model':'gpt-6-luna','reasoning_effort':'high','authentication':'ChatGPT subscription',
        'separate_api_calls_allowed':False,'states':states,'simulator_seed':0,'settling_actions':5,
        'maximum_control_steps_per_attempt':600,'maximum_attempts_per_case':2,'agent_seconds_per_case':1800,
        'initial_gripper_open_steps_included_in_budget':60,'controller':'existing joint position servo',
        'execution_vetoes':False,'current_conversation_inherited':False,'source_code_access':False,
        'native_object_truth_sent_to_agent':False,'shared_experience_frozen':True,
        'prompt_sha256':sha(out/'prompt.txt'),'guidelines_sha256':sha(out/'guidelines.json'),
        'dino':pins,'dino_runtime':json.loads(versions),'runtime_artifacts_sha256':runtime_pins,
        'smoke_result':str(ROOT/'results/dino-agent-tool-smoke-20261019/summary.json'),
        'primary_metrics':['first-attempt native success / 5','native case success within two attempts / 5'],
        'other_metrics':['DINO invocation count','total agent wall time including tool communication','tool and DINO latency','all physical outcomes'],
        'limits':'One task and asset, five reserved initial states. Model sampling and inherited numerical proposal sampling can vary. This tests optional tool availability, not DINOv3 versus DINOv2 quality or cross-task generalization.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'study':str(out),'cases':cases,'source_files_frozen':len(hashes)},indent=2))


if __name__=='__main__':main()
