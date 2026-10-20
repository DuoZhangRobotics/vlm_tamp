"""Transfer fixed drawer/rotary-control policies with no demonstrations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
from vlm_tamp import reliable_mechanisms as engine
from vlm_tamp.family_transfer import TASKS,STATES,BASELINE,TransferResources,frozen_configuration,manifest,core_audit
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_resources import api_snapshot
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.profiled_sim_client import execute_batch
from run_success_pipeline import SceneBoundaryClient
from run_interaction_pipeline import prior_case_evidence


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--states',nargs='+',type=int,default=STATES)
    parser.add_argument('--tasks',nargs='+',choices=[t['label'] for t in TASKS])
    parser.add_argument('--no-controls',action='store_true');args=parser.parse_args()
    tasks=[t for t in TASKS if not args.tasks or t['label'] in args.tasks]
    planned=manifest(tasks,args.states,controls=not args.no_controls);configuration=frozen_configuration()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output must be inside project results')
    if args.resume:
        protocol=read_json(out/'protocol.json');sources=protocol['sources_sha256']
        if read_json(out/'manifest.json')['episodes']!=planned or protocol['configuration']!=configuration:
            raise ValueError('Resume requires the same manifest and configuration')
    else:
        previous=prior_case_evidence(tasks,args.states)
        if previous:raise ValueError('Transfer cases already have physical outcome summaries: '+str(previous))
        audit=core_audit();out.mkdir(parents=True,exist_ok=False)
        before=api_snapshot();maximum_scenes=max(before['max_scene_requests'],before['api_attempts']+len(planned))
        files=list((ROOT/'src/vlm_tamp').rglob('*.py'))
        files += [Path(__file__),ROOT/'scripts/run_success_pipeline.py',ROOT/'scripts/run_all_prior_tasks.py',ROOT/'scripts/run_interaction_pipeline.py',
            ROOT/'scripts/reliable_microwave_final.py',ROOT/'scripts/try_rekep_bowl_v3.py',ROOT/'configs/articulated_libero.json',
            ROOT/'src/vlm_tamp/native/mcvamp_mechanism.cc']
        sources={str(p.relative_to(ROOT)):sha(p) for p in files}
        for name in sources:
            target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        write_json(out/'manifest.json',{'episodes':planned,'every_planned_case_in_denominator':True,
            'controls_excluded_from_transfer_score':True,'seed':0,'settling_actions':5,'maximum_control_steps':600})
        protocol={'sources_sha256':sources,'configuration':configuration,'configuration_source':str(BASELINE/'protocol.json'),
            'frozen_geometry_planner_and_controller':audit,'unchanged_configuration_for_all_families_and_cases':True,
            'api_before':before,'maximum_scene_requests':maximum_scenes,'original_compact_vlm_prompt_unchanged':True,
            'input_policy':'RGB-D only; VLM chooses part and primitive; family geometry code measures numerical values',
            'controls':'original tasks, state 1, observation-validated baseline VLM cache',
            'transfer':'original and related tasks, unused physical-evaluation states; fresh vision and compact VLM query',
            'vision_compatibility':'bounded finite DINO clustering; unchanged weights; at most five candidates per mask and 6cm merge',
            'coordinate_handling':'existing unambiguous collage-to-camera translation; no manual point edits',
            'provided_inventory':False,'simulator_object_truth_used_for_policy':False,'demonstration_video_used':False,
            'demonstration_trajectories_used':False,'neural_weights_finetuned':False,'manual_family_priors_retained':True,
            'held_out_scope':'case/layout transfer within existing LIBERO asset types, not novel appliance designs',
            'execution_vetoes':False,'task_success':'native LIBERO predicate; all failures remain in denominator',
            'automatic_api_retries':0,'model':'gpt-6-luna','reasoning_effort':'high','max_output_tokens':12000,
            'no_tuning_after_first_case':True,'future_work':['one demonstration video for hinge identification','RL tracking/controller adaptation; excluded from this experiment']}
        write_json(out/'protocol.json',protocol)
    if not all(sha(ROOT/n)==h for n,h in sources.items()):raise ValueError('Frozen sources changed')
    timeline=Timeline();resources=TransferResources(out,timeline,maximum_scenes=protocol['maximum_scene_requests'])
    resources.warm_models();engine.execute=execute_batch
    engine.CoupledClient=lambda *a,**k:SceneBoundaryClient(*a,**k,fast=True,timeline=timeline)
    # Restore the successful contact scene explicitly; no shared-interface
    # contact substitutions or arbitrary-shape fitting are used in this test.
    from vlm_tamp.reliability_geometry import SurfaceScene
    engine.SurfaceScene=SurfaceScene
    rows=[]
    for entry in planned:
        if not all(sha(ROOT/n)==h for n,h in sources.items()):raise ValueError('Frozen sources changed during cohort')
        task=next(t for t in tasks if t['label']==entry['task']);resources.phase=entry['phase'];timeline.context=entry
        folder=out/f'{entry["task"]}-state{entry["state"]:03d}'
        if (folder/'summary.json').exists():result=read_json(folder/'summary.json')
        else:
            if folder.exists():raise ValueError('Interrupted episode requires diagnosis; never silently rerun')
            print(json.dumps({'event':'started',**entry}),flush=True)
            result=engine.run_episode(task,entry['state'],folder,configuration,resources)
            resources.cleanup_proposals();compress_retained_observations(folder);result=read_json(folder/'summary.json')
            result.update(experiment_phase=entry['phase'],family=task['family'],demonstrations_used=False,
                frozen_family_configuration=True,scene_transfer_group=task['group'])
            write_json(folder/'summary.json',result)
        rows.append(result);timeline.save(out/'timing.json')
        groups={}
        for phase in ['reproduction_control','original_tasks','related_tasks']:
            group=[r for r in rows if r['experiment_phase']==phase]
            groups[phase]={'planned':sum(e['phase']==phase for e in planned),'completed':len(group),
                'native_successes':sum(r['official_task_success'] for r in group)}
        write_json(out/'summary.json',{'planned':len(planned),'completed':len(rows),'groups':groups,'episodes':rows,'api':api_snapshot()})
        print(json.dumps({'event':'completed',**entry,'success':result['official_task_success'],
            'failure':result.get('failure'),'steps':result['control_steps']}),flush=True)
    resources.cleanup_proposals()


if __name__=='__main__':main()
