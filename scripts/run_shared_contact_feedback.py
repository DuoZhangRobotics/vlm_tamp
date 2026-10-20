"""Paired shared-contact / feedback evaluation; frozen sources remain intact."""
import argparse
import contextlib
import hashlib
import json
import random
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.runtime/mc-vamp-deps')]
import numpy as np
from vlm_tamp import reliable_mechanisms as mechanisms, reliable_bowl_fallback as bowl
from vlm_tamp import shared_mechanism_pipeline as shared_mechanisms
import reliable_microwave_final as microwave
from run_success_pipeline import task_configuration
from run_all_prior_tasks import TASKS as ORIGINAL_TASKS
from vlm_tamp.family_transfer import TASKS as TRANSFER_TASKS, TransferResources
from vlm_tamp.cohort_resources import CohortResources
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.coupled_resources import api_snapshot, CoupledResources
from vlm_tamp.pipeline_timing import Timeline
from vlm_tamp.profiled_pipeline import compress_retained_observations
from vlm_tamp.shared_contact_planning import select_contact,motion_tangent
from vlm_tamp.shared_contact_geometry import grounded_geometry,closing_target,expand_contact
from vlm_tamp.shared_feedback_client import FeedbackClient,execute_shared
from vlm_tamp.shared_feedback import FeedbackConfig
from vlm_tamp.run_api_budget import api_run_scope,ledger_for_output

TASKS={t['label']:t for t in ORIGINAL_TASKS}
TASKS.update({t['label']:{**t,'adapter':'control rotation' if t['family']=='rotary_control' else 'line'} for t in TRANSFER_TASKS})
VARIANTS=['nominal','reanchor','reactive']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Resources(TransferResources):
    def __init__(self,out,timeline):
        super().__init__(out,timeline,maximum_scenes=100)
        self.case_cache=out/'paired-input-cache';self.case_cache.mkdir(exist_ok=True)

    def candidates(self,label,index):
        return [ROOT/'results/family-transfer-frozen-20261016'/f'{label}-state{index:03d}/grounding',
                *super().candidates(label,index)]

    def grounding(self,*args,**kwargs):
        return CohortResources.grounding(self,*args,**kwargs)

    def grasps(self,cloud,output):
        old=self.cache;self.cache=self.case_cache
        try:return CoupledResources.grasps(self,cloud,output)
        finally:self.cache=old

    def cleanup_proposals(self):
        # Same proposals across paired controller variants, then runner removes
        # the extra cache once complete evidence exists in all episode folders.
        pass

    def cleanup_case(self):
        if self.case_cache.exists():shutil.rmtree(self.case_cache)
        self.case_cache.mkdir()


@contextlib.contextmanager
def adapters(variant):
    saved=[];clients=[]
    def patch(obj,name,value):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    def factory(*args,**kwargs):
        sim=FeedbackClient(*args,**kwargs,variant=variant);clients.append(sim);return sim
    for engine in [mechanisms,bowl,microwave]:
        patch(engine,'CoupledClient',factory);patch(engine,'execute',execute_shared)
    patch(shared_mechanisms,'CoupledClient',factory)
    patch(mechanisms,'select_grasp',select_contact)
    patch(mechanisms,'ground_geometry',grounded_geometry)
    patch(mechanisms,'drawer_closing_target',closing_target)
    patch(mechanisms,'expand_handle_surface',expand_contact)
    original=microwave.select_grasp
    def hinge_contact(generated,anchors,state,ik,record,open_scene,closed_scene,spec,goal,folder,config):
        selected=original(generated,anchors,state,ik,record,open_scene,closed_scene,spec,goal,folder,config)
        context={**{k:v for k,v in selected.items() if k not in {'approach','planned_contact'}},
                 'mode':'pinch','contact_normal_world':spec['panel_normal_world'],'half_aperture_m':.04,
                 'motion_tangent_world':motion_tangent(spec,selected['pose_world_xyzw']),
                 'mode_selection':'existing admitted pinch; shared feedback interface'}
        clients[-1].contact_context(context,closed_scene.contact_reference)
        return selected
    patch(microwave,'select_grasp',hinge_contact)
    try:yield clients
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)
        for client in clients:
            if client.process.poll() is None:client.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--cases',nargs='+',required=True,help='task:state pairs')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    p.add_argument('--development',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--api-run-id',help='Shared $2 experiment ID; use the same ID across parallel cohorts. Resumes preserve it.')
    a=p.parse_args();out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output must be in project results')
    cases=[]
    for token in a.cases:
        name,index=token.rsplit(':',1);index=int(index)
        if name not in TASKS or not 0<=index<50:raise ValueError('Unknown task or official state')
        cases.append((name,index))
    planned=[{'task':name,'state':index,'variant':v} for name,index in cases for v in a.variants]
    if not a.resume:out.mkdir(parents=True,exist_ok=False)
    elif not (out/'protocol.json').exists():raise ValueError('Cannot resume an experiment without its protocol')
    ledger=ledger_for_output(out,a.api_run_id)
    with api_run_scope(ledger.run_id):
        run_experiment(a,out,cases,planned,ledger)


def run_experiment(a,out,cases,planned,ledger):
    if a.resume:
        protocol=read_json(out/'protocol.json');assert read_json(out/'manifest.json')['episodes']==planned
        if protocol.get('api_run_id')!=ledger.run_id:
            raise ValueError('Resume must preserve the frozen experiment API scope; legacy runs retain their saved sources')
    else:
        files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__),ROOT/'scripts/reliable_microwave_final.py',ROOT/'scripts/run_success_pipeline.py',ROOT/'configs/api_budget_policy.json']
        hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
        for name in hashes:
            target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        protocol={'sources_sha256':hashes,'feedback_configuration':FeedbackConfig().record(),
            'development':a.development,'frozen_before_first_episode':True,'per_case_tuning':False,
            'variants':a.variants,'contact_mode_menu_shared_across_variants':True,
            'planning_scene_admission_enabled':True,'execution_vetoes':False,
            'proprioception_update_hz':20,'depth_update':'between completed nominal trajectories',
            'controller_kind':'bounded reference accommodation; not force-sensor admittance or RL',
            'manual_priors':['finite contact-mode menu','part-frame approach tilt menu','gravity direction','fixed shared controller gains'],
            'api_before_spend_usd':api_snapshot()['cumulative_conservative_spend_usd'],
            'api_run_id':ledger.run_id,'api_budget_scope':'experiment_run','api_budget_usd':2.,
            'api_run_before_spend_usd':api_snapshot()['run_conservative_spend_usd'],
            'maximum_control_steps':600,'seed':0,'settling_actions':5,
            'task_configurations':{name:task_configuration(TASKS[name])[1] for name,_ in cases}}
        write_json(out/'protocol.json',protocol)
        write_json(out/'manifest.json',{'episodes':planned,'all_planned_episodes_in_denominator':True})
    timeline=Timeline();resources=Resources(out,timeline);rows=[]
    for name,index in cases:
        import torch
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
        for variant in a.variants:
            if (out/'stop-after-episode').exists():
                write_json(out/'stopped-between-episodes.json',{'completed':len(rows),'planned':len(planned),
                    'reason':'development superseded; every issued trajectory completed',
                    'remaining':planned[len(rows):]})
                resources.cleanup_case();resources.case_cache.rmdir();return
            assert all(sha(ROOT/n)==h for n,h in protocol['sources_sha256'].items()),'Frozen sources changed'
            folder=out/f'{name}-state{index:03d}-{variant}'
            if (folder/'summary.json').exists():result=read_json(folder/'summary.json')
            else:
                if folder.exists():raise ValueError('Interrupted case requires diagnosis; no silent rerun')
                print(json.dumps({'event':'started','task':name,'state':index,'variant':variant}),flush=True)
                fn,config,_=task_configuration(TASKS[name]);before=resources.new_api_calls
                if TASKS[name]['adapter']=='line' and 'legacy_index' not in TASKS[name]:
                    fn=shared_mechanisms.run_episode
                with adapters(variant):result=fn(TASKS[name],index,folder,config,resources)
                compress_retained_observations(folder);result=read_json(folder/'summary.json')
                result.update(controller_variant=variant,development=a.development,
                    new_api_calls=resources.new_api_calls-before,shared_feedback_configuration=True,
                    planning_scene_admission_enabled=True,baseline_sources_unmodified=True)
                write_json(folder/'summary.json',result)
            rows.append(result);timeline.save(out/'timing.json')
            groups={v:{'planned':sum(x['variant']==v for x in planned),'completed':sum(x['controller_variant']==v for x in rows),
                'native_successes':sum(x['official_task_success'] for x in rows if x['controller_variant']==v)} for v in a.variants}
            write_json(out/'summary.json',{'planned':len(planned),'completed':len(rows),'by_controller':groups,'episodes':rows,
                'api_run_id':ledger.run_id,'api_budget_scope':'experiment_run','api_budget_usd':2.,
                'run_api_spend_usd':api_snapshot()['run_conservative_spend_usd'],
                'remaining_run_api_budget_usd':api_snapshot()['remaining_run_budget_usd'],
                'cumulative_api_spend_usd':api_snapshot()['cumulative_conservative_spend_usd']})
            print(json.dumps({'event':'completed','task':name,'state':index,'variant':variant,
                'success':result['official_task_success'],'steps':result['control_steps'],'failure':result.get('failure')}),flush=True)
        resources.cleanup_case()
    resources.cleanup_case();resources.case_cache.rmdir()

if __name__=='__main__':main()
