"""Observation-validated caches and bounded Luna/GraspGen inputs for new states."""
import hashlib
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/'results/articulated-api-v1/ledger.json'


def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def api_snapshot(ledger=None):
    from .run_api_budget import snapshot
    return snapshot(ledger,legacy_path=LEDGER)


def fresh_task_template(previous, choices):
    """Carry task semantics, never a prior scene's camera or numeric geometry."""
    if previous.get('simulator_scene_state_used') is not False:
        raise ValueError('Task template needs non-oracle provenance')
    if any(previous[key]!=choices[key] for key in ['joint_family','motion']):
        raise ValueError('Current RGB primitive differs from the requested task')
    keys=['schema_version','task_type','instruction','joint_family','motion',
          'target_progress','progress_units','axis_prior','contact_policy','constraint']
    result={key:copy.deepcopy(previous[key]) for key in keys if key in previous}
    result.update(camera=choices['camera'],simulator_scene_state_used=False)
    return result


def _cached(*args):
    from .rekep_libero.mechanism_execution import cached_grounding
    return cached_grounding(*args)


def _fresh(*args,**kwargs):
    from .rekep_libero.articulated_grounding import ground
    return ground(*args,**kwargs)


class CoupledResources:
    def __init__(self,cache=None,*,api_output=None):
        self.cache=Path(cache or ROOT/'results/coupled-input-cache-v1')
        self.cache.mkdir(parents=True,exist_ok=True)
        self.perception=None;self.graspgen=None;self.pins=None
        self.api_output=Path(api_output) if api_output is not None else None

    def api_ledger(self,grounding_output=None):
        from .run_api_budget import ledger_for_output
        from .rekep_libero.api import BudgetError
        output=getattr(self,'output',None) or self.api_output
        if output is None and grounding_output is not None:
            # Older runners place episodes beneath a single results/<run> root.
            try:relative=Path(grounding_output).resolve().relative_to(ROOT/'results')
            except ValueError:relative=None
            if relative is not None and len(relative.parts)>1:
                output=ROOT/'results'/relative.parts[0]
        if output is None:
            raise BudgetError('Fresh grounding requires an experiment output directory for its shared API budget')
        return ledger_for_output(output)

    def grounding(self,label,state,observation,metadata,output):
        observation,output=Path(observation),Path(output)
        image_hash=hashlib.sha256(b''.join(np.asarray(Image.open(observation/(name+'.png'))).tobytes()
            for name in ['agentview','frontview'])).hexdigest()
        cache=self.cache/f'{label}-state{state:03d}-rgb-{image_hash}'
        candidates=[cache]
        if state==1:candidates.append(ROOT/'results/articulated-physics-pilot'/f'{label}-state001'/'grounding')
        for number,source in enumerate(candidates):
            if not (source/'grounding.json').exists():continue
            record=read(source/'grounding.json')
            if record.get('provided_inventory') is not False or record.get('simulator_scene_state_used') is not False:
                raise ValueError('Grounding cache must have RGB-only, non-oracle provenance')
            trial=output.with_name(output.name+f'-cache-check-{number}')
            try:
                _,choices,_,_=_cached(source,observation,metadata,trial)
            except (ValueError,FileNotFoundError) as error:
                if not trial.exists():trial.mkdir(parents=True)
                save(trial/'cache-rejection.json',{'source':str(source),'reason':str(error)})
                continue
            trial.rename(output)
            save(output/'input-provenance.json',{'rgb_input_sha256':image_hash,'new_api_calls':0,
                'cached_grounding_source':str(source),'rgb_and_selected_depth_verified':True})
            return choices
        ledger=self.api_ledger(output);before=api_snapshot(ledger)
        scene_id=f'coupled-rgb-v1-{label}-state{state:03d}-{image_hash[:16]}'
        if self.perception is None:
            from .rekep_libero.perception import Perception
            self.perception=Perception()
        output.mkdir(parents=True,exist_ok=True)
        try:
            _,choices,_,_=_fresh(self.perception,observation,metadata,output,scene_id,ledger=ledger)
        finally:
            after=api_snapshot(ledger)
            entry=after['requests'].get(scene_id)
            save(output/'input-provenance.json',{'rgb_input_sha256':image_hash,'scene_id':scene_id,
                'new_api_calls':int(scene_id not in before['requests'] and entry is not None),
                'shared_cumulative_spend_usd':after['cumulative_conservative_spend_usd'],
                'api_run_id':ledger.run_id,'budget_scope':'experiment_run',
                'run_spend_usd':after['run_conservative_spend_usd'],
                'shared_remaining_usd':after['remaining_run_budget_usd'],'automatic_api_retries':0,
                'rgb_and_selected_depth_verified':False,'provided_inventory':False})
        if not cache.exists():shutil.copytree(output,cache)
        return choices

    def grasps(self,cloud,output,*,num_grasps=8,topk=8):
        import yaml
        began=time.perf_counter()
        output=Path(output)
        xyz=np.asarray(cloud,np.float32)
        seed=getattr(self,'grasp_seed',0)
        if type(seed) is not int or not 0<=seed<2**31:
            raise ValueError('Grasp seed must be an integer in [0, 2**31)')
        if type(num_grasps) is not int or type(topk) is not int or not 1 <= topk <= num_grasps:
            raise ValueError('Grasp counts must be integers with 1 <= topk <= num_grasps')
        sampling={'num_grasps':num_grasps,'topk':topk,'tries':1}
        if self.pins is None:
            config=ROOT/'.runtime/graspgen-models/checkpoints/graspgen_franka_panda.yml'
            raw=yaml.safe_load(config.read_text())
            self.pins={'config_sha256':sha(config)}
            for key in ['eval','discriminator']:
                checkpoint=Path(raw[key]['checkpoint'])
                if not checkpoint.is_absolute():checkpoint=config.parent/checkpoint
                self.pins[key+'_checkpoint_sha256']=sha(checkpoint)
        seed_key=b'' if seed==0 else f'grasp-seed:{seed}'.encode()
        digest=hashlib.sha256(xyz.tobytes()+json.dumps(self.pins,sort_keys=True).encode()+seed_key+
            json.dumps(sampling,sort_keys=True).encode()).hexdigest()
        cache=self.cache/('graspgen-'+digest)
        lookup_seconds=time.perf_counter()-began
        if cache.exists():
            cache_started=time.perf_counter()
            provenance=read(cache/'input-provenance.json')
            if (provenance['cache_key']!=digest or provenance['checkpoint_pins']!=self.pins
                    or provenance.get('grasp_seed',0)!=seed or provenance.get('sampling')!=sampling):
                raise ValueError('Grasp cache checkpoint provenance differs')
            if not np.array_equal(np.load(cache/'input-cloud.npy'),xyz):
                raise ValueError('Grasp cache point cloud differs')
            shutil.copytree(cache,output)
            provenance.update(reused=True);save(output/'input-provenance.json',provenance)
            with np.load(output/'grasp_candidates.npz') as data:
                result={key:data[key].copy() for key in ['poses_world_base','scores','retained_indices']}
            save(output/'generation-timing.json',{'cache_reused':True,
                'checkpoint_and_cache_key_seconds':lookup_seconds,'cache_read_seconds':time.perf_counter()-cache_started,
                'model_initialization_seconds':0.,'generation_call_seconds':0.,'cache_write_seconds':0.,
                'total_seconds':time.perf_counter()-began,'historical_inference_time_is_not_current_work':True})
            return result
        initialization_seconds=0.
        if self.graspgen is None:
            model_started=time.perf_counter()
            from .rekep_libero.perception import GraspGen
            self.graspgen=GraspGen()
            initialization_seconds=time.perf_counter()-model_started
        generation_started=time.perf_counter()
        result=self.graspgen.generate(xyz,output,seed=seed,num_grasps=num_grasps,topk=topk)
        generation_seconds=time.perf_counter()-generation_started
        cache_started=time.perf_counter()
        np.save(output/'input-cloud.npy',xyz)
        save(output/'input-provenance.json',{'cache_key':digest,'reused':False,'checkpoint_pins':self.pins,'grasp_seed':seed,'sampling':sampling,
            'input_source':'current predicted RGB-D handle cloud','provided_inventory':False})
        shutil.copytree(output,cache)
        save(output/'generation-timing.json',{'cache_reused':False,
            'checkpoint_and_cache_key_seconds':lookup_seconds,'cache_read_seconds':0.,
            'model_initialization_seconds':initialization_seconds,'generation_call_seconds':generation_seconds,
            'cache_write_seconds':time.perf_counter()-cache_started,'total_seconds':time.perf_counter()-began,
            'generation_call_scope':'Sampler, host processing, candidate persistence and checkpoint hashes; not pure GPU utilization.'})
        return result
