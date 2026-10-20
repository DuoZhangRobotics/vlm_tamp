"""Observation-validated caches and bounded Luna/GraspGen inputs for new states."""
import hashlib
import copy
import json
from pathlib import Path
import shutil

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

    def grasps(self,cloud,output):
        import yaml
        output=Path(output)
        xyz=np.asarray(cloud,np.float32)
        if self.pins is None:
            config=ROOT/'.runtime/graspgen-models/checkpoints/graspgen_franka_panda.yml'
            raw=yaml.safe_load(config.read_text())
            self.pins={'config_sha256':sha(config)}
            for key in ['eval','discriminator']:
                checkpoint=Path(raw[key]['checkpoint'])
                if not checkpoint.is_absolute():checkpoint=config.parent/checkpoint
                self.pins[key+'_checkpoint_sha256']=sha(checkpoint)
        digest=hashlib.sha256(xyz.tobytes()+json.dumps(self.pins,sort_keys=True).encode()).hexdigest()
        cache=self.cache/('graspgen-'+digest)
        if cache.exists():
            provenance=read(cache/'input-provenance.json')
            if provenance['cache_key']!=digest or provenance['checkpoint_pins']!=self.pins:
                raise ValueError('Grasp cache checkpoint provenance differs')
            if not np.array_equal(np.load(cache/'input-cloud.npy'),xyz):
                raise ValueError('Grasp cache point cloud differs')
            shutil.copytree(cache,output)
            provenance.update(reused=True);save(output/'input-provenance.json',provenance)
            with np.load(output/'grasp_candidates.npz') as data:
                return {key:data[key].copy() for key in ['poses_world_base','scores','retained_indices']}
        if self.graspgen is None:
            from .rekep_libero.perception import GraspGen
            self.graspgen=GraspGen()
        result=self.graspgen.generate(xyz,output)
        np.save(output/'input-cloud.npy',xyz)
        save(output/'input-provenance.json',{'cache_key':digest,'reused':False,'checkpoint_pins':self.pins,
            'input_source':'current predicted RGB-D handle cloud','provided_inventory':False})
        shutil.copytree(output,cache)
        return result
