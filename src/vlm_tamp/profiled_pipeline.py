"""Timing and I/O substitutions around the unchanged mechanism planning engine."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw

from .mechanism_surface import MechanismResources
from .cohort_resources import prepare_contact
from .coupled_resources import api_snapshot
from .coupled_storage import read_json,write_json
from .rekep_libero.articulated_grounding import PROMPT,parse_choices,response_text
from .rekep_libero.api import request_arguments
from .profiled_sim_client import ProfiledClient,execute_batch

ROOT=Path(__file__).resolve().parents[2]


def fresh_request_arguments(image,prompt,benchmark_id):
    args=request_arguments(image,prompt,12000)
    # Documented request metadata separates this explicitly requested timing
    # trial from the durable response cache without changing model input.
    args['metadata']={'benchmark_id':benchmark_id}
    return args


class ProfiledResources(MechanismResources):
    def __init__(self,output,timeline,*,fresh=False):
        super().__init__(output);self.timeline=timeline;self.fresh=fresh;self.http_calls=0;self._vision_instrumented=False
    def ensure_perception(self):
        if self.perception is None:
            with self.timeline.span('initialization.vision_models'):
                super().ensure_perception()
        if not self._vision_instrumented:
            t=self.timeline;p=self.perception
            for obj,method,name in [(p.mask_generator,'generate','vision.sam_automatic_masks'),
                (p.point_predictor,'set_image','vision.sam_image_encoder'),(p.point_predictor,'predict','vision.sam_point_mask'),
                (p.proposer,'_get_features','vision.dinov2_features_and_transfer'),
                (p.proposer,'_cluster_features','vision.keypoint_clustering'),
                (p.proposer,'_merge_clusters','vision.keypoint_merging'),
                (p.proposer,'get_keypoints','vision.keypoint_proposal_total')]:
                setattr(obj,method,t.call(name,getattr(obj,method),gpu=True))
            p.prepare=t.call('vision.prepare_view_total',p.prepare,gpu=True)
            p.selected_cloud=t.call('vision.selected_contact_cloud',p.selected_cloud,gpu=True)
            self._vision_instrumented=True
        return self.perception
    def warm_models(self):
        self.ensure_perception()
        if self.graspgen is None:
            with self.timeline.span('initialization.grasp_model'):
                import random,torch
                from .rekep_libero.perception import GraspGen
                # Moving model loading earlier must not perturb the vision
                # proposer's NumPy/Torch/Python random streams.
                saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
                try:self.graspgen=GraspGen()
                finally:
                    random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])
                self.graspgen.generate=self.timeline.call('grasp.generation_and_artifact_total',self.graspgen.generate,gpu=True)
        self.timeline.mark('vision_and_grasp_models_loaded')
    def grounding(self,label,index,observation,metadata,output):
        with self.timeline.span('grounding.total',fresh_output=self.fresh):
            if not self.fresh:return super().grounding(label,index,observation,metadata,output)
            output=Path(output);output.mkdir(parents=True,exist_ok=False);observation=Path(observation)
            perception=self.ensure_perception();bundles={};sheet=Image.new('RGB',(1024,1072),'white');draw=ImageDraw.Draw(sheet)
            for row,name in enumerate(['agentview','frontview']):
                bundles[name]=perception.prepare(observation/'obs.npz',{**metadata,'main_camera':name},output/name)
                for column,kind in enumerate(['rgb','numbered']):
                    draw.text((column*512+10,row*536+5),f'{name}: {kind}',fill='black')
                    sheet.paste(Image.open(output/name/f'{kind}.png'),(column*512,row*536+24))
            with self.timeline.span('grounding.encode_request'):
                sheet.save(output/'role-choice-sheet.png');prompt=PROMPT.format(instruction=metadata['instruction'])
                (output/'prompt.txt').write_text(prompt)
                benchmark_id=f'{self.output.name}-{label}-{index:03d}'
                arguments=fresh_request_arguments(output/'role-choice-sheet.png',prompt,benchmark_id)
            from .cli import load_credentials
            from openai import OpenAI
            load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
            with self.timeline.span('initialization.luna_client'):
                client=OpenAI(base_url='https://api.openai.com/v1',timeout=240,max_retries=0)
            owner=self;http_before=self.http_calls
            class TimedResponses:
                def create(self,**kwargs):
                    owner.http_calls+=1
                    with owner.timeline.span('grounding.luna_round_trip',request_number=owner.http_calls) as event:
                        response=client.responses.create(**kwargs)
                    event['response_status']=response.status
                    event['usage']=response.usage.model_dump(mode='json') if response.usage else None
                    return response
            class TimedClient:responses=TimedResponses()
            ledger=self.api_ledger();before=api_snapshot(ledger);scene_id='timing-'+benchmark_id
            try:
                with self.timeline.span('grounding.luna_request_with_durable_accounting'):
                    raw=ledger.perform(arguments,scene_id,allow_paid=True,client=TimedClient())
            finally:
                client.close();after=api_snapshot(ledger)
                self.new_api_calls+=int(scene_id not in before['requests'] and scene_id in after['requests'])
                write_json(output/'input-provenance.json',{'new_api_calls':self.new_api_calls,'actual_http_calls':self.http_calls,
                    'fresh_dino_and_sam':True,'output_cache_disabled':True,'request_metadata_tag':benchmark_id,
                    'prompt_text_unchanged':True,'model':'gpt-6-luna','reasoning_effort':'high','max_output_tokens':12000,
                    'api_run_id':ledger.run_id,'budget_scope':'experiment_run',
                    'run_spend_usd':after['run_conservative_spend_usd'],
                    'automatic_retries':0,'cumulative_cost_usd':after['cumulative_conservative_spend_usd']})
            if self.http_calls!=http_before+1:raise ValueError('Fresh timing run unexpectedly reused a response')
            write_json(output/'response.json',raw);text=response_text(raw);(output/'response.txt').write_text(text)
            if raw.get('status')!='completed':raise ValueError('Luna did not complete; no automatic retry')
            with self.timeline.span('grounding.parse_validate_and_contact'):
                choices=parse_choices(text);prepare_contact(perception,bundles[choices['camera']],choices,output)
            return choices
    def grasps(self,*args,**kwargs):
        with self.timeline.span('grasp.resources_total'):return super().grasps(*args,**kwargs)


@contextmanager
def instrument_engine(timeline,*,fast):
    from . import mcvamp_articulated as engine,mcvamp_mechanism as native
    saved=[];clients=[];cache={};execution_index=0
    def patch(obj,name,value):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    def client_factory(*args,**kwargs):
        with timeline.span('initialization.simulator_process'):
            client=ProfiledClient(*args,**kwargs,fast=fast,timeline=timeline)
        clients.append(client);return client
    patch(engine,'CoupledClient',client_factory)
    execution=execute_batch if fast else engine.execute
    def run_commands(*args,**kwargs):
        nonlocal execution_index
        if execution_index==0:timeline.mark('first_planned_task_motion_requested')
        with timeline.span('execution.segment',segment_index=execution_index,commands=len(args[2])):
            result=execution(*args,**kwargs)
        execution_index+=1;return result
    patch(engine,'execute',run_commands)
    for name,label in [('read_cohort_observation','perception.load_observation'),('legacy_grounding','grounding.cached_legacy'),
        ('compile_scene','constraints.legacy_geometry'),('refine_contact','perception.refine_contact'),
        ('ground_geometry','constraints.metric_geometry'),('expand_handle_surface','perception.fuse_handle'),
        ('refine_control_axis','constraints.control_axis'),('SurfaceScene','collision.observed_scene_update'),
        ('select_grasp','grasp.selection_and_preplans'),('segment','planning.complete_segment'),
        ('observed_control_rotation','perception.control_motion_tracking')]:
        patch(engine,name,timeline.call(label,getattr(engine,name)))
    ik_type=engine.TaskIK
    def ik_factory(*args,**kwargs):
        with timeline.span('initialization.ik_worker_and_calibration'):ik=ik_type(*args,**kwargs)
        ik.solve_many=timeline.call('planning.ik_query_round_trip',ik.solve_many)
        return ik
    patch(engine,'TaskIK',ik_factory)
    native_type=engine.NativeScene
    patch(engine,'NativeScene',timeline.call('collision.native_context_total',native_type))
    if fast:
        from .exact_voxels import voxel_points as exact_voxel_points
        from .rekep_libero import scene_state_v7
        patch(scene_state_v7,'voxel_points',timeline.call('collision.voxel_deduplication',exact_voxel_points))
        patch(native,'voxel_points',timeline.call('collision.native_voxel_deduplication',exact_voxel_points))
        original=native.calibrated_frames
        def calibrated(record):
            key=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
            with timeline.span('collision.robot_frame_calibration',cache_hit=key in cache):
                if key not in cache:cache[key]=original(record)
                return cache[key]
        patch(native,'calibrated_frames',calibrated)
    try:yield engine,clients
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


def compress_retained_observations(folder):
    """Move compression after execution, checking lossless array equality."""
    folder=Path(folder);rows=[]
    for path in folder.glob('*/obs.npz'):
        before=path.stat().st_size
        with np.load(path) as archive:arrays={k:archive[k] for k in archive.files}
        temp=path.with_name('obs-retention.npz');np.savez_compressed(temp,**arrays)
        with np.load(temp) as check:
            if set(check.files)!=set(arrays) or any(not np.array_equal(arrays[k],check[k],equal_nan=True) for k in arrays):
                temp.unlink();raise ValueError('Lossless observation recompression failed')
        after=temp.stat().st_size
        if after<before:temp.replace(path)
        else:temp.unlink();after=before
        rows.append({'path':str(path.relative_to(folder)),'before_bytes':before,'after_bytes':after,'arrays_equal':True})
    reclaimed=sum(r['before_bytes']-r['after_bytes'] for r in rows)
    retention=read_json(folder/'retention.json');retention['lossless_after_run_recompression']=rows
    write_json(folder/'retention.json',retention)
    summary=read_json(folder/'summary.json');summary['storage']['lossless_recompression_bytes']=reclaimed
    summary['storage']['bytes_reclaimed_this_pass']+=reclaimed;write_json(folder/'summary.json',summary)
    return rows
