"""Fresh shared visual hypotheses with a durable per-experiment spending ledger."""
import hashlib
from pathlib import Path
import time
from PIL import Image,ImageDraw
import numpy as np

from .profiled_pipeline import ProfiledResources
from .coupled_resources import api_snapshot
from .coupled_storage import write_json
from .interaction_contract import parse_interaction
from .interaction_geometry import fit_interaction
from .rekep_libero.articulated_grounding import response_text
from .rekep_libero.api import request_arguments

ROOT=Path(__file__).resolve().parents[2]
PROMPT=ROOT/'configs/prompts/interaction_choices_v1.txt'


class InteractionResources(ProfiledResources):
    def ensure_perception(self):
        perception=super().ensure_perception()
        if not getattr(perception,'bounded_clustering_installed',False):
            from types import MethodType
            from .bounded_keypoints import cluster_features
            perception.proposer._cluster_features=self.timeline.call('vision.bounded_keypoint_clustering',MethodType(cluster_features,perception.proposer),gpu=True)
            perception.bounded_clustering_installed=True
        return perception

    def ground_interaction(self,observation,metadata,output):
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        perception=self.ensure_perception();bundles={}
        sheet=Image.new('RGB',(1024,1072),'white');draw=ImageDraw.Draw(sheet)
        for row,name in enumerate(['agentview','frontview']):
            bundles[name]=perception.prepare(Path(observation)/'obs.npz',{**metadata,'main_camera':name},output/name)
            write_json(output/name/'clustering-compatibility.json',{
                'released_dino_weights_unchanged':True,'normalization':'constant dimensions map to zero',
                'maximum_lloyd_iterations':100,'empty_clusters':'skipped; at most five proposals per mask',
                'records':perception.proposer.bounded_cluster_records})
            for column,kind in enumerate(['rgb','numbered']):
                draw.text((column*512+10,row*536+5),f'{name}: {kind}',fill='black')
                sheet.paste(Image.open(output/name/f'{kind}.png'),(column*512,row*536+24))
        sheet.save(output/'role-choice-sheet.png')
        # Only the instruction is inserted. Task label, state index, inventory,
        # simulator object identities, coordinates and evaluation outcomes are absent.
        prompt=PROMPT.read_text().replace('{instruction}',metadata['instruction'])
        (output/'prompt.txt').write_text(prompt)
        arguments=request_arguments(output/'role-choice-sheet.png',prompt,12000)
        digest=hashlib.sha256(prompt.encode()+ (output/'role-choice-sheet.png').read_bytes()).hexdigest()
        scene_id='interaction-v1-'+digest
        from .cli import load_credentials
        load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
        ledger=self.api_ledger();before=api_snapshot(ledger);started=time.perf_counter()
        try:
            with self.timeline.span('grounding.luna_round_trip'):
                raw=ledger.perform(arguments,scene_id,allow_paid=True)
        finally:
            after=api_snapshot(ledger);new=int(scene_id not in before['requests'] and scene_id in after['requests'])
            self.new_api_calls+=new
            write_json(output/'input-provenance.json',{'new_api_calls':new,'request_key':scene_id,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'model':'gpt-6-luna',
                'reasoning_effort':'high','max_output_tokens':12000,'automatic_api_retries':0,
                'provided_inventory':False,'simulator_scene_state_used':False,
                'fresh_dino_and_sam':True,'response_reused':not bool(new),
                'api_run_id':ledger.run_id,'budget_scope':'experiment_run',
                'run_spend_usd':after['run_conservative_spend_usd'],
                'cumulative_spend_usd':after['cumulative_conservative_spend_usd'],
                'remaining_usd':after['remaining_usd'],'request_wall_seconds':time.perf_counter()-started})
        write_json(output/'response.json',raw);text=response_text(raw);(output/'response.txt').write_text(text)
        if raw.get('status')!='completed':raise ValueError('Luna response incomplete; no automatic retry')
        selection=parse_interaction(text);write_json(output/'selection.json',selection)
        bundle=bundles[selection['camera']]
        from try_rekep_bowl_v3 import read_observation_v3
        cameras,_=read_observation_v3(Path(observation),metadata)
        bundle['cameras']=cameras
        with self.timeline.span('grounding.generic_geometry_fit'):
            result=fit_interaction(selection,cameras,metadata,bundle,perception,output)
        self._overlay(result,output)
        return result

    @staticmethod
    def _overlay(result,output):
        from .rekep_libero.mechanism_geometry import project
        selection=result.selection;camera=result.cameras[selection['camera']]
        image=Image.fromarray(camera['rgb']).convert('RGB');draw=ImageDraw.Draw(image)
        for box,color,label in [(selection['contact']['box_xyxy'],'lime','contact'),
            (selection['moving']['box_xyxy'],'cyan','moving'),(selection['goal']['reference_box_xyxy'],'yellow','goal reference')]:
            if box is not None:
                draw.rectangle(box,outline=color,width=2);draw.text((box[0],max(0,box[1]-12)),label,fill=color)
        model=result.evidence.get('contact_model',{})
        if 'endpoints_world_m' in model:
            points=project(model['endpoints_world_m'],camera)
            draw.line([tuple(p) for p in points],fill='red',width=3)
        if 'hinge_endpoints_world_m' in result.evidence:
            points=project(result.evidence['hinge_endpoints_world_m'],camera)
            draw.line([tuple(p) for p in points],fill='magenta',width=3)
        image.save(output/'grounding-overlay.png')
