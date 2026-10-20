"""One model-mediated grounding correction from validation feedback.

No pixel arithmetic, manual box, task-specific hint or response editing. The
original completed response remains intact beside the corrected response.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .coupled_storage import read_json,write_json
from .cohort_resources import prepare_contact
from .rekep_libero.api import request_arguments
from .rekep_libero.articulated_grounding import PROMPT,parse_choices,response_text
from .rekep_libero.perception import _read_observation


def correction_prompt(instruction,previous_response,validation_error):
    return PROMPT.format(instruction=instruction)+'''\nVALIDATION FEEDBACK\n
Your previous completed response could not be used. Review the original image
and the machine validator feedback, then return one corrected complete JSON
object in the original schema. Use local coordinates for the selected camera,
as specified above. Do not claim success or change the schema. This is a new
decision using feedback, not a repeated network request.\n'''+json.dumps({
        'previous_response':previous_response,'validator_feedback':validation_error},sort_keys=True)


class GroundingRepairMixin:
    def grounding(self,label,index,observation,metadata,output):
        try:return super().grounding(label,index,observation,metadata,output)
        except ValueError as error:
            output=Path(output);response=output/'response.json';sheet=output/'role-choice-sheet.png'
            if not response.exists() or not sheet.exists():raise
            original=read_json(response)
            if original.get('status')!='completed':raise
            failure=str(error)
        folder=output/'validation-repair';folder.mkdir(exist_ok=False)
        prompt=correction_prompt(metadata['instruction'],response_text(original),failure)
        (folder/'prompt.txt').write_text(prompt)
        arguments=request_arguments(sheet,prompt,32000)
        ledger=self.api_ledger();before=ledger.snapshot()
        key='grounding-repair-v1-'+hashlib.sha256(json.dumps(arguments,sort_keys=True).encode()).hexdigest()
        try:raw=ledger.perform(arguments,key,allow_paid=getattr(self,'allow_paid',False))
        finally:
            after=ledger.snapshot();new=after['api_attempts']-before['api_attempts'];self.new_api_calls+=new
            write_json(folder/'accounting.json',{'new_api_calls':new,'api_run_id':ledger.run_id,
                'max_output_tokens':32000,'reasoning_effort':'high','automatic_api_retries':0,
                'maximum_validation_corrections':1,'run_spend_usd':after['run_conservative_spend_usd']})
        write_json(folder/'response.json',raw);text=response_text(raw);(folder/'response.txt').write_text(text)
        if raw.get('status')!='completed':raise ValueError('Grounding correction incomplete; no automatic retry')
        choices=parse_choices(text)
        cameras,_=_read_observation(Path(observation)/'obs.npz',metadata);name=choices['camera']
        with np.load(output/name/'keypoints.npz') as values:bundle={k:values[k].copy() for k in values.files}
        bundle.update(cameras=cameras,meta={**metadata,'main_camera':name},main_camera=name)
        prepare_contact(self.ensure_perception(),bundle,choices,output)
        write_json(folder/'verification.json',{'previous_validation_error':failure,'corrected_choice_valid':True,
            'provided_inventory':False,'simulator_scene_state_used':False,'manual_pixel_correction':False,
            'task_specific_repair_hint':False})
        write_json(output/'validation-repair-provenance.json',{'model':'gpt-6-luna','reasoning_effort':'high',
            'source':'one model revision using original image and validator feedback','original_response_preserved':True})
        image_hash=hashlib.sha256(b''.join(np.asarray(Image.open(Path(observation)/f'{name}.png')).tobytes()
            for name in ['agentview','frontview'])).hexdigest()
        cache=self.cache/f'{label}-state{index:03d}-rgb-{image_hash}'
        if not cache.exists():shutil.copytree(output,cache)
        return choices
