"""Same region tools with independently labeled camera images in the API."""
import base64
import hashlib
import json
from pathlib import Path
import time
from PIL import Image
from .region_episode import RegionTools
from .region_decision import PROMPT,parse_decision
from .coupled_storage import write_json
from .rekep_libero.articulated_grounding import response_text

def separate_prompt():
    text=PROMPT.read_text();start=text.index('IMAGES\n');end=text.index('\nTOOLS\n')
    return text[:start]+'''IMAGES
Current camera images are supplied separately, each immediately after its camera
name. Image agentview and image frontview are each 512 by 512 pixels. Normalize
coordinates within that individual image: u=0 at its left edge, u=1 at its right
edge, v=0 at its top edge, v=1 at its bottom edge. There is no shared camera-sheet
coordinate frame. The grid labels are camera-local. Cyan outlines the estimated
moving panel; orange marks the contact patch; yellow marks estimated motion.
An optional final image contains rejected-pose diagnostics from both views. Use
it only to inspect the hypothetical purple robot and red overlap witness, never
as the coordinate image for a region selection. It uses the RGB observation from
planning time, which can precede the current views. No physical collision is
implied and the geometry estimate can be wrong.
'''+text[end:]

class SeparateRegionTools(RegionTools):
    def render_feedback(self,output):
        super().render_feedback(output)
        output=Path(output)
        # Extract exact, unrescaled current panels from the existing renderer.
        # This changes presentation only, not pixels, masks, fitting or tools.
        with Image.open(output) as sheet:
            sheet.crop((0,24,512,536)).save(output.parent/'camera-agentview.png')
            sheet.crop((512,24,1024,536)).save(output.parent/'camera-frontview.png')
            if self.last_collision and self.last_collision['representative_samples']:
                sheet.crop((0,548,1024,1096)).save(output.parent/'collision-preview.png')

def request_arguments(folder,prompt):
    folder=Path(folder);content=[{'type':'input_text','text':prompt}];files=[]
    for name in ('agentview','frontview'):
        path=folder/f'camera-{name}.png';files.append(path.name)
        content.append({'type':'input_text','text':f'CURRENT CAMERA: {name}. This individual image is 512x512. Use camera-local normalized coordinates.'})
        content.append({'type':'input_image','detail':'high','image_url':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()})
    diagnostic=folder/'collision-preview.png'
    if diagnostic.exists():
        files.append(diagnostic.name)
        content.append({'type':'input_text','text':'DIAGNOSTIC ONLY: rejected hypothetical robot poses. Do not select region coordinates from this image.'})
        content.append({'type':'input_image','detail':'high','image_url':'data:image/png;base64,'+base64.b64encode(diagnostic.read_bytes()).decode()})
    return {'model':'gpt-6-luna','input':[{'role':'user','content':content}],
        'reasoning':{'effort':'high'},'service_tier':'default','max_output_tokens':32000,'store':False},files

class SeparateLunaStrategy:
    def __init__(self,ledger,*,allow_paid=False):self.ledger=ledger;self.allow_paid=allow_paid;self.calls=0
    def choose(self,packet,image,folder,scene_id):
        folder=Path(folder)
        prompt=separate_prompt()+'\nCURRENT OBSERVATION AND TOOL HISTORY\n'+json.dumps(packet,sort_keys=True,allow_nan=False)
        (folder/'prompt.txt').write_text(prompt)
        arguments,files=request_arguments(folder,prompt)
        before=self.ledger.snapshot();started=time.perf_counter()
        try:raw=self.ledger.perform(arguments,scene_id,allow_paid=self.allow_paid)
        finally:
            after=self.ledger.snapshot();new=after['api_attempts']-before['api_attempts'];self.calls+=new
            write_json(folder/'api-accounting.json',{'api_run_id':self.ledger.run_id,'new_api_calls':new,
                'max_output_tokens':32000,'reasoning_effort':'high','round_trip_seconds':time.perf_counter()-started,
                'automatic_api_retries':0,'run_spend_usd':after['run_conservative_spend_usd'],
                'remaining_run_budget_usd':after['remaining_run_budget_usd'],
                'image_input_mode':'separate_cameras','image_files':files,
                'request_sha256':hashlib.sha256(json.dumps(arguments,sort_keys=True).encode()).hexdigest()})
        write_json(folder/'response.json',raw);text=response_text(raw);(folder/'response.txt').write_text(text)
        if raw.get('status')!='completed':raise ValueError('Strategy response incomplete; no automatic retry')
        decision=parse_decision(text);write_json(folder/'decision.json',decision);return decision
