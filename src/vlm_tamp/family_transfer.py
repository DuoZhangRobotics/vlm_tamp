"""Fresh visual role selection around the frozen successful mechanism policies."""
import copy
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

from PIL import Image,ImageDraw

from .bounded_keypoints import cluster_features
from .cohort_resources import CohortLedger,prepare_contact
from .coupled_resources import api_snapshot
from .coupled_storage import read_json,write_json
from .mechanism_surface import normalize_sheet_choices
from .profiled_pipeline import ProfiledResources
from .rekep_libero.api import BudgetError,request_arguments
from .rekep_libero.articulated_grounding import PROMPT,response_text

ROOT=Path(__file__).resolve().parents[2]
BASELINE=ROOT/'results/reliability-mechanisms-frozen-20261016'
STATES=[5,15,25,35,45]
TASKS=[
    {'label':'open_top_drawer','name':'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet','suite':'libero_90','family':'drawer','group':'original_tasks'},
    {'label':'close_top_drawer','name':'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet','suite':'libero_90','family':'drawer','group':'original_tasks'},
    {'label':'turn_on_stove','name':'KITCHEN_SCENE3_turn_on_the_stove','suite':'libero_90','family':'rotary_control','group':'original_tasks'},
    {'label':'open_bottom_drawer','name':'KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet','suite':'libero_90','family':'drawer','group':'related_tasks'},
    {'label':'close_bottom_drawer','name':'KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet','suite':'libero_90','family':'drawer','group':'related_tasks'},
    {'label':'turn_on_stove_scene9','name':'KITCHEN_SCENE9_turn_on_the_stove','suite':'libero_90','family':'rotary_control','group':'related_tasks'},
]
CORE_FILES=['src/vlm_tamp/reliability_geometry.py','src/vlm_tamp/mechanism_surface.py',
    'src/vlm_tamp/rekep_libero/mechanism_geometry.py','src/vlm_tamp/mcvamp_mechanism.py',
    'src/vlm_tamp/native/mcvamp_mechanism.cc','src/vlm_tamp/timed_execution.py',
    'src/vlm_tamp/rekep_libero/task_ik_curobo.py','src/vlm_tamp/rekep_libero/physical_tasks_v4.py']


def frozen_configuration():
    """No task-specific parameter overrides, including for the new cases."""
    return copy.deepcopy(read_json(BASELINE/'protocol.json')['configuration'])


def manifest(tasks=TASKS,states=STATES,*,controls=True):
    if len(set(states))!=len(states) or any(type(i) is not int or not 0<=i<50 for i in states):
        raise ValueError('Unique official state indices 0..49 are required')
    rows=[]
    if controls:
        rows.extend({'task':t['label'],'state':1,'phase':'reproduction_control'} for t in tasks if t['group']=='original_tasks')
    for group in ['original_tasks','related_tasks']:
        rows.extend({'task':t['label'],'state':i,'phase':group} for i in states for t in tasks if t['group']==group)
    return rows


def core_audit():
    baseline=read_json(BASELINE/'protocol.json')['sources_sha256'];rows=[]
    for name in CORE_FILES:
        digest=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        rows.append({'path':name,'baseline_sha256':baseline.get(name),'current_sha256':digest,'matches':baseline.get(name)==digest})
    if not all(r['matches'] for r in rows):raise ValueError('A frozen geometry/planning/controller component differs from the successful baseline')
    return rows


class TransferLedger(CohortLedger):
    """Raise only the internal request count for the authorized manifest."""
    def __init__(self,directory,*,maximum_scenes):
        super().__init__(directory,maximum_scenes=maximum_scenes)

    def _read(self):
        saved=read_json(self.path)
        prior_cap=saved.get('max_scene_requests')
        if (saved.get('version')!='rekep-libero-luna-1' or saved.get('model')!='gpt-6-luna'
            or saved.get('budget_usd')!=2. or saved.get('prior_conservative_spend_usd')!=self.prior_spend
            or type(prior_cap) is not int or prior_cap<30):
            raise BudgetError('Shared cumulative ledger changed')
        if prior_cap<self.max_calls:
            saved.setdefault('scene_limit_changes',[]).append({'previous':prior_cap,'current':self.max_calls,
                'reason':'User authorized frozen drawer and rotary-control family transfer trials',
                'dollar_budget_unchanged':2.})
            saved['max_scene_requests']=self.max_calls
        else:self.max_calls=prior_cap
        return saved


class TransferResources(ProfiledResources):
    def __init__(self,output,timeline,*,maximum_scenes):
        super().__init__(output,timeline,fresh=False)
        self.phase='reproduction_control';self.maximum_scenes=maximum_scenes

    def ensure_perception(self):
        perception=super().ensure_perception()
        if not getattr(perception,'bounded_clustering_installed',False):
            perception.proposer._cluster_features=self.timeline.call('vision.bounded_keypoint_clustering',
                MethodType(cluster_features,perception.proposer),gpu=True)
            perception.bounded_clustering_installed=True
        return perception

    def grounding(self,label,index,observation,metadata,output):
        if self.phase=='reproduction_control':
            # Preserve the baseline's observation-validated semantic cache for
            # reproduction controls only. Transfer uses a fresh visual sheet.
            return super().grounding(label,index,observation,metadata,output)
        output=Path(output);output.mkdir(parents=True,exist_ok=False);observation=Path(observation)
        perception=self.ensure_perception();bundles={}
        sheet=Image.new('RGB',(1024,1072),'white');draw=ImageDraw.Draw(sheet)
        for row,name in enumerate(['agentview','frontview']):
            bundles[name]=perception.prepare(observation/'obs.npz',{**metadata,'main_camera':name},output/name)
            write_json(output/name/'bounded-clustering.json',{'records':perception.proposer.bounded_cluster_records,
                'maximum_iterations':100,'weights_changed':False})
            for column,kind in enumerate(['rgb','numbered']):
                draw.text((column*512+10,row*536+5),f'{name}: {kind}',fill='black')
                sheet.paste(Image.open(output/name/f'{kind}.png'),(column*512,row*536+24))
        sheet.save(output/'role-choice-sheet.png')
        prompt=PROMPT.format(instruction=metadata['instruction']);(output/'prompt.txt').write_text(prompt)
        arguments=request_arguments(output/'role-choice-sheet.png',prompt,12000)
        digest=hashlib.sha256(prompt.encode()+(output/'role-choice-sheet.png').read_bytes()).hexdigest()
        scene_id='family-transfer-v1-'+digest
        ledger=self.api_ledger()
        from .cli import load_credentials
        load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
        before=api_snapshot(ledger);started=time.perf_counter()
        try:
            with self.timeline.span('grounding.luna_round_trip'):
                raw=ledger.perform(arguments,scene_id,allow_paid=True)
        finally:
            after=api_snapshot(ledger);new=int(scene_id not in before['requests'] and scene_id in after['requests'])
            self.new_api_calls+=new
            write_json(output/'input-provenance.json',{'request_key':scene_id,'new_api_calls':new,
                'fresh_rgbd_and_vision':True,'response_reused':not bool(new),'original_prompt_unchanged':True,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'model':'gpt-6-luna',
                'reasoning_effort':'high','max_output_tokens':12000,'automatic_retries':0,'provided_inventory':False,
                'demonstration_input':False,'simulator_scene_state_used':False,
                'request_wall_seconds':time.perf_counter()-started,
                'api_run_id':ledger.run_id,'budget_scope':'experiment_run',
                'run_spend_usd':after['run_conservative_spend_usd'],
                'cumulative_spend_usd':after['cumulative_conservative_spend_usd']})
        write_json(output/'response.json',raw);text=response_text(raw);(output/'response.txt').write_text(text)
        if raw.get('status')!='completed':raise ValueError('Luna did not complete; no automatic retry')
        # This is the baseline's existing coherent collage-coordinate repair,
        # applied uniformly before interpreting roles. No pixels are clamped.
        choices,repair=normalize_sheet_choices(json.loads(text.strip().removeprefix('```json').removesuffix('```').strip()))
        if repair:write_json(output/'coordinate-repair.json',repair)
        prepare_contact(perception,bundles[choices['camera']],choices,output)
        return choices
