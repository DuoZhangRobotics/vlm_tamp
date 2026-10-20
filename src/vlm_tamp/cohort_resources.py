"""RGB-validated grounding reuse and one bounded request per new cohort scene."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree

from .coupled_resources import CoupledResources,ROOT,api_snapshot
from .coupled_storage import read_json,write_json
from .rekep_libero.articulated_grounding import ArticulatedLedger,PROMPT,parse_choices,response_text
from .rekep_libero.api import BudgetError,request_arguments
from .rekep_libero.mechanism_execution import cached_grounding
from .rekep_libero.perception import _read_observation


class CohortLedger(ArticulatedLedger):
    """Extend the old internal scene-count bound for this user-requested cohort.

    The same durable ledger and dollar ceiling remain in force. Existing
    charges and pending reservations are never reset or transferred.
    """
    def __init__(self,directory,maximum_scenes=65):
        super().__init__(directory);self.max_calls=int(maximum_scenes)

    def _read(self):
        if not self.path.exists():raise BudgetError('Expected existing cumulative ledger')
        saved=read_json(self.path)
        prior_limit=saved.get('max_scene_requests')
        if (saved.get('version')!='rekep-libero-luna-1' or saved.get('model')!='gpt-6-luna'
                or saved.get('budget_usd')!=2. or saved.get('prior_conservative_spend_usd')!=self.prior_spend
                or type(prior_limit) is not int or prior_limit<30):
            raise BudgetError('Existing cumulative ledger or scene policy changed')
        # A later authorized cohort may have raised the shared request count.
        # Older readers preserve that limit; the same dollar cap still applies.
        self.max_calls=max(self.max_calls,prior_limit)
        if saved['max_scene_requests']!=self.max_calls:
            saved.setdefault('scene_limit_changes',[]).append({'previous':saved['max_scene_requests'],
                'current':self.max_calls,'reason':'User requested all seven prior physical tasks on five states',
                'dollar_budget_unchanged':2.})
            saved['max_scene_requests']=self.max_calls
        return saved


def prepare_contact(perception,bundle,choices,output):
    camera=bundle['cameras'][choices['camera']];source=choices['grasp_keypoint']
    x,y=np.rint(choices['grasp_pixel_xy']).astype(int);box=choices['contact_box_xyxy']
    supplemental=True
    if isinstance(source,int) and 0<=source<len(bundle['points_world']):
        sy,sx=bundle['pixels_yx'][source];supplemental=not(box[0]<=sx<=box[2] and box[1]<=sy<=box[3])
    if supplemental:
        point=camera['world_points'][y,x]
        if not np.isfinite(point).all():raise ValueError('RGB-selected contact has no measured depth')
        source=len(bundle['points_world']);bundle['points_world']=np.vstack([bundle['points_world'],point])
        bundle['pixels_yx']=np.vstack([bundle['pixels_yx'],[y,x]])
    cloud,mask,record=perception.selected_cloud(bundle,source,articulated=True,out_dir=output/'contact-region')
    yy,xx=np.indices(mask.shape)
    mask&=(xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])
    reference=camera['world_points'][mask]
    if len(reference)<30:raise ValueError('RGB contact mask lacks measured support')
    cloud=cloud[cKDTree(reference).query(cloud)[0]<=.008]
    if len(cloud)<30:raise ValueError('Predicted contact cloud is empty')
    np.save(output/'contact-cloud.npy',cloud);np.save(output/'contact-mask.npy',mask)
    record.update(supplemental_vlm_pixel_used=supplemental,selected_keypoint=int(source),source_box_from_rgb_vlm=True)
    write_json(output/'grounding.json',{'choices':choices,'source':record,'provided_inventory':False,
        'simulator_scene_state_used':False,'model':'gpt-6-luna','reasoning_effort':'high'})
    return choices


class CohortResources(CoupledResources):
    def __init__(self,output):
        super().__init__();self.output=Path(output);self.new_api_calls=0

    def ensure_perception(self):
        if self.perception is None:
            from .rekep_libero.perception import Perception
            self.perception=Perception()
        return self.perception

    def candidates(self,label,index):
        paths=[]
        if index==1:paths.append(ROOT/'results/articulated-physics-pilot'/f'{label}-state001/grounding')
        if label in {'open_microwave','close_microwave'}:
            base=ROOT/'results/coupled-planning-resumed-20261015'
            paths.append(base/f'final-development-{label.split("_")[0]}'/f'{label}-state001/grounding' if index==1 else
                base/'frozen-evaluation/runs'/f'{label}-state{index:03d}'/f'{label}-state{index:03d}/grounding')
        paths.extend(sorted(self.cache.glob(f'{label}-state{index:03d}-rgb-*')))
        return paths

    def grounding(self,label,index,observation,metadata,output):
        observation,output=Path(observation),Path(output)
        image_hash=hashlib.sha256(b''.join(np.asarray(Image.open(observation/f'{name}.png')).tobytes()
            for name in ['agentview','frontview'])).hexdigest()
        cache=self.cache/f'{label}-state{index:03d}-rgb-{image_hash}'
        rejections=[]
        for ordinal,source in enumerate([cache,*self.candidates(label,index)]):
            if not (source/'grounding.json').exists():continue
            record=read_json(source/'grounding.json')
            if record.get('provided_inventory') is not False or record.get('simulator_scene_state_used') is not False:
                continue
            attempt=output.with_name(output.name+f'-cache-{ordinal}')
            try:cached_grounding(source,observation,metadata,attempt)
            except (ValueError,FileNotFoundError) as error:
                rejections.append({'source':str(source),'reason':str(error)})
                if attempt.exists():shutil.rmtree(attempt)
                continue
            attempt.rename(output)
            write_json(output/'input-provenance.json',{'cached_source':str(source),'new_api_calls':0,
                'rgb_input_sha256':image_hash,'rgb_and_selected_depth_verified':True,'cache_rejections':rejections})
            return record['choices']
        # The earlier 12k responses have the exact original sheet but no
        # reconstructed contact bundle. Rebuild it without another API call.
        corrected=ROOT/'results/microwave-luna-12k-20261015'/f'{label}-state{index:03d}'
        if (corrected/'choices.json').exists():
            old=Path(read_json(corrected/'summary.json')['source'])
            cameras,_=_read_observation(observation/'obs.npz',metadata)
            if all(np.array_equal(cameras[n]['rgb'],np.asarray(Image.open(old/n/'rgb.png'))) for n in ['agentview','frontview']):
                output.mkdir(parents=True);choices=read_json(corrected/'choices.json');name=choices['camera']
                for n in ['agentview','frontview']:shutil.copytree(old/n,output/n)
                for f in ['prompt.txt','response.json','response.txt','role-choice-sheet.png']:
                    if (corrected/f).exists():shutil.copyfile(corrected/f,output/f)
                with np.load(old/name/'keypoints.npz') as data:bundle={k:data[k].copy() for k in data.files}
                bundle.update(cameras=cameras,meta={**metadata,'main_camera':name},main_camera=name)
                prepare_contact(self.ensure_perception(),bundle,choices,output)
                write_json(output/'input-provenance.json',{'new_api_calls':0,'cached_12k_response':str(corrected),
                    'rgb_input_sha256':image_hash,'both_rgb_views_byte_identical':True})
                if not cache.exists():shutil.copytree(output,cache)
                return choices
        output.mkdir(parents=True);perception=self.ensure_perception();bundles={}
        sheet=Image.new('RGB',(1024,1072),'white');draw=ImageDraw.Draw(sheet)
        for row,name in enumerate(['agentview','frontview']):
            bundles[name]=perception.prepare(observation/'obs.npz',{**metadata,'main_camera':name},output/name)
            for column,kind in enumerate(['rgb','numbered']):
                draw.text((column*512+10,row*536+5),f'{name}: {kind}',fill='black')
                sheet.paste(Image.open(output/name/f'{kind}.png'),(column*512,row*536+24))
        sheet.save(output/'role-choice-sheet.png');prompt=PROMPT.format(instruction=metadata['instruction'])
        (output/'prompt.txt').write_text(prompt)
        arguments=request_arguments(output/'role-choice-sheet.png',prompt,12000)
        from .cli import load_credentials
        allow_paid=getattr(self,'allow_paid',True)
        if allow_paid:load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
        ledger=self.api_ledger();before=api_snapshot(ledger)
        scene_id=f'all-prior-tasks-v1-{label}-{index:03d}-{image_hash[:16]}'
        try:raw=ledger.perform(arguments,scene_id,allow_paid=allow_paid)
        finally:
            after=api_snapshot(ledger);new=int(scene_id not in before['requests'] and scene_id in after['requests'])
            self.new_api_calls+=new
            write_json(output/'input-provenance.json',{'new_api_calls':new,'scene_id':scene_id,
                'rgb_input_sha256':image_hash,'max_output_tokens':12000,'automatic_retries':0,
                'api_run_id':ledger.run_id,'budget_scope':'experiment_run',
                'run_spend_usd':after['run_conservative_spend_usd'],
                'remaining_run_budget_usd':after['remaining_run_budget_usd'],
                'cumulative_conservative_spend_usd':after['cumulative_conservative_spend_usd']})
        write_json(output/'response.json',raw);text=response_text(raw);(output/'response.txt').write_text(text)
        if raw.get('status')!='completed':raise ValueError('Luna did not complete; no automatic retry')
        choices=parse_choices(text);prepare_contact(perception,bundles[choices['camera']],choices,output)
        if not cache.exists():shutil.copytree(output,cache)
        return choices
