"""Collision evidence and image-grounded region revision; existing motion tools."""
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw
from .recovery_episode import RecoveryTools
from .collision_evidence import capture_rejections,save_capture
from .region_grounding import revised_geometry
from .shared_mechanism_pipeline import ObservedProgress
from .coupled_storage import read_json,write_json
from .rekep_libero.mechanism_geometry import project

def projected_region(session):
    """Project the geometry actually being tracked, not old selection pixels."""
    name=session.selection['camera'];camera=session.views[name]
    h,w=camera['rgb'].shape[:2];scale=np.array([w-1,h-1],float)
    def box(points):
        uv=project(np.asarray(points).reshape(-1,3),camera)/scale
        valid=np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<=1).all(1)
        if not valid.any():return None
        lo,hi=np.quantile(uv[valid],[.01,.99],axis=0)
        return [*lo.tolist(),*hi.tolist()]
    contact=box(session.cloud)
    moving=session.scene.moving_reference
    face=moving[np.abs((moving-session.scene.panel_origin)@session.scene.panel_normal)<.008]
    panel=box(face if len(face) else moving)
    uv=project([session.spec['source_anchor_world_m']],camera)[0]/scale
    point=uv.tolist() if np.isfinite(uv).all() and (uv>=0).all() and (uv<=1).all() else None
    return {'camera':name,'contact_kind':session.selection['contact_kind'],'contact_uv':point,
        'contact_box_uv':contact,'moving_panel_box_uv':panel,
        'projection_at_control_step':int(session.state['control_steps']),
        'source':'current tracked metric geometry, projected into this camera',
        'occlusion_tested':False}

def search_coverage(folder,decision,epoch,control_steps):
    folder=Path(folder);out={'region_epoch':epoch,'control_steps_before_tool':control_steps,
        'operation':decision['operation'],'contact_mode':decision.get('contact_mode'),
        'step_mm':decision.get('step_mm'),'requested_posture_rank':decision.get('posture_rank')}
    menu=read_json(folder/'contact-menu.json')['candidates'] if (folder/'contact-menu.json').exists() else []
    admission=read_json(folder/'contact-mode-admission.json') if (folder/'contact-mode-admission.json').exists() else {}
    tested=defaultdict(list)
    for attempt in admission.get('attempts',[]):
        index=attempt.get('candidate');rank=attempt.get('ik_rank')
        if type(index) is int and type(rank) is int and 0<=index<len(menu):
            tested[str(menu[index]['approach_tilt_deg'])].append((rank,attempt.get('reason','unknown')))
    out['tested_contact_postures']=[{'tilt_deg':float(angle),'ik_ranks':sorted({r for r,_ in values}),
        'rejection_records':dict(Counter(reason for _,reason in values))} for angle,values in tested.items()]
    if (folder/'grasp-admission.json').exists():
        pinch=read_json(folder/'grasp-admission.json')
        out['pinch_search']={k:pinch[k] for k in ('tested_ik_postures','eligible_poses','posture_budget') if k in pinch}
    out['interpretation']='Tested candidates at this region/state; search failures do not prove physical impossibility.'
    return out

class EvidenceTools(RecoveryTools):
    """Logging-only control: the v2 policy still receives its original feedback."""
    def prepare(self):
        super().prepare();self.region_epoch=0;self.last_collision=None;self.coverage=[]
    def dispatch(self,decision,folder):
        before=int(self.state['control_steps']);epoch=self.region_epoch
        with capture_rejections() as capture:
            result=super().dispatch(decision,folder)
        diagnostics=save_capture(capture,self.views,folder/'collision-evidence')
        self.coverage.append(search_coverage(folder,decision,epoch,before))
        if diagnostics['representative_samples']:
            self.last_collision={'folder':str(folder/'collision-evidence'),'epoch':epoch,**diagnostics}
        write_json(self.folder/'search-coverage.json',self.coverage)
        return result

class RegionTools(EvidenceTools):
    def prepare(self):
        super().prepare()
        choices=read_json(self.folder/'grounding/grounding.json')['choices']
        self.selection={'camera':choices['camera'],'contact_kind':'bar'}
        w=self.views[choices['camera']]['rgb'].shape[1]-1;h=self.views[choices['camera']]['rgb'].shape[0]-1
        self.selection.update(contact_uv=(np.asarray(choices['grasp_pixel_xy'])/[w,h]).tolist(),
            contact_box_uv=(np.asarray(choices['contact_box_xyxy'])/[w,h,w,h]).tolist(),
            moving_panel_box_uv=(np.asarray(choices['moving_panel_box_xyxy'])/[w,h,w,h]).tolist())
    def dispatch(self,decision,folder):
        if decision['operation']!='revise_region':
            if decision['operation']=='try_contact' and decision['contact_mode']=='press_drag' and self.selection['contact_kind']=='surface':
                return {'status':'contact_mode_incompatible','manipulation_executed':False,
                    'reason':'The press_drag tool uses a raised bar; the current revision selected a surface patch.'}
            old={k:v for k,v in decision.items() if k!='region'};old['version']='strategy-v2'
            return super().dispatch(old,folder)
        if self.current_choice is not None:
            return {'status':'region_revision_requires_release','manipulation_executed':False,
                'region_epoch':self.region_epoch,'reason':'Release active contact before revising the selected region.'}
        self.stage='region_revision';output=folder/'region-revision';output.mkdir()
        try:
            spec,cloud,panel,scene=revised_geometry(self,decision['region'],output)
            generated=self.resources.grasps(cloud,output/'grasps')
        except (ValueError,RuntimeError) as error:
            write_json(output/'rejection.json',{'reason':str(error),'previous_region_retained':True})
            return {'status':'region_revision_rejected','manipulation_executed':False,
                'region_epoch':self.region_epoch,'reason':str(error),'previous_region_retained':True}
        previous={'region_epoch':self.region_epoch,'spec':self.spec,'progress_m':self.progress.progress}
        self.spec,self.cloud,self.panel,self.scene,self.generated=spec,cloud,panel,scene,generated
        self.region_epoch+=1;self.selection=decision['region'];self.progress=ObservedProgress(spec['joint_axis_world'])
        self.goal_reached=False;self.last_collision=None;self.latest_pose=None;self.geometry=self.geometry_record()
        self.sim.context=None;self.sim.depth_reference=None;self.sim.depth_hand=None;self.sim.visual_offset=np.zeros(3)
        write_json(output/'committed.json',{'previous':previous,'new_region_epoch':self.region_epoch,
            'remaining_travel_m':spec['target_progress'],'progress_reference_reset':True,'robot_motion_executed':False})
        write_json(self.folder/'current-task-spec.json',self.spec)
        write_json(self.folder/'current-region.json',{'region_epoch':self.region_epoch,'selection':self.selection,
            'initial_grounding_superseded':True,'revision_folder':str(output),
            'selection_at_control_step':int(self.state['control_steps']),
            'coordinates_reference':'RGB at revision time; current tracked projections are in each feedback.json'})
        return {'status':'region_revised','manipulation_executed':False,'region_epoch':self.region_epoch,
            'remaining_travel_m':spec['target_progress'],'progress_reference_reset':True,
            'camera':self.selection['camera'],'contact_kind':self.selection['contact_kind'],
            'reason':'New RGB region passed segmentation, depth support and geometry fitting.'}
    def extend_feedback(self,packet):
        packet['version']='strategy-feedback-v3';packet['current_region']=projected_region(self)
        packet['region_epoch']=self.region_epoch;packet['can_revise_region']=self.current_choice is None
        packet['search_coverage']=self.coverage[-8:]
        packet['collision_evidence']=None
        if self.last_collision:
            packet['collision_evidence']={'region_epoch':self.last_collision['epoch'],
                'representative_samples':self.last_collision['representative_samples'],
                'native_validation_rejections_observed':self.last_collision['native_validation_rejections_observed'],
                'image_time':'RGB observation used for the rejected plan; it may precede current RGB'}
        for out,entry in zip(packet['history'],self.history):
            out['decision']['region']=entry['decision'].get('region')
            out['result'].pop('solver_diagnostics',None) # cumulative native counters are not independent events
            for key in ('region_epoch','remaining_travel_m','progress_reference_reset','camera','contact_kind','previous_region_retained'):
                if key in entry['result']:out['result'][key]=entry['result'][key]
        return packet
    def render_feedback(self,output):
        sheet=Image.new('RGB',(1024,1120),'white');draw=ImageDraw.Draw(sheet)
        selection=projected_region(self)
        for column,name in enumerate(('agentview','frontview')):
            camera=self.views[name];view=Image.fromarray(np.asarray(camera['rgb'],np.uint8)).resize((512,512))
            overlay=ImageDraw.Draw(view)
            for value in (.25,.5,.75):
                pos=int(value*511);overlay.line((pos,0,pos,511),fill=(160,160,160),width=1)
                overlay.line((0,pos,511,pos),fill=(160,160,160),width=1)
                overlay.text((pos+2,2),str(value),fill='white',stroke_width=1,stroke_fill='black')
                overlay.text((2,pos+2),str(value),fill='white',stroke_width=1,stroke_fill='black')
            if name==selection['camera']:
                for key,color in (('moving_panel_box_uv','cyan'),('contact_box_uv','orange')):
                    if selection[key] is not None:
                        box=(np.asarray(selection[key])*511).tolist();overlay.rectangle(box,outline=color,width=2)
                if selection['contact_uv'] is not None:
                    u,v=np.asarray(selection['contact_uv'])*511
                    overlay.ellipse((u-4,v-4,u+4,v+4),outline='orange',width=2)
            origin=np.median(self.cloud,axis=0);axis=np.asarray(self.spec['joint_axis_world'])
            uv=project([origin,origin+.05*axis],camera)
            if np.isfinite(uv).all():overlay.line([tuple(p) for p in uv],fill='yellow',width=3)
            draw.text((column*512+8,6),f'CURRENT {name}: normalized camera-local u/right, v/down',fill='black')
            sheet.paste(view,(column*512,24))
            if self.last_collision and self.last_collision['representative_samples']:
                sample=self.last_collision['representative_samples'][-1]
                path=Path(self.last_collision['folder'])/f"sample-{sample['sample']:02d}"/f'{name}.png'
                preview=Image.open(path).convert('RGB').resize((512,512));sheet.paste(preview,(column*512,584))
                draw.text((column*512+8,548),f"REJECTED PLAN: {sample['robot_part']}",fill='black')
                draw.text((column*512+8,564),f"{sample['surface_role']}; overlap verified: {sample['witness_verified']}",fill='black')
            else:draw.text((column*512+30,720),'No rejected pose preview in this region epoch.',fill='black')
        draw.text((8,1103),'Top: current RGB. Bottom: hypothetical rejected robot pose on the observation used for planning. No simulator object truth.',fill='black')
        sheet.save(output)
