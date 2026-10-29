"""Reusable sensor-only tools for caller-selected parts and observed motion.

Registry IDs reference immutable captures in one episode. All operations are
read-only with respect to robot controls, plans and the planning scene.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .part_perception import (MobileSAMBackend,association,box_of,checked_folder,
                              compare_projected_mask,overlay,usable_pixels)
from .part_motion import (CoTrackerBackend,estimate_and_verify,fit_upright,
                          lift_tracks,lk_tracks,pair_points,query_points)
from .landmark_evidence import summarize_pairs


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


class PerceptionTools:
    def __init__(self,observations,output,workspace,*,attempt=1,segmenter=None,tracker=None,video_root=None):
        self.observations=observations;self.output=Path(output);self.workspace=Path(workspace)
        self.output.mkdir(parents=True,exist_ok=True);self.attempt=attempt
        self.segmenter=segmenter or MobileSAMBackend();self.tracker=tracker or CoTrackerBackend()
        self.video_root=Path(video_root) if video_root is not None else None
        self.parts={};self.associations={};self.tracks={};self.estimates={};self.cache=OrderedDict()

    def _record(self,observation_id):
        record=self.observations.get(observation_id)
        if record is None or record.get('attempt',self.attempt)!=self.attempt:
            raise ValueError('Use an observation ID from this episode')
        return record

    def _view(self,observation_id,camera):
        record=self._record(observation_id)
        if camera not in record['cameras']:raise ValueError('Camera is not present in the requested observation')
        cache_key=(observation_id,camera)
        if cache_key in self.cache:self.cache.move_to_end(cache_key);return self.cache[cache_key]
        sensors=Path(record['sensors']);calibration=Path(record.get('calibration',sensors.with_name('calibration.json')))
        for path,checksum_key in [(sensors,'sensor_sha256'),(calibration,'calibration_sha256')]:
            if checksum_key in record and hashlib.sha256(path.read_bytes()).hexdigest()!=record[checksum_key]:
                raise ValueError('Recorded sensor/calibration file changed after capture')
        cal=json.loads(calibration.read_text())[camera]
        with np.load(sensors,allow_pickle=False) as arrays:
            view={**cal,**{name:arrays[camera+'_'+name] for name in ['rgb','depth','world_points','known_robot_depth_prior'] if camera+'_'+name in arrays}}
        if np.asarray(view['world_points']).shape!=np.asarray(view['rgb']).shape:
            raise ValueError('RGB and metric world-point shapes disagree')
        self.cache[cache_key]=view
        while len(self.cache)>16:self.cache.popitem(last=False)
        return view

    def _relative(self,path):return str(Path(path).relative_to(self.workspace))

    def segment_parts(self,event_id,observation_id,camera,points_xy=None,point_labels=None,box_xyxy=None,label=None):
        began=time.perf_counter();view=self._view(observation_id,camera)
        height,width=view['rgb'].shape[:2];points=[] if points_xy is None else points_xy
        labels=[] if point_labels is None else point_labels
        if not points and box_xyxy is None:raise ValueError('Give a coarse RGB box or one or more positive/negative clicks')
        if len(points)!=len(labels) or any(v not in [0,1] for v in labels):raise ValueError('Supply one 0/1 label for each clicked point')
        if points and not any(labels) and box_xyxy is None:raise ValueError('A positive point or a box is needed to select a part')
        if any(not(0<=x<width and 0<=y<height) for x,y in points):raise ValueError('Clicks must use original image pixels')
        if box_xyxy is not None:
            x0,y0,x1,y1=box_xyxy
            if not(0<=x0<x1<width and 0<=y0<y1<height):raise ValueError('Use an ordered box inside the original image')
        folder=checked_folder(self.output,f'{event_id:03d}-segment');loaded=time.perf_counter()
        masks,scores,timing=self.segmenter.predict(view,(observation_id,camera),points,labels,box_xyxy)
        proposals=[];valid=usable_pixels(view)
        for index,(mask,score) in enumerate(zip(masks,scores)):
            mask=np.asarray(mask,bool)
            if mask.shape!=valid.shape or not np.isfinite(score):raise ValueError('Segmentation returned invalid mask/score')
            selected=mask&valid;identifier=f'part{event_id:03d}-{index}'
            path=folder/(identifier+'.npz');np.savez_compressed(path,raw_mask=mask,usable_mask=selected)
            picture=folder/(identifier+'.png');overlay(picture,view,selected,points=points)
            result={'part_id':identifier,'caller_label':label,'observation_id':observation_id,'camera':camera,
                'mask_box_xyxy':box_of(mask),'raw_pixels':int(mask.sum()),'usable_depth_pixels':int(selected.sum()),
                'masked_robot_or_invalid_pixels':int((mask&~valid).sum()),'sam_mask_score':float(score),
                'mask_file':self._relative(path),'overlay':self._relative(picture),
                'semantic_identity_verified':False}
            self.parts[identifier]={'observation_id':observation_id,'camera':camera,'mask':selected,'summary':result}
            proposals.append(result)
        response={'status':'mask_proposals','parts':proposals,'images':[p['overlay'] for p in proposals],
            'timing_seconds':{'observation_load':loaded-began,**timing,'total':time.perf_counter()-began},
            'robot_motion_executed':False,'planner_model_updated':False,
            'scope':'MobileSAM proposals from caller RGB prompts; filtered only by valid depth and known robot geometry. Scores do not certify physical part identity.'}
        save(folder/'result.json',response);return response

    def associate_views(self,event_id,part_id,cameras,comparison_part_id=None):
        began=time.perf_counter()
        if part_id not in self.parts:raise ValueError('Unknown part ID in this episode')
        part=self.parts[part_id];oid=part['observation_id'];source=self._view(oid,part['camera'])
        points=source['world_points'][part['mask']]
        if len(points)<20:raise ValueError('Selected part has insufficient valid non-robot depth')
        if len(set(cameras))!=len(cameras):raise ValueError('Choose distinct target cameras')
        comparison=None
        if comparison_part_id is not None:
            comparison=self.parts.get(comparison_part_id)
            if comparison is None or comparison['observation_id']!=oid or comparison['camera'] not in cameras:
                raise ValueError('Comparison part must be from a requested camera in the SAME observation')
        folder=checked_folder(self.output,f'{event_id:03d}-association');masks={part['camera']:part['mask']};reports=[];images=[]
        for name in cameras:
            view=self._view(oid,name);mask,report=association(points,view);masks[name]=mask
            picture=folder/(name+'.png');overlay(picture,view,mask);images.append(self._relative(picture))
            row={'camera':name,**report}
            if comparison is not None and comparison['camera']==name:
                row['comparison']=compare_projected_mask(mask,comparison['mask'])
            reports.append(row)
        identifier=f'association{event_id:03d}'
        self.associations[identifier]={'part_id':part_id,'observation_id':oid,'masks':masks}
        np.savez_compressed(folder/'masks.npz',**masks)
        response={'association_id':identifier,'part_id':part_id,'observation_id':oid,'views':reports,'images':images,
            'mask_file':self._relative(folder/'masks.npz'),'timing_seconds':{'total':time.perf_counter()-began},
            'robot_motion_executed':False,'planner_model_updated':False,'semantic_identity_verified':False,
            'scope':'Same-time cross-camera correspondence from measured surface points and 4mm depth agreement. Occluded/unsupported projections are not filled in.'}
        save(folder/'result.json',response);return response

    def track_part_motion(self,event_id,part_id,observation_ids,backend='lk',association_id=None,cameras=None):
        began=time.perf_counter()
        if part_id not in self.parts:raise ValueError('Unknown part ID in this episode')
        part=self.parts[part_id]
        if len(observation_ids)<2 or observation_ids!=sorted(set(observation_ids)) or part['observation_id'] not in observation_ids:
            raise ValueError('Use ordered distinct observation IDs CONTAINING the selected part capture; tracking runs around that reference')
        reference_index=observation_ids.index(part['observation_id'])
        if backend not in ['lk','cotracker3']:raise ValueError('Choose lk or cotracker3')
        for oid in observation_ids:self._record(oid)
        masks={part['camera']:part['mask']}
        if association_id is not None:
            value=self.associations.get(association_id)
            if value is None or value['part_id']!=part_id:raise ValueError('Association must belong to this part')
            masks=value['masks']
        names=cameras or [part['camera']]
        if len(set(names))!=len(names) or any(n not in masks for n in names):
            raise ValueError('Tracking cameras need masks from this part or its association')
        folder=checked_folder(self.output,f'{event_id:03d}-tracks');all_world=[];all_valid=[];reports=[];raw={};timing={};images=[]
        for name in names:
            start=time.perf_counter();views=[self._view(oid,name) for oid in observation_ids]
            if any(v['rgb'].shape!=views[0]['rgb'].shape for v in views):raise ValueError('Tracking requires matching image sizes within each camera')
            timing[name+'_load']=time.perf_counter()-start;start=time.perf_counter()
            queries=query_points(views[reference_index],masks[name]);timing[name+'_queries']=time.perf_counter()-start;start=time.perf_counter()
            if backend=='lk':tracks,visible=lk_tracks(views,queries,reference_index);backend_timing={}
            else:tracks,visible,backend_timing=self.tracker.predict(views,queries,reference_index)
            timing[name+'_tracking']=time.perf_counter()-start
            world,valid=lift_tracks(views,tracks,visible)
            all_world.append(world);all_valid.append(valid)
            raw.update({name+'_pixels':tracks,name+'_tracker_visible':visible,name+'_depth_valid':valid,name+'_queries':queries})
            # Saved observations with measured visible tracks, not a predicted
            # hinge overlay. Keep at most four image paths in the compact reply.
            if len(images)<4:
                endpoint=0 if reference_index>=len(views)/2 else len(views)-1
                for frame,label in [(reference_index,'reference'),(endpoint,'tracked')]:
                    picture=folder/(name+f'-obs{observation_ids[frame]:03d}-'+label+'.png')
                    shown=tracks[frame,valid[frame]]
                    overlay(picture,views[frame],np.zeros(views[frame]['rgb'].shape[:2],bool),points=shown)
                    images.append(self._relative(picture))
            reports.append({'camera':name,'query_points':len(queries),'reference_observation_id':part['observation_id'],'visible_depth_points_per_frame':valid.sum(1).tolist(),
                            'backend_timing_seconds':backend_timing})
        world=np.concatenate(all_world,axis=1);valid=np.concatenate(all_valid,axis=1)
        identifier=f'tracks{event_id:03d}';path=folder/'tracks.npz'
        np.savez_compressed(path,world_points=world,valid=valid,reference_frame_index=reference_index,**raw)
        self.tracks[identifier]={'part_id':part_id,'observation_ids':observation_ids,'reference_observation_id':part['observation_id'],
                                 'world':world,'valid':valid,'backend':backend}
        pairs=[]
        for i in range(len(observation_ids)-1):
            fit=fit_upright(world,valid,i,i+1)
            first,second=pair_points(world,valid,i,i+1)
            pairs.append({'observations':observation_ids[i:i+2],'finite_visible_pairs':fit['finite_visible_pairs'],
                          'motion_fit_accepted':fit['accepted'],'reason':fit.get('reason'),
                          'fit_correspondences':fit.get('correspondences',0),'fit_inliers':fit.get('inliers',0),
                          'fit_rms_m':fit.get('rms_m'),'fit_failed_checks':fit.get('failed_checks',[]),
                          'measured_rotation_deg':fit['measured_rotation_deg'],
                          'landmark_displacements':summarize_pairs(first,second)})
        response={'track_id':identifier,'part_id':part_id,'observation_ids':observation_ids,'tracking_backend':backend,
            'reference_observation_id':part['observation_id'],'reference_frame_index':reference_index,
            'camera_reports':reports,'interval_motion':pairs,'track_file':self._relative(path),'images':images,
            'timing_seconds':{**timing,'total':time.perf_counter()-began},'robot_motion_executed':False,'planner_model_updated':False,
            'scope':'RGB tracks lifted with each frame\'s measured depth/calibration. Invisible, invalid-depth and known-robot pixels are excluded. No hand-motion substitution.'}
        save(folder/'result.json',response);return response

    def track_video_motion(self,event_id,part_id,observation_ids,association_id=None,max_rgb_frames=64):
        """Track recorded RGB; lift only the caller's measured depth anchors."""
        from .observed_video import recorded_clip
        began=time.perf_counter()
        if self.video_root is None:raise ValueError('Live recorded RGB is unavailable; use track_part_motion on saved RGB-D captures')
        if part_id not in self.parts:raise ValueError('Unknown part ID in this episode')
        part=self.parts[part_id]
        if len(observation_ids)<2 or observation_ids!=sorted(set(observation_ids)) or part['observation_id'] not in observation_ids:
            raise ValueError('Use ordered distinct observation IDs containing the selected part capture')
        records=[self._record(oid) for oid in observation_ids]
        if any('control_steps' not in row for row in records):raise ValueError('Recorded depth captures require exact control-step timestamps')
        steps=[int(row['control_steps']) for row in records]
        if steps!=sorted(steps):raise ValueError('Depth capture control steps must be chronological')
        mask=part['mask'] if part['camera']=='agentview' else None
        if association_id is not None:
            association=self.associations.get(association_id)
            if association is None or association['part_id']!=part_id:raise ValueError('Association must belong to this part')
            mask=association['masks'].get('agentview')
        if mask is None:raise ValueError('Select the part in agentview or associate its mask into agentview first')
        folder=checked_folder(self.output,f'{event_id:03d}-video-tracks')
        anchor_views=[self._view(oid,'agentview') for oid in observation_ids]
        reference=observation_ids.index(part['observation_id'])
        queries=query_points(anchor_views[reference],mask)
        loaded=time.perf_counter()
        rgb_views,frames,sampling=recorded_clip(self.video_root,steps,max_rgb_frames,folder/'rgb',self.workspace)
        if any(v['rgb'].shape!=anchor_views[0]['rgb'].shape for v in rgb_views):raise ValueError('Video and depth capture image sizes disagree')
        rgb_steps=[r['control_step'] for r in frames];anchors=[rgb_steps.index(step) for step in steps]
        save(folder/'video-frames.json',{'camera':'agentview','frames':frames,'depth_control_steps':steps,'anchor_frame_indices':anchors,**sampling})
        decoded=time.perf_counter()
        pixels,visible,backend_timing=self.tracker.predict(rgb_views,queries,anchors[reference])
        tracked=time.perf_counter()
        world,valid=lift_tracks(anchor_views,pixels[anchors],visible[anchors])
        lifted=time.perf_counter();identifier=f'tracks{event_id:03d}';path=folder/'tracks.npz'
        np.savez_compressed(path,world_points=world,valid=valid,reference_frame_index=reference,
            agentview_pixels=pixels[anchors],agentview_tracker_visible=visible[anchors],agentview_depth_valid=valid,agentview_queries=queries,
            video_pixels=pixels,video_tracker_visible=visible,video_control_steps=rgb_steps,anchor_frame_indices=anchors)
        self.tracks[identifier]={'part_id':part_id,'observation_ids':observation_ids,'reference_observation_id':part['observation_id'],
            'world':world,'valid':valid,'backend':'cotracker3_video','control_steps':steps}
        images=[]
        for index,label in [(reference,'reference'),(0 if reference>=len(anchor_views)/2 else len(anchor_views)-1,'tracked')]:
            picture=folder/f'agentview-obs{observation_ids[index]:03d}-{label}.png'
            shown={**anchor_views[index],'rgb':rgb_views[anchors[index]]['rgb']}
            overlay(picture,shown,np.zeros(mask.shape,bool),points=pixels[anchors[index],valid[index]])
            images.append(self._relative(picture))
        pairs=[]
        for i in range(len(observation_ids)-1):
            fit=fit_upright(world,valid,i,i+1);first,second=pair_points(world,valid,i,i+1)
            pairs.append({'observations':observation_ids[i:i+2],'finite_visible_pairs':fit['finite_visible_pairs'],
                'motion_fit_accepted':fit['accepted'],'reason':fit.get('reason'),
                'fit_correspondences':fit.get('correspondences',0),'fit_inliers':fit.get('inliers',0),
                'fit_rms_m':fit.get('rms_m'),'fit_failed_checks':fit.get('failed_checks',[]),
                'measured_rotation_deg':fit['measured_rotation_deg'],'landmark_displacements':summarize_pairs(first,second)})
        response={'track_id':identifier,'part_id':part_id,'observation_ids':observation_ids,'tracking_backend':'cotracker3_video',
            'reference_observation_id':part['observation_id'],'reference_frame_index':reference,
            'video_reference_frame_index':anchors[reference],'video_sampling':sampling,
            'depth_control_steps':steps,'video_manifest':self._relative(folder/'video-frames.json'),
            'camera_reports':[{'camera':'agentview','query_points':len(queries),'reference_observation_id':part['observation_id'],
                'visible_depth_points_per_frame':valid.sum(1).tolist(),'rgb_frames':len(frames),'depth_anchors':len(anchor_views),
                'backend_timing_seconds':backend_timing}],
            'interval_motion':pairs,'track_file':self._relative(path),'images':images,
            'timing_seconds':{'observation_and_queries':loaded-began,'video_copy_decode':decoded-loaded,
                'tracking':tracked-decoded,'depth_lift':lifted-tracked,'total':time.perf_counter()-began},
            'robot_motion_executed':False,'planner_model_updated':False,
            'scope':'Recorded agentview RGB only up to the last supplied capture. 3D points use only measured depth anchors; no new render, interpolated depth or hand-motion substitution.'}
        save(folder/'result.json',response);return response

    def estimate_and_verify_joint(self,event_id,track_id,fit_observations,verify_observations=None):
        began=time.perf_counter()
        if track_id not in self.tracks:raise ValueError('Unknown track ID in this episode')
        track=self.tracks[track_id];ids=track['observation_ids']
        def indices(pair):
            if len(pair)!=2 or pair[0]>=pair[1] or any(p not in ids for p in pair):
                raise ValueError('Choose an ordered pair of observation IDs present in the track')
            return [ids.index(p) for p in pair]
        fit_indices=indices(fit_observations);verify_indices=indices(verify_observations) if verify_observations is not None else None
        result=estimate_and_verify(track['world'],track['valid'],fit_indices,verify_indices)
        folder=checked_folder(self.output,f'{event_id:03d}-joint');identifier=f'joint{event_id:03d}'
        result.update(estimate_id=identifier,track_id=track_id,part_id=track['part_id'],
            fit_observations=fit_observations,verify_observations=verify_observations,
            verification_observation_excluded_from_axis_fit=True,
            joint_type_scope='upright revolute hypothesis; not a general joint-type classifier',
            axis_sign_scope='unsigned world-Z axis; observed rotation does not establish the requested task direction',
            semantic_identity_verified=False,physical_contact_verified=False,
            timing_seconds={'total':time.perf_counter()-began},
            scope='No old hinge prior. Later-interval prediction support is distinct from semantic identity and task success. Offline CoTracker jointly sees the clip; this is not statistically independent tracking data.')
        self.estimates[identifier]=result;save(folder/'result.json',result)
        result['estimate_file']=self._relative(folder/'result.json');return result

    def close(self):
        self.segmenter.close();self.tracker.close();self.cache.clear()
