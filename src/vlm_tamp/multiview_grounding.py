"""Upright-door geometry from caller-selected RGB regions and measured RGB-D.

No task IDs, stored annotations, native object geometry, or model calls. Camera
calibration is an input. Agreement is a consistency screen, not semantic truth.
"""
import copy
import time
from pathlib import Path

import numpy as np

from .robot_contracts import validate_operation, observed_image_sizes
from .sensor_geometry_check import triangulate_vertical_hinge, cross_view_fit_agreement


def fit_multiview(cameras, metadata, task, arguments):
    """Return a candidate plus public checks; never mutate a planning service."""
    validate_operation('ground_multiview',arguments,observed_image_sizes(cameras))
    from .hinged_contact_tools import ground_handle
    from .handle_contact import axis_from_bulk, image_motion_axis
    began=time.perf_counter();phases={};fits=[];candidate=None
    review={'accepted':False,'status':'unresolved','failed_checks':[],
        'measurements':{},'observation_id':arguments['observation_id'],
        'calibration_source':'existing observation calibration; not estimated by caller or tool',
        'semantic_identity_verified':False,'simulator_object_geometry_used':False,
        'scope':'Upright hinge and vertical handle prior. Cross-view agreement does not prove part identity, hidden hinge accuracy, contact retention or task success.'}
    stage='primary_depth_fit'
    try:
        for name in ['regions','comparison_regions']:
            stage='primary_depth_fit' if name=='regions' else 'comparison_depth_fit'
            started=time.perf_counter();choices=copy.deepcopy(arguments[name])
            choices.update(motion=task['motion'],joint_family='revolute',axis_source='vertical_hinge')
            template={'schema_version':'articulated-spec-1.0','task_type':'articulated',
                'camera':choices['camera'],'instruction':task['instruction'],
                'joint_family':'revolute','motion':task['motion'],'target_progress':float(np.pi/2),
                'progress_units':'radians','axis_prior':'upright door',
                'contact_policy':'selected functional finger contact',
                'constraint':{'type':'point_on_circle_with_coupled_orientation'}}
            try:
                mechanism,cloud,mask,anchors,evidence=ground_handle(cameras,choices,template,metadata,allow_direction_conflict=True)
            finally:phases[stage]=time.perf_counter()-started
            fits.append({'spec':mechanism.spec,'cloud':cloud,'mask':mask,'anchors':anchors,
                         'evidence':evidence,'choices':choices})
        stage='cross_view_geometry';started=time.perf_counter()
        annotations={v['camera']:v['hinge_pixels_xy'] for v in
                     [arguments['regions'],arguments['comparison_regions'],arguments['check_view']]}
        hinge=triangulate_vertical_hinge(cameras,annotations,
            fit_cameras=[f['choices']['camera'] for f in fits],
            reference_z_m=fits[0]['spec']['source_anchor_world_m'][2])
        agreement=cross_view_fit_agreement(fits[0]['spec'],fits[1]['spec'])
        measurements={**agreement,'view_geometry_condition_number':hinge['condition_number'],
            'hinge_projection_error_px':{name:v['hinge_line_reprojection_error_px'] for name,v in hinge['views'].items()},
            'edge_verticality_error_px':{name:v['annotation_verticality_error_px'] for name,v in hinge['views'].items()}}
        checks={'handle_centres_within_10mm':agreement['handle_center_xy_difference_mm']<=10,
                'panel_normals_within_5deg':agreement['signed_panel_normal_difference_deg']<=5,
                'hinge_projections_within_4px':all(v['hinge_line_reprojection_error_px']<=4 for v in hinge['views'].values()),
                'annotated_edges_within_3px_of_vertical':all(v['annotation_verticality_error_px']<=3 for v in hinge['views'].values())}
        candidate=copy.copy(fits[0]);candidate['spec']=copy.deepcopy(fits[0]['spec'])
        spec=candidate['spec'];spec['joint_origin_world_m']=hinge['origin_world_m']
        spec['observed_handle_radius_m']=candidate['evidence']['cylinder_radius_m']
        phases[stage]=time.perf_counter()-started
        review.update(measurements=measurements,checks=checks,triangulation=hinge,
                      failed_checks=[key for key,value in checks.items() if not value])
        stage='motion_direction';direction=copy.deepcopy(candidate['evidence'].get('motion_direction_check',{}))
        bulk=direction.get('observed_bulk_world_m')
        axis,scores=image_motion_axis(spec['source_anchor_world_m'],np.asarray(hinge['origin_world_m']),
            cameras[arguments['regions']['camera']],arguments['regions']['initial_image_direction_xy'])
        if bulk is not None:
            depth_axis,score=axis_from_bulk(spec['source_anchor_world_m'],np.asarray(hinge['origin_world_m']),bulk,task['motion'])
            if axis is None:axis=depth_axis
            direction.update(positive_axis_toward_bulk_score_m=score,
                depth_prior_axis=depth_axis.tolist(),direction_conflict=bool(not np.allclose(axis,depth_axis)),
                task_direction_verified=False,
                source='caller image hint when supplied; adjacent-depth prior otherwise, using triangulated hinge')
        else:
            axis,scores=image_motion_axis(spec['source_anchor_world_m'],np.asarray(hinge['origin_world_m']),
                                         cameras[arguments['regions']['camera']],arguments['regions']['initial_image_direction_xy'])
            if axis is None:raise ValueError('Current sensor geometry and image hint do not resolve the rotation direction')
            direction.update(source='caller image-motion hint using triangulated hinge',projection_scores=scores)
        spec['joint_axis_world']=axis.tolist();direction['selected_axis']=axis.tolist()
        direction['vlm_image_axis']=axis.tolist() if arguments['regions']['initial_image_direction_xy'] is not None else None
        from .handle_contact import require_direction_agreement
        require_direction_agreement({'motion_direction_check':direction})
        spec['direction_source']=direction
        spec['geometry_source']='caller-selected RGB regions; calibrated multiview hinge triangulation and independent depth fits'
        spec['geometry_review']={'source':'multiview sensor consistency','semantic_identity_verified':False,
                                 'simulator_object_geometry_used':False}
        candidate['evidence']={**candidate['evidence'],'motion_direction_check':direction}
        review.update(accepted=all(checks.values()),status='consistent' if all(checks.values()) else 'inconsistent')
    except ValueError as error:
        review.update(status='unresolved',failed_checks=[*review['failed_checks'],stage],reason=str(error))
    review['timing_seconds']={'total':time.perf_counter()-began,'phases':phases}
    # Arrays remain host-side; explicit diagnostic receipts contain only fits.
    review['depth_fits']=[{'camera':f['choices']['camera'],'model':f['spec'],'fit':f['evidence']} for f in fits]
    return candidate,review


def write_overlays(folder,cameras,arguments,candidate):
    """Deterministic annotation/geometry plots on the original camera images."""
    from PIL import Image,ImageDraw
    from .rekep_libero.mechanism_geometry import project
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);paths=[]
    for view in [arguments['regions'],arguments['comparison_regions'],arguments['check_view']]:
        name=view['camera'];image=Image.fromarray(cameras[name]['rgb']).copy();draw=ImageDraw.Draw(image)
        draw.line([tuple(p) for p in view['hinge_pixels_xy']],fill='cyan',width=3)
        if 'contact_box_xyxy' in view:
            draw.rectangle(view['contact_box_xyxy'],outline='orange',width=2)
            draw.rectangle(view['moving_panel_box_xyxy'],outline='lime',width=2)
        if candidate is not None:
            spec=candidate['spec'];origin=np.asarray(spec['joint_origin_world_m']);points=np.repeat(origin[None],2,axis=0)
            points[:,2]+=np.array([-.20,.20]);uv=project(points,cameras[name])
            if np.isfinite(uv).all():draw.line([tuple(p) for p in uv],fill='red',width=2)
            point=project([spec['source_anchor_world_m']],cameras[name])[0]
            if np.isfinite(point).all():draw.ellipse([point[0]-4,point[1]-4,point[0]+4,point[1]+4],outline='yellow',width=2)
        destination=folder/(name+'.png');image.save(destination);paths.append(destination)
    return paths


def compact_review(review):
    return {key:review[key] for key in ['accepted','status','failed_checks','measurements','observation_id',
            'reason','calibration_source','semantic_identity_verified','simulator_object_geometry_used','scope','timing_seconds'] if key in review}
