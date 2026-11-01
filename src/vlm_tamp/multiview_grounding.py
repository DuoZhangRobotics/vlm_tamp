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


def fit_multiview(cameras,metadata,task,arguments):
    """Measure disagreement; the primary caller hypothesis is never rewritten."""
    from .explicit_grounding import ground_pixels
    from .rekep_libero.mechanism_geometry import project
    validate_operation('ground_multiview',arguments,observed_image_sizes(cameras))
    began=time.perf_counter();fits=[];candidate=None
    review={'accepted':False,'status':'unresolved','failed_checks':[],
        'measurements':{},'observation_id':arguments['observation_id'],
        'calibration_source':'existing RGB-D camera calibration',
        'semantic_identity_verified':False,'simulator_object_geometry_used':False,
        'scope':'Caller-selected 3D hinge line and raw selected depth. Cross-view errors are measurements, not a hidden admission threshold.'}
    try:
        for key in ('regions','comparison_regions'):
            fits.append(ground_pixels(cameras,metadata,arguments[key]))
        candidate=fits[0];candidate['spec']['motion']=task['motion']
        a,b=[f['spec'] for f in fits]
        origin=np.asarray(a['joint_origin_world_m']);axis=np.asarray(a['joint_axis_world'])
        other=np.asarray(b['joint_axis_world'])
        measurements={'axis_line_angle_deg':float(np.degrees(np.arccos(np.clip(abs(axis@other),0.,1.)))),
            'other_origin_to_primary_line_m':float(np.linalg.norm(np.cross(np.asarray(b['joint_origin_world_m'])-origin,axis))),
            'hinge_projection_error_px':{}}
        for key in ('regions','comparison_regions','check_view'):
            annotation=arguments[key];view=cameras[annotation['camera']]
            uv=project(np.asarray([origin,origin+axis]),view)
            direction=uv[1]-uv[0];norm=np.linalg.norm(direction)
            delta=np.asarray(annotation['hinge_pixels_xy'])-uv[0]
            error=(float(np.mean(np.abs(delta[:,0]*direction[1]-delta[:,1]*direction[0])/norm))
                if norm>0 and np.isfinite(uv).all() else None)
            measurements['hinge_projection_error_px'][annotation['camera']]=error
        review.update(accepted=True,status='hypothesis_with_measurements',measurements=measurements,
            consistency_cutoff_applied=False,primary_hypothesis_adjusted=False)
    except ValueError as error:
        review.update(reason=str(error),failed_checks=['required_observed_depth_unavailable'])
    review['timing_seconds']={'total':time.perf_counter()-began}
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
        if view.get('contact_box_xyxy') is not None:
            draw.rectangle(view['contact_box_xyxy'],outline='orange',width=2)
            draw.rectangle(view['moving_panel_box_xyxy'],outline='lime',width=2)
        if candidate is not None:
            spec=candidate['spec'];origin=np.asarray(spec['joint_origin_world_m']);points=np.repeat(origin[None],2,axis=0)
            points+=np.array([-.20,.20])[:,None]*np.asarray(spec['joint_axis_world']);uv=project(points,cameras[name])
            if np.isfinite(uv).all():draw.line([tuple(p) for p in uv],fill='red',width=2)
            point=project([spec['source_anchor_world_m']],cameras[name])[0]
            if np.isfinite(point).all():draw.ellipse([point[0]-4,point[1]-4,point[0]+4,point[1]+4],outline='yellow',width=2)
        destination=folder/(name+'.png');image.save(destination);paths.append(destination)
    return paths


def compact_review(review):
    return {key:review[key] for key in ['accepted','status','failed_checks','measurements','observation_id',
            'reason','calibration_source','semantic_identity_verified','simulator_object_geometry_used','scope','timing_seconds'] if key in review}
