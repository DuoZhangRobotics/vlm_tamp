"""Refit a VLM-selected contact and moving panel using current RGB-D only."""
import numpy as np
from scipy.spatial import cKDTree
from .region_decision import pixels
from .coupled_storage import write_json
from .cohort_resources import prepare_contact
from .shared_contact_geometry import grounded_geometry,closing_target,seed_component
from .rekep_libero.scene_state_v7 import observed_points,voxel_points
from .reliability_geometry import SurfaceScene

def revised_geometry(session,region,folder):
    name=region['camera'];camera=session.views[name];height,width=camera['rgb'].shape[:2]
    point,contact_box,panel_box=pixels(region,width,height);x,y=point
    anchor=np.asarray(camera['world_points'][y,x],float)
    if not np.isfinite(anchor).all() or camera['depth'][y,x]<=0:raise ValueError('Selected pixel has no measured depth')
    cloud_points,_=observed_points(session.views,session.state['metadata'],spacing=.003,
        robot_depth_tolerance=session.config['robot_self_depth_tolerance_m'])
    if len(cloud_points)<30 or cKDTree(cloud_points).query(anchor)[0]>.006:
        raise ValueError('Selected pixel lacks non-robot measured support')
    choices={'camera':name,'grasp_keypoint':None,'grasp_pixel_xy':point,'contact_box_xyxy':contact_box,
        'moving_panel_box_xyxy':panel_box,'joint_family':session.spec['joint_family'],'motion':session.spec['motion'],
        'travel_m':float(session.spec['target_progress']),'axis_source':'panel_normal'}
    # A point-prompted RGB mask avoids relying on stale feature IDs after motion.
    # DINO's original proposer is retained for initialization, not used as an oracle.
    bundle={'points_world':np.empty((0,3)),'pixels_yx':np.empty((0,2),int),'masks':[],
        'cameras':session.views,'main_camera':name,'meta':{**session.state['metadata'],'instruction':session.spec['instruction']}}
    prepare_contact(session.resources.ensure_perception(),bundle,choices,folder)
    cloud=np.load(folder/'contact-cloud.npy');mask=np.load(folder/'contact-mask.npy')
    # Keep a bounded patch around the requested point. A region change cannot
    # grant the fingers contact permission across an entire object or cabinet.
    cloud=cloud[np.linalg.norm(cloud-anchor,axis=1)<=.045]
    if len(cloud)<30:raise ValueError('Selected contact patch has insufficient metric support')
    mechanism,panel=grounded_geometry(bundle,choices,cloud,mask,session.resources.ensure_perception(),folder)
    spec=mechanism.spec
    spec['contact_kind']=region['contact_kind']
    if region['contact_kind']=='bar':
        normal=np.asarray(spec['panel_normal_world']);origin=np.asarray(spec['front_plane']['point_world_m'])
        raised=cloud[((cloud-origin)@normal>.004)&((cloud-origin)@normal<.06)]
        if len(raised)<30:raise ValueError('Selected bar lacks a measured raised component; surface contact remains a separate choice')
        cloud=seed_component(raised,anchor)
    else:
        normal=np.asarray(spec['panel_normal_world'])
        cloud=cloud[np.abs((cloud-anchor)@normal)<.008]
    cloud=voxel_points(cloud,.002)
    if len(cloud)<30 or np.linalg.norm(np.ptp(cloud,axis=0))<.012:
        raise ValueError('Selected contact lacks the required metric extent')
    spec['source_anchor_world_m']=np.median(cloud,axis=0).tolist()
    if spec['motion']=='close':spec=closing_target(session.views,session.state['metadata'],spec,cloud,session.config)
    spec['region_selection_source']='VLM camera-local normalized coordinates, current RGB MobileSAM and metric depth'
    scene=SurfaceScene(session.views,session.state['metadata'],spec,cloud,panel,session.config)
    np.save(folder/'grasp-cloud.npy',cloud);write_json(folder/'task-spec.json',spec)
    write_json(folder/'selection.json',{'request':region,'camera_pixels':{'contact':point,'contact_box':contact_box,'panel_box':panel_box},
        'contact_patch_radius_m':.045,'points':len(cloud),'simulator_object_truth_used':False,
        'human_supplied_replacement_coordinates':False,'collision_policy_changed':False})
    return spec,cloud,panel,scene
