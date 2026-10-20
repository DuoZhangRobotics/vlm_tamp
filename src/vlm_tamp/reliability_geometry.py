"""Observed geometry repairs for contact-rich regression tasks.

All decisions use calibrated RGB-D, prior VLM regions, and robot proprioception.
No evaluation labels, simulator object poses, or joint coordinates are read.
"""
import copy
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from .mechanism_surface import SurfaceScene as OriginalSurfaceScene
from .rekep_libero.geometry import transform_points
from .rekep_libero.scene_state_v7 import observed_points,voxel_points


def local_face_mask(points,contact,origin,normal,*,radius=.065,band=.004):
    points=np.asarray(points,float).reshape(-1,3)
    if not len(points):return np.zeros(0,bool)
    return (np.abs((points-origin)@normal)<band)&(cKDTree(contact).query(points)[0]<radius)


class SurfaceScene(OriginalSurfaceScene):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # SAM sometimes selects only the handle-height strip of the same face.
        # Reconcile measured coplanar local endpoints BEFORE assigning motion.
        local=local_face_mask(self.static_points,self.contact_reference,self.panel_origin,self.panel_normal)
        added=self.static_points[local]
        self.static_points=self.static_points[~local]
        self.moving_reference=np.vstack([self.moving_reference,added])
        distance=cKDTree(self.contact_reference).query(self.moving_reference)[0]
        self.selected_contact_mask=(distance<.028)|local_face_mask(self.moving_reference,self.contact_reference,self.panel_origin,self.panel_normal)
        self.reconciled_face_points=len(added)
    def record(self):
        result=super().record();result.update(schema_version='observed-mechanism-surface-2',
            coplanar_contact_role_reconciliation=True,reconciled_face_points=self.reconciled_face_points,
            local_face_contact_radius_m=.065,local_face_plane_band_m=.004)
        return result


def refine_control_axis(cameras,choices,spec,cloud):
    """Find the raised connected control, not the center of a loose RGB box."""
    camera=cameras[spec['camera']];world=np.asarray(camera['world_points'])
    origin=np.asarray(spec['front_plane']['point_world_m']);normal=np.asarray(spec['panel_normal_world'])
    normal=normal/np.linalg.norm(normal)
    box=np.asarray(choices['moving_panel_box_xyxy'],float)
    margin=max(12.,.35*max(box[2]-box[0],box[3]-box[1]))
    yy,xx=np.indices(world.shape[:2]);height=(world-origin)@normal
    bounded=(xx>=box[0]-margin)&(xx<=box[2]+margin)&(yy>=box[1]-margin)&(yy<=box[3]+margin)
    elevated=bounded&np.isfinite(world).all(2)&(height>.012)&(height<.095)
    labels,n=ndimage.label(elevated,np.ones((3,3),int))
    options=[]
    for label in range(1,n+1):
        pixels=np.argwhere(labels==label)
        if len(pixels)<20:continue
        points=world[labels==label]
        if np.linalg.norm(np.ptp(points,axis=0))>.13:continue
        options.append((len(pixels),label))
    if not options:raise ValueError('No raised connected control in the bounded RGB-D region')
    count,label=max(options);visible=world[labels==label]
    # Fuse only observed endpoints neighboring this selected physical component.
    all_points=np.concatenate([c['world_points'].reshape(-1,3) for c in cameras.values()])
    all_points=all_points[np.isfinite(all_points).all(1)]
    near=cKDTree(visible).query(all_points)[0]<.008
    raised=all_points[near&(((all_points-origin)@normal)>.010)]
    refined=voxel_points(raised,.0015)
    side=np.cross(normal,[1,0,0] if abs(normal[0])<.8 else [0,1,0]);side/=np.linalg.norm(side)
    basis=np.column_stack([side,np.cross(normal,side)])
    xy=(refined-origin)@basis;low,high=np.quantile(xy,[.01,.99],axis=0)
    center=origin+basis@((low+high)/2)
    new=copy.deepcopy(spec);new['joint_origin_world_m']=center.tolist()
    new['source_anchor_world_m']=np.median(refined,axis=0).tolist()
    new['control_axis_refinement']={'source':'raised connected RGB-D control component',
        'original_region_xyxy':box.tolist(),'bounded_expansion_pixels':margin,
        'component_pixels':count,'contact_points':len(refined),'center_world_m':center.tolist(),
        'footprint_extent_m':(high-low).tolist(),'minimum_surface_height_m':.012,
        'simulator_object_truth_used':False}
    return new,refined


def drawer_closing_target(cameras,metadata,spec,cloud,config):
    """Estimate the closed front plane from visible cabinet faces below the drawer."""
    result=copy.deepcopy(spec);normal=np.asarray(spec['panel_normal_world'])
    origin=np.asarray(spec['front_plane']['point_world_m']);side=np.cross(normal,[0,0,1.]);side/=np.linalg.norm(side)
    points,_=observed_points(cameras,metadata,spacing=.003,robot_depth_tolerance=config['robot_self_depth_tolerance_m'])
    relative=points-origin;depth=-relative@normal
    nearby=(abs(relative@side)<.16)&(relative[:,2]<-.045)&(relative[:,2]>-.26)&(depth>.03)&(depth<.25)
    selected=points[nearby];values=depth[nearby]
    if len(values)<80:raise ValueError('Insufficient observed cabinet-face support for closing distance')
    edges=np.arange(.03,.253,.003);counts,_=np.histogram(values,edges);peak=int(np.argmax(counts))
    inliers=(values>=edges[peak]-.002)&(values<edges[peak+1]+.002)
    distance=float(np.median(values[inliers]))
    if counts[peak]<40:raise ValueError('No dominant parallel cabinet face for closing target')
    result['nominal_target_progress']=result['target_progress'];result['closing_seating_allowance_m']=.010
    result['target_progress']=distance+.010
    result['closing_target_estimate']={'source':'depth of dominant parallel cabinet face below selected drawer',
        'observed_gap_m':distance,'support_points':int(inliers.sum()),'peak_points':int(counts[peak]),
        'seating_allowance_m':.010,'simulator_joint_state_used':False}
    return result
