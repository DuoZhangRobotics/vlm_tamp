"""Reconcile historical static samples with currently observed moving handle.

This repairs point roles using current RGB-D association, not a collision
callback or simulator identity. It never clears an obstacle under a hypothetical
robot pose. Current moving observations remain collision geometry for the arm
and palm; only the existing finger/handle contact permission is applied.
"""
import hashlib

import numpy as np
from scipy.spatial import cKDTree

from .coupled_scene import Scene,moving_mask
from .rekep_libero.scene_state_v7 import voxel_points


def reconcile_static(previous,current_static,moving,handle,association,*,support=.004,handle_band=.008):
    previous=np.asarray(previous,float).reshape(-1,3)
    current_static=np.asarray(current_static,float).reshape(-1,3)
    distance=cKDTree(handle).query(previous)[0]
    indices=np.flatnonzero(distance<=handle_band)
    remove=np.zeros(len(previous),bool)
    if len(indices):
        selected=np.asarray(association(previous[indices]),bool)
        supported=cKDTree(moving).query(previous[indices])[0]<=support
        remove[indices]=selected&supported
    return np.vstack([previous[~remove],current_static]),previous[remove]


def observed_motion_roles(previous_static,previous_moving,previous_handle,current_static,current_moving,
                          current_handle,transform,*,support=.004,persistence=.002):
    """Correct roles using measured target motion and present depth support.

    Current stationary evidence wins. No hypothetical robot volume clears data.
    """
    transform=np.asarray(transform,float);R=transform[:3,:3];t=transform[:3,3]
    old_static=np.asarray(previous_static,float).reshape(-1,3)
    current_static=np.asarray(current_static,float).reshape(-1,3)
    moved=np.asarray(previous_moving)@R.T+t
    current_distance=cKDTree(moved).query(current_static)[0]
    old_stationary=cKDTree(old_static).query(current_static)[0] if len(old_static) else np.full(len(current_static),np.inf)
    local=cKDTree(current_handle).query(current_static)[0]<.065
    retag=(current_distance<support)&(old_stationary>support)&local
    moving=np.vstack([current_moving,current_static[retag]])
    static=current_static[~retag]
    near_old=cKDTree(previous_handle).query(old_static)[0]<.012
    indices=np.flatnonzero(near_old);remove=np.zeros(len(old_static),bool)
    if len(indices):
        transported=old_static[indices]@R.T+t
        supported=cKDTree(moving).query(transported)[0]<support
        persists=cKDTree(static).query(old_static[indices])[0]<persistence if len(static) else np.zeros(len(indices),bool)
        remove[indices]=supported&~persists
    return old_static[~remove],static,moving,{'current_samples_retagged_moving':int(retag.sum()),
        'historical_moving_ghosts_removed':int(remove.sum()),'observed_transform_world':transform.tolist(),
        'stationary_observation_persistence_m':persistence,'current_depth_support_m':support,
        'hypothetical_robot_clearing':False}


class ArcContactScene(Scene):
    def __init__(self,cameras,metadata,spec,handle_cloud,panel_mask,choices=None,config=None,**kwargs):
        super().__init__(cameras,metadata,spec,handle_cloud,panel_mask,choices,config,**kwargs)
        name=choices['camera'] if choices is not None else spec['camera']
        self._association_inputs=(cameras[name],np.asarray(panel_mask,bool),choices)
        self.role_reconciliation={'enabled':True,'reconciled_prior_samples':0,
            'current_moving_support_m':.004,'selected_handle_distance_m':.008,
            'moving_handle_association_m':float(self.config.get('handle_association_m',.006)),
            'handle_ownership_is_geometric_proximity_not_object_identity':True,
            'uses_current_observations_only':True,'hypothetical_pose_clearing':False}

    def retain_static(self,previous):
        camera,mask,choices=self._association_inputs
        def association(points):
            return moving_mask(points,camera,self.spec,self.contact_reference,mask,choices,self.config)
        previous_static=previous.static_points
        if getattr(self,'observed_motion_transform',None) is not None:
            prior,static,moving,record=observed_motion_roles(previous.static_points,previous.moving_reference,
                previous.contact_reference,self.static_points,self.moving_reference,self.contact_reference,
                self.observed_motion_transform)
            filtering=self.self_filter;role=self.role_reconciliation
            self._initialize(static,moving,self.contact_reference,self.spec,self.config,self.robot,
                self.robot.reference_volumes if self.robot is not None else [])
            self.self_filter=filtering;self.role_reconciliation=role
            self.role_reconciliation['observed_temporal_association']=record
            previous_static=prior
        combined,reassigned=reconcile_static(previous_static,self.static_points,
            self.moving_reference,self.contact_reference,association)
        self.static_points=voxel_points(combined,float(self.config.get('scene_voxel_spacing_m',.003)))
        self.static_points.setflags(write=False);self.static_tree=cKDTree(self.static_points)
        self.static_prior_retained=True
        self.role_reconciliation.update(reconciled_prior_samples=len(reassigned),
            reassigned_points_sha256=hashlib.sha256(reassigned.tobytes()).hexdigest(),
            example_reassigned_world_m=reassigned[:20].tolist(),
            current_moving_geometry_retained=True)

    def record(self):
        return {**super().record(),'handle_role_reconciliation':self.role_reconciliation}
