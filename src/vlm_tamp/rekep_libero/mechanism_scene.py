"""Observed moving-panel geometry with pair-specific contact permissions."""

import numpy as np
from scipy.spatial import cKDTree

from .contact_policy_v8 import ContactPolicy
from .geometry import pose_matrix, transform_points
from .mechanism_geometry import project, unit
from .scene_state_v7 import observed_points, hull_volume
from .scene_state_v8 import ObservedScene as SceneV8


class MechanismContactPolicy(ContactPolicy):
    def local_panel_allowed(self, points, volume, contact_points, stage, transform):
        points = np.asarray(points, float).reshape(-1,3)
        if stage not in {'grasp','pull','release','retreat_contact'} or volume['robot_geom_name'] not in self.finger_names or not len(points):
            return np.zeros(len(points),bool)
        origin = np.asarray(self.spec['front_plane']['point_world_m'])
        normal = np.asarray(self.spec['panel_normal_world'])
        transform = pose_matrix(transform)
        moved = transform_points([origin],transform)[0]
        rotated = transform[:3,:3]@normal
        band = self.config['panel_contact_band_m']
        plane = (np.abs((points-origin)@normal)<=band) | (np.abs((points-moved)@rotated)<=band)
        return plane & (cKDTree(contact_points).query(points)[0]<=self.config['panel_contact_handle_radius_m'])


class PanelRegion:
    def __init__(self,spec,cloud,panel_mask,camera):
        self.spec,self.camera,self.panel_mask=spec,camera,np.asarray(panel_mask,bool)
        self.task_type='articulated'
        self.contact_reference=np.asarray(cloud,float)
        self.contact_tree=cKDTree(self.contact_reference)
        self.origin=np.asarray(spec['front_plane']['point_world_m'])
        self.axis=unit(spec['panel_normal_world'])
        self.side=unit(np.cross(self.axis,[0,0,1]) if abs(self.axis[2])<.9 else np.cross(self.axis,[0,1,0]))
        support=np.asarray(camera['world_points'])[self.panel_mask]
        support=support[np.isfinite(support).all(1)]
        side=(support-self.origin)@self.side
        self.side_low,self.side_high=np.quantile(side,[.01,.99])
        self.z_low,self.z_high=np.quantile(support[:,2],[.01,.99])
        self.initial_support=np.empty((0,3))
        self.description='VLM RGB panel box refined by MobileSAM, measured face band, and selected handle/control surface'

    def mask(self,points):
        points=np.asarray(points,float).reshape(-1,3)
        image=project(points,self.camera)
        pixel=np.rint(np.nan_to_num(image,nan=-1.,posinf=-1.,neginf=-1.).clip(-1e6,1e6)).astype(int)
        x,y=pixel.T
        valid=(x>=0)&(y>=0)&(x<self.panel_mask.shape[1])&(y<self.panel_mask.shape[0])
        selected=np.zeros(len(points),bool)
        selected[valid]=self.panel_mask[y[valid],x[valid]]
        selected &= np.abs((points-self.origin)@self.axis)<=self.spec['panel_depth_association_m']
        return selected | (self.contact_tree.query(points)[0]<=.008)


class MechanismScene(SceneV8):
    def __init__(self,cameras,metadata,spec,cloud,panel_mask,config):
        self.config=config
        self.margin=float(config['observed_clearance_margin_m'])
        self.spacing=float(config['scene_voxel_spacing_m'])
        self.depth_tolerance=float(config['robot_self_depth_tolerance_m'])
        self.region=PanelRegion(spec,cloud,panel_mask,cameras[spec['camera']])
        self.policy=MechanismContactPolicy(metadata['robot_collision_volumes_world'],spec,config)
        self.required_robot_parts=frozenset(v['robot_geom_name'] for v in metadata['robot_collision_volumes_world'] if not v.get('fixed'))
        points,self.self_filter=observed_points(cameras,metadata,spacing=self.spacing,robot_depth_tolerance=self.depth_tolerance)
        moving=self.region.mask(points)
        self.moving_reference=points[moving]
        if len(self.moving_reference)<75:
            raise ValueError('Insufficient observed moving mechanism surface')
        self.static_points=points[~moving]
        self.static_tree=cKDTree(self.static_points)
        self.payload_volume=hull_volume(self.moving_reference,'observed_moving_panel')
        self.target_points=np.empty((0,3))
        self.observed_transform=np.eye(4)
        self.revision=0
        self.guide_points=np.empty((0,3))
        self.prediction_events=[]

    def check(self,volumes,tcp_pose,**kwargs):
        result=super().check(volumes,tcp_pose,**kwargs)
        enforce=self.config.get('scene_mode','strict')=='strict'
        predicted=result['accepted']
        if not predicted:
            self.prediction_events.append({**result,'tcp_pose_world_xyzw':np.asarray(tcp_pose).tolist(),
                                           'scene_admission_enforced':enforce})
        return {**result,'accepted':predicted or not enforce,
                'predicted_scene_admission_accepted':predicted,'scene_admission_enforced':enforce}

    def record(self):
        result=super().record()
        result.update(schema_version='articulated-scene-1.0',joint_family=self.region.spec['joint_family'],
            moving_geometry_source='RGB-predicted panel mask and measured depth',
            panel_depth_association_m=self.region.spec['panel_depth_association_m'],
            hidden_mechanism_body_validated=False,simulator_scene_state_used=False)
        result['scene_admission_enforced']=self.config.get('scene_mode','strict')=='strict'
        return result
