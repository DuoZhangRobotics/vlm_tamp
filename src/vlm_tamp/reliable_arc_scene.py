"""Remove historical depth ghosts only where current cameras observe free space."""
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from .arc_contact_scene import ArcContactScene
from .rekep_libero.geometry import transform_points
from .rekep_libero.scene_state_v7 import voxel_points


def observed_free_space(points,cameras,*,margin=.008):
    points=np.asarray(points,float).reshape(-1,3);free=np.zeros(len(points),bool)
    for camera in cameras.values():
        local=transform_points(points,np.linalg.inv(np.asarray(camera['camera_to_world'])))
        projected=local@np.asarray(camera['intrinsic']).T
        xy=projected[:,:2]/np.maximum(projected[:,2:3],1e-10)
        pixel=np.rint(np.nan_to_num(xy,nan=-1,posinf=-1,neginf=-1).clip(-1e6,1e6)).astype(int)
        depth=np.asarray(camera['depth']).squeeze();h,w=depth.shape
        valid=(local[:,2]>0)&(pixel[:,0]>=1)&(pixel[:,0]<w-1)&(pixel[:,1]>=1)&(pixel[:,1]<h-1)
        ids=np.flatnonzero(valid)
        # A foreground pixel anywhere in the 3x3 footprint prevents carving.
        conservative=minimum_filter(np.where(np.isfinite(depth)&(depth>0),depth,0.),size=3)
        measured=conservative[pixel[ids,1],pixel[ids,0]]
        free[ids]|=(measured>0)&(local[ids,2]<measured-margin)
    return free


class CurrentDepthScene(ArcContactScene):
    def __init__(self,cameras,*args,**kwargs):
        super().__init__(cameras,*args,**kwargs);self.current_cameras=cameras
        self.free_space_update={'enabled':True,'removed_historical_samples':0}
    def retain_static(self,previous):
        current_static=self.static_points.copy()
        super().retain_static(previous)
        contradicted=observed_free_space(self.static_points,self.current_cameras)
        # A measured endpoint in any current view takes priority over carving.
        supported=cKDTree(current_static).query(self.static_points)[0]<.004 if len(current_static) else np.zeros(len(self.static_points),bool)
        remove=contradicted&~supported
        self.static_points=voxel_points(self.static_points[~remove],float(self.config.get('scene_voxel_spacing_m',.003)))
        self.static_points.setflags(write=False);self.static_tree=cKDTree(self.static_points)
        self.free_space_update={'enabled':True,'removed_historical_samples':int(remove.sum()),
            'current_endpoints_preserved':True,'minimum_depth_footprint_pixels':3,'free_space_margin_m':.008,
            'hypothetical_robot_clearing':False,'simulator_object_truth_used':False}
    def record(self):return {**super().record(),'observed_free_space_update':self.free_space_update}
