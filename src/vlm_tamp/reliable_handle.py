"""Validate upright handle fits and recover a bar within the observed door region."""
import copy
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .handle_contact import ground_handle as original_ground_handle,upright_plane
from .rigid_hinge_tracking import robot_mask
from .rekep_libero.mechanism_geometry import project
from .rekep_libero.scene_state_v7 import observed_points,voxel_points


def ground_handle(cameras,choices,template,metadata=None,*,fixed_joint=None):
    original=original_ground_handle(cameras,choices,template,metadata,fixed_joint=fixed_joint)
    mech,cloud,mask,anchors,evidence=original
    adequate=np.ptp(cloud[:,2])>.065 and evidence['cylinder_robust_residual_m']<.0035
    if adequate:return refine_bar(original)
    camera=cameras[choices['camera']];world=np.asarray(camera['world_points'])
    box=np.asarray(choices['moving_panel_box_xyxy']);yy,xx=np.indices(world.shape[:2])
    region=(xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])&np.isfinite(world).all(2)
    region &= ~robot_mask(camera)
    face=upright_plane(world[region],np.asarray(camera['camera_to_world'])[:3,3]);center,normal=face['point'],face['normal']
    zlo,zhi=np.quantile(world[region][face['inliers'],2],[.05,.95])
    occupied,_=observed_points(cameras,metadata,spacing=.0015,robot_depth_tolerance=.0001)
    pixel=project(occupied,camera)
    bounded=(pixel[:,0]>=box[0]-15)&(pixel[:,0]<=box[2]+15)&(pixel[:,1]>=box[1]-15)&(pixel[:,1]<=box[3]+15)
    signed=(occupied-center)@normal
    candidates=[]
    for sign,near,far in [(1.,.008,.075),(-1.,.035,.09)]:
        selection=bounded&(sign*signed>near)&(sign*signed<far)&(occupied[:,2]>zlo+.025)&(occupied[:,2]<zhi-.01)
        points=occupied[selection]
        middle=points[(points[:,2]>zlo+.3*(zhi-zlo))&(points[:,2]<zlo+.75*(zhi-zlo))]
        if len(middle)<75:continue
        low=middle[:,:2].min(0);cells=np.floor((middle[:,:2]-low)/.003).astype(int)
        dims=cells.max(0)+1
        if np.prod(dims)>1000000:continue
        grid=np.zeros(dims,bool);grid[tuple(cells.T)]=True
        labels,n=ndimage.label(grid,np.ones((3,3),int));ids=labels[tuple(cells.T)]
        for label in range(1,n+1):
            part=middle[ids==label]
            if len(part)<60 or np.linalg.norm(np.ptp(part[:,:2],axis=0))>.065:continue
            xy=np.median(part[:,:2],axis=0)
            fit=least_squares(lambda x:np.linalg.norm(part[:,:2]-x[:2],axis=1)-x[2],np.r_[xy,.01],
                bounds=(np.r_[xy-.025,.005],np.r_[xy+.025,.018]),loss='soft_l1',f_scale=.001,max_nfev=150)
            error=np.abs(np.linalg.norm(part[:,:2]-fit.x[:2],axis=1)-fit.x[2])
            if np.median(error)>.0025:continue
            radial=np.abs(np.linalg.norm(points[:,:2]-fit.x[:2],axis=1)-fit.x[2])
            cylinder=points[radial<.003]
            if len(cylinder)<100 or np.ptp(cylinder[:,2])<.065:continue
            # A bar must have dense support across height, not just two mount caps.
            hist,_=np.histogram(cylinder[:,2],bins=np.arange(cylinder[:,2].min(),cylinder[:,2].max()+.005,.005))
            coverage=float(np.mean(hist>=4))
            if coverage<.75:continue
            candidates.append((np.ptp(cylinder[:,2])*coverage,len(cylinder),sign,fit.x,cylinder,float(np.median(error))))
    if not candidates:raise ValueError('RGB handle fit invalid and no supported vertical bar in observed door region')
    # A cylinder elsewhere in the door box may be a hinge or cabinet edge.
    # Geometric regularity cannot authorize changing the selected semantic part.
    proposals=[]
    for _,_,sign,fit,cylinder,error in sorted(candidates,key=lambda row:(row[0],row[1]),reverse=True)[:4]:
        pixels=project(cylinder,camera);lo,hi=np.quantile(pixels,[.01,.99],axis=0)
        proposals.append({'camera':choices['camera'],'box_xyxy':np.r_[lo,hi].tolist(),
            'cylinder_residual_m':error,'semantic_identity_verified':False})
    failure=ValueError('Selected contact region does not support the vertical-handle model. '
        'No replacement region was installed. Inspect another view and explicitly select the intended contact.')
    failure.planning_diagnostic={'status':'contact_region_unresolved','module':'reliable_handle.contact_identity',
        'selected_region':choices['contact_box_xyxy'],'height_m':float(np.ptp(cloud[:,2])),
        'radial_residual_m':evidence['cylinder_robust_residual_m'],'allowed_radial_residual_m':.0035,
        'minimum_height_m':.065,'unconfirmed_vertical_features':proposals,
        'physical_motion_issued':False,'model_changed':False,
        'interpretation':'A different vertical feature is not a verified handle. Use the camera overview and select a visible contact in original pixels.'}
    raise failure


def refine_bar(result):
    """Refit the central bar after excluding mounts and noncylindrical endpoints."""
    mechanism,cloud,mask,anchors,evidence=result
    lo,hi=np.quantile(cloud[:,2],[.2,.8]);points=cloud[(cloud[:,2]>=lo)&(cloud[:,2]<=hi)]
    old=np.asarray(anchors[0][:2]);fit=least_squares(lambda x:np.linalg.norm(points[:,:2]-x[:2],axis=1)-x[2],
        np.r_[old,evidence['cylinder_radius_m']],bounds=(np.r_[old-.02,.005],np.r_[old+.02,.018]),
        loss='soft_l1',f_scale=.0005,max_nfev=150)
    residual=float(np.median(np.abs(fit.fun)))
    if residual>.0025:raise ValueError('Central handle surface is inconsistent with a cylinder')
    evidence['central_bar_refinement']={'old_axis_xy':old.tolist(),'new_axis_xy':fit.x[:2].tolist(),
        'radial_residual_m':residual,'excluded_height_quantiles':[.2,.8],'simulator_geometry_used':False}
    evidence['cylinder_radius_m']=float(fit.x[2])
    anchors=[np.r_[fit.x[:2],a[2]] for a in anchors]
    mechanism.spec['source_anchor_world_m']=anchors[0].tolist()
    return mechanism,cloud,mask,anchors,evidence
