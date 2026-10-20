"""Condition numerical drawer geometry on its existing upright-panel prior.

The former dominant-plane fit could select a table and normalize its almost-zero
horizontal normal into an arbitrary drawer axis. Fit a supported upright surface
in the RGB-selected box instead. No object identities or asset axes are used.
"""
import copy
import numpy as np
from .coupled_storage import write_json
from .rekep_libero.mechanism_geometry import Mechanism,ground_geometry as original_ground
from .rekep_libero.mechanism_geometry import expand_handle_surface as original_expand
from .rekep_libero.scene_state_v7 import observed_points
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def upright_plane_candidates(points, camera, anchor, *, iterations=256):
    points=np.asarray(points,float);points=points[np.isfinite(points).all(1)]
    if len(points)<60:return []
    rng=np.random.default_rng(0);sample=points
    if len(sample)>6000:sample=sample[rng.choice(len(sample),6000,replace=False)]
    candidates=[]
    for _ in range(iterations):
        a,b=sample[rng.choice(len(sample),2,replace=False)];delta=b[:2]-a[:2]
        if np.linalg.norm(delta)<.03:continue
        normal=np.r_[delta[1],-delta[0],0.];normal/=np.linalg.norm(normal)
        if normal@(np.asarray(camera)-a)<0:normal=-normal
        keep=abs((sample-a)@normal)<.0035
        support=sample[keep]
        if len(support)<50 or np.ptp(np.quantile(support[:,2],[.05,.95]))<.025:continue
        center=support.mean(0);gap=float((anchor-center)@normal)
        if not -.004<=gap<=.09:continue
        # A vertical face must have horizontal extent as well as height.
        tangent=np.array([-normal[1],normal[0],0.])
        if np.ptp(np.quantile(support@tangent,[.05,.95]))<.035:continue
        candidates.append({'center':center,'normal':normal,'support':len(support),'handle_protrusion_m':gap})
    candidates.sort(key=lambda x:-x['support'])
    if not candidates:return []
    top=candidates[0];mask=abs((points-top['center'])@top['normal'])<.004
    support=points[mask];xy=support[:,:2];_,_,vt=np.linalg.svd(xy-xy.mean(0),full_matrices=False)
    normal=np.r_[vt[-1],0.]
    if normal@top['normal']<0:normal=-normal
    center=support.mean(0);top={**top,'center':center,'normal':normal,
        'support':int(mask.sum()),'rms':float(np.sqrt(np.mean(((support-center)@normal)**2)))}
    return [top]


def grounded_geometry(bundle,choices,contact_cloud,contact_mask,perception,output):
    if choices['joint_family']!='prismatic':
        return original_ground(bundle,choices,contact_cloud,contact_mask,perception,output)
    camera=bundle['cameras'][choices['camera']];world=np.asarray(camera['world_points'])
    a,b,c,d=choices['moving_panel_box_xyxy'];yy,xx=np.indices(world.shape[:2])
    bounded=(xx>=a)&(xx<=c)&(yy>=b)&(yy<=d)&np.isfinite(world).all(2)
    anchor=np.median(contact_cloud,axis=0);points=world[bounded&~contact_mask]
    candidates=upright_plane_candidates(points,np.asarray(camera['camera_to_world'])[:3,3],anchor)
    if not candidates:raise ValueError('Selected drawer region lacks an observed upright face; no arbitrary axis projection')
    plane=candidates[0];normal=plane['normal'];center=plane['center']
    axis=normal*(-1 if choices['motion']=='close' else 1)
    mask=bounded&(abs((world-center)@normal)<.008)
    spec={'schema_version':'shared-contact-prismatic-1','task_type':'articulated',
        'instruction':bundle['meta']['instruction'],'joint_family':'prismatic','motion':choices['motion'],
        'joint_origin_world_m':anchor.tolist(),'joint_axis_world':axis.tolist(),
        'target_progress':float(choices['travel_m']),'progress_units':'meters','source_anchor_world_m':anchor.tolist(),
        'camera':choices['camera'],'front_plane':{'point_world_m':center.tolist(),'normal_world':normal.tolist(),
            'rms_m':plane['rms'],'inliers':plane['support']},'panel_normal_world':normal.tolist(),
        'pull_axis_world':normal.tolist(),'axis_prior':'upright drawer face; existing family prior',
        'direction_source':'open outward / close inward from observed upright face',
        'panel_depth_association_m':.020,'simulator_scene_state_used':False,
        'geometry_source':'RGB-selected box and supported upright depth plane',
        'contact_policy':'functional fingers at selected region; palm and arm checked',
        'constraint':{'type':'point_on_line','position_tolerance_m':.015,'orientation_tolerance_rad':.10}}
    np.save(output/'panel-mask.npy',mask);np.save(output/'panel-plane-points.npy',world[mask])
    write_json(output/'upright-plane-evidence.json',{**plane,'upright_prior_explicit':True,'object_axis_or_label_used':False})
    return Mechanism(spec),mask


def closing_target(cameras,metadata,spec,cloud,config):
    """Find the cabinet face above OR below the moving drawer, excluding table-only bins."""
    result=copy.deepcopy(spec);normal=np.asarray(spec['panel_normal_world']);origin=np.asarray(spec['front_plane']['point_world_m'])
    side=np.cross(normal,[0,0,1.]);side/=np.linalg.norm(side)
    points,_=observed_points(cameras,metadata,spacing=.003,robot_depth_tolerance=config['robot_self_depth_tolerance_m'])
    relative=points-origin;depth=-relative@normal
    nearby=(abs(relative@side)<.16)&(abs(relative[:,2])>.035)&(abs(relative[:,2])<.30)&(depth>.03)&(depth<.25)
    points=points[nearby];values=depth[nearby];choices=[]
    for lo in np.arange(.03,.25,.003):
        inliers=abs(values-(lo+.0015))<.0045;selected=points[inliers]
        if len(selected)<50:continue
        height=np.ptp(np.quantile(selected[:,2],[.05,.95]))
        if height<.035:continue
        choices.append((len(selected),float(np.median(values[inliers])),float(height)))
    if not choices:raise ValueError('No observed parallel cabinet face above or below selected drawer')
    count,distance,height=max(choices)
    result['nominal_target_progress']=result['target_progress'];result['target_progress']=distance+.010
    result['closing_seating_allowance_m']=.010
    result['closing_target_estimate']={'source':'parallel observed cabinet face above or below selected drawer',
        'observed_gap_m':distance,'support_points':count,'support_vertical_span_m':height,'seating_allowance_m':.010,
        'table_only_support_rejected':True,'simulator_joint_state_used':False}
    return result


def seed_component(points,seed,spacing=.008):
    points=np.asarray(points,float);tree=cKDTree(points);pairs=tree.query_pairs(spacing,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    _,labels=connected_components(graph,directed=False)
    _,index=tree.query(seed)
    return points[labels==labels[index]]


def expand_contact(cameras,spec,choices,reference):
    expanded=original_expand(cameras,spec,choices,reference)
    if spec['joint_family']!='prismatic':return expanded
    normal=np.asarray(spec['panel_normal_world']);origin=np.asarray(spec['front_plane']['point_world_m'])
    height=(expanded-origin)@normal
    raised=expanded[(height>.004)&(height<.06)]
    if len(raised)<30:raise ValueError('Selected handle has no raised depth component')
    x,y=np.rint(choices['grasp_pixel_xy']).astype(int)
    seed=cameras[choices['camera']]['world_points'][y,x]
    selected=seed_component(raised,seed)
    if len(selected)<30:raise ValueError('RGB contact pixel has insufficient connected raised support')
    return selected
