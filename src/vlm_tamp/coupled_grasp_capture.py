"""Validate selected observed target capture using configured finger-pad geometry."""
import numpy as np
from .rekep_libero.geometry import pose_matrix,transform_points


def observed_pad_capture(metadata,handle_cloud,radius,*,padding=.003,minimum_points=10,axis=(0.,0.,1.)):
    radius=float(radius)
    if not np.isfinite(radius) or not .005<=radius<=.025:
        raise ValueError('Observed handle radius is outside the supported grasp model')
    pose=pose_matrix(metadata['eef_pose_world_xyzw']);inverse=np.linalg.inv(pose)
    pads=[transform_points(v['vertices_world'],inverse) for v in metadata['robot_collision_volumes_world']
          if v['robot_geom_name'].endswith('_pad_collision') and v['type']=='convex']
    if len(pads)!=2:raise ValueError('Two calibrated finger-pad volumes are required')
    cloud=np.asarray(handle_cloud,float)
    if cloud.ndim!=2 or cloud.shape[1]!=3 or not np.isfinite(cloud).all():
        raise ValueError('Finite selected handle observations are required')
    vertices=np.vstack(pads);midpoint=np.mean([p.mean(0) for p in pads],axis=0)
    low,high=vertices.min(0)-padding,vertices.max(0)+padding
    target=transform_points(cloud,inverse)
    captured=((target>=low)&(target<=high)).all(1)
    separation=float(np.linalg.norm(pads[0].mean(0)-pads[1].mean(0)))
    half_thickness=max(float(np.ptp(p[:,0])/2) for p in pads)
    bar_axis=np.asarray(axis,float).copy()
    if bar_axis.shape!=(3,) or not np.isfinite(bar_axis).all() or np.linalg.norm(bar_axis)<1e-9:
        raise ValueError('Finite observed bar axis required')
    bar_axis/=np.linalg.norm(bar_axis)
    transverse=float(np.sqrt(max(0.,1.-float(pose[:3,0]@bar_axis)**2)))
    if transverse<.25:raise ValueError('Closing direction is too parallel to the observed bar')
    maximum_separation=2*radius/transverse+2*half_thickness+2*padding
    if separation>maximum_separation:
        raise ValueError('Measured fingers remain too far apart to capture the selected bar')
    if int(captured.sum())<minimum_points:
        raise ValueError('Selected RGB-D handle surface is outside the measured finger-pad channel')
    bound=float(np.max(np.linalg.norm(vertices-midpoint,axis=1))+radius+padding)
    return {'observed_points_in_pad_channel':int(captured.sum()),'required_observed_points':minimum_points,
        'pad_channel_bounds_tcp_m':[low.tolist(),high.tolist()],'sensor_padding_m':padding,
        'pad_centre_separation_m':separation,'maximum_supported_separation_m':maximum_separation,
        'maximum_axis_distance_m':bound,'observed_radius_m':radius,'closing_axis_transverse_fraction':transverse,
        'source':'selected measured handle surface inside configured finger-pad channel',
        'continued_observed_retention_required':True,'force_closure_certified':False,
        'simulator_scene_geometry_used':False}
