"""Planning-model sensitivity, not a probabilistic or collision certificate."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotate_cloud(points,origin,axis,angle):
    origin=np.asarray(origin,float);axis=np.asarray(axis,float)
    axis=axis/np.linalg.norm(axis)
    return origin+Rotation.from_rotvec(float(angle)*axis).apply(np.asarray(points,float)-origin)


def hinge_sensitivity(points,origin,axis,angles=(15.,45.,90.)):
    points=np.asarray(points,float);origin=np.asarray(origin,float)
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    reference=np.eye(3)[np.argmin(np.abs(axis))]
    u=np.cross(axis,reference);u/=np.linalg.norm(u);v=np.cross(axis,u)
    cases=[]
    for magnitude in [.005,.010]:
        for direction,name in [(u,'u'),(v,'v')]:
            for sign in [-1.,1.]:
                cases.append((f'origin_{sign*magnitude*1000:+g}mm_{name}',origin+sign*magnitude*direction,axis,0.))
    for degrees in [1.,3.]:
        for direction,name in [(u,'u'),(v,'v')]:
            for sign in [-1.,1.]:
                tilted=Rotation.from_rotvec(np.radians(sign*degrees)*direction).apply(axis)
                cases.append((f'axis_{sign*degrees:+g}deg_{name}',origin,tilted,0.))
    for degrees in [-5.,-2.,2.,5.]:cases.append((f'progress_{degrees:+g}deg',origin,axis,np.radians(degrees)))
    rows=[]
    for degrees in angles:
        angle=np.radians(degrees);nominal=rotate_cloud(points,origin,axis,angle)
        for name,hinge,direction,error in cases:
            changed=rotate_cloud(points,hinge,direction,angle+error)
            distance=np.linalg.norm(changed-nominal,axis=1)
            rows.append({'nominal_progress_deg':float(degrees),'perturbation':name,
                'median_displacement_m':float(np.median(distance)),
                'p95_displacement_m':float(np.quantile(distance,.95)),
                'maximum_displacement_m':float(np.max(distance))})
    return {'rows':rows,'matched_point_displacement_without_registration':True,
        'uncertainty_calibrated_from_data':False,'probability_of_collision_estimated':False,
        'perturbations_are_sensitivity_probes':True,'execution_veto':False,
        'nominal_geometry_is_estimated':True,
        'parallel_axis_offset_error_formula':'delta_p=(I-R(theta))*delta_h',
        'parallel_axis_offset_10mm_at_90deg_displacement_m':float(2*np.sin(np.pi/4)*.01)}
