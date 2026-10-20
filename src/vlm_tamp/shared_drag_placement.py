"""Place the finger contact envelope behind the observed bar cross-section.

This is an explicit geometric contact prior, shared across cases, rather than
an assertion that a cloud support point provides force closure.
"""
import numpy as np
from .shared_contact import contact_frame,local_gripper,unit
from .rekep_libero.geometry import pose_matrix,matrix_pose


def rear_surface_contacts(menu,cloud,spec,metadata):
    f=contact_frame(cloud,spec);cloud=np.asarray(cloud,float)
    axial=(cloud-f['center'])@f['bar'];lo,hi=np.quantile(axial,[.25,.75])
    middle=cloud[(axial>=lo)&(axial<=hi)]
    shapes=local_gripper(metadata)
    pads=[v for k,v in shapes.items() if k.endswith('_pad_collision')]
    pad_center=np.mean([v.mean(0) for v in pads],axis=0)
    pad_depth=float(np.ptp(pads[0][:,1]));pad_height=float(np.ptp(pads[0][:,2]))
    low,high=np.quantile(middle@f['basis'],[.02,.98],axis=0)
    contact=f['basis']@((low+high)/2)
    # A bar can be pulled with fingers on its rear/upper side. The planned
    # contact region allows the resulting finger contact, while arm/palm
    # volumes remain in the scene checker. Physics determines actual loading.
    contact-=f['normal']*((high-low)[1]/2+pad_depth/2-.001)
    contact-=f['across']*(pad_height/2)
    result=[]
    for original in menu:
        row=dict(original)
        if row['mode']=='press_drag':
            pose=pose_matrix(row['pose_world_xyzw']);pose[:3,3]=contact-pose[:3,:3]@pad_center
            row.update(pose_world_xyzw=matrix_pose(pose).tolist(),contact_normal_world=f['across'].tolist(),
                selected_surface='bar rear side from observed cross-section and known pad dimensions',
                inferred_contact_anchor_world_m=contact.tolist(),
                contact_geometry_prior='bar cross-section envelope; retention resolved physically',
                contact_establishment_depth_m=None,object_shape_completion_used=False)
        result.append(row)
    return result
