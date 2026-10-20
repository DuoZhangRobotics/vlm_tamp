"""Reference accommodation transverse to planned progress.

Preserve tangential motion, accommodate transverse position error, and retain a
small contact preload for press/drag. This is proprioceptive reference control,
not a force-measuring controller. Sensor-inferred offsets never veto execution.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .shared_feedback import FeedbackConfig,bounded_vector,corrected_joints
from .rekep_libero.geometry import pose_matrix


class DirectionalFeedback:
    def __init__(self,chain,references,measured_pose,*,variant,normal,phase,config=None,
                 tangent=None,mode='pinch',visual_offset=None):
        if variant not in {'nominal','reanchor','reactive'}:raise ValueError('Unknown controller')
        self.chain=chain;self.config=config or FeedbackConfig();self.variant=variant;self.phase=phase;self.mode=mode
        self.poses=[pose_matrix(chain.forward(q)) for q in references]
        self.normal=np.asarray(normal,float);self.normal/=np.linalg.norm(self.normal)
        if tangent is None:
            tangent=self.poses[-1][:3,3]-self.poses[0][:3,3]
            if np.linalg.norm(tangent)<1e-8:
                tangent=np.cross(self.normal,[1,0,0] if abs(self.normal[0])<.8 else [0,1,0])
        self.tangent=np.asarray(tangent,float);self.tangent/=np.linalg.norm(self.tangent)
        self.projector=np.eye(3)-np.outer(self.tangent,self.tangent)
        projected=self.projector@self.normal
        self.loading_direction=projected/max(np.linalg.norm(projected),1e-12)
        self.preload=self.config.preload_m if mode=='press_drag' else 0.
        self.offset=np.zeros(3);self.rotation=np.zeros(3)
        measured=pose_matrix(measured_pose)
        if phase=='manipulate' and variant!='nominal':
            self.offset=bounded_vector(self.projector@(measured[:3,3]-self.poses[0][:3,3]),self.config.maximum_offset_m)
            self.offset-=self.preload*self.loading_direction
            if variant=='reactive' and visual_offset is not None:
                self.offset+=self.projector@np.asarray(visual_offset,float)
            self.rotation=bounded_vector(Rotation.from_matrix(measured[:3,:3]@self.poses[0][:3,:3].T).as_rotvec(),self.config.maximum_rotation_rad)

    def command(self,index,nominal,measured_pose):
        if self.variant=='nominal' or self.phase=='free':
            return np.asarray(nominal,float),{'offset_world_m':[0.,0.,0.],'tangential_offset_m':0.}
        base=self.poses[index];measured=pose_matrix(measured_pose);c=self.config
        # Rotate the task frame along the nominal manipulation. It does not
        # follow the free-approach tangent, which points into the contact.
        rotation=base[:3,:3]@self.poses[0][:3,:3].T if self.phase=='manipulate' else np.eye(3)
        tangent=rotation@self.tangent;projector=np.eye(3)-np.outer(tangent,tangent)
        loading=rotation@self.loading_direction
        if self.variant=='reactive':
            observed=projector@(measured[:3,3]-base[:3,3])
            if self.mode!='press_drag':
                # Ignore small free-space servo lag; no force proxy is needed
                # along the task's commanded progress direction.
                norm=np.linalg.norm(observed)
                observed*=max(0.,norm-c.deflection_deadband_m)/max(norm,1e-12)
            desired=observed-self.preload*loading
            self.offset=(1-c.update_fraction)*(projector@self.offset)+c.update_fraction*desired
            self.offset=bounded_vector(self.offset,c.maximum_offset_m)
        target=base.copy();target[:3,3]+=self.offset
        target[:3,:3]=Rotation.from_rotvec(self.rotation).as_matrix()@base[:3,:3]
        command=corrected_joints(self.chain,nominal,target,c)
        return command,{'offset_world_m':self.offset.tolist(),'rotation_offset_rad':self.rotation.tolist(),
            'tangential_offset_m':float(self.offset@tangent),'contact_preload_m':self.preload,
            'normal_world':loading.tolist(),'target_pose_position_world_m':target[:3,3].tolist(),
            'reference_accommodation':True,'force_feedback_used':False}
