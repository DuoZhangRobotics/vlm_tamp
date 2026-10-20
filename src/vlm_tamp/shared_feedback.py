"""Bounded proprioceptive reference accommodation, with no execution veto.

Not RL and not force-sensor admittance: tracking deflection supplies a contact
proxy. Every nominal sample advances. Only robot kinematics and measurements
enter the correction; a small bounded offset does not certify collision freedom.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation
from .rekep_libero.geometry import pose_matrix


@dataclass(frozen=True)
class FeedbackConfig:
    maximum_offset_m: float = .035
    maximum_rotation_rad: float = .12
    update_fraction: float = .25
    preload_m: float = .002
    deflection_deadband_m: float = .004
    damping: float = .03
    maximum_joint_correction_rad: float = .12
    ik_iterations: int = 4

    def record(self): return asdict(self)


def bounded_vector(x, bound):
    x = np.asarray(x, float)
    return x * min(1., bound / max(np.linalg.norm(x), 1e-12))


def pose_error(target, current):
    return np.r_[target[:3,3]-current[:3,3],
                 Rotation.from_matrix(current[:3,:3].T @ target[:3,:3]).as_rotvec()]


def corrected_joints(chain, nominal, target, config):
    q0 = np.asarray(nominal, float); q = q0.copy()
    for _ in range(config.ik_iterations):
        pose, jac = chain.forward(q, jacobian=True)
        error = pose_error(target, pose_matrix(pose))
        # Rotation is in the same body frame as RobotChain's lower Jacobian.
        weights = np.array([1.,1.,1.,.15,.15,.15])
        j = weights[:,None]*jac; e = weights*error
        dq = j.T @ np.linalg.solve(j@j.T + config.damping**2*np.eye(6), e)
        q += np.clip(dq, -.05, .05)
        q = np.clip(q, q0-config.maximum_joint_correction_rad, q0+config.maximum_joint_correction_rad)
        q = np.clip(q, chain.limits[:,0]+.002, chain.limits[:,1]-.002)
    return q


class ReferenceFeedback:
    def __init__(self, chain, references, measured_pose, *, variant, normal, phase, config=None):
        if variant not in {'nominal','reanchor','reactive'}:
            raise ValueError('Unknown reference controller')
        self.chain=chain; self.config=config or FeedbackConfig(); self.variant=variant; self.phase=phase
        self.normal=np.asarray(normal,float); self.normal/=np.linalg.norm(self.normal)
        self.poses=[pose_matrix(chain.forward(q)) for q in references]
        self.offset=np.zeros(3); self.rotation=np.zeros(3)
        measured=pose_matrix(measured_pose)
        if phase == 'manipulate' and variant in {'reanchor','reactive'}:
            self.offset=bounded_vector(measured[:3,3]-self.poses[0][:3,3], self.config.maximum_offset_m)
            self.offset-=self.config.preload_m*self.normal
            self.rotation=bounded_vector(Rotation.from_matrix(measured[:3,:3]@self.poses[0][:3,:3].T).as_rotvec(),self.config.maximum_rotation_rad)

    def command(self, index, nominal, measured_pose):
        base=self.poses[index]; measured=pose_matrix(measured_pose); c=self.config
        if self.variant=='nominal' or self.phase=='free':
            return np.asarray(nominal,float), {'offset_world_m':[0.,0.,0.], 'rotation_offset_rad':[0.,0.,0.]}
        if self.variant=='reactive':
            # Accommodate deflection normal to the intended contact. Tangential
            # motion keeps its planned reference and is never stopped for lag.
            deflection=float((measured[:3,3]-base[:3,3])@self.normal)
            desired=max(0.,deflection-c.deflection_deadband_m)
            current=float(self.offset@self.normal)
            self.offset += c.update_fraction*(desired-current)*self.normal
            self.offset=bounded_vector(self.offset,c.maximum_offset_m)
        target=base.copy();target[:3,3]+=self.offset
        target[:3,:3]=Rotation.from_rotvec(self.rotation).as_matrix()@base[:3,:3]
        command=corrected_joints(self.chain,nominal,target,c)
        return command, {'offset_world_m':self.offset.tolist(), 'rotation_offset_rad':self.rotation.tolist(),
                          'target_pose_position_world_m':target[:3,3].tolist(),
                          'reference_accommodation':True,'force_feedback_used':False}
