"""Measured push progress during physics execution; no scene-based admission."""
from collections import deque
import numpy as np
from .rekep_libero.geometry import pose_matrix


def contact_point(pose,local):
    transform=pose_matrix(pose)
    return transform[:3,3]+transform[:3,:3]@np.asarray(local,float)


class PushProgress:
    """A one-second physical stall needs moving commands and a stationary hand."""
    def __init__(self,configuration,initial_pose):
        self.local=np.asarray(configuration['contact_local_m'],float)
        self.references=np.asarray(configuration['reference_points_world_m'],float)
        self.initial=contact_point(initial_pose,self.local)
        previous=configuration.get('previous_history',[])
        self.history=deque(((np.asarray(command,float),np.asarray(actual,float))
            for command,actual in previous),maxlen=21)
        self.carried_samples=len(self.history)
        if not self.history:self.history.append((self.initial,self.initial))
        self.actual=self.initial;self.commanded=self.initial;self.samples=0;self.stalled=False
        self.window=None

    def update(self,pose,index):
        self.actual=contact_point(pose,self.local);self.commanded=self.references[index]
        self.history.append((self.commanded,self.actual));self.samples+=1
        if len(self.history)==21:
            first_command,first_actual=self.history[0]
            requested=float(np.linalg.norm(self.commanded-first_command))
            measured=float((self.actual-first_actual)@(self.commanded-first_command)/max(requested,1e-12))
            self.window={'simulation_seconds':1.,'commanded_travel_m':requested,'measured_directional_progress_m':measured,
                'measured_total_displacement_m':float(np.linalg.norm(self.actual-first_actual))}
            self.stalled=requested>=.01 and measured<.001
        return self.stalled

    def result(self):
        return {'source':'measured robot pose during physics execution',
            'samples':self.samples,'stalled':self.stalled,'stall_window':self.window,
            'carried_history_samples':self.carried_samples,
            '_continuation_history':[[command.tolist(),actual.tolist()] for command,actual in self.history],
            'commanded_contact_displacement_m':float(np.linalg.norm(self.commanded-self.initial)),
            'measured_contact_displacement_m':float(np.linalg.norm(self.actual-self.initial)),
            'object_motion_inferred':False,'scene_prediction_veto':False}
