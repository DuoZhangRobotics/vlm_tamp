"""Short complete plans and hinge fitting from observed handle motion only."""
import numpy as np

from .handle_contact import fit_upright_hinge


def motion_horizons(remaining,config):
    if config.get('full_task_only',False):return [float(remaining)]
    if config.get('motion_policy','legacy')=='short':
        cap=8. if config.get('hinge_refinement_accepted',False) else 3.
        degrees=[cap,2.,1.]
        candidates=[min(float(remaining),float(np.radians(value))) for value in degrees]
    else:
        candidates=[float(remaining)]+[float(np.radians(v)) for v in [20.,10.] if np.radians(v)<remaining-1e-8]
    unique=[]
    for value in candidates:
        if value>0 and not any(abs(value-old)<1e-9 for old in unique):unique.append(value)
    return unique


class ObservedHingeHistory:
    def __init__(self):
        self.points=[];self.frames=[];self.skipped=0;self.last_fit={'accepted':False,'reason':'insufficient_observed_motion'}

    def add(self,point,*,observed,control_step):
        point=np.asarray(point,float)
        if not observed or point.shape!=(3,) or not np.isfinite(point).all():
            self.skipped+=1
            self.last_fit={'accepted':False,'reason':'not_an_observed_measurement'}
            return self.last_fit
        self.points.append(point.copy());self.frames.append(int(control_step))
        self.last_fit=fit_upright_hinge(self.points)
        return self.last_fit

    def record(self,*,refined=False):
        return {'status':'refined' if refined else 'collecting_observed_motion',
            'observed_centres_world_m':[p.tolist() for p in self.points],
            'control_steps':self.frames,'observed_samples':len(self.points),
            'predicted_samples_excluded':self.skipped,'last_fit':self.last_fit,
            'refinement_used_in_planning':bool(refined),'simulator_hinge_used':False}
