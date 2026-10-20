import numpy as np

from vlm_tamp.short_motion_policy import motion_horizons,ObservedHingeHistory


def test_calibration_precedes_longer_motions_and_never_demands_full_goal():
    config={'motion_policy':'short'}
    assert np.allclose(np.degrees(motion_horizons(np.pi/2,config)),[3,2,1])
    assert np.allclose(np.degrees(motion_horizons(np.pi/2,{**config,'hinge_refinement_accepted':True})),[8,2,1])
    assert motion_horizons(np.pi/2,{**config,'full_task_only':True})==[np.pi/2]
    assert np.allclose(np.degrees(motion_horizons(np.radians(.5),config)),[.5])


def test_predicted_motion_cannot_create_an_observed_hinge_fit():
    history=ObservedHingeHistory()
    for step,t in enumerate(np.linspace(0,.4,6)):
        point=[.2*np.cos(t),.2*np.sin(t),1.]
        result=history.add(point,observed=step==0,control_step=step)
    assert len(history.points)==1 and history.skipped==5
    assert not result['accepted']


def test_observed_circle_recovers_hinge():
    history=ObservedHingeHistory();origin=np.array([.1,-.2,1.])
    for step,t in enumerate(np.linspace(0,.4,7)):
        result=history.add(origin+[.2*np.cos(t),.2*np.sin(t),0],observed=True,control_step=step)
    assert result['accepted']
    assert np.linalg.norm(np.asarray(result['hinge_world_m'])[:2]-origin[:2])<1e-7
