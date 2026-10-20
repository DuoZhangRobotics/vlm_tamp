import copy
import numpy as np
from vlm_tamp.rigid_hinge_tracking import estimate_rigid_hinge
from vlm_tamp.subscription_measurements import hinge_fit_diagnostics


def observations():
    pivot=np.array([.2,-.15,1.])
    phi=np.linspace(-.5,.5,30)
    reference=pivot+np.column_stack([.2*np.cos(phi),.2*np.sin(phi),np.linspace(-.04,.04,30)])
    a=.12;rotation=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    current=(reference-pivot)@rotation.T+pivot
    return pivot,reference,current


def test_wrong_prior_is_distinguished_from_bad_sensor_fit():
    pivot,first,second=observations();prior=pivot+[-.28,0,0]
    fit=estimate_rigid_hinge(first,second,prior=prior);saved=copy.deepcopy(fit)
    assert not fit['accepted']
    evidence=hinge_fit_diagnostics(first,second,prior,fit)
    assert evidence['sensor_fit_without_initial_pivot_prior']['accepted']
    assert evidence['sensor_fit_passed_but_initial_pivot_proximity_failed']
    assert np.isclose(evidence['candidate_distance_from_initial_pivot_xy_m'],.28)
    assert fit==saved


def test_consistent_prior_is_not_reported_as_a_conflict():
    pivot,first,second=observations()
    fit=estimate_rigid_hinge(first,second,prior=pivot)
    evidence=hinge_fit_diagnostics(first,second,pivot,fit)
    assert fit['accepted']
    assert evidence['initial_pivot_proximity_passed']
    assert not evidence['sensor_fit_passed_but_initial_pivot_proximity_failed']
