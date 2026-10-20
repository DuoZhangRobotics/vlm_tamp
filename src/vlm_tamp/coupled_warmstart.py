"""Locally retarget complete contact plans, then validate their native edges.

Call the native validator only in a short-lived planner process.  Door progress
is supplied by the caller; this module never changes the task's requested goal.
"""

from pathlib import Path
import sys
import time
import numpy as np

from .coupled_local_projection import project_nearby


def _path(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite eight-coordinate warm path')
    return values


def retarget_path(problem, start_x8, warm_path, valid, *,
                  maximum_joint_correction_rad=.04, residual_tolerance=1e-5,
                  native_validation=True, delta=.02, interpolation_count=None):
    """Return (path, result); failures return an empty path with their reason.

    The caller maps warm-path angles into the new reference frame and trims
    already executed motion. Each angle remains fixed during local projection.
    A projected current start is prepended, so native edge checking includes
    its connector to the previous plan. The actual-to-projected physical
    binding connector remains the executor's responsibility.
    """
    warm = _path(warm_path)
    start = np.asarray(start_x8, dtype=float)
    if start.shape != (8,) or not np.isfinite(start).all():
        raise ValueError('Expected a finite eight-coordinate current start')
    before = time.perf_counter()
    output, corrections, projections = [], [], []
    result = {'planner': 'retargeted complete path with OMPL Atlas edge validation',
        'exact_solution': False, 'retargeted': False, 'warm_states': len(warm),
        'goal_angle_rad': float(warm[-1, 7]), 'native_validation_pending': True,
        'continuous_clearance_certified': False}
    for index, values in enumerate([start, *warm]):
        projection = project_nearby(problem, values, fixed_progress=float(values[7]),
            maximum_joint_correction_rad=maximum_joint_correction_rad,
            residual_tolerance=residual_tolerance)
        state = projection['state']
        if not projection['accepted'] or not valid(state):
            result.update(status='warm_start_projection_or_validity_failed', failed_state=index - 1,
                failure_is_start=index == 0, projection={k: v for k, v in projection.items() if k != 'state'},
                failed_state_values=state.tolist(), retarget_seconds=time.perf_counter() - before)
            return np.empty((0, 8)), result
        corrections.append(projection['maximum_joint_correction_rad'])
        projections.append(projection['position_error_m'])
        if not output or np.max(np.abs(state - output[-1])) > 1e-12:
            output.append(state)
    path = np.asarray(output)
    if len(path) < 2 or abs(path[-1, 7] - warm[-1, 7]) > 1e-12:
        result.update(status='warm_start_has_no_complete_motion')
        return np.empty((0, 8)), result
    result.update(retargeted=True, retarget_seconds=time.perf_counter() - before,
        maximum_joint_projection_correction_rad=float(max(corrections)),
        maximum_retargeted_position_error_m=float(max(projections)),
        retargeted_states=len(path), final_planned_angle_rad=float(path[-1, 7]))
    if not native_validation:
        result['status'] = 'retargeted_states_valid_native_edges_pending'
        return path, result
    checked_path, checked = validate_native_path(problem, path, valid,
        residual_tolerance=residual_tolerance, delta=delta,
        interpolation_count=interpolation_count)
    return checked_path, {**result, **checked}


def validate_native_path(problem, path, valid, *, residual_tolerance=1e-5,
                         delta=.02, interpolation_count=None):
    """Native constrained motion checking, not merely endpoint collision tests."""
    path = _path(path)
    if residual_tolerance <= 0 or delta <= 0:
        raise ValueError('Positive constrained validation tolerances are required')
    dependency = Path(__file__).resolve().parents[2] / '.runtime' / 'ompl-deps'
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))
    from ompl import base as ob, geometric as og

    class Constraint(ob.Constraint):
        def __init__(self):
            super().__init__(8, problem.codimension, residual_tolerance)

        def function(self, x, out):
            out[:] = problem.evaluate(x)[0]

        def jacobian(self, x, out):
            out[:] = problem.evaluate(x)[1]

    constraint = Constraint()
    constraint.setMaxIterations(100)
    ambient, bounds = ob.RealVectorStateSpace(8), ob.RealVectorBounds(8)
    for index, (low, high) in enumerate(problem.bounds):
        bounds.setLow(index, float(low))
        bounds.setHigh(index, float(high))
    ambient.setBounds(bounds)
    space = ob.AtlasStateSpace(ambient, constraint)
    information = ob.ConstrainedSpaceInformation(space)
    space.setSpaceInformation(information)
    space.setDelta(float(delta))
    space.setRho(.15)
    space.setEpsilon(.01)
    space.setup()
    calls, invalid = 0, []

    def checked(state):
        nonlocal calls
        calls += 1
        values = np.asarray([state[i] for i in range(8)])
        errors = problem.errors(values)
        accepted = (problem.satisfied(values)
                    and errors['weighted_residual_norm'] <= residual_tolerance * (1. + 1e-6)
                    and bool(valid(values)))
        if not accepted and len(invalid) < 4:
            invalid.append({'state': values.tolist(), 'errors': errors})
        return bool(accepted)

    information.setStateValidityChecker(checked)
    information.setup()
    native, retained = og.PathGeometric(information), []
    for values in path:
        state = space.allocState()
        state.copy(values)
        retained.append(state)  # Do not manually free nanobind-owned states.
        space.anchorChart(state)
        native.append(state)
    before = time.perf_counter()
    initial_check = bool(native.check())
    interpolated_check = False
    output = np.empty((0, 8))
    if initial_check:
        if interpolation_count is None:
            steps = np.maximum(np.max(np.abs(np.diff(path[:, :7], axis=0)), axis=1) / .015,
                               np.abs(np.diff(path[:, 7])) / .02)
            interpolation_count = max(len(path), int(np.ceil(steps).sum()) + 1)
        native.interpolate(max(int(interpolation_count), len(path)))
        interpolated_check = bool(native.check())
        if interpolated_check:
            output = np.asarray([[state[i] for i in range(8)] for state in native.getStates()])
    accepted = initial_check and interpolated_check and len(output) > 1
    result = {'exact_solution': bool(accepted), 'native_validation_pending': False,
        'status': 'validated_warm_start' if accepted else 'warm_start_native_edge_check_failed',
        'ompl_path_check_passed': initial_check, 'interpolated_path_check_passed': interpolated_check,
        'state_validity_checks': calls, 'first_invalid_states': invalid,
        'validation_seconds': time.perf_counter() - before, 'states': len(output),
        'constraint_tolerance': residual_tolerance, 'delta': delta,
        'continuous_clearance_certified': False,
        'binding_lifetime': 'caller must use a short-lived process'}
    if accepted:
        result.update(maximum_position_error_m=max(problem.errors(x)['position_error_m'] for x in output),
                      maximum_joint_edge_rad=float(np.max(np.abs(np.diff(output[:, :7], axis=0)))))
    return output, result
