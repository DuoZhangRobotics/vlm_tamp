"""Local start-state projection without underdetermined IK nullspace drift.

This is a contact-binding correction, not a global IK search.  The door angle
stays fixed, and all robot corrections remain inside both the configured joint
bounds and an explicit neighbourhood of the measured posture.
"""

import numpy as np


def project_nearby(problem, values, *, fixed_progress=None,
                   maximum_joint_correction_rad=.04, residual_tolerance=1e-5,
                   max_iterations=50):
    """Return an explicit success/failure record for a local minimum-norm fit.

    A damped pseudoinverse chooses no arbitrary Jacobian nullspace motion.
    Active joint bounds and a bounded line search preserve locality.  Failure
    means this local projection did not succeed; it does not prove that the
    complete task has no solution.
    """
    measured = np.asarray(values, dtype=float)
    bounds = np.asarray(problem.bounds, dtype=float)
    if measured.shape != (8,) or not np.isfinite(measured).all():
        raise ValueError('Expected seven finite robot joints and one door angle')
    if (bounds.shape != (8, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError('Expected eight finite, ordered state bounds')
    if (not np.isfinite(maximum_joint_correction_rad) or maximum_joint_correction_rad <= 0
            or not np.isfinite(residual_tolerance) or residual_tolerance <= 0
            or isinstance(max_iterations, bool) or int(max_iterations) != max_iterations
            or not 1 <= max_iterations <= 1000):
        raise ValueError('Positive local projection limits are required')
    progress = measured[7] if fixed_progress is None else float(fixed_progress)
    if not np.isfinite(progress):
        raise ValueError('Fixed progress must be finite')
    measured = measured.copy()
    state = measured.copy()
    state[7] = progress
    evaluations, iterations, history = 0, 0, []

    def finish(accepted, reason):
        correction = state[:7] - measured[:7]
        return {'accepted': bool(accepted), 'state': state.copy(), 'reason': reason,
                'evaluations': evaluations, 'iterations': iterations,
                'correction_norm': float(np.linalg.norm(correction)),
                'maximum_joint_correction_rad': float(np.max(np.abs(correction))),
                'maximum_allowed_joint_correction_rad': float(maximum_joint_correction_rad),
                'strict_residual_tolerance': float(residual_tolerance),
                'fixed_progress': float(progress), 'history': history,
                **problem.errors(state)}

    if not bounds[7, 0] <= progress <= bounds[7, 1]:
        return finish(False, 'fixed_progress_outside_bounds')
    if (np.any(measured[:7] < bounds[:7, 0] - 1e-9)
            or np.any(measured[:7] > bounds[:7, 1] + 1e-9)):
        return finish(False, 'measured_start_outside_command_bounds')
    lower = np.maximum(bounds[:7, 0], measured[:7] - maximum_joint_correction_rad)
    upper = np.minimum(bounds[:7, 1], measured[:7] + maximum_joint_correction_rad)
    state[:7] = np.clip(state[:7], lower, upper)
    residual, jacobian = problem.evaluate(state)
    evaluations += 1
    for iterations in range(int(max_iterations) + 1):
        residual = np.asarray(residual, dtype=float)
        jacobian = np.asarray(jacobian, dtype=float)[:, :7]
        if (residual.ndim != 1 or jacobian.shape != (len(residual), 7)
                or not np.isfinite(residual).all() or not np.isfinite(jacobian).all()):
            return finish(False, 'nonfinite_or_malformed_constraint')
        norm = float(np.linalg.norm(residual))
        history.append({'iteration': iterations, 'weighted_residual_norm': norm,
                        'maximum_joint_correction_rad': float(np.max(np.abs(state[:7] - measured[:7])))})
        if norm <= residual_tolerance:
            return finish(problem.satisfied(state),
                          'satisfied' if problem.satisfied(state)
                          else 'equality_satisfied_but_state_inequality_failed')
        if iterations == max_iterations:
            return finish(False, 'local_projection_iteration_limit')

        # Eliminate joints already at a bound when the minimum-norm direction
        # points outward; re-solve using the remaining available coordinates.
        free = np.ones(7, dtype=bool)
        delta = np.zeros(7)
        for _ in range(8):
            if not free.any():
                return finish(False, 'local_projection_blocked_by_bounds')
            u, singular, vt = np.linalg.svd(jacobian[:, free], full_matrices=False)
            cutoff = max(1e-12, float(singular[0]) * 1e-10) if len(singular) else 1e-12
            inverse = np.where(singular > cutoff, singular / (singular**2 + 1e-16), 0.)
            delta[:] = 0.
            delta[free] = -(vt.T * inverse) @ (u.T @ residual)
            outward = (((state[:7] <= lower + 1e-12) & (delta < 0))
                       | ((state[:7] >= upper - 1e-12) & (delta > 0))) & free
            if not outward.any():
                break
            free[outward] = False
        if np.max(np.abs(delta)) < 1e-12:
            return finish(False, 'local_projection_zero_descent')
        delta *= min(1., min(.01, maximum_joint_correction_rad) / np.max(np.abs(delta)))
        step = 1.
        positive, negative = delta > 0, delta < 0
        if positive.any():
            step = min(step, float(np.min((upper[positive] - state[:7][positive]) / delta[positive])))
        if negative.any():
            step = min(step, float(np.min((lower[negative] - state[:7][negative]) / delta[negative])))
        cost = .5 * norm**2
        accepted_step = False
        for _ in range(24):
            candidate = state.copy()
            candidate[:7] = np.clip(state[:7] + step * delta, lower, upper)
            if np.max(np.abs(candidate[:7] - state[:7])) < 1e-13:
                break
            next_residual, next_jacobian = problem.evaluate(candidate)
            evaluations += 1
            next_cost = .5 * float(np.asarray(next_residual) @ np.asarray(next_residual))
            if (np.isfinite(next_cost)
                    and (next_cost <= .5 * residual_tolerance**2
                         or next_cost < cost * (1. - 1e-4 * step))):
                state = candidate
                residual, jacobian = next_residual, next_jacobian
                accepted_step = True
                break
            step *= .5
        if not accepted_step:
            return finish(False, 'local_projection_line_search_stalled')
    return finish(False, 'local_projection_iteration_limit')
