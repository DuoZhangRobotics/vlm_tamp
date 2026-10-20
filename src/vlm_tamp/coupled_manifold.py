"""Robot-door fixed-grasp constraints and an isolated OMPL planning adapter.

The environmental articulation is supplied by perception.  The chain contains
only calibrated robot geometry.  This module does not inspect simulator objects
or infer that a physical gripper has a rigid attachment.
"""

from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .pose_constraints import skew, perpendicular_basis
from .rekep_libero.geometry import pose_matrix
from .rekep_libero.robot_chain_v4 import fixed_orientation_residual


class CoupledHinge:
    """Eight ambient coordinates, six equalities, two manifold dimensions.

    ``x[:7]`` are Panda joints and ``x[7]`` is progress about the supplied
    signed hinge axis.  Progress zero uses the measured reference TCP.  The
    angular residual is scaled into metres for numerical conditioning, while
    acceptance always reports position and orientation separately.
    """

    codimension = 6
    constraint_name = 'rigid_fixed_grasp'

    def __init__(self, chain, origin, axis, reference_tcp, *,
                 progress_bounds=(0., np.pi / 2), rotation_scale=.15,
                 joint_margin=0., position_tolerance=.001,
                 orientation_tolerance=.01):
        self.chain = chain
        self.origin = np.asarray(origin, dtype=float)
        axis = np.asarray(axis, dtype=float)
        self.reference_tcp = np.asarray(reference_tcp, dtype=float)
        if (self.origin.shape != (3,) or axis.shape != (3,)
                or not np.isfinite(self.origin).all() or not np.isfinite(axis).all()
                or np.linalg.norm(axis) < 1e-9):
            raise ValueError('Finite origin and nonzero hinge axis are required')
        self.axis = axis / np.linalg.norm(axis)
        if (self.reference_tcp.shape != (4, 4)
                or not np.isfinite(self.reference_tcp).all()
                or not np.allclose(self.reference_tcp[3], [0, 0, 0, 1])
                or not np.allclose(self.reference_tcp[:3, :3].T @ self.reference_tcp[:3, :3], np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(self.reference_tcp[:3, :3]), 1., atol=1e-7)):
            raise ValueError('Reference TCP must be a rigid homogeneous transform')
        bounds = np.asarray(progress_bounds, dtype=float)
        limits = np.asarray(chain.limits, dtype=float)
        if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]
                or limits.shape != (7, 2) or not np.isfinite(limits).all()
                or not np.isfinite(joint_margin) or joint_margin < 0):
            raise ValueError('Finite, ordered robot and progress bounds are required')
        self.bounds = np.vstack([limits + [joint_margin, -joint_margin], bounds])
        if not np.all(self.bounds[:, 0] < self.bounds[:, 1]):
            raise ValueError('Joint margin exhausts a joint range')
        self.rotation_scale = float(rotation_scale)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        if not all(np.isfinite(value) and value > 0 for value in
                   [self.rotation_scale, self.position_tolerance, self.orientation_tolerance]):
            raise ValueError('Positive residual scale and acceptance tolerances are required')

    @staticmethod
    def _state(values):
        values = np.asarray(values, dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError('Expected seven robot joints and one finite door angle')
        return values

    def transform(self, progress):
        """World-frame rigid motion relative to the observed reference door."""
        rotation = Rotation.from_rotvec(float(progress) * self.axis).as_matrix()
        result = np.eye(4)
        result[:3, :3] = rotation
        result[:3, 3] = self.origin - rotation @ self.origin
        return result

    def pose(self, progress):
        return self.transform(progress) @ self.reference_tcp

    def evaluate(self, values):
        """Return residual (6,) and exact Jacobian (6,8), away from SO(3) pi."""
        values = self._state(values)
        robot_pose, robot_jacobian = self.chain.forward(values[:7], jacobian=True)
        robot = pose_matrix(robot_pose)
        desired = self.pose(values[7])
        # The helper uses right/body perturbations.  Desired hinge rotation
        # differentiates to minus the world axis expressed in the current TCP.
        body = np.column_stack([robot_jacobian[3:], -robot[:3, :3].T @ self.axis])
        angular, angular_jacobian = fixed_orientation_residual(
            desired[:3, :3], robot[:3, :3], body)
        position_jacobian = np.column_stack([
            robot_jacobian[:3], -np.cross(self.axis, desired[:3, 3] - self.origin)])
        residual = np.r_[robot[:3, 3] - desired[:3, 3], self.rotation_scale * angular]
        jacobian = np.vstack([position_jacobian, self.rotation_scale * angular_jacobian])
        return residual, jacobian

    def errors(self, values):
        residual, _ = self.evaluate(values)
        return {'position_error_m': float(np.linalg.norm(residual[:3])),
                'orientation_error_rad': float(np.linalg.norm(residual[3:]) / self.rotation_scale),
                'weighted_residual_norm': float(np.linalg.norm(residual))}

    def in_bounds(self, values, tolerance=1e-9):
        values = self._state(values)
        return bool(np.all(values >= self.bounds[:, 0] - tolerance)
                    and np.all(values <= self.bounds[:, 1] + tolerance))

    def satisfied(self, values):
        errors = self.errors(values)
        return bool(self.in_bounds(values)
                    and errors['position_error_m'] <= self.position_tolerance
                    and errors['orientation_error_rad'] <= self.orientation_tolerance)

    def project(self, values, *, fixed_progress=None, max_iterations=100):
        """Bounded numerical projection; a failed fit remains an explicit failure.

        ``fixed_progress`` is useful for finding a candidate endpoint at a
        requested angle.  The ordinary eight-coordinate projection may change
        both robot posture and door angle, as OMPL manifold projection does.
        """
        values = self._state(values).copy()
        low, high = self.bounds.T
        if fixed_progress is not None:
            fixed_progress = float(fixed_progress)
            if not np.isfinite(fixed_progress) or not low[7] <= fixed_progress <= high[7]:
                raise ValueError('Fixed progress must be inside the planned door range')
            values[7] = fixed_progress
            unpack = lambda q: np.r_[q, fixed_progress]
            initial, lower, upper = values[:7], low[:7], high[:7]
        else:
            unpack = lambda x: x
            initial, lower, upper = values, low, high
        initial = np.clip(initial, lower + 1e-10, upper - 1e-10)
        cached, result, derivative = None, None, None

        def both(x):
            nonlocal cached, result, derivative
            if cached is None or not np.array_equal(cached, x):
                result, derivative = self.evaluate(unpack(x))
                if fixed_progress is not None:
                    derivative = derivative[:, :7]
                cached = x.copy()
            return result, derivative

        solved = least_squares(lambda x: both(x)[0], initial, jac=lambda x: both(x)[1],
            bounds=(lower, upper), max_nfev=int(max_iterations),
            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        projected = unpack(solved.x)
        errors = self.errors(projected)
        return {'accepted': self.satisfied(projected), 'state': projected,
                'evaluations': int(solved.nfev), 'optimizer_success': bool(solved.success),
                'correction_norm': float(np.linalg.norm(projected - values)), **errors}


class CoupledHandlePoint(CoupledHinge):
    """A cylindrical-handle contact point, with orientation as an inequality.

    The selected point on the handle and the robot's finger-pad midpoint must
    coincide throughout the hinge arc.  Rotational symmetry at contact is not
    replaced with a fictitious rigid attachment.  The known gripper's
    local up axis must stay inside a cone about the observed vertical bar.
    The cone is a state-validity condition, separate from the three equalities.
    With ``align_axis=True``, two additional equalities keep the gripper up
    axis parallel to the bar while preserving its free axial twist.
    Alternatively, ``jaw_axis_local`` adds one equality keeping jaw closure
    perpendicular to the bar, with no restriction on the gripper up axis.
    """

    codimension = 3
    constraint_name = 'cylindrical_handle_point_with_orientation_cone'

    def __init__(self, chain, origin, axis, reference_tcp, *, pad_local,
                 reference_anchor=None, up_local=(0., 1., 0.),
                 orientation_cone=np.deg2rad(20.), align_axis=False,
                 jaw_axis_local=None, jaw_tilt_limit=None, **kwargs):
        super().__init__(chain, origin, axis, reference_tcp, **kwargs)
        self.pad_local = np.asarray(pad_local, dtype=float)
        up = np.asarray(up_local, dtype=float)
        if (self.pad_local.shape != (3,) or not np.isfinite(self.pad_local).all()
                or up.shape != (3,) or not np.isfinite(up).all()
                or np.linalg.norm(up) < 1e-9):
            raise ValueError('Finite local pad midpoint and nonzero gripper up axis are required')
        self.up_local = up / np.linalg.norm(up)
        self.align_axis = bool(align_axis)
        self.jaw_tilt_limit=jaw_tilt_limit
        if jaw_tilt_limit is not None and (not np.isfinite(jaw_tilt_limit) or not 0<jaw_tilt_limit<=np.pi/2):
            raise ValueError('Jaw tilt limit must be finite and in (0, pi/2]')
        self.jaw_axis_local = None
        if jaw_axis_local is not None:
            jaw = np.asarray(jaw_axis_local, dtype=float)
            if jaw.shape != (3,) or not np.isfinite(jaw).all() or np.linalg.norm(jaw) < 1e-9:
                raise ValueError('Jaw closure axis must be a finite nonzero local vector')
            if self.align_axis:
                raise ValueError('Choose bar-up alignment or jaw-perpendicular contact, not both')
            self.jaw_axis_local = jaw / np.linalg.norm(jaw)
        self.axis_basis = perpendicular_basis(self.axis)
        if getattr(self, 'align_axis', False):
            self.codimension = 5
            self.constraint_name = 'cylindrical_handle_point_with_axis_alignment'
        if self.jaw_axis_local is not None:
            self.codimension = 4
            self.constraint_name = 'cylindrical_handle_point_with_perpendicular_jaw_axis'
        self.orientation_cone = (np.pi / 2 if self.jaw_axis_local is not None else float(orientation_cone))
        if not np.isfinite(self.orientation_cone) or not 0 < self.orientation_cone <= np.pi / 2:
            raise ValueError('Handle orientation cone must be positive and at most pi/2')
        default_anchor = self.reference_tcp[:3, 3] + self.reference_tcp[:3, :3] @ self.pad_local
        self.reference_anchor = np.asarray(default_anchor if reference_anchor is None else reference_anchor, dtype=float)
        if self.reference_anchor.shape != (3,) or not np.isfinite(self.reference_anchor).all():
            raise ValueError('Finite observed handle reference anchor is required')

    def target_point(self, progress):
        transform = self.transform(progress)
        return transform[:3, :3] @ self.reference_anchor + transform[:3, 3]

    def evaluate(self, values):
        values = self._state(values)
        pose, jacobian = self.chain.forward(values[:7], jacobian=True)
        robot = pose_matrix(pose)
        point = robot[:3, 3] + robot[:3, :3] @ self.pad_local
        target = self.target_point(values[7])
        point_jacobian = jacobian[:3] - robot[:3, :3] @ skew(self.pad_local) @ jacobian[3:]
        derivative = np.column_stack([
            point_jacobian, -np.cross(self.axis, target - self.origin)])
        residual = point - target
        if getattr(self, 'align_axis', False):
            up = robot[:3, :3] @ self.up_local
            angular = self.axis_basis.T @ up
            angular_jacobian = -self.axis_basis.T @ robot[:3, :3] @ skew(self.up_local) @ jacobian[3:]
            residual = np.r_[residual, self.rotation_scale * angular]
            derivative = np.vstack([derivative,
                self.rotation_scale * np.column_stack([angular_jacobian, np.zeros(2)])])
        jaw = getattr(self, 'jaw_axis_local', None)
        if jaw is not None:
            angular = float(self.axis @ robot[:3, :3] @ jaw)
            angular_jacobian = -self.axis @ robot[:3, :3] @ skew(jaw) @ jacobian[3:]
            residual = np.r_[residual, self.rotation_scale * angular]
            derivative = np.vstack([derivative, self.rotation_scale * np.r_[angular_jacobian, 0.]])
        return residual, derivative

    def errors(self, values):
        values = self._state(values)
        residual, _ = self.evaluate(values)
        robot = pose_matrix(self.chain.forward(values[:7]))
        up = robot[:3, :3] @ self.up_local
        cone_angle = float(np.arccos(np.clip(abs(up @ self.axis), 0., 1.)))
        jaw = getattr(self, 'jaw_axis_local', None)
        jaw_error = (float(np.arcsin(np.clip(abs(self.axis @ robot[:3, :3] @ jaw), 0., 1.)))
                     if jaw is not None else 0.)
        return {'position_error_m': float(np.linalg.norm(residual[:3])),
                'orientation_error_rad': jaw_error if jaw is not None else cone_angle,
                'orientation_cone_angle_rad': cone_angle,
                'orientation_cone_violation_rad': max(0., cone_angle - self.orientation_cone),
                'jaw_axis_error_rad': jaw_error,
                'measured_jaw_tilt_rad':float(np.arcsin(np.clip(abs(self.axis@robot[:3,0]),0.,1.))),
                'weighted_residual_norm': float(np.linalg.norm(residual))}

    def satisfied(self, values):
        errors = self.errors(values)
        return bool(self.in_bounds(values)
                    and errors['position_error_m'] <= self.position_tolerance
                    and errors['orientation_cone_angle_rad'] <= self.orientation_cone + 1e-9
                    and (getattr(self,'jaw_tilt_limit',None) is None
                         or errors['measured_jaw_tilt_rad']<=self.jaw_tilt_limit+1e-9)
                    and (not getattr(self, 'align_axis', False)
                         or errors['orientation_cone_angle_rad'] <= self.orientation_tolerance)
                    and (getattr(self, 'jaw_axis_local', None) is None
                         or errors['jaw_axis_error_rad'] <= self.orientation_tolerance))


def projected_path(problem, path, *, maximum_joint_step=.02,
                   maximum_progress_step=.02, valid=None):
    """Sample and project a sparse path; report every rejected sample.

    This is a sampled diagnostic, not a continuous collision certificate and
    not a replacement for OMPL's constrained motion validator.
    """
    raw = np.asarray(path, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 8 or not np.isfinite(raw).all() or len(raw) < 1:
        raise ValueError('A nonempty finite path of eight-coordinate states is required')
    if maximum_joint_step <= 0 or maximum_progress_step <= 0:
        raise ValueError('Positive path sampling steps are required')
    result, rejected = [raw[0].copy()], []
    if not problem.satisfied(raw[0]) or (valid is not None and not valid(raw[0])):
        rejected.append({'segment': 0, 'fraction': 0., 'reason': 'invalid_initial_state'})
    for segment, (first, last) in enumerate(zip(raw[:-1], raw[1:])):
        count = max(1, int(np.ceil(max(np.max(np.abs(last[:7] - first[:7])) / maximum_joint_step,
                                       abs(last[7] - first[7]) / maximum_progress_step))))
        for fraction in np.linspace(0., 1., count + 1)[1:]:
            seed = first + fraction * (last - first)
            projected = problem.project(seed)
            state = projected['state']
            accepted = projected['accepted'] and (valid is None or bool(valid(state)))
            if not accepted:
                rejected.append({'segment': segment, 'fraction': float(fraction),
                                 'reason': 'projection_or_state_validity_failed',
                                 **{key: projected[key] for key in ['position_error_m', 'orientation_error_rad']}})
            result.append(state)
    return np.asarray(result), {'accepted': not rejected, 'rejected': rejected,
        'samples': len(result), 'continuous_clearance_certified': False}


def ompl_plan(problem, start, goal, valid, seconds=10., *, kind='atlas',
              constraint_tolerance=1e-5, delta=.025, planner_range=.15,
              interpolation_count=200, seed=1):
    """Plan in a dedicated calling process with OMPL 2.0.1 Atlas/RRTConnect.

    The caller owns process isolation: use a short-lived worker rather than
    importing this wheel into a long-lived GPU/simulator service.  ``valid``
    receives the complete eight-coordinate state, including the door angle.
    It must check both robot and angle-dependent scene collision geometry.
    """
    if kind not in {'atlas', 'projected'}:
        raise ValueError('Expected atlas or projected constrained space')
    if seconds <= 0 or constraint_tolerance <= 0 or delta <= 0 or planner_range <= 0:
        raise ValueError('Positive planner budgets and resolutions are required')
    if seed is not None and seed <= 0:
        raise ValueError('OMPL requires a positive RNG seed; LIBERO seed remains separate')
    start, goal = problem._state(start), problem._state(goal)
    checks, invalid = 0, 0

    def checked(values):
        nonlocal checks, invalid
        checks += 1
        values = np.asarray(values, dtype=float)
        accepted = problem.satisfied(values) and bool(valid(values))
        invalid += int(not accepted)
        return accepted

    for label, state in [('start', start), ('goal', goal)]:
        errors = problem.errors(state)
        if errors['weighted_residual_norm'] > constraint_tolerance:
            return np.empty((0, 8)), {'exact_solution': False,
                'status': f'invalid_{label}_constraint', 'state_validity_checks': checks,
                'invalid_state_checks': invalid, 'solve_seconds': 0.,
                'constraint_tolerance': constraint_tolerance,
                'endpoint_errors': {label: errors}}
        if not checked(state):
            return np.empty((0, 8)), {'exact_solution': False,
                'status': f'invalid_{label}', 'state_validity_checks': checks,
                'invalid_state_checks': invalid, 'solve_seconds': 0.,
                'endpoint_errors': {label: errors}}
    dependency = Path(__file__).resolve().parents[2] / '.runtime' / 'ompl-deps'
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))
    from ompl import base as ob, geometric as og, util as ou

    class Constraint(ob.Constraint):
        def __init__(self):
            super().__init__(8, problem.codimension, constraint_tolerance)

        def function(self, values, out):
            out[:] = problem.evaluate(values)[0]

        def jacobian(self, values, out):
            out[:] = problem.evaluate(values)[1]

    if seed is not None:
        ou.RNG.setSeed(int(seed))
    constraint = Constraint()
    constraint.setMaxIterations(100)
    ambient = ob.RealVectorStateSpace(8)
    bounds = ob.RealVectorBounds(8)
    for index, (low, high) in enumerate(problem.bounds):
        bounds.setLow(index, float(low))
        bounds.setHigh(index, float(high))
    ambient.setBounds(bounds)
    space = (ob.AtlasStateSpace if kind == 'atlas' else ob.ProjectedStateSpace)(ambient, constraint)
    information = ob.ConstrainedSpaceInformation(space)
    space.setSpaceInformation(information)  # Required by the 2.0.1 nanobind wheel.
    space.setDelta(float(delta))
    if kind == 'atlas':
        space.setRho(.15)
        space.setEpsilon(.01)
    space.setup()
    setup = og.SimpleSetup(information)
    setup.setStateValidityChecker(lambda state: checked([state[index] for index in range(8)]))
    initial, target = space.allocState(), space.allocState()
    initial.copy(start)
    target.copy(goal)
    if kind == 'atlas':
        space.anchorChart(initial)
        space.anchorChart(target)
    setup.setStartAndGoalStates(initial, target, 1e-6)
    planner = og.RRTConnect(information)
    planner.setRange(float(planner_range))
    setup.setPlanner(planner)
    before = time.perf_counter()
    status = setup.solve(float(seconds))
    elapsed = time.perf_counter() - before
    exact = bool(setup.haveExactSolutionPath())
    path, native_checked = np.empty((0, 8)), False
    if exact:
        solved = setup.getSolutionPath()
        solved.interpolate(int(interpolation_count))
        native_checked = bool(solved.check())
        path = np.asarray([[state[index] for index in range(8)] for state in solved.getStates()])
        exact = native_checked and all(checked(state) for state in path)
    errors = [problem.errors(state) for state in path]
    metadata = {'planner': f'OMPL {kind} + RRTConnect', 'status': str(status),
        'exact_solution': bool(exact), 'solve_seconds': elapsed,
        'maximum_solve_seconds': float(seconds), 'ompl_path_check_passed': native_checked,
        'state_validity_checks': checks, 'invalid_state_checks': invalid,
        'states': len(path), 'constraint_tolerance': constraint_tolerance,
        'rotation_scale_m_per_rad': problem.rotation_scale,
        'delta': delta, 'planner_range': planner_range, 'ompl_rng_seed': None if seed is None else int(seed),
        'maximum_position_error_m': max((row['position_error_m'] for row in errors), default=0.),
        'maximum_orientation_error_rad': max((row['orientation_error_rad'] for row in errors), default=0.),
        'ambient_dimension': 8, 'constraint_codimension': problem.codimension,
        'manifold_dimension': 8 - problem.codimension, 'constraint_model': problem.constraint_name,
        'continuous_clearance_certified': False,
        'binding_lifetime': 'caller must isolate each invocation in a short-lived process'}
    # Leave nanobind-owned states to the binding; freeing them manually crashes.
    return path, metadata
