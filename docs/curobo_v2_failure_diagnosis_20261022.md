# Why the cuRobo v2 migration failed — 22 October 2026

Latest: [integrated continuation and fresh simulator success](curobo_v2_continuation_20261022.md).

Follow-up: [3 mm tolerance and a successful local-continuation prototype](curobo_v2_tolerance_and_continuation_20261022.md).
The original diagnosis below is retained; the later prototype now connects
the previously failed saved state, but has not yet been integrated or executed.

The saved-input regression and the fresh-physics replay failed for different
reasons. Neither is evidence that Luna made a bad decision or that cuRobo v2
cannot solve this robot. The migration changed the numerical goal generator;
MC-VAMP still connects goals and performs final constrained-path admission.

[Open the diagnostic report](../results/curobo-v2-failure-diagnosis-20261022/report.html)
with VS Code Live Preview. This is an observer report, not a robot interface.
The [migration report](../results/curobo-v2-migration-20261022/report.html)
retains the original runs and profiling.

## The controlled comparison

Use the exact observed cloud, calibration, start joints, estimated hinge,
30-degree requested arc, and native checker from failed physical replay event
10. Change only the endpoint generator. The legacy generator produces three
goals in **3.217 s**; the first connects in **0.219 s**, with 72 sampled states,
no collision rejection, and 36.70 degrees maximum wrist excursion against the
63-degree limit. No simulator or robot motion is used in this comparison.

The first legacy endpoint is 0.563 radians from the start in seven-joint
Euclidean distance. The nearest admitted v2 endpoint is 4.576 radians away.
This is a posture-distance diagnostic, not a single joint angle or path length.

Evidence: [endpoint crossover results](../results/curobo-v2-failure-diagnosis-20261022/endpoint-crossovers002/results.json),
[generator timing](../results/curobo-v2-failure-diagnosis-20261022/endpoint-crossovers002/new-state-legacy-goals.json),
[reproduction script](../results/curobo-v2-failure-diagnosis-20261022/cross_endpoints.py).
This proves a feasible sampled constrained connection exists in the supplied
planning model. It does not prove that executing it would complete the task.

## Failure 1: premature projection termination, now corrected

On the original saved event-10 state, the first three v2 endpoints passed the
native sparse connection check immediately (two sparse nodes, zero search
iterations), but dense path reconstruction returned `dense_projection_failed`.
An isolated instrumented native build found radial residuals of
1.028–1.132 micrometres, just above the strict 1-micrometre constraint threshold,
with no collision rejection in those connections.

Upstream MC-VAMP can terminate projection when the **squared joint update** is
at most 1e-6, while the separate task residual has not yet reached 1e-6.
Resuming from the intermediate solution reduces the three failing residuals to
0.015–0.076 micrometres. This distinguishes premature numerical termination
from an infeasible contact constraint.

The production native adapter now permits up to three projection resumptions,
requiring finite, strictly improving residuals and retaining the original
cumulative projection-distance cap. The final 1e-6 residual threshold,
collision checks, and wrist checks are unchanged. Upstream source is unchanged.

All three complete paths now pass in 0.605–0.682 s, and the original legacy
path still passes in 0.238 s. Every returned sample was rechecked for collision
and for already satisfying the constraint without further projection changes.
The broader five-positive-case GPU replay now passes **5/5**, in
7.612, 4.709, 4.499, 4.175, and 4.114 s including scene setup and final validation.
The earlier final migration sequence passed 4/5.

Evidence: [native production regression](../results/curobo-v2-failure-diagnosis-20261022/projection-refine/production-results/results.json),
[five-case replay](../results/curobo-v2-failure-diagnosis-20261022/saved-positive-projection-fix/results.json).
Code: [native resampler](../src/vlm_tamp/native/mcvamp_contact_numerical.cc),
[reusable integration checker](../scripts/validate_native_projection_recovery.py).

Limit: independent projection can increase adjacent joint spacing above the
raw 0.006-radian interpolation spacing. The recovered v2 paths have maxima
0.038, 0.037, and 0.057 rad. These are sampled collision checks, not a continuous
collision certificate. Do not describe the raw interpolation spacing as a
bound on final projected path spacing.

## Failure 2: distant goal postures and palm collisions

In `physical-replay002-query-state/attempt01/010-arc`, goal generation took
2.609 s and admitted four endpoints. Connection attempts consumed about
17.60 s and failed under their search deadlines. The original calls reported
89, 151, 122, and 37 moving-geometry rejections. No second-pull trajectory was
executed, and the full-arc continuation fallback received no time.

The instrumented replay identifies the rejected robot part as the **palm**
(`part=1000`, verified against the scene's gripper mapping), against moving
noncontact geometry and the selected handle/contact region (`surface=1,2`).
This is not an intended finger/handle contact being incorrectly forbidden.
The endpoints themselves pass; intermediate attempts to reach the distant
postures collide. A search deadline does not prove those endpoints are
unreachable by every possible path.

Evidence: [per-call traces](../results/curobo-v2-failure-diagnosis-20261022/traces/results.json),
[original failure receipt](../results/curobo-v2-migration-20261022/physical-replay002-query-state/attempt01/010-arc/result.json).

### Incorrect posture objective

Our v2 worker requests `seed_velocity_weight=.01` but supplies no timestep.
In this pinned upstream implementation, the seed solver resets its velocity
reference to zero and its timestep to one; it only loads the actual current
posture when a timestep is present. Its velocity residual remains active when
the weight is positive. Consequently the configured term penalizes distance
from **zero joint angles**, not distance from the current posture. The main
rollout's intended velocity regularization is inactive without a timestep.

The diagnostic GPU run reads the actual internal buffers and confirms seven
zero reference angles. Only one supplied IK seed is the measured start; the
other 63 seeds are global samples. Sorting outputs by distance afterwards
cannot recover a nearby solution that the optimizer did not retain.

Evidence: [original new-state GPU probe](../results/curobo-v2-failure-diagnosis-20261022/objective-probe/new_state/current/result.json),
[upstream reference initialization](../.runtime/curobo-v0.8.0/curobo/_src/solver/seed_ik/seed_ik_solver.py),
[upstream residual](../.runtime/curobo-v0.8.0/curobo/_src/solver/seed_ik/seed_ik_error_calculator.py).

Removing this penalty alone does **not** solve the fresh-state case: no
endpoint survives all native admission checks. Adding an arbitrary timestep
also introduces velocity-dependent joint bounds, so it is not a clean fix for
an untimed endpoint IK query. These remain diagnostic variants, not production
changes.

### Different collision geometry

The GPU fills occupied 5-mm voxels around observed points. Native palm narrow
phase checks measured points against the calibrated convex palm with zero
extra padding. The resulting proposal geometries differ materially near
contact. For a palm sphere at the known-good legacy endpoint:

| State | Clearance to measured moving point | Clearance to 5-mm occupied voxel |
| --- | ---: | ---: |
| Original saved state | +2.751 mm | -2.789 mm |
| Failed replay state | +4.027 mm | -1.419 mm |

Negative values are proxy overlaps. CPU geometry reproduces the GPU collision
costs. The finer 2-mm moving mesh removes these particular overlaps, while
stationary resolution remains 5 mm. This comparison establishes disagreement
with native admission, not ground-truth free space or calibrated uncertainty.

The GPU also adds `contact_reference` as an obstacle. Native code deliberately
uses that transported handle estimate only for association and queries current
moving measurements as obstacles. However, the measured overlaps above come
from current moving points, so removing transported points alone cannot explain
or repair them.

Evidence: [geometric reconstruction and pitch sweep](../results/curobo-v2-failure-diagnosis-20261022/legacy-goal-proxy-geometry.json),
[native comparison of objective variants](../results/curobo-v2-failure-diagnosis-20261022/objective-validation/results.json),
[surface variants](../results/curobo-v2-failure-diagnosis-20261022/objective-validation-surface/results.json),
[resolution variants](../results/curobo-v2-failure-diagnosis-20261022/objective-validation-resolution/results.json).

Both stale-reference removal and finer resolution were tested independently
with and without the wrong posture penalty. They do not, by themselves, restore
the fresh-state connection. A verified configuration defect or geometry mismatch
must not be described as the sole cause of the remaining failure when its
isolated correction does not restore success.

### Local sampling is not sufficient either

The final diagnostic kept the 64-seed budget but supplied 32 local seeds
(the measured start and bounded 0.15-radian perturbations) plus 32 global
seeds, with zero seed-velocity penalty, current moving observations, and the
2-mm moving mesh. It produces eight admitted goals and a valid connection on
the older state. On the newer state it produces **zero** admitted goals:
51/64 are outside the 2-mm equality-polish capture range; the remaining 13
fail both scene collision and the 63-degree wrist limit after polishing.
These categories overlap as stated, rather than representing 26 different
candidates. All checks use the same current native policy.

Evidence: [local-seed validation](../results/curobo-v2-failure-diagnosis-20261022/objective-validation-local/results.json).
The GPU currently optimizes endpoint position, weak orientation preference,
and approximate collision geometry. It does not impose the native wrist limit
and full partial-pose/contact rules inside the optimizer. Native rejection
afterward is necessary, but cannot make the candidate generator search the
right constrained posture region.

## What is fixed and what comes next

Only the proven native projection repair is promoted to production in this
diagnosis. The GPU objective, mesh, and local-seed changes remain explicitly
labelled experiments because they do not restore the newer-state connection.
The original production worker is unchanged. Do not report this as a complete
end-to-end recovery or as a repaired GPU goal formulation.

The next implementation should preserve the current posture branch during
goal generation and express compatible wrist/contact constraints, then rerun
the newer-state crossover and all five positive cases before another physical
replay. An incremental constrained IK continuation is a concrete candidate;
it must receive reserved planning time instead of being starved by full-goal
searches. Use the existing CPU goal as a feasibility reference, not hidden
task information for Luna. Increasing the timeout alone does not repair the
measured objective/geometry mismatches.

## Reporting failures from now on

Report the stage, actual rejecting check, supporting measurements, whether any
motion executed, and the controlled comparison used for causal attribution.
`search_budget_exhausted` describes how work stopped, not why a proposal failed.
Use per-event collision traces to identify the dominant robot part; the last
rejected part alone may be stale. Distinguish an attempted continuation angle
from reached or executed progress. Do not count failed dense reconstruction as
zero work merely because the existing native timing fields are filled only on
success.

The diagnosis used saved observations and robot calibration. It did not expose
private simulator state to Luna, add reflection, retrain any model, or run a new
adaptive Luna practice episode. The fresh-physics replay has not been repeated
after this numerical correction; the original failure remains archived.
