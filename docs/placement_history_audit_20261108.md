# Historical placement audit and object-point interface repair

The historical success claim was real in the project reports, but it describes a different
experiment from the recent fresh Luna episode. `docs/success_first_regression.md` records 5/5
bowl successes on states 1, 11, 21, 31 and 41, within a 35/35 seven-task regression. The referenced
raw result directories are absent from this checkout and from the supplied 55 MB handoff archive.
The count was therefore checked against retained documentation, not independently recounted from
the original action traces. Do not describe this audit as reproducing the historical score.

## What the earlier runner did

`scripts/run_all_prior_tasks.py` maps `bowl_on_plate` to `libero_goal/put_the_bowl_on_the_plate`.
`src/vlm_tamp/subscription_tasks.py` maps the current `place_bowl` case to
`libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`. State 1 in
one suite is not a matched initial state in the other.

The old task-family runner used cached, RGB/depth-validated Luna role selections and a supplied
manipulation sequence. It was not a fresh Luna policy choosing every robot action. Its placement
machinery included:

1. An RGB-selected object body and destination region. `task_geometry_v4.bowl_spec` estimates a
   lower support reference from the object's measured bounds, and a destination on the selected
   support surface. Its historical default adds 6 mm release clearance.
2. Closure registration and a measured/estimated object-to-hand relation. `bind_bowl_goal`
   expresses the object support point in the measured TCP frame. The reliable runners contain
   an identity-transform substitution when closure registration is unverified; that fallback
   has not been restored here.
3. Explicit lift, transport, lower and release stages in the task-family code. Point IK allowed
   wrist adjustment; the result was not equivalent to pinning an arbitrary hand quaternion.
4. Observed payload geometry and placement-specific contact handling. The old final bowl policy
   also used larger grasp searches and a second fused-cloud proposal pool when the first failed.

The newer direct-action service exposed TCP poses and named robot-contact points, but omitted an
explicit object-point goal. Asking for a finger tip on a plate is not equivalent to asking for
the object's support point there. This is a missing interface capability, not evidence that the
old placement function suddenly began returning incorrect coordinates. It was bypassed entirely.

## Repair

The public `plan_object_point` operation now accepts:

- `source_point_world_m`: the caller's current object material point;
- `target_point_world_m`: the desired destination of that same point;
- `attachment_model="rigid"`: an explicit caller assumption;
- the existing collision policy and planning budget.

The numerical adapter converts the current source point into measured TCP coordinates and uses
the existing cuRobo point IK and VAMP path planner. Wrist/object orientation remains free. The
gripper stays at its current command during execution. Point-arrival feedback is explicitly a
prediction from measured robot pose and the assumed attachment, not observed object arrival.

No object bottom, destination, attachment verification, carried-object collision model, lift
sequence, release, clearance or alternate solver is synthesized. The old task-specific recipes,
CPU IK, identity fallback and large grasp fallback were not restored. The shared contact planner
was factored to reuse the same numerical query/path logic for robot and object point goals.
Existing robot-contact tools retain their behavior.

The operation is registered in request schema v27, the actual service dispatch, structured Luna
operation list, compact tool index, generated help and both feedback formats. Its assumptions
and limits remain in the actor-visible result. Existing surface freshness rules still apply.

## Physical diagnostics

`results/placement-regression-20261108/watch.html` preserves the three checks below and their
videos. All replay the recent Luna object's and plate's selected pixels/regions; no alternate
contact coordinates or task labels were supplied to a model. There were no model calls.
Initial images matched the source recording. Small differences in returned grasp solutions
occurred across processes; these are comparable component diagnostics, not identical-trajectory
counterfactuals or a restoration of the older task cohort.

| Check | Measured result | Native placement success |
| --- | --- | --- |
| Old object-to-TCP conversion, fixed wrist quaternion | No valid IK returned; no placement motion; stopped at control 166 with 434 controls unused | False |
| Object-point constraint, free wrist, operator adapter | Point goal reached within 0.552 mm; release completed at control 233 | False |
| Repaired public object-point tool and executor | Point goal reached within 0.550 mm; release completed at control 232 | False |

The source support estimate in these diagnostics uses the old percentile-bound rule on the
current selected region, and explicitly assumes a rigid relation at grasp arrival. It does not
measure closure-induced object motion. The bowl remains visibly offset from the plate after
release. The submillimetre errors are errors of the modeled point relative to robot FK, not
measurements of the true bowl support point. This distinction explains why they cannot establish
successful placement or correct perception. A rectangle containing background and object motion
during acquisition can invalidate the reference; these contributions have not been isolated.

The fixed-orientation case returned zero IK solutions before any path query, rather than timing
out a path search. The point-only queries use the existing 8 s per-path limit; the public call has
a 30 s total planning budget. Each operator job has a 600 s runtime cap, 600 physical controls,
32 GiB memory limit, no swap, and the existing memory/I/O/SSH monitor. Operators stop after the
declared diagnostic, so unused controls are not recorded as task-budget exhaustion. No fault
was injected. Exact stage/control intervals and wall timings are shown beside each recorded video.

## Validation and remaining work

123 focused tests pass, covering frame invariance, rotation/offset conversion, native failure
propagation, public schemas, contact-tool behavior, gripper preservation and feedback uncertainty.
Scoped Ruff, byte compilation and `git diff --check` pass. No native rebuild was needed. One
existing synthetic contact fixture was updated to declare that it has no Panda sphere geometry;
no production metadata fallback was introduced.

The original five-state placement score was not rerun because its frozen outputs and cached
grounding inputs are absent, and the recent task is different. No fresh Luna episode has yet
evaluated selection of the newly exposed tool. The missing object-point interface is repaired;
full placement reliability remains unresolved. Next validation needs an object reference derived
from the selected body and updated by observed motion, followed by matched task/state evaluation.
Do not promote this candidate over the historical reference or report the component tests as
native placement successes.
