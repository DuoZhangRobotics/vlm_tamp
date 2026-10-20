# Full-arc partial-pose planning

The microwave runner now defaults to short completed motions with this
partial-pose backend; see `docs/short_planned_hinge.md`. This document describes
its full-arc option, selected with `--motion-policy full` or `--full-task-only`.
Frozen comparisons and their source snapshots are retained. The original
backend is still available as `--backend endpoint_menu`.

## What is optimized

The gripper pad midpoint follows the full perceived handle circle. All seven
Panda joints are optimization variables. There is no prescribed wrist
quaternion and no orientation menu. Jaw tilt is bounded by 60 degrees; the model
does not fix two individual material contact points or certify no-slip grasping.

Joint-space seeds are projected onto the partial-pose goal. Local palm-convex
and native-arm-sphere clearance objectives help propose postures, then a pure
equality projection restores the contact target. These objectives are not the
admission authority: native MC-VAMP validates the endpoints and every returned
connection using the observed clouds and configured gripper.

The backend first tries C-RRTC directly to admissible full-angle joint goals.
It can also search intermediate arc waypoints with free wrist orientation and
C-RRTC connections. All connections are assembled into a complete arc before
motion. These are internal planning waypoints, not online execution segments.
Failure to connect the complete arc produces a planning failure; the partial
arc is not executed. The output preserves which search succeeded or failed.

## Contact roles

The selected-handle nearest-observation association is 8 mm, matching the
existing selected-surface fusion tolerance. The prior 6 mm threshold left a
thin observed handle shell tagged as static. This is a geometric association,
not a guarantee of exact object identity when neighboring surfaces are close.
Arm and palm still collide with this geometry; only configured fingers receive
the existing selected-handle contact permission.

When retaining old observations, historical static samples are reconciled only
when they are within 8 mm of the current selected handle, current RGB-D moving
association agrees, and a current moving observation supports them within
4 mm. The current moving surface stays in the collision model. Occlusion alone,
or a hypothetical robot pose, cannot clear stationary obstacles. In state 11,
the previous three flagged pad samples are corrected by the current association;
the additional historical reconciliation removed zero samples.

## Estimated door motion and uncertainty

Moving observations are rigidly rotated around the estimated hinge during
planning. This prediction is conditional on that hinge, cloud ownership, grasp
retention and rigid-door assumptions. Collision-free in this model is not a
certificate for the true simulated scene.

For parallel hinge axes with perpendicular origin error `delta_h`, the point
error is `(I - R(theta)) delta_h`; its magnitude is
`2 sin(theta / 2) ||delta_h||`. A 10 mm offset therefore causes 14.1 mm error at
90 degrees. Axis tilt and progress error are separately evaluated as sensitivity
probes, not calibrated probabilities. Extra blanket padding is not introduced.

`evaluate_hinge_prediction.py` runs only after the policy search. It uses the
simulator hinge for evaluation, comparing predictions of the same observed
partial cloud. No simulator hinge, identity or scene mesh is read by the policy
backend. The policy result hash is checked before and after evaluation.

## Reproduction

Use the existing isolated research environment and limit CPU math threads:

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python scripts/run_mcvamp_microwave.py \
  --output results/my-full-arc-test \
  --tasks open_microwave --states 11 \
  --backend partial_pose_arc --reconcile-handle-roles --full-task-only \
  --replay-grasp-source results/mcvamp-microwave-frozen-20261015/open_microwave-state011 \
  --arc-proposals 96 --arc-seconds 80
```

`--replay-grasp-source` is an explicit development diagnostic. Initial RGB,
proprioception, predicted handle cloud and approach admission are checked.
Only the original approach and physical gripper closure are reused. The original
20-degree pull is never executed. After closure, a fresh complete remaining arc
must be found. No new physical attachment is created. The normal fresh-grasp
pipeline is used when this flag is omitted.

No API calls are made by this experiment. Luna/high decisions remain cached
under the shared $2 ledger. Every issued timed trajectory is followed in full
with simulator contact physics and zero execution-time collision, constraint,
progress or tracking-error vetoes. The existing 600-control-step limit and
automatic post-run retention remain active. Native LIBERO predicates decide
task success, and pre-motion planning failures stay in the trial denominator.
