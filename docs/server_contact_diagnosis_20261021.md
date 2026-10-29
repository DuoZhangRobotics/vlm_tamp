# State 44 contact-scene diagnosis, 2026-10-21

The fresh state 44 prefix replay isolates a scene-estimation failure: observed
handle samples become stationary obstacles during a scene rebuild. They remain
at the old handle position after the next pull and block finger 1 in the next
planning query. The configured collision checker is detecting real overlap with
its supplied points; the error is their ownership and persistence in the model.

This explains the reproduced rejection. The original run's raw RGB-D and collision
clouds are unavailable, so exact identity of the original offending pixels cannot
be established. The new replay reproduces the original physical state and the
same first-layer finger-1/stationary rejection, supporting this mechanism for the
historical failure.

Environment restoration, state 32's separate contact-loss timeline, and previous
full saved-command replays remain documented in
[the restoration report](server_restore_diagnosis_20261020.md).

## Experiment and matched comparison

These are development diagnostics on previously examined states, with no agent
calls and no changes to numerical production code, thresholds, contact
permissions, physics, controllers, or success predicates. Prefixes use the exact
retained joint references. Recorded sensor operations are repeated on **new**
RGB-D observations. Earlier grasp/arc searches are skipped because their issued
commands are already available; diagnostic continuation queries execute the
unchanged `full_arc_plan` backend. Generated paths are never executed.

The original 20-second search budget, 32 goal proposals, seed 0, and wrist policy
are retained. A wrapper records Python-level validation inputs/results in memory;
it does not alter the native check or native search internals. Wall-limited search
details and original sensor equivalence are not claimed to be identical.

| Case | Prefix endpoint | Native door opening | Continuation queries |
|---|---:|---:|---|
| State 44, learned memory | Step 275, after original event 17 | 19.0623° | Original 10° point and 5° point requests both blocked |
| State 2, learned memory | Step 298, after original event 16 | 16.2501° | Original 10° rigid request solved; additional matched 5° point request solved |

State 2's original complete episode succeeded at step 497. This diagnostic stops
at its comparison checkpoint and is not a new task-success trial. Its opening is
close to, but not exactly matched to, state 44's opening.

Both prefixes issued exactly the retained references. Maximum absolute joint
differences from the historical executions were `1.2311e-11` radians for state 44
and `1.2958e-10` radians for state 2. Maximum native opening differences were
`1.10e-13` and `7.99e-14` degrees, respectively. Neither prefix had any difference
in the per-step sets of native robot contact pairs. Source hashes for all recorded
production planner/service files remained unchanged.

See [comparison.json](../results/server-contact-diagnosis-20261021/comparison.json)
and each case's `protocol.json`, `summary.json`, and retained native action/contact
logs for the exact comparison.

## State 44 failure chain

1. **Grounding, step 60:** native opening is 0.0326°. The model's handle center is
   within 0.31 mm horizontally of the native handle center.
2. **Approach and closure, steps 61–247:** the door moves to 4.2467° while the
   stored model remains at its grounding pose. The handle-center error reaches
   17.61 mm. `RobotService.issue()` synchronizes the articulated scene only for
   arc execution with a held/closing gripper, so this earlier object movement is
   absent from its stored scene pose.
3. **First pull, step 259:** native opening reaches 15.5589°. The fresh RGB-D fit
   accepts a 10.5718° relative rotation. It updates the old model incrementally,
   leaving its handle center 19.86 mm from the current native center. The fit is
   accepted as a pose; its hinge estimate is rejected. There is no hand-motion
   fallback in this part of the replay.
4. **Explicit `apply_tracking`, event 11 at step 259:** the same pose was already
   auto-applied, so the incremental transform is effectively identity. The call
   nevertheless rebuilds the scene from current depth using the lagging contact
   reference. Newly visible handle samples are classified as stationary. The
   stationary cloud grows from 249,225 to 260,403 points; the three decisive
   samples first appear here. This is not evidence that all newly added points
   are incorrect.
5. **Next pull, step 275:** native opening reaches 19.0623°. A second accepted
   RGB-D pose rotates moving/contact points another 3.5420°. Stationary points
   and their roles are deliberately unchanged by `transport_scene()`. The
   misclassified handle points therefore stay behind.
6. **Continuation:** both the 10° and 5° point requests fail at their first 2.5°
   continuation layer. All four candidates are rejected as `1810/0`, meaning
   `gripper0_finger1_collision` against stationary points. The original archived
   5° request has that same first-layer rejection.

The archived episode then opens the gripper and spends the remaining control
budget attempting recovery, ending at 600 steps. That later sequence was already
verified in the earlier full command replay; it was not repeated here.

## Point-level evidence

For the 10° query, all four continuation candidates overlap the same three
stationary samples. Re-evaluating their positions against the retained native
float32 convex halfspaces recovers all three points for every rejected candidate;
there are no unexplained gripper/stationary rejections in the trace. Halfspace
penetrations are approximately **3.00, 2.29, and 0.94 mm**, with zero configured
gripper padding.

The points match raw step-259 samples from `arm_top` pixel `(391, 284)` and
`arm_back` pixels `(391, 162)` and `(390, 161)`, to within `1.51e-8` m after native
float32 conversion. The robot depth prior does not classify those endpoints as
robot pixels: the top-view prior is absent there; the back-view robot surfaces
are approximately 27–29 mm farther away.

At the rebuild, the points are 9.95–13.15 mm from the transported contact cloud,
beyond its 8 mm nearest-neighbor association distance. The other existing
association predicates also leave them stationary. The subsequent static-role
reconciliation retags zero current samples and removes zero historical ghosts.

For an **evaluator-only** identity check, the native XML's handle capsule and the
task BDDL's fixed fixture yaw are combined with private logged hinge angles.
The three points lie within 0.17–0.85 mm of the handle surface at step 259. At step
275, those fixed coordinates lie 6.74–10.11 mm outside the moved handle. Native
geometry is never supplied to the perception code, actor, or planner.

Fresh depth support independently agrees with this interpretation: all three
points have supporting views at step 259. None has support within 5 mm at step
275; two have contradictory free-space evidence from at least two views, and the
third remains occluded/unresolved. This evidence does not authorize clearing
arbitrary obstacles.

The 5° query has five unique stationary samples across its goal and continuation
checks; its first-layer candidates overlap the same three decisive samples.

Evidence:

- [Point analysis, 10° query](../results/server-contact-diagnosis-20261021/state044-prefix/probes/event018/point-analysis.json)
- [Point analysis, 5° query](../results/server-contact-diagnosis-20261021/state044-prefix/probes/event019/point-analysis.json)
- [Private native-handle attribution](../results/server-contact-diagnosis-20261021/state044-prefix/evaluator-handle-provenance.json)
- [Figure](../results/server-contact-diagnosis-20261021/state044-stale-handle-points.png), with an [SVG version](../results/server-contact-diagnosis-20261021/state044-stale-handle-points.svg)

## What the successful comparison establishes

State 2 also moves the door during approach without updating its stored model.
Its handle-center error is 17.59 mm after closure and 17.18 mm at the comparison
checkpoint. Therefore, model lag alone does not explain failure.

Unlike state 44, state 2 does not rebuild the scene during this prefix. Its
253,643 stationary points remain exactly identical to the grounding cloud; it
does not introduce these newly observed handle samples into that cloud. Both
diagnostic continuation requests solve, with no rejected Python-level validation
checks. The distinction is the combination of **model lag, rebuilding point
ownership against that lagging model, and retaining those roles after motion**.

This also separates the issue from state 32's observed contact loss and later
hand-motion prediction. Initial closure failure, a required two-finger pinch,
and hand-motion fallback are not needed to explain this state 44 rejection.

## Implementation status and next experiment

No numerical fix is installed. The baseline remains runnable and its checks
remain active. New source files are evaluator replay, point-analysis, comparison,
and plotting scripts only.

The smallest justified next experiment is an **offline matched scene update**
using the RGB-D and point arrays now retained for these two prefixes. Test
sensor-based registration of the moving/contact reference across approach and
closure, before assigning fresh point roles during a rebuild. Compare the
unchanged baseline and that isolated generic update on the same state 44 and
state 2 observations, then run their unchanged continuation queries. Preserve
bounded contact permissions and retain unrelated stationary geometry. Rejected
visual motion must remain uncertain rather than borrowing hand motion as proof.

The next experiment should establish whether registration repairs ownership
without creating spurious moving regions. It must not delete these particular
three points, widen contact exemptions to force a pass, use native handle poses
for registration, or insert a state-specific recovery sequence. No new agent
cohort is warranted yet.

## Validation, storage, and reproduction

Focused existing geometry, synchronization, depth-support, hinge-tracking, and
storage tests: **22 passed, 1 skipped**. The skip requires an absent historical
saved failure fixture. Ruff and `git diff --check` pass. See
[tests.log](../results/server-contact-diagnosis-20261021/tests.log).

The two case folders retain approximately **775 MB** of logical file contents,
including fresh RGB-D at relevant checkpoints, private metadata, full point-role
arrays, native planner geometry, validation traces, exact issued trajectories,
source/configuration hashes, and verified videos. Run cleanup and verified
lossless compression saved approximately **1.20 GB**. Large diagnostic JSON uses
the repository's existing transparent `.json.gz` reader. Original evidence and
the immutable measured source snapshot were untouched. Details are in
[storage.json](../results/server-contact-diagnosis-20261021/storage.json).

Commands used for the physical prefixes are below. Their output directories now
exist and are preserved; choose new output names to repeat the runs. GPU device
access is required for rendering and the service's cuRobo initialization.

```bash
source scripts/server_runtime_env.sh
python scripts/diagnose_contact_prefix.py \
  --source results/recovery-learning-transfer-20261020/transfer/state044-with_learned_memory \
  --output results/server-contact-diagnosis-20261021/state044-prefix \
  --through 17 --probe-events 18 19
python scripts/diagnose_contact_prefix.py \
  --source results/recovery-learning-transfer-20261020/transfer/state002-with_learned_memory \
  --output results/server-contact-diagnosis-20261021/state002-prefix \
  --through 16 --probe-events 17 --extra-point-probe 5
```

The retained data can be reanalyzed without simulator/GPU execution:

```bash
source scripts/server_runtime_env.sh
python scripts/analyze_contact_prefix.py \
  results/server-contact-diagnosis-20261021/state044-prefix --probe event018
python scripts/analyze_contact_prefix.py \
  results/server-contact-diagnosis-20261021/state044-prefix --probe event019
python scripts/analyze_contact_prefix.py \
  results/server-contact-diagnosis-20261021/state002-prefix --probe event017
python scripts/analyze_contact_prefix.py \
  results/server-contact-diagnosis-20261021/state002-prefix --probe matched-point-probe
python scripts/summarize_contact_prefix.py results/server-contact-diagnosis-20261021
python scripts/plot_contact_prefix.py results/server-contact-diagnosis-20261021/state044-prefix
```
