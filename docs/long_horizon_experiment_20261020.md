# Long-horizon execution and scene synchronization

The pre-experiment checkpoint is `792474c95bd6750380780386eaf2f739213342da`
on GitHub. Changes are on `codex/grasp-consistent-long-horizon`.

## Scope and rules

The user authorized autonomous debugging, long physical experiments, and
extension from microwave opening to bowls, drawers and stoves. Establish
repeatable success before optimizing planning calls, trajectory length and
execution time. Use native LIBERO success, stop immediately on success, add no
execution-time collision veto, retain all failures, and clean each run. The API
cap is $2 per experiment, not cumulative. Subscription Luna practice uses high
reasoning and its isolated public workspace; implementation code, human
diagnostic solutions and simulator object state remain unavailable to it.

## Implementation

- `scene_pose_sync.py` transports the moving and contact geometry after an
  executed arc, preserving stationary observations and point roles. RGB-D rigid
  tracking is preferred; measured-hand motion is an explicitly labeled fallback
  that assumes retained contact. The hinge is not silently refined.
- The native microwave adapter and Python goal projector support `point`,
  `rigid`, and `jaw_axis` contact models. Rigid orientation is transported with
  the estimated mechanism; the contact-line model leaves rotation about the
  closing axis free. The legacy wrist-excursion heuristic is not imposed on
  these two new modes. The working point mode remains available.
- `refined` is documented accurately as the legacy point-mode wrist allowance,
  not hinge estimation.
- Failed arc requests expose current-query endpoint-projection, collision and
  search diagnostics to the actor. These are not proofs of infeasibility.
- The family regression runner stops established numerical engines on native
  success and records the complete requested trajectory, issued prefix and
  tracking. It retains existing family priors and does not claim that the
  microwave acting agent transfers to other tasks.

## Matched microwave component results

Source: `results/subscription-success-first-continuation-20261019/attempt01`.
The initial sensor arrays and post-grasp joint state were verified against that
source before replaying only the approach and closure prefix. Subsequent plans
were generated afresh. This is one development scene, state 47, seed 0, with
600 control steps maximum; it is not held-out generalization.

| Condition | Native successes | Control steps | Manipulation plans |
|---|---:|---:|---:|
| Preserved point-mode 15° baseline, old cloud lifecycle | 1/1 | 329 | 6 |
| Synchronized point-mode 15° | 3/3 | 339 each | 6 each |
| Synchronized point-mode 90° | 3/3 | 330 each | 1 each |
| Synchronized rigid-mode 15° | 0/1 | 321 | 2, then planning failure |
| Synchronized rigid-mode 90° | 0/1 | 228 | 0 |
| Synchronized jaw-axis mode 90° | 0/1 | 228 | 0 |

The first synchronized short and long runs took about 161 s and 56 s,
respectively, including simulator startup and physical execution but no actor
reasoning or fresh VLM grounding. These are diagnostic timings, not a formal
hardware-isolated speed benchmark. New rigid-mode paths independently satisfy
their declared orientation constraint; the unsuccessful endpoints had large
projection residuals before collision admission.

Results and exact source snapshot:

- `results/grasp-horizon-physical-20261020`
- `results/grasp-horizon-repeatability-20261020`

All three repeats of each successful condition terminated immediately on native
success. There were no control steps after first success.

## Physical interaction audit

Native contact logs were read only after the episodes, outside the actor.
At the end of the shared closure prefix, the pads contact each other and the
mean absolute finger-joint position is about 0.58 mm. This does not establish a
rigid pinch attachment. The successful long motion subsequently opens the
microwave through physical contact; its palm contacts microwave geometry during
27 manipulation steps. No unrelated-object contacts were recorded during that
long manipulation. The short conditions had no palm/microwave contact.

The full long plan permits about 168° of orientation difference from a rigid
attachment frame at some planned states. Therefore native success must not be
reported as verification of a maintained rigid grasp or collision-free physical
execution. The native opening predicate terminates the run at roughly 75° of
actual door opening, before the entire requested 90° command sequence is sent.

The point-cloud model still omits unobserved space and door-versus-environment
collisions. Automatic rigid transport is not fresh reconstruction of all depth
surfaces. In the tested short runs, tracking was insufficient and the fallback
was labeled as predicted hand motion.

## Validation and continuation

69 relevant regression tests passed, including scene transport, orientation
Jacobians, storage, adaptive-practice phase gating, success termination and
actor-visible failure diagnostics. A separate independent audit checked the
orientation contract of saved native rigid paths.

The next numerical cohort is
`results/success-first-family-baseline-20261020`: bowl placement, middle and top
drawer opening, top drawer closing, and stove activation, initially state 1.
The subsequent agent practice is configured as 5 attempts initially, extendable
to 20, with success-first phase gating and a 10,800-second session ceiling:
`results/subscription-scene-sync-practice-20261020`.
These paths name planned work; consult their result files for actual completion.

## Subsequent complete-sequence experiments

The first seven-task/five-state cohort finished with 32/35 native successes.
Its failures were bowl placement on states 21 and 31, and middle-drawer opening
on state 21. Both state-21 cases already had admitted complete manipulation
paths during grasp selection. A separate composition executed those paths
after closure instead of replanning from the newly estimated scene.

That comparison recovered both failures. Initial sensor arrays, selected grasp
candidates, and every approach/closure reference were exactly equal to their
failed counterparts. The prepared execution then passed 15/15 family cases on
states 1, 11 and 21, and 10/10 on states 31 and 41. Rotary controls retain their
existing observed direction probe; cases without a full admissible motion
retain the short-path fallback. Bowl preflight retreat points are excluded:
execution ends at placement/release or earlier native success.

A final uniform 35-case cohort is recorded separately in
`results/prepared-final-all-tasks-20261020`. Its result file is authoritative
while it runs; the earlier cohorts are not combined into a synthetic score.

## Public sequence tool and agent continuation

`plan_sequence` plans the approach and complete requested manipulation before
motion. `execute_sequence` checks the whole sequence's remaining-step budget,
then executes approach, physical closure and manipulation. It checks freshness
before starting, permits its own stage transitions, and does no collision
admission or replanning after closure. Native success cancels all remaining
stages immediately. The physical public-interface smoke test succeeded at 340
steps with one prepared manipulation.

Successful planner replies now retain concise scalar summaries and point to
full numerical JSON artifacts. The first agent practice used seven attempts,
with five native successes. Its later attempts used a neutral gripper command
that accidentally bypassed scene synchronization. That trigger is now corrected
for both continuing-close and neutral-hold commands; the earlier outcomes remain
unchanged and their efficiency interpretation is explicitly qualified.

The continuation in
`results/subscription-prepared-sequence-practice-20261020` is limited to thirteen
additional attempts, preserving twenty combined attempts. It receives the prior
agent's abstract principles and an automatic whitelist of its own relative tool
parameters. Image regions, world poses, joint paths, plan IDs and raw historical
replies are excluded. The note describes the tool correction and asks the agent
to test longer prepared sequences after current-version success is established.
Human numerical solutions and the successful smoke-test trajectory are withheld.

The acting agent has been evaluated on microwave opening. The broader family
cohorts use established numerical adapters and inherited geometry priors. In
particular, the microwave contact proposals selected in these runs come from
the existing RGB-D frontal-handle generator within the hybrid grasp tool.
These results do not establish that Luna learned every family or that GraspGen
alone produced the successful contacts.
