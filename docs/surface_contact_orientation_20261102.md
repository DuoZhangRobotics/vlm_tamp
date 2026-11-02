# Explicit surface-contact orientation

The point-only contact tool left wrist orientation free. A numerically correct
point could therefore correspond to an unsuitable physical hand posture. The
user approved an explicit point-plus-surface-normal model on November 1/2.
This is an optional model, not an automatic fallback or a task-specific action.

## Contract

`push`, `hook_pull`, and `plan_contact_pose` accept
`contact_constraint="surface_normal"` and a signed outward
`surface_normal_world`. The robot face comes from the selected calibrated
`contact_name`. cuRobo aligns that face against the supplied surface normal and
leaves twist free. The implementation creates a contact frame whose local Z is
the robot face normal, then uses native goal-frame weights `[1,1,1,1,1,0]`.
Existing numerical tolerances and returned joint solutions are unchanged.
Acquisition and continuation solvers remain cached separately for each model.

The normal selects a face and orientation. It does not select an opening/closing
sign or a motion direction. A world direction or explicitly associated joint
increment is still required. A joint-linked stroke rotates its normal using the
same declared joint increment while its contact-point waypoints remain a straight
chord. Accepted RGB-D motion transports the same material face; it is never
re-flipped toward a new camera. Unresolved tracking remains unresolved.

Completion matches the chosen model: point-only approaches check the selected
contact point; normal-constrained approaches/strokes check that point and its
normal, with the existing 4 mm / 0.05 rad controller tolerances. Twist is not an
extra completion condition. Retargeting responds to measured target movement;
it does not run another IK search to polish residual error on an unchanged target.
Contact-pose plans default to holding their planned gripper aperture. Explicit
aperture changes require replanning because the contact offset would change.

No CPU IK correction, analytical Jacobian, posture menu, collision buffer, dense
returned-path validation, or automatic normal-to-point fallback was added.
Free approaches use VAMP RRTC through the existing MC-VAMP adapter. Push strokes
remain direct cuRobo IK. Grasped arcs retain MC-VAMP CRRTC.

## Communication

Previews identify the robot face role (inner jaw, distal pad, body-mesh reference,
or palm), the chosen normal, initial robot contact's signed plane offset, and the
motion component into/away from that face. Candidate camera sides are geometric
view information, not visibility or reachability certificates. The normal and
motion arrows are rendered together. Measured orientation reports carry control
steps and surface-pose freshness.

`measure_surface` supplies a camera-oriented normal without replacing the active
scene. `select_surface_contact` installs a surface scene. Local normal support,
spatial spread and fit residual are exposed; no whole-mask completeness gate is
added. A partial mask can support a plane, while an incorrect depth association
can still describe background geometry instead of the intended moving part.

Opening/closing requires a goal relationship to the fixed structure. Unsigned
plane angles are 0–90 degrees and fold larger angles; camera-facing normal angles
also depend on which material faces were selected. Neither alone certifies task
progress. The existing material-motion report now exposes the observed point
translation, with its measurement interval, instead of requiring the actor to
extract it from a 4x4 transform. No hand-motion substitute is used.

Compact contact-search failures now preserve native endpoint rejection causes,
including robot part, observed surface and representative sphere coordinates.
Previously the wrapper displayed `unknown` for these nested native reasons.

## Evidence before Luna trials

- GPU-only checks: 6–30 returned solutions across current/opposite/arbitrary
  signed faces; maximum observed normal error 0.323 degrees; maximum position
  error 1.65 mm under the existing 3 mm numerical tolerance. Warm calls took
  about 6–11 ms. Initial acquisition was 2.30 s; initial local continuation
  1.17 s. No robot motion was requested. See
  `results/contact-normal-ik-check-20261102`.
- Same old Luna-selected inner-pad inputs: state 17's normal-model endpoints
  were rejected by native palm/arm collision checks before motion; state 49's
  normal approach stopped 52.2 mm short during physical contact. Normal alignment
  alone did not make those selections usable. See
  `results/contact-normal-matched-20261102`.
- State 49, observed front face/end pad: approach plus a 2 cm inward stroke
  repeated 3/3, moving the door from 107.26 to 111.91 degrees open. This validates
  contact execution, **not closing**. See `results/contact-normal-face-checks-20261102`.
- State 17, observed outer handle face: approach reached in 3/3 and the push
  interval closed from 87.65 to 84.49 degrees (initial scene 88.92 degrees).
  The requested 2 cm stroke stopped short. This is repeatable partial closing
  motion, **not full task success**. See `results/contact-normal-handle-face-check-20261102`.
- Rejected closing-side candidates are visualized in
  `results/contact-normal-rejections-20261102`. A 24 mm palm sphere overlaps
  observed handle points by 3.43 mm for the end-pad candidate; the higher inner-pad
  candidate overlaps observed top-edge points by 5.47 mm. These are native
  spherical-model overlaps, not proof of exact physical palm penetration.

## Collision-aware IK (November 2)

**Current scope after the user's clarification:** environment collision checking
applies to acquiring the initial contact pose and its approach, not to subsequent
push strokes. A push uses direct native cuRobo IK initialized from the previous
joint solution, with robot self-collision and existing joint/speed limits retained.
No observed collision world is prepared, transported, or queried for that push.
Acquisition and push optimizers have separate persistent caches; disabling the
push's environment checker cannot disable a later acquisition. Hook-pull and
grasped-arc behavior was not changed by this clarification.

The initial implementation described below applied environment collision costs
too broadly to push continuation. Its measurements remain historical evidence.

The earlier contact and full-pose IK calls had environment and self-collision
disabled. Their success flag meant kinematic convergence; native VAMP subsequently
rejected colliding endpoints. The user explicitly requested collision-aware IK.

The v2 worker now optimizes with the same 59 robot spheres and self-collision pairs
as the pinned VAMP model. Environment queries use exactly the existing sampled
observed points, with the same stationary/moving/contact ownership. cuRobo's native
zero-size cuboid distance is the distance to a point. This uses upstream distance
and gradient kernels without reconstructing a mesh, adding occupied thickness,
or introducing a custom IK/gradient solver. The robot base and fingers are exempt
from environment queries, as in VAMP. Explicit palm contact exempts the palm only
against the selected contact partition. Arm and other palm collisions remain active.

The obstacle storage, collision masks, and contact solvers remain resident. Large
particle batches are partitioned only to respect Warp's signed 32-bit kernel-index
limit; every configuration still sees every point. Collision activation distance,
point radius, sphere padding and self-collision padding are zero. No post-IK repair,
extra returned-path check, posture gate, or fallback was introduced.

Free approaches and grasp selection configure the shared scene before solving IK.
Arc goals rotate observed moving geometry by the requested angle. Direct push IK
transports the moving target with the requested stroke while retaining stationary
geometry. This prediction assumes contact; RGB-D feedback remains the measurement.
Direct strokes do not construct a VAMP path planner. The redundant local CPU self
check was removed because cuRobo now uses the same native sphere pairs itself.

Validation:

- 224 targeted CPU/native tests passed; two historical-recording fixtures skipped.
- 14 native GPU tests passed, including exact boundary distances, gradients,
  finger/palm permissions, and geometry/permission updates under CUDA graph replay.
- Across 235 configurations in three saved scenes, GPU/native self and scene
  admission agreed in every case. Maximum sphere-coordinate discrepancy was
  2.06 micrometres. See `results/curobo-collision-ik-zero-20261102`.
- The two rejected state-17 panel contacts returned no successful IK solution;
  warm searches took approximately 1.46 s. This is a finite optimization result,
  not proof that all postures are infeasible. The previously reachable handle
  contact returned 22 warm solutions, all admitted by VAMP, in 15.8 ms end to end.
- Full-pose and zero-angle arc smoke checks returned 31 solutions each; the arc
  sphere/contact-frame checks agreed to 0.24 micrometres. No motion was executed
  by those checks. See `results/curobo-collision-arc-check-20261102`.

The first adapter test accidentally retained cuRobo's default 1 cm cached box
dimensions. Boundary regression tests caught this; both stores now explicitly
zero their dimensions. The earlier `curobo-collision-ik-20261102` and
`curobo-collision-physical-20261102` results are marked invalid for zero-padding
comparisons and preserved for audit. Use the `-zero-` reruns above and below.

The corrected physical replay is **not a task success**. In
`results/curobo-collision-physical-zero-20261102/state017-handle-outer-face-r1`,
the approach reached at control 130 (6.50 simulated seconds), with 3.61 mm contact
error. The requested 2 cm stroke produced a large change between IK postures;
the scheduled references span 2.55 radians in one joint and were truncated to the
470 remaining controls. After advancing 20 references and waiting another 20
controls, execution stopped at control 171 (8.55 s): the last issued reference
was 23.6 mm away, normal error 0.0707 rad, joint lead 0.0562 rad. Stop reason:
`push_reference_not_reached`. No additional collision veto ran in execution.
The door went from 88.92 to 87.36 degrees open; 104.6 s of skill time remained.
The recorded execution is `attempt01/initial/rollout.mp4` in that result folder.

This exposes an unresolved continuity issue: collision-free IK endpoints do not
make their joint interpolation a straight usable contact trajectory. The next
diagnostic is native cuRobo continuation/seed selection at that stroke. No new
continuity cutoff, alternate solver or motion-planning fallback was added to
conceal this failure. Full Luna task evaluations have not been repeated with
collision-aware IK. The preceding point/normal-interface trials, before this
change, failed both states 17 and 49 at 600 controls.

## Corrected push scope and matched replay

`results/curobo-approach-only-push-20261102/watch.html` compares the same observed
contact, starting state, direction and 2 cm request with the previous run.
The approach was identical: reached at control 130 with 3.61 mm contact error.
Environment checking was true for acquisition and false for all seven push IK
queries; native self checking remained true. All seven push solutions succeeded.
The largest adjacent IK joint change was 0.0306 rad (1.75 degrees), and only
14 nominal push controls were needed, versus the previously truncated 470.
Warm IK calls took 10–11 ms; first-use initialization took 1.24 s. Total stroke
planning was 1.31 s, versus 1.79 s with the environment constraint.

The robot executed the push: the door moved from 87.65 to 84.49 degrees open
during the stroke (initial scene: 88.92 degrees). It did not complete the requested
2 cm hand motion: after 14 references and 20 extra settling controls, the measured
contact remained 19.05 mm short. It stopped at control 164 (8.20 simulated seconds)
with `push_reference_not_reached`. The orientation normal error was 0.0389 rad,
within the existing 0.05 rad tolerance. No IK rejection, environment veto, path
planner, or extra collision validation stopped the push. Approximately 105.6 s
of the 120 s skill allowance remained. This is partial physical closing, not full
task success. Video: `state017-handle-outer-face-r1/attempt01/initial/rollout.mp4`.

132 targeted tests passed for this scope correction, including solver-cache
separation, keeping acquisition collisions enabled, previous-posture seeding, and
ensuring a push does not prepare or query an environment world. No new continuity
limit, fallback, seed menu, posture correction, or controller change was introduced.

## Native local IK continuation

The next audit found that the method called "local" disabled cuRobo's LM seed
stage and forced its particle/L-BFGS optimizer at every waypoint. One supplied
seed did not prevent particle sampling. Sorting the single returned candidate
also could not establish that it was the nearest possible configuration.

Continuation now calls the pinned SDK's native `seed_ik_solver.solve_single`
with exactly one supplied seed: the preceding joint configuration. The result
passes through the SDK's standard result evaluation using
`solve_pose(..., run_optimizer=False)`. That evaluates the existing pose, joint
limit and self-collision conditions without modifying the LM result or running
the main optimizer. Both native success flags must pass. A failed local solve
is returned directly, with no sampled retry, CPU correction or substitute pose.
The same point/normal criteria and existing numerical tolerances are preserved.
Acquisition retains its collision-aware batch. Push continuation still has no
environment collision constraint. The local solver and CUDA graphs stay cached.

The corrected call follows the incremental pattern used in
[Bullet's IK](https://github.com/bulletphysics/bullet3/blob/master/examples/SharedMemory/IKTrajectoryHelper.cpp)
and [MoveIt's Cartesian interpolation](https://moveit.picknik.ai/main/api/html/cartesian__interpolator_8cpp_source.html).
This follows a nearby local solution; it is not a proof of a global minimum over
all IK branches. No new runtime joint-jump threshold or timing constraint was added.

Validation: 133 targeted tests passed, plus three real GPU tests. The GPU tests
forbid invocation of the main optimizer, exercise seven successive nearby targets
for both point and surface-normal models, and verify that an unreachable request
fails without a retry. The CPU tests also inject an LM failure while the metrics
would accept its output, ensuring that a failed solver result is not rescued.

Matched physical replay: `results/curobo-local-lm-push-20261102/watch.html`.
The initial approach again reached at control 130. All seven push IK calls
succeeded; maximum adjacent joint change was 0.0254 rad (1.46 degrees), versus
0.0306 rad before the local-solver correction. Both used 14 nominal push controls.
Warm worker calls took 3.3–3.7 ms (native LM computation about 1.1 ms), versus
10–11 ms before. First local use took 0.809 s; the complete stroke computation
took 0.834 s versus 1.310 s. The whole skill took 14.13 s versus 14.37 s.

This remains a partial-closing diagnostic, not full task success. The door moved
from 87.65 to 84.48 degrees open during the push, with no IK failures or collision
veto. At control 164 (8.20 simulated seconds), the measured contact was 19.09 mm
short of the final reference after the existing 20-control settling allowance.
The stop reason was `push_reference_not_reached`; 105.9 s of skill time remained.
The remaining physical contact/target tracking issue is separate from IK branch
selection. Video: `state017-handle-outer-face-r1/attempt01/initial/rollout.mp4`.
Full Luna task evaluations were not rerun for this local-solver change; validation
focused on native numerical behavior and the matched physical push.

## Closing-oriented direction: three matched operator replays

The user then requested a test in the closing direction. Only the requested world
direction changed; the contact pixel, robot face, orientation constraint, distance,
speed, controller, tolerances and runtime code stayed identical. This is an
operator input comparison, not a Luna-selected action or learned policy.

The operator selected an opaque, flat door-rim patch in public RGB-D
(`arm_lower_right`, pixel `[300,318]`, region `[297,314,304,323]`). Its measured
inward normal, approximately `[0.999816,0.019185,0.000065]`, supplies a closing-
oriented push. This approximates the local closing motion; it is not an exact
hinge-derived tangent. No simulator hinge coordinates or object geometry selected
the command. The curved handle's normal still specifies the robot contact face.
These diagnostic pixels and vectors are not supplied to Luna or promoted to memory.

`results/curobo-closing-direction-push-20261102/watch.html` contains all three
videos, full inputs/outputs, IK traces and time breakdowns. Every replay returned
`stroke_executed`: seven local IK solutions succeeded, 14 nominal controls ran,
and no additional settling controls were required. The hand advanced 18.09 mm
along the 20 mm command; final 3D tracking error was 3.81 mm, within the unchanged
4 mm completion tolerance. Maximum adjacent IK joint change was 0.0210 rad.
The approach was unchanged and reached at control 130; the push ended at control
144 (7.20 simulated seconds). Three identical-state replays establish only local
repeatability.

Evaluation-only simulator angles changed from 87.65 to 83.64 degrees open during
the push: 4.01 degrees of closing. The microwave was not fully closed. RGB-D
object-motion estimation still reported unresolved in all three runs, so successful
robot execution must not be confused with validated visual progress feedback.

Total skill wall times were 17.41, 13.30 and 14.07 s; contact search took 7.34,
3.23 and 3.61 s. Stroke computation including first local use took 0.85–0.92 s,
with warm IK calls around 3.3 ms. Peak measured job memory was 4.25 GiB. Host-wide
full I/O pressure peaked at 18.02%, while eleven fresh SSH checks succeeded; the
run completed without a resource stop. Host pressure does not identify which
workload caused the I/O. No runtime code changes or new unit tests were needed
for this input-only experiment.

## Fresh Luna evaluation after local IK correction

The user requested new trials on states 17 and 49. Results are in
`results/luna-local-ik-evaluation-20261102/watch.html`, with event-level explanations
and both videos in `failure-summary.html`. Runtime commit `7361cc1` includes the
native local IK fix. Each case used a fresh Luna/high actor, 600 controls, 1800 wall
seconds, no inherited notes, no worked example and no operator coordinates. Both
memory audits and the 623-file/native-binary integrity checks passed. No runtime
source, prompt or configuration changed during the study.

Neither task succeeded. These are two development-state episodes, not an estimate
of a general success rate:

| State | Initial → final open angle | Controls | Episode wall | Actor wall |
| --- | --- | --- | --- | --- |
| 17 | 88.92° → 122.21° | 600 | 256.9 s | 332.3 s |
| 49 | 107.26° → 122.15° | 600 | 166.3 s | 223.8 s |

Angles are evaluator-only simulator measurements, excluded from actor inputs.
Actor wall time includes setup and writing final reports outside the episode.

In state 17, Luna selected the visible face and repeatedly requested world −X.
Its first 15 cm `push_stroke` opened the door from 88.36° to 122.71° over controls
160–280. Later requests kept −X. Two initial-contact approaches also failed to
physically arrive, with 32.4 and 87.7 mm final reference errors. Luna's final notes
interpreted some image changes as progress toward closure; native angles contradict
that interpretation. The raw notes are preserved, not promoted to validated memory.

In state 49, Luna first selected a background pixel as one hinge endpoint, noticed
the implausible depth, and revised the annotation. Its selected-part plane still
had a 52 mm RMS fit residual. The revised closing sign remained a caller hypothesis.
The actual commands used explicit +Y, not the hinge model to generate an arc.
The first push ended at 120.15° open; the continuation kept +Y and exhausted the
budget at 122.15°. That final call issued 355 controls, including 282 repeated holds,
and advanced only 72 of its 100 nominal references. It ended 67.7 mm from its
planned final reference. RGB-D motion feedback had too few consistent visible
correspondences and did not establish progress toward the task goal.

Numerical continuation remained sound in these runs: 117/117 local IK queries in
state 17 and 74/74 in state 49 succeeded, without sampled optimization or push
environment checks. Median warm calls were 3.11 and 3.07 ms. Maximum adjacent
push-IK joint changes were 0.0109 and 0.0036 rad. Neither actor selected the optional
surface-normal contact model. No grasp or arc planning was requested.

Service calls totaled 96.7 s (state 17) and 64.4 s (state 49). Outside-service time
was 235.6 and 159.5 s, including Luna, transport and local analysis; it is not an
isolated reasoning measurement. Nested IK time was 6.44 and 3.63 s; native free-
approach path search was 0.195 and 0.108 s. These nested times overlap service time.
Peak job RAM was 6.12 GiB, peak GPU memory 1885 MiB, and host full I/O pressure
peaked at 14.24%. All 39 fresh SSH probes succeeded; no resource interruption occurred.

The observed failure pattern is wrong closing-direction selection and lack of
correction from progress evidence, alongside the physical approach/holding issues
above. These trials do not implicate a local IK failure, and they do not establish
whether the deeper cause is the actor's visual interpretation or the way public
goal/progress evidence is communicated. The next focused diagnostic should test
that interpretation without supplying successful coordinates or action sequences.

## Communication audit and recorded-video access

The user requested an audit of actual inputs/feedback before another 17/49
evaluation. The earlier actors received the correct task instruction. State 17
requested six visual inspections, including three before/after comparisons, six
overviews and one history lookup. State 49 did not call that local comparison
helper. Neither called `track_video_motion`; no direct RGB video-inspection command
was exposed then. These facts do not establish that the model cannot recognize
opening/closing. State 17 did receive past images, but some feedback captures had
shrunk to agentview/wrist and its interpretation of the visible movement was wrong.

Confirmed communication defects and repairs:

- Per-operation push help described only point IK even though the schema/index
  offered the surface-normal model. Help now documents both and distinguishes
  contact orientation from desired task motion.
- The compact grounding reply dropped the measured 52 mm plane-fit residual and
  literal hinge depth samples. Those fields are now retained. Public geometry
  reports task intent separately and does not echo "close" as the meaning of an
  unverified positive joint angle. A positive contact tangent is calculated only
  from the caller's declared geometry, with no task sign inferred.
- Motion replies now lead with separate hand execution, object-motion status and
  task-progress status. A large hand displacement cannot become a goal-progress
  claim. Stale object measurements carry their original control step.
- Feedback keeps the requested view, fixed agentview/frontview context and wrist
  within the existing four-view cap. Each physical action supplies a paired
  fixed-camera before/after sheet from existing captures. No joint angle or
  semantic opening/closing label is added to those images.
- `inspect_video(start_control_step, end_control_step, max_frames=8,
  region_xyxy=null)` exposes bounded chronological recorded agentview RGB. It
  reads live JPEGs or the completed MP4; it does not step physics, render new
  sensors, interpolate depth or read private object state. Exact-step requests
  return a frame only if recorded. Maximum 12 frames per reply; original images
  and control timestamps remain available. A motion comparison provides a ready
  interval for optional video lookup.

Generic guidance asks the actor to identify the desired moving-part/fixed-reference
relationship, distinguish motion toward a pivot from rotation about it, and inspect
evidence before extrapolating an uncertain direction. It supplies no world-axis
choice, successful pixel, hinge location, state-dependent trigger or action script.
No IK, planner, collision permission, controller tolerance or budget was changed.

Validation before evaluation: 139 targeted tests passed, plus the real isolation
test in the same host environment used by Luna. The physical preflight in
`results/communication-preflight-20261102` completed the previously tested short
stroke, generated a fixed agentview/frontview comparison and returned eight
requested video frames with no additional controls. This preflight was an operator
test; its coordinates and results are excluded from the fresh actor inputs.

All diagnostic pixels/settings above were operator inputs, excluded from Luna
prompts and memory. Native object angles/contacts are evaluation-only evidence.

The preceding frozen point-model study completed four clean practice episodes
and four fresh evaluations: 0/4 practice success; 0/2 with learner notes and 0/2
without on held-out states 8 and 35. Notes were frozen verbatim and explicitly
acknowledge no successful method. See
`results/luna-contact-handoff-practice-v2-20261101/watch.html`. These results do
not evaluate the new surface-normal model.
