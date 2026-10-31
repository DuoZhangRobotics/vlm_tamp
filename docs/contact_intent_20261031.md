# Explicit motion intent and acquisition corrections

The previous physical-push cohort completed at 6/12 native task successes;
closing state 17 was 1/3. Its first closing run exposed two interface problems:
an omitted direction silently became the selected face's inward normal, and a
legacy acquisition rule rejected the later explicit opposite direction before IK.
Positive movement along that normal did not establish progress toward closing.

## Current contract

- Targeted `push` and `hook_pull` require either `direction_world` or a signed
  `joint_degrees` increment with the associated model. This includes previews.
  A missing direction is an incomplete request; the tool no longer invents it.
  `select_surface_contact` remains the geometry-only point/normal query.
- `push_stroke` and `continue_push` retain their direction/distance interface
  and physical execution policy. No new geometric margin or execution gate was added.
- `reference_direction_world` replaces `suggested_direction_world`. It describes
  the previous request transported by observed part rotation, not a task policy.
- Compact replies expose `measured_object_motion_along_request` and its units/scope.
  The full record retains `observed_progress` for historical readers. Neither
  quantity certifies progress toward the task goal; native success is separate.
- Reversing an open-hand stroke can withdraw from a surface. Contact placement
  and desired direction remain policy decisions rather than assumed attachment.

## Acquisition and diagnostics

The camera-facing normal no longer vetoes a push direction. Its alignment is
reported as geometric evidence. Existing push postures near the selected surface
can be reused regardless of the requested direction, before new scene admission.

Controlled-push preview planning now matches execution: the free approach is
scene-planned, while contact insertion uses robot-valid IK references and physics.
It no longer rejects the intended contact endpoint or adds a dense environment
validation pass to an insertion that will execute physically. Equivalent scaled
direction vectors reuse the same preview.

Hook aspect ratio is descriptive rather than a hard 1.8 cutoff. Other detector
support requirements and observed retaining-surface checks remain; failed
conditions are returned with measured values. A detector failure is not a proof
that the physical object cannot be hooked.

Pixel errors identify the camera dimensions and requested region before expensive
work. Rectangle boundaries at the image edge are accepted consistently. Depth
support failures state the eligible/finite point counts. Dashboard explanations
prioritize the actual candidate rejection or shape condition over generic timing.

## Recorded-case checks

For event 44 of the first closing-state-17 trial in
`luna-physical-push-rerun-20261031`, the original code rejected all five points by
`wrong_side_contact` in about 1 ms, before IK. The corrected request found an
approach in 2.02 s in the saved-data probe. A matched physical prefix reproduced
control 437 with 5.8e-15 rad maximum robot-joint difference; the explicit preview
took 3.99 s, was reused by execution, and the hand moved 26.99 mm for a 30 mm
request. Execution took 9.62 s including approach and captures. This demonstrates
that the request is now attempted, not full task closure.

Event 45's 1.73-aspect-ratio feature now passes the shape/back-surface stage.
Its four tested hook postures still fail predicted arm/finger endpoint checks.
No physical hook success is claimed. The remaining cause is reported explicitly.

Regression validation: 205 passed, three missing historical-fixture skips.
The new clean-actor cohort uses the same Luna model, 600-control/1800-second limits,
GPU/EGL device 3, a 16 GiB memory limit, zero swap and 2-second resource monitoring.
Diagnostic directions, pixel regions and recorded prefixes are not supplied to Luna.

Runtime page: `results/luna-explicit-motion-rerun-20261031/watch.html`.
Repair evidence: `results/push-intent-validation-20261031/index.html`.

## Stroke feedback and contact positioning repair

The later `luna-explicit-motion-rerun-20261031` recordings exposed a separate
execution bug. Opening state 37, event 57 requested 100 mm in 20 mm segments.
Each segment reset the one-second physical-stall history before it could fill;
the remaining-distance loop repeatedly restarted from the blocked measured
position. That call used 349 controls and 64.71 seconds without opening the door.

Sub-strokes now advance finite waypoints on the original requested path. Tracking
lag is compensated toward the next waypoint; it does not create an unlimited
sequence of replacement strokes. The same physical-stall history crosses
sub-stroke boundaries. No environment admission check, collision padding, force
sensor, or new execution threshold was added. A new public call starts a new
history so an actor can deliberately attempt a different recovery.

Contact acquisition applies its existing hand/scene check before IK. An arm
branch cannot resolve collision of a fixed hand pose. Duplicate standoff queries
are cached, and target IK waits for an accepted standoff. Contact insertion and
controlled strokes still use physical execution. GraspGen remains unchanged.

`plan_contact_pose(point_world_m, quaternion_xyzw=None, contact_name=None)` accepts
an observed contact point and computes the calibrated fingertip/pad-to-TCP
offset. It tries current/nearby orientations, or the supplied fixed orientation,
and the closest scene-valid IK per candidate. It returns a normal plan for
`execute(..., gripper=0)`. This operation is registered in callable schemas,
compact tool discovery, and detailed help; `plan_pose` retains its TCP interface.

Closing state 17 showed why face-normal alignment cannot establish task progress:
the initial +X request reduced the evaluator's opening angle from 88.9 to 69.2
degrees, but Luna reversed after interpreting a geometry warning as a task
direction judgment. The warning now explicitly lacks task-direction/contact
evidence, and stale alignment predictions are removed on continuation. Replies
retain the last measured part change separately from an unresolved later fit.
These diagnostic directions and native angles are not supplied to fresh actors.

Tracking advances its image references after accepted motion while retaining the
original world frame and other camera references. A failed physical-push fit can
retry with two additional reference views; this does not gate execution. Explicit
camera selection no longer has a four-view cap inconsistent with full captures.
Rigid feedback retains the 4 mm residual, eight-inlier and 25 mm span requirements,
but requires a strict majority rather than an arbitrary 60%: the observed 11/19
consistent matches were previously rejected. Tied-motion regression cases remain
rejected. The last closing stroke now yields measured motion with 2.09 mm RMS.

Matched evidence is in `results/push-feedback-repairs-v2-20261031` and
`results/push-feedback-closing-v5-20261031`. The opening prefix reproduced the
recorded robot joints exactly. The blocked call returned at control 339 with
261 controls retained, in 9.64 seconds. The two old push searches fell from
14.47/10.88 to 5.43/1.25 seconds; the new contact-point planner found a plan in
0.72 seconds. The closing diagnostic physically reduced the opening angle from
88.92 to 68.55 degrees and now measured the final movement. These are targeted
regressions, not full-task success or fresh-policy repeatability claims.

The evaluator `scripts/validate_push_feedback_repairs.py` keeps transient RGB-D
in its own local temporary run and archives small inspection artifacts. Service
cleanup is restricted to that service's own output root, retaining finished-run
and video verification requirements. Peak sampled memory was 5.24 GiB in the
combined check; the final closing check had zero full I/O and memory pressure.
