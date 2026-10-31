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
