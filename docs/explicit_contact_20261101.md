# Explicit contact and unchanged solver output — 2026-11-01

This continues the removal audit in `heuristic_audit_20261101.md`. The user explicitly rejected a new analytical Jacobian and CPU pose correction, selected explicit contact pixels and attachment models, and chose IK-only push strokes. No new Jacobian or motion-planning algorithm was implemented.

## Active execution boundary

- cuRobo success flags determine IK success. Returned joint values are not clipped, polished with SciPy, or rescued after a failed flag. FK measurements are diagnostic. Native VAMP owns path collision admission.
- Full-pose and arc candidates are ranked by distance from the current joints. Only exact duplicates are removed. The wrapper no longer imposes an eight-goal arc cap or approximate 0.12-radian duplicate filter.
- Point IK has no soft wrist objective or custom optimization time-step clamp. Existing positional accuracy and physical controller limits are unchanged. The numerical problem is supplied to the existing cuRobo API.
- `plan_arc` and `plan_sequence` require `grasp_constraint` explicitly: point, jaw-axis or rigid. Full requested arcs go to MC-VAMP first. Previously authorized 3-degree continuation remains a reported fallback after that attempt. Large-angle coordinate segmentation now occurs at the atan2 half-turn boundary, rather than an arbitrary 90 degrees. Native constraint projection remains part of MC-VAMP; no separate returned-path correction or dense validation is added.
- Targeted push/hook acquisition requires `contact_pixel_xy` and `contact_name`. The pixel is lifted through measured depth without snapping. The name selects a calibrated robot contact point. cuRobo solves that point with free orientation; VAMP plans the free-space approach. No nearby-point, wrist-angle, stand-off or robot-only fallback menu remains in this route.
- A controlled push starts from the measured hand position after approach. It uses straight Cartesian contact-point targets, one current-state cuRobo seed per waypoint, and the existing timed joint controller. There is no planned-push alternative or automatic second insertion. Simulator contacts determine the physical outcome. VAMP self checking remains the single robot self check in the point-IK adapter.
- `continue_push` requires retained contact. `push_stroke` explicitly requests an unassociated current-hand stroke. Neither tool silently substitutes the other. Feedback uncertainty and stall measurements do not shorten a requested direction stroke.
- Gripper calls retain the existing arm-controller reference; they no longer invent an inward arm correction. Sequence closure no longer stops solely because tracking is unresolved. Gap/object-motion measurements remain evidence, not automatic regrasp policy.

## Geometry ownership and remaining scope

Public `ground` now uses the exact contact depth sample and the line through two caller-selected hinge depth samples. It passes raw selected contact points to grasp generation. It does not fit a vertical cylinder, force a vertical hinge, complete a door plane, choose a mount, or infer task direction from surrounding geometry. A supplied image arrow selects a sign; otherwise the ordered hinge samples only define coordinates. A pixel line is still the actor's hinge hypothesis, not recognition of the hidden physical hinge.

Multiview grounding reports independent depth/reprojection disagreement without replacing the primary hypothesis or applying the former 10 mm/5 degree/4 pixel consistency vetoes. Validity checks still require finite, available depth and correctly associated current images. Masks preserve holes; proximity matching cannot relabel pixels excluded by the caller.

The existing 3 mm observation representation and calibrated robot geometry remain. Unselected observed points remain obstacles. Current measured-robot occupancy is removed with zero padding, as previously requested; hypothetical goal poses do not erase obstacles. Fingers remain exempt against all environment surfaces; arm/palm and upstream self checks remain. Explicit palm contact is target-specific.

Existing tracking algorithms still have correspondence/fit-quality criteria, and the optional motion-based hinge-estimation tool still declares an upright-model scope. Those criteria are not new planning vetoes, calibrated uncertainty guarantees, or evidence that a failed physical action is impossible. This change does not claim to replace all perception algorithms. Native MC-VAMP's finite-difference task callback and its numerical tolerance are unchanged. No analytical-Jacobian investigation is part of this work.

Historical offline fitters/planners remain available for reading saved experiments; they are not fallbacks of the registered Luna contact route. Supplied tool knowledge stays separate from learner findings. Fresh evaluation actors receive no inherited notebook or successful coordinates.

## Timing and resource checks

The original service smoke test's four contact IK calls took 2.203 s, 10.9 ms, 9.5 ms and 9.4 ms end to end. Calling the 2.23 s sum simply “IK generation” concealed first-use initialization. The worker now records solver creation, first-use solve wall time, warm solve wall time, and upstream-reported solve time separately. Acquisition and continuation have independent persistent cache slots; switching between them does not evict the other solver. Changing the calibrated contact offset replaces the corresponding slot, rather than accumulating solvers indefinitely.

A real GPU test on restored arcla GPU 0 returned 31 full-pose solutions, 42 point-acquisition solutions and a successful point continuation, with no joint-result modification. A 1 cm simulated service stroke executed, taking 4.04 s including observation; it is a plumbing test, not a microwave-success result.

The fresh study is `results/luna-explicit-contact-ik-push-20261101/watch.html`: closing states 17 and 49, sequential fresh Luna/high actors, 600 controls and 1800 wall seconds each, no inherited memory. Runs retain the 16 GiB memory cap, zero swap and existing RAM/I/O watchdog. The dashboard preserves video selection and exposes the selected pixel/contact, stroke method, nested IK/approach planning, execution and observation timing. Final outcomes are recorded in that study, not inferred from unit tests.

Warm physical repeats completed: the second and third 1 cm strokes spent 38.8 ms and 39.1 ms generating all four Cartesian IK targets. Total tool time was 1.820 s and 1.808 s; execution plus capture was 1.666 s and 1.654 s. Instrumentation located the initial delay inside the first solve call (2.166 s), while solver construction itself took 39.3 ms. Subsequent solve calls were 8.4–10.0 ms. Source: `results/explicit-contact-warm-strokes-20261101/check.json` and `attempt01/curobo-queries.json`.

Validation before fresh trials: 317 targeted tests passed; two historical recorded-data fixtures were unavailable. The Codex read-isolation integration check passed separately in the host systemd environment (it cannot run inside the nested tool sandbox). Real GPU alternation between acquisition and continuation reused both cached solvers with zero setup time; measured reused calls were 10.8 ms and 10.3 ms end to end.
