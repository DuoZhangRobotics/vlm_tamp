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


## Fresh trial results (frozen code 7b325e2)

| Closing state | Native success | Controls | Actor wall | Outside robot service | All IK wall, including first use | Warm IK solve calls | Native approach planning |
|---|---|---:|---:|---:|---:|---:|---:|
| 17 | No | 600 | 335.1 s | 233.3 s | 6.10 s | 1.53 s | 0.087 s |
| 49 | No | 600 | 318.0 s | 235.0 s | 7.26 s | 0.94 s | 0.209 s |

These timing columns overlap; outside-service time includes Luna calls, transport and local analysis, not solely model reasoning. Both episodes exhausted controls, not wall time. The contact pipeline is **not yet reliable**; these results do not establish an improvement in task success.

- State 17: initial pad pushes closed the door partly. Controls 275–372 had no microwave contact. Later contact was arm link 5; the final upward/sideways strokes reopened it. Final opening was 67.4 degrees. All seven issued trajectories completed, with the last skill ending at the control cap.
- State 49: actual palm/pad contacts during approaches moved the door substantially. The intended strokes barely engaged it. Controls 411–600 had no microwave contact, so the final strokes could not make progress. Final opening was 65.3 degrees. Joint-linked calls reported unresolved visual feedback after their first executed strokes; direction-only continuations then moved in free space.

These physical contacts come from operator-only simulator diagnostics. They were not sent to Luna or the planners. The exact division between observation coverage, sphere approximation and controller tracking in the unpredicted approach contacts is not yet isolated. No task-specific correction, contact-loss veto or automatic fallback was added to manufacture success.

One timing qualification: the process stays alive, but a changed calibrated contact offset requires a different solver. State 17's switch from a pad-based stroke to a TCP-based `push_stroke` incurred another first use (0.91 s end to end). Startup is therefore per solver/contact geometry, not strictly once per trial. Acquisition and continuation at an unchanged offset do retain their independent warm solvers.

Resource monitoring: 340 samples; peak RAM 6.60 GiB, GPU 1802 MiB; maximum host full-I/O pressure 1.54%; memory pressure zero; no OOM or resource interruption. Both simulations ended and GPU 0 returned to 2 MiB residency.

See the study's `watch.html`, `failure-summary.html`, `contact-timeline.png` and `outcome-analysis.json` for videos, per-stage evidence and source timing. Each trial also has `evaluator-diagnostics.json`, explicitly excluded from actor input.

## Measured-arrival repair

The subsequent user request requires reaching the commanded approach before pushing and accounting for physical tracking lag at the 20 Hz control rate. The old executor consumed a timed reference list regardless of measured arrival. In state 17 that allowed a push to start with the TCP 74 mm from its approach goal.

The repair keeps the native path and every IK joint solution unchanged. During free approaches, intermediate references wait when measured joint lag exceeds the existing one-control reference increment: 1 rad/s divided by 20 Hz = 0.05 rad. The final approach requires the existing controller reporting tolerance, 4 mm position and 0.05 rad orientation. The executor repeats the same reference; it does not solve another IK, change a target, validate collisions again, or assign simulator state. The default catch-up allowance is 20 extra controls per reference (one simulated second), ending as soon as tracking is sufficient. `approach_wait_steps` exposes that allowance for targeted contact tools. Every extra tick counts toward the existing episode control budget.

An approach that does not arrive returns `approach_not_reached`, with its current-reference error, final-goal error, and actual issued controls. It does not start the following push or gripper closure and does not create a retained contact. Free-pose/approach execution and grasp-sequence approach stages use the same completion contract. IK push strokes also use measured lag feedback; their final criterion is the selected contact point, with no new wrist-orientation constraint. A physically blocked stroke returns `push_reference_not_reached` after attempting motion. Explicit current-hand strokes remain available; no environment planning was added to them.

Recorded reference files include the actual repeated commands, so physical replay remains reproducible. Nominal path commands, issued physical controls, extra catch-up controls and final arrival are separate fields. The webpage displays arrival and explains incomplete execution with numerical errors. Native success still stops motion immediately.

The recorded approaches were physically replayed with zero joint-state discrepancy. Holding state 17's *unchanged* final target for three simulated seconds reduced its position error only from 74.0 to 66.3 mm: this posture was obstructed, not merely waiting for normal settling. State 49's original 8.58 mm endpoint lag fell to 0.67 mm in five controls (0.25 s). The repair therefore must distinguish command completion, settling, and obstruction.

A tested 4 mm condition at every intermediate reference was too restrictive and is **not the shipped policy**. Its evidence remains in `results/measured-arrival-validation-20261101`; the shipped intermediate rule uses the existing joint-reference increment instead. In `results/bounded-lead-arrival-validation-20261101`, the old selected contact requests stop with measured failures and issue no blind push. A reachable clear-space positive control arrives at 2.25 mm in five controls, including three repeats. Three ordinary 1 cm IK strokes all reach their point goals with about 2.6 mm error and no added controls; warm IK remains approximately 40 ms per stroke (`results/arrival-paced-stroke-check-20261101`). This fixes false completion; it does not assert that the old chosen contact postures became physically achievable.

Validation: 342 targeted tests passed, two historical fixtures were unavailable. Fresh Luna runs use `results/luna-measured-arrival-20261101/watch.html`, closing states 17 and 49, unchanged 600-control/1800-second limits, no inherited memory, GPU 0, and the existing 16 GiB/no-swap RAM/I/O monitoring.
