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

### Arrival-repair trial outcomes and terminal-accounting correction

The fresh measured-arrival trials used frozen commit `56e9bed`. Both physical episodes reached 600 controls without native success. State 17 reached none of its four targeted approach goals; state 49 reached none of its three targeted approach goals. **No automatic push followed an unreached approach in either run.** Luna separately selected current-hand strokes. Final native door openings were 108.6 degrees (17) and 32.9 degrees (49). Catch-up consumed 244 and 205 real control steps respectively. This is a completion/feedback repair, not a demonstrated contact-acquisition solution or task-success improvement.

A reporting edge case was found in state 49: the final request had 11 nominal references and 11 controls remaining. Four catch-up steps meant only seven nominal references advanced before the 600-control cap. The worker incorrectly labelled that partial result `trajectory_complete`; the study validator correctly noticed that its summary was inconsistent and marked a driver error. The executor now reports `control_budget_exhausted` and `terminated_on_control_budget`, and the episode summary separately counts budget-terminated trajectories. All original trial records remain unchanged.

The exact recorded command prefix and final request were physically replayed in `results/arrival-budget-edge-replay-20261101`. The prefix joint-state difference was 8.05e-13 rad and final difference 9.55e-14 rad. The corrected worker stopped at the identical 600-control physical state and reported the budget termination accurately. This verifies an accounting correction; it is not a fresh Luna trial or a successful outcome. Forty targeted checks passed after this correction.

The ordinary 1 cm stroke check still used seven controls per stroke, no catch-up, approximately 2.6 mm endpoint error, and 40 ms warm IK. Reference-rate peaks in the two fresh trials were 1 rad/s as configured; measured joint-speed peaks were approximately 0.567 and 0.656 rad/s. The reference limit is not a claim that physical arrival is guaranteed within one 50 ms tick.

A separate diagnostic used the existing explicit-quaternion contact API to align a fingertip with the measured surface normal at the same recorded pixels. It did not solve acquisition: state 17's 18 IK endpoints were rejected by VAMP for observed arm-link 4/5/6 collisions; state 49's 22 endpoints were rejected mainly at link 6. That particular full-orientation construction was **not adopted** as a hidden heuristic, menu or fallback. More general contact-posture construction remains unresolved.

The updated webpage is `results/luna-measured-arrival-20261101/watch.html`; its `failure-summary.html` links the per-stage arrival/contact timeline, `arrival-outcomes.json`, and the terminal-replay proof. Resource monitoring over 445 samples found peak RAM 6.64 GiB, GPU 1822 MiB, maximum host full-I/O pressure 0.65%, no memory pressure and no OOM kills. The original state-49 collector error remains visible for provenance, with the verified correction documented beside it.

## Inspecting Luna's original pixel selections

The observer webpage now includes `Luna’s pixel and hinge selections` and a standalone `results/luna-measured-arrival-20261101/selections.html`. Choose a trial and event to inspect the exact source capture, pixel coordinates, hinge endpoints, requested direction, selected SAM mask, and the grounding-model event requested by a joint-linked push. Zoom and mask toggles preserve original-image coordinates. Images load on demand and the selection view does not auto-refresh.

Yellow contact crosses and purple hinge points come from Luna's logged request arguments. Blue marks are explicitly labelled projections of world/hand references; `push_stroke` and `continue_push` do not invent a new selected pixel from carried metadata. The input image remains separate from an optional link to the unannotated result capture. No annotation, perception, planner, controller or Luna prompt was changed by this viewer work.

An optional evaluator hinge line is hidden by default and is exported outside the actor workspace for human comparison only. It uses the native hinge at the capture's exact control step and existing camera calibration. It is not input to the actor or planning. To regenerate this study's viewer and human-only reference, run `python scripts/publish_luna_selections.py --study results/luna-measured-arrival-20261101 --evaluator-joint microwave_1_microjoint` in the server runtime environment. Omitting `--evaluator-joint` adds no new evaluator reference.

Useful examples: state 17 event 5 shows contact pixel `(280, 200)` on the visible door; later current-hand requests repeatedly chose `[-1, 0, 0]`. State 49 event 4 shows its early table-point/hinge hypothesis, but it was superseded before pushing. State 49 event 10 uses pixel `(210, 380)` in `arm_wide_back` and explicitly requests model epoch 4 from grounding event 7 in `arm_wide_front`. The viewer links these events so abandoned hypotheses are not blamed for later motions.

Validation includes request/capture provenance, continuation-versus-new-pixel distinction, model-epoch links, mask ownership, camera projection and unchanged source images. Browser checks confirmed exact overlay alignment at 200% zoom, lazy image loading, and separate mask/evaluator toggles. The main HTML remains under 5 KiB; the recorded selections live in per-trial detail JSON, not a repeatedly rewritten embedded-image page.

## Moving-target handoff repair

The read-only readiness audit found a separate service integration bug: after a
successful robot approach, accepted RGB-D tracking could move the selected
material point while the first stroke still started at the obsolete hand pose.
The deterministic reproduction moved the target 100 mm after approach; the old
service issued a 10 mm stroke from the old location. This is a control-flow
reproduction, not attribution of every real approach failure to this branch.

Targeted acquisition now measures the selected robot contact against the updated
material point. If accepted tracking leaves it outside the existing 4 mm arrival
tolerance, the service approaches that same updated point with the same selected
contact name, cuRobo IK and native free-path planner. It updates the moving cloud
and reuses the static scene, fixed joint parameters and tracker. Subsequent
searches use the existing five-second allowance and remaining skill/episode
budgets. There is no new clearance, contact guess, pose menu, IK correction or
returned-path validation. Acquisition motion is excluded from requested stroke
progress. A failed reacquisition returns its cause without issuing a stroke from
the obsolete target. Unresolved tracking remains explicitly unresolved; it does
not create a new veto on direction-only physical experiments or continuations.

`contact_handoff` reports each measured target/finger pair, error and reacquisition.
It appears in the compact actor reply and watch dashboard. Reaching a point still
does not certify physical engagement: point IK leaves wrist orientation free.
The existing explicit-quaternion `plan_contact_pose` remains available; no
surface-normal alignment heuristic was introduced. Practice instructions now
explicitly separate observations, unverified hypotheses and tested procedures.
No previous evaluation notebook is supplied to clean practice.

Validation covers moving targets during initial and repeated acquisition,
stationary/unresolved targets, failed reacquisition, local search expiration,
unchanged fixed hinge/static map, correct stroke progress origin, and the 600-step
budget. Physical qualification and practice evidence are recorded separately
under `results/contact-handoff-physical-check*-20261101` and
`results/luna-contact-handoff-practice-20261101`; interrupted startup is not scored
as a robot task failure.

### Configured pad faces versus passive mesh tips

Physical qualification exposed another interface defect. The old advertised
`finger*_collision_distal_tip` describes the body mesh. In this robot XML that
mesh has `contype=0, conaffinity=1`; the microwave surfaces are also passive.
The pad boxes have `contype=1, conaffinity=1`. Three matched requests reached the
body-tip target and issued a short stroke, yet produced no native door contact.
This is not a failed IK or a measured engagement certificate.

The robot-only geometry export now carries its configured collision masks.
Existing contact names remain available and are labelled. New
`finger*_pad_collision_distal_face` points use the actual calibrated pad box's
distal face center. The tools do not change simulator masks, native collisions,
VAMP spheres, solver settings, or automatically select a different surface.
Luna sees these as supplied robot/tool knowledge, not a discovered task policy.

In `results/contact-handoff-active-pad-check-20261101`, all three matched
approaches reached the selected pad pose (3.45 mm TCP error), followed by real
pad/door contacts at controls 73–94. The deliberately downward 5 mm probe stopped
against the top surface with 5.49 mm final robot error. That is a physically
blocked normal press, not a door-closing demonstration. The 24 native contacts
per replay are operator-only evaluation evidence, never actor input. The prior
misaligned diagnostic selected a front-face point 8 mm below the top while
keeping a downward wrist pose; its video and 7.16 mm stopped-goal evidence remain
in `results/contact-handoff-physical-check-v6-20261101`.

### Cold-start I/O diagnosis

After the host restart, the existing I/O-percentage watchdog interrupted runtime
initialization before any requested approach/push. Tracing showed cold dependency
imports from NFS, including SciPy. Bulk library caching alone was insufficient.
Direct SSH checks stayed responsive (roughly 0.34–0.46 s) during the reads. One
bounded initialization-only check, with a 16 GiB/no-swap cap, memory/I/O sampling
and direct SSH checks, completed in 34.08 s at 2.64 GiB peak job memory and zero
memory pressure, despite 27.34% I/O wait. I/O wait is not by itself a measured
login outage. The subsequent physical checks kept the existing experiment
watchdog. See `results/contact-handoff-startup-readiness-20261101`; no task success
is assigned to startup checks or interrupted initialization.
