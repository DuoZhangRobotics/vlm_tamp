# Planner range recovery: close microwave state 17

The original motion-assisted Luna episode maintained its grasp but stalled at control 380. The repaired tools now plan and physically execute the previously failed 3° point, 7° jaw-axis and 15° rigid requests. These are engineering replay results, not evidence that Luna learned a skill or that the entire task is reliable.

Open the existing [watch.html](../results/luna-live-hinge-comparison-20261027/watch.html) with VS Code Live Preview. Its new **State 17: planner repair and physical replay** section contains the measurements, time breakdown, videos, tests and fresh Luna follow-up. The original six-trial cohort remains unchanged below it.

## What actually blocked planning

| Module | Measured evidence | Old restriction | Repair |
|---|---|---|---|
| `curobo_v2_arc.py` candidate admission | Rigid local seed missed position by 3.755 mm; jaw-axis seed by 3.575 mm | Failed cuRobo seeds above 3 mm were discarded before CPU correction | Refine finite seeds and judge the corrected result |
| `partial_pose_arc.py` wrist policy | One point-goal seed had 0.046 mm input position error and 29.607° total hand rotation; native endpoint checks passed | Hidden 8° wrist excursion cap on short requests | No extra cap by default; caller may request an explicit bound |
| `curobo_v2_arc.py` local connection policy | A valid endpoint could require a larger joint change | A 0.6 rad joint-change veto skipped native connection search | Optional explicit limit; otherwise attempt a checked complete path |
| Partial-pose and native orientation constraints | At 12° progress, the rigid solution had 0.424 mm positional and 0.703° orientation error | Weighted angular tolerance `1e-6`, equivalent to about 0.000382° in rigid mode | Explicit angular tolerance, applied consistently in CPU endpoint projection and native path admission |
| Joint bounds | Endpoint optimizer and path checker used different joint margins | Projection could return a posture the path checker rejected | Both use the existing native 5 mrad buffer within physical limits; the old projection-only 25 mrad margin is removed |
| Search scheduling | Many nearly identical endpoints were refined before any connection | Candidate generation could consume the deadline | Connect the first admissible endpoint immediately; after failure, reuse the GPU batch and explore remaining seeds |

The final 15° local rigid posture needs about 2.43° orientation error with consistent bounds. The generic default is 3°, with an explicit override. This is an operating allowance, not an estimated confidence bound. Rigid, jaw-axis and point modes retain different contact assumptions.

## Current interface and grasp search

`plan_arc` accepts `position_tolerance_m`, `orientation_tolerance_deg`, and `max_wrist_reorientation_deg`. Defaults are **6 mm**, **3°** for rigid/jaw-axis deviation, and **no independent total wrist-reorientation cap**. `orientation_tolerance_deg=0` selects the legacy effectively exact angular constraint. The legacy `refined` argument is accepted but no longer changes constraints or estimates a hinge.

Do not confuse orientation error with total wrist motion. A hand can rotate substantially along a valid articulation while remaining close to its transported grasp orientation. Position tolerance also does not change collision clearance. Physical joint limits, collision geometry, contact permissions, progress bounds, full-path checking and the 600-control episode cap remain in force.

Grasp proposals now interleave frontal, oblique and GraspGen orientations across observed handle heights. Frontality and original GraspGen translation are preferences rather than hard rejection tests; candidate translations are grounded on the measured handle. Search can examine up to 32 IK branches per pose, initially trying two per pose before deeper branches. Returned grasps still require IK, collision checks, an approach and a complete requested contact path.

The old late-grasp rejection reserved 200 manipulation controls and 40 closure controls, then required a 60-control approach hold. At control 380 this yielded −20 available approach controls despite 220 physical controls remaining. The new policy prices the actual complete approach and contact paths using the documented default speed 0.35 rad/s and six endpoint holds, plus closure. Callers choosing other execution timing still face the actual trajectory budget check before motion.

## Validation evidence

All saved scenes are hash-checked. Planners receive public observed geometry and calibrated robot information. Native object state and contact labels are used only after rollouts for grading.

1. **Identical recorded GPU seeds:** the original 3° point request admitted zero endpoints. With refined admission at 3 mm, the local 1.5° query accepts 60/64 seeds, producing 23 distinct endpoints. A complete 3° connection also succeeds. This isolates admission changes from new GPU proposals.
2. **Fresh cuRobo planning:** all three complete requests solve at both 3 mm and 6 mm. The final 6 mm 15° query takes 4.246 s: scene setup 0.550 s, GPU IK 2.585 s, CPU refinement 0.129 s, native connections 0.446 s, remaining admission/bookkeeping about 0.535 s. First-query and cache costs differ; do not infer a tolerance speedup from this ordered run.
3. **Fresh physical replays:** restore the exact original 380 issued controls, verify zero maximum joint difference at the planning start, and execute the whole new request. At both position tolerances: 3° → **3.026° closing**, 7° → **7.131°**, 15° → **15.248°**. Both finger pads contact the microwave at every new control step: respectively 17/17, 17/17 and 29/29. These bounded moves do not close the whole door; it starts about 68.52° open.
4. **Native boundary checks:** inside/outside positional and angular allowances behave correctly for rigid and jaw-axis modes; injected palm collisions remain rejected. Legacy exact-angular behavior remains available and passes the previous tolerance test.
5. **Regression tests:** 74 focused tests cover the interface/schema, progressive continuation, failed-seed refinement, optional wrist policy, grasp diversity/budget, explicit timing, free-pose branches/diagnostics and live motion tools.

The position tolerance expansion alone did not cause the replay recovery: 3 mm and 6 mm produce identical physical results here. The demonstrated causes were premature seed rejection, hidden wrist limits, exact angular constraints, mismatched bounds and search scheduling.

Hinge uncertainty and IK residual are different quantities. For an ideal revolute motion with pivot error δc and known axis, target displacement error is `(I − R) δc`. A 6 mm transverse pivot error contributes about 0.31 mm at 3° and 1.57 mm at 15°. Axis-direction error, contact offsets and perception bias add other effects. A larger IK tolerance is not an uncertainty-aware collision or grasp certificate.

## Fresh Luna outcome and remaining failure

One fresh Luna episode used the same motion-route prompt and budgets, without the successful replay commands or coordinates. **Task failure: 598 controls, 1,800.04 actor seconds. All nine requested arc plans succeeded.** Every issued trajectory completed. Runtime sources were frozen throughout the episode, and the episode's source-integrity audit passed.

The final native diagnostics show a different failure from the original numerical stall:

- Initial visual hinge error: **254.49 mm**. Grounding selected **−Z** for positive requested motion, whereas closing this scene requires **+Z**. This direction comparison is an offline grader fact, never provided to Luna.
- The first approach opened the door by **13.64°**. After the following gripper closure neither finger pad touched the microwave. The subsequent requested 6° arc moved the robot without moving the door.
- Applied motion fits improved pivot error to **15.14 mm** and **13.90 mm**, but preserved −Z. A motion-supported *unsigned* axis fit did not establish the correct task direction or a retained grasp.
- Later sequences opened the door farther. Re-grounding changed the direction to +Z only at control 573. Subsequent arcs had no useful target motion, with zero or one finger pad touching at their endpoints.
- Final opening: **119.98°**, versus **88.92°** initially. These facts support a grounding/direction/contact diagnosis; they do not prove which policy change would make the next fresh episode succeed.

The 1,800.04 s actor session splits into **863.15 s outside robot calls**, **398.80 s grasp/sequence planning**, **385.81 s execution/gripper calls including observations**, **65.33 s requested arc planning**, and **86.94 s other robot work**. Outside calls include model inference, image inspection, analysis, scheduling and transport, rather than pure reasoning time.

Two interface defects surfaced during that run and were fixed **after** it completed:

1. Planning now accepts `speed`, `hold`, and `close_steps` in `plan_grasp` / `plan_sequence`, and returns its budget assumptions. The saved sequence cost **129 controls under defaults**, but **67 under Luna's actual requested timing**, exactly matching execution. At control 529, default closure reserved 40 of 71 remaining controls; requested closure of 12 leaves 59 for approach instead of 31. Complete-path cost is still checked. This deterministic check does not establish that a particular late regrasp succeeds physically.
2. Free-pose planning searches all generated robot-valid IK solutions, rather than only eight, and exposes endpoint versus path-search failures. Recorded call 45 had eight IK solutions: **three goals rejected at arm link 6 and five at link 5**, all against stationary observed geometry. It was an endpoint collision rejection, not a stalled IK solver. Goal joints and planning inputs are now saved for reproducible diagnosis.

Applied hinge estimates also explicitly report that task direction is **not verified by the motion fit**, the change in axis location, and that the previous sign was retained. This is honest feedback, not an automatic direction correction. The generic introduction now calls for rechecking signed task direction after a pivot correction.

The post-episode changes passed regression and deterministic checks. No second fresh Luna episode was run with them. The next end-to-end blocker is reliable **task-direction and physical-contact verification from public observations**, rather than further indiscriminate tolerance expansion. Do not promote this episode as a successful learned method.

## Artifacts and continuation

Evidence root: [planner-range-recovery-20261027](../results/planner-range-recovery-20261027/).

- `source-before/`, `source-revision1/`, `source-final/`: preserved implementation stages; `source-final/` is the frozen fresh-Luna runtime. `source-current/` records the post-episode interface changes. The original native library/build receipt is also retained.
- `candidate-replay.json`: same-seed analysis; no GPU or simulator rerun.
- `fresh-replan.json`, `progressive-replan.json`, `angular-replan.json`: intermediate failures retained.
- `final-replan.json`, `physical-final-replay.json`: final complete planning and physical measurements.
- `tolerance-validation.json`, `native-legacy-validation/summary.json`, `tests-final.log`: checks.
- `luna-protocol.json`, `luna-status.json`, `luna-summary.json`: one fresh Luna follow-up, same original motion-route prompt, one episode, 600 controls, 1,800 seconds, no worked example, teacher, reflection or prior result exposure. Consult the status/summary for completion; do not infer full-task success from replay success.
- `luna-completed-analysis.json`, `luna-physical-effects.json`, `luna-failure-diagnostics.json`: completed-episode geometry, motion/contact and tool-failure evidence. `post-trial-interface-checks.json` verifies timing and error interpretation without another live episode.

For future failures, report **module → measured error → allowed range → consequence**. Distinguish an unexamined seed, rejected endpoint, failed path search, controller outcome and loss of physical contact. A finite search failure is not a proof of infeasibility. Do not attribute a task failure to Luna or grounding when the numerical tool discarded a valid alternative.
