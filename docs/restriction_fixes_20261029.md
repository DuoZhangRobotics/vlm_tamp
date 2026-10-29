# Recovery, control tracking and restriction fixes — 2026-10-29

The implementation follows `restriction_audit_20261029.md`. There is **zero
added collision padding**. Environment predictions are advisory by default in
simulation; physical contacts remain enabled. Real joint bounds, known robot
self checks, finite inputs and observation freshness remain checked. Luna
continues to choose strategy, targets and recovery from public observations.

## What changed

- A single per-episode control budget reaches the worker, service, planners,
  schemas, prompts and reports. The normal learning runner defaults to 1,200;
  comparisons explicitly request 600 or 1,200. The wall limit remains 30 minutes.
- Removed extra 0.025/0.005 rad joint reserves. Full-pose cuRobo generation and
  CPU admission use the same 3 mm tolerance. Failed optimizer flags are
  diagnostic: actual FK residuals, bounds and robot checks decide admission.
- Grasp search chooses low-control-cost approaches, prefers observed clear
  routes and retains a robot-valid advisory fallback. Direct, short and longer
  standoffs and calibrated pad depths are candidates. An exhausted search is
  reported as incomplete, not physical infeasibility.
- Sequence execution opens when needed, approaches with matching aperture,
  physically closes, then replans from measured geometry. It uses short initial
  joint-calibration probes and complete feedback arcs. An independent observed
  motion interval checks a revised hinge before installation. Gap collapse or
  two ineffective motions return recovery to Luna.
- Signed motions and requests above 90 degrees use internal segments. Current
  observed part geometry can bootstrap a joint model without a vertical handle.
- Contact stroke length, approach speed, search budgets, ineffective-stroke
  allowance and withdrawal distance are explicit settings. Push, hook-pull and
  pinch-grasp evidence remain distinct.

## Controller defect and matched physical test

The old adapter limited target minus *measured position* to 0.05 rad. This is a
position-lead cap, not a limit on consecutive commanded references. At Kp=300 it
caps proportional lead at 15 Nm and can cause large tracking lag against damping.
The corrected `slew_limited` adapter limits consecutive desired references to
1 rad/s using robosuite's existing `set_goal(..., set_qpos=reference)` API.
It changes a controller goal, **not simulator qpos/qvel**. Gains, actuator limits,
physics and contacts are unchanged. Actual joint speed is measured separately.

An identical saved approach from state 49, with its original prefix replayed
under the original adapter, produced:

| Adapter / requested speed | Approach controls | Endpoint position error | Orientation error |
| --- | ---: | ---: | ---: |
| Old / 0.35 rad/s | 315 | 0.249 mm | 0.0173 degrees |
| Old / 0.60 rad/s | 186 | 6.730 mm | 24.467 degrees |
| Corrected / 0.60 rad/s | 186 | 0.344 mm | 0.0175 degrees |
| Old / 1.00 rad/s | 114 | 27.451 mm | 95.263 degrees |
| Corrected / 1.00 rad/s | 114 | 36.133 mm | 9.031 degrees |

0.60 rad/s is the approach default. The 1.00 results do not support using the
maximum as a universal default. Faster requested motion is not guaranteed to
reach its endpoint or retain a grasp. Preserve `measured_delta` for historical
prefixes; `source_reference_mode` reads the mode recorded in newer runs.

## Evidence so far

- A budget-only fresh Luna run with old numerical/controller policy and frozen
  memory closed state 49 at control **692** (623.02 s actor wall). The earlier
  600-control run failed. This actor knew its larger budget; decisions were not
  held identical, so this is evidence of recovery opportunity, not a deterministic
  counterfactual replay.
- A scripted diagnostic using the revised sequence opened state 13 at control
  **249**, including one motion-supported hinge revision. It reused historical
  public selections and is not a fresh Luna success.
- A closing diagnostic still failed to acquire the handle and returned
  `grasp_not_established` at control 284. Other retained diagnostics lost grip
  after partial progress. These are unresolved contact/model problems; a solved
  numerical plan is not a grasp-retention certificate.

Fresh fixed-memory comparisons use closing states 17/49 and opening state 13 at
both budgets. These are previously examined development cases. The manifest
freezes source hashes, Luna/high, and the prior learner memory. Report first
native success, success by 600 controls, total controls, wall time and failure
cause. Separate varied practice and later evaluations must use a new manifest.

## Running and finding evidence

From the repository root, on this server:

```bash
source scripts/server_runtime_env.sh
CUDA_VISIBLE_DEVICES=3 python scripts/run_restriction_study.py --tag cleaned-v1
```

Use VS Code Live Preview on
`results/restriction-fixes-20261029/watch.html`. The existing
`results/luna-live-hinge-comparison-20261027/watch.html` links/embeds it too.
The report includes actor I/O, tool selections and returns, videos, contact
overlays, measured motion/speed, stop reasons and nested profiling. Outside-service
time includes model, transport and local analysis; it is not pure reasoning time.

Each run keeps `tool-events.jsonl`, `agent-events.jsonl`, `agent-prompt.txt`,
`protocol.json`, `evaluation-private.json` and `timing-breakdown.json`. Raw data
stays in ignored `results/`; code and this handoff are versioned. Private native
angles/contact diagnostics are for offline evaluation, never actor/planner input.

For future agents: preserve all failed runs, do not add collision buffers, do not
change runtime sources during a frozen comparison, and never present scripted
checkpoint success as fresh policy transfer. Read `motion_settings`, the current
tool schema and source-reference mode instead of assuming historical defaults.

## Automated checks

The full server suite passed 1,494 tests, skipped 36, failed 44 and had eight
fixture-setup errors. The remaining failures require unavailable historical
recordings/calibration fixtures, the separate ReKep checkout, LIBERO parser
imports in the planner environment, or the optional Google provider package.
They are retained in `results/restriction-fixes-20261029/test-validation.json`;
the complete suite is **not** claimed to pass. Updated controller, recovery,
sequence, contact, budget, schema and isolation checks pass.

## Frozen comparison before the follow-up repairs

These six fresh actors used commit `8eb4503`, Luna/high and the unchanged prior learner memory. Each case/budget has one rollout, not a statistical success-rate estimate. The source-integrity checks passed.

| Case | Budget | Native success | Controls | Actor wall (s) | Services (s) | Outside services (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| close_microwave / 17 | 600 | False | 600 | 1007.0 | 435.5 | 571.5 |
| close_microwave / 49 | 600 | False | 600 | 1309.4 | 419.3 | 890.2 |
| open_microwave / 13 | 600 | True | 425 | 537.9 | 252.5 | 285.4 |
| close_microwave / 17 | 1200 | True | 135 | 489.4 | 130.2 | 359.2 |
| close_microwave / 49 | 1200 | False | 1200 | 1395.8 | 581.9 | 813.9 |
| open_microwave / 13 | 1200 | True | 1097 | 1023.9 | 511.1 | 512.8 |

Outside-service time includes model calls, transport and local analysis. GPU 3 carried another workload during the later tests; wall times include those hardware conditions. See `hardware-conditions.json`.

The 600-control condition succeeded in 1/3 runs; the 1,200-control condition in 2/3. The latter opening rollout needed 1,097 controls, while closing state 17 succeeded at 135 and did not use the extra allowance. Closing state 49 failed at both limits. Do not infer that budget alone caused the differences between independent actor decisions.

### Concrete failure and success mechanisms

- Closing state 17 / 600: early pushes opened the door slightly; grasped motion and later approaches reduced the opening to 32.25 degrees, but engagement and tracking failed before closure. The first automatic hinge update fit tiny 1.77/2.24-degree motions better while worsening native axis-location error from 23.9 to 36.2 mm.

- Closing state 49 / 600: it ended 120.37 degrees open from 107.26. Requested local progress often served the opposite task direction.

- Closing state 49 / 1,200: the observed-motion hinge estimate was accurate to 0.708 mm, yet all seven physically attempted signed joint pushes requested the opening direction. It ended 86.86 degrees open. Accurate grounding alone did not solve task-direction selection. These native comparisons are offline grading only.

- Closing state 17 / 1,200: native success occurred during approach, with measured peak hand speed 0.173 m/s, before the requested controlled contact stroke. It is task success, not a validated 3 cm/s controlled-stroke result. Separate withdrawal/coasting validation is required.

### Follow-up implementation under validation

Preserve a signed axis when reinstalling a generic part model. Retain prior observed geometry as inactive, timestamped history for explicit measured part-motion updates. Use a combined initial motion baseline and a third checking interval before automatic hinge revision. Make the expected plan executor and arc search budget explicit. Preserve detailed grasp and specialized arc IK timings in the report.

The earlier suspicion that compact replies omitted `grasp_stage_contract` was incorrect: the field was present in actual actor logs. The wrong-executor call was an action-selection error. The response now names the expected executor directly for clarity.

The first two physical follow-up diagnostics completed their motion but hit a reporting-harness error afterwards: direct `RobotService.dispatch` returns the result itself, not the RPC envelope. Original exit codes and logs are retained. Recovered data showed an 8.6 mm hinge error with the longer baseline, but grasp loss persisted; without revision, subsequent planning was incomplete. Intermediate replanned probes were not identical, so this is not a strict matched causal comparison. Corrected harness checks and local contact replays follow under fresh tags.

## Repaired implementation and local validation

The promoted repair preserves signed joint directions during model reuse and
retains inactive observed geometry for explicit measured-motion recovery. A
track can originate from another mask of the same object; mask-ID equality is
not a motion constraint. Timestamp, rigid-fit, fixed-hinge and observed-depth
checks remain. Arc search budgets and expected plan executors are explicit.

The complete repaired server suite passed **1,513 tests**, skipped 36, and failed
42 because historical recordings or optional dependencies were unavailable.
There were no fixture-setup errors after the numerical geometry tests switched
to the versioned calibrated-Panda fixture. The LIBERO parser tests separately
passed all nine checks in the simulator environment. The full suite is not
claimed to be green; see `test-validation-repaired.json`.

All 15 local physical checks completed under fresh output names:

| Local condition | Replays | Native closure | Retained closure | No robot/environment contact during release coast |
| --- | ---: | ---: | ---: | ---: |
| Controlled push, 3 cm/s | 3 | 3 | 3 | 0 (final pose held) |
| Push-and-release, 3 cm/s | 3 | 3 | 3 | 3 |
| Push-and-release, 6 cm/s | 3 | 3 | 3 | 3 |
| Push-and-release, 12 cm/s | 3 | 3 | 3 | 3 |

Hook-pull reached its separate non-pinching motion goal in 3/3 replays: about
2.72 degrees for a 3-degree request, with a minimum gripper joint span of
79.4 mm. It deliberately pulled in a closing-task scene, so native closing
success is not that skill's criterion.

The release strokes commanded 30 mm and measured approximately 21 mm of forward
hand travel. Mean measured peak speeds were 2.87, 5.43 and 9.39 cm/s, with first
native success 0.75, 0.45 and 0.30 simulated seconds after the pulse began.
Push replays also used palm contact. They establish these local outcomes, not
strict fingertip-only execution qualification or general task success. The
physical source files were preserved and verified against all 12 recorded hashes.
A later change clarifies validation-status wording; contact motion code is unchanged.

The five-degree feedback experiment retained a roughly 22 mm pad gap but
stopped after 19 commanded degrees: eight endpoint goals were admitted and
eight connection searches returned no path. This does not prove physical
infeasibility and did not justify changing the default feedback interval.

Separate replays of the successful live state-17 rollout matched its robot
prefix to 1e-14 rad. Closure persisted after 15, 50 and 100 mm withdrawals and
one second of coasting. At 15 mm, palm/handle contact remained. At 50 and 100 mm,
microwave contact was gone but finger/mug contact remained. Those three tests
therefore establish retained closure with the reported contacts, not completely
contact-free withdrawal. See `retention-summary.json` for forces and geometry names.

Varied practice uses six attempts in one persistent Luna context across the
three development cases. Freeze and inspect its learner-authored notes before
fresh evaluation. Final-version evaluation covers the three development
regressions and two separately declared cases; report those groups separately.
No native coordinates or successful action sequence is supplied to the learner.

## Varied practice and camera-selection repair

The persistent six-attempt practice block on commit `2313406` completed with
2/6 native successes: closing state17 at855 controls on its first attempt and
121 on the repeat. Closing state49 failed twice (1182 and1131 controls, wall
budget); opening state13 failed twice (1200 controls and1179 controls). Luna
requested further practice after reaching the declared cap; this block is kept
intact while its frozen notes are assessed in fresh agents.

Luna wrote four scoped principles. The reviewed file is preserved verbatim in
`prompts/learned/contact-practice-20261029/guidelines.json`, SHA256
`d3d4d3a27a7c3da748701ffcc26ef721258b2eb62030f4333c34d5b3f07bd931`.
It records incomplete opening results and explicitly declines a repeatability
claim. The review found no saved coordinates, robot joint vectors, object-state
indices or replay recipe. Supporting attempt numbers identify evidence only.

A recorded push exposed a separate observation inefficiency: its original
four-camera subset kept being used after arm/target visibility changed. Four
failed subset fits triggered four full-camera recoveries, costing32.9 seconds;
contact and stroke planning together took about4.4 seconds.

The repair chooses up to four views from each new tracker's current reference
features and retains the requested view. The quality thresholds are unchanged.
On the same four recorded physical intervals, a closed-loop observation replay
used only actually available reference views: both policies accepted all four
intervals, while full captures fell from4 to1. Final measured progress differed
by0.42mm. This is an observer replay, not a claim of faster live task completion.
The baseline replay reproduces the recorded progress to floating-point precision.
See `camera-selection-replay/closed-loop-comparison.json` and `analyze.py`.

The camera repair passed45 focused contact/feedback tests. It was prepared in
an isolated copy and applied only after the six-attempt practice finished.
Final fresh-agent evaluation therefore tests the repaired system plus frozen
learner memory; it does not isolate the causal contribution of memory alone.
Controller, planner, contact permissions, measurement-quality thresholds and
native success criteria are unchanged by this camera-selection update.

## Generic-joint to handle conversion repair

A no-motion regression exposed a schema/type-transition bug. A part-only joint
model intentionally has no handle. After a valid handle fit, refresh_contact
passed that sparse model directly to the legacy manifold constructor, which
requires joint_family and target_progress. It could raise KeyError before
constructing the handle model. The old generic/no-handle flag also survived the
fit and would reject grasp planning afterwards.

The repair merges the standard handle schema with the preserved fixed-joint
parameters and clears the part-only flag only after the handle fit succeeds.
Error messages now identify the required model and the available update tools.
The regression failed with KeyError: joint_family before repair;71 related tests
passed afterwards (5 optional-fixture skips). A real recorded RGB-D check also
fitted the9.72mm-radius handle without motion or native object geometry, while
preserving the previous hinge origin and signed axis exactly.

The v2 evaluation driver was suspended while its active episode finished on
unchanged code. Its two completed outcomes remain: closing17 succeeded at224
controls and closing49 failed at1200. No logged KeyError occurred in those two
runs, so their outcomes are not attributed to this latent conversion bug. The
cohort was stopped before its remaining cases; a new complete v3 five-case
evaluation follows after repair. Memory remains frozen and unchanged.
