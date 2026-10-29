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
