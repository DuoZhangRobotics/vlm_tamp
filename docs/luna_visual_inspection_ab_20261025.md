# Luna visual-inspection pilot — 25 October 2026

**Completed: the helper works, but this pilot does not demonstrate a success
improvement. Keep it opt-in.** Baseline succeeded in 4/5 episodes; the condition
with visual inspection available succeeded in 3/5. All ten simulator episodes
completed without an infrastructure failure.

[Watch the comparison](../results/luna-visual-inspection-ab-20261025/watch.html) ·
[Protocol](../results/luna-visual-inspection-ab-20261025/protocol.json) ·
[Implementation patch against the preceding working files](../results/luna-visual-inspection-ab-20261025/implementation.patch)

## Results

| State | Baseline: outcome / controls / actor seconds | Helper available: outcome / controls / actor seconds |
| --- | --- | --- |
| 20 | Success / 346 / 583.2 | Failure / 600 / 809.0 |
| 24 | Success / 289 / 394.4 | Success / 287 / 378.5 |
| 35 | Failure / 600 / 836.1 | Failure / 600 / 739.5 |
| 39 | Success / 277 / 377.6 | Success / 292 / 336.8 |
| 49 | Success / 277 / 322.3 | Success / 291 / 436.1 |

There are three pairs where both succeeded, one where both failed, and one
baseline-only success. This is too small and narrow to establish that the
helper systematically helps or harms. No memory or default interface was
promoted from these results.

Mean actor wall time, including failed episodes, was 502.7 s baseline and 540.0 s
with the helper available. On the three paired successful episodes, the helper
condition averaged 19.0 s longer and 9 more controls. Raw CLI totals were 15.52M
input / 89,236 output tokens for baseline and 10.53M input / 74,026 output tokens for
the helper condition. Most input tokens were cached; different action histories
also affect usage. These counts do not establish a monetary-cost improvement.

### Adoption and demonstrated behavior

Luna used the helper in three of five treatment episodes: 11 requests total,
10 successful and one rejected format error that Luna corrected. The tool's
local work totaled **3.11 seconds** across the whole study, excluding model
processing and image delivery. All 10 generated comparisons were delivered to
the model in recorded image-tool batches.

- State 20: crops from two cameras at the same observation; failed.
- State 35: one four-camera view and one crop of a later observation; failed.
- State 39: seven successful inspections, including **five comparisons across
  observation IDs**; succeeded. The first malformed request supplied strings
  instead of view objects and was rejected before any artifact or robot action.
- States 24 and 49: no helper use; both succeeded.

State 39 is a useful process example: Luna explicitly recognized that tracking
had fallen back to motion predicted from the hand, compared current and earlier views,
and used visible door displacement before planning further motion. This
demonstrates that Luna can use the interface as intended. Its matched baseline
also succeeded, so it does not establish a causal success benefit.

Use was optional and often occurred during difficult recovery. The higher
failure frequency among some users is not evidence that inspection caused
failure. Raw image-tool receipts also cover baseline image access. Their
count of batches containing multiple observations is a lower bound; it does not measure all
comparisons a model can make with images already in context.

### Failure evidence

On state 20, the treatment initially opened the door to about 71°, then lost
finger contact. Several later arcs moved the robot without moving the door;
recovery ended around 60° and exhausted 600 controls. The baseline revised its
explicit pivot estimate from 24.5 to 3.7 mm error and succeeded. That difference
is descriptive, not proof that hinge error alone caused the outcome.

Both state 35 episodes failed. Their maximum opening was approximately 39°
(baseline) and 52° (treatment). Wrist limits and rejected goals, rejected handle
refits, and unsuccessful contact recovery consumed budget. The existing grasp
reserve rule rejected late searches; the helper does not change that rule or
make an unreachable pose reachable. There were no cuRobo CUDA-graph crashes or
model-access failures in this study.

[Private failure evidence](../results/luna-visual-inspection-ab-20261025/failure-diagnoses-private.json)
contains exact native pose/contact measurements at the recorded control steps.
They were never provided to actors. Small motion on a trajectory terminated by
native success is excluded from the ineffective-arc proxy; stopping early at
the goal is correct behavior.

## Question and controls

Does convenient retrieval and comparison of already captured observations help
Luna use available evidence? This tests one idea motivated by
[VISTA](https://arxiv.org/abs/2610.02200), not the complete VISTA system.

The five initial states are 20, 24, 35, 39 and 49 of the existing microwave task/asset.
They are reused diagnostic states, not held-out objects. Each condition gets
one fresh actor context per state, one episode, 600 controls and 1,800 actor seconds.
Condition order alternates across states. Runs are serialized on GPU 0; it is a
shared server, so wall times include machine and model-service variability.

Both conditions use `gpt-6-luna`, high reasoning, the same initial prompt,
abstract example, Python/Pillow access, calibrated RGB-D and existing numerical
tools. The frozen `luna-scene-motion-v1` guide has SHA256
`65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2`.
Per-trial findings do not update another actor's guide. No prediction-note
prompt, Reflection reviewer, stronger-model teacher, RoArtNet integration or
geometry/planning/controller change is included.

The treatment adds exactly three public-input differences: `visual.py`,
`tools/visual_inspection.txt`, and an appended discovery entry in `tools.txt`.
The robot client, full robot manual, schemas, guide and example remain identical.
The baseline may use Python/Pillow to inspect the same underlying files.
This is a test of convenience and optional adoption, not new sensory information.

## Helper behavior

[Implementation](../src/vlm_tamp/visual_inspection.py) is copied into the public
workspace as a read-only `visual.py`. Its two operations are:

```bash
python3 visual.py history '{"limit":8}'
python3 visual.py inspect '{"views":[{"observation_id":1,"camera":"frontview"},{"observation_id":4,"camera":"frontview"}]}'
```

`history` lists captured observation IDs, control steps and cameras. `inspect`
accepts one to four selected views, optionally cropped using original-pixel
`region_xyxy` rectangles. It preserves display order, labels every panel, saves
original-resolution crop files and gives a comparison-image path for the image
viewer. Metadata includes source hashes, original dimensions and an affine
mapping from displayed pixel centers back to original pixels.

The helper reads existing PNGs and public calibration/state metadata. It does
not capture new sensors, advance the simulator, call the robot service, infer
objects/hinges, fit motion or alter planner geometry. Only derived images and
`visual-events.jsonl` are written. In particular, local inspection does not
renumber robot operations or consume control steps; it still consumes wall time
and model/tool processing within the same actor budget.

Enable it in a future independently registered run with
`scripts/run_subscription_agent_pilot.py --visual-inspection` alongside that
run's other declared options. It is disabled by default. The commands above run
inside an agent workspace containing `visual.py`. For replay, first copy that
public workspace so new helper receipts do not alter the original study logs.

## Verification and observability

Before launch, 46 tests passed across the new helper, compact presentation,
robot contracts and runtime portability. Tests check exact crop pixels,
coordinate mapping, bounded history, invalid requests, unchanged source data
and absence of robot RPC. Paired workspace preparation verifies the three
allowed differences. Sandbox checks verify that the helper is readable but
cannot be edited, and private geometry/implementation reads remain denied.

The CLI event feed omits image-view tool calls. An observer-only extractor uses
the exact research thread UUIDs to record image-tool requests and returned-image
counts from those sessions. It does not copy raw session instructions,
reasoning, image bytes or other threads. These receipts are never actor inputs.
An image delivered to the actor does not prove correct interpretation; a batch
can include multiple images.

The report retains native success, control count, full actor wall time, robot
service time, nested observation/grasp/IK/arc profiles, helper activity and raw
CLI token usage. Time outside recorded tools includes inference, scheduling,
delivery and overhead; it is not pure reasoning time. Token totals include
repeated/cached inputs and are not monetary cost or peak context usage.

Private completed-trial diagnostics compare explicit hinge revisions with the
native axis, commanded arcs with actual door movement, contact endpoints and
controller tracking. They do not advise the actors. All task failures remain
in the denominator; infrastructure failures stop the suite without silent
replacement. Compare efficiency on paired successful episodes, since a quick
failure is not a performance improvement.

## Final integrity checks and decision

All 210 per-trial native/control/source/isolation checks pass. All five pairs
match on initial metric sensor arrays, calibration, robot state, analysis
libraries and the declared common public inputs. All 30 view crops in the 10
comparison artifacts exactly match their selected source pixels.

**Exact RGB identity was not achieved in three pairs:** state 20 has 6 differing
channel values, state 24 has 54, and state 35 has 6; each difference is 1 out of 255.
States 39 and 49 match exactly. This remains a failed exact-identity check,
documented separately from matching physical/metric state. No post-hoc
tolerance or hidden renderer change was used to erase the deviation.

All study processes have exited. The 46 focused tests and sandbox preflight
passed; the frozen guide and runtime source hashes remain unchanged throughout
the ten episodes. The helper remains disabled unless explicitly requested with
`--visual-inspection`.

The supported conclusion is limited: optional convenient visual access did not
improve success in this pilot, although one successful actor used temporal
inspection as intended. A separate controlled test of brief expected-versus-
observed checks could assess the other proposed VISTA idea. It was intentionally
not bundled into this experiment, and has not been run.
