# Recorded goal-reference gap test

The user requested a recorded-data test on closing states 17 and 49, with Luna/Molmo2 choosing
the references, no hardcoded closing direction, and marked points plus measured gaps in
`watch.html`. No new robot motion was run, and no live goal-feedback tool was installed.

Report: `results/recorded-goal-gap-20261103/watch.html`.

## Selection protocol

One fresh isolated `gpt-6-luna`/high session per state received the saved `agentview` and wrist
captures, an observation catalog, the existing local Molmo2 pointing operation, and a literal
saved-depth sampling operation. Later saved views were available; both actors nevertheless
selected observation 1. Neither actor received old notebooks, native object state, expected
coordinates, expected motion labels or measured gap trends. Each session received ten actual
image-input blocks. Input provenance, isolation checks, exact prompts and public tool logs are
saved per case. Private reasoning was not exported.

Molmo2 remained loaded across selection sessions. Its proposals were inspected and rejected or
adjusted by Luna. The final references below were Luna's own selections; no operator coordinates
were substituted. Their file hashes were frozen before numerical measurement and remained
unchanged across both tracking-density conditions.

| State | Moving reference, agentview | Fixed-reference hypothesis, agentview | Luna's assessment |
| --- | --- | --- | --- |
| 17 | obs1, (262,235) | obs1, (486,235) | `can_select=true` |
| 49 | obs1, (270,190) | obs1, (461,247) | `can_select=false`; tentative matching edge |

Both fixed marks sample the appliance's upper surface at world Z about 1.1078 m. Their heights
differ from the moving marks by 129.4 mm and 90.8 mm respectively. They are not verified literal
mating points. A distance to a nearby fixed surface can still correlate with closure over a
limited range; that does not make the numerical distance an actual door-clearance measurement.

## Measurement method

The same CoTracker3 `scaled_offline` checkpoint tracked both selected point hypotheses. The
initial condition used 64 RGB frames per case. A separately declared density comparison used
every recorded external RGB frame from controls 60–600: 541 inputs for state 17 and 542 for state
49, including two distinct captures at control 440. Exact capture PNGs replaced compressed-video
RGB at their timestamps. Dense video decoding ran in memory, without dumping hundreds of frames
to shared storage. Checkpoint, algorithm, references and depth policy were unchanged.

At a saved depth capture, each predicted pixel is rounded to its nearest original pixel and read
from the recorded world-point array. A supported pair requires tracker visibility and finite
non-robot measured depth for both points. The reported distance is simply
`norm(moving_world - fixed_world)` at that capture. Both points are tracked; neither is held at
an earlier world position. No hinge, CW/CCW convention or desired world direction enters this
calculation. There is no point snapping, inferred plane, alternate tracker or interpolated depth.

Raw depth distances at unconfirmed tracks remain visible as diagnostic samples, not supported
goal measurements. Tracker visibility and finite depth do not certify material correspondence.
Native angles are read only by the subsequent evaluator/report, after references and measured
gaps have been saved. They are not used to choose points, fit tracks or alter measurements.

## Results

| State | 64-frame supported captures | All-recorded-frame supported captures | Retained depth captures |
| --- | --- | --- | --- |
| 17 | 4/15 | 6/15 | 15/15 |
| 49 | 1/15 | 2/15 | 4/15 |

The 64-frame state-17 distances were 369.57 mm at control 60, 327.76 at 590, 332.64 at 597 and
332.83 at 600. Those sparse samples broadly agreed with net closing and late reopening, but did
not cover most manipulation intervals.

Using all RGB frames exposed a false progress signal in state 17:

| Interval | Native opening | Measured selected-point distance |
| --- | --- | --- |
| Controls 573–590 | 67.44° → 72.56°: **opened 5.13°** | 337.31 → 327.10 mm: **decreased 10.21 mm** |

Thus a controller interpreting every decreasing distance as closing would receive the wrong
signal here. Tracking/depth association is not sufficiently reliable, even when the tracker
reports a point visible. The moving sample's world height also drifted by about 29.6 mm in this
condition; the evaluator's hinge axis/anchor remained vertical and fixed.

State 49's all-frame distances were 382.32 mm initially and 545.20 mm at control 600. The final
tracked moving sample was 115.5 mm below its initial height, inconsistent with the same material
point on this vertically hinged door. This is not a validated gap increase, despite agreeing
in sign with the opening trajectory. Its fixed reference was already marked tentative by Luna.

More frames improved coverage slightly, but did not repair correspondence. The report preserves
both conditions rather than selecting whichever happens to look better.

## Missing depth and the reader correction

The original run's post-episode retention kept all RGB, calibration and robot metadata but removed
unreferenced intermediate sensor arrays. State 49 retains depth only at observations 1, 15, 26 and
38. This limits the retrospective measurement; it does not explain the live tracker failures
during that earlier episode, when the arrays still existed.

The first postprocessor assumed every saved RGB capture still had depth and failed on obs006.
The reader was corrected to represent missing depth explicitly. The completed 64-frame CoTracker
outputs were reused for CPU remeasurement with unchanged references. No depth was reconstructed
from simulator state or nearby frames. The subsequent all-frame comparison used the corrected
reader from the start. The original error log remains under `tracking-resources/`.

## Timing, resources and validation

- Reference selection: 187.38 seconds for state 17; 166.11 seconds for state 49.
- State-17 initial tracking inference: 3.76 seconds; the state-49 initial group timing was not
  persisted before the reader interruption and is reported unavailable.
- All-frame tracking inference: 3.31 seconds for state 17 and 2.98 seconds for state 49. These
  single measurements are not a controlled speed benchmark; caching and first-use costs differ.
- Peak monitored job RAM including file cache: 24.45 GiB. Peak GPU memory: 36.14 GiB during dense
  tracking. Host available memory stayed above 95 GiB. All 31 SSH probes succeeded; zero OOM kills.
- All ML phases used the existing 32 GiB host limit, zero swap and resource watchdog. Workers
  were shut down afterward. CPU remeasurement used the already saved small track arrays.
- Measurement unit tests cover world-coordinate distance, occlusion, robot depth, invalid pixels
  and missing archived depth. Related tracking/video tests and Ruff were run.
- Input alignment audits verified all 15 exact capture images per condition and unchanged
  reference hashes. Browser checks cover chosen marks, both density settings, capture values,
  input/output links and narrow layout.

## Reproduction and next boundary

Code: `goal_gap.py`, `select_recorded_goal_references.py`, `measure_recorded_goal_gap.py`,
`report_recorded_goal_gap.py`, and `templates/recorded_goal_gap.html`.
The monitored launchers are saved under `.runtime/recorded-goal-gap-20261103/`.
Protocols, frozen choices, both track conditions, data-availability reasons and evaluator-only
comparisons are retained in the result directory. No old actor workspace was modified.

This test does **not** justify deploying a reliable closing-progress signal. A useful next
recorded-data experiment would carry the selected reference through the existing measured
rigid-part pose, using multiple visible landmarks, and label the result as an estimated gap
when the reference itself is occluded. That representation has not been introduced here.
Reference correspondence also needs improvement, and future recordings need to retain the depth
at the action boundaries required for evaluation. None of these should be replaced by a guessed
closing sign or an automatic switch to another tracker.
