# Two-camera region perception — 2026-11-03

The new interface correctly identified the two substantial opening intervals in recorded
state 49. Luna classified five of six tested intervals correctly. The remaining mismatch
was a 0.37-degree closing movement labelled unchanged. This is an offline perception result,
not a new task-success result or an online-policy evaluation.

Open `results/two-view-regions-20261103/watch.html` in VS Code Live Preview. The page shows both
camera masks, paired interval images, every tracking capture, three camera conditions, actual
model inputs/outputs, native scoring, source videos and timing/resource monitoring. It is static
and does not periodically refresh or rewrite media.

## Interface and implementation

1. A fresh Luna/high session per case used the existing Molmo2 pointer and MobileSAM to select
   door material and fixed opening/rim regions in agentview and wrist. The selected capture
   is common to both views. Empty selections remain explicit when the part is not visible.
   No exact pair of mating pixels is required.
2. CoTracker3 tracked up to 64 queries per part per camera. Agentview used all 541/542 recorded
   RGB frames; wrist used its 15 saved captures because no dense wrist video exists in these
   recordings. Correspondences were lifted using the depth and calibration at each capture.
3. The existing general 3D rigid fitter combined both cameras' world correspondences. Its
   4 mm residual, eight-inlier, majority and spatial-span settings were unchanged. Agentview-only
   and wrist-only fits used exactly the same selections and tracker outputs as comparisons.
   There is no automatic switch between these three conditions.
4. Selected-region proximity is descriptive: nearest-surface distances between the observed
   door and opening clouds carried by their fitted poses. A smaller value is not labelled
   closing automatically. An unresolved pose or absent region produces no distance.
5. A separate fresh Luna/high session inspected before/after image pairs from both cameras,
   region descriptions and compact measured feedback. It assigned opening, closing, unchanged
   or unclear based on the visible relationship. No native state, orbit, signed hinge angle,
   successful coordinates, previous notebook or expected movement label was provided.

Three intervals per case were fixed by selecting four uniformly spaced retained-depth captures,
without native scoring. Both model outputs were frozen before the evaluator read native labels.
The report and evaluator files contain privileged diagnostic labels and must not be supplied
as Luna practice input.

The old motion-circle prompt generator was deleted, along with its dedicated test. Historical
artifacts remain intact; its former implementation is available at commit `d0be4a1`. No new
planner restrictions, collision checks, fallback algorithms or simulated motion were added.

## Frozen motion judgments

| State | Controls | Native opening change | Luna | Strict directional score |
|---|---|---:|---|---|
| 17 | 60 → 276 | −31.38° | closing | correct |
| 17 | 276 → 437 | +6.19° | opening | correct |
| 17 | 437 → 600 | +11.19° | opening | correct |
| 49 | 60 → 338 | +12.71° | opening | correct |
| 49 | 338 → 468 | +2.90° | opening | correct |
| 49 | 468 → 600 | −0.37° | unchanged | incorrect; very small movement |

The exact native direction determines this score; no tolerance was introduced after seeing
Luna's answers. The last mismatch is a small-motion classification error, not a new robot
failure or a motion rejected by the planner. Its source video and paired images are shown
at interval 3 in state 49. No fault was injected.

## Tracking and remaining limitations

| State / camera condition | Available door poses | Non-registration poses | Max / median non-registration centroid error |
|---|---:|---:|---:|
| 17 / agentview | 15/15 | 14/14 | 4.89 / 2.42 mm |
| 17 / wrist | 5/15 | 4/14 | 6.31 / 3.54 mm |
| 17 / both | 15/15 | 14/14 | 4.89 / 2.17 mm |
| 49 / agentview | 3/15 | 2/14 | 5.81 / 5.18 mm |
| 49 / wrist | 1/15 | 0/14 | unavailable |
| 49 / both | 3/15 | 2/14 | 5.81 / 5.18 mm |

The registration capture is fitted to itself; it is excluded from error summaries and the
non-registration tracking count. In state 49, only four captures retain depth. Its wrist-only
result is just the reference capture, not evidence of successful subsequent tracking.

With the same new selections, adding wrist did not increase availability over agentview alone.
Wrist views are clipped, occluded and temporally sparse. Do not attribute the change from the
earlier experiment solely to a second camera: region selection and the reference capture also
changed. State 17 selected observation 55 and uses offline backward tracking; state 49 selected
observation 1. This does not establish an online result using only an episode prefix.

At state 49 control 338, the door fit is unresolved: agentview contributes 31 measured pairs
and wrist contributes six. The best group has 11/37 matches within the existing 4 mm residual,
below the required majority. The arm obscures much of the selected upper rail. This is
inconsistent correspondence evidence, not demonstrated geometric infeasibility or a planner
timeout. The webpage shows all query locations and independent native-motion residuals.

State 17's selection agent could not confidently identify a fixed opening/rim in either view,
so the numerical door-to-rim distance remains unavailable. Luna nevertheless identified the
motion relative to the visible housing correctly. State 49 selected partial fixed regions;
its full-region proximity is available at three captures. Neither result establishes a
generally reliable automatic closed-state detector.

The original recorded robot episodes both ended at the 600-control limit. This analysis has
no new planning, IK, grasping or execution stage, and does not change their task outcomes.

## Time and resources

| Stage | State 17 | State 49 |
|---|---:|---:|
| Luna region-selection session, including tools | 279.65 s | 172.98 s |
| Molmo2 point-tool service time | 10.42 s | 2.55 s |
| Segmentation service time | 6.38 s | 1.88 s |
| CoTracker inference, both cameras | 6.74 s | 4.27 s |
| Complete tracking and measurement | 12.26 s | 6.61 s |
| Luna motion-judgment session, including tools | 110.08 s | 153.43 s |

The two numerical tracking/measurement passes took 18.87 seconds total. Model sessions dominate
this test's wall time. Session time outside tool service includes inference, image inspection,
transport and orchestration; it is not claimed to be pure model inference.

All three phases used the existing monitored launcher, a 32 GiB cgroup, zero swap allowance
and whole-group OOM handling. Peak accounted job RAM was 3.54 GiB, peak selected-GPU memory
36.14 GiB, and minimum available host memory 86.47 GiB. All 51 SSH probes succeeded, no OOM
kills occurred, and maximum sampled full I/O pressure average was 0.62%. All model/tracker
workers exited. Other users' jobs were left running.

## Validation and entry points

- 84 relevant tests passed; one older saved-failure-fixture test was skipped because its
  fixture is absent. New tests verify actual two-view fusion, missing-depth behavior and
  orientation-invariant region proximity without task labels. Ruff and diff checks passed.
- Source PNGs exactly match the RGB arrays in both cameras' reference RGB-D bundles.
- Browser checks covered 180 capture/condition/region views, six interval pairs, 20 model/data
  links, both videos and 360 px layout. The page is approximately 15 KB plus a 511 KB data file.
- No CMake build or fresh simulator episode was needed for the recorded test.

The numerical module is `src/vlm_tamp/region_motion.py`. The four explicit study phases are:

```bash
python scripts/select_region_motion.py --study <new-output-directory>
python scripts/measure_region_motion.py --study <new-output-directory>
python scripts/judge_region_motion.py --study <new-output-directory>
python scripts/report_region_motion.py --study <new-output-directory>
```

Use the existing configured robotics environment and monitored launchers, saved under
`.runtime/two-view-regions-20261103/`. These are offline study interfaces; they have not replaced
the live actor's tool contract. The next live test should use only past/current frames, retain
both RGB streams and the required action-boundary depth, and keep task success separate from
motion-judgment accuracy.
