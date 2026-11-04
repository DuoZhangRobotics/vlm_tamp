# Whole-door tracking and reference selection — 2026-11-03

Whole-part tracking reduces point drift when its rigid fit is supported. It does **not** yet
provide reliable closing-progress feedback for both recordings. No new robot motion ran, and
the tested goal-reference interface has not been added to live control.

Open `results/whole-part-gap-20261103/watch.html` in VS Code Live Preview. It shows the exact
model selections, masks, all tracking queries, rigid inliers, estimated gaps, independent native
evaluation, timing, source videos and working input/output links. The page does not refresh.

## Changes and protocol

- Fresh Luna/high sessions used Molmo2 proposals, existing MobileSAM segmentation and measured
  RGB-D patch geometry to select the moving reference, fixed reference and whole-part masks.
  Both original cameras and all saved captures were available. No successful coordinates,
  simulator geometry, native angles or closing sign were supplied.
- Track up to 64 spatially distributed landmarks on each selected part using CoTracker3.
  The separate marked-point track is excluded from the pose fit. All 541/542 recorded external
  RGB frames are used; this includes offline backward tracking when the reference is late.
- Fit each retained-depth capture directly against its registered reference. Reuse the existing
  general 3D `fit_object_motion` estimator: 4 mm residual, at least 8 inliers, strict majority,
  25 mm spatial span. These existing settings were not changed. No upright-axis assumption is
  used. A failed fit remains missing; it is not replaced with a prior pose.
- Carry the marked point using the fitted part transform. Label the resulting gap as estimated,
  distinct from direct depth sampled at the individual point track.
- Corrected an actual query-sampling bug: the previous row-ordered fill placed most added
  landmarks near the mask's top edge. The fill now uses farthest-point spacing across the mask.
  The earlier state-17 result is preserved in `state017/whole-part-row-biased`.
- Corrected `lift_tracks` requiring an unused RGB array just to determine image dimensions.
  It now uses the world-point array's dimensions. The first GPU pass completed but its reader
  raised `KeyError: 'rgb'`; its failed resource record remains visible. Saved outputs were
  explicitly reprocessed before the sampling correction prompted a new tracking pass.
- The existing rigid fitter now returns original inlier indices for visualization. Its numerical
  solution and acceptance criteria are unchanged.

The live planner/controller, collision policy, control budget and motion direction interface
were not changed. The only shared tracking behavior changes are query distribution and the
depth-lifting interface correction.

## Recorded results

| Measurement | State 17 | State 49 |
|---|---:|---:|
| Captures with a resolved whole-part gap | 5/15 | 3/15 |
| Captures retaining depth | 15 | 4 |
| Maximum carried-reference error in accepted fits | 1.44 mm | 9.96 mm |
| Maximum supported direct-point error on the same new moving reference | 16.82 mm | 12.10 mm |
| Initial reference-selection session, including tools | 304.60 s | 323.43 s |
| CoTracker inference | 4.64 s | 7.79 s |
| Decode, tracking and measurement | 8.60 s | 8.80 s |

The error maxima have different supported-capture sets and are not a paired statistical claim.
The older single-point experiment also used different model-selected references. Its results
remain at `results/recorded-goal-gap-20261103/watch.html`.

State 17 has ten unresolved moving-part fits. At control 276, for example, only 20 of 63 measured
matches agreed on a rigid transform within the existing 4 mm residual. CoTracker visibility is
not equivalent to a correct material-point correspondence. Accepted poses have small error;
the unresolved intervals must not be presented as confirmed progress.

State 49 has eleven missing-depth captures due to post-episode retention, plus one inconsistent
fit at control 60 (27 of 57 matches). This retention is a limitation of the recorded test, not
the cause of the earlier live task failure. The initial selected gap changes in the wrong
direction at controls 338→468: the door opens 2.90°, while estimated gap decreases 0.80 mm.
The 468→600 change also has the wrong sign, but is only 0.47 mm. Neither is reliable feedback.

Native labels are read only by the report after selections and fits are frozen. The original
robot videos are embedded in the page. Both original episodes stopped at 600 controls; these
recorded-only tests have no motion-planning budget, collision rejection or executed push stage.

## Reference-selection revision and its limitation

Geometry feedback reduced the earlier 129/91 mm height mismatch to approximately 3 mm, but
neither first session claimed a verified matching closure reference. The fixed choices were
still tentative appliance points, not demonstrated closed-position targets.

A separate offline experiment showed Luna the full orbit inferred from the accepted door poses.
It supplied neither a preferred angle nor a direction. Luna declined to select a final fixed
reference in both revisions; its exact reasoning summaries, inspected proposals and candidate
geometry remain in `orbit-refinement/state017` and `state049`.

**Do not treat these rejections as proof that the scene is visually impossible.** In state 49,
the accepted poses contain only 0.18–0.20° estimated rotation. The resulting orbit axis is
76.66° from the withheld native axis despite a 0.35 mm equation residual. Small fitting residual
does not establish an accurate articulation axis. State 17 has 6.2–16.2° measured rotations and
0.53° axis error, but still lacks a confirmed matching fixed feature.

The orbit prompt generator was removed during the subsequent two-camera region experiment,
as explicitly requested. Its original source remains in commit `d0be4a1`; historical results
and exact prompts are preserved. Its `supported` field meant the numerical equations produced
an orbit, not that an axis was independently verified. Do not promote that field into an action decision.
The core module `part_goal_reference.py` contains only rigid-part reference carrying and gap
measurement, with semantic verification explicitly false.

The next useful diagnostic is visibility and identity of the fixed opening reference, plus
recovering moving-part correspondences through occlusion. More frames alone did not solve
either issue. A new live closing trial is not justified by these offline results alone.

Follow-up: [two-camera region perception](two_view_regions_20261103.md) removes the orbit hint,
compares both cameras with matched single-camera conditions, and tests fresh Luna motion judgments.

## Resource use and validation

- All GPU workers exited. Other users' jobs were left running.
- 32 GiB cgroup limit; swap disabled. Peak accounted memory reached 32 GiB during first model
  loading, mostly file cache; peak anonymous/shared memory was about 3 GiB. No OOM kills.
- Minimum available host RAM: 81.38 GiB. Peak selected-GPU memory: 36.15 GiB.
- 75/75 SSH checks succeeded. Maximum sampled full I/O pressure average: 5.09%.
- 82 relevant tests passed, one saved-fixture test skipped. Tests cover outlier-resistant rigid
  carrying, missing poses, arbitrary-axis geometry, depth lifting, spatial sampling and existing
  perception/contact tracking behavior. Ruff passed.
- Browser checks passed: 30 saved capture rows, both reference revisions, 26 document links,
  actual selected-pixel coordinates and 360 px layout. No new simulator episode or CMake build
  was required.

Reproduction entry points:

```bash
python scripts/select_part_goal_references.py --study <new-output-directory>
python scripts/measure_part_goal_gap.py --study <new-output-directory>
python scripts/report_part_goal_gap.py --study <new-output-directory>
```

Use the configured robotics environment and the existing monitored, memory-limited launcher
for GPU work. Saved tracking outputs can be explicitly reprocessed with `--use-saved-tracks`;
capture/video hashes and the query set are checked. This is not an automatic runtime retry.
