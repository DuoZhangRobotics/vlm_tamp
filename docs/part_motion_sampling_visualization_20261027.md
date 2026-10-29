# Inspecting the perception pipeline and testing denser capture

More captures improved tracking and one matched-window hinge estimate, but the
fresh Luna trials still ended without a selected supported estimate. This is
development evidence on the previously examined microwave, not generalization
or a new autonomous manipulation success.

A subsequent [five-versus-58 RGB-video comparison](video_sparse_depth_tracking_20261027.md)
uses the existing agent-view movie and the same five depth timestamps. It keeps
encoding fixed between the main conditions and reports mixed accuracy results.

- [Interactive walkthrough](../results/part-motion-pipeline-visualization-20261027/watch.html)
- [Full-resolution stage figure](../results/part-motion-pipeline-visualization-20261027/figures/pipeline-overview.png)
- [PDF figure](../results/part-motion-pipeline-visualization-20261027/figures/pipeline-overview.pdf)
- [Paired numerical comparison](../results/part-motion-pipeline-visualization-20261027/density-comparison/summary.json)
- [Private accuracy grading, performed after predictions](../results/part-motion-pipeline-visualization-20261027/grading-private.json)

The walkthrough uses retained measurements, not illustrative masks or invented
tracks. Its stages show raw RGB, Luna's coarse box, the selected MobileSAM mask,
calibrated associations, corner/grid query points, measured depth, CoTracker
tracks, rigid-fit membership and subsequent motion checks. Camera and capture
controls expose the intermediate observations. The three recording choices
separate the original failure from the paired five/sixteen-frame replay.
The displayed query points are tracking seeds, not semantic embeddings.

Previews are compressed and the depth preview uses 16 grayscale bins. Coordinates
remain in the original 512-pixel convention, and all estimates use the original
precision arrays. The standalone figures preserve higher image detail. Inlier
membership was reconstructed with the production fitter's exact selection
procedure and checked against its counts and numerical transforms before
drawing overlays. A projected upright axis is an estimate, not a native label.

## Sampling experiment

The original clip contained captures at control steps 221, 242, 259, 277 and 278.
No intermediate RGB-D captures were retained. The last pair spans one control
step and contains almost no rotation. We replayed the exact previously issued
joint references, retaining the original initialization, gripper commands,
controller and native-success stop, and captured at:

`221, 226, 231, 236, 241, 242, 246, 251, 256, 259, 261, 266, 271, 276, 277, 278`.

The first two anchor captures match the source exactly. Later differences are
small but nonzero: at step 259 only 46 agent-view color channels differ by one
intensity level; at step 278 the maximum agent-view depth difference is about
5.3 micrometres. All anchor calibrations match. These comparisons and the
issued-reference audit are retained under `dense-capture/`.

The causal sampling comparison therefore uses **two subsets of the same new
replay**, rather than assuming the whole replay is byte-identical to the old
run. Both subsets use the same final-frame Luna mask, associated masks, query
points, CoTracker checkpoint and estimation thresholds. The old mask is held
fixed as an experimental input; no new manual mask or known axis is supplied.

For fitting on controls 242→259 and checking 259→277:

| Measurement | Five captures | Sixteen captures |
| --- | ---: | ---: |
| Rigid inliers / filtered correspondences | 40 / 232 | 58 / 239 |
| Later axis-prediction RMS | 4.16 mm | 2.92 mm |
| Existing motion check | Disagrees | Supported |
| Grader-only axis-line error | 10.25 mm | 6.70 mm |

Across the three checks at the original physical time intervals, supported
estimates increased from zero to one. These are correlated windows, not three
independent trials. Zero unsigned direction error follows from the supplied
upright-axis assumption.

## What still fails

Additional short-window checks exposed a limitation of the existing admission
rule: six adjacent dense-window checks passed motion consistency, but three
had axis-line errors of 11.9, 17.3 and 24.1 mm, outside the pre-existing 10 mm
grading screen. These failures are retained, not filtered from the report.

There is a geometric reason that a small motion residual is insufficient. With
known planar rotation R, shifting a fixed pivot by delta changes predicted
motion by `(I - R) delta`. The magnitude multiplier is `2 sin(|theta| / 2)`.
For example, at 5 degrees a 3 mm displacement discrepancy can correspond to
about 34 mm of pivot displacement. This is an idealized sensitivity calculation,
not a calibrated error bound in the presence of tracking/depth/rotation error.

Dense images help maintain correspondences. The physical motion span used to
estimate and check the axis must also constrain its location. Repeated static
images do not supply that constraint. The next generic numerical improvement
should expose axis observability over a motion window and test predictions
over sufficient cumulative rotation. No threshold was loosened here.

Track quality also remains a problem. The paired experiment's agent-view query set contains
15 detected corners and 185 grid samples. In the matched interval, the denser
tracker retains 58 global rigid inliers, of which only two originated from
detected corners across the three cameras. This does not support simply deleting
all grid queries or blaming one detector. The figure shows many inconsistent
tracks on the frame; mask identity, low texture, occlusion and depth association
have not been causally separated. See the
[query-origin measurements](../results/part-motion-pipeline-visualization-20261027/feature-origin-diagnostic.json).

## Luna retries and timing

Two fresh Luna sessions received raw observations, the generic introduction,
the current four-tool interface, 14 requests and 480 seconds. No prior masks,
successful intervals, native labels, teacher or reflection reviewer entered
either actor. Both selected their own masks and attempted alternatives. Both
ended unresolved after nine requests. The five-frame session also made one
invalid request treating `interval_motion`, a reply field, as an operation.

| Session | Wall time | Time in numerical tools | Selected supported estimate |
| --- | ---: | ---: | --- |
| Original five captures, current interface | 80.98 s | 11.38 s | None |
| Sixteen replay captures, current interface | 87.13 s | 14.81 s | None |

The remainder includes model inference, image inspection, shell work and
transport; it is not a pure reasoning timer. The actor comparisons also differ
in selected masks/windows and slightly in replay observations, so their results
do not isolate sampling density as the paired numerical comparison does.

- [Five-frame actor result](../results/part-motion-pipeline-visualization-20261027/recovery/result.json)
- [Five-frame tool requests/replies](../results/part-motion-pipeline-visualization-20261027/recovery/trials/clip2/tool-events.jsonl)
- [Sixteen-frame actor result](../results/part-motion-pipeline-visualization-20261027/dense-recovery/result.json)
- [Sixteen-frame tool requests/replies](../results/part-motion-pipeline-visualization-20261027/dense-recovery/trials/clip2_dense/tool-events.jsonl)

The simulator replay took 76.17 seconds, with 45.52 seconds in the 16
capture/export operations. CoTracker inference across three cameras took 1.12
seconds for the first five-frame condition and 0.73 seconds for the subsequent
sixteen-frame condition; the first condition also loaded the model. These are
single sequential timings with warm-up effects, not evidence that more frames
are faster. Capture/export remains substantial and is not optimized here.

No live capture policy, estimator, planner, controller, learner guide or memory
promotion changed. Denser captures were collected only in this explicit replay.
Current motion-supported outputs remain consistency evidence rather than a
calibrated reliability certificate. New numerical changes need generic tests
and unfamiliar-asset validation before any claim of dependable transfer.

## Reproduce and inspect

The experiment drivers and frozen actor sources are under
`results/part-motion-pipeline-visualization-20261027/`. Run the visualization
builder after sourcing `scripts/server_runtime_env.sh`:

```bash
python scripts/build_part_motion_walkthrough.py --output results/part-motion-pipeline-visualization-20261027
```

The literal interactive template is
[`templates/part_motion_walkthrough.html`](../templates/part_motion_walkthrough.html).
The browser check covers eight stages in three recording conditions, mobile
layout, stage-dependent controls and theme changes. Screenshots and source-data
hashes are retained alongside the report. Original experiment artifacts were
not rewritten.
