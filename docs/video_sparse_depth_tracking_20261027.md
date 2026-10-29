# Five versus 58 RGB frames with the same sparse depth

The registered comparison is complete. Denser RGB tracking improved the later
hinge fit but worsened the middle fit; both video conditions produced one
motion-supported estimate. This does not establish a universally better frame
count or resolve the earlier autonomous Luna failure.

A subsequent [live Luna comparison](luna_live_hinge_comparison_20261027.md)
integrated recorded-video tracking and explicit estimate application. The
offline results below remain unchanged; the live study reports adoption,
task outcomes and planning limitations separately.

- [Interactive comparison](../results/video-sparse-depth-tracking-20261027/watch.html)
- [Accuracy, support and timing figure](../results/video-sparse-depth-tracking-20261027/comparison-summary.png)
- [PDF figure](../results/video-sparse-depth-tracking-20261027/comparison-summary.pdf)
- [All interval measurements](../results/video-sparse-depth-tracking-20261027/interval-results.csv)
- [Registered protocol](../results/video-sparse-depth-tracking-20261027/protocol.json)
- [Prediction integrity](../results/video-sparse-depth-tracking-20261027/prediction-integrity.json)

## What was held fixed

The original retained agent-view movie contains a frame for every control step
from 221 through 278: 58 RGB frames. Its retained frame/control-step index
provides alignment; the encoded playback rate of four frames per second is not
used as physical sensor time. No simulator replay or rendering was needed.

The main conditions use either all 58 decoded frames or the exact five-frame
subset at controls `221, 242, 259, 277, 278`. Both use the original Luna-selected
`part001-0` mask, its exact 200 agent-view queries, the same CoTracker checkpoint,
the same final reference frame, and unchanged fitting/checking thresholds.
Segmentation and feature selection were not rerun.

CoTracker receives RGB-only dictionaries. After tracking, only its predictions
at the five known depth timestamps are lifted using the original measured
world points, calibration and known-robot exclusion. The 53 other timestamps
have no measured depth and supply no invented/interpolated 3D points.

This is a single-camera comparison because the existing movie records only
agentview. The earlier five/sixteen-capture study used three cameras and is a
different experiment. No actor, mask or interval was selected using grading
labels. All predictions were saved before private accuracy scoring.

## Results

Axis errors below are private post-prediction measurements. The unsigned upright
axis is supplied as a prior; the table tests its position, not arbitrary-axis
recognition.

| Fit interval, controls | Five video frames | 58 video frames | Later check |
| --- | ---: | ---: | --- |
| 221→242 | No admitted fit | 39.20 mm error, 10/50 inliers | Dense candidate disagrees with later motion |
| 242→259 | 3.13 mm error, 44/159 inliers | 6.69 mm error, 46/161 inliers | Both motion-supported |
| 259→277 | 6.29 mm error, 75/166 inliers | 1.35 mm error, 138/170 inliers | Both inconclusive: final checking interval lacks informative motion |
| 277→278 | No admitted fit | No admitted fit | No later interval available |

The 1.35 mm estimate is accurate according to the grader but is not a verified
estimate available to Luna. The small final motion cannot validate it. Do not
choose it using the grading result and report that as autonomous recovery.

Three balanced-order timed passes were recorded for each video condition after
warming up both sequence lengths. Pixel tracks, visibility and lifted points
were identical across those repeated passes. These repeats measure runtime
stability; they are not independent task trials.

| Cost | Five video frames | 58 video frames |
| --- | ---: | ---: |
| Median warm tracking wall time | 0.159 s | 0.906 s |
| Median lifting of the same five depth snapshots | 0.027 s | 0.025 s |
| Peak allocated CUDA tensor memory | 0.56 GiB | 8.67 GiB |
| Four fits and three checks, one pass | 0.279 s | 0.366 s |

Decoding the movie and exporting the 58 review PNGs took 2.95 seconds once.
That includes artifact writing, not just streaming decode. Initial model load
and warm-ups are separate in [tracking-timing.json](../results/video-sparse-depth-tracking-20261027/tracking-timing.json).
Peak allocated tensor memory is not total device-reserved memory.

## Input-format diagnostic

A separately registered five-PNG condition uses the original RGB-D images at the
same timestamps and the same queries/depth. Its middle fit had 10.71 mm error
and failed the later-motion check, while the five decoded-video images produced
the supported 3.13 mm fit. The later PNG fit had 3.11 mm error and remained
unverified.

This shows sensitivity to the RGB input representation. It is not proof that
compression improves tracking: the original PNG and retained JPEG/MP4 paths
are not a controlled re-encoding of one identical raw image sequence. The
primary video5/video58 comparison does control encoding by sharing exact
decoded frames. Anchor RGB differences and every result are retained.

## Interpretation and limits

Dense RGB with sparse depth is technically viable and adds less than a second
of warm tracking time in this clip. Additional temporal context changes the
learned correspondences and visibility; the resulting 3D support and axis
accuracy are not monotonic in frame count. The current hinge estimator still
fits selected pairs of measured 3D snapshots. It does not jointly optimize the
hinge over every intermediate RGB observation.

This experiment supports offering dense video as an information source. It does
not justify making the maximum available frame count an accuracy guarantee.
Reliable motion-window selection, correspondence quality and axis observability
remain separate issues. The prior short-window false-acceptance findings remain
valid; this experiment does not replace them.

No production perception RPC, live capture policy, estimator, grasp planner,
controller or learner memory changed. The live `track_part_motion` interface
still accepts at most 16 recorded RGB-D observation IDs. This comparison calls
the existing tracker directly with 58 RGB frames and lifts only sparse anchors;
exposing that distinction through a production tool is separate work.

## Visual and code inspection

The viewer shows solid dots for depth-backed points and hollow circles for
RGB-only tracks. Selecting Depth at an intermediate timestamp explicitly shows
that no depth exists there. Track lines connect sampled RGB positions; Rigid fit
connectors show displacement between the selected interval's endpoints.
Grader-only axis errors are for this human report and were not tracker inputs.

- [Experiment driver](../results/video-sparse-depth-tracking-20261027/run_comparison.py)
- [Video comparison builder](../scripts/build_video_tracking_comparison.py)
- [Shared visualization template](../templates/part_motion_walkthrough.html)
- [Post-prediction grading](../results/video-sparse-depth-tracking-20261027/grading-private.json)
- [Browser checks](../results/video-sparse-depth-tracking-20261027/browser-check.json)

Input hashes, exact subset alignment and repeated output equality were checked.
The browser checks cover seven available stages across three input conditions,
explicit missing-depth handling, stage-specific controls and narrow layouts.
Visual previews are compressed; numerical calculations retain full precision.
Original experiment artifacts are preserved.
