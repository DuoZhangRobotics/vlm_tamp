# Part masks and observed-motion tools for Luna

Four optional perception tools are implemented and connected to the ordinary
Luna robot interface. Luna supplies coarse mask prompts and chooses parts,
views and motion intervals. Numerical tools produce masks, calibrated
correspondences, tracks and upright-hinge estimates. No assistant masks or
hinge coordinates were supplied to the actors.

Across successive development passes, Luna selected motion-supported estimates
with **6.25, 4.52, 4.03 and 3.46 mm** hinge-line position errors on four recorded
trajectories. A fifth recording remained unresolved. These are perception
replays of prior Luna motion on a microwave asset, not new manipulation
successes or an independent pooled success-rate benchmark. The two additional
recordings used to test the final estimator correction were predicted before
their native scoring. Earlier failures, versions and source snapshots remain.

- [Dashboard: prompts, masks, associations, tracks, decisions and timings](../results/luna-part-motion-tools-20261027/watch.html)
- [Summary and limitations](../results/luna-part-motion-tools-20261027/conclusion.json)
- [Full native scoring](../results/luna-part-motion-tools-20261027/evaluation-private.json)
- [Every tool-call timing](../results/luna-part-motion-tools-20261027/tool-timing.csv)
- [Tracker timing breakdown](../results/luna-part-motion-tools-20261027/tracking-timing.csv)
- [Every interval’s estimate, rejection and accuracy](../results/luna-part-motion-tools-20261027/accuracy.csv)
- [Later visual walkthrough and five/sixteen-capture experiment](part_motion_sampling_visualization_20261027.md)

## Runtime interface

Add `--part-motion-tools` to `scripts/run_subscription_agent_pilot.py`.
The flag enables the service operations and generates their individual help
files and JSON schemas. It is off by default. The existing compact interface
keeps arrays in files and presents IDs, support measurements, failures and a
bounded number of overlay paths. Observed replies had a median size of about
2.5 kB and a maximum of 5.5 kB; these are bytes, not token/context-size estimates.

The later [live comparison](luna_live_hinge_comparison_20261027.md) adds
`track_video_motion` for recorded RGB with measured depth anchors and
`apply_joint_estimate` for an explicit planner-model update. The original
perception operations remain read-only. Current onboarding includes these
live operations; the earlier introduction/replay artifacts remain archived.

New runs with that flag also receive a
[372-word introduction](../prompts/part_motion_introduction.txt) at the start
of `tools.txt`, in both full and compact modes. A separate
`pipeline-introduction.txt` copy is saved in the actor workspace and run output;
its checksum is recorded in `protocol.json` and its source is included in the
runner's source snapshot. Detailed arguments remain in the operation contracts.
The introduction explains tool relationships, evidence limits, the need for
observed movement, and Luna's responsibility for experiments and memories.
It contains no recording-specific selections, hinge coordinates or solution.

See the [generated compact tool index](../results/luna-tool-introduction-20261027/compact-enabled/tools.txt)
for the actual delivered text. This onboarding is supplied documentation, not
learner-authored memory. Existing example and memory options are unchanged;
an introduction-only experiment must configure and record those separately.
Preparing this introduction does not resolve the outstanding recording or
establish new manipulation success.

| Operation | Luna supplies | Numerical output |
| --- | --- | --- |
| `segment_parts` | Observation, camera, coarse box and/or positive/negative clicks | Up to three MobileSAM masks, part IDs, scores, usable-depth counts and overlays |
| `associate_views` | Part ID and other camera names; optional comparison part | Depth-supported projections, occlusion/free-space disagreement and overlap evidence |
| `track_part_motion` | Part ID, ordered observations, LK or CoTracker3, optional associated camera masks | Visible measured 3D tracks, interval motion summaries and reference/endpoint overlays |
| `estimate_and_verify_joint` | Track ID, fitting interval and optional later verification interval | Independently initialized upright hinge, explicit observability failures and fixed-axis prediction residuals |

MobileSAM is the existing installed segmentation backend. Its `label` argument
is a caller description, **not** a text prompt or a detected category. SAM 3
was not installed or evaluated in this implementation.

Association uses the existing camera calibration and a 4 mm depth-agreement
test. Only supported projections enter transferred masks. Occluded points are
not filled with guessed geometry. High overlap can also mean containment, so
the tool does not claim that overlap alone proves physical part identity.

Tracking may start from a part selected in **any** supplied frame, including
the last. Observation IDs remain chronological. LK runs forward/backward around
the reference; CoTracker3 uses its official query-time and backward-tracking
support. Each pixel track is lifted using that frame’s measured depth/world
points. Invisible predictions, invalid depth and known robot pixels are
excluded. Gripper motion is never substituted for object motion.

The estimator uses `prior=None`: it can initialize an axis independently of
Luna’s old hinge guess. Its model is explicitly an **upright revolute
hypothesis**, with an unsigned world-Z axis. It does not classify general joints
or infer the requested opening/closing direction. Axis height is a gauge choice
along a line; evaluation uses a line metric, not distance between arbitrary
points chosen along that line.

All four operations leave robot controls, the planning scene and geometry epoch
unchanged. They do not automatically install an estimate or certify contact,
semantic identity, collision-free execution, or task success. In the live
service they follow the episode lifecycle; the same numerical engine can also
operate on archived observations, as in this experiment.

Implementation:

- [Mask proposals and calibrated associations](../src/vlm_tamp/part_perception.py)
- [LK/CoTracker adapters and independent fitting/verification](../src/vlm_tamp/part_motion.py)
- [Episode registries and artifact-producing tools](../src/vlm_tamp/perception_tools.py)
- [Request contracts](../src/vlm_tamp/robot_contracts.py)
- [Production service integration](../scripts/serve_subscription_robot.py)
- [Component and regression tests](../tests/test_part_motion_tools.py)

## What the development runs established

Ten fresh Luna sessions used raw recordings, generic tool contracts and six
initial/later raw image attachments per session. Additional views were available
on demand. No human mask examples, native object geometry, teacher model,
reflection reviewer, prior answer guide or cross-session learner memory was
provided. Luna wrote its own decisions and procedural notes. Native scoring
was separate from inference.

| Recording | Latest actor pass | Selected estimate | Hinge-line position error |
| --- | --- | --- | ---: |
| Opening state 20 | CoTracker/feedback follow-up | Motion-supported | 6.25 mm |
| Opening state 13 | Final independent-estimator follow-up | Unresolved | — |
| Closing state 49 | CoTracker/feedback follow-up | Motion-supported | 4.52 mm |
| Opening state 10 | Final independent-estimator follow-up | Motion-supported | 4.03 mm |
| Opening state 19 | Final independent-estimator follow-up | Motion-supported | 3.46 mm |

This table describes the latest development result per recording, not the best
native-scoring candidate. Versions, instructions and some mask/reference choices
changed, so these rows must not be presented as a controlled improvement rate.
Zero unsigned axis-angle error follows from the supplied upright prior; it is
not learned arbitrary-axis recognition.

The initial LK sessions returned no selected hinge estimates. Points were lost
rapidly in the sparse recordings, and some tracks appeared to follow the
cabinet’s rear/opening surface rather than the moving door, as Luna also noted.
This does not fully separate mask-selection error from tracking or depth-association
error. CoTracker retained
more depth-backed points, but retention alone did not establish correctness.
For example, an initial single-interval fit had roughly 200 mm position error
and received no later-motion support.

Luna improved its choices when the interface showed track overlays and raw
displacement summaries and the generic instructions encouraged another view,
mask or reference frame after failure. On opening state 20 it selected a later,
more informative reference. In the paired numerical comparison using that new
selection, **LK also produced a supported 2.36 mm fit**, while CoTracker produced
the actor-selected 6.25 mm fit. Better part/reference selection mattered; the
result is not solely a tracker-model gain.

The remaining opening-state-13 case is a concrete limitation. Luna identified
the visible open panel in a later image, but early measurements were weak or
inconsistent. In one retained test, a 17.9 mm-biased candidate predicted another
interval with 4.97 mm RMS and 6.57 mm P90 error, failing the 3 mm/5 mm verification
gates. Luna also tried alternative masks/intervals and retained an unresolved
decision. We did not supply corrected pixels or relax prediction residuals to
force acceptance.

## Corrections found through actual tool use

**Fixed-axis validation and initialization are different questions.** The first
implementation incorrectly required the later interval to independently admit
another pivot. A known axis can be tested from an observable rigid rotation even
when that view alone cannot initialize a pivot. Verification now requires an
observable rigid pose and at least 1.5° rotation, then checks the fixed-axis
prediction with the unchanged 3 mm RMS and 5 mm P90 gates. The original results
and [correction registration](../results/luna-part-motion-tools-20261027/verification-fix-registration.json)
are preserved.

**The reference image need not be the first image.** Luna’s own request selected
a part in observation 5 and requested chronological observations 1–5. Our initial
wrapper rejected it despite CoTracker supporting that use. The interface now
supports arbitrary references. An exact replay of those original Luna requests
through `RobotService.dispatch` passed with identical mask pixels, a no-RPC
sentinel, and unchanged controls, scene and epoch. This was recorded-service
integration, not a new simulator manipulation. A test-harness archive-path
lookup error was corrected and preserved in `reference-request-history`.

**The tracked points’ radius is not the handle’s lever arm or door size.**
Post-run scoring revealed accurate later fits rejected because their inlier
points lay within 6 cm of the axis. A synthetic spatially spread 4 cm-radius
patch demonstrated the generic problem. The shared fitter now has an explicit
`minimum_radius_m` parameter whose default remains 0.06; the new independent
tool passes 0.0. Existing `RGBDHingeTracker` behavior is unchanged.

The independent fit retains the other existing gates: at least six measured
moving landmarks, 6 cm XY span, 1.5–70° rotation, rigid RMS below 2 mm, bootstrap
spread below 15 mm, and median visible-point radius below 70 cm. It now reports
failed checks explicitly. These are estimator assumptions, not motor execution
tolerances or calibrated probabilities. The correction was registered after
earlier scoring and checked prospectively on the two additional recordings;
the old problematic recording remained an explicitly developmental recheck.

**Subsequent interface correction: expose fit support and distinguish missing
verification evidence from disagreement.** The interval summaries now include
the number of filtered correspondences, rigid-motion inliers, their RMS error,
and failed admission checks. Compact replies retain these measurements. On
the unresolved recording, the successive moving intervals used 8/105, 42/231
and 111/273 filtered correspondences. Those ratios are support measurements,
not calibrated probabilities or proof of the cause of the correspondence errors.

An insufficiently observable checking interval now returns
`verification_inconclusive`; measurable motion that disagrees with the candidate
still returns `verification_failed`. Both leave `motion_supported=false` and
leave the planner unchanged. A disagreement does not by itself isolate an axis
error from bad tracks. The numeric fits, verification residuals and thresholds
were preserved. A replay through the current tracking tool reused Luna's
original masks, queries and recorded tracker outputs, reproduced the lifted
tracks exactly, and checked the compact replies. The focused suite passed
53 tests. See the [diagnosis and replay record](../results/part-motion-evidence-diagnostics-20261027/summary.json).

The original actor stopped after five of fourteen allowed tool calls without
another mask/camera selection. It also had no robot actions: this was a fixed
recording. The later interval had too little observed rotation to verify a
candidate fitted from the preceding interval. This is not evidence that every
possible selection would fail or that a live probe would succeed. The next
useful experiment is a live Luna recovery with fixed tools and the generic
introduction, including the existing optional `select_surface_contact` and
`plan_pose` interfaces for probes that do not require a hinge estimate first.
Luna must select the contact and experiment; a surface contact/path plan alone
does not establish actual object motion. No such live recovery was run during
this interface correction, and the unresolved case remains unresolved.

## Backend comparison and timing

Every paired comparison used the exact mask selected by Luna, identical camera
sets, identical query points and identical RGB-D frames. Predictions were saved
before native scoring. Initial and later passes have different selections and
horizons, so their aggregate tables are shown separately in the dashboard.

CoTracker3’s warm calls took roughly **0.08–0.18 seconds per camera clip** in
these runs. A process’s first CoTracker call took about **4–5 seconds**, including
imports/model setup. Recorded sensor loading and query preparation added about
0.5–3.3 seconds in the earlier comparisons. These are single-run component
timings on sparse replay clips, not full robot episode latency. Capture/rendering,
grasp planning and IK are absent from this study.

Luna’s CoTracker follow-up sessions took **37–91 seconds**, with about **3–16
seconds** inside the numerical tool calls. The two final additional-recording
sessions took 53.3 and 69.3 seconds, with 11.4 and 7.8 seconds in tools. Session
wall time also contains inference, image inspection, shell commands and
transport; it is not pure reasoning time. Tool timers and their nested model
timers must not be added twice.

The practical choice is to keep **LK as an inexpensive option and CoTracker3
available when correspondence retention needs it**. A different mask or
reference can help more than a larger tracker. No automatic backend-policy
change was made.

The [ArtiPoint adapter](../src/vlm_tamp/artipoint_adapter.py) runs the authors’
unmodified factor-graph estimation core on the same measured 3D pairs, with
author defaults and `theta_prior=None`. It does not reproduce the full
human-hand segmentation, tracking, smoothing and data-selection pipeline.
The core passed a synthetic frame/unit check (about 0.062 mm line error), but
its raw estimates were unreliable on these sparse robot correspondences. A
separate broader-horizon diagnostic produced one useful 9.96 mm/4.0° result on
closing-state-49 LK tracks; other windowed results remained inaccurate.
Its raw proposal semantics also differ from our fitter’s admission gates.
These results do not justify replacing the current estimator, and are not a
verdict on the paper’s full pipeline.

## Reproducibility and future-agent notes

- [Initial registration](../results/luna-part-motion-tools-20261027/protocol.json)
- [CoTracker/feedback follow-up](../results/luna-part-motion-tools-20261027/cotracker-followup/protocol.json)
- [Arbitrary-reference follow-up](../results/luna-part-motion-tools-20261027/reference-followup/protocol.json)
- [Final independent-estimator follow-up](../results/luna-part-motion-tools-20261027/radius-prior-followup/protocol.json)
- [Frozen-source/input audit](../results/luna-part-motion-tools-20261027/audit.json)
- [Production dispatch replay](../results/luna-part-motion-tools-20261027/service-dispatch-integration.json)
- [Pinned upstream sources/checkpoint](../results/luna-part-motion-tools-20261027/upstream-sources.json)

The final focused suite passed **69 tests**. Source snapshots preserve each
version; old registrations must be checked against their snapshots, not against
the newer working tree. Actor sensors/tool files stayed unchanged, private
source/answers were denied by sandbox preflights, and backend comparisons used
the same actor-selected masks and query sets.

A [calibration arithmetic audit](../results/luna-part-motion-tools-20261027/calibration-audit.json)
checked 97 used frame/camera combinations. Supplied depth, world points and
calibration round-trip consistently. This checks numerical frame conventions;
it is not a physical calibration-accuracy or material-correspondence guarantee.

CoTracker source: `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` from
[the official repository](https://github.com/facebookresearch/co-tracker).
Its official `scaled_offline.pth` checksum is
`2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`.
ArtiPoint source: `ec88e43e178616697fbc948d0fdda48c0c880702` from
[the official repository](https://github.com/robot-learning-freiburg/artipoint).
GTSAM 4.2 is isolated under `.runtime/artipoint-deps`; the existing robot
environment was not upgraded. ArtiPoint’s eager package initializer initially
required unrelated full-pipeline dependencies. The adapter imports the unchanged
core/helper files under temporary namespaces instead; that repair and smoke
failure remain recorded.

The normal cuRobo v2 backend, 3 mm arc tolerance, controller, whole-trajectory
execution/native-success stop, grasp timing and wrist policies were not changed.
The previous planning/budgeting bottlenecks remain separate. A live use of
motion-based initialization still needs a feasible contact/probe that produces
informative motion; this replay study does not establish that capability.

Luna’s `findings.txt` files are retained as candidate procedural lessons with
tool IDs. They were not automatically promoted into the learner guide. Reuse
generic lessons about mask identity, reference choice, visibility and validation;
do not transfer per-recording coordinates or stale mask IDs to a new scene.
No cross-case learned-memory transfer or end-to-end manipulation improvement
was measured here.
