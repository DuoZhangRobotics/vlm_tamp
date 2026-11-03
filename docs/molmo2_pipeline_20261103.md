# Molmo2 in the Luna robot pipeline

Implementation commit: `8ff8d96` on `codex/grasp-consistent-long-horizon`.
The earlier recorded-image diagnostic is in `molmo2_pointing_20261103.md`.

## Interface and division of work

Luna can call `point(observation_id, camera, query)`. The tool reads the two saved RGB images
(`agentview`, `wrist`) from that observation. Luna writes the target description and chooses the
marking camera. Molmo2 receives no depth, simulator geometry, previous answers or successful
coordinates. The operation neither captures a new scene nor changes the robot/planner state.

The model request is deliberately short:

```text
Point to: <Luna's exact query>
Image 1 is agentview. Image 2 is wrist. Only point in image <requested index>.
```

Replies retain the native response, image index, raw coordinates, converted floating-point pixels,
nearest-integer `contact_pixel_xy`, observation/camera IDs, annotated images and timing. A mark in
the wrong camera keeps its actual camera and a mismatch flag. Out-of-image points are not clipped;
their raw coordinates remain available. No coordinate is substituted when the model returns none.

The point can seed `segment_parts`. The existing selected mask/cloud then supplies grasp planning
or moving-part tracking. Targeted push/hook tools still require a selected region or current part
ID in addition to the contact pixel. GraspGen generates grasp poses; Molmo2 supplies no grasp pose,
joint model, force direction, reachability certificate or engagement evidence. Window marks remain
allowed image-space selections. Raw-depth limitations are a separate question.

The actual operation registry, compact help, workspace schema export, service dispatch and compact
response all expose this contract. A separate process holds Molmo2 weights for the study. Robot
services connect through one local JSON socket; each query has fresh model context. The worker
uses the existing isolated Molmo2 environment and its packaged CUDA 13 runtime, separately from
the simulator/planner CUDA 12.9 environment. The socket read timeout is 600 seconds; there is no
automatic retry, worker replacement, alternate model or device fallback.

Main files:

- `src/vlm_tamp/pointing_worker.py`: persistent model and local transport.
- `src/vlm_tamp/pointing_tools.py`: two-camera request, coordinate conversion and public evidence.
- `src/vlm_tamp/perception_tools.py`: shared perception entry point.
- `src/vlm_tamp/robot_contracts.py`: contract v21 and callable schema.
- `scripts/run_molmo2_luna_study.py`: worker lifecycle and fresh-actor evaluation batch.
- `configs/molmo2_luna_eval_20261103.json`: closing 17, closing 49 and opening 13.

Enable the feature with an explicit worker endpoint (`--pointing-socket` on the pilot/service, or
`VLM_TAMP_POINTING_SOCKET` for the study driver). Without it, the operation is not enabled in the
actor's exported help/schema. The shared worker stays alive across episode resets; stopping a
robot episode does not unload it. The batch owner shuts it down when the batch ends.

## Integration problems found and corrected

Two diagnostic launches are preserved outside the final evaluation counts:

1. `results/luna-molmo2-integration-20261103`: an overlong wrapper produced “no visible door”
   responses. Paired calls on identical images showed that the previously tested prompt still
   returned a point. The extra wrapper wording was removed. The launch stopped at 60 controls.
2. `results/luna-molmo2-integration-v2-20261103`: Luna wrote “Mark a point ...”; passing that as
   free-form chat produced Asymptote drawing code instead of coordinates. Paired calls using the
   exact Luna query showed that the native `Point to:` instruction returned a coordinate tag.
   This launch stopped at 129 controls. Its 6.45-second recorded video is retained.

These fixes change the model's instruction format, not the requested task or numerical geometry.
The word-for-word Luna request and complete composed prompt are both saved. Structured output
does not guarantee that the model selects the requested object, feature or side.

## Evaluation and inspection

The frozen final study is `results/luna-molmo2-integration-v3-20261103/watch.html`. It uses fresh
Luna/high agents, no inherited notebooks or worked examples, two native cameras, 600 controls
and 1,800 seconds per episode. The existing controller, IK, planners and collision policy remain
unchanged. `watch.html` adds a Luna-to-Molmo2 section with each query, point, overlay, actual model
input/output and timing. Perception/generation timings are nested within the `point` operation,
not additional disjoint execution time. Updates remain manual and publication remains throttled.

The first opening run was interrupted at 134 controls. At 2026-11-03 14:01 EST, the service journal
records SIGTERM and permission denials while saving supervisor/GPU status; Luna also logged
permission denials while recording its session. Why write access changed is not established.
The final resource samples show zero OOM kills, no memory-limit hit and no I/O-pressure event.
The interrupted run and its 6.7-second video are preserved and excluded from task outcomes.
An explicitly recorded fresh opening replacement uses the same frozen sources and limits.
Its model worker was explicitly restarted; do not claim one uninterrupted worker for all trials.
Earlier monitor records are in `resources-before-permission-interruption/`.

The observer report `failure-summary.html` and machine-readable `integration-summary.json` contain
task outcomes, actual selected primitives, model calls, exact later pixel matches and native
door-angle timelines. Native angles/contact forces in that report are evaluator-only diagnostics;
they are never delivered to Luna, Molmo2 or planning. Exact pixel matching is evidence of a later
request, not proof that the model caused an action.

## Validation

All 96 relevant tests and Ruff passed on the final code. Tests cover camera indexing, coordinate
scaling, retained raw points, no-point output, observation provenance, actual schema/help export,
point-to-SAM/depth handoff, existing push/hook request compatibility and report timing export.
Grasp/hook compatibility tests exercise the connection, not physical grasp/hook success. The real
preflight queries include grasp and hook descriptions; one hook query selected a mug feature,
demonstrating that semantic correctness remains an open issue.

Browser validation opened the live model records and both point overlays and verified that the
composed `Point to:` prompt is visible. Completed and interrupted episodes have recorded videos.
Physical grasp/hook coverage must be reported from what Luna actually selected, separately from
tool availability and interface tests.

## Completed results

All three completed episodes passed the frozen-source/native-binary integrity check (631 source
files). The interrupted opening run is excluded and its replacement is explicitly recorded in
the study manifest. These are single fresh trials, not a matched success-rate improvement study.

| Task / state | Native success | Controls | Door opening: initial → final | Actor wall time | Robot tools | Molmo2 calls / time |
| --- | --- | --- | --- | --- | --- | --- |
| Close / 17 | No | 600 | 88.92° → 74.93° | 491.98 s | 98.57 s | 2 / 3.18 s |
| Close / 49 | No | 600 | 107.26° → 122.51° | 594.98 s | 89.48 s | 2 / 3.75 s |
| Open / 13 | Yes | 584 | 0.28° → 74.74° | 1115.58 s | 130.44 s | 5 / 7.47 s |

All nine actor-authored requests returned native coordinate tags: 14.40 seconds total, median
1.46 seconds per call including transport and artifact work. This is slower than the short-query
recorded-image median of 0.79 seconds; real queries and their echoed output labels are longer.
The outside-tool time remains Luna calls, transport and local analysis, not isolated reasoning.

Closing 17 reached its best 57.54° opening at control 276. A later 10 cm +X hand stroke did not
move the door, reacquisition reopened it, and the final -X strokes opened it further. Its first
approach reported a measured 5.9 mm arrival error. Closing 49's first push opened the door to
120.3°; later inward-normal pushes kept pressing toward the fully open stop. Evaluator-only
contacts show sustained finger-pad contact (about 79 N total normal force at the end), while the
hand could not reach commanded references. A later approach ended 30.1 mm short. The actual
executed requests, control intervals and measured door changes are in the observer report.
Neither closing episode exactly reused a Molmo2 point in later contact/segmentation arguments.

The opening episode reused Molmo2's `[180,255]` handle pixel in segmentation and grounding,
planned a GraspGen approach and executed grasped arcs. A subsequent rigid 20° arc failed local
IK admission at 8.6° with 64 cuRobo failures; this does not prove global infeasibility. Luna
explicitly requested point attachment next, then released/repositioned and finished with a push.
The native opening predicate is `qpos < -1.3`, about 74.48°; success was first observed at control
584 and execution stopped immediately. The wrist query had also marked two robot-finger
locations incorrectly. Thus valid output format is not proof of correct semantic grounding.

No actor selected `hook_pull`. Grasp/pull execution is demonstrated in the opening episode;
physical hook engagement is not newly validated. The shared point-to-mask/contact interfaces
and hook request compatibility are covered by the interface tests, not by an invented hook win.

The opening actor initially passed unsupported camera/part fields directly to `plan_grasp`,
then corrected the call and grounded the scene. After evaluation, the pointing help was clarified
to state that `ground` installs the contact/model and `plan_grasp` reads that installed scene.
This is a documentation clarification; the physical results above used commit `8ff8d96` and do
not claim a rerun with that later help wording.

Across the two monitored final-study phases, peak job host memory was **8.53 GiB**, peak GPU
memory **22.68 GiB**, all **164/164 SSH probes succeeded**, and there were **zero OOM kills**.
The permission/SIGTERM interruption is retained separately and is not called a memory failure.
After completion the model/robot workers exited and GPU 0 returned to approximately 2 MiB used.
At the final headroom check, the server had about 108 GiB RAM available.

The results support a usable, fast pointing connection and one successful grasp/pull/push
episode. They do not establish reliable closing or a general model advantage. The next closing
diagnostic is the relation between the selected face, intended closed configuration and commanded
motion, followed by contact loss/reacquisition. Repeating an inward-normal push is not evidence
that it moves toward the task goal.
