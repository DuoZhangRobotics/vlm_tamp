# Small GraspGen experiment and removal of duplicate path validation

The active approach now uses eight native GraspGen poses, the closest robot-valid
returned cuRobo IK solution for each, and one VAMP path query per examined pose.
It stops at the first executable approach. There is no geometric fallback,
height relocation, angle menu or pad-depth sweep in this route. The old hybrid
selector remains available only through explicit `historical_hybrid`
configuration for comparisons; it is never invoked automatically.

At the user's explicit request, the MC-VAMP wrapper now returns the upstream
planner's checked path without its additional 0.006-radian collision/projection
pass. This applies to both RRTC free paths and CRRTC constrained paths. The
existing time scheduler interpolates along those returned segments. Bounds,
robot calibration, upstream collision checks and controller speed limits remain.
No extra collision padding was introduced. The native library was rebuilt.

**The speed result is positive; reliable grasp acquisition is not established.**
All three direct physical tests closed empty. Do not promote these timings as a
task-success improvement, or describe this as a fresh Luna evaluation.

## Meaning of the previous terms

- Frontal: approach along the negative observed panel normal, with jaw alignment
  based on the upright-panel prior. It was our geometric seed family.
- Angled/oblique: prescribed yaw/tilt variants around that approach, including
  yaw ±30/60 degrees and smaller tilt variants. These were not neural predictions.
- Contact depth: shift the entire TCP along the approach direction so the
  selected bar occupies a different part of the pad face. The old fractions
  were 0.5, 0.8 and 1.0. This was not a penetration allowance or gripper width.
- Frame conversion: still necessary. The trained Panda fingers attach at
  58.4 mm; our configured robot attaches them at 52.4 mm. A fixed +6 mm local-z
  asset correction, followed by the calibrated 97 mm base-to-TCP transform,
  preserves the predicted contact placement. This is a single robot-only
  calibration, not a menu or a scene-specific adjustment.

## What upstream systems actually do

There is no universal candidate count or exclusively learned workflow.

- [GraspGen](https://github.com/NVlabs/GraspGen) predicts complete 6D poses and
  discriminator scores. Its official examples use 200 diffusion samples and
  optionally top-100 ranking. Those numbers were upstream defaults; they were
  not invented in this project. Their cost should be measured separately from
  downstream IK and route search. Our eight-sample choice is an experiment.
- [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) predicts grasps
  from scene clouds, and recommends segmentation/cropping and filtering contacts
  to the intended object. Learned pose generation is combined with geometric
  filtering, reachability and subsequent motion planning.
- [GPD](https://github.com/atenpas/gpd) samples candidates and classifies them.
  [MoveIt Grasps](https://github.com/moveit/moveit_grasps) deliberately generates
  geometric approaches for simple shapes and filters with IK. Rule-based
  sampling is an established method.
- Current GraspGen also includes **GraspMoE**, combining diffusion and geometric
  OBB proposals and scoring the union with its discriminator. The
  [official collision-free demo](https://github.com/NVlabs/GraspGen/blob/main/scripts/demo_collision_free_grasps.py)
  then checks scene-cloud collisions. This is different from changing the
  translation/depth of a scored pose without re-evaluating its learned score.
- [MoveIt Task Constructor](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html)
  separates approach, grasp, closure and subsequent motion. A pregrasp approach
  can be a deliberate motion stage; it does not require changing the learned
  final grasp into a grid of new poses.
- The [cuRobo report](https://curobo.org/reports/curobo_report.pdf) uses hundreds
  of parallel IK seeds in experiments and discusses checking interpolated
  trajectories. Internal GPU multistart is normal. Our former 1,125 queued
  grasp postures and repeated serial route alternatives were a separate cost.

No controlled ablation demonstrating a need for this project's always-on
frontal/oblique/depth expansion was found. The old hybrid results do not isolate
the reliability of original GraspGen poses, since translations were replaced.

## Measurements

Main output: `results/small-graspgen-20261029-planner-path/watch.html`.
The existing clean600 and live-hinge watch pages link to it. Use VS Code Live
Preview; no new browser port or server is required.

| Case | Historical grasp call | Small grasp call | Model load inside call | Path call | Extra validation |
|---|---:|---:|---:|---:|---:|
| open37 | 55.90 s | 16.50 s | 10.54 s | 0.125 s | 0 |
| open13 | 54.52 s | 5.85 s | 0 | 0.141 s | 0 |
| close49 | 86.25 s | 4.96 s | 0 | 0.141 s | 0 |

Each case used one IK request and one path request. Each had eight original
learned poses available, no geometric substitutions and no fallback. Native
RRTC solving took approximately 4–5 ms; the path-call column also includes
wrapper/worker overhead. GPU inference for the warm cases was approximately
0.16 s; the roughly 0.80 s generation stage also includes saving/checkpoint
provenance. IK includes first-query worker setup (1.9–3.0 s).

One model instance stayed resident across the three fresh simulator episodes.
This does not yet make GraspGen persist across separately launched robot-service
processes. Those still load their own model. Historical episodes each loaded
fresh models, so the full-call comparison includes residency and cold-start
differences, as well as the smaller search. Do not attribute the whole gain to
removing dense validation. The initial small-grasp run, before that removal,
is preserved under `results/small-graspgen-20261029-v1`.

Initial sensor arrays matched the historical sources exactly. Luna's recorded
public region selections were reused. Network CUDA sampling was not bitwise
identical between the first two runs despite the same seed. The final
scene-policy diagnostic explicitly reused the archived generated-pose file and
verified its identical SHA256. Source-integrity checks passed in the final
physical run and that matched diagnostic.

## Physical failures, separately from returned plans

The test executed approach, 40 closure controls, and a requested 10-degree
probe. It did not attempt a full episode recovery or provide new hints to Luna.

- open37: actual approach endpoint was **89.2 mm** from the planned endpoint.
  Native contact logs show palm/microwave contact, peaking around 224 N in this
  simulator. The approach pushed the door; the fingers then closed empty.
- open13: **88.6 mm** endpoint error, palm/microwave contact around 219 N,
  followed by empty closure.
- close49: **76.9 mm** endpoint error, arm/mug and finger/microwave contacts,
  followed by empty closure. The door opened further overall.

All final pad gaps were effectively zero. All native task-success flags were
false. Solver/API completion is not grasp success. Simulator object angles
and contact identities were read only after the runs for this diagnostic;
they were not available to planning or an actor.

The default simulation environment policy remains advisory, as previously
requested. It checks the robot but does not veto predicted scene collisions.
This explains why a returned approach can be physically blocked by scene
contacts. The learned score considers the selected handle cloud; a high score
does not certify whole-arm or surrounding-scene compatibility.

The explicit `veto` diagnostic at
`results/small-graspgen-20261029-scene-frozen` reused **the exact same eight
poses** for open37. It took 6.72 s including cache lookup, scene preparation
and eight IK requests. Two poses had no returned robot-valid IK. Six were
rejected at their selected endpoints: one finger-body/contact-patch prediction,
three palm/stationary-scene predictions, and two arm/stationary-scene
predictions. **Zero path queries** were needed. These are predictions under the
current collision representation, not a proof that every pose/IK branch is
physically impossible. This test did not change the default advisory policy.

An earlier diagnostic requested a disposable cache that finish_attempt had
already cleared, so it regenerated poses. It is retained at
`results/small-graspgen-20261029-scene-query`; do not call it a frozen-candidate
comparison. The driver now restores immutable archived generation folders.

The remaining question is whether a small set of learned poses can provide a
scene-compatible reachable grasp with the calibrated robot. A useful next
experiment should inspect those predicted collisions and approach directions,
then test standard scene filtering/approach stages. It should not restore the
old combinatorial pose menu or dense validation automatically. Eight failed or
rejected candidates do not establish that GraspGen itself is unreliable.

## Validation and code

86 focused tests passed; 3 optional tests skipped. The native adapter rebuilt,
all three physical cases completed, and the HTML rendered in headless Chrome.
The tests cover fixed frame conversion, first-feasible selection without pose
expansion, sampling counts in cache identity, budgets, contacts, scheduling,
schemas and compact presentation.

- `src/vlm_tamp/graspgen_selection.py`: small proposal selection.
- `src/vlm_tamp/rekep_libero/perception.py`: configurable one-pass sampling.
- `src/vlm_tamp/coupled_resources.py`: sample-count-aware cache and model reuse.
- `src/vlm_tamp/native/mcvamp_contact_numerical.cc`: direct planner path return.
- `scripts/test_small_graspgen.py`: matched public-input component experiment.
- `scripts/report_small_graspgen.py`: timing, physical contacts and video report.

The pre-existing handoff tarball remains untracked. Raw results and runtime
models are not committed.
