# cuRobo v2 migration and development tests — 22 October 2026

Latest: [production continuation and successful fresh replay](curobo_v2_continuation_20261022.md).

Latest: [3 mm positional tolerance and tested cuRobo continuation](curobo_v2_tolerance_and_continuation_20261022.md).

Follow-up: [failure diagnosis and verified native fix](curobo_v2_failure_diagnosis_20261022.md)
restored all five positive saved cases. The fresh-state GPU goal regression
remains unresolved; the original migration results below are retained unchanged.

Status: migration implemented and tests complete. Arc optimizer reuse was
corrected after the initial learner trials; one complete-path regression remains.
Results distinguish numerical replay, fixed
policy replay and adaptive practice. The user requested this migration while
deferring further work on grasp uncertainty.

[Open the profiling report](../results/curobo-v2-migration-20261022/report.html)
with VS Code Live Preview. The report reads retained receipts; it is not a robot
control interface.

## What changed

The shared robot service now defaults to cuRobo v2 for grasp IK and arc-goal
generation. It uses the calibrated controller TCP, and a separate fixed tool
frame at the finger-pad midpoint for arc goals. Point-mode orientation remains
a soft proposal preference. CPU equality correction, native robot/scene
admission, wrist limits and complete constrained path connections remain.

The expensive CPU nearest-neighbor clearance objective is replaced in v2 arc
goal generation and its intermediate-goal fallback. Native MC-VAMP still
connects the admitted configurations along the articulation constraint. This
is **not** a replacement of that constrained planner with cuRobo MotionPlanner,
and does not enable cuRobo torque/dynamics planning.

GraspGen, its candidate geometry, the inherited contact permissions, physical
servo and native LIBERO success predicate are unchanged. Different IK
solutions can nevertheless change the selected posture, path length and
physical outcome; identical grasp proposals do not imply identical execution.

Relevant code:

- [Process/environment selection](../src/vlm_tamp/rekep_libero/curobo_client.py)
- [v2 numerical worker](../src/vlm_tamp/rekep_libero/curobo_v2_worker.py)
- [Calibrated collision guidance](../src/vlm_tamp/rekep_libero/curobo_collision_model.py)
- [Arc goals and independent admission](../src/vlm_tamp/curobo_v2_arc.py)
- [Shared arc integration/profiling](../src/vlm_tamp/arc_execution.py)
- [Complete constrained search](../src/vlm_tamp/partial_pose_arc.py)

## Collision representation and scope

The worker receives only calibrated robot geometry and hash-verified observed
scene arrays. It uses upstream `Mesh.from_pointcloud` at 5 mm resolution and
GPU mesh SDF queries. Stationary and moving observations remain separate; the
moving mesh is transformed according to the configured hinge and requested
progress. This uses cuRobo v2 collision checking, **not** its raw-depth
TSDF/ESDF mapper. The latter remains a possible subsequent mapping change.

The final proposal proxy has 82 spheres: 34 movable arm spheres from the
existing CPU clearance model and 48 inscribed palm spheres. Fixed arm spheres
are omitted. Fingers and the exact configured convex robot geometry are still
checked by native admission, including the original bounded finger/target
contact permissions. The proxy is an optimization aid, not a collision
certificate. Two posture checks at each scene configuration verify the contact
frame and sphere transforms against independent CPU kinematics.

GPU flags alone cannot authorize a goal. Every candidate needs equality
correction, native projection, scene admission, requested progress and wrist
checks. A GPU-rejected candidate within 2 mm of the positional goal can undergo
correction and independent admission. More distant rejected proposals are
discarded. No calibrated uncertainty or attachment guarantee was added.

## Environment and compatibility

The [upstream v2 API](https://github.com/NVlabs/curobo/tree/v0.8.0) is a rewrite.
The installation is pinned to release `0.8.0`, revision
`4ea77366ca48ee453e7df139e39fa6532af49f3b`, in
`.runtime/curobo-v0.8.0`. Its environment is `.runtime/envs/curobo-v2`, separate
from the retained v1, simulator and GraspGen environments.

Validated core versions: Python 3.10.22, Torch 2.7.1+cu128, Warp 1.18.0,
cuda-core 1.2.2, NumPy 1.26.4 and SciPy 1.14.1. Physics and Luna trials used
server GPU 1; the final numerical sequence and history probes used GPU 2.
Both are RTX PRO 6000 Blackwell cards with approximately 96 GB memory.

[Setup script](../scripts/setup_curobo_v2.sh),
[core dependency pins](../requirements-curobo-v2.txt) and
[full installed dependency receipt](../results/curobo-v2-migration-20261022/environment-freeze.txt).
The source checkout is unmodified. Two adapter details matter:

1. The pinned mesh implementation calls `wp.torch.device_from_torch`; Warp 1.18
   exposes that function directly. A process-local alias bridges the namespace.
2. The v0.8.0 factory's self-collision flag disables optimizer terms but leaves
   the metrics configuration enabled. The adapter makes both consistent with
   the intended setting. Native VAMP still performs the authoritative robot
   self-collision check, as before.

`JointState.from_position` also initializes the velocity/acceleration arrays
needed by the new API. The first failing smoke test and its corrected rerun
are retained. First-use kernel compilation can take tens of seconds; warm
throughput must be distinguished from cold setup. This server's validation
runs have populated those caches.

Arc optimization now uses fresh solver state for each request while retaining
the CPU surface cache. The original reuse implementation recovered four of
five positive cases in isolation but only one when mixed with the failed
queries. Seed reset alone recovered two. Fresh solver state restored four in
the full mixed sequence. This is a measured workaround for our partial-pose
integration, not a claim of a general upstream cuRobo defect or universal
bitwise determinism. It costs additional initialization/graph capture time,
included in the worker solve-call timings.

## Numerical validation

The corrected [full ten-case sequence](../results/curobo-v2-migration-20261022/saved-arcs006-query-state/results.json)
recovered four of five originally successful cases, and did not solve the five
original failures: 4/10 complete numerical paths versus the archived 5/10.
The [original reused-state sequence](../results/curobo-v2-migration-20261022/saved-arcs004/results.json),
[seed-reset-only experiment](../results/curobo-v2-migration-20261022/saved-arcs005-seed-reset/results.json)
and [controlled reuse diagnostic](../results/curobo-v2-migration-20261022/reuse-diagnostic002/results.json)
are preserved, including their negative results.

[Basic GPU smoke](../results/curobo-v2-migration-20261022/basic-smoke002/result.json)
passed 25 FK comparisons and eight reachable-target IK queries. Maximum TCP
position discrepancy was 0.185 micrometres; orientation discrepancy was
0.000000399 radians. Warm basic IK calls took approximately 3.5–3.9 ms in the
worker, or 11–14 ms including IPC and CPU filtering. This is not the cost of
collision-aware arc planning.

The corrected mixed-case replay used identical retained robot state and
observed scene arrays, with native path checking and no simulator execution:

| Saved event | Original search, s | v2 wall, s | v2 complete path |
|---|---:|---:|---|
| 7 | 13.477 | 7.801 | Solved |
| 10 | 13.867 | 22.248 | Failed: search budget exhausted |
| 13 | 13.608 | 4.296 | Solved |
| 18 | 13.746 | 5.004 | Solved |
| 21 | 14.323 | 3.749 | Solved |

[Results and inputs](../results/curobo-v2-migration-20261022/saved-arcs006-query-state/results.json).
The original column is search time; v2 wall includes scene setup and final
sampled validation, so they are not exactly identical timer boundaries.
Four of five previously successful queries were recovered. The remaining
failure has admitted endpoints but unsuccessful complete connections; speed
alone does not resolve this. Earlier adapter iterations and failures are kept
in `saved-arcs001` and `saved-arcs002`.

[Regression tests](../results/curobo-v2-migration-20261022/tests-final-query-state.log):
69 passed, 5 skipped because older optional scene fixtures are unavailable.
Two stale test fixtures were repaired: robot-frame tests now use a portable
robot-only JSON fixture, and the planning-feedback fake service includes the
specification/epoch required by the earlier compact-interface change.

## Fresh-physics fixed-policy replay

[Replay receipt](../results/curobo-v2-migration-20261022/physical-replay001/summary.json)
uses the first archived Luna decision sequence with fresh observations, grasp
selection and plans. It never executes the old joint trajectories.

The grasp and first 30-degree pull executed. The subsequent 30-degree request
failed to produce a complete path, so the fixed sequence could not continue.
Native success was false at 316 controls. Attempt time was 122.56 seconds,
126.59 seconds including cleanup. All three issued trajectories completed,
with zero execution vetoes; source integrity passed.

This replay is a diagnostic of a fixed decision sequence, not a normal adaptive
Luna trial terminated early. It has no recovery branch. The changed posture and
path also consumed more controls: approach 175 versus the archived 149, first
pull 41 versus 14. The same old action sequence is therefore not a reliable
drop-in policy under the changed numerical proposals.

The [corrected-backend replay](../results/curobo-v2-migration-20261022/physical-replay002-query-state/summary.json)
also stopped at the unavailable second 30-degree plan, at the same 316 controls.
Its attempt took 154.57 seconds, 158.76 including cleanup. Grasp planning took
72.57 seconds because the corrected solver state also increases the cost of
unsuccessful contact probes. The first executed arc planned in 3.37 seconds;
the failed next request took 21.43 seconds. Native success remained false, all
issued trajectories completed, and source integrity passed. Thus this migration
has not demonstrated an end-to-end reliability improvement.

## Luna practice

Two continuous Luna/high attempts completed with 600 control steps and
1,800 seconds per attempt, and 4,200 seconds for the session. They use the same
earlier Luna-authored guide as the compact-interface study, with no separate
reflection reviewer and no new hand-authored recovery hints.

[Registration](../results/curobo-v2-migration-20261022/luna-practice001-registration.json),
[exact prompt](../results/curobo-v2-migration-20261022/luna-practice001/agent-prompt.txt),
[process log](../results/curobo-v2-migration-20261022/luna-practice001/agent-events.jsonl),
[robot input/output log](../results/curobo-v2-migration-20261022/luna-practice001/tool-events.jsonl).

Both failed at the 600-control limit, taking 1,030.37 and 618.34 seconds per
attempt. They ran **before** the optimizer-lifetime correction. Luna tried
smaller arcs, fresh observations, alternate contact strategies and a measured
pivot hypothesis; it explicitly reported that no successful method was
established. The corrected backend has not received another adaptive Luna
practice study in this migration experiment.

The session lasted 1,757.01 seconds: 862.56 in robot service calls and 894.45
outside them. The residual includes inference, local analysis, transport and
scheduling; pure reasoning time is not separately available. All 506 registered
source hashes matched at trial completion and API ledgers were unchanged.
Subsequent corrective code edits are distinguished from the preserved
`source-snapshot` used for these attempts.

[Learner summary](../results/curobo-v2-migration-20261022/luna-practice001/agent-workspace/practice-summary.json),
[candidate guidelines](../results/curobo-v2-migration-20261022/luna-practice001/agent-workspace/guidelines.json),
[audit](../results/curobo-v2-migration-20261022/luna-practice001/audit.json) and
[evaluation scope](../results/curobo-v2-migration-20261022/luna-practice001/evaluation-scope.json).
The candidate memory was not promoted and the earlier default guide is unchanged.

## Lessons for the next agent

Validate the original mixed sequence as well as isolated successful queries.
Failed requests and mode changes can affect a reused numerical solver; record
solver lifetime and sampling policy in the experiment. Seed reset alone was
insufficient in this integration. Preserve both the pre-correction trials and
the final implementation so their outcomes are not conflated.

The remaining concrete problem is endpoint/path quality: scene-admitted goals
can still be difficult to connect, and changed postures can consume more
controls. A logical next comparison is to rank proposals by complete-path cost
and connection feasibility. Grasp uncertainty, raw-depth fusion and broader
memory/learning changes remain separate questions; this experiment does not
establish a limitation of Luna's reasoning ability.

## Running and reverting

The service, pilot runner and compact-practice runner accept
`--numerical-backend curobo_v2` (the new default) or `--numerical-backend legacy`.
Legacy selects the preserved cuRobo v1 grasp IK and CPU arc goals. Neither
choice changes the grasp generator or contact permissions.

```sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .runtime/envs/vlm-tamp/bin/python scripts/check_curobo_v2.py \
  --output results/my-v2-kinematics-check

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .runtime/envs/vlm-tamp/bin/python scripts/benchmark_curobo_v2.py \
  --output results/my-v2-saved-arc-check
```

Use a new output directory each time. GPU commands need the server execution
context with device access. The normal Codex filesystem sandbox cannot access
the driver. Original evidence and pre-migration source copies remain under
`results/curobo-v2-migration-20261022/before`.
