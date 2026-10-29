# Integrated cuRobo continuation and fresh simulation replay

Follow-up: [adaptive Luna practice succeeded on all five attempts](luna_curobo_continuation_practice_20261022.md),
with learner-authored candidate memory retained separately from the default guide.

The user authorized integrating the successful local-continuation prototype
and testing it with fresh physics. Local continuation is now the default
cuRobo arc strategy, with a bounded retained global IK fallback. The 3 mm
positional tolerance remains active. No Luna prompt or memory changes are part
of this integration.

**Result: native LIBERO success in the fresh replay, at control step 289.**
Reset-to-success time was 209.812 seconds (215.982 including cleanup). The
second pull that previously failed now executes. This is one successful
fixed-decision development replay, not an adaptive learner success-rate study.

[Open the observer report](../results/curobo-v2-continuation-20261022/report.html).
[Prior prototype and tolerance work](curobo_v2_tolerance_and_continuation_20261022.md).

## Production behavior

An arc is planned through targets at most 3 degrees apart. All targets use the
original estimated hinge circle and original wrist reference. Each IK query
starts from the preceding admitted posture, supplies all seeds locally, and
disables global LM reseeding. Assigned local time comes from the requested arc
speed (default 15 degrees/second), so cuRobo's local velocity-related posture
cost refers to the current posture. Moving observations use a 2 mm mesh and
stationary observations use 5 mm; transported contact references remain
association data in this local mode.

The optimizer and local random sampler are fresh for every configured arc.
Steps within that arc share them. Arc IDs reject stale, missing, and closed
requests; normal completion and errors close the scope. CPU mesh caching can
survive across requests, but optimizer history cannot.

Native MC-VAMP connects and validates each edge. Every returned sample is
checked for the arc constraint and collision, and the whole-arc wrist limit
applies to each edge. A failure returns **no executable partial trajectory**.
The first returned state comes from the native validated path, including its
existing small start-projection policy, rather than reintroducing an unchecked
measured start.

Local continuation alone lost one old positive case: event 21 reached 2.5
degrees in planning, then found 21 near-target candidates colliding with the
palm and 43 outside the polishing capture range. The previous global IK branch
still solves this complete 5-degree request. The integrated planner therefore
retains a global fallback when local continuation stalls and at least three
seconds remain. It tries at most three global endpoints, with each connection
capped at 1.5 seconds and the remaining overall budget. It replans from the
original query start and cannot execute the local partial path.

This fallback intentionally retains the earlier global proposal formulation,
including its zero-joint prior and its original observed-surface mesh setup.
It is not described as a current-posture regularizer. Both proposal branches
use the same native final constraints, wrist limits, collision checks, and
20-second planning budget. Neither branch changes physical contact permissions.
Results identify the branch, scope ID, setup time, candidate failures, connection
failures, and whether fallback supplied the final path.

## Numerical validation

- **41 focused tests passed**, with three skips for unavailable old fixtures.
  These cover full-path admission, rejecting partial progress, scope cleanup,
  stale IDs, local seeds, and fallback validation.
- GPU lifecycle regression uses one persistent worker: A, an unrelated failed
  request B, then A again. All 23 checks pass; admitted endpoints and raw GPU
  solutions reproduce with zero measured joint difference. The second step
  within A reuses its optimizer; B and the repeated A get fresh generations.
- Final mixed saved-case run restores **all five old positives**. Their wall
  times are 18.732, 17.096, 7.135, 4.744, and 10.410 seconds. Event 21 uses the
  bounded global fallback; the other four use local continuation. All five old
  failed requests remain failures with recorded reasons.
- The previously failing fresh-state event 10 now solves through production
  continuation in 17.910 seconds. This is a saved-state check, not a physical
  success result.

Evidence: [tests](../results/curobo-v2-continuation-20261022/tests-final.log),
[GPU reset/reuse regression on final sources](../results/curobo-v2-continuation-20261022/lifecycle003/results.json),
[final mixed run](../results/curobo-v2-continuation-20261022/saved-arcs003/results.json),
[previously failed state](../results/curobo-v2-continuation-20261022/failed-state001/results.json),
[integrated late fallback](../results/curobo-v2-continuation-20261022/late-fallback001/results.json).

The first mixed integration run caught a path-assembly bug before simulation:
event 18's measured start was collision-free but slightly outside the jaw-angle
constraint. Native planning corrected it by at most 0.00244 radians in any
joint, under its existing 0.01-radian start gate. The assembly incorrectly kept
the raw start instead of the native first sample. The correction and a
regression test preserve the validated first sample; subsequent mixed testing
passes event 18. Original failed runs remain archived.

## Fresh physics

The retained decision sequence comes from the successful compact Luna practice
run on LIBERO-90 microwave opening, state 45, seed 0. Replay resets physics and
RGB-D, regenerates/selects grasps, replans every requested trajectory, and uses
the native success predicate. Historical joint trajectories are not executed.
This is a development engineering replay, not adaptive Luna learning or a
held-out evaluation.

`physical-replay001` stopped at grasp planning because the controller was
launched with the simulator Python environment, which lacks `omegaconf` needed
by GraspGen. Only 60 reset/settling controls occurred; zero trajectories were
issued. This is an environment-launch failure, not a cuRobo planning result.

`physical-replay002` uses the established controller environment and succeeds.
Its four executed pulls request 30, 30, 15, and 5 degrees, all planned by local
continuation. Native success occurs during the last pull, so that final
trajectory stops immediately. Five issued command batches complete and one
terminates on success; no controls follow success. The run uses zero new API
calls and zero execution vetoes. Grasp selection benefits from the new arc
admission and selects a different posture; this fresh replay is not a frozen
grasp comparison. The separate saved-state experiment isolates the planner
change from that selection effect.

The larger rejected requests are retained in the original decision tape:

| Event / request | Local continuation failure | Global fallback failure |
| --- | --- | --- |
| 12 / 30-degree point arc | At 18 degrees, 23 candidates collide and 41 miss the position capture range | All 64 miss position; best error 6.60 mm |
| 15 / 15-degree point arc | At 9 degrees, four candidates collide and 60 miss position | Two candidates fail wrist allowance, 62 miss position |
| 16 / 15-degree jaw-axis arc | At 12 degrees, 17 candidates collide and 47 miss position | All 64 miss position; best error 14.22 mm |
| 17 / 15-degree rigid arc | At 9 degrees, all 64 miss position | All 64 miss position; best error 24.09 mm |

The local angles above are internal planning progress, never executed partial
motions. These finite search results do not prove the requests impossible.
The later 15- and 5-degree point requests produce complete paths and finish
the task. Detailed per-candidate robot parts, surface roles, and wrist values
are in the raw query logs linked from the report.

[Replay summary](../results/curobo-v2-continuation-20261022/physical-replay002/summary.json),
[complete video](../results/curobo-v2-continuation-20261022/physical-replay002/attempt01/initial/rollout.mp4),
[arc outcomes](../results/curobo-v2-continuation-20261022/physical-arc-outcomes.json),
[independent execution audit](../results/curobo-v2-continuation-20261022/physical-audit.json).

The audit confirms 504 source files match the executed snapshot and all 229
joint references match what reached the driver (plus 60 initial reset/settling
controls). The existing driver caps each joint delta relative to measured
position at 0.05 radians. That limiter activates on 101 controls, mostly the
approach; its maximum correction is 0.01625 radians. Controller goals match
that documented bound exactly. Do not confuse unchanged issued references
with exact physical tracking or unbounded controller goals. This driver was
not changed by the continuation work.

## Reproduction and environment ownership

The controller needs the GraspGen dependencies in `rekep-libero`. The simulator
runs separately in `vlm-tamp`, selected by `VLM_TAMP_SIM_PYTHON`. cuRobo runs in
its own `curobo-v2` worker environment. Offline arc benchmarks can run in
`vlm-tamp` because they do not instantiate GraspGen.

```bash
source scripts/server_runtime_env.sh
CUDA_VISIBLE_DEVICES=1 .runtime/envs/rekep-libero/bin/python \
  scripts/run_curobo_v2_policy_replay.py --output results/NEW-OUTPUT-DIRECTORY
```

GPU regression and benchmark commands use a new output directory:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .runtime/envs/vlm-tamp/bin/python scripts/check_curobo_arc_lifecycle.py \
  --output results/NEW-LIFECYCLE-DIRECTORY

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .runtime/envs/vlm-tamp/bin/python scripts/benchmark_curobo_v2.py \
  --output results/NEW-SAVED-CASE-DIRECTORY
```

Production code: [arc planner and admission](../src/vlm_tamp/curobo_v2_arc.py),
[worker lifecycle and local solver](../src/vlm_tamp/rekep_libero/curobo_v2_worker.py),
[shared defaults](../src/vlm_tamp/hinged_contact_tools.py),
[full-arc dispatch](../src/vlm_tamp/partial_pose_arc.py),
[execution boundary](../src/vlm_tamp/arc_execution.py),
[actor-visible failure summary](../src/vlm_tamp/planning_feedback.py).
