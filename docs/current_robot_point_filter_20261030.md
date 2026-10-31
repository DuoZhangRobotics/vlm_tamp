# Measured current robot occupancy is authoritative for planning

The user requested that observed points overlapping an existing simulator robot
state be removed before all downstream planning, without adding collision buffers
or another current-state rejection gate.

The motivating case is closing49, event 36/control 406 in
`results/luna-contact-repair-rerun-20261030/trials/baseline-r1-close_microwave-state049-600-launch-r2`.
The planner reported two stationary-cloud points inside the palm by 0.045 and
0.164 mm. An independent evaluator reconstruction found a **positive 0.581 mm
native palm/handle separation**, with identical palm mesh vertices and matching
robot joints and target hinge frame. The depth samples lay 0.816–0.843 mm outside
the native handle capsule. This was a false current-start rejection from a
representation mismatch. The native object geometry used to diagnose it is not
an input to Luna or the runtime filter.

## Implementation

`src/vlm_tamp/current_robot_scene.py` subtracts the configured robot's exact
current convex/sphere volumes from collision clouds. It has zero padding and no
overlap-depth/count rejection threshold. A boundary point is inside the closed
robot volume. Points outside it are unchanged.

The helper returns a planning copy and rebuilds its spatial indices. Canonical
observations remain available for perception and for the next measured state;
filtering does not accumulate holes along an imagined or historical robot path.
Hypothetical continuations transport the already-cleaned moving points. They
never clear additional points against a proposed pose or aperture.

Active integration points:

- Free pose and joint plans: `RobotService._query_scene`.
- Arc and post-closure plans: `sequence_execution.measured_robot_scene`.
- Grasp search: target points and collision geometry are filtered against the
  actual aperture before constructing a hypothetical open-hand approach.
- Push and hook-pull: fresh planning copies at contact search, insertion,
  each physical stroke boundary and withdrawal. Tracking uses the canonical
  measured geometry.

Static, moving and inferred obstacle clouds are filtered. Contact references
remain association metadata; they are not extra obstacle geometry. cuRobo global
fallback formerly appended those references to its obstacle cloud, unlike local
continuation and the native checker. Both GPU modes now use the same current
moving collision cloud. Empty collision roles are supported without placeholder
points. GraspGen's target input is separately filtered as well.

The previous C++ exception carrying initial finger/target point contacts is
removed. The common cleaned map replaces it. The redundant contact-skill
current-overlap check/full-observation rejection loop is also removed. Planner
checks of candidate configurations, constraints, joint bounds and self collision
remain. No extra dense post-planner validation was added.

`current_robot_point_filter` records the measured control step, joints/aperture,
per-cloud and per-part removal counts, zero padding and elapsed time. Scene
exports and native reports retain this provenance. Contact-skill timing includes
`current_robot_filter_seconds`; grasp profiling separates filtering from network
generation. Compact feedback retains the filter summary and start-rejection
details. The generic tool introduction explains the policy without supplying
successful coordinates, case-specific triggers or evaluator lessons.

## Validation

Artifacts: `results/current-state-cloud-filter-20261030/watch.html`.

- **152 tests passed** across the main and worker suites. Five historical fixture
  tests were skipped because the older calibrated recordings are unavailable.
- All four saved failing starts now pass: closing49 event 36 and opening13 events
  50, 53 and 56. Filtering took approximately 0.32–0.34 seconds in these probes.
- Actual GPU local and global queries handled empty static/moving clouds while
  retaining one contact-reference point as metadata. Each returned an admitted
  candidate; no phantom contact-reference obstacle was created.
- Tests cover exact zero-padding boundaries, sphere/convex overlap counting,
  canonical-data preservation, new physical poses, hypothetical object movement,
  actual versus hypothetical gripper aperture, coherent indices/exports,
  all contact-skill stages and rejection of new future collisions.
- Opening13 events 53 and 56 produced complete new arc plans. Event 50's fresh
  search failed to connect its selected goals after 2.7°; all three connections
  of the previously saved 8° route still solve under the cleaned map. This is a
  finite-search miss, not proof that the filter prohibits the route. The earlier
  route is evaluator-only and is not used as a runtime fallback or actor hint.
- The palm-case start passes, but the requested future arc still rejects palm/
  arm candidates against remaining observed geometry. Those future collisions
  were not used to clear more points. New physical task success is not established
  by these numerical checks.

## Experiment handling

The previous cohort used commit `15afb67`. It was stopped for this requested
runtime update after five completed baseline trials (two successes); the next
partial trial is preserved as interrupted, not scored as a task failure. Do not
pool a changed-runtime continuation into that frozen cohort.

The replacement study is `results/luna-current-state-filter-rerun-20261030`:
12 fresh baseline trials, opening13/opening37/closing17/closing49 three times each,
600 controls and 1,800 seconds per episode, Luna/high, no inherited notebooks.
Closing49 is the resource canary. Continue only after the existing continuous
watchdog qualifies it. CUDA and EGL use GPU 3; the service has a 16 GiB hard memory
limit, zero swap and whole-group termination. The source/native binary freeze
starts with that new cohort. Keep all evaluator diagnostics outside learner input.

The background service must prepend `/common/home/dz323/.local/bin` to `PATH`
before sourcing `scripts/server_runtime_env.sh`, otherwise it cannot find the
ChatGPT-authenticated `codex` executable. This caused an excluded startup failure
in the previous cohort; no simulator controls were issued by that attempt.

Use `scripts/validate_current_robot_filter.py` for saved-state/GPU regressions,
with a new output directory and a bounded systemd job. It does not execute any
generated motion or invoke Luna. The original task recordings remain unchanged.
