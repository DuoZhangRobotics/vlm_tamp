# Constrained planner feasibility and physical task adaptation

This follows the strict ReKep/GraspGen experiment and the successful physical
bowl grasp diagnostic. The new physical baseline uses RGB-selected roles,
metric depth, GraspGen and typed geometric templates with IK/OSC execution.
It does **not** call the native ReKep optimizers, OMPL or MC-VAMP during physical
episodes. Planner compatibility tests are separate experiments.

## Why numerical export is possible

There is no new VLM numerical capability. Luna's cached high-reasoning outputs
still supply landmark roles and stage structure. The project already contained
`pose_constraints.py` and `drawer_constraints.py`, including pose residuals and
Jacobians. The new adapter connects those components to two predefined tasks:

1. Bowl placement: select a separate plate RGB mask, estimate its visible
   surface center and height from depth, and propose a bowl support point above
   that surface. This is a point-target proxy for the semantic relation On.
   It is not an exact containment/support model of the entire bowl and plate.
2. Drawer pulling: fit a local front plane, separate protruding handle points,
   and assume horizontal prismatic motion normal to the visible front.
   A visible front normal is an axis estimate, not discovery of an arbitrary
   articulated mechanism. The travel request is a configured 0.14 m policy
   parameter, not a number inferred from the instruction or simulator joint.

For bowl placement, an observed retained lift supplies a measured initial-object
to-held-object transform. That transform and the measured TCP pose bind the
estimated object support point into the robot TCP frame. Closure-induced object
motion is registered before testing the lift. Without that observation, an
object goal remains an object-frame proposal and cannot be treated as a verified
robot goal. Drawer line binding remains marked unverified pending observed pull.

Depth estimates, role/mask association, grasp retention, commanded travel and
tracking can still fail. Numeric JSON export does not certify geometric accuracy,
attachment, collision clearance or task success. It also does not compile
arbitrary generated ReKep Python into an arbitrary joint constraint.

## OMPL

[OMPL's constrained planning interface](https://ompl.kavrakilab.org/constrainedPlanning.html)
accepts a differentiable equality residual `f(q) = 0` and an optional analytic
Jacobian. `AtlasStateSpace` provides the atlas representation; it can be combined
with `RRTConnect`. This is the concrete interpretation of trying Atlas RRT here.
Travel bounds and collisions remain state-validity conditions, separate from
the equality manifold. OMPL's documentation recommends projection as a starting
point because it is generally less sensitive to parameter tuning; an individual
fixture can still favor Atlas.

We installed **OMPL 2.0.1** into `.runtime/ompl-deps`, without upgrading either
existing environment. A robot-only product-of-exponentials model exports the
calibrated world/TCP transform and all seven joint axes, anchors and limits.
Its analytic translation/body-rotation Jacobian composes with the geometric
residual. The drawer fixture has two line equations and three fixed-orientation
equations: seven ambient joints, five equalities, two remaining dimensions.

The Atlas fixture solved and passed interpolated-path checking; the projection
fixture timed out with the recorded settings. These are single compatibility
runs on a 0.10 m segment with native VAMP self-collision checks and an empty
environment. They do not establish obstacle avoidance or physical drawer success.
The selected grasp did not yield an accurate full 0.14 m IK goal in eight
deterministic local searches; this is a failed IK search, not a proof of global
infeasibility. The shorter fixture keeps that workspace issue separate from API
compatibility.

The wheel requires explicitly associating `SpaceInformation` before constrained
space setup. Manual freeing of nanobind-owned states crashed the first attempts;
the working adapter leaves ownership to the bindings and isolates each run in
a process. Shutdown reference diagnostics remain a reason to validate a C++
integration or updated bindings before adopting a persistent production service.
OMPL does not permit RNG seed zero, so its planner seed is **1**; physical LIBERO
episode seed remains **0**.

## MC-VAMP

The released code linked by [Purdue's CoMMA Lab](https://commalab.org/) is the
[`constrained_planner` branch](https://github.com/CoMMALab/vamp/tree/constrained_planner).
It accompanies [Vectorizing Projection in Manifold-Constrained Motion Planning
for Real-Time Whole-Body Control](https://arxiv.org/abs/2604.13323). The paper
evaluates Panda line/plane constraints with and without fixed orientation, which
closely matches our drawer template. Its acceleration is CPU SIMD projection
and collision checking; published benchmark speedups do not transfer automatically
to this pipeline.

We cloned revision **24e0f8adef0649a957fec77bf33f8c2f04515a18** and built only
the Panda Python module into `.runtime/mc-vamp-deps`. We used that branch's
`crrtc`, `TaskSpaceConstraint` and `Composable_TaskSpaceConstraint` bindings,
not an unconstrained PyPI VAMP wheel. The TSR transforms use quaternion
**wxyz followed by xyz**, unlike our external world xyz/xyzw pose convention.

The native Panda FK agrees with LIBERO after a fixed base and end-effector/TCP
conversion. Across 25 configurations, maximum position error was 0.000111 mm
and maximum rotation error was 2.25e-7 rad. This validates kinematics and the
frame conversion, not the spherical collision decomposition or finger contact.
The custom FK/Jacobian agrees with isolated MuJoCo to floating-point precision;
independent finite differences also pass.

MC-VAMP solved the same 0.10 m fixture. Its direct-connection result contained
only two endpoints, although the native connector checks projected edges. The
Python adaptation therefore reconstructs sampled motion with native projection
and verifies each output. Plain linear interpolation of that sparse result had
3.32 mm maximum line deviation; after projection, measured maximum line
deviation was 0.672 mm, with 0.000937 rad maximum orientation error. The binding
discards the native projection success flag, so residual checking is essential.
The native convergence distance is not equivalent to OMPL's equality-norm
tolerance. We have not performed a matched-tolerance performance comparison.

[VAMP supports observed point clouds through CAPT](https://github.com/CoMMALab/vamp/tree/constrained_planner#environment-representation),
so a new SDF backend is not required to supply RGB-D geometry. We demonstrated
ingestion of 74,615 observed points. Both grasp-fixture endpoints were rejected
with the unadapted sphere model and no intended-contact exceptions. This test
establishes input compatibility, not a working scene collision checker.

MC-VAMP remains model-specific and uses spherical robot geometry. For generic
numeric constraints beyond its compiled types, custom tracing or C++ constraints
would be required. Switching from pRRTC to MC-VAMP does not remove that modeling
work.

## pRRTC and nvblox

The accessible `/home/duo/Rutgers/ARC/pRRTC` checkout does not expose the newer
manifold interface mentioned in the conversation. We did not modify it or infer
that the user's newer code lacks that capability. A future adapter needs the
same seven-joint FK/Jacobian, stage-specific residuals, projection/constrained
edge handling, and actual robot collision geometry.

[nvblox's ESDF interface](https://nvidia-isaac.github.io/nvblox/v0.0.9/pages/torch_examples_esdf.html)
can build and query an observed distance map on the GPU and distinguishes
unsuccessful/unknown queries. Its
[C++ interface](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/core_library_interface.html)
also permits direct GPU voxel access. It is a feasible mapping backend for
pRRTC, where batch robot sphere centers can stay on the GPU. Sphere radii and
geometry remain necessary: distance-to-surface values alone do not describe
collision of an entire robot link.

For CPU SIMD MC-VAMP, per-state GPU round trips may negate its speed advantage.
Native CAPT is the simpler first integration; an ESDF alternative should use
batched queries or a consistent CPU snapshot. This is an engineering assessment,
not a measured transfer benchmark. Unknown space must have an explicit policy.
Known robot surfaces, selected finger contacts and the moving drawer require
distinct treatment; masking a whole cabinet or declaring unobserved space free
would invalidate the intended collision test. Current nvblox packages also have
different Torch/CUDA requirements from GraspGen's pinned Torch 2.1/cu121 worker,
so any installation should use another isolated process/environment.

## Recommended integration order

Use Atlas/RRT-Connect as the first general constraint reference, and MC-VAMP as
the fast compiled counterpart for the same drawer template. Before executing
either result, validate collision admission on the actual robot's measured
posture and an observed feasible grasp, including intended finger contact and
self-point removal. Check manipulation reachability while selecting a grasp,
not just endpoint grasp IK. Then execute and reobserve progress and retention.
The current physical experiment shows that a successful grasp or a satisfied
robot line can coexist with a failed overall task. The frozen ten-episode run
completed with **2/5 bowl successes and 0/5 drawer successes**. Two bowl states
were rejected by an overly broad source-mask association rule even though the
cached program contained separate plate landmarks. The drawer that completed
its commands moved the selected surface approximately **0.124 m**, below the
evaluation predicate's opening threshold. Other failures involved front-plane
estimation and observed grasp/closure tracking. These remain failures in the
denominator; the diagnosis does not replace any frozen result.

Keep ordinary bowl pick-and-place free of extra geometric path constraints;
its point target is a terminal goal. Keep drawer line/orientation constraints
stage-scoped to pulling. Preserve the existing strict ReKep benchmark as a
separate baseline, and later port the validated interface into the user's newer
pRRTC manifold implementation.

## Reproduction

```bash
# Isolated OMPL package; does not modify either conda environment.
.runtime/envs/rekep-libero/bin/python -m pip install \
  --target .runtime/ompl-deps --no-deps ompl==2.0.1

# MC-VAMP source is pinned to the revision stated above.
.runtime/envs/rekep-libero/bin/python -m pip install \
  --target .runtime/mc-vamp-deps --no-deps \
  -Ccmake.define.VAMP_ROBOT_MODULES=panda \
  -Ccmake.define.VAMP_ROBOT_STRUCTS=Panda \
  -Cbuild.tool-args=-j8 ./third_party/mc-vamp

.runtime/envs/rekep-libero/bin/python scripts/check_planner_models_v4.py \
  --output results/planner-feasibility-v4/models-new

# Run once for each of projected, atlas, and mc-vamp, using fresh output names.
.runtime/envs/rekep-libero/bin/python scripts/try_constrained_planners_v4.py \
  --planner atlas --output results/planner-feasibility-v4/atlas-new

# Frozen physical task protocol: five states per task, 600 controls each.
.runtime/envs/rekep-libero/bin/python scripts/run_physical_tasks_v4.py \
  --output results/physical-tasks-v4-new --tasks 8 0 --states 1 11 21 31 41
```

Results include exact source/configuration snapshots, task specifications,
bound targets, cached functions, detector candidates, command logs, observed
retention evidence, images and videos. The physical runner preserves earlier
results and refuses to reuse nonempty output directories. All requested states
remain in the denominator. State 1 was used for development before the freeze,
so the five-state results are not a five-state held-out generalization estimate.
All model requests are cached; this work makes **zero new paid API calls**.
