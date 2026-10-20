# Corrected grounding, complete-task grasp selection and observed feedback

The updated runner is `scripts/run_physical_tasks_v6.py`. It keeps the protected
LIBERO environment unchanged and uses the existing isolated Python 3.10 grasp
worker. Every episode uses seed 0, five settling actions and at most 600 control
steps. The official states are 1, 11, 21, 31 and 41 for each of the two tasks.
There are no new paid model requests or automatic API retries.

## Grounding changes

The bowl adapter now excludes source landmarks using the **admitted source
mask**, rather than every overlapping SAM proposal. A broad mask covering bowl
and plate previously caused two separate plate references to be discarded.
The revised adapter compiles all five cached bowl scenes.

Drawer failures were not all numerical plane-fitting failures. The original
cached Luna grasp landmark selected a different handle height in states 11, 21
and 41. Some numbered images did not have a landmark on the middle handle.
The new adapter finds a source-containing RGB parent mask, checks its measured
vertical front plane, extracts three separate protruding horizontal bars, and
selects the middle bar by observed height. It then refits the panel near that
bar, excluding the protruding material. All five cached drawer scenes compile.

This is an explicit **task-template geometry correction**, not an improvement
in Luna's original answers. The original numbered image, functions, landmark ID
and position are retained. The specification records the observed handle
centers, new geometry anchor, and whether the ordinal role changed. No simulator
object identities, instance segmentation, object meshes, joint axes or object
poses enter that decision. The three-bar/front-view assumption limits the
adapter; occluded handles and rotated views need separate validation.

## Grasp selection and execution

The detector still generates 200 Panda diffusion candidates and retains its top
100 discriminator scores. Bowl grasps reuse the original candidates; drawer
grasps are generated once from the corrected observed middle-bar cloud.

For each geometrically admitted grasp, calibrated seven-joint IK checks the
subsequent manipulation and its approach. The closest fully admitted grasp is
selected, with confidence breaking ties. The robot-only kinematic chain was
previously checked against 25 isolated MuJoCo configurations. VAMP's native
Panda checker supplies robot self-collision admission. No scene identity is
used in FK, IK or the self-collision check.

The first updated frozen experiment, v5, kept a fixed wrist orientation for
placement. It completed all ten episodes, with **1/5 bowl and 2/5 drawer
successes**. It exposed two additional problems: a nominal grasp can be
reachable while its palm intersects another handle, and the observed bowl
attachment can invalidate a placement that was reachable under its nominal
attachment assumption. The v5 records and exact source snapshots are preserved.

In v6, bowl lifting and placement solve for the estimated object support point.
Wrist orientation is a **soft preference**, not an extra equality constraint.
After a retained lift, the measured object transform binds that support point
to the TCP, and the complete placement is checked again. A short correction
loop reduces the measured TCP-bound point error before opening. The semantic
On relation remains an approximation represented by a support-point goal;
only LIBERO's actual predicate establishes task success.

Drawer selection checks a 16 cm continuation. The execution goal is 15 cm of
**observed** handle travel, using a fixed margin for measurement/controller
error relative to the earlier 14 cm request. Pulling proceeds in short chunks.
After every chunk, the runner fits displacement of the selected handle surface
along the measured normal, checks hand/handle separation, and uses the remaining
observed distance to set the next command. Tracking loss or repeated lack of
progress becomes a recorded failure. Simulator success is not used to steer
that feedback loop; it is read only for final evaluation.

## Collision scope

An occupied-point probe compares measured RGB-D surfaces against configured
robot convex collision volumes. Known robot endpoints are filtered using the
calibrated robot-only depth prior. Intended selected-surface contacts are
restricted to inward finger faces.

The initial full-robot probe rejected every development candidate. Arm/finger
intersections are therefore retained as **diagnostics**, while admission uses
the palm intersection check to address the demonstrated upper-handle collision.
It is evaluated at sampled preflight states. This does not validate unknown
space, continuous motion, the carried object's complete shape, or all arm
clearance. The original physical contact dynamics remain enabled throughout.
No object is snapped, attached or moved by a state-setting command.

Atlas execution is not integrated in this runner. The earlier isolated Atlas
and MC-VAMP fixture results remain separate. Full robot/environment collision
admission needs calibration before either can be described as an integrated
collision-safe planner. See `docs/planner_feasibility_v4.md` for those experiments.

## Experiment records and reproduction

```bash
.runtime/envs/rekep-libero/bin/python scripts/run_physical_tasks_v6.py \
  --output results/physical-tasks-v6-new \
  --tasks 8 0 --states 1 11 21 31 41

PYTHONPATH=src .runtime/envs/rekep-libero/bin/python -m pytest -q \
  tests/test_physical_tasks_v5.py tests/test_scene_clearance_v6.py \
  tests/test_planner_geometry_v4.py
```

The existing pinned MC-VAMP Panda module must be present in
`.runtime/mc-vamp-deps`; setup is documented in the earlier feasibility report.
Nonempty result directories are rejected. Outputs include exact source/config
snapshots, cache evidence, corrected region overlays, geometry specifications,
all detector candidates and rejection reasons, preflight paths, observed
attachment and feedback records, commands, frames and videos. An API ledger
checksum confirms that no paid requests were added.

The report compares v4, v5 and v6 **separately**, retaining five planned episodes
per task in every frozen cohort, including failures before closure or movement.
Development attempts are not substituted into those denominators. All these
states were inspected previously, so these are repeated development comparisons,
not a held-out generalization estimate. Several changes were made together;
the experiment does not isolate each change's causal effect.
