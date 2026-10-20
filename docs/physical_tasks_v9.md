# Contact-policy correction on LIBERO

v7.2's 0/10 was a software-admission result: none of the episodes reached physical closure. It cannot establish that the grasp model or task grounding failed. A counterexample is bowl state 11: the same successful v6 grasp was vetoed solely for a 0.660 mm finger-mesh overlap with the selected bowl. The recorded physical episode includes that finger-to-bowl contact. The pad-only allowance was too narrow.

## Contact semantics

v8 automatically builds touch pairs from the configured Panda links, selected RGB-D task regions and stage. GPT does not create a global collision ignore list.

- Both functional finger meshes and pad geometries can contact the selected bowl/handle during grasp, carry/pull, placement, release and the first 12 mm of release withdrawal. Contact association is 4 mm in the selected observed target region.
- Drawer fingers can brace against the observed flat front panel within 5 cm of the selected handle, using a 4 mm plane-association band. Other protruding handles, distant panel regions, arm and palm remain checked.
- The bowl can contact the observed initial support while lifting and selected plate while placing/releasing. This permission covers the identified support pair rather than only an arbitrary lowest-2-mm slice of an estimated convex envelope.
- A moving drawer's observed panel boundary can have shallow contact with the cabinet seam. This is not a complete model of hidden rails or the drawer body.
- Fingers and the observed moving-payload envelope require no penetration into unrelated surfaces. Positive gaps under 1 mm are logged separately and do not become false collision labels. Arm/palm keep the original 1 mm clearance margin.

Allowing a pair affects the software admission test only. Simulator contact forces stay enabled. Closure and observed retention still establish a grasp, with no object snapping, welding, position injection or artificial attachment.

The hand/object touch-link convention follows the [MoveIt Task Constructor pick-and-place example](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html). Our implementation retains physical closure and tracking instead of introducing simulator rigid attachment.

## Geometry corrections

The support-plane cutoff for bowl scene updates is evaluated in stationary world coordinates. Applying it after rotating the bowl into its canonical frame left legitimate base observations in the stationary map. Newly visible surfaces within a fixed 6 mm association band of the initial selected body are fused into the moving surface; the initial association model does not grow recursively. The table plane and selected plate stay separate. Observation association is bounded, not a full shape reconstruction.

MuJoCo soft finger-joint stops can exceed the nominal interval by a few micrometres. Isolated FK now accepts the exact current measured gripper posture without clamping it; an unmeasured out-of-limit hypothetical posture is still rejected. The live simulator state is not modified by these queries.

## Reproduction and evaluation

Use the existing isolated worker environment and protected LIBERO environment. No dependencies, checkpoints or earlier results are changed. Grounding and GraspGen candidates are reused only after the same recorded fresh-image/depth agreement checks as v7.2. No new API requests or grasp-model inference are made.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .runtime/envs/rekep-libero/bin/python scripts/validate_contact_policy_v8.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .runtime/envs/rekep-libero/bin/python scripts/run_physical_tasks_v9.py \
  --output results/physical-tasks-v9-eval
.runtime/envs/rekep-libero/bin/python scripts/report_physical_tasks_v9.py
```

Every runner output snapshots its sources and configuration. Existing destinations are refused. Development pilots are preserved separately. The frozen regression uses seed 0 and official states 1, 11, 21, 31, 41 on bowl/plate and middle-drawer tasks, with five settling actions and a 600-control-step maximum. All planned failures remain in the denominator. Native LIBERO success is recomputed from the action log, and final-state success is recorded separately.

The offline audit script uses known successful poses solely to diagnose regressions. Runtime grasp selection retains the original geometric ranking and cannot use that diagnostic pose, simulator contact labels or the native success predicate as input. Simulator identities in evaluation contact logs remain explanation-only.

## Scope

This is our ReKep-inspired integration; it does not run ReKep's native optimizers. Seven-joint IK, task templates and sampled path admission remain unchanged. VAMP provides self-collision checks only. All moving configured arm/finger/palm geometry and an observed convex payload envelope are tested against measured surfaces. Convex fill, incomplete visibility, tracking uncertainty and limited IK/route alternatives remain limitations. No certificate is made for unseen geometry, hidden bodies, unknown space or continuous motion. All official states were previously inspected; these results are regression measurements, not held-out generalization.

The preserved development trail is under `results/contact-policy-v8-development`: v8.0 exposed the measured-finger soft-stop issue, v8.1 reached physical closure and exposed bowl-base misclassification, v8.2 corrected that association and exposed two positive-clearance payload points. v8.3 separates those positive gaps from penetration. Corrections are made before the final source freeze.

## Connected selected support (v9)

The RGB-selected plate seed can cover only its flat center. v8 still rejected bowl contact with the raised rim, 18–21 mm from that seed. v9 extends the selected support through connected measured surfaces in a bounded 2.5 cm lateral ROI above the stationary tabletop. Its seed/expanded counts and height cutoff are recorded. The rim becomes permitted placement support; the table and disconnected objects remain obstacles. The default touch pairs, measured FK, margins and task-motion algorithm are otherwise unchanged from v8.3. Earlier frozen sources and v8.3 episodes are preserved.

During later lifts, a fixed nearest-initial-surface cutoff can still leave newly seen source surfaces as stationary ghosts. v9.1 owns the current connected measured component, seeded by the tracked source inside a fixed initial-reference ROI (2 cm lateral, 1.5 cm vertical), above the stationary tabletop and excluding the selected plate. The ROI reference does not grow recursively. Other disconnected components remain obstacles. This remains bounded observational association, not an object-instance oracle.
