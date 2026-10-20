# Bowl grasp development pilot

This pilot diagnoses the bowl grasp before repeating the frozen two-task
experiment. It uses LIBERO Goal task 8, official initial state 1 and seed 0.
The original ten-episode results in `results/rekep-libero-v1` and
`results/rekep-libero-v2` remain unchanged. Development attempts with different
settings are reported separately; their results are not a new benchmark score.

## Run and inspect

From the repository root:

```bash
PYTHONPATH=src .runtime/envs/rekep-libero/bin/python scripts/try_rekep_bowl_v3.py \
  --output results/rekep-bowl-pilot-v3/new-attempt \
  --candidate-indices 54 179 14 102 159 --approach-limit 20 \
  --waypoint-steps 25 --servo-scale .5 --follow-ik-posture

.runtime/envs/rekep-libero/bin/python scripts/report_rekep_bowl_v3.py
```

The output directory must be new. The worker environment runs perception and
geometry. The unchanged `vlm-tamp` environment runs LIBERO through the JSON-lines
observation/action interface. Each new run stores a source snapshot, hashes,
configuration and package versions in `protocol.json`. Upstream/checkpoint
revisions remain those recorded in the frozen version 2 protocol.

## Adaptations being checked

- Reuse the original RGB-predicted bowl mask, numbered landmarks, Luna program
  and 200 GraspGen poses. The bowl/plate RGB evidence must still match. No API
  call, manual landmark rebinding or simulator object inventory is permitted.
- Transfer the bowl region to other views with MobileSAM using projected
  measured points. Reject masks with weak seed support or excessive area.
  Surface proximity alone cannot label the table as intended finger contact.
- Keep raw Panda grasps and the checkpoint's antipodal half-turn symmetry.
  Separate variants account for the measured 6 mm difference in finger body
  attachment; the calibrated base-to-TCP translation remains 97 mm.
- The grasp development diagnostic fixes the detector pose after region and
  proximity filtering. It records, but does not enforce, the cached stage 1
  TCP-to-landmark distance function: an object landmark is not necessarily the
  detector's contact center. This substitution must not be reported as
  satisfaction of that original numerical function.
- Use multi-start, seven-joint IK in an isolated robot kinematic state. Keep
  alternative feasible solutions and reject discontinuous path branches.
- Fuse RGB-D with 1 cm scene voxels and a local 3 mm field. Unknown space stays
  blocked and the collision margin remains 5 mm. Dense tool surface samples
  distinguish inward contact pads from palms and outer finger surfaces.
- Identify observed robot depth returns by agreement with a robot-only render
  at the current proprioceptive pose. This removes false robot obstacles while
  retaining closer scene occluders and unknown space behind the robot.
- Use exact current physical robot hulls and closed visual mesh material to
  handle grid rounding at the robot surface. Open visual components are
  ignored. Closed CAD components with crossing seam triangles exclude every
  intersection-pair bounding region; holes are not filled or convexified.
- Additional fixed cameras cover the nominal Panda workspace. The latest
  pilot has twelve views, including the original agent, wrist and frontal
  views. This is a sensor-rich development adaptation; it is not a result
  obtained with the original three-camera configuration.
- Before motion, check the complete interpolated approach and its IK posture.
  Optional `--follow-ik-posture` updates the existing OSC controller's nullspace
  reference to the checked robot joint target. It does not set joint positions
  or attach objects. Check the actual robot posture after each approach
  waypoint, and stop on a clearance or controller failure.
  The commanded hand pose is the exact forward-kinematic pose of the checked
  joint configuration, and dense tool geometry is checked there as well.
  `--precise-controller` tightens settling to 1 mm translation, 0.01 rad
  orientation and 0.01 rad maximum joint error; pose-only defaults stay
  unchanged. `--closure-steps` controls physical closing duration.

## Physical outcomes and evidence

The robot opens its physical gripper for 60 controls after the five settling
actions, approaches a detector pose, closes normally, then attempts a 5 cm
lift and hold. The simulator retains the 600-control limit.

A valid detector pose or approach is an intermediate measurement. Lift
verification compares RGB-D object surfaces against stationary and
hand-following hypotheses. It requires at least 35 mm of observed transport,
adequate surface coverage, a better fit to the moving hypothesis, and a
nonempty physical gripper gap. Placement is attempted with the cached ReKep
stage functions and native optimizers only after that verification. Official
task success uses LIBERO's actual predicate.

Saved artifacts include mask overlays, all grasp rejections, approach paths,
actual robot posture checks, closure/lift observations, retention evidence,
optimizer output and videos. Seven-joint position/velocity logs help diagnose
control drift. Native contact identities and forces are saved in a separate
evaluation-only file and never enter policy observations or grasp selection.

The remaining placement implementation must be evaluated before claiming full
method success: an attached object's swept collision geometry and retention
throughout placement are not established by a successful tool path alone.
Only the completed stages in each saved attempt may be claimed.

## Budget and preservation

This continuation makes no new API requests. The cumulative conservative API
ledger remains $0.181847 of the authorized $2 cap. Cached Luna requests use
high reasoning. Existing environments, vendor sources and earlier results are
preserved; version 3 changes use separate modules, runner and report files.

The report is generated at `results/rekep-bowl-pilot-report/index.html` from
completed attempt records. Its machine-readable summary includes each
attempt's sensor, grasp and controller configuration.

## Measured status after development attempt 11

All eleven development attempts stopped before physical gripper closure.
There is no verified bowl grasp, lift or placement. These are repeated attempts
on one initial state with changing configurations, not eleven independent
benchmark episodes. The furthest attempt passed seventeen actual robot
posture checks before a controller settling failure.

Nominal grasp approaches now pass admission after correcting observed robot
self-surfaces, increasing camera coverage and checking coherent FK/joint
commands. Actual execution still encounters settling errors or shortfalls in
the unchanged 5 mm clearance requirement. GraspGen's physical grasp quality
has not yet been measured by this pilot.

The final comparison used the official OSC posture torque helper at gain 80
(default 10). The configured robot has 0.1 Nm joint friction loss; this is a
possible source of small settling errors, not a demonstrated sole cause.
No simulator physics parameters, Cartesian gains or object states were changed.
To reproduce that final failed configuration, choose a new output directory:

```bash
PYTHONPATH=src .runtime/envs/rekep-libero/bin/python scripts/try_rekep_bowl_v3.py \
  --output results/rekep-bowl-pilot-v3/replay-final \
  --candidate-indices 179 --approach-limit 4 --waypoint-steps 40 \
  --servo-scale 1 --follow-ik-posture --closure-steps 40 --nullspace-kp 80
```

`results/rekep-bowl-pilot-v3/final-verification.json` independently checks
actual control-log events, source snapshots and API accounting. Sixty geometry
regressions and seven simulator/controller checks, including the live check,
passed. Path and settled-posture sampling do not constitute a continuous-time
clearance certificate.

## Follow-up: contact-physics diagnostic

The separate `scripts/try_rekep_bowl_physics_only.py` runner bypasses the
software collision admission, clearance checks and tracking-error aborts while
preserving normal simulator contact physics. It uses the same cached RGB-D
GraspGen candidate, ordinary Panda actions and physical gripper closure.

The first trial succeeded: the bowl moved 50.25 mm with the hand and stayed held
for twenty controls (about one second). Both finger pads had positive physical
contacts with the same bowl throughout that hold. No object state was set or
attachment created, and the API ledger was unchanged. This establishes a
physical grasp and lift in one scene; placement was not attempted. The eleven
strict-check outcomes above remain unchanged.

Run with a fresh output directory:

```bash
PYTHONPATH=src /home/duo/miniforge3/envs/vlm-tamp/bin/python \
  scripts/try_rekep_bowl_physics_only.py \
  --output results/rekep-bowl-physics-only-v1/new-attempt

PYTHONPATH=src /home/duo/miniforge3/envs/vlm-tamp/bin/python \
  scripts/report_rekep_bowl_physics_only.py
```

The video, RGB-D transport evidence and evaluation-only physical contact
verification are linked from `results/rekep-bowl-physics-only-report/index.html`.
