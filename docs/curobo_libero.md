# cuRobo IK adaptation on LIBERO

This experiment changes the grasp IK backend and adds a 0.025 rad command joint
reserve. It retains the v11 RGB-D reconstruction, task contact permissions,
physical Panda controller, and native LIBERO success predicate. Contact-rich
execution is an empirical test; collision-free operation is not guaranteed.

cuRobo v0.7.8, revision `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`, is isolated
in `.runtime/envs/curobo-libero` (Python 3.10, PyTorch 2.5.1+cu124, Warp 1.18.0). The upstream
checkout is `.runtime/curobo-v0.7.8`. Existing LIBERO and GraspGen environments
are unchanged. JSON-lines requests send robot-only calibration and target TCP
poses to this process; credentials and simulator scene metadata are excluded.
`scripts/setup_curobo_libero.sh` reproduces this project-local install using
`requirements-curobo-libero.txt`. Building requires the local CUDA compiler;
this machine uses CUDA 12.6 to compile the Torch 12.4 extensions for its RTX 4090.
Warp 1.18.0 fixes a driver-entry-point initialization failure seen with 1.7.2
on this WSL driver. No upstream source changes were needed.

The exporter derives a seven-joint URDF from calibrated robot axes, anchors,
joint limits, and the controller grip_site. It aligns every joint axis to local
z for cuRobo's parser, preserving its physical reference angle. Runtime admission
requires FK parity on 25 configurations, including four isolated simulator FK
queries. cuRobo uses wxyz quaternions; the simulator uses xyzw.

For each eligible GraspGen pose, cuRobo optimizes 64 seeds and returns 32
solutions. The adapter rejects invalid solver flags, pose residuals, inadequate
joint reserve, duplicates, and VAMP self collisions. Up to 16 distinct postures
are tried, ordered by joint travel with a joint-reserve penalty. Rejecting one
posture no longer rejects that grasp. Each posture must pass open/closed grasp,
sampled approach, and predicted manipulation checks against observed geometry.

cuRobo's internal world/self collision functions and MotionGen are not used in
this first adapter. The known VAMP robot checker and full robot-mesh observed
scene checker remain authoritative for admission. The approach uses joint
interpolation to a standoff followed by a Cartesian segment. The segment is
solved backward from the admitted grasp posture, ensuring that its terminal
seven-joint state is exactly the one checked for closure and manipulation.
Local continuation chooses the closest posture within the hard joint reserve;
the joint-reserve cost only ranks initial grasp alternatives. Drawer pulling is
still the observed-axis template with RGB-D feedback. Point-goal bowl placement
uses bounded CPU refinement with soft orientation. This isolates the IK
question; it is not a full cuRobo motion-planner benchmark.

Finger-to-selected-target and local drawer-panel permissions remain explicit;
the palm, arm, unrelated handles, and other objects receive no new exemptions.
Observed occupied surfaces, sampled paths, and discrete controller boundaries
do not certify unseen geometry or continuous collision avoidance. Simulator
contact physics remains active; grasping requires closure and observed retention.

Run with the existing orchestration environment (single-threaded BLAS):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .runtime/envs/rekep-libero/bin/python scripts/run_physical_tasks_curobo.py \
  --output results/physical-tasks-curobo-eval
```

The default cohort is tasks 8 and 0, states 1, 11, 21, 31, 41, seed 0, at most
600 control steps per episode. Every planned failure counts. These initial
states were previously inspected, so there is no held-out generalization claim.
The runner saves dependencies/configuration source snapshots, frame checks, IK
queries and rejected postures, contact checks, videos, trajectories, and native
success measurements. Cached Luna and GraspGen outputs avoid additional API
calls. New output directories are required; old results remain preserved.
