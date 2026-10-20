# ReKep + GraspGen on LIBERO

This experiment joins actual RGB-D perception, generated ReKep Python stage costs,
Panda-specific grasp detection, the released ReKep optimization routines and
physical Panda execution. It measures two LIBERO Goal tasks, not a reproduction
of ReKep's reported real-robot success rate or the standard VLA camera protocol.

## Reproduce

From `/home/duo/Rutgers/ARC/random/vlm_tamp`:

```bash
/home/duo/miniforge3/envs/vlm-tamp/bin/python scripts/setup_rekep_libero.py
PYTHONPATH=src .runtime/envs/rekep-libero/bin/python scripts/run_rekep_libero.py --phase development --allow-paid
PYTHONPATH=src .runtime/envs/rekep-libero/bin/python scripts/run_rekep_libero.py --phase evaluation --allow-paid
PYTHONPATH=src .runtime/envs/rekep-libero/bin/python scripts/run_rekep_libero.py --phase report
```

The existing simulator environment stays intact. The new worker uses Python3.10,
Torch2.1/CUDA12.1, NumPy1.26 and a worker-local CUDA compiler targeting the RTX4090
(compute capability8.9). Its inference requirements omit upstream `pickle5`,
which is incompatible with Python3.10; upstream already falls back to built-in
`pickle`. The installed dependency list is saved under
`results/rekep-libero-setup/requirements-installed.txt`.

API credentials are loaded from the previously authorized local environment file,
never written into observations, logs, manifests or model requests. The API
ledger admits requests against a cumulative $2 cap, including the earlier
$0.154725 and reservations for pending or uncertain calls. No automatic API
retries or manually repaired semantic bindings are used.

## Declared adaptations

- Native DINOv2 ViT-S/14 proposal uses five candidates per predicted RGB mask and
  global6cm merging. MobileSAM replaces simulator-provided segmentation.
- Luna-high replaces the original VLM. The numbered-image prompt retains native
  stage-function semantics, with ordinary pick/place excluding extra geometric
  path constraints and drawer pull constraints restricted to pulling.
- Generated functions run through a restricted numerical AST interpreter, not
  unrestricted Python execution. Syntax/index/numeric errors remain failures.
- GraspGen's pinned Panda diffusion and discriminator produce200 base-link
  transforms. The top100 scores are retained; feasible grasps are ranked by
  distance to the instructed landmark, with confidence breaking ties. There is
  no analytic grasp fallback or assumption that a confidence score is calibrated.
- Known Panda geometry converts base-link transforms to the real controller TCP.
  Detector grasps replace ReKep's top-down prior. Both released motion optimizers
  remain in use for paths and later manipulation subgoals, with all seven Panda
  joints regularized and5mm collision margins. Native NumPy function bodies run
  without JIT decorators to avoid cross-process cache imports; source files stay
  unchanged.
- Calibrated agentview and wrist RGB-D are supplemented by the existing fixed
  frontal drawer view. No camera is positioned using an object pose or joint axis.
- Observed-free-space occupancy/ESDF blocks unknown space. Only known robot
  geometry and observed held-region surfaces are excluded. Intended finger
  contact is allowed near the predicted target; whole cabinets are not removed.
- DINO/depth tracking preserves the original numbered-array indexing. Only used
  landmarks need tracking. Drawer points become movable through selected-handle
  membership and observed co-motion, rather than a simulator object identifier.
- Gripper closure and visual relative motion establish retention. Fully closed
  empty fingers or a stationary object do not establish attachment. No snapping,
  teleportation or assisted grasp is used.

## Evaluation contract

Development uses bowl task8/state5 and drawer task0/state0. Evaluation uses official
states1,11,21,31,41 for each task, seed0, five settling actions and at most600
physical controller steps. Gripper opening, closure and retention checks count
against this horizon. Every planned episode stays in the denominator, including
failures before reaching an object. Development does not enter the evaluation
success rate.

Only the official LIBERO success predicate establishes task completion. Robot
pose/proprioception and camera calibration are policy inputs; simulator scene
identities, masks, meshes, joint axes and object poses are withheld from the
policy. A valid program, selected grasp or optimizer result is an intermediate
measurement.

The frozen protocol hashes worker code, native helpers, checkpoint files and
dependency metadata before evaluation. Completed episode outputs can be replayed
without new model calls. An incomplete episode is preserved and requires
inspection; the runner refuses to overwrite it. Reports and videos are under
`results/rekep-libero-report` and `results/rekep-libero-v1`.

The initial drawer development response exhausted its 6,000-token allowance
(5,927 reasoning tokens). Its incomplete answer is preserved without retry.
Evaluation uses a12,000-token allowance under the same cumulative cost cap.
The bowl development collision diagnostic also exposed unobserved moving-arm
space. Exact known-current robot geometry is excluded, while unknown future arm
space remains blocked. This limitation must be distinguished from grasp-model
generation quality in interpreting the results.

## Sources

- [ReKep released code and missing real-world perception components](https://github.com/huangwl18/ReKep)
- [GraspGen code](https://github.com/NVlabs/GraspGen)
- [Public Panda generator/discriminator checkpoints](https://huggingface.co/adithyamurali/GraspGenModels/tree/ec1ccbb5eec0680db669246ac312a3636f16ee43/checkpoints)

GraspGen's released license permits research/evaluation use. Exact source and
checkpoint revisions are recorded in the experiment configuration and manifests.
