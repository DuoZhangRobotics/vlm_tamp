# Full-arm collision admission and moving-source geometry (v7)

Version 7.2 implements a common geometric validity interface for later Atlas
or pRRTC integration. Execution still uses task waypoints, seven-joint IK and
LIBERO's Panda pose controller. It does not integrate a global search planner.
All earlier sources, configurations and reports remain intact.

## Scene and robot representation

Calibrated metric depth from the existing twelve views is fused at 2 mm spacing.
Only measured occupied surfaces enter scene collision checks. Known current
robot pixels are identified through a robot-only depth render. The v7 prior
mirrors every configured robot geometry enabled in the normal camera render,
including contact-pad boxes when their render group is enabled.
Same-pixel depth matching allows 0.1 mm numerical disagreement. Closer scene
measurements beyond this tolerance remain obstacles. Neither current nor future
robot volumes erase unobserved space.

The first frozen v7 cohort is preserved at `results/physical-tasks-v7-eval`.
Four initial states failed on one robot pixel because the first images preceded
the later camera calls' kinematic refresh. Version 7.1 refreshes MuJoCo kinematics
before any image capture and renders agentview/wrist directly from that state.
Depth, camera calibration and robot-only priors now share the same physical
state. No control step, object pose override or collision exception is added by
this synchronization correction. The revised frozen cohort uses a fresh output.

Every moving configured Panda collision volume is checked, including all arm
links, palm, finger meshes and pads. The fixed mounting base is excluded.
Convex link volumes come from the known configured robot model. Hypothetical
gripper positions are evaluated in isolated kinematic state; live robot/object
state and MuJoCo contact settings are unchanged. VAMP supplies the existing
Panda sphere-model self-collision check, while IK supplies arm joint bounds.

The RGB-selected bowl surface is expanded over a connected measured component
above an observed support plane, within bounded lateral growth. An observed
convex envelope is used for payload/environment checks; this overapproximates
the bowl's concavity but cannot reconstruct unseen exterior geometry.

For the drawer, the RGB parent mask, front plane and neighboring observed handle
heights identify the middle front-panel band. This panel and selected handle are
represented separately from stationary observed cabinet surfaces. The hidden
drawer body and rails are not reconstructed. The moving-part pose is a horizontal
translation along the observed panel normal, with progress measured from RGB-D.
The handle contact region is extended to multi-view observed protrusions within
4 mm of the RGB-seeded handle bounds, so measured undersides can be contacted
without allowing contact with the flat drawer panel or adjacent handles.
During preflight the hypothetical displacement is coupled to hand travel; during
execution observed displacement corrects that prediction after each pull.
Fresh depth rebuilds the stationary map after closure and observed manipulation,
so the old moving surface is not retained as a stationary obstacle.

## Contact compilation

No GPT-generated robot-geometry ignore list is used. Grounded source/target roles
are compiled into local geometric rules:

- Only known inward Panda pad patches can touch the predicted selected source
  during grasp, carrying, pulling or placement, within 3 mm pad/region tolerance.
- Where a configured convex finger mesh overlaps its pad patch, that same small
  patch receives the allowance. The remaining finger mesh, palm and arm retain
  ordinary collision admission. A 0.5 mm tangential pad boundary tolerance is
  recorded explicitly.
- Near a pad, the same inflated clearance shell is permitted for points outside
  the actual finger mesh. This shell exception never permits penetrating an
  outer finger shoulder beyond the physical pad patch.
- The lowest 2 mm object support band may touch its observed initial support
  while lifting, or the grounded placement support during placement.
- Initial drawer/front-seam adjacency is allowed only at observed region edges,
  within 3 mm initial displacement and no more than 0.2 mm point penetration.
  This does not exempt a whole cabinet, guide rail or robot link.
- Stationary scene contacts are never granted robot-pad permissions merely
  because they are near the selected object.

The first development attempt exposed a gripper-frame bug in the new aperture
estimator: spatial TCP side is not the joint-vector order. An isolated FK probe
now calibrates each named pad's prismatic direction, and an asymmetric contact
fixture verifies the conversion. This is a v7 development correction; v6 remains
unchanged and the first attempt is preserved.

## Planning and execution admission

Grasp candidates and cached high-reasoning Luna outputs are reused only after
selected RGB-D evidence is validated. Source geometry and keypoint centres must
agree within 0.2 mm; selected RGB regions and keypoint patches must have mean
absolute channel change at most 0.25 and maximum change at most 16 on the 8-bit
scale. Larger movement, recoloring or missing depth rejects reuse. These are
tight simulation-capture tolerances, not a hardware noise model.

The synchronized v7.1 cohort is preserved at `results/physical-tasks-v7.1-eval`.
Its one cache failure involved a source depth difference below 0.1 mm and mean
RGB channel difference 0.0384, with unchanged keypoint centres. Version 7.2 records
these differences instead of declaring them byte-identical. Cached GraspGen poses
remain proposals, accepted only when the paired selected source agrees within
0.2 mm and every candidate passes current-geometry admission. No source/obstacle
geometry is replaced by a cached cloud. This spends no additional API budget.
Detector poses, pose variants and
ranking follow the previous runner. Candidate admission checks open-gripper
geometry, the estimated contact aperture, the complete approach and the complete
nominal manipulation. Attachment during preflight is explicitly hypothetical;
only physical closure and observed retention establish an execution binding.

Joint edges are interpolated at a maximum 0.025 rad per joint. All generated
samples undergo self-collision, full-arm observed-scene and, where applicable,
moving-payload checks. Before each controller command the path from current
proprioception to its requested joint guide is rechecked. Actual geometry is
checked at action boundaries, with closure and drawer checks deferred until
the corresponding RGB-D moving-surface update. Checks do not cover every internal
MuJoCo integration step or certify continuous swept-volume clearance.

Unknown space, complete hidden geometry, tracker errors and controller deviations
remain limitations. No full collision-safety claim follows from passing these
checks. An empty admitted candidate set is a recorded failure, with rejection
stage, robot part, scene part and example collision points retained.

## Reproduction and evaluation

The existing isolated Python 3.10 environments and pinned upstream/checkpoint
revisions remain unchanged. Development runs use fresh output directories. After
development, freeze the configuration and source snapshot before the ten planned
episodes: tasks 8 and 0, official states 1, 11, 21, 31 and 41, seed 0, five settling
actions and at most 600 control steps. Every failure is counted, even before motion.
Success is recomputed from the actual LIBERO predicate in the saved action log.
These states were already inspected in v6; this is a regression experiment.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/envs/rekep-libero/bin/python \
  scripts/run_physical_tasks_v7.py --output results/physical-tasks-v7.2-eval
.runtime/envs/rekep-libero/bin/python scripts/report_physical_tasks_v7.py
```

The report separates initial scene calibration, candidate admission, physical
closure, tracking, motion admission and task success. It links videos, point-cloud
plots, contact policies, candidate rejections, geometry snapshots, commands and
observed drawer feedback when available.

`scripts/replay_moving_drawer_v7.py` separately exercises the moving-scene updates
on preserved v6 RGB-D drawer motion. Its report explicitly states that it is a
replay, not a new v7 physical success. Synthetic regression fixtures also check
that moved surfaces vacate their old location while stationary cabinet obstacles
and payload/environment collisions remain active.

## Later geometry/planner backends

[RNDF](https://arxiv.org/pdf/2309.16085), the paper supplied by the user, learns
configuration-conditioned, per-link signed distances and joint-space gradients.
It is a possible robot-geometry query backend. It does not replace RGB-D scene
reconstruction, object tracking, contact permissions or task constraints. A future
Panda model would require matching the configured robot geometry and validating
distance error around the contact tolerances used here. No RNDF checkpoint is
downloaded, trained or used in v7. Atlas/pRRTC search can later consume the same
robot-plus-drawer state validity interface; space-time planning is deferred.
