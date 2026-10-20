# State 11: wrist, moving geometry and hinge recovery

This is a development diagnosis of `libero_90`, `KITCHEN_SCENE7_open_the_microwave`, initial-state index 11, seed 0. It replays the original physical grasp/approach and changes subsequent perception/planning. It is not a new held-out benchmark or a planner-only ablation. Native LIBERO task predicates determine success.

## Original failure

The original short-motion run ended at 8.8802 degrees after three complete paths: 3, 3 and 2 degrees. The third path requested a 33.958-degree wrist excursion and a 0.729-radian maximum joint excursion. The physical wrist turned about 32 degrees while the door advanced only about 0.5 degree. Post-hoc simulator contacts show 34.9 N palm-to-handle contact, strong finger/handle contact and finger/door contact. The initial grasp was near the upper handle mount.

The immediate termination was the next planner rejecting its start state. All three requested horizons returned `invalid_start`, identifying configured `gripper0_finger2_collision` against the stationary point cloud. Some reconstructed static points penetrated that volume by about 3.4 mm and lay 9–13 mm from the lagging selected handle cloud, outside its 8 mm moving association. This was both a poor manipulation posture and imperfect surface tracking. No execution-time checker stopped a path.

Simulator contact names and hinge state are confined to post-hoc reporting. They do not select geometry, grasps, trajectories or contact permissions in the policy.

## Changes

- For a small planned door motion, use a soft orientation-continuity objective in collision-aided projection. Admit locally continuous endpoint and path choices: at most 8 degrees of wrist excursion during calibration, then `max(10, 2*door_degrees + 3)` after hinge refinement. Full-motion mode retains unrestricted wrist rotation. These are planning-time branch choices, not fixed wrist equalities or execution vetoes.
- Track spatially separated door texture corners through calibrated RGB-D images using forward/backward pyramidal optical flow. Lift corresponding material landmarks into measured world points. Reject robot pixels with known robot-only rendered depth. Robustly estimate upright SE(2) target motion and derive the hinge from `(I-R)h=t`.
- Require spatial baseline, rigid residual and geometric-bootstrap stability before using a pivot update. A measured rigid pose can be usable before the hinge is stable. The bootstrap spread is not a calibrated confidence interval. The axis is assumed upright; this does not solve arbitrary-axis identification.
- Use measured motion to reconcile previous/current moving geometry. Preserve current stationary evidence, require depth support, and confine reassignment to the selected handle neighborhood. A hypothetical robot pose cannot erase geometry.
- Revision 1 used rigid landmarks for hinge and surface roles but retained the old translation-only handle-cloud update. It reached 20.03 degrees, then again encountered an invalid-start point-cloud collision. Every issued path completed.
- Revision 2 also uses the measured rigid pose to transport the selected handle cloud, accounts for motion during physical grasp closure, rotates the selected anchor consistently, and renews material landmarks after a measured 8-degree arc. The original hinge, geometry and failed results remain saved.
- Revision 2 stopped at 12.49 degrees because the native contact tree also contained the transported handle reference. Nine reference points penetrated the configured palm, up to 0.45 mm, although current measured moving points were outside by at least 1.03 mm and the simulator recorded no palm contact. Revision 3 uses the reference only for semantic association; the contact collision tree contains current measured moving endpoints. Future queries still transform those measured endpoints about the estimated hinge. There is no blanket palm exemption. A regression checks both this false positive and rejection when the same point is truly observed.

Each planned short path runs in full before the next RGB-D observation. There are no runtime collision, residual, tracking or progress admission checks. Real physics and gripper closure remain active. Native task success is evaluation-only and does not cause hidden policy changes.

## Physical result

Revision 3 reached **89.7584 degrees**, passed the native LIBERO predicate, and completed the policy's visual goal at control step **460**. All **12** issued manipulation paths completed; execution-time veto checks: **0**. The same initial physical grasp/approach was replayed in all three development trials. Native success first occurred at step 420 with both pads still contacting the handle/mount. Bilateral contact was lost near the end, and the final contact was a finger against the panel: continuous grasp retention remains a limitation.

The 12 post-pull hinge estimates had perpendicular-axis error **3.31–7.57 mm**, ending at **5.52 mm**, versus **44.16 mm** initially. These are evaluation-only errors; the policy uses RGB-D correspondences and an upright-axis prior, never the simulator's hinge state. The grasp-induced bootstrap estimate before the first pull had about 19 mm error.

No robot–mug contacts and no palm contacts after closure were recorded in the corrected trial. The shared original approach still had brief palm–handle contacts, with a peak of about 50 N. Success therefore measures the native task outcome; the experiment does not establish contact-free execution or guaranteed grasp retention.

Automatic cleanup reclaimed **259,591,066**, **85,020,260** and **617,368,595** bytes in revisions 1–3, respectively. Essential videos, lossless planning inputs, source snapshots, landmark correspondences and outcome/contact logs remain. The API ledger did not change.

## Reproduction

Use the existing isolated research environment; do not change the simulator environment. Every output directory must be new. The runner freezes source snapshots, configuration and native build metadata before the first episode and automatically compacts observations/frames after each episode.

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python scripts/run_mcvamp_microwave.py \
  --output results/state11-observed-contact-recovery-NEW \
  --tasks open_microwave --states 11 \
  --backend partial_pose_arc --motion-policy short --hinge-estimator hybrid \
  --reconcile-handle-roles \
  --replay-grasp-source results/mcvamp-microwave-frozen-20261015/open_microwave-state011 \
  --arc-proposals 32 --arc-seconds 20
```

The frozen grasp replay is explicitly a development diagnostic. It is physically re-executed, not an assisted attachment. No new VLM requests are allowed by this runner; it reuses observation-validated cached High-reasoning grounding. The cumulative API ledger is checked before and after.

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python -m pytest -q \
  tests/test_rigid_hinge_tracking.py tests/test_short_motion_policy.py \
  tests/test_partial_pose_arc.py tests/test_mcvamp_microwave.py \
  tests/test_mcvamp_goal_proposals.py tests/test_timed_execution.py \
  tests/test_coupled_storage.py
```

36 focused tests passed before revision 3. The suite includes noisy/outlier rigid landmark recovery, rejection of unobservable single-bar motion, measured short-arc transport without a stable hinge claim, preservation of stationary observations, rejection of the saved 34-degree wrist path during planning, and the tracking-reference collision regression.

The report script reads native contacts/hinges only after execution and saves its evaluation separately:

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
.runtime/envs/rekep-libero/bin/python scripts/report_state11_recovery.py \
  --runs results/state11-hybrid-recovery-20261015 \
         results/state11-rigid-pose-recovery-20261015 \
         results/state11-observed-contact-recovery-20261015
```

Author-hosted real-robot microwave videos are illustrative, not matched LIBERO comparisons. Geometry, clutter, robot embodiment, sensing and controllers differ. No VLA has been evaluated on this exact state in this experiment.
