# Existing finger contacts and local planning deadlines

The user identified three `invalid_start` arc failures and correctly challenged
rejecting an existing physical finger contact. The repaired native planner accepts
all three saved measured starts and produces complete plans for their original
8°, 10° and 5° requests. Physical execution of those new paths has not been tested.

## Scope and evidence

- Original cohort: `results/luna-cache-repair-validation-20261030`.
- Exact arc inputs: `baseline-r2-open_microwave-state013-600`, events 50, 53, 56.
- Repair artifacts: `results/contact-continuation-repair-20261030/watch.html`.
- Machine-readable checks: `saved-start-checks.json`, `saved-arc-checks.json`;
  original inputs and actor logs are preserved.
- Actor-authored findings/guidelines were not edited. Evaluator addenda are outside
  the agent workspaces and must not be supplied to the clean practice learner.

The original checker reported `gripper0_finger1_collision` against moving target
points outside the selected contact patch. It tried zero goal proposals. There
were no joint-bound or self-collision rejections. The observed overlaps were
0.259–0.885 mm, 0.027–0.358 mm and 0.242–0.893 mm respectively, with zero added
gripper padding. Points were roughly 23–26 mm from the selected patch. These are
point-cloud/model overlaps, not native-contact measurements. Event 50 had an open
gripper; events 53 and 56 were closed, so permission is not a grasp certificate.

The implementation carries an existing target-point/finger-part contact from
the articulated query's reference configuration. Native convex narrow phase
recognizes points inside that same finger part at the reference pose, without
adding a distance threshold or buffer. It applies to moving target surfaces,
including points that the selected visual patch missed. New point collisions,
stationary obstacles, arm and palm remain checked. Free approaches retain their
original policy. No post-planner dense validation was added. The native cache
remains bounded to two indexed scenes.

The repaired requests produced complete plans in 15.37, 4.86 and 4.39 seconds.
Shared IK worker startup was another 5.01 seconds before the first request.
Detailed GPU setup, IK, projection and native connection timings are in the
report. These are numerical replays, not Luna success-rate measurements.

Push previews 48 and 51 are a separate issue: all four points had
`wrong_side_contact`, with force alignment about −0.0007. The requested motion
was nearly tangent to the fitted surface. They issued no physical motion. Do not
attribute those previews to the start-state rejection or claim this repair fixes
their surface/direction selection.

## State-49 deadline repair

`run_skill` previously derived defaults as `budget_seconds / 4` for contact search
and `budget_seconds / 24` for each subsequent plan. A 20–45 second total request
therefore allowed only 0.83–1.88 seconds for insertion/stroke planning.

Defaults are now independent: 30 seconds initial search, 5 seconds per insertion,
stroke or withdrawal. Each child is capped by the remaining whole-skill/episode
deadline; explicit caller settings are preserved. The 600-control episode limit
and 1 rad/s joint reference limit did not change.

Planning failures retain stage, requested/effective local allowance, elapsed time,
remaining skill time, nested timing and which deadline expired. A timeout is an
incomplete search, not proof of physical infeasibility. Insertion and withdrawal
timing are now retained on exceptions as well as success.

At closing49 r2 event 17, a paired reconstructed stroke query timed out under the
old 1.67-second default (1.70s elapsed) and planned under the independent 5-second
default (2.30s elapsed). Both used the same reconstructed scene and robot state
at control 474. The internal control-460 depth arrays had been pruned; its
occupancy refresh could not be reproduced. A retained observation at the same
control/joint state supplied final depth. This is explicitly not an exact replay
of the historical collision map. See `stroke-budget-r2/result.json` for substitutions.

## Approach feedback

In closing49 r1 event 25, the approach ended at control 388 with two usable
tracked correspondences and an unresolved pose; the scene still referred to
control 298. The automatic sequence then spent 40 controls closing at the old
grasp endpoint. It now returns `approach_target_pose_unresolved` with current
feedback before that stale automatic closure. Explicit gripper control remains
available. This change avoids an unsupported automatic continuation; it does not
repair tracking or establish that a new grasp will succeed.

## Validation and reporting

102 focused tests passed. An additional historical collision-preview test could
not run because `results/vlm-recovery-frozen-20261018/close_bottom_drawer-state017-vlm/opened/metadata.json`
is absent; it is excluded, not passed. New checks cover preserved initial contact,
rejection of a newly encountered finger point, palm/static collisions, deadline
scope and accounting, the actual compact skill flow with a 20-second budget,
stale approach feedback and accurate diagnostics. The former general
`collision_evidence.py` module is unchanged; new contact-specific functions live
in `contact_collision_evidence.py`.

Browser verification passed with repair addenda present: all videos available,
no video before Play, no page navigation on updates, selected run/expanded issue/
paused playback preserved, no console errors. The repair page is approximately
6.6 KB and static. The original watch page links it and shows evaluator addenda
without changing historical tool results. The complete new artifact directory
was approximately 15 MB after validation.

Native/GPU diagnostics used GPU 3, 8–16 GiB systemd memory limits, no swap and
whole-group termination. They have exited. Sampled I/O full pressure was 0–0.02%;
this was not a new continuous Luna capacity qualification.

## Next experiments

The original batch stopped on `sustained_host_io_pressure` during baseline r2
opening37. Seven baselines completed; the eighth is preserved as interrupted.
This is distinct from the earlier false-positive memory-pressure interruption.
The current repair changes numerical behavior, so the old study's runtime freeze
has ended. Do not silently resume or pool repaired-runtime outcomes into that
cohort. Use a new declared revision/cohort for physical and fresh Luna tests,
600 controls/1800 seconds, GPU 3 and the existing continuous resource watchdog.

No practice or transfer run has completed in the old cohort. Do not feed baseline
notebooks or evaluator repair diagnoses into the clean practice learner. Keep
provided tool knowledge separate from learner-authored procedural discoveries.

Reproduction entry points (source `scripts/server_runtime_env.sh` first):

```bash
python scripts/check_saved_contact_starts.py --source results/luna-cache-repair-validation-20261030/trials/baseline-r2-open_microwave-state013-600 --events 50 53 56 --plan-arcs --output results/NEW-REPAIR-RUN/saved-arcs.json
python scripts/check_saved_stroke_budget.py --source results/luna-cache-repair-validation-20261030/trials/baseline-r2-close_microwave-state049-600 --event 17 --geometry 14 --output results/NEW-STROKE-RUN
```

Run native/GPU commands in a bounded job, with `CUDA_VISIBLE_DEVICES=3` and
`MUJOCO_EGL_DEVICE_ID=3`. These entry points do not invoke Luna or execute generated
paths; they deliberately distinguish planning evidence from physical outcomes.
