# All prior physical tasks: frozen regression

The user requested testing every previously physically attempted task, including bowls, drawers and microwave doors. The cohort contains seven task types on official initial states 1, 11, 21, 31 and 41, seed 0: 35 episodes. It uses five settling actions and at most 600 control steps per episode. All manifest failures count, including failures before motion. These tasks/states have been inspected previously; no held-out generalization claim is made.

The primary frozen cohort is `results/all-prior-tasks-frozen-v3-20261015`, version 1.2. The earlier integration run at `results/all-prior-tasks-frozen-20261015` used an older observation reader in the non-microwave adapters: its RGB/depth were correct, but synchronized known-robot depth arrays were dropped. This left robot surfaces in the stationary obstacle map. The corrected reader removed about 19,000 such samples in each saved bowl/middle-drawer fixture and restored clear initial poses. A second integration trial at `results/all-prior-tasks-frozen-v2-20261015` completed bowl placement, release and native task success at step 331, but post-run cleanup failed because its grasp-retention log shared the storage-retention filename. Its completed result/video were recovered by renaming the physical log and finishing cleanup. The final adapter uses `grasp-retention.json` for physical retention and `retention.json` only for storage.

Regressions explicitly check robot-depth preservation and repeated cleanup with a separate physical retention log. All completed integration attempts are retained and reported separately; unstarted initial-manifest cases were cancelled. These corrections change no grasp-search budgets, task goals, planner parameters or contact permissions.

| Task | Official suite | Task name | Adapter |
|---|---|---|---|
| Bowl onto plate | libero_goal | put_the_bowl_on_the_plate | Point goals, cuRobo IK continuation |
| Open middle drawer | libero_goal | open_the_middle_drawer_of_the_cabinet | Line-constrained IK continuation |
| Open top drawer | libero_90 | KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet | Line-constrained IK continuation |
| Close top drawer | libero_90 | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet | Line-constrained IK continuation |
| Open microwave | libero_90 | KITCHEN_SCENE7_open_the_microwave | MC-VAMP partial-pose hinge planner |
| Close microwave | libero_90 | KITCHEN_SCENE6_close_the_microwave | MC-VAMP partial-pose hinge planner |
| Turn on stove | libero_90 | KITCHEN_SCENE3_turn_on_the_stove | Coupled control-rotation IK continuation |

## Frozen choices

All arm execution uses the same official joint-position controller at kp 300, following a reference-time schedule. Each issued segment finishes completely before another perception/planning decision. There are no execution-time collision, constraint-residual, tracking-progress, retention or native-success vetoes. Physics, actuator limits and physical gripper closure remain active. No object snapping, assisted attachment or scene-state assignment is used.

The planner adapters differ explicitly. The existing MC-VAMP adapter supports microwave-door hinge arcs; it is not silently reused as a sliding-drawer or payload planner. The other adapters use the existing cuRobo multiple-IK and sampled observed-surface collision/payload admission code. Bowl motion has no additional geometric path constraints. The bowl stages have ordinary lift/transport/place goals. Line and rotation constraints apply to articulated manipulation. This is a pipeline regression across task families, not a single-planner speed comparison.

There is fresh grasp/IK selection in every episode, without replaying selected per-state grasp poses or old command trajectories. GraspGen generates 200 Panda candidates and retains the top 100. Its content-addressed cache includes current cloud bytes and checkpoint hashes, and may reuse identical proposals. The frozen non-microwave selector considers the 24 closest detector candidates, their four existing frame variants, up to 8 IK roots per pose, a total 96-posture budget. Bowl/middle-drawer selection also uses a 120-second bound between queries; the existing top-drawer/stove selector uses the posture-count bound. Individual queries are allowed to finish. Different poses are visited at the same IK rank first. The microwave selector retains its 24-pose/short-probe policy, 32 terminal proposals and 20-second arc-search budget. Search failures record the budget/geometry limitation; they do not prove physical infeasibility. These budgets differ from some older larger-search experiments.

The task roles come only from RGB. Legacy bowl/middle-drawer ReKep-style cached choices are rebound to current depth after selected RGB, landmark and selected-surface agreement checks. Old raw snapshots were intentionally pruned; retained RGB, landmark arrays and selected source clouds provide the independent evidence. The integration smoke check found byte-identical selected RGB and zero landmark/surface displacement for state 1 of both tasks.

Articulated grounding reuses matching cached Luna High decisions. The three completed 12k microwave-closing responses are reconstructed using their exact original RGB views, avoiding repeats. New scenes use the existing numbered RGB fill-in template, Luna High, a 12,000-token output bound and no automatic retries. The same durable cumulative API ledger and $2 dollar cap remain active. The old internal 30-scene ceiling is extended to 65 records solely to cover this explicitly requested cohort; all prior charges and pending reservations remain unchanged and a migration record is retained. The cohort snapshots the ledger before any new requests. Historical ledger consumers with the older scene-count policy must use the new cohort gateway for new requests; no monetary budget is reset.

The simulator's object identities, contact labels, axes and state remain evaluation-only. Known Panda kinematics and robot-only depth rendering are allowed. Only measured occupied surfaces are checked; hidden geometry and continuous clearance are not certified. Grasp retention and incidental contacts remain diagnostics even when the native task succeeds.

## Reproduction

Use the existing research, cuRobo and simulator environments unchanged:

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python scripts/run_all_prior_tasks.py \
  --output results/all-prior-tasks-frozen-NEW
```

Sources/configuration are frozen before the first episode. The runner refuses overwrites and checks the source hashes before every episode. `--resume` skips completed episodes only when the manifest and source hashes are identical; an incomplete episode needs explicit diagnosis first. All historical protected baseline source hashes remain checked.

Independent simulation workers partition the manifest into states 1/11, state 21, and states 31/41. Additional workers import and verifies the exact parent-frozen policy functions/configuration. GraspGen resets seed 0 for every generation. Its runtime cache is private to prevent concurrent cache writes, and is deleted after its episodes finish; every episode retains the exact proposals and provenance. The same locked cumulative API ledger is shared. This is ordinary parallel simulation, not a change in planner policy; operational timings are measured under concurrent load. A partition protocol records the worker source hash and assigned episodes.

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python scripts/run_frozen_task_partition.py \
  --root results/all-prior-tasks-frozen-v3-20261015 --states 31 41
```

The original main loop refuses to overwrite externally completed cases when it reaches state 21. After all workers finish, run the main command with `--resume` and the same output path to aggregate all completed cases without executing duplicates. Its last resume elapsed time is an accounting-pass duration, not total experiment wall time.

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
.runtime/envs/rekep-libero/bin/python scripts/report_all_prior_tasks.py
```

Use `--partial` for an in-progress view. Every episode is compacted after simulator and IK workers exit: retain essential initial/latest observations, lossless planning geometry/paths, selected keypoints/functions, outcome/contact logs and a verified video/frame index; remove redundant raw observations and JPEG sequences. The report references retained movies directly instead of encoding duplicates.

41 focused checks passed before the corrected cohort, including full timed execution, intended finger contact, gripper convex geometry, moving-point roles, RGB-D hinge estimation, preservation of synchronized robot depth, separate physical/storage retention logs, the seven-task manifest and preservation of costs during the internal scene-count extension.


## Resource interruption

A concurrent system out-of-memory event killed the original main worker (exit 137) while preparing `plan-09` for microwave opening state 11. The episode had reached 67.8 degrees at control step 356. Its 296 joint commands exactly match all saved references (approach, closure and nine completed contact paths), so no issued trajectory was cut short. The episode is archived under `infrastructure-interruptions/open_microwave-state011-attempt1`, with recovered tracking from the unchanged native action log and recorded relocation of artifact-only absolute paths. Its temporary data were compacted.

This is an infrastructure interruption, not a completed policy failure. The same official state is rerun using the exact frozen policy/configuration; the interrupted attempt remains separately reported. Subsequent concurrency is capped at two workers, and the restarted process limits joblib CPU workers via `LOKY_MAX_CPU_COUNT=2`. No grasp, contact, motion or API parameters are changed. Operational timing measurements therefore reflect differing concurrent system load.
