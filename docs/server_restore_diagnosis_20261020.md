The active simulator, grasp/perception, cuRobo IK, and MC-VAMP stack is runnable on the server. Three saved-command replays reproduced the original success/failure outcomes. The two diagnosed failures differ: state 32 failed to reach its grasp pose and later moved without contact; state 44 acquired contact, made progress, then encountered planning failures and unsuccessful reacquisition. No manipulation algorithm was changed.

This is a development diagnosis, not a new transfer evaluation. No fresh agent trials or separately billed model calls were made. Optional DINO remains unavailable because the server has neither its authorized checkpoint nor an existing Hugging Face login. The GPU runtime differs from the original machine, so this does not establish complete reproduction of perception, planning, or the agent experiment.

**Baseline and evidence.** The starting tree was clean at `e45dac1772d15857b0207fb0432107c4da9d0ccd`, branch `codex/grasp-consistent-long-horizon`. A Git source archive and tracked-file hashes were saved before edits. The supplied 55,162,110-byte archive matched SHA256 `f3279e61b32f9bccf6fe6fd11cba657a5b94fecbad11369b1dd2c02fe6aa0371`. All 4,130 manifest entries were verified. Restoration added 3,271 absent files; 860 existing files already matched. Archived source and the working patch were not applied over Git.

All 15 episode videos, retained trajectories, prompts, frozen lessons, outcome records, historical runner inputs, and dependency inventories are present. The guide, successful-method file, public-library manifest, transfer-profile prompt, and frozen learned-memory hashes match their recorded values. A final check verified 3,290 archived evidence/input files without mismatches; all 477 measured snapshot files still match and have no write permission bits. See [archive restoration](../results/server-restoration-20261020/archive-restoration.json), [historical inputs](../results/server-restoration-20261020/historical-input-verification.json), and [final evidence verification](../results/server-restoration-20261020/final-evidence-verification.json).

Original RGB-D arrays and the diagnosed cases' collision-cloud arrays are absent. Some historical regression fixtures and the old full state-11 run are also absent; the approved public example backup exists. Replays use the retained joint references with a fresh simulator reset. They do not reconstruct the original sensor stream or rerun historical perception/planning.

The original finding remains guided practice with five failures followed by successes at 416 and 414 steps, then 2/4 transfer successes with memory versus 1/4 without. Memory is written conditional guidance, not a weight update. These are same-asset development findings. The older 35/35 result belongs to another pipeline.

**Environment and portability.** The server has Ubuntu 24.04, an AMD EPYC 9354, and four RTX PRO 6000 Blackwell GPUs with compute capability 12.0, driver 580.178.04. The active environments are project-local:

| Component | Restored version and evidence |
| --- | --- |
| Simulator | Python 3.10.22, MuJoCo 2.3.7, robosuite 1.4.0, NumPy 1.22.4, SciPy 1.10.1, Torch 2.2.2 CPU; original simulator pins retained |
| Grasp/perception | Python 3.10.22, NumPy 1.26.4, SciPy 1.10.1, GraspGen and MobileSAM at original Git revisions; original GraspGen checkpoint hashes verified |
| Grasp CUDA runtime | Torch 2.7.1+cu128 and torchvision 0.22.1+cu128; original Torch 2.1.0+cu121 failed a basic GPU operation, including a forced-PTX attempt |
| cuRobo | Original 0.7.8 Git revision; Torch 2.7.1+cu128 replaces 2.5.1+cu124; original Warp 1.18.0 and NumPy/SciPy pins retained |
| Native planning | Original MC-VAMP revision and unchanged numerical adapter source; rebuilt with GCC 13.3, Eigen 3.4.0, `-O3 -march=native -mavx2` |
| CUDA extensions | PointNet and cuRobo rebuilt using nvcc 12.9.86 and `TORCH_CUDA_ARCH_LIST=12.0`; CUDA target include/library paths supplied explicitly |
| Optional DINO | No checkpoint or existing HF authorization; its separate runtime was not installed without usable weights |

All five restored upstream repositories match their recorded revisions and have no tracked source modifications. Exact revisions, package freezes, compiler flags, and hashes are in [the build manifest](../results/server-restoration-20261020/server-build-manifest.json), [upstream verification](../results/server-restoration-20261020/upstream-verification.json), and the `*-installed.txt` files beside them. The CUDA 12.9 top-level include directory lacks the required headers; builds used `targets/x86_64-linux/include` and its library directory. The unavailable workstation-local `packaging` wheel reference was replaced with public `packaging==26.2` in cuRobo's derived installation requirements.

`pip check` passes for the simulator and cuRobo environments. GraspGen's upstream metadata still requires its old Torch/torchvision versions and several omitted training/visualization dependencies (`pickle5`, `scene-synthesizer`, `sharedarray`, `viser`). These were not installed solely to satisfy training metadata; actual inference and the integrated service passed. This exception is recorded in [dependency checks](../results/server-restoration-20261020/dependency-checks.json).

Infrastructure edits select the local simulator interpreter, honor an explicitly supplied CUDA location, locate Eigen, and fix standalone Codex runtime access under the restricted sandbox. The previous permission layout denied the runtime's ancestor directory and prevented the launcher from executing. The new layout denies private siblings while retaining the executable path. No contact permission, controller, physics, task predicate, numerical source, prompt, or frozen memory was changed. Torch/CUDA upgrades can change numerical proposals; matching original proposal distributions has not been established.

**Validation.** The unchanged baseline passed 79 relevant tests with eight missing-fixture skips. After restoration and the portability fixes, 82 passed with the same eight skips. Lint passed on the edited and added Python files. Fresh, explicitly labeled interface tests covered the missing local robot calibration rather than substituting new files into historical fixture paths.

| Check | Observed result |
| --- | --- |
| EGL rendering and RGB-D | Twelve camera views; finite metric depth, calibrated capture steps, float32 world/depth round trips, and pixel reprojection errors below 0.00045 pixels |
| Robot kinematics | 25 configurations against isolated MuJoCo FK/Jacobians; PoE errors around 1e-15; MC-VAMP maximum position error 1.16e-7 m and orientation error 2.61e-7 rad |
| cuRobo frames | 25 configurations accepted by existing checks; maximum TCP position error 3.77e-7 m, orientation error 7.46e-7 rad; joint order `joint1` through `joint7` |
| CUDA/model inference | Torch, scatter, neighbor search, sparse convolution, and rebuilt PointNet executed on GPU; MobileSAM produced a mask and GraspGen generated 200 finite candidates, retaining 100 |
| Planning/contact permissions | Free-motion and constrained arcs solved in `point`, `rigid`, and `jaw_axis` modes on declared synthetic geometry; stationary collision rejected while selected bounded finger contact remained permitted |
| Physical interface | Small joint motion and empty closure; then one complete service smoke on practice state 1: grounding, grasp planning, approach, closure, and one five-degree probe |
| Live tracking/provenance | Integrated probe used accepted RGB-D pose motion: -0.07204 rad, RMS 1.47 mm. Hinge refinement was rejected as unstable, separately from pose acceptance. Scene update correctly reported `rgbd_rigid_motion` and no contact-retention assumption |
| Isolation and episode rules | Actual sandbox denied private reads and sockets; frozen memory remained readable but unwritable. Unit checks covered persistence/budgets/success termination; successful replay stopped at the first native success with no retreat |

The integrated interface smoke ended after its declared probe at 270 control steps and 8.47° native opening. It is not a task success and is excluded from success rates. Its elapsed time was approximately 137 seconds, including startup. See [robot validation](../results/server-restoration-20261020/robot-validation-v3/summary.json), [service validation](../results/server-restoration-20261020/contact-service-smoke/compact-validation.json), and [isolation](../results/server-restoration-20261020/isolation-complete.json).

One calibration-only run produced a single frame that did not pass the movie retention check. The check preserved the frame; no evidence was discarded. The validation harness now retains the still image and losslessly compresses sensor arrays for zero-control-step sessions. Full replay video cleanup passed.

**Matched physical replays.** Each case below was replayed once with the original issued references, original gripper commands, seed 0, the same reset/60-step open hold/joint servo, a 600-step limit, and native-success stopping. No inference, new policy decisions, execution vetoes, or physics modifications were introduced.

| Original case | Server result | Steps | Maximum joint difference from archive | Maximum native-opening difference |
| --- | --- | ---: | ---: | ---: |
| State 32 with memory | Failure | 600 | 3.41e-7 rad | 0.002862° |
| State 44 with memory | Failure | 600 | 1.24e-11 rad | 1.79e-13° |
| State 2 with memory | Success | 497 | 1.30e-10 rad | 6.37e-13° |

Every retained command matched. State 2 stopped on step 497 with zero subsequent commands. State 32 had 11 per-step robot contact-set differences at a 1e-4 N reporting cutoff; the other two had none. Thus physical behavior closely reproduced this subset, with small numerical/contact differences in state 32. This is not bitwise reproduction or a fresh model comparison. See [replay comparisons](../results/server-restoration-20261020/replay-comparison.json).

![Archived and server native opening with robot contact intervals](../results/server-restoration-20261020/contact-diagnosis.png)

**State 32 timeline.** The first decisive discrepancy precedes closure. At the end of the approach, step 187, the TCP was 52.55 mm from the requested pose and the maximum joint reference error at that instant was 0.0805 rad. The second finger was pressing a microwave surface with approximately 35.4 N normal force. The door's episode maximum of 16.59° occurred during this approach, at step 173; it relaxed to roughly 1.35° by closure.

The gripper command at steps 188–227 holds the measured arm joints, as the existing implementation specifies. It therefore held the reached pose rather than completing the missed approach. Its small final error of 0.70 mm is relative to that new hold reference. At closure end, the pads touched each other and only finger 2 contacted the microwave. This is poor acquisition evidence, not a rule requiring a pinch: that contact still opened the door during later probes.

The first three arcs ended at 4.91°, 8.29°, and 10.90°. Accepted RGB-D pose estimates tracked those increments reasonably. The last robot–microwave contact occurred at step 266. From step 287 onward, repeated hand arcs moved without moving the door. Rejected tracking, typically `insufficient_moving_landmarks`, triggered `hand_motion_prediction`; four early fallback arcs each advanced the represented scene by approximately five degrees despite zero native progress. The tool's own contact-cloud free-space conflict rose to approximately 80%, 99%, and 100% on the first three such updates. Subsequent model revisions, rejected plans, and attempted reacquisition did not restore contact. The run ended at 600 steps, 10.90°, and approximately 1,736 seconds.

The supported failure chain is approach tracking error under contact, closure at the displaced pose, limited initial interaction, complete contact loss, and scene prediction continuing under an invalid retention assumption. The archive cannot establish whether a different settling duration or approach geometry would have prevented it. Complete trajectory issuance is not evidence that the target pose was reached.

**State 44 timeline.** This case reached its initial approach pose within 0.43 mm at step 207. By the end of closure, step 247, both pads contacted the microwave and the door was open 4.25°. Its first 15° point-constrained request produced 11.31° of actual door motion, reaching 15.56° at step 259; the final TCP reference error was 12.37 mm. Contact persisted, and the accepted RGB-D pose estimate measured approximately 10.52° of opening. Hinge refinement was rejected, separately from pose acceptance.

Ten-degree rigid/point requests then failed continuation/search checks. A five-degree point arc succeeded, reaching 19.06° at step 275; RGB-D measured approximately 3.49° versus 3.50° actual motion. The next ten- and five-degree requests failed. For the five-degree request, all four first continuation candidates at 2.5° were rejected for `1810/0`: the configured finger-1 collision volume against the represented stationary surface. This is a planning-scene collision result, not an execution veto or proof that no physical continuation exists.

The agent released at steps 276–315, losing the useful contact early in that hold. The door settled around 18.21°. Much of the remaining budget went to rejected plans, repositioning, and gripper holds without object contact. Later single-finger interaction was real and valid, but the final one-degree jaw-axis probe reached only 19.41° at step 600. The run lasted about 832 seconds, leaving wall time but no control steps.

State 44 therefore does not share state 32's initial acquisition failure or its hand-fallback divergence: all three executed arc updates used RGB-D pose motion. The unresolved issue is why the represented stationary geometry blocked continuation while useful contact remained. The initial estimated hinge was approximately 24.6 mm from the evaluator-side native hinge in the horizontal plane and remained unrefined in those requests; this is a possible contributor, not an isolated cause. Missing original collision clouds prevent distinguishing a true obstacle from stale/misassigned surface points with certainty.

**Successful comparisons and fix decision.** State 2 with memory reached its approach pose within 0.43 mm, closed with both pads in contact, retained contact through its manipulations, and succeeded at step 497. The archived state-44 control also succeeded, at step 600, and its first three arc updates used hand-motion fallback while the door actually moved. Removing fallback outright is therefore not justified by the failed state-32 episode alone. No numerical behavior fix was applied, and no improved success-rate claim is made. The only fixes are portability/infrastructure changes, supported by the tests and matched execution checks above. Detailed command timelines and error counts are in [saved contact timelines](../results/server-restoration-20261020/saved-contact-timelines.json).

The smallest justified next experiment is an evaluator-only prefix replay of state 44 through step 275, retaining fresh RGB-D and all point-role arrays before its rejected continuation. Reconstruct the same tool inputs and identify the stationary points responsible for the finger-1 rejection, alongside the analogous snapshot from successful state 2. Keep privileged native contacts/poses outside the planner and actor. This would test the scene-label/hinge hypothesis without changing checks or launching a new agent cohort. Fresh observations would be newly collected evidence, not the missing original arrays.

**Resume commands.** Run from the repository in a session with GPU access. Every output path below must be new; the scripts refuse to overwrite prior runs.

```bash
cd /common/users/dz323/vlm_tamp
source scripts/server_runtime_env.sh

# Regression checks; no physical episode or model calls.
python -m pytest -q tests/test_runtime_portability.py tests/test_recovery_learning.py \
  tests/test_persistent_episode.py tests/test_success_termination.py \
  tests/test_robot_sequence.py tests/test_shared_contact_tools.py \
  tests/test_surface_contact_tools.py tests/test_rigid_hinge_tracking.py \
  tests/test_grasp_scene_sync.py tests/test_storage_budget.py \
  tests/test_coupled_storage.py tests/test_timed_execution.py \
  tests/test_subscription_practice.py tests/test_subscription_measurements.py \
  tests/test_partial_pose_arc.py tests/test_short_motion_policy.py

# One saved-command replay; not a fresh agent trial.
python scripts/replay_compact_episode.py \
  --source results/recovery-learning-transfer-20261020/transfer/state032-with_learned_memory \
  --output results/server-followup-state032-replay

# Recheck calibration and native planning using the retained fresh fixture.
python scripts/validate_restored_robot.py \
  --fixture results/server-restoration-20261020/active-simulator-smoke \
  --output results/server-followup-robot-validation

# Optional repeat of the explicitly declared practice-state interface smoke.
python scripts/smoke_restored_contact_service.py \
  --regions results/server-restoration-20261020/practice-smoke-regions.json \
  --output results/server-followup-contact-interface
```

The original agent launcher still selects `gpt-6-luna`, high reasoning, and forced ChatGPT subscription authentication. Existing subscription login was verified. Do not treat a run with missing DINO as an identical original-tool comparison; obtain its authorized checkpoint and validate its isolated runtime first. No full-study/cohort launch is included here.

**Storage.** The retained restoration/diagnostic directory is approximately 0.742 GB, the restored measured study 0.153 GB, and the original archive 0.055 GB. The three runtime environments occupy approximately 19.4 GB of apparent file size; pinned GraspGen checkpoints occupy 1.07 GB and upstream sources/builds about 1.45 GB. Runtime directory totals are not physical block accounting across hardlinks. New-run retention removed about 696 MB and lossless sensor compression saved about 427 MB. The project-local mamba cache namespace decreased from 928 MB to roughly 2 KB; shared hardlinks mean that is not a claim of 928 MB of newly freed disk blocks. No original evidence or sole required observation was deleted. The original archive, baseline source archive, hashes, prompts, memory, commands, outcomes, videos, and necessary fresh observations remain available.
