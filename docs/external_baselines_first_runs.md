# Public grounding implementations: first runs

Checked and executed on 2026-10-12. This study covers the four works discussed with the user: RoboPoint, ReKep, ViLaIn-TAMP and GroundedPlanBench. **Three have usable public implementation components; GroundedPlanBench's official code link still says “Code (Coming Soon).”** These runs are component tests and an adapted RGB-role comparison, not reproductions of complete published robot systems.

## Source availability

| Work | Verified release | Source revision used |
|---|---|---|
| [RoboPoint](https://github.com/wentaoyuan/RoboPoint) | Public code, inference/evaluation entry points, and pretrained checkpoints | `206cb1e086fc3960a8673725a8540766655cd9c5` |
| [ReKep](https://github.com/huangwl18/ReKep) | Public OmniGibson demo, keypoint proposer and constraint generator. The README explicitly omits its real-world keypoint/mask tracking and SDF perception pipeline. | `63c43fdba60354980258beaeb8a7d48e088e1e3e` |
| [ViLaIn-TAMP development branch](https://github.com/omron-sinicx/ViLaIn-TAMP/tree/jeremy_dev) | The default main branch contains a README and data. Public `jeremy_dev` and `shirai_dev` branches contain interpreter code. The inspected release is not a standalone complete robot/TAMP system. | `9e36ee0df43bf7ef018c5bb0a78c3c7b6ad72669` (`jeremy_dev`) |
| [GroundedPlanBench](https://groundedplanning.github.io/GroundedPlanning/) | Public paper and project website. The official code button says Coming Soon; the account's repository is the project website. No runnable benchmark/annotated-data release was found through these official channels. | Availability snapshot saved in `results/external-baseline-audit/` |

ReKep's DINOv2 repository revision is saved in `results/external-baseline-audit/source-status.json`. Third-party source is under ignored `third_party/`; checkpoints and environments are under ignored `.runtime/`.

## RoboPoint on the shared RGB scenes

The actual released `wentao-yuan/robopoint-v1-vicuna-v1.5-13b` checkpoint is downloaded at model revision **`508ed28687caeeb923e1d7d5905e37dccc16c991`** and loaded through the upstream model loader. The RTX 4090 has insufficient spare memory for a complete 13B FP16 model, so this run uses its documented **8-bit option**.

The same twenty task images, instruction-derived roles, goal compiler and frozen `rgb-semantic-1.0` evaluator as the Mini generalization test are used. RoboPoint gets one separate question per role, using untouched RGB and the instruction/role phrase; the model's native preprocessing resizes/pads the image for its vision encoder. It returns lists of normalized image points. The **first returned point** is selected according to the prespecified protocol, without masks, ranking, coordinate snapping or label-based rescue.

| Shared RGB-role metric | Saved GPT-5.4 Mini | RoboPoint adapter |
|---|---:|---:|
| Complete symbolic final-goal agreement | **8/20** | **7/20** |
| Correct individual roles | 25/39 | 23/39 |
| Compiled declarations | 20/20 | 20/20 |
| Queries | 20, one per task | 39 local queries, one per role |
| Wrong-instance points | 3 | 9 |
| Background/occluder points | 11 | 7 |

Four tasks are correct in both arms, three only in RoboPoint, four only in Mini, and nine in neither. A one-task difference on this small sample is not a justified model ranking. This adapter measures instance binding on LIBERO rather than RoboPoint's original spatial-affordance evaluation. Quantization, greedy decoding (instead of the upstream evaluation script's default temperature 0.2), the custom role queries, and selecting one point from a returned set are material comparison limitations. No claim about the paper's reported accuracy is made.

The first local setup run used an incorrectly short 128-token limit. This truncated many native multi-point lists and is retained as `results/robopoint-generalization/`, **excluded from the comparison**. The corrected protocol uses the official **1,024-token maximum** and is saved as `configs/frozen_robopoint_native_cap.json`; the results are `results/robopoint-native-cap/`. Both runs and raw responses remain intact. There was no fine-tuning and no API charge for either local model run.

## ReKep keypoint and constraint components

The original keypoint proposer was run on a real rendered RGB-D observation for `libero_goal-08-state005-seed0` (“put the bowl on the plate”). RGB is verified pixel-identical to the earlier frozen image. Sensor depth is converted to world points using the official pinhole, camera-axis and MuJoCo depth formulas. MobileSAM supplies masks from RGB alone; simulator instance labels are withheld from proposal generation.

The upstream DINOv2/PCA/k-means/mean-shift proposer produced **33 keypoints**. The adapter loads only the pure bounds helper it uses, avoiding unrelated Open3D/OmniGibson imports; the published proposer itself is unchanged. Workspace bounds are adapted to LIBERO and parallelism is reduced to four workers. This is an RGB-D/predicted-mask component test, not the RGB-only Mini condition or the native OmniGibson setup. The background mask remains subject to the upstream area heuristic; predicted segments are not guaranteed to be complete rigid-object masks.

One live call to the original constraint generator produced **two stages and three parseable functions**, using its unchanged prompt. The original `chatgpt-4o-latest` alias is [retired](https://developers.openai.com/api/docs/deprecations); the available GPT-4o-2024-11-20 snapshot is used. The transport is ported to the bounded official SDK and the output cap is 1,536 rather than the original 2,048 tokens.

The generated grasp index **23** independently matches the bowl mask. However, the draft describes index **24** as the plate, while that pixel belongs to **no reference object**. Index **20** matches the actual plate. The resulting placement cost therefore binds the target incorrectly despite valid Python and in-range indices. This one adapted example cannot establish the published method's general failure rate.

Generated functions are only saved and AST-parsed. They are **not executed**, numerically certified, optimized into motions or sent to a robot. No physical constraint-accuracy or execution success score is claimed.

Artifacts: `results/rekep-component-input/`, `results/rekep-keypoint-smoke/`, `results/rekep-constraint-smoke/`, and `results/rekep-component-summary.json`.

## ViLaIn interpreter component

The public development branch's actual detector/initial-state/goal functions were exercised on its native `cooking/moving/problem2` image and instruction: “Move the cucumber to the plate.” Reference object lists, fixed-object boxes and initial states are withheld from inference. The method's predefined cooking category list and PDDL domain vocabulary are supplied as upstream domain knowledge, not as an assertion of which instances are in the scene.

Portability changes are explicit:

- The unused ROS import is removed when loading the interpreter.
- Its hard-coded private lab proxy is bypassed using the official OpenAI SDK.
- GPT-4o substitutes for the original local Qwen2.5-VL detector in this smoke test.
- A structured detector output format is added; native prompts and parsers remain.
- The native fixture rescaler assumes a nonempty provided-box list and raises `min() arg is an empty sequence` when it is empty. A zero-fixture guard returns an empty list instead of injecting oracle fixtures.

The empty-fixture setup failure caused four detector requests through the native retry loop. These actual requests/replies and the failed result are retained. The corrected run reuses the **first** actual detector reply, with exact request/format checks, rather than choosing a favorable reply or paying for another detection. Two subsequent real calls generated the initial state and goal.

The completed pipeline produces a cucumber/plate object declaration and the desired `(At cucumber plate)` goal. Its initial state incorrectly includes `(At cucumber plate)` and `(isNotFree plate)`, while the held-back native reference places the cucumber on the tray. It therefore declares its own requested goal already achieved. It also refers to `a_bot` in initial predicates without declaring a robot in its image-derived object list. This stricter RGB-only port omits the fixed environment information assumed by the native setup, so the result must not be represented as a faithful complete-system reproduction or general ViLaIn accuracy.

Artifacts: `results/vilain-component-smoke/` (preserved setup failure), `results/vilain-component-smoke-fixed/` (completed component calls), and `results/external-component-api/` (actual SDK arguments and raw responses).

## Cost, setup and validation

All live external component tests share a **$0.10 estimated admission limit**. Actual usage is **seven calls / $0.03400750**: four detector setup attempts, two ViLaIn state/goal calls, and one ReKep constraint call. All use GPT-4o-2024-11-20, at $2.50/M input and $10/M output checked against [official documentation](https://developers.openai.com/api/docs/models/gpt-4o). There are no unknown-cost reservations. Invoices remain authoritative. Project paid-inference totals are now **301 calls / $3.93033260 conservative estimated usage**, with local RoboPoint queries tracked separately.

An isolated Conda environment is at `.runtime/envs/robopoint`, with Python 3.10.22, Torch 2.1.2+CUDA12.1, torchvision 0.16.2, Transformers 4.43.1, NumPy 1.26.4 and bitsandbytes 0.43.3. Exact observed runtime versions are recorded in `results/external-baseline-audit/robopoint-runtime.json`. The main LIBERO environment is unchanged. An overlapping setup installation initially introduced newer Torch files; those stale files were preserved under `.runtime/recovered-installation/` and the isolated environment was cleanly repaired before the benchmark.

**178 tests pass**, including real LIBERO rendering, with twenty existing dependency warnings. New tests verify literal-only point parsing and API admission, price/model locking, streaming adaptation, accounting, and uncertain-cost reservations. New source/scripts pass lint. Saved evidence is independently replayed by `scripts/verify_external_baselines.py`; prior experiment integrity is checked by the original verifier. Browser rendering is not independently tested.

The report is `results/external-baselines-report/index.html`, with twenty paired Mini/RoboPoint galleries, raw point lists, exact grades, ReKep's numbered input and constraint draft, the ViLaIn scene/specification, source links and explicit reproduction limits.

Rebuild and verify saved results with no new inference:

```bash
conda activate vlm-tamp
python scripts/report_external_baselines.py
python scripts/verify_external_baselines.py
python scripts/verify_rgb_role_generalization.py
```

A fresh local RoboPoint run uses the corrected frozen protocol and a new result directory:

```bash
conda activate /home/duo/Rutgers/ARC/random/vlm_tamp/.runtime/envs/robopoint
python scripts/run_robopoint_baseline.py --output results/robopoint-new
```

The full OmniGibson robot demo, ROS/TAMP execution, real-world perception stack and unreleased GroundedPlanBench assets remain outside these completed grounding-component tests. ST-pRRTC is unchanged.
