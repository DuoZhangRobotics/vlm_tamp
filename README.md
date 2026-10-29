# VLM grounding for constraint-aware TAMP — Stage 1

## Luna contact recovery (29 October 2026)

The current experiment is [clean practice with 600 controls](docs/clean_start_600_20261029.md):
one learner starts without inherited task memory, then fresh agents are evaluated
with and without its frozen notes. Evaluator notebooks are retained separately.
Open `results/clean-start-600-20261029/watch.html` with VS Code Live Preview.
The completed pilot recorded 5/6 practice successes and 2/3 fresh successes with
memory versus 1/3 without. Opening state 13 remains unreliable in fresh actors;
these small counts do not establish a general success-rate improvement.

The current experimental branch adds push/hook skills, observed-motion joint
tools, configurable recovery budgets and corrected joint-reference tracking.
Read the [implementation and evidence handoff](docs/restriction_fixes_20261029.md)
and [restriction audit](docs/restriction_audit_20261029.md). The preceding study's
page remains at `results/restriction-fixes-20261029/watch.html`.
Versioned learner memories and their provenance live under `prompts/learned/`.

The experiments retain zero added collision padding and physical simulator
contacts. Luna chooses strategies from public observations; native object state
is reserved for offline grading. Local repeatability, fresh actor outcomes and
memory transfer are reported separately. Acceptance as a replacement for the
original seven-task reference still requires the paired regression described below.

## Episodic memory prototype (19 October 2026)

`scripts/run_subscription_remembered_agent.py` starts a fresh Luna/high subscription
session with the registered public episodic library enabled. It preserves the
recorded failure, corrected procedure, images, geometry, and tool calls, and
requires one physical attempt with fresh plans. `configs/subscription_memory.json`
pins the public library, prompt, and shared guide. The library is staged read-only
and actual retrieval is checked before simulation starts. New episode records are
saved, but do not silently rewrite the shared library.
The active data lives in `data/agent_memory/luna-microwave-v1`, outside transient
results; keep the profile's `protected_inputs` during cleanup.

This is an experimental microwave-opening entry point, not a replacement for the
accepted seven-task reference below. See `docs/subscription_episodic_memory.md`
and `results/subscription-episodic-memory-v2-20261019/index.html` for the separate
exact-replay, same-scene, and transfer comparisons. Existing numerical tool priors
remain inherited; this experiment tests experience reuse.

## Current execution experiment (18 October 2026)

The **accepted reference remains the original successful task-family pipeline**, represented by the frozen runs in `results/success-first-report/evaluation.json` (35/35 observed successes). Use its recorded source snapshots and configurations for reproduction; the current working tree contains later changes. `configs/execution_comparison_policy.json` records the reference versions and the requirements for accepting a replacement: reproduce the original cases, preserve all their successes, then demonstrate improvement on matched harder cases. The full paired comparison has not yet run.

The candidate runner `scripts/run_shared_contact_feedback.py` remains **experimental and unpromoted**; saved evidence is in `results/shared-contact-feedback-report/index.html`. Its `nominal` variant means the **new pipeline with standard joint servo feedback and between-segment replanning, but without the added outer reference correction**. It is an ablation within the candidate, not a replay of the original full pipeline. The original 35/35 regression covered seven tasks; bottom-drawer opening and closing were outside that set. The earlier 4/4 bottom-opening diagnosis bypassed planning-time scene collision admission. The report keeps these settings and task sets separate.

The newer `scripts/run_strategy_feedback.py` pilot exposes pinch, press-and-drag, push, continued motion, reobservation and finishing as declarative choices. Luna high receives measured feedback and an exact image sheet after each operation, with a 32,000-token decision ceiling and one synthetic tool-use example. Its selected mode is enforced without automatic fallback. Both the VLM and the geometry-heuristic control use the same numerical tools, reactive controller, five decision rounds and 600 total control steps, with no scene resets between attempts. `configs/strategy_feedback_pilot.json` declares the matched cases; `scripts/validate_strategy_feedback.py` verifies actual commands, input pairing and absence of evaluation labels in feedback. This first implementation covers sliding drawers. Contact templates, upright geometry and completion priors remain engineered; autonomous geometry revision and cross-episode rule learning are not implemented. Results belong in `results/vlm-strategy-feedback-report/index.html` and do not promote the candidate over the original reference.

The first frozen strategy pilot scored 1/5 for Luna and 2/5 for the heuristic. A subsequent, separate two-episode follow-up adds one bounded grounding correction: the completed response and validator error are returned to Luna with the original image, with no manual pixel transformation. Luna corrected the invalid image coordinates, but both physical closing attempts still failed. Those follow-up outcomes remain separate from the frozen comparison. Current initial grounding and correction requests honor the runner's paid-inference opt-in, share the experiment ledger, and do not automatically retry network failures or incomplete responses.

The API allowance is **$2 per experiment run**, as clarified by the user on 18 October. All episodes, variants, parallel cohorts/workers and resumptions of that experiment share one ledger. Use the same `--api-run-id EXPERIMENT_ID` for every cohort, or set `VLM_TAMP_API_RUN_ID=EXPERIMENT_ID` for workers using the shared resource classes. A single-output experiment otherwise derives a stable ID from its output directory. `api-run.json` preserves that binding; resuming cannot replace it with a fresh balance. Ledgers live under `results/api-runs/`. Recorded cumulative spending is informational, exact cached responses are free to reuse, and unknown outcomes retain their reservations without automatic retries. Historical ledgers and frozen source snapshots are preserved. The current policy is recorded in `configs/api_budget_policy.json`; older spending statements below describe the accounting used at the time.

## Earlier Stage 1 work

This repository evaluates instruction, object, goal, stage, and manipulation-constraint grounding on **real LIBERO scenes**. It is the first experimental stage of a TAMP system whose existing motion planner is **ST-pRRTC**. It does not implement or integrate a motion planner, robot policy, or controller.

The first implementation includes official task loading and headless rendering, exact BDDL goal parsing, Pydantic predictions, OpenAI and Gemini SDK adapters, four initial input settings plus a separate object-label oracle condition, offline inference, reviewed annotation tooling, evaluation, HTML overlays, retries, caching, and cost admission controls.

## Active task scope

The next geometry experiment replaces the three VLM-selected face points with a fixed, sensor-only local plane fit around an unchanged drawer seed. `src/vlm_tamp/drawer_local_plane.py` uses a 32-pixel ROI, handle exclusion, depth-edge filtering, deterministic RANSAC and predeclared consensus/extent/seed/ambiguity gates. Its policy was fixed using synthetic tests before scoring real predictions; no simulator part mask, joint axis or object pose enters fitting, and neither selected pixel is moved.

The zero-call replay of the four saved Luna-high source/handle predictions produces four local estimates with **0.25–0.35° axis error**, compared with about **0.11°** for the preserved three-point geometry. Local support is only **51–54%** the intended middle drawer body because neighboring fronts are coplanar. The old five-anchor criterion remains **2/4**; the new two-anchor-plus-axis criterion is **4/4**, explicitly a changed contract rather than repair of old answers. A separate ≥90% reference-pure-support diagnostic is **0/4**. See `results/drawer-plane-ablation-report/index.html`. This stage adds no API cost.

A fresh test then uses only two point blanks plus joint/motion/constraint choices, with **Luna high reasoning** and the same frozen local fitting policy. Four predetermined captures vary left/right camera position or rotate the fixed cabinet by **±20°** and partially open the middle drawer to **q=−0.06 m**. These are labeled simulation state interventions, not original benchmark initial states or robot executions. Camera cases bind the correct drawer and handle; both rotated/partial-open cases select the wrong drawer. All four plane estimates still have low axis error (**0.16–0.50°**), demonstrating that numerical accuracy on parallel surfaces does not establish correct task grounding. Middle-body support purity is **54.0%, 48.1%, 0.70%, 0.04%**, respectively; none meets the 90% diagnostic. See `results/drawer-geometry-stress-report/index.html`.

New API usage is **four calls / $0.002404 estimated**; cumulative spending in the authorized budget sequence is **$0.154725 / $2**. Captures pass exact RGB/depth/segmentation repeat and inverse-calibration checks. The new suites pass 95 tests plus Ruff. Offline replay checks four unchanged cached replies, 13,986 support pixels, 967 protected artifacts and 49 report assets. The stress replay checks four real replies, 12,143 support pixels, three distinct reference axis directions, four line/Jacobian checks, 1,027 protected artifacts and 55 report assets. No contact, TCP attachment, numerical opening target or planner execution is certified.

Rebuild and verify these saved results without further API calls:

```bash
python scripts/report_drawer_plane_ablation.py
python scripts/verify_drawer_plane_ablation.py
python scripts/report_drawer_geometry_stress.py
python scripts/verify_drawer_geometry_stress.py
```

The immediate remaining issue is part selection for partially opened drawers and support localization across coplanar parts. Symbolic PDDL goals/actions can complement the typed geometric constraint records, but do not infer visual identities or certify motion feasibility; no PDDL planner integration was added in this experiment.

The latest matched drawer experiment uses **GPT-6 Luna and GPT-6 Astra with high reasoning**, the same four frontal RGB observations, the unchanged prompt/schema/compiler/grader, standard service, no retries and a shared 6,000-output-token cap. Both models return four completed forms and correctly bind the middle drawer and handle in **4/4** cases, versus **0/4** for the saved Luna-none run with its older 1,000-token cap. All five landmark bindings are correct in **2/4** Luna-high cases and **0/4** Astra-high cases; the remaining error is always the third face point. All eight estimated axes pass the five-degree proxy check, which does not repair a wrong face binding or prove execution. The four observations share one cabinet pose and joint axis, so this is a development diagnostic rather than a population model ranking.

Estimated new usage is **$0.002523 for Luna / $0.140600 for Astra**, **$0.143123 total**. Cumulative usage in the authorized budget sequence is **$0.152321 / $2**, including previous Luna trials. Actual response metadata confirms `reasoning.effort=high` for every call; no responses truncate and no unknown charges remain. Luna-high is the economical configuration to use for subsequent grounding development. Earlier frozen none/medium configurations remain available for exact historical reproduction. The current report is `results/drawer-models-high-report/index.html`; its successful verifier replays all eight responses, unchanged numerical compilation, costs, four matched sensor captures, eight line equations/Jacobians, 837 protected artifacts and 87 report assets. The new comparison checks pass 63 relevant tests plus Ruff. Grasp attachment, opening distance, joint-space compilation and robot execution remain unresolved.

Rebuild and verify this comparison without API calls:

```bash
python scripts/report_drawer_models_high.py
python scripts/verify_drawer_models_high.py
```

Ordinary rigid-object pick-and-place uses `src/vlm_tamp/pick_place_scope.py` and
`configs/prompts/pick_place_blanks_v1.txt`: five landmarks (source, grasp,
opposing contacts, target) plus grasp mode. There are no object-axis blanks,
path-constraint choices or extra task-specific path relations. Placement goals,
normal collision/grasp requirements and unresolved placement offsets remain.
Eight saved ordinary Luna predictions are reused with this reduced output scope;
synthetic upright variants are excluded and no new pick-place calls were made.

Opening the middle drawer uses five different landmarks (drawer, handle, three
front-face points), joint type, motion and constraint template. Local calibrated
depth estimates a front normal. A typed `point_on_line` residual with analytic
pose Jacobians applies during **pull only**, after verified handle attachment.
The front-normal-to-slider-axis assumption is explicitly unverified; the
constraint uses an abstract handle-tracking point until robot TCP, grasp and
attachment transforms are bound. Fixed grasp orientation, joint travel limits
and the metric stopping threshold for the word “open” remain unresolved.

The original oblique camera produced four valid/correct-template forms but zero
complete drawer/handle bindings or numerical pull lines. The RGB-selected fixed
frontal camera retry preserves those failures and improves plane/axis estimation:
three of four direct-RGB cases yield axes within about 0.11 degrees of the
held-back simulator axis. Complete drawer/handle binding is still zero of four;
one direct case has the correct handle and axis while its separate drawer-point
binding is wrong. DINO proposals omit the required handle in all original
observations and two frontal observations. Missing-proposal calls are skipped
by an explicitly evaluation-assisted **development setup audit**, not a deployable
perception gate. References never enter API requests or repair selected pixels.
The two eligible frontal DINO calls also have no complete drawer/handle binding.
This is a small camera intervention study, not a model/backbone ranking.

Before the matched high-reasoning comparison, drawer usage was **10 actual calls / $0.00236025 estimated**. Together with
the earlier Luna comparison, cumulative Luna usage is **$0.00919800** under the
user-authorized **$2 ceiling**. All ten forms are valid and use no reasoning
tokens. Robot execution is not tested. The preserved frontal-view none-reasoning report is
`results/task-grounding-frontview-report/index.html`; the preserved initial
report is `results/task-grounding-scope-report/index.html`.

The new schema/compiler/scope checks pass 126 tests; original numerical tests
also pass. Captures pass deterministic rendering and inverse-calibration checks.
Rebuild and replay saved results without new API calls:

```bash
python scripts/report_task_grounding_scope.py
python scripts/verify_task_grounding_scope.py
python scripts/report_task_grounding_frontview.py
python scripts/verify_task_grounding_frontview_v2.py
```

The second front-view verifier repairs only an overly strict HTML wording
assertion in the preserved first verifier. Empirical replies, numerical
compilation, grading and report contents are unchanged.

## Tested environment and installation

The actual development session used Ubuntu **22.04 under WSL2**, an RTX 4090, and a new Conda environment named `vlm-tamp`. Python **3.10.22** was selected because upstream LIBERO pins NumPy 1.22.4, which predates Python 3.11 support. Robosuite 1.4.0 and MuJoCo 2.3.7 are pinned; no legacy simulation upgrade or policy-training stack is required. Torch is CPU-only and used by LIBERO's task/initial-state API. The renderer works headlessly with EGL; GPU inference is unnecessary.

From a fresh checkout:

```bash
cd /home/duo/Rutgers/ARC/random/vlm_tamp
source ~/miniforge3/etc/profile.d/conda.sh
bash scripts/setup.sh
conda activate vlm-tamp
python -m pip check
```

`scripts/setup.sh` creates a separate environment, checks out official LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, and installs the minimal simulation and API dependencies. Initial-state files and assets are already in the official checkout; full demonstration downloads are unnecessary. Existing Conda environments are unaffected.

`environment.yml` records the intended environment, `requirements-sim.txt` explains simulation pins, and `requirements-lock.txt` / `conda-linux-64.lock` capture the exact tested environment. To reproduce the resolved package set in another new Linux environment:

```bash
conda create -y -n vlm-tamp-repro --file conda-linux-64.lock
conda activate vlm-tamp-repro
python -m pip install -r requirements-lock.txt
bash scripts/setup.sh vlm-tamp-repro
```

The CPU Torch wheel in the simulation requirements targets Linux x86-64 and Python 3.10. Other platforms need a compatible wheel. Read `docs/setup.md` for renderer and dependency details.

## Validate LIBERO first

```bash
python scripts/smoke_libero.py --output data/smoke --gl egl
vlm-tamp validate --dataset data/smoke
RUN_LIBERO_INTEGRATION=1 python -m pytest -q -m integration
```

The smoke test loads an official task and initial state, renders a 256×256 RGB image, saves visible object masks, and extracts BDDL goals, object metadata, camera metadata, and joint kinematics. The integration test performs two independent collections and checks identical RGB bytes. EGL works with `DISPLAY` unset. The collector flips Robosuite's OpenGL image convention to a top-left origin for both RGB and segmentation.

## Collect the small benchmark

```bash
vlm-tamp collect --suites libero_goal libero_spatial \
  --tasks 0 1 2 3 --states 0 1 --seed 0 --gl egl --output data/pilot
vlm-tamp validate --dataset data/pilot
vlm-tamp contact-sheet --dataset data/pilot --output data/pilot/contact-sheet.png
```

This creates 16 observations across eight tasks. Inspect the contact sheet before using the dataset. The same interface supports `libero_object` and `libero_10`, for example:

```bash
vlm-tamp collect --suites libero_object libero_10 \
  --tasks 0 --states 0 --seed 0 --output data/additional-suites
```

Each dataset has `records.jsonl` plus example directories containing `rgb.png`, `record.json`, `initial_state.npy`, `segmentation.npy`, and visible instance masks. The manifest separates `model_inputs`, `reference`, and `metadata`. Model requests use an explicit allowlist and do not include task names, filenames, BDDL goals, references, annotations, or simulator internals.

## Run the complete offline pipeline

```bash
vlm-tamp run --dataset data/pilot --config configs/offline.json --output results/offline
vlm-tamp annotate --dataset data/pilot --output annotations/pilot
vlm-tamp report --dataset data/pilot --experiment results/offline --annotations annotations/pilot
```

The deterministic mock is a small language-rule baseline. It ignores RGB and does not read ground truth. **Its scores validate the pipeline; they are not empirical VLM results.** Missing credentials trigger an explicitly labelled mock fallback. Dry-run uses the requested provider's pricing without loading a provider or making API calls.

## Configure credentials and inspect cost

Credentials stay outside the repository. A literal environment file can contain `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY`; quoted values and `export` are supported. Values are loaded without executing shell code. The file is never printed, cached, or copied to results.

```bash
vlm-tamp run --dataset data/pilot --config configs/openai_pilot.json \
  --output results/openai-estimate --dry-run
```

The pilot selects four observations and three conditions: language, RGB, and RGB with candidate inventory. It allows **12 actual request attempts including retries** and a **$0.50 estimated budget**. Model names, output caps, pricing, prompts, and all conditions live in version-controlled configurations. Current pricing sources and verification dates are recorded in paid configs; update these when using another backbone.

## Run the authorized small paid pilot

```bash
vlm-tamp run --dataset data/pilot --config configs/openai_pilot.json \
  --output results/openai-pilot \
  --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
vlm-tamp report --dataset data/pilot --experiment results/openai-pilot --annotations annotations/pilot
```

Paid inference requires `--allow-paid`. SDK internal retries are disabled, so the runner accounts for every attempt. Before each attempt it reserves a conservative text-byte estimate, a configurable image-token allowance, and the maximum output cost. Errors without token usage retain their reservation. Actual response token usage is stored and used for cost estimates, including invalid model output. These are **estimated admission limits**, not a guaranteed provider invoice cap; use provider-side limits for a hard billing cap. Provider invoices are authoritative, and cache discounts are conservatively charged at the configured input rate.

Select a new experiment directory for each run; existing runs are protected. Identical requests reuse `results/cache/` with zero new API cost. Cache keys include provider, model, full prompt, schema, image content, visible input, condition, output cap, and seed. Cloud calls need not be deterministic; the seed controls collection and mock behavior. A cache hit records original usage separately from the current run's usage.

`configs/openai_small.json` exposes all four conditions, including geometric metadata. Gemini is optional through `configs/gemini_small.json`, which records a verified model ID and pricing source. No live Gemini accuracy is implied by the SDK adapter.

## Evaluate saved responses and generate reports

```bash
vlm-tamp evaluate --dataset data/pilot --experiment results/openai-pilot --annotations annotations/pilot
vlm-tamp report --dataset data/pilot --experiment results/openai-pilot --annotations annotations/pilot
```

Evaluation requires the same dataset manifest hash used during inference. Outputs include:

- `manifest.json`: model, full prompt/version hash, configuration, data hash, planned coverage, and cost ledger.
- `predictions.jsonl` and `raw/`: per-example outputs, validation status, attempts, latency, usage and cost.
- `metrics.jsonl`, `per_example.csv`, `summary.csv`, and `summary.json`: exact metrics and denominators.
- `per_task_summary.json`: task-level aggregation across initial states.
- `report.md`: the four requested symbolic, visual, constraint-pilot, and cost tables.
- `gallery.html` and `gallery-assets/`: RGB, reference/predicted goals, entities, stages, dependencies, constraints, overlays, and errors.

Language/RGB settings do not receive hidden simulator IDs. Object instance identity can be resolved offline from a unique predicted point-in-mask or box match. Category recognition alone earns no instance credit. Whole-object masks do not supply drawer-part or arbitrary-region masks. Missing localization counts as failure for visible final-goal targets; missing reference coverage is reported as N/A. See `docs/evaluation.md` for exact scoring semantics.

## Manipulation-constraint annotations

```bash
vlm-tamp annotate --dataset data/pilot --output annotations/pilot
vlm-tamp validate-annotations --annotations annotations/pilot
```

Templates begin with `status: "draft"` and explicit coverage gaps. Add reference entities, stages, alternative valid partial orders, constraints, evidence, and a named reviewer before setting `status: "reviewed"`. Drafts are excluded from Track B. Incomplete references allow recall while leaving unlisted predictions unjudged. Do not infer required grasp/contact/path labels from BDDL final goals. Simulator joint type/axis metadata is kinematic evidence, not a complete task-constraint annotation. Read `docs/annotations.md` before editing.

## Tests

```bash
python -m pytest -q
RUN_LIBERO_INTEGRATION=1 python -m pytest -q -m integration
ruff check src scripts tests
python -m pip check
```

Unit tests cover actual official BDDL files against LIBERO's parser, unsupported logical constructs, ordered arguments, schema invariants, data leakage, distractor inclusion, pixel grounding, frame/sign-aware axis errors, partial annotations and alternatives, paid guards, retry accounting, corrupted/changed data, raw response storage, cache isolation, and official OpenAI SDK transport with a local fake server. They do not make paid calls. Rendering is an explicit integration-test opt-in.

## Continued pilot and verified part references

The revised pilot is configured in `configs/openai_pilot_v2.json`. It uses the same four observations, adds geometric metadata, and allows at most 16 attempts with a $0.25 estimated budget. The two executed pilots together cost an estimated **$0.0406916**. Paired counts and remaining errors are recorded in `docs/continuation_results.md`.

To estimate and run a new follow-up (use new output directories if these already exist):

```bash
vlm-tamp run --dataset data/pilot --config configs/openai_pilot_v2.json \
  --output results/v2-estimate-new --dry-run
vlm-tamp run --dataset data/pilot --config configs/openai_pilot_v2.json \
  --output results/v2-new --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
```

Verified simulator references are separate from draft expert annotations. The following command re-renders original observations and checks drawer site/joint/body links, actual visible geometry pixels, parent-mask consistency, and the official `Open` predicate at joint limits. It adds derived masks without changing the collection manifest, original images, or model-visible fields:

```bash
vlm-tamp simulator-references --dataset data/pilot --output annotations/simulator-new \
  --examples libero_goal-00-state000-seed0 libero_goal-00-state001-seed0 \
  libero_goal-03-state000-seed0 libero_goal-03-state001-seed0
vlm-tamp report --dataset data/pilot --experiment results/openai-pilot \
  --output results/v1-facts-new --annotations annotations/pilot \
  --simulator-references annotations/simulator-new
vlm-tamp report --dataset data/pilot --experiment results/openai-pilot-v2 \
  --output results/v2-facts-new --annotations annotations/pilot \
  --simulator-references annotations/simulator-new
vlm-tamp compare --baseline results/v1-facts-new --updated results/v2-facts-new \
  --output results/comparison-new
```

The factual references cover articulation identity/type, visible drawer bodies, and axes in explicit local/world frames. They do not assert a required grasp, contact strategy, task sequence, or path constraint. Gold gallery boxes mark verified part references. A drawer-body mask is not a mask of its interior volume or a handle affordance.

Comparison requires identical dataset content, evaluator version, expert annotations, simulator references and matched image content. It compares common example/condition pairs only; the four new geometry predictions are excluded from the v1/v2 paired counts. Both prompt and schema descriptions changed, so this is an interface/prompt refinement comparison rather than a prompt-only causal ablation.

## Frozen backbone comparison and schema 1.1

Schema 1.1 introduced `joint_type` (`prismatic`, `revolute`, or `unknown`) and `articulated_part` on articulation constraints. The part must be a declared local part/region entity. Other categories keep these fields null, and parents use local IDs. Saved 1.0 predictions remain readable. Each experiment now uses its configured/frozen schema: current semantic presets use 1.2, while older frozen comparisons retain 1.1 and their original cache keys.

The executed comparison used dated GPT-4.1 Mini and GPT-5.4 Mini snapshots, the same `structured_v3` prompt, the same four observations/conditions, temperature 0, high image detail, and a 1,500-token output cap. The latter model's reasoning was explicitly disabled. `configs/frozen_backbone_protocol.json` and `configs/schemas/grounding_v1_1.json` record the protocol. Config loading rejects a changed schema, prompt, matrix or generation limit.

```bash
vlm-tamp run --dataset data/pilot --config configs/backbone_41mini.json \
  --output results/41-new-estimate --dry-run
vlm-tamp run --dataset data/pilot --config configs/backbone_54mini.json \
  --output results/54-new-estimate --dry-run
vlm-tamp run --dataset data/pilot --config configs/backbone_41mini.json \
  --output results/41-new --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
vlm-tamp run --dataset data/pilot --config configs/backbone_54mini.json \
  --output results/54-new --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
vlm-tamp report --dataset data/pilot --experiment results/41-new \
  --annotations annotations/pilot --simulator-references annotations/simulator
vlm-tamp report --dataset data/pilot --experiment results/54-new \
  --annotations annotations/pilot --simulator-references annotations/simulator
vlm-tamp compare --baseline results/41-new --updated results/54-new \
  --output results/backbones-new --controlled-backbone
```

The controlled mode requires matching prompt/schema, full example-condition coverage, images, scoring code/reference provenance, output cap, temperature and effective reasoning settings. New caps are $0.15 and $0.30 with no retries and 16 attempts per model. Both executed runs completed for an estimated **$0.1011967**, bringing total project inference cost to **$0.1418883**. See `docs/backbone_results.md` for the measured result and the logical-status ambiguity discovered in the stronger candidate.

Strict goal agreement continues to respect `goal_logic`. `symbolic_content` is a separately labelled exploratory diagnostic that compares predicate content while ignoring that flag; it does not replace strict scores. The clarification excerpt in `structured_v4_logic_clarification.txt` is incorporated into the separate follow-up below.

## Logical uncertainty and object correspondence follow-up

`structured_v4.txt` distinguishes the logical connective joining goals from unknown identities, numerical geometry and execution. It retains schema 1.1. The separately frozen follow-up uses the same four observations and dated models, temperature 0, high image detail and 1,500-token output cap. Its conditions are language, RGB, inventory and labelled-image/inventory; object poses are not rerun. The three common conditions permit a paired prompt comparison with `structured_v3`.

The labelled condition uses **all collected visible whole-object masks**, including distractors, to place shuffled `M01` labels and provide label-to-candidate mappings. It is explicitly simulator-assisted. Labels never select goal targets or expose BDDL goals, drawer part masks, joint types/axes, expert annotations or geometric parameters. The original images, dimensions, manifests and inventories remain unchanged. Part/region candidates receive no direct labels. `object-markers` prepares this separate bundle; the runner/evaluator check its source images, masks, complete coverage, original dimensions, anchor pixels and frozen hashes.

For a freshly collected matching dataset, prepare the labels once:

```bash
vlm-tamp object-markers --dataset data/pilot --output data/markers-v1 \
  --examples libero_goal-00-state000-seed0 libero_goal-01-state000-seed0 \
  libero_goal-03-state000-seed0 libero_spatial-00-state000-seed0
```

The bundle already exists in this workspace. The frozen protocol records its exact hash; changed data or marker inputs require a deliberately new protocol. Reuse it to run new experiment directories:

```bash
vlm-tamp run --dataset data/pilot --config configs/logic_markers_41mini.json \
  --output results/logic41-new-estimate --dry-run
vlm-tamp run --dataset data/pilot --config configs/logic_markers_54mini.json \
  --output results/logic54-new-estimate --dry-run
vlm-tamp run --dataset data/pilot --config configs/logic_markers_41mini.json \
  --output results/logic41-new --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
vlm-tamp run --dataset data/pilot --config configs/logic_markers_54mini.json \
  --output results/logic54-new --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
vlm-tamp report --dataset data/pilot --experiment results/logic41-new \
  --annotations annotations/pilot --simulator-references annotations/simulator
vlm-tamp report --dataset data/pilot --experiment results/logic54-new \
  --annotations annotations/pilot --simulator-references annotations/simulator
vlm-tamp compare --baseline results/logic41-new --updated results/logic54-new \
  --output results/logic-backbones-new --controlled-backbone
vlm-tamp compare-conditions --experiment results/logic41-new --output results/markers41-new
vlm-tamp compare-conditions --experiment results/logic54-new --output results/markers54-new
```

Each model permits 16 attempts and zero retries, with $0.10/$0.22 admission budgets. The protocol prespecifies schema validity, strict complete-goal agreement, logical status and ordered visible whole-object final-goal identity. Galleries show the actual labelled image sent to the model alongside the original RGB with evaluation overlays. A separate object-role score excludes part/region arguments and does not turn identity agreement into complete-goal or unaided vision credit.

The executed follow-up completed 32 calls with 31 valid predictions for an estimated **$0.10182805**; cumulative project inference is **92 calls / $0.24371635**. In the 12 common GPT-5.4 Mini cases, erroneous logical abstentions fell from 11 to zero. Both models still matched only 2/4 complete goals with labelled inputs. The newer model corrected bowl identity but introduced an unresolved support-region goal argument. See `docs/correspondence_results.md` for the measured results, coverage and remaining representation errors. All 74 tests, including real rendering, passed.

## Entity/geometry separation and task-held-out evaluation

Schema **1.2** adds separate `geometry_targets` and terminal-goal intent/evidence. It validates identity, kind and parent ownership against the exact allowlisted request, checks parent cycles and instruction evidence, and keeps geometric descriptor IDs out of final goals. Unknown geometry stays unresolved. Legacy scoring remains unchanged.

`structured_v5` was frozen for one backbone and 20 calls: 12 familiar diagnostics and eight responses from four previously unqueried task groups. A symbol linker was frozen on development outputs before held-out inference. It binds declared aliases/ownership and makes conflicting identities abstain. Final goal declarations and numerical geometry are unchanged; raw and linked artifacts stay separate.

```bash
vlm-tamp run --dataset data/pilot --config configs/semantic_development.json \
  --output results/semantic-dev-new-estimate --dry-run
vlm-tamp run --dataset data/pilot --config configs/semantic_heldout.json \
  --output results/semantic-held-new-estimate --dry-run
vlm-tamp run --dataset data/pilot --config configs/semantic_development.json \
  --output results/semantic-dev-new --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
vlm-tamp run --dataset data/pilot --config configs/semantic_heldout.json \
  --output results/semantic-held-new --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
vlm-tamp link-outputs --dataset data/pilot --experiment results/semantic-held-new \
  --output results/semantic-held-new-linked --protocol configs/frozen_symbol_linker_protocol.json
vlm-tamp report --dataset data/pilot --experiment results/semantic-held-new \
  --annotations annotations/constraint-review/annotations --simulator-references annotations/simulator
vlm-tamp report --dataset data/pilot --experiment results/semantic-held-new-linked \
  --annotations annotations/constraint-review/annotations --simulator-references annotations/simulator
```

The runs cost an estimated **$0.10099350**, bringing project inference to **112 calls / $0.34470985**. Outputs were valid in 14/20 cases raw and 19/20 after zero-call linking. Held-out inventory-assisted complete goals scored **2/4 raw / 3/4 linked**; RGB scored **0/4**. Raw development validity regressed from 12/12 in v4 to 7/12 in v5. See `docs/semantic_results.md` for these mixed findings and remaining region-selection errors.

Four evidence-backed drafts are ready in `annotations/constraint-review/review.html`. To create another review set:

```bash
vlm-tamp review-set --dataset data/pilot --output annotations/review-new \
  --simulator-references annotations/simulator \
  --examples libero_goal-00-state000-seed0 libero_goal-01-state000-seed0 \
  libero_goal-03-state000-seed0 libero_spatial-00-state000-seed0
```

The local page shows observations, measured references, independent kinematics and editable conditional proposals. A human name, explicit track coverage and confirmation are required to export reviewed JSON. Drafts stay excluded from Track B; validate exported files before scoring. Known joint axes do not establish task necessity or contact strategy.

## Automatic physical verification trial

`physical-probes` independently measures actual joint mechanisms, axes and signed limits, checks finite static placements using official predicates plus collision geometry, and restores the original scene. It never reads benchmark goal answers for verification. Probe images are direct state manipulations, not robot execution.

`verification-trial` runs a bounded VLM proposal/check/correction loop. First-pass numerical geometry remains unknown; explicitly oracle-assisted verifier feedback supplies measured parameters for the second pass. Every check has a `verified`, `contradicted` or `unresolved` status and scope. Benchmark scoring occurs only after both calls and stays out of feedback.

The saved trial used two calls for an estimated **$0.01332975**, bringing project inference to **114 calls / $0.35803960**. It verified the drawer mechanism and corrected axis/limit parameters without expert annotations. Full containment, grasp/contact strategy and executable motion remain unresolved. Exact benchmark agreement also remained false because the model retains an additional terminal `Open` goal. See `docs/verification_trial_results.md` and `results/drawer-verifier-trial/report.html`.

To reproduce physics on the same observation in a new directory:

```bash
vlm-tamp physical-probes --dataset data/pilot \
  --example libero_goal-03-state001-seed0 --moving-object akita_black_bowl_1 \
  --target-region wooden_cabinet_1_top_region --grid-size 5 --gl egl \
  --output results/physics-new
```

The frozen trial preset uses the existing `results/drawer-physics/facts.json`. Reuse that validated evidence with a new run directory:

```bash
vlm-tamp verification-trial --dataset data/pilot --config configs/verification_trial.json \
  --output results/verifier-new-estimate --dry-run
vlm-tamp verification-trial --dataset data/pilot --config configs/verification_trial.json \
  --output results/verifier-new --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
```

The preset permits at most two attempts, no retries and $0.08 estimated admission. Changing evidence, prompt or verifier implementation requires a new frozen protocol. Existing trials and earlier benchmark reports remain protected. Goal/contact/path necessity is not automatically certified from joint kinematics or static placement tests.

## Grounding across ten new tasks

A bounded expansion tested ten previously unqueried task groups across Goal, Spatial, Object and LIBERO-10, with one observation and two input conditions per task. It used the same frozen v5 schema/prompt/model settings. **Complete goals matched 0/10 with RGB alone and 2/10 with the supplied simulator inventory**, after the existing structural linker. Raw outputs were valid in 18/20 cases and linked outputs in 19/20. No final goals were changed by linking.

The paired visual report is `results/expanded-grounding-review/index.html`: each original scene appears beside both predictions, final goals, unverified stages/constraints, validation errors and descriptive inspection notes. The inventory improved some identities, but basket/microwave region binding, multiple-object references and domain predicate conventions remain problems. Stage/contact/path accuracy is N/A; no robot trajectories were tested. See `docs/expanded_grounding_results.md` for exact counts and limitations.

The 20 new calls cost an estimated **$0.09733725**, bringing project totals to **134 calls / $0.45537685**. Selection, request boundaries and prior task history were audited before inference. Rebuild the paired report from saved outputs without additional calls:

```bash
vlm-tamp report --dataset data/expanded-grounding --experiment results/expanded-grounding-raw
vlm-tamp report --dataset data/expanded-grounding --experiment results/expanded-grounding-linked
python scripts/report_expanded_grounding.py
```

## Reasoning-enabled Mini and GPT-6 Astra

The same ten diagnostic tasks were evaluated with medium reasoning using both `gpt-5.4-mini-2026-03-17` and `gpt-6-astra`. Both new arms used matching settings, omitted temperature, allowed 25,000 total output tokens including reasoning, and made 20 calls with no retries. The saved Mini-none baseline was reused with its original temperature and 1,500-token cap; that baseline comparison is a configuration change rather than an isolated reasoning test.

**Complete-goal matches were 0/10 RGB and 2/10 inventory for Mini-none; 2/10 and 5/10 for Mini-medium; and 4/10 and 9/10 for Astra-medium.** The provided inventory is the full simulator object/region list with names, kinds and ownership, not task-goal answers. Inventory results are simulator-assisted. Three successful Astra inventory scores rely on the evaluator's independent pixel-to-instance correspondence, so runtime scene binding is still needed. Grasp/path constraints and robot execution remain unverified.

The three-way visual report is `results/reasoning-comparison/index.html`; `docs/reasoning_comparison_results.md` explains remaining vocabulary, region and geometry gaps. The 40 new calls cost a conservative estimated **$2.38115025**, bringing project totals to **174 calls / $2.83652710**. The runner covers cache-write input rates, includes reasoning in output cost once, and records actual reasoning/cache usage without changing historical configurations.

Rebuild the comparison from saved metrics without model calls:

```bash
python scripts/report_reasoning_comparison.py
```

The frozen presets are `configs/reasoning_mini_medium.json` and `configs/reasoning_astra_medium.json`, with a combined $8.75 estimated new-usage admission cap. Any new paid run needs its own output directory and current budget/protocol review; saved runs are protected.

## Research limits and next steps

**RGB-only repair and retry (2026-10-12):** A separate compact semantic contract uses a 190-word prompt and agrees with its API/backend relation vocabulary and arities. It evaluates actual image referents plus desired relations/owned-part selectors without requiring hidden simulator region IDs. The ten-call Mini RGB-only retry scored **3/10 semantic complete goals** for an estimated **$0.09524325**. Saved RGB-only Mini/Astra outputs, explicitly adapted to this same new evaluator, score **3/10 and 9/10** with no additional calls. These are changed evaluation criteria, not revisions to historical BDDL scores or proof of robot execution. No inventory was supplied in the new trial. See `docs/rgb_repair_results.md` and `results/rgb-repair-report/index.html`; numerical geometry and manipulation verification remain unresolved.

**RGB-derived multiple-choice trial (2026-10-12):** The same Mini model now fills five final-goal blanks using numbered regions generated from RGB alone. No simulator inventory or reference masks enter inference. **8/10 complete semantic goals matched**, versus the saved free-form RGB baseline's **3/10**, under the same unchanged semantic evaluator. Choices cover 20/21 required object owners across nine tasks; the cream-cheese packet is missed by the proposal generator, and soup is confused with tomato sauce. All ten forms are valid; new estimated usage is **$0.08391675** within a $0.25 cap. Form, image-region proposals and crop presentation change together; this is a development-sample comparison, not robot execution. See `docs/rgb_choices_results.md` and `results/rgb-choice-report/index.html`. Rebuild/verify without API calls using `python scripts/report_rgb_choices.py` and `python scripts/verify_rgb_choices.py`. The existing Conda environment already contains the image-processing dependencies; standalone installations can select the `rgb-choices` extra.

**GPT-6 Astra on the same choices (2026-10-12):** Ten fresh `gpt-6-astra` RGB-choice calls scored **6/10**, versus Mini's **8/10**, for a conservative estimated **$0.41441250** within a $1 cap. All forms were valid; Astra correctly used not-found for the absent cream-cheese choice. Two additional strict misses selected the correct bowls and plates but used `Stack` instead of `On`, exposing ambiguity between the form's `stacked_on` and `supported_on` options. Both models still confuse soup with tomato sauce. Inputs, choice mappings, prompt, schema and grading are identical; Astra's output allowance is 4,000 versus Mini's 8,000, with no truncations. No simulator inventory was supplied and no prior result was rewritten. See `docs/rgb_choices_astra_results.md` and `results/rgb-choice-astra-report/index.html`; clearer choice definitions and reliable RGB proposal coverage remain the next useful improvements.

**Smaller Mini job and RGB repair (2026-10-12):** Disjoint state choices and genuine 512 RGB reached **10/10** on the original states but only **6/10** on new arrangements. RGB-only MobileSAM segments reduced choice clutter; a further controlled language compiler now leaves Mini only the job of matching instruction-derived noun phrases to image numbers. The final pipeline scores **9/10 on the previously failed state2 scenes and 9/10 on prospectively unqueried state3 scenes**, under unchanged semantic grading. It receives no simulator inventory or reference masks. One missing bowl proposal and one food-can confusion remain. All earlier failures are preserved; final templates support a controlled placement/pushing/activation grammar, not arbitrary specifications or robot execution. The entire repair costs **$0.387963** for sixty calls, within $0.50. See `docs/mini_role_grounding_results.md` and `results/rgb-role-choice-report/index.html`. Rebuild and verify without API calls using `python scripts/report_rgb_role_choices.py` and `python scripts/verify_rgb_role_choices.py`. The local segmentation dependencies are isolated in the workspace and the Conda environment is unchanged.

**Direct RGB without segmentation (2026-10-12):** Mini now also has a role-matching path that takes one untouched 512 RGB image and returns its own image points. On the same ten state3 scenes it scores **9/10**, matching the saved MobileSAM + numbered-scene + crops pipeline and failing the same soup/tomato-sauce case. The ten calls cost **$0.024723 estimated**, about 46% less. The instruction roles, goal compiler and semantic grader are unchanged; no simulator inventory or candidate proposals enter inference. This small paired replay does not establish general reliability or robot execution. See `docs/rgb_role_points_results.md` and `results/rgb-role-points-report/index.html`. Rebuild/verify without API calls using `python scripts/report_rgb_role_points.py` and `python scripts/verify_rgb_role_points.py`.

**New-task RGB validation (2026-10-12):** The exact same direct-RGB prompt, coordinate schema, role parser and semantic grader score **8/20 complete goals on twenty previously unqueried task groups**, with **25/39 correct image roles**. All twenty replies are valid JSON. Eleven points miss visible object masks and three select another instance, so the earlier 9/10 familiar-task result does not establish broad reliability. The catalogue preflight finds 70/130 tasks pass the current instruction/reference goal-structure audit; this sample covers twenty compatible contracts, rather than whole-LIBERO accuracy. New usage is **$0.05353950 estimated**, within $0.10. See `docs/rgb_role_generalization_results.md` and `results/rgb-role-generalization-report/index.html`. Rebuild/verify without API calls using `python scripts/report_rgb_role_generalization.py` and `python scripts/verify_rgb_role_generalization.py`.

**Existing-code baseline runs (2026-10-12):** The released RoboPoint checkpoint, loaded locally with its supported 8-bit option, scores **7/20 complete goals and 23/39 image roles** on an adapted shared RGB benchmark, versus saved Mini 8/20 and 25/39. Its native 1024-token cap is used; a truncated 128-token setup trial is retained separately. ReKep's actual keypoint/constraint modules produced 33 points and a two-stage draft, with an observed wrong placement binding. ViLaIn-TAMP's public development-branch interpreter generated a specification but an incorrect initial state in a ported RGB-only smoke test. GroundedPlanBench's official code link still says Coming Soon. These are component tests with documented adaptations, not full robot-system reproductions. Shared live API usage is **$0.03400750 estimated** within $0.10; the local model adds no API charge. See `docs/external_baselines_first_runs.md` and `results/external-baselines-report/index.html`.

**RGB-D numerical blanks (2026-10-13):** Six Mini calls on four existing scenes fill seven image landmarks and two method choices. Aligned LIBERO depth supplies metric points; a typed geometry library generates grasp-frame proposals and point/axis residuals with analytic pose Jacobians. **2/6 complete source/target bindings** are correct. Depth checks accept 34/42 landmarks, but only 14/34 are on the intended object; one grasp frame is proposed and none has fully correct source/contact-anchor binding. All six path choices and ten equation/Jacobian checks pass. Actual new usage is **$0.039708 estimated**, within $0.10. Two upright instructions are explicit variants, not original LIBERO path requirements. Independent object-axis compilation reuses the same replies with zero additional calls. TCP/attachment transforms, joint-space compilation, IK, stable grasping and robot execution remain unresolved. See `docs/rgbd_numerical_results.md` and the interactive `results/rgbd-numerical-report/index.html`. Rebuild/verify using `python scripts/report_rgbd_numericals.py` and `python scripts/verify_rgbd_numericals.py`.

**DINO landmark choices (2026-10-13):** Actual local DINOv2 and approved DINOv3 features were compared through the same RGB-D candidate proposer. Both final candidate sets cover the expected source/target whole objects in all four scenes after correcting an inappropriate height cutoff and sparse mask prompts. Mini then selects numbered landmark IDs. The selection comparison is **incomplete**: nine of twelve planned calls cost **$0.12728175 estimated**; four replies hit the output-token limit and three DINOv3 cases were not attempted under the shared $0.15 admission cap. DINOv2 has three complete replies/two complete goals; DINOv3 has two complete replies/one complete goal. This does not rank the backbones. No grasp passes primitive screening and no upright case completes. See `docs/dino_landmark_comparison_results.md` and `results/rgbd-dino-comparison-report/index.html`. Rebuild/verify with `python scripts/report_dino_landmark_comparison.py` and `python scripts/verify_dino_landmark_comparison.py`. DINOv3 uses a separate Python3.11 GPU environment; existing simulator/RoboPoint runtimes are preserved.

**GPT-6 Luna on the same DINO landmarks (2026-10-13):** With reasoning disabled, Luna completes **12/12 valid forms** for **$0.00683775 estimated** under the newly authorized **$2 ceiling**. Both DINOv2 and DINOv3 candidate arms bind the correct source and target in **4/6 cases**; their whole-object landmark memberships are 37/42 and 39/42. The candidate images, prompt, geometry and grader are unchanged. Model, reasoning and output allowance differ from the partial Mini trial, so this is a practical configuration comparison. All depth samples pass local checks, but estimated upright axes and grasp/contact semantics remain unresolved. See `docs/dino_luna_landmark_results.md` and `results/rgbd-dino-luna-report/index.html`. Rebuild/verify with `python scripts/report_dino_luna_comparison.py` and `python scripts/verify_dino_luna_comparison.py`.

**Prompt/metric audit (2026-10-12):** The recent ten-task RGB condition cannot obtain exact simulator-region identities for five tasks under its current prompt and reference coverage, capping its strict complete-goal score at 5/10. The 4/10 Astra RGB result is consequently not a general task-understanding accuracy estimate. A separate domain-convention mismatch (`In` versus the cheese task's `On`) and the schema/backend goal-name mismatch also affect scores. All historical predictions, prompts and measured counts are preserved; exact instructions, schema and source-verified reconstructed requests are available in `results/prompt-audit-2026-10-12/`. Audit input/output contracts and grading coverage before drawing further architecture or fine-tuning conclusions.

Track A verifies supported positive conjunctive BDDL goals and available visible-object correspondences. Track B requires reviewed expert annotations for intermediate stages, dependencies, manipulable parts and path/contact requirements. Proposed constraints are never self-verified by a model. Numerical axes are compared only in compatible frames, with an explicit sign convention. Model confidence is not a calibrated probability.

The initial pilot is too small for a claim that images improve grounding or for backbone rankings. Numerical pose-level relation prototypes now exist, while joint-space compilation and ST-pRRTC integration remain separate work. Next, improve and independently review part/landmark binding, complete representative articulation/contact annotations, and compare backbones under matching prompts and task-level uncertainty. See `docs/related_research.md` for methodological references and compatibility limits.

## Success-focused regression runner

Use `scripts/run_success_pipeline.py --describe` to inspect the current seven-task pipeline. The implementation, evaluation scope, and failure diagnoses are documented in [docs/success_first_regression.md](docs/success_first_regression.md). Results and retained development attempts are in `results/success-first-report/index.html`.

## Shared interaction interface experiment

`scripts/run_interaction_pipeline.py` uses one Luna-high prompt for all tasks:
[configs/prompts/interaction_choices_v1.txt](configs/prompts/interaction_choices_v1.txt).
The model selects a contact shape, mechanism, visible face patch, moving region,
goal reference and contact roles. `interaction_geometry.py` fits metric planes,
arbitrary-axis bars, control axes and relational goals using current RGB-D.
Dispatch uses those predictions rather than benchmark task names. The existing
executors accept these inputs through an optional `prepared` boundary; their
original entry points remain available.

Run with the existing environment, selecting an unused state explicitly:

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
.runtime/envs/rekep-libero/bin/python scripts/run_interaction_pipeline.py \
  --output results/interaction-NEW --states 4
```

The manifest and source snapshot are frozen before inference. `--resume` requires
the identical sources and never reruns completed cases. Reproducing an inspected
state requires `--development`. Every case, including invalid VLM output and
pre-motion failures, remains in the denominator. Requests use the existing $2
cumulative ledger, 12,000 output tokens, high reasoning and no automatic retries.
Each completed episode verifies its video, removes redundant observations and
proposal caches, and losslessly compresses retained observations.

The primitive library, robot-scaled opening/completion policies and collision
representation remain assumptions. Arbitrary-axis fitting has rotation tests;
it is not a claim of arbitrary-axis physical reliability. The mature partial-pose
hinge executor supports upright bars/hinges; other supported geometry routes to
the full-pose mechanism planner. The new bottom-drawer tasks reuse existing
LIBERO cabinet assets. Initial results are reported separately from the tuned
35-case regression in `results/interaction-generalization-report/index.html`.

## Fixed mechanism-family transfer

`scripts/run_family_transfer.py` restores the successful drawer/rotary-control
geometry and contact policies and uses their exact saved configuration. It runs
three reproduction controls, then five new initial states per original or related
task. `scripts/build_family_transfer_report.py` verifies native outcomes, issued
references, videos, source snapshots and cleanup, and reports the controls
separately. The report is `results/family-transfer-report/index.html`.

The original compact Luna prompt is unchanged. Transfer inputs contain fresh
RGB-D and no demonstration video or simulator object geometry. The shared API
ledger retains the cumulative $2 cap, high reasoning, a 12,000-token output cap,
and zero automatic retries. The new manifest extends only the internal request
count. The finite clustering fix and existing unambiguous image-coordinate repair
apply uniformly; no physical-case-specific thresholds are added.

Reproduce the saved cohort with `--resume` and its exact manifest, or select new
unused state indices in a new output directory. The default cohort is 33 episodes:
three state-1 controls and thirty transfer cases on states 5, 15, 25, 35 and 45.
Related LIBERO tasks reuse cabinet and stove assets; they are not a novel-design
benchmark. Geometry and controller parameters stay frozen during evaluation.

Future work, excluded from this experiment: estimate a hinge from a single
demonstration video, including camera/metric-scale handling; and assess a learned
controller that tracks planner references while correcting execution/model error.
The suggested [ALPHA project](https://alpha-nist.github.io/) is a reference for the
latter investigation. Current trials use no demonstration input or RL adaptation.

The frozen family-transfer evaluation completed with 15/15 successes on new
states of the original tasks, 5/5 in the second stove layout, and 0/10 on bottom
drawer opening/closing. The three reproduction controls also succeeded. All ten
bottom-drawer failures occurred before task motion: six scene-collision admission
failures, two without valid grasp IK, one panel-plane fit failure and one ambiguous
coordinate response. These are case/layout results on reused assets, not evidence
of transfer to arbitrary appliance designs.

After this evaluation, AlphaNIST and its linked
[SGS paper](https://arxiv.org/html/2610.12465v1) were reviewed as future-control
references. AlphaNIST learns complete assembly behavior; adapting its ideas into
a planner-conditioned residual controller would require a separate implementation
and training study. A proposed comparison is the current servo, a nonlearned
visual/compliance controller, and an RL correction policy trained with perturbed
mechanism estimates. The policy should observe task/contact motion as well as
tracking error: tracking an inaccurate reference precisely need not complete the
task. This study remains separate from the unresolved pre-motion drawer failures.
