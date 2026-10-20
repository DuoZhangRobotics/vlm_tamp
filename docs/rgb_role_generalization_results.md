# Fixed RGB prompt on new LIBERO task groups

Date: 2026-10-12. Model: `gpt-5.4-mini-2026-03-17`.

**The unchanged direct-RGB pipeline matched complete final goals in 8/20 new task groups.** All twenty replies are valid JSON and compile. Individual image roles are correct in **25/39 cases**: eleven returned points land on background or unassigned/occluding pixels, and three point to another object instance. The experiment used **twenty actual API calls / $0.05353950 conservative estimated usage**, within the frozen $0.10 budget, without retries.

The earlier 9/10 on familiar task groups does not establish reliable grounding across broader tasks. These are different tasks, rather than a matched difficulty comparison, but the new prospective sample provides direct evidence of substantial remaining failures.

## Selection and fixed interface

Task selection was saved before rendering the selected images or obtaining any model response. The selected groups are disjoint from all earlier paid query history by suite/index, official task name and BDDL-file hash. One previously unqueried arrangement (state4 or state5, alternating) is collected for each task; all twenty original RGB images have distinct hashes. The sample is purposive, emphasizing new package targets, contextual bowl references, front/back, middle and left/right distinctions, and stove deactivation.

The input is one untouched 512 × 512 RGB image plus the instruction and the same instruction-only role parser's phrases. The prompt, output coordinate schema, SDK transport, model snapshot, high image detail, medium reasoning, default tier, 3,000-output-token limit, goal compiler and primary `rgb-semantic-1.0` evaluator are unchanged from the earlier direct-point experiment. The new runner only increases the maximum admitted calls from ten to twenty and selects a new frozen dataset/reference file.

No segmentation inference, cropped views, numbered candidates, object inventory, expected object identities, reference masks, physical poses, tools, or previous replies enter inference. Returned coordinates are not adjusted, snapped to masks, or reassigned to nearby objects. The renderer collects simulator references solely for offline grading. The existing collector's suite allowlist is extended only in this preparation process to permit LIBERO-90; historical source and collection behavior remain intact.

All **39 required role instances have visible reference pixels** in the selected images. This removes absence of a visible reference object as an explanation for the reported failures. It does not establish that the instruction or appearance is unambiguous to a viewer.

## Measured results

| Suite | New task groups | Complete final-goal matches | Correct individual image roles |
|---|---:|---:|---:|
| LIBERO Goal | 2 | 2/2 | 4/4 |
| LIBERO Spatial | 4 | 2/4 | 6/8 |
| LIBERO Object | 8 | 3/8 | 11/16 |
| LIBERO 90 | 6 | 1/6 | 4/11 |
| **Total** | **20** | **8/20** | **25/39** |

All twenty responses are valid and compile, with zero abstentions. Relation-type agreement is 20/20 because the unchanged instruction compiler supplies the relations; this is not a model-generated relation score. The primary complete-goal score requires the correct relation, exact object instances and semantic part selectors, with no missing or extra goals. The additional point diagnostics never change that score.

The new usage is 13,390 input tokens and 9,666 output tokens, of which 8,971 are reasoning tokens already included in output. Rates remain $0.75/M input and $4.50/M output from [official Mini documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini). Discounts are ignored and provider invoices remain authoritative. Cumulative project inference is **294 calls / $3.89632510 conservative estimated usage**.

## What failed

The eleven background/occluder points range from a near miss to a substantial location or referent error. The plate in Spatial7 is missed by **one pixel**. The front butter box in LIBERO-90 task4 is missed by about **2.24 pixels**, inside its bounding rectangle but outside its visible mask. These remain failed complete goals under the frozen point-membership criterion. At the other extreme, the stove-deactivation point selects the background cooker area, about **203 pixels** from the reference tabletop stove mask. The original RGB visibly contains the background appliance as well as the intended tabletop stove, so this also illustrates referent ambiguity and foreground/background confusion.

Other unresolved points include cream cheese, BBQ sauce, chocolate pudding, the Spatial9 plate, and mugs/plates near scene edges. The three wrong-instance selections are:

- Salad dressing → ketchup.
- Butter → chocolate pudding.
- Middle black bowl → the upper/back bowl instance (`akita_black_bowl_3` instead of `akita_black_bowl_2`).

A point-only reply does not let us cleanly attribute every failure to recognition versus coordinate precision. The diagnostic categories describe actual mask membership; they are not a causal classifier. Looking at the original images and overlays is required to interpret these cases. Bounding rectangles and distances are offline diagnostic information, not alternative acceptance criteria.

## Language and reference coverage

Before selection, all 130 official tasks across the five local catalogues were audited. **70/130 pass the current instruction/reference goal-structure check**, including 55 tasks new to project queries. The remaining sixty have:

- 44 instructions rejected by the existing grammar.
- 13 unsupported reference-region contracts.
- 3 goal-structure mismatches where the parser accepts text but fails to express the task's required multiplicity or conjunction.

For example, “put both objects in the basket” can become a single compound source role, and an activation-plus-placement instruction can be accepted as one activation role. The frozen parser is not repaired in this experiment. The audit is a structural contract check, not proof of arbitrary-language semantic equivalence; some structurally compatible multi-goal tasks also lack a unique offline role alignment. The selected twenty have compatible structures and unambiguous offline role alignment.

The 8/20 score applies only to this selected supported-interface sample. It is not whole-LIBERO accuracy, random-sample accuracy or a claim that the tasks are absent from the model's pretraining.

## Implication and next work

Direct coordinate grounding is a useful low-cost baseline but is not yet reliable enough to assume perception will be a minor planning bottleneck. The narrower 9/10 result is preserved and should be interpreted alongside this broader 8/20 result. This experiment does not prove that MobileSAM or cropping is required, because no segmentation-assisted arm was run on these twenty tasks.

The next comparison should keep semantic role matching fixed while improving or verifying the selected pixels using an RGB-based localization method. Near-boundary errors, scene-edge objects and distractor instances need separate attention. Language multiplicity/conjunction and unsupported reference mappings also need explicit coverage work. Numerical geometry, physical part localization, action sequencing, path/contact constraints, grasp feasibility and robot execution remain untested; ST-pRRTC is not modified or integrated.

## Artifacts and validation

- `configs/rgb_role_generalization_selection.json` and `results/rgb-role-generalization-plan.json`: fixed selection, prior query history and the complete catalogue coverage audit, saved before rendering/inference.
- `configs/rgb_role_generalization_trial.json`, `configs/frozen_rgb_role_generalization_trial.json`, and the separate offline semantic/role reference files: exact frozen inputs, settings and source/reference hashes.
- `results/rgb-role-generalization-mini/`: actual SDK arguments, untouched RGB, raw model responses, points, compiled declarations, unchanged grades and usage ledger.
- `results/rgb-role-generalization-report/index.html`: all twenty scenes, original input, returned-point overlays added after inference, offline reference overlays, JSON, strict goals and point diagnostics.
- `results/rgb-role-generalization-report/final-verification.json`: twenty independent input/output/compilation/grade/diagnostic checks, task disjointness, raw accounting, preservation of 88 earlier artifacts, and sixty checked report images. Browser rendering has not been independently tested.

**168 tests pass**, including real LIBERO rendering, with twenty existing dependency warnings. New coverage tests catch compound-task misparsing and verify independent role alignment. Pixel tests retain wrong-instance and background failures even when a point lies inside the object's bounding rectangle. New source and scripts pass lint. The Conda environment and previous prompts, experiments and graders are unchanged.

Rebuild and verify without API calls:

```bash
conda activate vlm-tamp
python scripts/report_rgb_role_generalization.py
python scripts/verify_rgb_role_generalization.py
```
