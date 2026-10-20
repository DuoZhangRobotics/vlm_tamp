# Making GPT-5.4 Mini's grounding job smaller — 2026-10-12

The final **role-only Mini pipeline scored 9/10 on previously failed scenes and 9/10 on a prospective set of ten unqueried initial arrangements**. It uses `gpt-5.4-mini-2026-03-17`, RGB-derived visual choices and role descriptions extracted from the published instruction. Code handles supported final-goal templates and semantic part selectors. The model generates only image-choice numbers. No simulator inventory, reference masks, expected task answers, physical poses or benchmark instance IDs enter inference.

This is evidence that Mini can serve as the visual role matcher in this small, controlled pipeline. It does not establish universal reliability, full VLM specification generation, arbitrary-language support or successful robot execution.

## What changed

The first repair removed the overlapping `supported_on`/`stacked_on` options, preserved small contrasting RGB regions instead of selecting only the largest patches, and collected genuine 512-pixel RGB images. It fixed the known cream-cheese and soup cases and scored 10/10 on the original initial states. A separately frozen test on different initial states scored only 6/10, exposing real selection errors rather than a formatting issue.

The second repair uses the pretrained, class-agnostic [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) to generate object-sized image segments. Its point prompts come from RGB color regions and a uniform grid; task language and reference labels do not influence segmentation. Nested proposals are deduplicated. Whole-segment crops dim pixels outside the **predicted** segment. The same original RGB is also supplied, with a numbered scene and image-derived two-dimensional points/boxes. This reduces 64 patch choices to roughly 13–27 segment choices on the tested scenes. It supplies no object category names. MobileSAM has not been fine-tuned on these benchmark labels.

The five-field Mini form then scored 9/10 on the previously failed state1 scenes, but again only 6/10 on unqueried state2 scenes. Errors included additional mug goals, loss of referential qualifiers and wrong object selection. Those outcomes remain saved unchanged.

The final repair separates language compilation from visual matching. A controlled instruction grammar handles placement, pushing, activation, multiple placements and requested final closure. It preserves complete source noun phrases, including initial drawer/proximity qualifiers. A phrase such as “yellow and white mug” creates one visual role. Mini receives those roles as blanks and returns one image number for each; it cannot add objects, goals or change part selectors. A code adapter assembles the existing RGB semantic declaration from the chosen numbers and instruction-derived templates. Unsupported language is rejected before inference, rather than guessed.

Role descriptions are **requirements from the instruction**, not a scene inventory or an answer about which image segment is correct. The final input boundary permits only instruction, instruction-derived roles, and image-derived segment numbers/points/boxes, plus the three RGB images. It independently reconstructs that content and rejects altered coordinates or hidden identity fields before an API call.

## Saved trials and costs

| Trial | Initial-state set | Complete semantic final goals | New calls | Estimated cost |
|---|---|---:|---:|---:|
| Disjoint five-field form + clearer RGB | state0, previously inspected | 10/10 | 10 | $0.07388025 |
| Same frozen color-region repair | state1, then unqueried | 6/10 | 10 | $0.08348775 |
| Object segments + five-field form | state1, development repeat | 9/10 | 10 | $0.07843725 |
| Same frozen object interface | state2, then unqueried | 6/10 | 10 | $0.05898225 |
| Instruction templates + visual roles | state2, development repeat | 9/10 | 10 | $0.04723575 |
| Same frozen role-only pipeline | state3, prospective validation | 9/10 | 10 | $0.04593975 |

**Total for this repair request: 60 calls / $0.387963**, below the announced $0.50 spending ceiling. The final role-only pair costs **$0.09317550** for twenty calls. It uses 69,706 input tokens, 9,088 total output tokens and 8,616 reasoning tokens. Reasoning is included in output and charged once. Cached discounts are ignored; provider invoices remain authoritative. Rates were checked against [official Mini documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

Project totals are **264 inference calls / $3.81806260 conservative estimated usage**. Earlier manifests, prompts, grades and predictions are preserved. No new Astra calls were made.

## Remaining failures and scope

In final development, the cream-cheese scene has **no usable bowl representative in the segment choices**. Mini chooses a plate for the requested bowl role. This is a proposal-coverage limitation plus failure to use the unknown option; it remains a failed complete goal. The required cream-cheese candidate is correct.

In final prospective validation, Mini confuses alphabet soup with tomato sauce. The soup candidate is available; this is an image-recognition error. Both failures remain in the primary 9/10 counts. No expected object was injected, source choice relabeled, predicate softened or failed case excluded.

Final candidate coverage is **41/42 required owner instances across 19/20 scenes**. All ten prospective state3 scenes have full candidate coverage. Coverage was checked offline after both protocols and all candidate inputs were frozen, and is not counted as model success. Each final response is valid and compiles; valid format alone is not success.

Every version uses the unchanged `rgb-semantic-1.0` final-goal evaluator. The comparison changes the model's job, image presentation and perception pipeline; it is not a pure prompt ablation. State3 is a previously unqueried initial state of the same ten task groups, not a held-out task or object-class benchmark. The controlled instruction grammar is narrower than arbitrary language. Constraint accuracy, metric pose estimation, physical part localization, grasp/contact feasibility and robot execution remain unverified. ST-pRRTC was not modified or integrated.

## Review and reproduce

The current report is `results/rgb-role-choice-report/index.html`: actual original RGB, numbered segments, masked crops, instruction-derived roles, returned numbers, code-assembled goals, and separate offline evaluation overlays. `results/rgb-role-choice-report/final-verification.json` verifies all twenty final model responses, exact request contents, schema/instruction echoes, deterministic language/goal compilation, image/mapping hashes, raw token accounting and unchanged prior graders. Full browser rendering is not claimed.

The two earlier repair galleries are `results/rgb-choice-v2-report/index.html` and `results/rgb-object-choice-report/index.html`; their unsuccessful prospective results remain visible. The final configurations are `configs/rgb_role_choices_dev.json` and `configs/rgb_role_choices_validation.json`, with separately frozen protocols. Both were saved before either final inference partition ran. Checkpoint checksum and MobileSAM source revision are retained in each protocol.

**151 tests pass**, including real LIBERO rendering, with twenty existing simulator/dependency warnings. Tests cover forbidden identity fields and modified image mappings, isolated cached segmentation mappings, instruction-template cardinality, preservation of qualifiers, unsupported language, unknown choices, extra goals, wrong image identities, semantic part ownership, paid opt-in, bounded usage and error redaction. Source lint checks pass.

MobileSAM is stored in ignored `third_party/MobileSAM`. Torchvision `0.17.2+cpu` and timm `0.6.13` are isolated under `.runtime/rgb-object-deps`; the existing Conda environment was not upgraded. The model runs locally on CPU. It adds preprocessing cost/time but no model API charge.

Rebuild/verify saved results without API credentials or additional paid calls:

```bash
conda activate vlm-tamp
python scripts/report_rgb_role_choices.py
python scripts/verify_rgb_role_choices.py
```

The next engineering requirement is an uncertainty/consistency check and a fallback for missing image candidates. These measured 9/10 results should not be treated as execution-ready specifications.
