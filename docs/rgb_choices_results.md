# RGB grounding by filling blanks — 2026-10-12

The new finite-choice trial obtained **8/10 complete image-grounded final goals**, compared with **3/10** for the saved free-form RGB trial. Both use `gpt-5.4-mini-2026-03-17`, medium reasoning, standard service, an 8,000-token output allowance and the **same unchanged `rgb-semantic-1.0` evaluator and frozen semantic references**. All ten new forms were valid and compiled. Estimated new usage was **$0.08391675**, within a $0.25 admission cap. There were no retries, API failures, truncated responses, abstentions or budget stops.

The visual report is `results/rgb-choice-report/index.html`. It shows the original RGB, the actual numbered scene and crop sheet sent to the model, the filled form, and separately labeled offline evaluation overlays. This trial used no Astra calls and no simulator inventory.

## What the model chooses

Each requested final goal has five blanks: desired state, object number, object part, destination number and destination part. The generic state library includes inside, supported on, in front of, open, closed, turned on, turned off and stacked on. Part selectors include whole, interior, top/middle/bottom drawer and door. Every choice set includes not-found/unknown options. Multiple final goals use multiple entries in the same form and one API call per observation. Single-object states have no destination.

Code maps those choices to the existing RGB semantic goal declaration. It creates local referents and image points from the selected RGB regions; it does not invent simulator identities or fill physical geometry. The model generates no object descriptions, coordinates, stages, constraints, code or trajectories. This is a smaller **final-goal grounding** job, not a complete TAMP specification or an implemented skill library.

## Where visual choices come from

A local, fixed image-processing procedure accepts only the original RGB bytes: mean-shift color smoothing, quantized Lab color components, an interior pixel per component and the largest 48 components. Choices are subsequently numbered in image order. No task language influences proposal generation or ranking. These are image patches, including background and robot distractors; they are not whole-object detections, privileged instance masks or an object catalogue.

The input contains the published instruction, numeric choice labels, the original RGB, an enlarged numbered scene and a numbered context-crop sheet. No object category list, benchmark ID, simulator region definition, pose, joint parameter, expected goal or reference mask is supplied. Each public SDK argument dictionary, without authentication, is saved under `results/rgb-choice-mini/input-XX/request.json`. The provider independently reconstructs the choices from RGB and rejects a modified mapping or extra input field before making an API call. Requests have no conversation, previous-response link or tools.

The algorithm, prompt, schema, images, semantic evaluator and previous baseline manifest were frozen before candidate coverage was measured and before paid inference. Offline masks then showed that **20/21 required owner instances are selectable across 9/10 tasks**. The cream-cheese packet has no selected image pixel. No proposals were modified or filtered using that information. Candidate coverage is a ceiling and is not counted as model success; the primary result remains **8/10**, including the uncovered case.

## Observed results

| Task | Previous free-form RGB | New choices |
|---|---|---|
| Push plate to front of stove | Fail | Match |
| Put cream cheese in bowl | Fail | Fail |
| Turn on stove | Match | Match |
| Retrieve top-drawer bowl and place on plate | Fail | Match |
| Bowl next to plate onto plate | Match | Match |
| Alphabet soup into basket | Fail | Fail |
| Orange juice into basket | Fail | Match |
| Bowl into bottom drawer, then close | Match | Match |
| Two mugs onto their respective plates | Fail | Match |
| Mug into microwave, then close | Fail | Match |

The cream-cheese failure starts in the image-choice generator: the target has no selectable dot. The model then chooses a nearby background point instead of abstaining. In the soup case, a soup choice is available, but the model chooses tomato sauce; basket identity and containment relation are correct. Thus failures distinguish candidate coverage, incorrect visual recognition and failure to use the unknown option.

All ten relation-type multisets match. This was also true for the previous free-form trial, so the gain comes from better measured image correspondence rather than fixing the relation vocabulary. In the eight complete matches, all requested object roles and semantic part selectors agree with the references. Selecting a cabinet's bottom-drawer selector does not establish the actual drawer pixel location or numerical joint geometry.

## Cost and limits

The new ten-call trial uses 29,341 input tokens, 13,758 total output tokens and 13,172 reasoning tokens. Reasoning is included in output and charged once. Compared with the prior ten-call trial, visible answer tokens drop from 5,127 to 586; total output drops from 19,000 to 13,758. Extra annotated images and crops increase input tokens from 12,991 to 29,341. Estimated usage drops from $0.09524325 to $0.08391675, about 12%; the short form does not eliminate reasoning cost. Cached-token discounts are ignored. Provider invoices remain authoritative. Pricing was checked against [official Mini documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

Project totals are **194 inference calls / $3.01568710 estimated usage**. Earlier experiments, graders, inventories, predictions and scores remain unchanged.

This is a descriptive comparison on ten already inspected development tasks. Form restriction, RGB candidate proposals, image enlargement and context crops change together. It does not isolate a pure multiple-choice effect or establish held-out accuracy. It establishes that this combined RGB-derived selection pipeline worked better for the small model on this sample. Constraint accuracy, required manipulation method such as pushing, part-level pixel geometry, grasp/contact feasibility and robot execution remain unverified; ST-pRRTC was not modified or integrated.

The next useful improvement is a more reliable RGB proposal generator, followed by evaluation on new scenes with similar-object distractors and absent-choice cases. Preserve the finite goal interface while measuring candidate coverage and conditional selection separately.

## Verification and reproduction

**133 tests passed**, including real LIBERO rendering, with 20 existing simulator/dependency deprecation warnings. Twelve new checks cover image-only proposal generation, forbidden inventory fields, modified image-to-choice mappings, correct/wrong image identity, semantic selectors and multiple final goals, missing/unknown choices, extra goals, paid opt-in, bounded costs, saved request arguments, output-directory protection and error redaction. All source lint checks pass.

`results/rgb-choice-report/final-verification.json` rechecks all ten actual completed response snapshots, instruction/schema echoes, saved request bytes, frozen RGB-only choice mappings, code-assembled declarations, unchanged baseline/evaluator, raw usage and report assets. A full browser-rendering test is not claimed.

Rebuild and verify from saved outputs without credentials or additional API calls:

```bash
conda activate vlm-tamp
python scripts/report_rgb_choices.py
python scripts/verify_rgb_choices.py
```

To repeat the authorized protocol as a separate paid experiment, choose a new output directory. The runner refuses to overwrite an existing experiment, allows ten attempts with no retries, reserves estimated input/output costs before each call, and stops after an API failure or a budget boundary. Credentials are read literally from the authorized environment file without being copied or printed.
