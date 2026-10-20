# Grounding on ten new tasks — 2026-10-11

**Audit notice, 2026-10-12:** The strict RGB complete-goal metric in this dataset is structurally capped at 5/10: five tasks require named regions with no visual identity reference, while the prompt requires all unsupplied simulator IDs to be null. Historical counts remain unchanged and should not be read as pure task-understanding accuracy. See `results/prompt-audit-2026-10-12/audit.json` and the audit notice in `docs/reasoning_comparison_results.md`.

Twenty real model calls tested ten task groups that had not appeared in this project's paid inference history. The same frozen v5 prompt, schema 1.2, model snapshot and generation settings were used as in the earlier task-held-out trial. **Inventory-assisted complete goals matched 2/10; RGB-only matched 0/10.** This is a purposive task expansion, not a representative accuracy estimate or a paired comparison with earlier tasks.

The visual report is `results/expanded-grounding-review/index.html`. It shows each original scene beside RGB-only and inventory-assisted predictions, measured object boxes, predicted locations, final goals, proposed stages, declared constraints and validation errors. `summary.json` contains exact per-task results. Descriptive inspection notes are explicitly separate from scoring and are not human-reviewed constraint annotations.

## Task coverage and frozen design

The selected tasks cover plate pushing, cheese placement in a bowl, stove activation, drawer retrieval, spatial bowl selection, two packaged-object basket placements, drawer insertion followed by closing, two-mug/two-plate assignment, and microwave insertion followed by closing. They span LIBERO Goal, Spatial, Object and LIBERO-10. Each uses official initial state 0, collection seed 0, and a 256×256 agent-view image.

The model was `gpt-5.4-mini-2026-03-17`: temperature 0, reasoning `none`, high image detail, a 1,500-token output cap, at most 20 calls, zero retries and a $0.14 estimated admission budget. Every returned model ID matched the requested snapshot. All 20 calls completed; no outputs were truncated, no cached responses were used, and no API failures or budget stops occurred.

Selection, task/query history, input hashes, schema, prompt, evaluator and existing structural-linker protocol were recorded before inference in `configs/frozen_expanded_grounding_protocol.json`. The public task instructions determined diversity selection; model outputs were not used to choose tasks. New relative to project query history does not establish that tasks were absent from model pretraining.

The two conditions share exactly the same original RGB. Inventory-assisted requests additionally contain the full supplied scene inventory and ownership relationships. They are **oracle-assisted**. No reference goals, expert labels, initial-state vectors, task IDs, filenames, masks or numerical geometry were transmitted. `results/payload-audit-expanded-grounding.json` records all 20 request boundaries and hashes. Credentials were used for authentication only.

## Results

| Condition | Raw valid | Valid after frozen linking | Raw complete goals | Linked complete goals | Correct visible final-goal object roles |
|---|---:|---:|---:|---:|---:|
| RGB only | 9/10 | 10/10 | 0/10 | 0/10 | 2/17 |
| RGB + supplied inventory | 9/10 | 9/10 | 2/10 | 2/10 | 10/17 |

All failed outputs remain in denominators. Object-role identity requires the expected predicate name and ordered argument position; it is not an accuracy count for every entity the model mentions. Region arguments are excluded from that whole-object metric. Thus correctly naming the cheese and bowl still earns no expected `On`-role credit if the prediction instead declares `In`.

Image localization also remains weak: **2/10 RGB points and 3/17 inventory-assisted points** fall inside their corresponding measured object masks, among evaluable supplied points with resolved identities. These are conditional attempt denominators, not overall target-success rates. Even correct simulator IDs do not establish accurate image coordinates. The detailed reports separately count missing localization against visible goal targets.

The linker resolved one RGB output's undeclared stage alias through a geometric-owner relationship. It did not change any final goal, add an identity, alter numerical geometry or make additional calls. The inventory pushing output remains invalid because it uses a geometric descriptor as a semantic final-goal argument. Both invalid raw outputs were complete API responses; their failures were structural contract errors rather than truncation.

| Suite | Tasks | RGB complete goals | Inventory complete goals |
|---|---:|---:|---:|
| Goal | 3 | 0/3 | 1/3 |
| Spatial | 2 | 0/2 | 0/2 |
| Object | 2 | 0/2 | 0/2 |
| LIBERO-10 | 3 | 0/3 | 1/3 |

## What the examples reveal

The inventory response correctly grounded **stove activation** and **bowl insertion followed by drawer closure**. The latter preserves both containment and terminal closure and proposes opening, placement and closing stages. Goal correctness does not establish that these stages or their grasps are executable.

Several failures distinguish entity recognition from predicate/region binding:

- **Basket placement:** both inventory responses correctly identify the requested package and declare the basket's containment region, but their final goals reference the whole basket. The orange-juice placement stage already uses the proper region, exposing an inconsistency between stage and final goal.
- **Cheese in a bowl:** natural-language `In` is plausible, while the published task encodes `On(cream_cheese_1, akita_black_bowl_1)`. Exact benchmark disagreement alone does not show that the intended physical placement was misunderstood. A separately defined domain-predicate contract is needed to resolve this convention.
- **Drawer retrieval:** the inventory response identifies the bowl and plate but declares `In` despite the instruction's explicit placement *on* the plate. This differs from the cheese convention case.
- **Spatial bowl selection:** the inventory output chooses bowl 2 when the task requires bowl 1.
- **Two mugs and two plates:** the inventory output assigns the same yellow-and-white mug to both plates and omits the separate white mug.
- **Microwave insertion:** the inventory output turns the single phrase “yellow and white mug” into two mug-placement goals and uses the microwave object as the final containment argument, despite using its heating region in placement stages.

The strict results are preserved. Inspection notes explain the disagreements; they do not silently canonicalize predictions or claim that every mismatch is a physically invalid plan.

## Constraint and planning limits

Zero constraint annotations are human-reviewed, so stage, dependency, grasp, contact and path accuracy remains **N/A**. All generated constraints remain `verified=false`, and all numerical constraint-parameter lists are empty. This batch contains no independently verified new joint facts, numerical constraint compilation, trajectory planning or robot execution. Proposed contact/access strategies and a prismatic drawer claim remain hypotheses.

The broader trial supports continuing a checked single-task pilot while improving target-region binding, domain predicate alignment, multi-object references and visual localization. It does not establish that grounding is reliable enough to eliminate it as a possible end-to-end bottleneck.

## Cost, validation and reproduction

The 20 calls used 62,649 input tokens and 11,189 output tokens. Estimated new usage was **$0.09733725**, bringing project totals to **134 calls / $0.45537685**. Pricing was checked on the official [GPT-5.4 Mini page](https://developers.openai.com/api/docs/models/gpt-5.4-mini) on 2026-10-11. Cached-input discounts are conservatively ignored; the provider invoice is authoritative.

Budget, request-bound schema and structural-linking checks passed **31 tests**. Both new helper scripts pass formatting and static checks. Dataset, request, protocol, prior-history, raw/linked-goal preservation and report assets are checked in `results/expanded-grounding-final-verification.json`. The real contact sheet and selected prediction overlays were inspected; a browser-rendering test is not claimed.

The dataset/configuration and paid outputs are already saved. Rebuild reports without new API calls:

```bash
cd /home/duo/Rutgers/ARC/random/vlm_tamp
conda activate vlm-tamp
vlm-tamp report --dataset data/expanded-grounding --experiment results/expanded-grounding-raw
vlm-tamp report --dataset data/expanded-grounding --experiment results/expanded-grounding-linked
python scripts/report_expanded_grounding.py
```

`scripts/prepare_expanded_grounding.py` records how the ten scenes and frozen configuration were created. It refuses to overwrite an existing preparation. Any new paid expansion needs a distinct selection and a budget based on current usage.
