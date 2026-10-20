# Reasoning-enabled Mini and GPT-6 Astra — 2026-10-11

**Prompt/evaluation audit, 2026-10-12:** The historical counts below are unchanged. Five of the ten RGB-only tasks require region IDs that the prompt forbids inventing and the current pixel resolver cannot recover because their visual references are absent. The strict RGB complete-goal metric therefore has a structural ceiling of **5/10** for this dataset. Its 4/10 result must not be interpreted as 40% task-understanding accuracy. The prompt's containment vocabulary also differs from the cheese task's published `On` predicate, and the schema allows goal-name strings that backend validation later rejects. Exact instructions, full schema, source-verified reconstructed requests and per-task coverage evidence are in `results/prompt-audit-2026-10-12/`. Contract conformance, perception, task interpretation and reference coverage need separate evaluation before further model-selection or fine-tuning conclusions.

The stronger-model trial completed **40 new calls** on the same ten diagnostic tasks. **GPT-6 Astra matched 9/10 complete goals with the provided inventory and 4/10 from RGB alone.** Reasoning-enabled Mini matched 5/10 and 2/10; the reused Mini-none baseline matched 2/10 and 0/10. These are descriptive results on previously inspected tasks, not general model rankings or fresh held-out accuracy.

The visual comparison is `results/reasoning-comparison/index.html`. It includes a measured count chart and twenty paired panels, each showing an original scene and all three configurations under the same input condition. `summary.json` preserves per-task goals, entities, image points, stage/constraint proposals and failures.

## What “provided inventory” means

The inventory is a simulator-generated list of scene objects and named regions, each with an identifier, category, kind and parent/owner. A microwave scene includes the microwave, a yellow-and-white mug, and the microwave's heating region. Both conditions contain the same image and instruction; the assisted condition adds the inventory list. The list is not a task-goal answer and includes scene candidates beyond the intended targets.

The inventory supplies privileged scene identities and region definitions, so these results are **simulator-assisted**. Numerical poses, joint axes/limits, object masks, reference goals, expert labels, filenames and task IDs were excluded from both inference conditions. A real robot would need perception and a scene/domain model to construct the corresponding inventory.

## Frozen comparison

The saved `gpt-5.4-mini-2026-03-17` baseline used reasoning `none`, temperature 0 and a 1,500-token output cap. Both new arms used medium reasoning, omitted temperature, requested standard service, and shared a 25,000-token total output allowance. Reasoning tokens consume that allowance. Inputs, v5 prompt, schema 1.2, structural linker, evaluator and reference coverage were unchanged.

The new Mini-medium versus Astra-medium pair is a controlled backbone comparison with matching generation settings. The baseline-versus-Mini comparison changes reasoning, temperature and the output cap together; it is a practical configuration comparison, not an isolated causal test of reasoning. The baseline's original responses were complete and were not truncated.

`configs/frozen_reasoning_comparison_protocol.json` and the two arm protocols were saved before inference. API metadata access was checked without generation; every actual inference returned the requested model ID. Mini uses its dated snapshot. Astra's official model identifier is `gpt-6-astra`; no dated Astra snapshot was invented, and the actual returned identifiers are retained.

At most 20 serial calls per arm, no retries, and $0.75/$8.00 estimated admission budgets were allowed. Both arms completed all 20 calls. There were no API failures, incomplete responses, local cache replays or budget stops. SDK retries were disabled. Reasoning, cloud-cache reads/writes, completion status and service tier were recorded.

Current API compatibility and pricing were verified in official documentation: [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [reasoning output budgets](https://developers.openai.com/api/docs/guides/reasoning), and [cache-write accounting](https://developers.openai.com/api/docs/guides/prompt-caching). Cache-write pricing replaces the ordinary input rate for those tokens; it is not additive. Admission and usage estimates conservatively price every Astra input token at the highest applicable input/cache-write rate, $12.50 per million. All output, including reasoning, is charged once at the configured output rate. Legacy configurations retain their prior accounting.

## Measured results

All counts retain invalid outputs in denominators. Linking did not change final goals or improve new-arm goal scores.

| Configuration | Input | Raw valid | Linked valid | Complete goals | Correct visible final-goal object roles | Points inside corresponding object masks |
|---|---|---:|---:|---:|---:|---:|
| Mini, reasoning none | RGB | 9/10 | 10/10 | 0/10 | 2/17 | 2/10 |
| Mini, reasoning none | RGB + inventory | 9/10 | 9/10 | 2/10 | 10/17 | 3/17 |
| Mini, medium reasoning | RGB | 10/10 | 10/10 | 2/10 | 6/17 | 5/10 |
| Mini, medium reasoning | RGB + inventory | 10/10 | 10/10 | 5/10 | 12/17 | 11/21 |
| GPT-6 Astra, medium reasoning | RGB | 9/10 | 9/10 | 4/10 | 13/17 | 8/9 |
| GPT-6 Astra, medium reasoning | RGB + inventory | 10/10 | 10/10 | 9/10 | 15/17 | 19/20 |

Object-role identity requires the expected predicate name and ordered argument, so correctly identifying the cheese and bowl receives no expected `On`-role credit when the model declares `In`. Point-in-mask counts are conditional on a resolved identity and a supplied point; they are not overall target-success rates or grasp-affordance checks. Missing localization is reported separately in the individual reports.

## Examples and remaining gaps

- **Two mugs and two plates:** Astra correctly distinguishes the white mug and the single yellow-and-white mug and assigns them to their respective plates under both conditions. The original inventory baseline assigned the same mug to both plates. Medium Mini identifies both mugs but still lacks successful plate correspondence.
- **Basket placements:** Astra binds final containment goals to the supplied basket-interior region. The baseline names the interior but references the whole basket in its final goals.
- **Microwave insertion and closure:** both new models treat “yellow and white mug” as one mug, bind containment to the supplied heating region, and retain closure as a final goal. The original inventory baseline adds a second mug and references the microwave object for containment.
- **Spatial bowl selection:** Astra supplies image localization that matches the requested bowl; it can abstain on an arbitrary simulator instance ID while still locating the referent accurately.
- **Cheese in a bowl:** all three configurations declare `In`, while the published task encodes `On(cream_cheese_1, akita_black_bowl_1)`. This is Astra's sole inventory complete-goal mismatch. It illustrates a domain-predicate convention issue rather than establishing that the physical placement was misunderstood.
- **Plate pushing from RGB:** Astra declares `InFrontOf(plate, stove)`. That is a plausible natural-language relation, but the current final-goal ontology does not support it, so the response remains invalid. With the provided front-region candidate, Astra uses the accepted `On(plate, front_region)` representation and matches the task.

Some successful scores require a further identity-binding step. In **three of Astra's nine inventory goal matches**, final object IDs remain null: the bowl in each spatial task and both plates in the two-mug task. The evaluator independently resolves the accurate image points/boxes against measured simulator masks. It does not insert those identities into model outputs. Six of nine Astra inventory goal matches therefore have every final argument explicitly bound to a supplied identifier, compared with all five Mini-medium matches. A runtime perception/scene-binding layer is still required before planner use.

All constraints remain `verified=false`, and no generated numerical constraint parameter was independently established in this trial. Zero human-reviewed grasp/contact/path labels and zero robot trajectories were added. A complete symbolic goal match is not an executable manipulation specification.

## Cost and interpretation

| New arm | Calls | Input tokens | Output tokens, including reasoning | Reasoning tokens | Conservative usage estimate | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| Mini, medium | 20 | 62,649 | 58,978 | 46,470 | $0.31238775 | 17.87 s |
| Astra, medium | 20 | 62,649 | 25,713 | 2,979 | $2.06876250 | 20.49 s |

New conservative estimated usage totals **$2.38115025**; project totals are **174 calls / $2.83652710**. The saved baseline incurs no new charge. Estimates deliberately ignore cloud-cache read discounts and, for Astra, apply its cache-write ceiling to all input tokens; provider invoices remain authoritative. Actual reasoning-token counts differ substantially between models despite the same nominal effort setting.

These results support using Astra as the stronger reference grounder for the next checked pilot, while retaining Mini-medium as a lower-cost comparison. Fine-tuning should follow a larger independently validated dataset and fresh evaluation tasks. The immediate unresolved issues are domain predicate definitions, runtime referent/region binding, numerical geometry and manipulation verification; higher goal counts do not resolve those automatically.

## Validation and saved artifacts

**105 tests passed**, including real LIBERO collection and independent physics checks. New checks cover standard-service request settings, reasoning-token accounting without double billing, cache-write price ceilings and stopping after an API failure. Formatting/static checks pass. The fixed old four-case wording in the comparison reporter was generalized without changing scoring.

`results/reasoning-comparison/final-verification.json` checks frozen inputs and prior manifests, exact response model/settings and full completion, image/payload equality across models, unchanged linked final goals, scored counts, budget accounting and report images. Original source-image and inference artifacts are preserved. Generated charts were visually inspected; a full browser rendering test is not claimed.

The full raw/linked galleries are in `results/reasoning-mini-medium-raw`, `results/reasoning-mini-medium-linked`, `results/reasoning-astra-medium-raw` and `results/reasoning-astra-medium-linked`. The controlled raw comparison is `results/reasoning-controlled-backbone-comparison/comparison.json`.

Rebuild reports from saved outputs with no API calls:

```bash
cd /home/duo/Rutgers/ARC/random/vlm_tamp
conda activate vlm-tamp
vlm-tamp report --dataset data/expanded-grounding --experiment results/reasoning-mini-medium-raw
vlm-tamp report --dataset data/expanded-grounding --experiment results/reasoning-mini-medium-linked
vlm-tamp report --dataset data/expanded-grounding --experiment results/reasoning-astra-medium-raw
vlm-tamp report --dataset data/expanded-grounding --experiment results/reasoning-astra-medium-linked
python scripts/report_reasoning_comparison.py
```
