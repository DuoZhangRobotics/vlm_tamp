# Logical uncertainty and object correspondence — executed 2026-10-11

The bounded follow-up completed **32 real API calls**, with **31 schema-valid responses**, no retries, no cache hits and no budget stops. The logical clarification removed erroneous logical abstentions in the matched GPT-5.4 Mini cases. Simulator-assisted labels corrected its ambiguous bowl selection, but complete-goal agreement remained limited by entity/region representations and additional final goals. This is a four-observation diagnostic, not a general model ranking or evidence of unaided visual grounding.

Estimated new usage was **$0.10182805**, bringing the project to **92 inference calls and $0.24371635** of estimated cumulative usage. Original responses and reports were preserved.

## Frozen design

- Dated backbones: `gpt-4.1-mini-2025-04-14` and `gpt-5.4-mini-2026-03-17`. All 32 returned model IDs matched the requested snapshots.
- The same four observations used previously: Goal tasks 0, 1 and 3 and Spatial task 0, official initial state 0, collection seed 0.
- `structured_v4.txt`, unchanged strict schema **1.1**, temperature 0, high image detail, 1,500 output tokens. The newer model's reasoning remained explicitly `none`; the older model had no reasoning parameter. Returned settings matched.
- Language, RGB, RGB/inventory, and labelled RGB/inventory: 16 predictions per model. Geometry was not rerun. The three common conditions provide 12 matched prompt pairs per model against `structured_v3`.
- Maximum 32 inference attempts, zero retries, $0.10/$0.22 admission budgets. Byte-bound worst-case estimates exceeded these caps; the runner admitted each call against observed usage plus its next reservation. Both runs completed comfortably within the caps.
- Primary endpoints were frozen before inference: schema validity, strict complete instance-goal agreement, logical status, and exact ordered visible whole-object final-goal argument identity. See `configs/frozen_logic_markers_protocol.json`.

The full new prompt clarifies logical versus geometric uncertainty and the ambiguity/identity contract, and explains the optional marker inputs. The output schema and generation settings did not change. These are descriptive prompt comparisons with one response per case, not replicated causal estimates for a single sentence.

## Logical clarification

`goal_logic` describes the connective between final goal predicates. Single predicates and positive conjunctions use `and`; missing identities, uncertain joint axes or grasp geometry, and unexecuted actions remain uncertain in their own fields. Old predictions were not edited or reinterpreted.

| GPT-5.4 Mini condition | Conjunctive logic declared, v3 → v4 | Strict complete goal, v3 → v4 |
|---|---:|---:|
| Language | 0/4 → 4/4 | 0/4 → 0/4 |
| RGB | 1/4 → 4/4 | 0/4 → 1/4 |
| Inventory, oracle-assisted | 0/4 → 4/4 | 0/4 → 2/4 |

On these **12 matched cases**, erroneous logical abstentions fell from **11 to zero**. All 16 new GPT-5.4 Mini responses were valid and declared `and`. Its predictions still contained **11 unresolved numerical parameters**; the prompt did not force numerical geometry to become known. All generated constraints remained `verified=false`.

GPT-4.1 Mini declared `and` in all 15 valid new responses and retained three unresolved numerical parameters. Its Goal 3 inventory response failed the local-parent-reference contract, as in the earlier backbone run. That response and its cost are retained; it was not repaired into a successful prediction.

Strict scoring still respects the model's logical status. The content-only diagnostic remains separate: for example, GPT-5.4 Mini's inventory predicate-content matches decreased from 3/4 to 2/4 even while strict agreement rose from 0/4 to 2/4. Removing an incorrect uncertainty flag did not solve all grounding errors.

## Simulator-assisted object labels

The separately prepared bundle uses **28 measured whole-object masks**, seven per observation. It marks every collected visible benchmark-object mask, including distractors, with a contour and an `M01` label attached to an interior mask pixel. Numbering is shuffled using only the source image hash, independent of instruction and goals. Robot/workspace/background surfaces without collected object masks are not labelled. The original 256×256 dimensions, RGB files, inventory and dataset manifest are unchanged.

Only labelled RGB bytes and label-to-candidate mappings are added to the inventory condition. Neither mask paths nor numeric anchor coordinates are sent. There are no goal-target filters, part masks, joint labels/axes, expert annotations or task IDs in requests. Whole-object labels do not identify drawer interiors or arbitrary support regions. The intervention combines contours, text and mappings; its effect cannot be attributed to the mapping alone, and labels alter some image pixels. All results for this condition are explicitly **oracle-assisted**.

| Model | Schema-valid inventory / labels | Strict complete goals inventory / labels | Correct visible whole-object goal arguments inventory / labels |
|---|---:|---:|---:|
| GPT-4.1 Mini | 3/4 / 4/4 | 2/4 / 2/4 | 2/4 / 3/4 |
| GPT-5.4 Mini | 4/4 / 4/4 | 2/4 / 2/4 | 3/4 / 3/4 |

The object-argument diagnostic evaluates exact identities in matching ordered predicate roles, excluding part/region reference arguments and ignoring the logical-status flag. It is not an object-category score, a count of every correctly named entity, or a substitute for complete-goal correctness. The four observations contain four evaluable object roles per condition; the drawer-only task contributes none. Missing/invalid predictions retain their original role denominator.

The aggregate hides an instructive change in the stronger model: inventory alone selected `akita_black_bowl_2` in the spatial task; labels selected the correct `akita_black_bowl_1`. However, the labelled response used an unresolved `plate_region` as the goal destination instead of the supplied `plate_1`, despite also declaring the plate entity correctly. Bowl identity improved while the plate's goal argument became unresolved, leaving the aggregate at 3/4 and the full goal wrong. The smaller model continued to abstain on bowl identity with labels; its one-role aggregate gain came from the drawer-insertion response becoming valid.

Both models' labelled top-drawer insertion responses included an additional final `Open` predicate relative to the published BDDL containment goal. The smaller model also introduced a separate unresolved drawer-interior entity. These are representation/alignment errors against the benchmark; an extra predicate is not by itself proof that a physical plan would be invalid. The evaluator does not silently map an invented region to its parent or remove additional goals.

New RGB-only complete-goal counts were 0/4 for GPT-4.1 Mini and 1/4 for GPT-5.4 Mini. Language-only counts were 0/4 for both. Neither labels nor candidate inventory establish unaided visual ability, and four observations do not support general accuracy or vision-benefit claims.

## Accounting, validation and coverage

| Model | Calls | Valid | Input tokens | Output tokens | Estimated usage USD |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 Mini | 16 | 15 | 43,427 | 6,350 | 0.02753080 |
| GPT-5.4 Mini | 16 | 16 | 43,071 | 9,332 | 0.07429725 |
| Combined | 32 | 31 | 86,498 | 15,682 | 0.10182805 |

Prices were checked against the official [GPT-4.1 Mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) and [GPT-5.4 Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini) pages on 2026-10-11. Input usage is conservatively charged at the uncached configured rate; provider invoices remain authoritative. No response had unknown usage. Credentials were used only for API authentication, never stored in payloads or results.

**74 tests passed**, including real LIBERO rendering and independent drawer site/joint/body/mask/state verification. Formatting, static checks and dependency checks passed. The pinned upstream stack emitted 20 legacy deprecation warnings. New checks exercise source/marker/mask tampering, full distractor coverage, instruction/goal-independent labels, the input allowlist, actual marker galleries, matched inventory comparisons and frozen-bundle protection.

Earlier backbone responses were re-evaluated into `results/backbone-41mini-eval13` and `results/backbone-54mini-eval13` with evaluator **1.3** and identical expert/simulator references. All **32 earlier strict-goal scores were unchanged**. Their original outputs remain in place. Both new runs use the same evaluator source hash and reference hashes for paired comparisons.

Reviewed expert stage/contact/path annotations remain **zero**; Track B accuracy is N/A. Existing independent drawer facts supplied four correct matched prismatic-type claims for the smaller model and three for the newer model. These conditional claims do not validate complete manipulation constraints, and the prismatic-only references do not establish articulation generalization. No motion planning or ST-pRRTC integration was added.

## Saved evidence and practical next work

- `results/logic-markers-41mini` and `results/logic-markers-54mini`: all raw responses, predictions, accounting, four-table reports and galleries. The actual labelled model input appears alongside the original RGB with scoring overlays.
- `results/logic-prompt-41mini` and `results/logic-prompt-54mini`: 12 matched v3/v4 prompt pairs per model; unpaired geometry/marker conditions are excluded.
- `results/marker-comparison-41mini` and `results/marker-comparison-54mini`: four matched inventory/marker observations per model.
- `results/logic-backbone-comparison`: complete 16-pair backbone comparison under the new frozen prompt/schema/settings.
- `results/payload-audit-logic-markers.json`, `results/logic-followup-summary.json`, and `data/markers-v1/markers.json`: public-data input audit, accounting summary and marker provenance.

The next implementation issue is the symbolic entity contract: preserving a resolved object's identity when referring to its support surface, keeping local parent IDs declared, and distinguishing final benchmark goals from intermediate actions. These errors should be addressed without using hidden BDDL answers or silently forgiving old predictions. Expert-reviewed stage/contact references and more diverse articulated tasks are still needed before geometric compilation and planner integration.
