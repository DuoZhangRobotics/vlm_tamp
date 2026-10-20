# Controlled backbone experiment — 2026-10-11

This Stage 1 study compared the two models under one frozen prompt/output contract. It did not integrate geometric compilation, robot control or ST-pRRTC.

## Executed protocol

- Models: `gpt-4.1-mini-2025-04-14` and `gpt-5.4-mini-2026-03-17`. Metadata access and returned inference model IDs were checked.
- Four observations: middle-drawer opening, bowl-on-stove placement, combined top-drawer opening/insertion, and the multiple-bowl Spatial task.
- Four conditions per observation: language, RGB, RGB/candidate inventory and RGB/candidate inventory/object poses. The last two are oracle-assisted.
- Prompt: `structured_v3`; output schema: 1.1; temperature: 0; image detail: high; maximum output: 1,500 tokens.
- Reasoning: no reasoning parameter for GPT-4.1 Mini; `none` explicitly requested for GPT-5.4 Mini. Returned settings were checked.
- Thirty-two calls, sixteen per model, no retries and no budget stops. No paid tuning calls followed the comparison.

The original dataset and model-visible fields were unchanged. Reference BDDL goals, simulator joint/part labels, expert annotations and unrelated files were excluded from inference. Schema/prompt/configuration drift is checked against `configs/frozen_backbone_protocol.json`.

Official sources were checked for [GPT-4.1 Mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [GPT-5.4 Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini), and [reasoning/parameter compatibility](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4). These sources establish API support and configured pricing, not robotics performance.

## Output contract

Every articulation constraint now has a dedicated `joint_type` (`prismatic`, `revolute`, or `unknown`) and `articulated_part`. The part must be a declared part/region entity included in the constraint's entities. Non-articulation constraints have null articulation-specific fields. Parent references use declared local semantic IDs. Numerical geometry remains independently known or unresolved, and model predictions cannot mark themselves verified.

Old schema 1.0 results remain readable. New inference and cache entries use schema 1.1. Generation settings and the actual strict schema participate in cache keys, preventing reuse across changed controls.

## Measured results

Each entry below uses four observations and retains invalid responses in denominators.

| Condition | Valid schema, 4.1 / 5.4 | Strict instance-goal exact, 4.1 / 5.4 | Predicate-content exact, 4.1 / 5.4 |
|---|---:|---:|---:|
| Language | 4/4 / 4/4 | 0/4 / 0/4 | 0/4 / 0/4 |
| RGB | 4/4 / 4/4 | 0/4 / 0/4 | 0/4 / 0/4 |
| Candidate inventory, oracle-assisted | 3/4 / 4/4 | 2/4 / 0/4 | 2/4 / 3/4 |
| Object poses, oracle-assisted | 4/4 / 3/4 | 3/4 / 0/4 | 3/4 / 2/4 |

Both models produced 15/16 schema-valid predictions. The baseline failure used a parent ID not declared in the entity list. The comparison failure simultaneously marked an entity ambiguous and assigned it a specific instance ID, violating the abstention contract. Neither failure was repaired or selectively re-queried for scoring.

Both now supply explicit prismatic declarations on drawer tasks. Across valid responses with simulator facts, the baseline produced seven explicit known-type declarations and the comparison produced eight. Three matched part-linked type claims per model were evaluable, and all three matched the verified prismatic type. Both also produce known type claims in language-only conditions, showing that this can come from the word “drawer” rather than visual articulation understanding. All verified articulated bodies in this pilot are prismatic; the study does not test revolute discrimination.

Standard RGB produced no complete instance-goal match for either model. The updated images, point/box overlays, visible-mask denominators and per-example metrics show remaining localization errors. Unique candidate names and object-pose assistance help some identities, but candidate inventories do not visibly label arbitrary instance IDs on RGB images.

## Logical-status ambiguity

GPT-4.1 Mini returned `goal_logic=and` in all fifteen valid responses. GPT-5.4 Mini returned `unresolved` in fourteen and `and` in one. Some of its unresolved responses nevertheless contain exact benchmark predicate/instance tuples. Its abstention explanations discuss missing joint axes, grasp geometry, or lack of an articulated manipulation rather than an unsupported logical operator.

This exposes an output-contract ambiguity: unknown geometry is being conflated with uncertainty about the logical form of a symbolic goal. The strict metric continues to withhold complete-goal credit for `goal_logic=unresolved`; those scores were not changed to favor a model.

**Predicate-content agreement is an exploratory diagnostic introduced during error analysis**, after the frozen calls were completed. It compares exact normalized predicates and ordered instance arguments, but ignores the logical-status flag. It is not a predeclared primary measure or a replacement for strict goal agreement. Both primary and diagnostic values are retained in the report.

The additional clarification in `configs/prompts/structured_v4_logic_clarification.txt` explicitly separates logical conjunction from entity/geometry/execution uncertainty. It is prepared for a future separately frozen study and has not been used in paid calls.

## Usage and cost

| Model | Calls | Input tokens | Output tokens | Estimated usage cost USD |
|---|---:|---:|---:|---:|
| GPT-4.1 Mini | 16 | 41,338 | 6,855 | 0.0275032 |
| GPT-5.4 Mini | 16 | 40,982 | 9,546 | 0.0736935 |

Combined new cost is **$0.1011967**. Including the earlier 28 calls, project total is **60 inference calls** and an estimated **$0.1418883**. Prices are applied conservatively to reported usage; invoices and possible cache discounts are authoritative. The allocated new estimated-budget caps totaled $0.45, keeping the maximum total admission budget below $0.50.

## Interpretation and next Stage 1 work

This four-case comparison cannot establish a general model ranking or a statistically meaningful benefit from vision. Prompt/schema were fixed across models, but strict scoring is affected by the logical-status interpretation. Category understanding, physical localization, exact region identity and benchmark logical form must remain separate. Extra final `Open` predicates also require a specification-completeness audit rather than an automatic claim of physical invalidity.

The next useful experiments are a separately frozen logical-status clarification, a labelled object/part correspondence diagnostic reported as oracle-assisted, and reviewed stage/contact annotations with more diverse articulated tasks. In particular, prismatic-only references cannot establish articulation generalization. No numerical axes or complete task-plan labels should be invented to fill missing coverage.

The logical clarification and a whole-object-only correspondence diagnostic have now been executed separately; see `correspondence_results.md`. The original predictions, protocol and results in this document remain unchanged.

Results are in `results/backbone-comparison/`, `results/backbone-41mini/` and `results/backbone-54mini/`. The galleries preserve all successes/failures and verified simulator fact overlays. The core offline pipeline, both SDK adapters and real-render/reference checks remain covered by regression tests.

Final verification: **62 tests passed**, including both real simulator integration tests. Formatting/static checks and dependency consistency passed.
