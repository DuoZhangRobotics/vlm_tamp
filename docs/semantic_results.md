# Entity/geometry representation and held-out tasks — 2026-10-11

Schema **1.2** and `structured_v5` now separate geometric descriptions from symbolic scene identities, declare terminal-goal intent, and validate references against the exact supplied inputs. A separately frozen deterministic linker handles explicit namespace and ownership relationships. **20 real calls** completed; the four held-out tasks scored **2/4 inventory-assisted complete goals raw and 3/4 after linking**, versus **0/4 RGB-only**. The revised prompt/schema regressed on familiar diagnostic cases. These are mixed findings, not a demonstrated general improvement or readiness for planning.

Four evidence-backed constraint drafts and a local review interface are ready. **Zero annotations are human-reviewed**, so stage/contact/path accuracy remains N/A.

## Implemented contract

- `geometry_targets` describes a support surface, interior, handle or articulated body belonging to a declared entity. Its ID is distinct from entity IDs and cannot be a final-goal argument. Describing a plate's surface need not discard the plate's identity; real supplied semantic regions retain their own identities.
- Every new final goal carries `intent` (`explicit_terminal`, `task_outcome` or `uncertain`) and an instruction evidence phrase. Request-bound validation checks that the phrase appears in the instruction. Intermediate stage subgoals remain distinct. Uncertain terminal intent requires an abstention and receives no strict complete-goal credit; uncertainty about geometry alone does not change logical status.
- Parts/regions require declared local parents; parent links are acyclic. With inventory inputs, claimed identities, candidate kinds and resolved ownership must match the supplied inventory. Without inventory, hidden IDs are rejected. Image coordinates require an actual image. Invented support-region goals with no corresponding supplied region under a resolved parent are rejected.
- Validation reads only instruction, candidate inventory and image-presence information from the allowlisted request. It never consults benchmark goal answers, expert references or simulator joint facts.

Legacy 1.0/1.1 predictions remain readable, including unversioned historical payloads. New inference explicitly declares its configured version. Frozen 1.1 configurations use the saved 1.1 schema and retain their original cache keys. Prompt and schema changed together here; this is not a prompt-only ablation.

## Frozen design and held-out definition

The model was `gpt-5.4-mini-2026-03-17`, with temperature 0, reasoning `none`, high image detail and a 1,500-token output cap. All 20 returned model IDs matched the requested snapshot. There were no retries, cache hits, API failures, unknown-usage reservations or budget stops.

Development used the same Goal 0/1/3 and Spatial 0 observations under RGB, inventory and labelled-inventory conditions: **12 calls**, $0.13 admission budget. The held-out partition used Goal 2 and Spatial 1/2/3, official state 0, under RGB and inventory: **eight calls**, $0.10 budget. These four task groups had never appeared in our paid inference history; different initial states of previously queried tasks were not treated as task-held-out tests. Prior-query manifests and hashes are captured in the frozen protocols. This does not establish that these benchmark tasks were unseen during model pretraining.

Schema, prompt and both partitions were frozen before inference. Development responses exposed structural-link errors. The linker was then tested on development outputs and separately frozen **before held-out inference**. The schema/prompt were not changed between partitions, and the linker was not tuned on held-out answers. All model requests used public LIBERO inputs; the exact input audit is `results/payload-audit-semantics.json`.

## Raw outputs and deterministic linking

| Partition | Responses | Raw schema/request-valid | Valid after frozen linking |
|---|---:|---:|---:|
| Development | 12 | 7/12 | 11/12 |
| Task-held-out | 8 | 7/8 | 8/8 |
| Combined | 20 | 14/20 | 19/20 |

The linker uses relationships already declared in the output and ownership in the supplied candidate inventory. It binds geometric-owner aliases in intermediate stages/constraints, binds unique reference/local-ID aliases, and fixes ownership links when the corresponding parent is already declared. A conflicting `ambiguous=true`/non-null identity becomes an explicit abstention rather than a forced resolution.

It adds no entities or identity claims, changes no final predicate names/argument IDs/intent/evidence, and changes no numerical geometry. It does not remove extra goals, choose an unresolved final region, change candidate kinds, or repair an invented final-goal geometry argument. No model calls or inference-cache writes occur. Every event is retained in a separate derived run with its source status and source response path. Original responses, failures and raw scores are preserved.

Removing a conflicting identity claim can allow the evaluator to resolve identity from the model's unchanged image coordinates against independently measured masks. That pixel resolution occurs during offline evaluation, not inside the linker. Derived runs have zero new usage and retain their source ledger, avoiding double counting. Their galleries explicitly identify pipeline outputs and show linking provenance.

One development response remains invalid after linking because it assigns a supplied region identifier to an entity declared as a part. The linker does not silently coerce that type. The held-out failure was a geometric descriptor used as a stage entity; its explicit owner relationship resolved it.

## Complete-goal results

| Partition | Condition | Previous v4 raw | v5 raw | v5 with frozen linker |
|---|---|---:|---:|---:|
| Development | RGB | 1/4 | 0/4 | 0/4 |
| Development | Inventory, oracle-assisted | 2/4 | 1/4 | 2/4 |
| Development | Labelled inventory, oracle-assisted | 2/4 | 1/4 | 1/4 |
| Task-held-out | RGB | N/A | 0/4 | 0/4 |
| Task-held-out | Inventory, oracle-assisted | N/A | 2/4 | 3/4 |

All failures remain in denominators. Previous v4 raw validity on the 12 common development cases was 12/12, versus 7/12 here. Linking addresses wire-format relationships but does not establish a successful semantic improvement on those familiar cases. The prior and new raw responses were compared with identical evaluator/reference provenance; old scores were not softened.

The held-out inventory pipeline correctly identified **7/7 visible whole-object final-goal argument roles**; RGB identified **2/7**. Complete goals still differed because the cabinet-top task used the whole cabinet as the support argument instead of the published semantic top-side region. This illustrates the distinction between object identity and domain-specific predicate grounding. Candidate conditions supply privileged identifiers/ownership and may benefit from naming/layout regularities; these counts do not establish unaided perception or a statistically meaningful vision benefit.

Remaining errors include selecting a stove object instead of its supplied cooking region, selecting a cabinet object instead of its top-side region, extra final `Open` predicates in insertion tasks, and wrong/uncertain bowl correspondence or pixel coordinates. The model still sometimes labels access opening as `task_outcome`; adding an intent field does not guarantee correct interpretation. An extra final predicate is a disagreement with the published goal specification, not by itself proof of an invalid physical plan.

## Constraint review set

`annotations/constraint-review/review.html` provides four observations covering middle-drawer opening, opening/insertion, bowl placement on a stove and the multi-bowl spatial task. Each includes original RGB, measured object/part overlays, published goal evidence, image/state/BDDL hashes, verified drawer kinematics where available, and editable conditional stage/dependency/contact proposals.

All saved references have `status=draft`, disabled scoring coverage and no named human reviewer. Proposed constraints remain `verified=false`; independent axis facts do not prove task necessity or a grasp/contact strategy. The review interface requires a human reviewer name, selected track coverage and explicit confirmation before exporting a reviewed JSON file. Completeness remains optional and separate. Export leaves repository drafts unchanged; validate exported references before using them for scoring.

The interface's reviewer/coverage/confirmation guards and partial-coverage export were exercised in isolated JavaScript with mocked DOM/downloads. This was a workflow test, not a human review or a full browser-layout test. No data is sent from the local page. See `results/review-ui-validation.json`.

## Cost, validation and saved evidence

| Partition | Calls | Input tokens | Output tokens | Estimated usage USD |
|---|---:|---:|---:|---:|
| Development | 12 | 40,671 | 6,628 | 0.06032925 |
| Held-out | 8 | 25,761 | 4,743 | 0.04066425 |
| Combined | 20 | 66,432 | 11,371 | 0.10099350 |

Project totals are **112 inference calls / $0.34470985** of estimated usage. Prices were verified on the official [GPT-5.4 Mini page](https://developers.openai.com/api/docs/models/gpt-5.4-mini) on 2026-10-11. Input usage is conservatively priced at the uncached rate; provider invoices remain authoritative. Credentials were used only for authentication and were not copied into inputs or artifacts.

**96 tests passed**, including real LIBERO rendering and independent part verification. They cover actual saved-error regression, geometric/identity separation, real supplied regions, ownership and parent cycles, evidence phrases, request-bound IDs/image claims, terminal-intent abstentions, legacy replay/cache compatibility, draft-only review status, deterministic linking and source preservation, zero new derived usage, and explicit mock/processed-result labelling. Formatting and dependency checks passed; the pinned legacy stack emitted 20 deprecation warnings.

All **92 prior strict-goal scores were unchanged** when re-evaluated with matching original reference coverage into separate directories. Legacy schema/cache replay checks and all current frozen protocol checks passed. Final verification made no model calls. See `results/semantic-final-verification.json`.

- `results/semantic-development` / `results/semantic-heldout`: all raw calls, validation failures, metrics and galleries.
- `results/semantic-development-linked` / `results/semantic-heldout-linked`: separately labelled pipeline outputs and per-response linking events; zero additional API calls.
- `results/semantic-raw-comparison`: 12 matching v4/v5 raw development pairs, with coupled prompt/schema changes clearly identified.
- `results/semantic-summary.json`: exact raw/linked counts and source-only cost totals.
- `configs/frozen_semantic_development_protocol.json`, `configs/frozen_semantic_heldout_protocol.json`, `configs/frozen_symbol_linker_protocol.json`: separate inference and processing protocols.
- `annotations/constraint-review`: four evidence-backed drafts and the human review workflow.

The observed next gap is choosing the appropriate domain region for a symbolic predicate while keeping its geometric description separate. It requires an independently specified domain/affordance contract and stronger perception evidence; deleting predicates or selecting benchmark regions from hidden answers would mask the problem. Human-reviewed constraint labels remain necessary. No geometric compiler, policy, controller or ST-pRRTC integration was added.
