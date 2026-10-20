# Scoring contract

Current evaluator version: **1.4**. Historical-version sections below describe earlier additions.

## Track A

The BDDL parser retains exact S-expression structure and exact predicate spelling/argument order. Only positive conjunctions are scored as unordered goal sets. `Or`, `Not`, quantifiers and nested unsupported constructs are saved explicitly and disable goal-set evaluation for the entire example; no partial score silently drops them. Official LIBERO task files are tested against upstream `robosuite_parse_problem`.

Canonicalization uses explicit tables in `bddl.py` and `evaluation.py`. `On` and `In` remain different. `Close`/`Closed` are declared aliases. Unlisted predicates retain their spelling. Category aliases map a few explicit benchmark categories (for example `akita_black_bowl` to `bowl`) and do not resolve an instance. Cabinet middle/top/bottom regions are categorized as drawer parts for category scoring only.

Predicate precision/recall/F1 use `(normalized predicate, ordered benchmark instance arguments)` sets. Instance IDs come from supplied inventory resolution or independent unique pixel correspondence with simulator masks (a point-in-mask or box IoU ≥0.5). Semantic descriptions and category agreement alone never resolve an instance. The evaluator retains missing/invalid responses as empty predictions in evaluable denominators. It reports complete goal exact match, predicate-type exact match, category-goal exact match, and argument-category accuracy separately. Aggregate predicate scores are micro-averaged. Per-example results retain all counts.

Whole-object visible masks provide point-in-mask and box-IoU references. In unconstrained settings, a unique ordered goal-argument role correspondence gives an expected object for pixel scoring; measured pixels independently determine resolved identity. Known candidate IDs are scored against their actual object's mask. Reported IoU/point means include only localization attempts with applicable references; localized final-goal target recall includes missing attempts as failure. No drawer-part mask or region box is synthesized from a parent-object box. Task-relevant category/instance identification and manipulable-part identification require reviewed mappings; the BDDL scene inventory alone does not label every required grasp part.

## Track B

Only `status: reviewed` annotations with evidence and reviewer provenance are evaluated. Reference alternatives represent valid plans rather than a mandatory single total order. Stages match declared action names and ordered entity identities; synonyms and differences in stage granularity need explicit alternative annotations. Dependency comparison uses transitive closure, so equivalent partial orders are not penalized for different transitive edge listings. Cycles and contradictory goals are reported independently.

Constraints match category, type, ordered entities, goal/path/transition scope, required/conditional/optional status, and applicable stage semantics. Per-category results are saved for reviewed references. Incomplete annotation coverage disables false-positive counts, precision, exact match and F1 for that track, preserving recall and an unjudged-prediction count. This prevents alternative plausible plans from being labelled wrong by an incomplete reference. Choosing the best reviewed alternative is explicit in the per-example record.

Articulation-axis angular error is available only when a matched predicted/reference parameter has the same non-null coordinate frame and valid nonzero 3D vectors. Axis-sign invariance is controlled by the annotation. No implicit frame transforms or invented axes are used. Missing or incompatible geometry is not counted as a numerical match. Simulator axes alone do not identify which constraints are required by a task.

## Reliability and accounting

Invented simulator IDs, out-of-ontology predicates, contradictory goals, dependency cycles, ambiguity, identity abstention, JSON/schema failures, API failures, budget stops and unverified numeric parameters are stored. “Unverified numeric” is a reliability flag, not proof of hallucination. Unsupported task constraints are judged only when reviewed coverage supports such a judgment. Model confidence is stored but not treated as calibrated probability.

Latency is wall time per prediction, including retries, with cache hits distinguished. Tokens and cost are new usage in the current experiment; cached rows record original usage separately. SDK retries are off. Unknown-usage failed requests retain a conservative cost reservation. Cost admission depends on configured current prices and a vision-token allowance and is an estimate; it cannot guarantee a provider's future invoice. The manifest includes planned versus completed coverage and every actual API attempt.

The initial four inputs are instruction only, instruction/RGB, instruction/RGB/all candidate inventory, and instruction/RGB/inventory/object geometry. The latter two are oracle-assisted. A separate `markers` condition adds all-visible-whole-object labels to RGB and label-to-candidate mappings, using the same inventory and original image dimensions. It is also oracle-assisted. All settings omit BDDL goals and successful action/constraint references. `build_request()` is the audited allowlist boundary.

## Evaluator 1.1 and simulator fact coverage

`--simulator-references` adds independently verified articulation facts without modifying the original dataset or model inputs. Loading checks the benchmark commit, BDDL/initial-state/image hashes, measured part-mask hash/dimensions/box, and containment in the parent-object mask. Derived part masks cover the visible articulated body associated with a semantic site, not its interior volume. Unresolved object references are matched only to object masks; part/region references are matched only to measured part masks, avoiding false matches to a parent object.

Hidden instance IDs are not accepted as identity evidence in language/RGB conditions. Language-only image coordinates are excluded from pixel/instance scoring and counted as unsupported claims. Standard RGB identity can still be resolved through independent unique measured pixel correspondence. This policy also applies to entity-linked Track B scores.

Fact scoring distinguishes part identification, explicit joint-type claims, joint identity, and frame-compatible axis error. Type aliases are a closed list in `evaluation.py`; `drawer_open` is an action label, not an implicit prismatic claim. Missing type claims produce no conditional-accuracy denominator and no type-recall credit. Verifying a joint does not prove a particular manipulation strategy or required path constraint. Expert labels remain independently reviewed with their original coverage rules.

`evaluation_manifest.json` records scoring version and reference/annotation hashes. `compare` requires matching provenance and image content, keeps validation failures in common-pair denominators, and excludes unpaired conditions. Strict BDDL matching measures agreement with the published goal specification; additional predicates should be audited for reference completeness rather than automatically described as physically invalid plans.

Initial-state observations from one task are correlated. Larger experiments should report paired differences at the task level, intervals over tasks, prompt/backbone identifiers, failures and annotation denominators. Do not infer a statistically meaningful vision benefit from this small pilot.

## Evaluator 1.3 and whole-object correspondence

Schema 1.1 and strict goal scoring are unchanged. `goal_logic` describes the logical operators between final goal predicates. Positive conjunctions, including single predicates, use `and` even when localization, instance identity, joint type, axis, grasp geometry or execution remain unresolved. These uncertainties belong in entity identity/ambiguity, parameter status, joint type and abstentions. Strict scoring still gives no logical-goal credit to `unresolved`; the content-only diagnostic remains separate. The clarified prompt is tested through new responses, without editing or reinterpreting old predictions.

The new `visual.visible_whole_object_goal_roles` diagnostic considers only visible reference entities with kind `object` in ordered final-goal argument roles. A role is correct when a predicted predicate has the same name and arity and its independently resolved or supplied identity matches that exact argument position. Categories do not resolve identities. Extra predicates do not erase an otherwise correct individual identity role, so this diagnostic is separate from strict complete-goal agreement and ignores the logical-status flag. Missing or invalid predictions count as zero correct roles with the original reference denominator. Part/region arguments are excluded; whole-object identity never substitutes for a drawer region. The four frozen observations contain four such roles per condition.

`markers.py` prepares a separate oracle bundle from all visible whole-object masks, without reading final goals or expert/kinematic files. Numbering is shuffled using only the source image hash. Every visible object is included, including distractors. Each label anchor must lie in that object's mask. The bundle records source/mask/labelled-image hashes and preserves original dimensions. Only label/identifier mappings and the labelled RGB bytes cross the model-input boundary; mask paths, anchors, source hashes and provenance remain local. Normal model inputs and dataset manifests are unchanged.

Inference records the actual image hash, original image hash, full inventory hash, payload hash and oracle provenance. Evaluation checks the unchanged bundle and both image sources. Galleries include the actual marked image. `compare-conditions` requires complete matching inventory/marker coverage, the same original images and full inventories, and recorded bundle provenance. `compare` also checks payload hashes when both runs record them. Old saved responses are re-evaluated into new directories with the same evaluator/reference hashes for prompt comparisons; their original reports are preserved. The extra object-role diagnostic and marker support do not alter old strict-goal scores.

## Evaluator 1.4 and schema 1.2

New final goals declare terminal intent and an instruction evidence phrase. `intent=uncertain` disables strict complete-goal credit while retaining the content diagnostic; geometry and identity uncertainty remain independent. Legacy 1.0/1.1 scoring is unchanged. In 1.2, the diagnostic ignores logical-status and terminal-intent uncertainty while preserving exact ordered predicate/instance arguments.

Input-bound validation uses only instruction, inventory and image presence. It rejects hidden/invented IDs, candidate-kind/parent mismatches, unsupported derived-region goals under known parents, cycles, invalid goal arity and absent instruction evidence. Geometric descriptions have a separate namespace and do not receive entity mask credit. An intent field does not prove that the model classified an access action correctly.

`link-outputs` creates a separate derived experiment. It binds declared geometric-owner/local-reference aliases in intermediate stages/constraints, uses supplied ownership for existing parents, and makes contradictory ambiguity/identity claims abstain. It does not edit final goal declarations, insert entities/identities or numerical geometry, read benchmark answers, call a model or write caches. Final semantic errors remain errors. Offline pixel resolution can recover identity from unchanged coordinates after an abstention. Raw/linked scores retain source/event provenance and zero additional derived usage.

Controlled backbone comparisons require matching postprocessing protocols. Raw model validity and linked system validity are distinct endpoints. Four task-held-out observations remain a descriptive diagnostic; inventory is privileged assistance. `docs/semantic_results.md` records positive held-out counts and the negative development regression.
