# Continued grounding validation — 2026-10-11

The continuation tested the revised prompt/schema descriptions on the same four public LIBERO observations and added the geometric-metadata condition. No private source, reference goals, or expert annotations were transmitted. The new run was limited to 16 attempts and a $0.25 estimated budget.

The revised run used `gpt-4.1-mini-2025-04-14` and `structured_v2`. It made **16 calls**, producing **15 valid predictions** and one referential-integrity failure. Reported usage was **37,578 input tokens** and **5,546 output tokens**, with estimated cost **$0.0239048**. Combined with the first pilot, this is **28 calls** and an estimated **$0.0406916**. No additional paid calls were made after this run.

## Matched comparison

Both saved runs were re-evaluated under evaluator version 1.1 and the same verified simulator references. Original reports and raw responses were preserved. Failures remain in denominators. Twelve predictions have matching example/condition pairs; the four geometry predictions have no v1 counterpart.

| Condition | Pairs | Schema valid v1 → v2 | Predicate-type exact v1 → v2 | Category-goal exact v1 → v2 | Strict instance-goal exact v1 → v2 |
|---|---:|---:|---:|---:|---:|
| Language | 4 | 4/4 → 4/4 | 3/4 → 3/4 | 2/4 → 2/4 | 0/4 → 0/4 |
| RGB | 4 | 2/4 → 4/4 | 2/4 → 3/4 | 1/4 → 2/4 | 0/4 → 0/4 |
| Candidates, oracle-assisted | 4 | 4/4 → 3/4 | 3/4 → 3/4 | 3/4 → 3/4 | 0/4 → 1/4 |

The updated-only geometry condition returned 4/4 valid predictions and **2/4 strict BDDL instance-goal matches**. These are oracle-assisted results. The same four tasks are far too few for a statistically meaningful vision-benefit claim. Prompt and schema descriptions changed together, so the comparison does not isolate a causal prompt effect or establish general backbone performance.

## Verified articulation references

Twelve drawer-part facts were generated across four original observations (middle-drawer and top-drawer tasks, two initial states each). All twelve parts have measured visible masks. Checks established:

- Re-rendered RGB exactly matches the original image array.
- The official BDDL site's joint belongs to the corresponding articulated body subtree.
- Visible element-segmentation pixels belong to that subtree and are contained in the parent object's instance mask.
- Moving the associated joint to its two limits independently toggles LIBERO's official `Open` predicate.
- Local/world axes are explicit, normalized and tied to the original scene state.

The references are stored in `annotations/simulator/`, with hashed derived masks under `data/pilot/derived/`. The original dataset manifest and model-visible compartments remain unchanged. These references verify kinematics and visible body correspondence, not required stages, grasp/contact strategy, drawer access dependencies or whole-path constraints.

The revised geometric-metadata condition correctly names one of the two goal-related drawer parts. RGB predictions localize neither goal drawer part correctly under measured references. **No matched explicit prismatic/revolute claims were produced.** Generic `drawer_open` action labels are not reinterpreted as joint-type predictions. Conditional joint-type accuracy therefore remains N/A, with missing-claim coverage reported separately.

## Remaining failures and interpretation

Bounding-box ordering failures disappeared in this small revised sample, but localization remains inaccurate. The candidate-assisted Spatial response referenced an undeclared constraint entity or stage and failed validation. Some categories remain broad superclasses despite the revised prompt. Candidate and geometry assistance improve identity resolution for a few cases without establishing unconstrained visual grounding.

Strict BDDL goal matching can also penalize an additional final `Open` predicate in the combined opening/insertion instruction because the official goal records only containment. This is a benchmark-specification discrepancy; it is not by itself proof that an additional open final state or a different valid task sequence is physically incorrect. Expert task-plan/constraint annotations still require review.

Candidate inventories also contain arbitrary instance IDs that are not marked on RGB images. Multiple identical bowls cannot always be mapped to those IDs from the inventory alone. Abstention in such cases is appropriate; independent pixel matching can still verify physical target selection. The geometry condition supplies selected object poses, but it does not expose the new joint-type or part-mask reference labels.

Evaluator 1.1 now excludes invented hidden-ID credit and image-localization credit from language-only inputs, prevents whole-object/part mask confusion during pixel resolution, and checks simulator references against state/image/mask hashes. Factual articulation checks are reported separately from expert Track B scores. Paired comparisons reject mismatched scoring/reference provenance and distinguish mock results from empirical calls.

The final complete suite passed **48 tests**, including both real simulator integration tests and the part-reference checks. Static checks and dependency consistency checks passed. Known deprecation warnings remain in the pinned upstream stack.

## Outputs and next research work

- `results/prompt-comparison/comparison.md` and `.json`: matched counts and per-example transitions.
- `results/openai-pilot-v2-facts/report.md` and `gallery.html`: full four-condition results, gold part overlays and kinematic-fact coverage.
- `results/openai-pilot-v1-facts/`: baseline re-evaluated with exactly the same references and scoring policy.
- `annotations/simulator/`: verified physical facts; `annotations/pilot/` remains draft expert task-plan labels.

Next, compare another backbone with a fixed schema/prompt and review representative task dependencies, grasp/contact requirements, and alternative strategies. Articulation-focused prompting should require an explicit joint-type declaration or an unresolved parameter. Geometric compilation and ST-pRRTC integration remain subsequent stages.
