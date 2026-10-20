# GPT-6 Astra on the same RGB choice form — 2026-10-12

The fresh `gpt-6-astra` trial obtained **6/10 complete semantic final-goal matches**, versus the saved Mini choice run's **8/10**, under the same frozen `rgb-semantic-1.0` evaluator. The ten new calls completed for a conservative estimated **$0.41441250**, within a $1 admission cap. All ten forms were valid; nine compiled and one explicitly chose not found. There were no retries, API failures, truncated responses or budget stops.

The comparison report is `results/rgb-choice-astra-report/index.html`. It selects only the saved Mini RGB-choice baseline and the new Astra RGB-choice outputs. Neither condition receives simulator inventory. This is fresh Astra inference, distinct from the previously rescored free-form Astra outputs.

## What stayed identical

All ten original RGB images, numbered scenes, context-crop sheets, numeric choices, task instructions, the 185-word prompt, the strict five-field form, and image-derived choice mappings are byte-identical to the Mini choice run. Both use medium reasoning, high image detail, standard service, omitted temperature and one independent request per observation. References, relation names, compilation and grading remain unchanged. Exact public request dictionaries are saved under `results/rgb-choice-astra/input-XX/`, without authentication.

The only differing inference arguments are model and maximum output allowance: **Astra 4,000 total output tokens versus Mini 8,000**, to contain cost. Every Astra response completed, and none reached that allowance; the smaller limit did not cause literal truncation. Nominal medium reasoning is not an equal-compute guarantee across models.

The model identifier `gpt-6-astra` was verified in [official documentation](https://developers.openai.com/api/docs/models/gpt-6-astra) and against every actual response. No dated Astra snapshot was invented. The frozen arm configuration is `configs/rgb_choices_astra_trial.json` and the protocol is `configs/frozen_rgb_choices_astra_trial.json`.

## Failure inspection

| Task | Mini choices | Astra choices | Evidence |
|---|---|---|---|
| Push plate in front of stove | Match | Match | Correct object roles and relation |
| Cream cheese into bowl | Fail | Not found | No cream-cheese choice exists; Astra returns object choice 0 |
| Turn on stove | Match | Match | Correct stove and final state |
| Top-drawer bowl onto plate | Match | Relation mismatch | Correct bowl and plate; Astra chooses `stacked_on`, yielding `Stack` instead of `On` |
| Bowl next to plate onto plate | Match | Relation mismatch | Same vocabulary issue; object identities are correct |
| Alphabet soup into basket | Fail | Fail | Both choose the tomato-sauce instance |
| Orange juice into basket | Match | Match | Correct juice and basket |
| Bowl into bottom drawer and close | Match | Match | Correct owner, drawer selector and final closure |
| Two mugs onto respective plates | Match | Match | Both mug-to-plate assignments agree |
| Mug into microwave and close | Match | Match | Correct containment owner and closure |

The two additional strict misses are **state-choice vocabulary mismatches with correct visual object identities**. Our form offers `supported_on` and `stacked_on` without clearly defining their distinction for ordinary bowl-on-plate placement. This ambiguity should be resolved in a new prompt or contract before another model-selection conclusion. It is not evidence that Astra failed to see those objects. The two `Stack` outputs remain failures under the original frozen scoring rule; no alias was introduced, relation was rewritten, or score softened after seeing the responses.

Candidate coverage remains **20/21 required owner instances across 9/10 tasks**. The absent cream-cheese choice cannot be repaired by selecting a stronger model. Astra's explicit not-found answer avoids Mini's nearby-background guess, but it does not earn a complete-goal success. In the soup case, an appropriate soup choice exists and both models select a different benchmark instance. This is an image-recognition error in this test.

The compiled-declaration relation-type metric is 7/10: two `Stack` mismatches and the uncompiled not-found declaration. This metric discards an abstained declaration's partial known goals; the original valid form is preserved for inspection. Thus it must not be described as seven correctly interpreted instructions.

## Usage and interpretation

Astra uses **29,341 input tokens, 953 total output tokens, and 375 reasoning tokens**. Mini used the same 29,341 input tokens, but 13,758 output tokens with 13,172 reasoning tokens. Reasoning tokens are already included in output and charged once. Astra's higher input rate makes the conservative estimate approximately five times the Mini run's $0.08391675 despite its much shorter reasoning/output.

Official standard rates are $10/million ordinary input, $1/million cached reads, $12.50/million cache writes and $50/million output. For both budget admission and actual-usage estimates, the Astra configuration intentionally prices **every input token at the $12.50 ceiling** and ignores discounts. Actual usage reports zero cached reads and 29,311 cache-write tokens. Provider invoices remain authoritative. The 4,096 image-token reservation allowance is an accounting setting, not an inference argument or a change to the images.

Project totals are now **204 inference calls / $3.43009960 conservative estimated usage**. Previous manifests, prompts, images, predictions, grader implementations and scores remain unchanged.

This small development-sample comparison does not establish a general ranking of the models. It gives no demonstrated overall benefit from upgrading this particular choice interface to Astra. The immediate improvements are clearer, non-overlapping state choices and a more reliable RGB proposal generator. Any revised prompt should receive a new frozen protocol and a separate evaluation, rather than revising this trial. Constraint accuracy, numerical part geometry, required motion method, grasp/contact feasibility and robot execution remain unverified. ST-pRRTC was not modified or integrated.

## Verification and reproduction

The twelve targeted choice-pipeline checks pass, and all source lint checks pass. The previous full suite had 133 passing tests including real LIBERO rendering; core choice generation, inference, compilation and grading code was not changed for this backbone trial. Reporting and verification commands now accept alternate experiment/baseline paths and preserve their prior defaults.

`results/rgb-choice-astra-report/final-verification.json` verifies all ten completed actual model responses, response schema/instruction echoes, RGB-only request contents, exact image/mapping hashes, deterministic compilation or explicit abstention, raw token usage, unchanged Mini baseline/grader, and all report assets. It also confirms that all ten public inference requests differ from Mini only in model and output allowance. A full browser-rendering test is not claimed.

Rebuild and verify without credentials or API calls:

```bash
conda activate vlm-tamp
python scripts/report_rgb_choices.py \
  --experiment results/rgb-choice-astra \
  --baseline results/rgb-choice-mini \
  --output results/rgb-choice-astra-report \
  --label 'GPT-6 Astra: same choice form' \
  --baseline-label 'GPT-5.4 Mini: same choice form'
python scripts/verify_rgb_choices.py \
  --config configs/rgb_choices_astra_trial.json \
  --experiment results/rgb-choice-astra \
  --baseline results/rgb-choice-mini \
  --report results/rgb-choice-astra-report
```
