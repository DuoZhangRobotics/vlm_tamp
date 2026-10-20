# Mini role grounding without MobileSAM

Date: 2026-10-12. Model: `gpt-5.4-mini-2026-03-17`.

**Direct grounding from one untouched RGB image scored 9/10 complete final-goal matches, matching the saved MobileSAM-assisted pipeline on the same ten scenes.** Both pipelines fail the same alphabet-soup scene by selecting tomato sauce. The new run made ten actual API calls, with no retries, for **$0.024723 estimated usage**, within the frozen $0.10 admission budget.

## What changed

The model receives the original 512 × 512 RGB image, the task instruction, and exactly the same instruction-derived role phrases as the saved state3 role experiment. It returns a normalized `[x,y]` point per role, or `null` if uncertain. No segmentation inference, automatic proposal points, numbered scene, candidate-coordinate table, crop sheet, simulator inventory, reference masks, physical poses, tools, or earlier responses enter inference.

The controlled instruction grammar and goal compiler are unchanged. `rgb_role_points.py` feeds the returned points through the existing compiler using internal adapter IDs; these are not predefined candidates supplied to the model. The coordinates pass through unchanged. There is no nearest-object lookup, mask snapping, coordinate refinement, or answer-dependent rescue.

The unchanged `rgb-semantic-1.0` evaluator identifies each returned point against all independently rendered whole-object masks offline, then compares the complete set of normalized final-goal predicates with the frozen references. Whole-object identity and semantic part selectors are checked. Reference masks and expected answers are used only for offline grading and post-inference overlays.

## Paired results

| Metric on the same ten state3 scenes | Saved MobileSAM + scene + crops | Original RGB + direct points |
|---|---:|---:|
| Actual API calls | 10 | 10 |
| Valid JSON | 10/10 | 10/10 |
| Compiled declarations | 10/10 | 10/10 |
| Complete semantic final goals | **9/10** | **9/10** |
| Abstentions | 0 | 0 |
| Input tokens | 34,685 | 6,798 |
| Output tokens including reasoning | 4,428 | 4,361 |
| Reasoning tokens, already included in output | 4,192 | 3,999 |
| Estimated usage | $0.04593975 | **$0.024723** |

The new usage estimate is about **46.2% lower**. Both arms use the same pinned Mini snapshot, medium reasoning, high image detail, default service tier, and 3,000-output-token cap. Rates are $0.75/M input and $4.50/M output, checked against [official OpenAI Mini documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini). Output reasoning is charged once; cache discounts are ignored; invoices remain authoritative.

Nine scenes are correct in both arms; one is incorrect in both. The 21 direct points all resolve to visible object instances, but one resolves to the wrong requested object: for “pick up the alphabet soup and place it in the basket,” the returned source point `[0.53,0.68]` belongs to `tomato_sauce_1`, while the basket point `[0.84,0.55]` is correct. The expected source is `alphabet_soup_1`. This remains a failed complete goal, with no relabeling or change to scoring.

Project cumulative inference is **274 calls / $3.84278560 conservative estimated usage**. The preparation records hashes of 85 earlier artifacts; the verifier confirms they remain unchanged.

## Interpretation and limits

These results support using direct RGB role grounding as a simpler baseline for the tested scenes. They do not establish that segmentation is unnecessary across arbitrary tasks. This is a paired replay of ten previously studied scene arrangements, rather than new held-out tasks or a large reliability study. The model now estimates coordinates itself, so removing MobileSAM also changes the localization interface; this is not an isolated crop-sheet ablation.

The known missing-bowl proposal was in the earlier state2 development partition, which this ten-scene state3 comparison does not retest. No claim is made that the new pipeline has repaired that particular case.

The model still only matches noun phrases. Relations, multiplicity and part selectors come from the same controlled instruction grammar. Numerical poses, physical part localization, task sequencing, path/contact constraints, grasp feasibility, and robot execution remain unverified. ST-pRRTC is not modified or integrated.

## Artifacts and validation

- `configs/rgb_role_points_trial.json` and `configs/frozen_rgb_role_points_trial.json`: frozen before any direct-point response, with exact paired RGB/role checks and source/reference hashes.
- `results/rgb-role-points-mini/`: exact SDK argument dictionaries, original RGB bytes, raw model replies, returned points, compiled declarations, unchanged semantic scores and usage ledger.
- `results/rgb-role-points-report/index.html`: ten scenes showing the actual RGB input, returned-point overlays added after inference, role tables, JSON, goal comparisons and separately labeled offline reference overlays.
- `results/rgb-role-points-report/final-verification.json`: ten exact request/output/compilation/score checks, raw token accounting and preservation of earlier results; thirty report images are checked. Browser rendering has not been independently tested.

**165 tests pass**, including real LIBERO rendering, with twenty existing dependency warnings. Fourteen new cases cover direct-RGB transport without invoking segmentation, hidden-input rejection, unchanged multi-goal/drawer compilation, invalid coordinates, abstention, and strict wrong-instance/background-point failures. New source and scripts pass lint checks. The existing Conda environment and segmentation installation are unchanged.

Rebuild and verify the saved result without API calls:

```bash
conda activate vlm-tamp
python scripts/report_rgb_role_points.py
python scripts/verify_rgb_role_points.py
```

To inspect the frozen input without paid inference, choose a new output directory:

```bash
python scripts/run_rgb_role_points.py --output results/rgb-role-points-new-dry-run --dry-run
```

The original candidate-based experiments, prompts and graders remain saved separately.
