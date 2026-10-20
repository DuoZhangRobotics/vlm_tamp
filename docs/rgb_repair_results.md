# RGB-only prompt and evaluation repair — 2026-10-12

The authorized retry used **only the original RGB images and published instructions**. Ten new calls to `gpt-5.4-mini-2026-03-17` completed for an estimated **$0.09524325**, within a $0.25 admission cap. No inventory, object labels, masks, scene IDs, poses or reference answers were provided to the model. All ten responses passed the new contract. **Complete image-grounded semantic goals matched 3/10.** This is a real localization limitation of the small model; formatting success is not counted as grounding success.

The visual report is `results/rgb-repair-report/index.html`. It shows original RGB beside the fresh retry and separately adapted saved RGB-only baselines. Every score uses the same new semantic evaluator. Raw historical BDDL scores, prompts and predictions remain unchanged.

## Repairs

The 772-word v5 prompt was replaced by a **190-word** prompt in `configs/prompts/rgb_semantics_v1.txt`. The compact `rgb-1.0` declaration describes image referents, local ownership/part selectors, desired relations, stages and unverified constraint hypotheses. It does not require hidden simulator identities, exact metric geometry, benchmark region names or redundant identity bookkeeping.

Allowed goal names and their argument counts are now enforced in both the API JSON schema and backend. `InFrontOf` is supported, so a natural plate-in-front-of-stove declaration is no longer rejected merely because the old schema allowed a string that the old validator could not consume. Wrong argument counts and invented output fields are rejected. Self-verification is forbidden.

The new evaluation measures **natural-language semantic goal agreement**, not raw BDDL-ID equality. Independent image evidence establishes the actual object identity. A part or region is described through its image-identified owner plus a semantic selector, such as `bottom_drawer` or `interior`, without needing to emit a simulator's hidden region ID. Different owners and drawer levels remain distinct.

Reference meanings were frozen **before inference** from published instructions and benchmark instance labels through a documented scene-role contract. Containment in a basket or microwave normalizes the named region to its owner's interior. A front-of-stove surface normalizes to `InFrontOf`. For the single cheese-in-bowl task, the published instruction establishes `In` intent while the legacy BDDL `On` encoding is retained separately. There is no blanket `On`/`In` equivalence, and ambiguous multi-clause bowl references are not automatically converted.

The identity resolver considers all independently measured object masks; it does not read expected task goals to choose an object. A uniquely matching point or tight box is required. Conflicting point/box identities remain unresolved. Correct category words, local IDs resembling benchmark names, or descriptions alone do not earn instance credit. Missing or wrong object locations fail. Part-selector meaning is scored separately from actual part-pixel localization, which remains unverified.

## Results and honest comparison

| RGB-only configuration | Semantic complete goals | New model calls |
|---|---:|---:|
| Saved Mini-medium, old prompt, explicitly adapted goals | 3/10 | 0 |
| Saved Astra-medium, old prompt, explicitly adapted goals | 9/10 | 0 |
| Fresh Mini-medium, compact RGB contract | 3/10 | 10 |

The two saved rows are **new evaluations of existing outputs**, not fresh model runs or raw old-schema validity results. The adapter uses only the saved model declaration to translate representation; it does not read the task answer. In particular, the saved Astra `InFrontOf` response remains an invalid old-schema result even though its task meaning is interpretable in the new contract.

The fresh Mini's relation-name multiset matches the desired structure in all ten cases. Complete grounding succeeds for stove activation, selecting the bowl next to the plate, and bowl insertion followed by bottom-drawer closure. Its other failures are image localization: points/boxes miss the cheese, soup, orange juice, mugs or plates, or identify the cabinet as the stove. The shorter prompt changed which tasks succeeded but did **not** improve aggregate Mini grounding from the adapted baseline's 3/10.

The saved Astra RGB-only outputs match 9/10 under this evaluator; soup-in-basket remains unresolved by image identity. This establishes that meaningful RGB-only grounding was already present in those stronger-model outputs. It does not demonstrate a new-prompt improvement or nine executed robot tasks.

The earlier strict RGB metric was structurally capped at 5/10 because five tasks needed region IDs absent from the model input and visual identity references. All ten semantic references are now constructible. Ten explicitly **synthetic offline controls** demonstrate that correct image/selector declarations can be scored and that added wrong goals fail. These controls are not model successes and are never sent to the model.

## Protocol, costs and limitations

`configs/frozen_rgb_semantics_trial.json` freezes prompt/schema, semantic references, implementation hashes, original image/dataset hashes, model settings and prior paid-history manifests. Settings were medium reasoning, temperature omitted, standard service, high image detail, 8,000 total output tokens, ten calls maximum, no retries, and stopping after an API failure. There were no API failures, truncated responses or budget stops. No new inventory condition was run.

Usage was 12,991 input tokens and 19,000 output tokens, including 13,873 reasoning tokens. Estimated new usage was $0.09524325; project totals are **184 calls / $2.93177035**. Pricing was checked in the official [Mini model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini) on 2026-10-12. Estimates ignore cache discounts; invoices are authoritative.

These are ten previously inspected development tasks and a changed representation/evaluator. The old 4/10-versus-new-9/10 Astra values cannot be advertised as a model improvement. The small model still needs stronger RGB localization, a local image-grounding component, or a stronger model. Numerically executable constraints, full containment, actual part localization, grasp/contact feasibility, stability and robot execution remain unverified. ST-pRRTC was not modified or integrated.

## Validation and reproduction

**121 tests passed**, including real LIBERO rendering and the unchanged legacy evaluation/cache checks. Sixteen targeted new checks cover schema/backend relation and arity agreement, forbidden hidden-ID fields, ownership cycles, incorrect image identities, wrong region owners and drawer selectors, extra goals, task-goal-independent image resolution, RGB-only API inputs, paid opt-in, budget admission and error redaction.

`results/rgb-repair-preflight.json` records source payload/image hashes and synthetic coverage controls. `results/rgb-repair-report/final-verification.json` checks the actual ten response echoes, no-inventory request payloads, new costs, frozen references, unchanged old histories and report assets. A full browser-rendering test is not claimed.

Rebuild the report without model calls:

```bash
cd /home/duo/Rutgers/ARC/random/vlm_tamp
conda activate vlm-tamp
python scripts/report_rgb_trial.py
```

The actual generation command was:

```bash
python scripts/run_rgb_trial.py --output results/rgb-repair-mini \
  --env-file /home/duo/.codex/keys/vlm-tamp.env --allow-paid
```

Saved run directories are protected. Use a new directory and a freshly reviewed budget/protocol for any further paid experiment.
