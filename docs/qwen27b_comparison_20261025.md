# Qwen3.5-27B versus the corrected 9B baseline

The 27B model did not show a consistent grounding or tool-use advantage that
justifies adopting it for this pipeline. It used more serving memory and was
slower on the matched six-request batch. Luna remains primary. This is a small
same-asset development comparison, not a verdict on the Qwen family or a test
of its ability to learn through physical practice.

[Interactive input/output and timing report](../results/qwen27b-comparison-20261025/watch.html) ·
[Machine-readable summary](../results/qwen27b-comparison-20261025/summary.json) ·
[Per-request timings](../results/qwen27b-comparison-20261025/timing.csv) ·
[Concrete failure diagnoses](../results/qwen27b-comparison-20261025/failure-diagnoses.json).

## Fixed conditions and checkpoint

The official [Qwen3.5-27B checkpoint](https://huggingface.co/Qwen/Qwen3.5-27B)
was pinned to `fc05daec18b0a78c049392ed2e771dde82bdf654`. All eleven weight
shards matched their remote SHA-256 hashes. The checkpoint contains
27,781,427,952 parameters including vision/MTP and 55,562,872,800 tensor bytes.
It remains cached under `.runtime/models/Qwen3.5-27B`.

Both models used BF16, the existing vLLM runtime, 65,536 total context,
32,768 output tokens per request, temperature 1.0, presence penalty 1.5,
top-p 0.95, top-k 20 and thinking mode. The 900-second request guard remained.
The six saved one-step requests changed only their serialized `model` field;
messages, image bytes, schemas and seeds were preserved. Exact prompt counts
were 1,658 for each model in that batch.

Tool-assisted sessions used the corrected workspace/scratch and array-layout
contracts, 16 model turns, at most three ground calls and 1,200 seconds. Initial
requests for the two matched sessions differ from the 9B baseline only in
model. Initial requests for each fresh-state pair also differ only in model.
There were no correction retries, teacher hints, model fine-tuning, Reflection
reviewers, prior-model traces or cross-case memories.

The retained 9B baselines and new runs used shared server hardware at different
times. The new 27B server used GPU0; the fresh 9B controls used GPU1 with another
job present. Timings are descriptive, not an isolated throughput benchmark.
These are different checkpoints in one family, not a pure causal experiment
that changes only parameter count and holds training fixed.

## Matched one-step results

| Measurement | 9B | 27B |
| --- | ---: | ---: |
| Completed valid function calls | 6/6 | 6/6 |
| Admitted numerical fits | 4/6 | 5/6 |
| Model-client time for six requests | 585.29 s | 939.60 s |
| Generated tokens | 35,911 | 24,793 |
| Median raw hinge-line image error | 37.67 px | 44.95 px |
| Admitted fits needing handle recovery | 3/4 | 4/5 |

Raw image error is the mean distance of the proposed hinge points to the
projected native hinge line in the selected camera's original pixels. Native
data is private observer information. It never enters the actor's prompts or
feedback. Camera choice is part of the model proposal; pixel error should be
read alongside world-space fit error and image overlays.

The median native hinge error among admitted fits was 63.51 mm for 9B and
51.81 mm for 27B. **Those admitted subsets differ.** In the three cases admitted
for both models, errors changed as follows:

| Case | 9B error | 27B error |
| --- | ---: | ---: |
| vision2-r1 | 66.48 mm | 45.43 mm |
| vision3-r1 | 88.48 mm | 94.17 mm |
| vision3-r2 | 60.53 mm | 149.22 mm |

27B also admitted two cases rejected by the 9B fit, but lost one formerly
admitted case. Its rejected proposal lacked a supported vertical handle within
the selected door region. The five admitted errors were 24.60, 149.22, 51.81,
45.43 and 94.17 mm. An upright plane/cylinder fit can succeed while the proposed
hinge remains wrong. Numerical handle recovery must not be credited as accurate
model localization.

## Tool-assisted outcomes

| Case | 9B workflow / hinge error | 27B workflow / hinge error | 9B / 27B actor seconds |
| --- | --- | --- | ---: |
| Matched state020 | Finished in 7 turns; 23.28 mm | Finished in 10 turns; 47.75 mm | 157.86 / 319.66 |
| Matched state035 | Finished in 9 turns; 89.97 mm | 16-turn cap; no ground call | 220.48 / 270.08 |
| Fresh state000 | 16-turn cap; no ground call | 16-turn cap; no ground call | 170.03 / 286.53 |
| Fresh state041 | Finished in 11 turns; 23.33 mm | Finished in 14 turns; 63.98 mm | 300.95 / 271.85 |

States 000 and 041 were absent from prior Qwen input images; the hash audit
confirms this. They are retained initial observations from older Luna development
runs of the same microwave asset. Only public observations/calibration were
copied. No earlier strategy, learner memory or outcome was supplied. The model
order was counterbalanced across these two states.

27B's matched state020 and fresh state041 fits did not need numerical handle
recovery: its Python/depth analysis identified a usable handle region. That is
a useful capability. Its hinge hypotheses were still displaced. The state020
trace chose a hinge based on a depth transition; that transition did not identify
the true pivot. In fresh state041, its final text described the hinge placement
as correct despite a 64 mm private error, while separately noting that physical
success was unverified.

The three sessions ending at the turn cap made **no ground call**. They spent
their allotted turns reading, creating/inspecting images and running Python.
This was an analysis/decision-budget failure, not an unsuccessful IK, grasp,
collision or geometric search. The fixed 16-turn cap limits conclusions about
what the model might achieve with more interaction.

Ten Python tool errors occurred across the six new tool sessions. These include
requests for `matplotlib`, which is absent from the documented NumPy/SciPy/Pillow
bundle, and actual array-indexing mistakes. The traces distinguish unavailable
dependencies from coding errors. No model request failed or exhausted its
output-token allowance. Generation stalls and repeated analysis without a
submission are separate problems.

## Time and memory

Across all new work there were 89 model requests: 62 to 27B and 27 to 9B. All retained
at least 32k allowance; none truncated or raised a model-request error. Exact
tokenizer counts matched generation prompt counts for all 89.

- 27B model-client time: 2,067.59 s; 53,883 generated tokens.
- 9B fresh-control model-client time: 461.41 s; 24,322 generated tokens.
- All new tool dispatch: 28.22 s, including 17.54 s of Python and 10.55 s of grounding.
- Observer-only geometry scoring: 45.50 s for one-step proposals and 10.58 s after
  tool sessions. These measurements were excluded from actor feedback.
- Grasp planning, IK, arc planning and physical execution: 0 s; not exercised.

The two model totals cover different case sets and request counts; they are not
a direct speed ratio. Use the matched rows and timing CSV for comparisons.
Client times include exact context admission and HTTP streaming. Model-written
Python ran in the existing OS sandbox with private project data blocked.

The 27B server reported 51.01 GiB during weight loading. Peak owned serving
allocation was 74.75 GiB, versus 32.89 GiB for 9B, with configured GPU fractions
0.78 and0.34. These peaks include cache/runtime allocation and are **not measured
minimum VRAM requirements**. Other users' GPU allocations were excluded. Both
owned servers exited cleanly and their recorded worker PIDs are gone.

## Manipulation gate and remaining questions

No simulated manipulation episode ran. The registered gate required at least
four admitted one-step fits, at least 30% lower admitted-fit median hinge error,
no worse selected error in both matched tool cases, both fresh 27B selected
errors below 30 mm, and no more one-step handle recoveries than 9B. Only the first
condition passed. Recovery counts are conservative here because admitted-fit
denominators differ; the report retains both counts and denominators. This was
a screening rule for one development episode, not a universal accuracy standard
or the existing 3 mm IK goal tolerance.

Keep Luna primary and retain 9B as a cheaper diagnostic baseline. Further useful
tests should examine hypothesis submission and mechanism verification:

- The offline wrapper states a turn limit initially but does not return a
  remaining-turn counter after each tool call. Making budget/progress explicit
  is a generic harness hypothesis, not a proven explanation for these loops.
- Fit admission verifies supported observed geometry. Stronger independent
  mechanism checks or uncertainty feedback may help prevent treating a fitted
  plane/handle as evidence of a correct pivot. The existing warning alone did
  not prevent overconfident wording.

Neither change was introduced during this frozen model comparison. This study
does not measure physical self-practice learning, memory transfer or performance
on new assets. Do not feed private reference coordinates into learner memory.

## Code and retained evidence

`serve_local_qwen.py` now accepts an explicit checkpoint path and served model
name; its 9B defaults remain. The offline probe accepts a model and calibrated
fixture/observer callbacks without changing model-facing tools. Each finished
case writes a completion status so the live report does not show a stale active
case. This progress-only change was made between frozen phases.

`compare_qwen_sizes.py` implements the registered replay, matched and fresh
phases. `summarize_qwen_sizes.py` applies the declared gate and retains actor
choices rather than selecting the best private fit. `report_qwen_sizes.py`
renders model inputs/outputs, image artifacts, errors and profiling. The local
full-robot actor remains untested with 27B; this run used the offline controller.

The result directory preserves the download/checksum manifest, protocols, fresh
fixture provenance, frozen source versions, exact inputs, all outputs and tool
events, native scoring, timing CSV and GPU records. Ninety-one focused tests
passed, including a known-projection scoring check and a check that scoring uses
the actor-selected candidate. The Luna guide remains unchanged.
