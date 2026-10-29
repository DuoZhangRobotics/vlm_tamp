# Qwen sampling, tool contracts and remaining grounding limits

Later follow-up: the [27B comparison](qwen27b_comparison_20261025.md) preserved
these corrected settings and did not find a consistent benefit from the larger
checkpoint. The results below retain the 9B diagnostic conditions.

The documented general-thinking sampling recipe removed the observed 32k
generation stalls in this diagnostic. Six of six one-step requests completed
with valid function calls. Visual grounding remains inaccurate, and numerical
fit acceptance must not be reported as a correct hinge or physical success.
Luna remains the primary actor; Qwen9B remains experimental.

[Input/output report](../results/qwen-sampling-diagnostic-20261025/watch.html) ·
[Overall results](../results/qwen-sampling-diagnostic-20261025/summary.json) ·
[Request timing CSV](../results/qwen-sampling-diagnostic-20261025/timing.csv) ·
[Tool timing CSV](../results/qwen-sampling-diagnostic-20261025/tool-timing.csv).

## Sampling experiment

The pinned checkpoint README recommends temperature 1.0, top-p 0.95, top-k 20
and presence penalty 1.5 for general thinking tasks. Our previous temperature
0.6 with an omitted/default-zero presence penalty resembled its precise-coding
recipe. The README specifically discusses presence penalty as a way to reduce
endless repetition. [Official guidance](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices).

The three formerly truncated requests were replayed with the same messages,
schemas, original image bytes, model checkpoint, BF16 runtime, thinking mode,
seeds, 65,536 context and 32,768 output allowance. Only temperature and presence
penalty changed in their request bodies. Baselines are retained runs, not
concurrent controls. GPU 0 was shared with unrelated work; timings describe
these executions rather than isolated throughput. This tests a settings
package, not the independent effect of either sampling parameter.

| Formerly stalled request | Old 32k seconds | New seconds | New output tokens | New outcome |
| --- | ---: | ---: | ---: | --- |
| vision1-r1 | 587.72 | 10.60 | 562 | Valid ground call |
| vision2-r2 | 556.60 | 76.18 | 4,665 | Valid ground call |
| vision3-r2 | 542.10 | 104.84 | 6,454 | Valid ground call |

Together these requests changed from 98,304 output tokens / 1,686.43 seconds
without a call to 11,681 tokens / 191.62 seconds with three valid calls.
The exact repeated-long-paragraph metric was zero for all six new one-step
responses; this does not imply that every sentence was useful reasoning.

The three previously completed cases also completed with valid calls. They
took 168.99, 78.41 and 146.26 seconds. Their historical 4k runs totaled 79.76
seconds, while the new 32k calls totaled 393.67 seconds. **This is not a universal
latency improvement.** That confirmation changes both sampling and allowance;
it does not isolate which caused the longer output.

The predeclared gate required all six generations to finish and at least five
valid calls before proceeding to tool-assisted probes. It passed 6/6. The
conditional two-factor follow-up was therefore not run.

## Geometry is a separate outcome

Two proposals failed the existing numerical fit: one selected a panel region
over the tabletop with no supported upright plane; another lacked a supported
vertical handle in its selected door region. Four were admitted, but their
native hinge-line errors were 34.64, 60.53, 66.48 and 88.48 mm. Three admitted
fits used the existing depth-based handle-recovery routine to replace the
model's inaccurate handle region.

These checks were private observer work on retained calibrated RGB-D. No native
hinge, answer annotation or fit feedback was supplied to the one-step model.
The 3 mm IK goal tolerance is a different quantity from these mechanism-estimate
errors; it is not a grounding accuracy criterion.

## Tool-assisted probes and one documented retest

Two recorded initial observations, states 020 and 035, had not appeared in the
prior Qwen probes. They are the same microwave asset previously evaluated by
Luna, not new objects or a held-out transfer benchmark. Fresh model contexts
received public RGB-D, calibration, Python/image/file tools and the same
numerical grounding implementation. There was no robot motion, teacher,
Reflection reviewer, prior Qwen trace or cross-case memory.

Both initial eight-turn sessions reached their cap without submitting ground.
The first encountered undocumented tool boundaries: it saved crops to isolated
`/tmp`, which the image viewer could not read, then tried writing to the
read-only observation folder. Both sessions also contained ordinary Python
mistakes. The first RGB indexing error occurred after the model had printed
the correct shape; it was not evidence of a malformed depth array. The second
tried calling `.save()` on a NumPy array.

One retest was registered before changing the interface. It added explicit
workspace-relative paths, a precreated writable `scratch/`, read-only input
documentation, and public array shapes/dtypes/indexing. It raised the turn cap
from 8 to 16 while retaining 1,200 seconds, three ground attempts and 32k output
per request. It reused the same observations with fresh contexts and no answers
or previous traces. Documentation and turn budget changed together, so their
individual effects are not isolated. All initial failures remain in the report.

| Retest | Model calls | Model seconds | Tool seconds | Ground fit seconds (inside tool time) | Hinge error |
| --- | ---: | ---: | ---: | ---: | ---: |
| fresh01 / state020 | 7 | 147.14 | 10.59 | 9.58 | 23.28 mm |
| fresh02 / state035 | 9 | 207.13 | 13.20 | 11.15 | 89.97 mm |

Both actors created and inspected image annotations, submitted a ground call
and selected the admitted fit. Both fits relied on existing depth-based handle
recovery. In the second retest, a missing `ImageDraw` import was corrected after
tool feedback. No path errors recurred. The private observer's recomputed hinge
origin and axis matched each actor's actual tool reply exactly. Native scoring
happened after the actor ended and was never fed back.

These results establish that Qwen can use the tools under the revised contract.
They do not establish reliable localization, manipulation success, self-practice
learning or transfer. A hypothesis can fit the observed plane/handle while its
proposed hinge line remains displaced. Depth recovery also must not be credited
as successful model localization.

## Current implementation and reproducibility

`LocalChatClient` and the experimental full-actor runner now default to
temperature 1.0 and presence penalty 1.5. The runner exposes both flags. The
32,768 output / 65,536 server-context / 900-second request settings remain.
Exact prompt admission still refuses insufficient context instead of shrinking
the output allowance. Truncated responses are still blocked before dispatch.

Historical capability, coordinate, I/O and output-budget reproduction drivers
explicitly preserve their old temperature and omitted presence penalty. The
three older 4k drivers remain historical reproductions, not new-run defaults.
Final default promotion happened after the experiments ended; frozen sources
and completion audits preserve each actual experimental version.

The local analysis adapter now documents image paths and precreates `scratch/`.
Array-layout metadata is supplied by the offline probe; this study did not
change the live robot sensor export. It also did not validate the experimental
local full-actor runner in a physical episode.

Reproduction entry points:

- `diagnose_qwen_sampling.py --run NEW_RESULT_DIRECTORY` runs the registered
  one-step schedule against an already running pinned local server.
- `evaluate_qwen_sampling_geometry.py --run RESULT_DIRECTORY` adds observer-only
  fits for newly completed immutable requests.
- `probe_qwen_tool_assisted_grounding.py` provides the offline actor and a
  no-generation `--preflight-only` check. The original eight-turn source is
  frozen under `tool-assisted/source-snapshot`; current code includes the
  clarified contracts. `--retest` requires the separately registered retest
  protocol. Do not silently substitute the current prompt for the old condition.
- `summarize_qwen_sampling.py` and `report_qwen_sampling.py` regenerate summaries,
  timing tables and HTML without a model or robot call.

All 38 new model requests retained at least 32k allowance; none truncated or
raised a model-request error. Total generated output was 62,588 tokens and
model-client time was 1,026.72 seconds. Exact tokenizer and generation prompt
counts matched for all 38 requests. Tool dispatch totaled 27.20 seconds.
Observer-only scoring was separate. Grasp planning, IK, arc planning and
physical execution were not exercised.

Eighty-seven focused tests passed, both real sandbox preflights passed, source
snapshots and input hashes were retained, and the report rendered in Chrome.
The owned server processes are stopped; other GPU jobs were untouched.
The Luna guide hash remains unchanged.

For future work: retain the sampling and tool-contract fixes, keep Luna primary,
and measure localization and mechanism uncertainty separately from JSON validity
and fit acceptance. Do not promote these reused-scene diagnostics as evidence
of learning or copy evaluator coordinates into learner memory.
