# Private checkpoint diagnostic, 2026-10-21

Later runtime audit: eight response logs contain NumPy import failures despite
the supplied documentation promising numerical libraries. The scores below
remain the observed responses, with an additional environment limitation on
capability comparisons. See [the runtime defect and repair](public_analysis_runtime_20261021.md).

All 24 registered responses completed, with valid decision files, distinct
contexts, passing isolation checks and unchanged packet contents. Luna handled
basic prediction/measurement distinctions, but missed or incompletely addressed
two related model-consistency problems that Sol identified. This supports a
specific development target, not a conclusion that Luna cannot learn recovery.

## Results and learning targets

Scores below average two responses per model at each checkpoint, on the
registered four-dimension rubric (8 points maximum). They are qualitative
assistant judgments with model identities masked, not physical success rates.

| Checkpoint | Luna mean | Sol mean | Main observation |
| --- | ---: | ---: | --- |
| Accepted visual motion | 8/8 | 8/8 | Both use measured motion and preserve contact uncertainty. |
| Finite planning failure | 8/8 | 8/8 | Both avoid treating exhausted search as global infeasibility. |
| Uncertain hand prediction | 8/8 | 8/8 | Both request measurement using the existing tracker reference. |
| Refit changes the selected object part | 6/8 | 8/8 | Sol identifies the hinge-side frame substituted for the handle. |
| Subsequent mistaken hinge edit | 6/8 | 8/8 | Sol identifies the coordinate mixup and the preceding wrong contact. |
| Strong depth disagreement after motion | 8/8 | 8/8 | Both reject treating predicted travel as verified opening and request fresh grounding. |

Total rubric scores were **88/96 for Luna and 96/96 for Sol**. A maximum score
means the response met this coarse offline rubric; it does not establish an
optimal or executable recovery policy. There is a substantial ceiling effect,
and the two discriminating checkpoints share one physical episode.

At the first inconsistent refit, Luna's two responses proposed grasp planning
for the newly fitted region or another arc. One noticed the hand-to-fit distance,
but neither identified that automatic recovery had selected a different object
part. Sol's two responses compared the recovered box, current image and retained
depth, identified the wrong region, and proposed a new observation-derived
grounding request. At the next checkpoint, both Sol responses additionally
identified the copied handle-coordinate component in the hinge edit. One Luna
response chose a useful inspection but missed the edit's provenance; the other
called the current model corrected and proposed another arc.

The decisive information was already public. Event 11's fit diagnostics name
the requested contact box and the recovered box on the opposite edge. The
returned anchor moved near the hinge. Event 12's requested hinge x matches an
earlier contact-anchor x. Independent checks of the retained agentview depth
support Sol's approximate handle-position claims. No native object geometry was
needed for those checks. The proposed new grounding regions have not been run
through the numerical fitter or physically executed.

This suggests two concrete abilities for future development practice:

1. Verify that an automatically recovered or refitted geometric candidate still
   refers to the intended object part. Numerical fit success alone is inadequate.
2. Track the origin and meaning of values used in model changes, and check what
   actually changed before using the result as a corrected scene.

These are evaluator-side learning targets. No response, diagnosis or new lesson
has been supplied to Luna's practice memory. A later task-independent prompt
ablation could test a short verification record. It should include ordinary
correct fits as well as inconsistent ones, and measure whether decisions improve
without causing unnecessary re-grounding. Any resulting memory should still be
Luna-authored, tested on separate development conditions, and frozen before a
fresh transfer evaluation. Physical continuations are needed to establish that
better identification leads to useful recovery.

Luna averaged **29.4 seconds** per response; Sol averaged **94.7 seconds**.
These are end-to-end times including local preparation/checks, not isolated model
latencies or subscription price estimates. Both ran at nominal high reasoning
effort. Input consumption differed because each agent chose its own file and
image inspections. All 24 proposed operations passed signature and obvious
staleness checks; none was executed by this diagnostic.

[Masked scores](../results/checkpoint-diagnostic-20261021/scores-blind.json) ·
[Scored records and evidence notes](../results/checkpoint-diagnostic-20261021/scored-results-private.json) ·
[Runtime and usage totals](../results/checkpoint-diagnostic-20261021/runtime-summary.json)

## Question and information boundary

Can Luna recognize inconsistencies and choose an evidence-based next operation
using the information already available to the robot agent? This tests a narrower
ability than physical recovery. A fresh model reads a recorded prefix; it does
not resume the original agent's hidden reasoning or operate the simulator.

Six development checkpoints come from four completed source episodes. Three
checkpoints share the failed state 45 memory episode. The other checkpoints
cover accepted visual motion, a finite planning failure, and uncertain
hand-motion prediction in source episodes that eventually succeeded. Later
outcomes are not included in any acting-model input or used as actor knowledge
when grading. These are deliberately selected development cases, not held-out
generalization evidence.

Each packet contains the exact original public requests and responses through
its cutoff, referenced public artifacts, retained images, the original tool
documentation, the original abstract example, and the exact frozen memory
already provided in that episode. Four current camera images are attached in
the same order. Some historical sensor arrays were pruned by the original
study; their absence is explicitly listed. Paired model inputs are byte-identical.

Case names, our diagnoses, grading criteria, native object geometry, future
events and final agent notes remain outside the acting workspace. The packet
builder copies only referenced artifacts. Original evidence is not hardlinked
into a writable workspace. An OS isolation probe checks private files and a
symlink, and the input directory is mounted read-only. Separate fresh model
contexts prevent sharing answers across cases or repetitions.

## Registered comparison

- Models: `gpt-6-luna` and `gpt-6.1-sol`, both at high reasoning effort.
- Six checkpoints, two responses per model/checkpoint: 24 responses total.
- Existing ChatGPT subscription; no separately billed model API.
- 600 seconds per response, at most two concurrent requests, no automatic retries.
- Identical short output contract: next request, brief justification, expected
  observation and relevant uncertainty. No detailed reasoning transcript.
- No robot motion, simulator resets, new practice attempts or memory promotion.

The four qualitative grading dimensions are tool-contract use, evidence
fidelity, value of the next investigation/action, and checkable predictions with
appropriate uncertainty. Each has a registered 0–2 scale. Request signatures and
obvious stale-plan references are checked separately against the service source.
Those checks do not establish numerical feasibility. Multiple next actions can
be defensible; the rubric does not require matching an expert action sequence.

The qualitative reviewer is this coding assistant with model labels masked,
not an independent human panel. Repeated responses and the overlapping state 45
prefixes are correlated. Scores will be exploratory diagnostic evidence, not
physical success rates or a statistical ranking of model intelligence.

## Implementation and validation

The [packet builder](../src/vlm_tamp/checkpoint_diagnostic.py),
[subscription runner](../scripts/run_checkpoint_diagnostic.py), and
[review utility](../scripts/report_checkpoint_diagnostic.py) are separate from
the numerical robot service. Nine targeted tests passed, covering causal
cutoffs, exclusion of future notes/evaluator files, path traversal, external
symlinks, explicit missing assets, evidence-copy independence, hash changes and
malformed decisions. A separate preflight confirmed that input reads work and
input writes fail with the filesystem's read-only error. The initial probe
expected a different exception class; its original output is retained alongside
the corrected probe. No model run was repeated for that probe correction.

All source files listed in the earlier numerical freeze retain their evaluated
hashes except the previously documented reporting-interpreter repair. The new
diagnostic sources, packets, prompt, case registration and rubric have their own
hashes and source snapshot. After all responses and masked reviews completed,
one unused import was removed from the new test file; its original is retained
and the test-only change has a separate maintenance record. Runtime sources and
model inputs were unchanged. Final targeted tests again passed (9 tests), and
lint passed. The evaluated practice memory remains unchanged.

The CLI emitted an arg0 temporary-directory cleanup warning. Three reference
responses also attempted to open optional derived images that had not been
created; they completed using available inputs. All four original attached
images were present in every response workspace. These tool errors remain in
the logs; no failed case was retried or dropped.

Artifacts: [protocol](../results/checkpoint-diagnostic-20261021/protocol.json),
[private rubric](../results/checkpoint-diagnostic-20261021/rubric-private.json),
[private case provenance](../results/checkpoint-diagnostic-20261021/cases-private.json),
[response records](../results/checkpoint-diagnostic-20261021/results.json),
[tests](../results/checkpoint-diagnostic-20261021/final-tests.log),
[lint](../results/checkpoint-diagnostic-20261021/final-lint.log),
[post-run test cleanup](../results/checkpoint-diagnostic-20261021/post-run-maintenance.json).

The [completion audit](../results/checkpoint-diagnostic-20261021/completion-verification.json)
passes all recorded checks. Approximately 0.73 GB of packets, provenance, code,
responses and review evidence remain. All 24 temporary workspaces were removed,
including about 2.88 GB of redundant input copies. Original source evidence and
the sole retained diagnostic inputs were preserved.

## Reasoning prompts and the meaning of learning

The current experiment updates written memory and context, not model weights.
Because practice includes prior guidance and experience, it is not strictly
zero-shot. A local model can also be used with frozen weights; fine-tuning is not
required merely to run it.

Luna already runs at high reasoning effort. Official OpenAI documentation
describes reasoning-effort support for Luna and generally recommends clear,
direct prompts rather than asking reasoning models for a full step-by-step
transcript. That general guidance does not establish which prompt works best in
this robotics system. [Luna documentation](https://developers.openai.com/api/docs/models/gpt-6-luna),
[reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning-best-practices).

A later controlled prompt experiment can compare the existing interface with a
short, task-independent decision record: current belief, supporting observation,
important uncertainty, next test, and predicted result. Measure whether that
changes decisions and physical outcomes. Do not treat a fluent explanation as
proof of correct reasoning. Giving Luna a worked solution to the diagnosed case
would introduce additional supervision; it should be a separately labeled
experiment, not silently folded into self-authored practice memory.

## Local foundation-model feasibility

The server reports four NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
GPUs, each with 97,887 MiB (about 95.6 GiB) visible memory. At inspection, GPUs 2
and 3 were essentially idle, while 0 and 1 had allocations. The reported links
are PCIe/host-bridge paths, with no NVLink shown. These observations establish
capacity, not exclusive resource availability or inference performance.

For scale, a nominal dense 32-billion-parameter model uses about 59.6 GiB for
BF16 weights alone; a nominal 70-billion-parameter model uses about 130.4 GiB.
These are arithmetic estimates excluding additional vision components, KV cache,
activations and framework workspaces. A vision-language model in the smaller
range is therefore a plausible single-GPU candidate, subject to context/image
limits and actual load measurements. [Hardware record](../results/checkpoint-diagnostic-20261021/hardware-feasibility.json).

For example, Qwen publishes a 32B vision-language model with Transformers and
vLLM serving instructions. This is a concrete feasibility example, not a claim
that it matches Luna on this task. Current vLLM documentation provides Blackwell
CUDA guidance and recommends an isolated environment. The robot environment
currently has Torch 2.7.1+cu128 and Transformers 4.43.1, and no vLLM installation;
local serving should use a separate validated environment and a dedicated GPU.
[Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct),
[vLLM installation guidance](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

No local weights have been downloaded and no local inference benchmark has run.
A useful later comparison would reuse these exact packets, then test physical
continuations with matched tools, information and budgets. Model weights,
quantization, image processing, context length, tool-call parsing and end-to-end
latency would need to be recorded.

## Subsequent prompt test

The [completed generic-verification comparison](verification_prompt_study_20261021.md)
tested 24 Luna responses and four new physical episodes. The prompt mostly
increased inspection requests: familiar calibration scores improved slightly,
additional-checkpoint scores did not, and both physical conditions succeeded
2/2. Verification used fewer steps in the two pairs but elapsed time was mixed.
Neither the default prompt nor the evaluated learner memory was replaced.

## Commands

The registered study is complete. Do not rerun or overwrite its directory.
For a separately authorized repeat, prepare a new directory with a new recorded
source snapshot, then run it. The runner resumes only jobs that have never
started and refuses automatic retries of partial runs:

```bash
python3 scripts/run_checkpoint_diagnostic.py \
  --output results/checkpoint-diagnostic-NEW --prepare
python3 scripts/run_checkpoint_diagnostic.py \
  --output results/checkpoint-diagnostic-NEW --run
```

The completed comparison can be regenerated from its saved masked scores:

```bash
python3 scripts/report_checkpoint_diagnostic.py \
  --output results/checkpoint-diagnostic-20261021 --summarize
```

Do not run a second driver while one is active. These commands use the existing
subscription and require the same sandbox/network launch permissions as the
earlier subscription experiments. The model's own shell remains network-disabled
and restricted to its public workspace.
