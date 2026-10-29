# Generic verification prompt study, 2026-10-21

Later runtime audit: two offline response logs contain NumPy import failures;
the isolated learner's promised numerical-library environment was unavailable.
The physical outcomes remain valid, but capability interpretations must include
this limitation. See [the runtime defect and repair](public_analysis_runtime_20261021.md).

Status: **complete**. Both conditions succeeded in both physical episodes.
Verification used fewer control steps in these two pairs, but elapsed time was
mixed and the additional offline checkpoints did not improve. The paragraph
remains an optional experimental scaffold; neither the default prompt nor the
evaluated learner memory was replaced.

## Offline findings

| Subset | Baseline | Verification |
| --- | ---: | ---: |
| Two reused calibration checkpoints, four responses/condition | 25/32 | 29/32 |
| Four additional development checkpoints, eight responses/condition | 63/64 | 62/64 |
| Total qualitative rubric points | 88/96 | 91/96 |

The added prompt produced a small improvement on the familiar calibration
examples and no improvement on the additional examples. Evidence-fidelity
scores were identical at 19/24 in both conditions. The difference in total
points came from choosing more relevant checks at the familiar problem states.
These coarse, ceiling-limited scores do not establish physical recovery or a
reliable prompt advantage.

Baseline responses chose six arc queries, three grasp queries and three scene
inspections. Verification responses chose two arc queries and ten inspections.
One verification response explicitly identified the roughly 20 cm discrepancy
introduced by the model edit. Neither condition consistently identified the
underlying wrong object-part selection. The extra prompt therefore changed the
next-operation preference more clearly than it improved diagnosis.

One verification response compared positive hand progress with negative RGB-D
yaw without reconciling the negative-Z hinge axis with world-Z yaw. The signs
alone do not establish opposite motion. This illustrates why asking for
verification does not by itself guarantee correct comparisons of quantities.
Its requested inspection was still supported by independent depth disagreement.

All 24 responses had valid decision files and passed signature/staleness checks.
Mean response times were 27.7 seconds for baseline and 28.7 for verification.
Packet copies were unchanged, every context was distinct, and all isolation
checks passed. Model options and readable/writable grants match within each
checkpoint. Literal filesystem-policy strings differ in two checkpoints only
because private SQLite journal files appeared/disappeared and were denied when
present; the differing entries grant no read or write access.

[Masked reviews](../results/verification-prompt-study-20261021/scores-blind.json) ·
[Scored responses](../results/verification-prompt-study-20261021/scored-results-private.json) ·
[Offline audit](../results/verification-prompt-study-20261021/offline-audit.json)

## Question

Does a short, task-independent verification instruction help Luna use tool
results accurately and complete physical manipulation? The earlier checkpoint
comparison identified failures to notice changed object-part identity and to
trace a mistaken model edit. This study tests a human-authored process scaffold;
it does not count that scaffold as an agent-learned memory.

The [added paragraph](../prompts/generic_result_verification.txt) asks the agent
to check the entity and quantity represented by a result, compare changes with
observations and prior estimates, distinguish measured changes from edits and
assumptions, and choose an informative check when needed. It also explicitly
permits continued useful action when evidence is consistent. It contains no
task name, coordinates, prescribed operation, angle or recovery sequence.

Baseline prompts remain unchanged. Treatment prompts append only that paragraph.
Both conditions use Luna/high through the existing ChatGPT subscription, the
same previously evaluated frozen memory, original abstract example, numerical
tools, physics and native success predicate. No reference-model answers or
evaluator diagnoses are supplied to the actor.

## Registered offline comparison

Six checkpoint prefixes, two conditions and two fresh responses per condition
produce 24 responses. Two checkpoints are the previously examined state 45
calibration cases. Four additional checkpoints come from different source
episodes and were not used in the earlier model comparison. All are deliberately
selected development data; they are not fresh held-out tasks or new assets.
Report calibration and additional-development scores separately.

The four additional prefixes originally came from agents without the newly
learned memory. Here both prompt conditions receive the same pre-existing frozen
memory, which predates those source episodes. These are counterfactual fresh
decisions on a recorded prefix, not resumptions of the original agent context.
Historical sensor gaps are listed, and all paired packet bytes are identical.

The output contract and grading rubric are the same in both conditions. The
qualitative coding-assistant reviewer sees responses with condition labels
masked. Grading does not reward verbosity, merely repeating the verification
instruction, or indiscriminate re-grounding. Masking does not prevent a reviewer
from inferring a condition from wording. An offline score cannot establish
physical feasibility or recovery.

## Registered physical follow-up

Completed outcomes:

| State | Baseline | Verification |
| --- | --- | --- |
| 45 | Success, 355 steps, 467 s agent time | Success, 345 steps, 414 s agent time |
| 0 | Success, 354 steps, 263 s agent time | Success, 323 steps, 387 s agent time |

Native success was **2/2 in each condition**. Mean control steps were 354.5 for
baseline and 334 for verification; mean agent time was 365 s versus 401 s.
Verification saved 10 and 31 steps in the two pairs, but was faster on state 45
and slower on state 0. With one sample per condition/state and different early
grasp/motion choices, those differences do not establish a reliable efficiency
advantage or isolate the verification mechanism.

The state 45 initial sensor arrays and robot state match exactly. State 0 has one
frontview RGB pixel differing by one intensity level in three channels; all
other arrays, including depth, world points and robot priors, match exactly.
Robot state matches exactly in both pairs. State 0 is therefore not claimed to
have bit-identical images.

All episodes used one reset, ended on native success, and issued no post-success
control. All retained issued references match the execution logs, with no
execution vetoes or forbidden evaluator fields in actor feedback. Each condition
received its registered prompt and the same frozen memory through the existing
ChatGPT subscription. There were no separate API ledger changes.

All four episodes made useful opening progress despite intermittent weak visual
tracking. None had a hand-prediction event over one degree with native motion
below 0.05 degrees, using the previously defined evaluator-only diagnostic.
The baseline state 0 approach even ended about 22 mm from its requested TCP pose
and subsequently succeeded; that tracking error alone did not establish an
ineffective interaction. No mandatory two-finger-pinch or exact-pose success
criterion was introduced.

The trials recovered from finite planning failures, but they did not demonstrate
reliable reacquisition after lost contact or repair of a wrong object-part
assignment. The added paragraph is therefore not established as a recovery fix.
The next targeted experiment should start from saved failure prefixes so that
the relevant inconsistency is present in both conditions, rather than depending
on it arising during a fresh episode. Keep that recovery test separate from
these completed task-success results.

[Full audit and evaluator diagnostics](../results/verification-prompt-study-20261021/analysis-private.json) ·
[State 45 baseline video](../results/verification-prompt-study-20261021/physical/state045-baseline/attempt01/initial/rollout.mp4) ·
[State 45 verification video](../results/verification-prompt-study-20261021/physical/state045-verification/attempt01/initial/rollout.mp4) ·
[State 0 baseline video](../results/verification-prompt-study-20261021/physical/state000-baseline/attempt01/initial/rollout.mp4) ·
[State 0 verification video](../results/verification-prompt-study-20261021/physical/state000-verification/attempt01/initial/rollout.mp4)

Four new episodes start from reset, with one episode per condition on each of
known development states 45 and 0. The schedule was frozen before offline
responses: 45 baseline, 45 verification, 0 verification, 0 baseline. Each episode
has at most 600 control steps and 1,800 seconds, persists until native success
or a genuine budget limit, and stops immediately on success. There is no reset
retry, post-success retreat or execution-time veto. Optional DINO is disabled
uniformly as in the previous validated study.

Native success is the primary outcome. Step/time consumption, planning failures,
model revisions and the recovery trace are secondary evidence. All four episodes
remain in the denominator, including failures and infrastructure problems. With
only two states and one episode per condition/state, any success difference is
development evidence, not a reliability or generalization estimate. A better
episode may also reflect avoiding a failure earlier, rather than recovering
from an identical failed physical state.

No per-case notes or outputs are passed to another episode. The existing learner
memory is not rewritten or promoted by this comparison. Failure sensor retention
is evaluator-only and bounded to eight early planning-failure observations per
episode, in addition to normal initial/final evidence retention.

## Preparation and reproducibility

Preparation initially stopped before any model or physical trial because the
loader expected contiguous public event IDs. A successful multi-stage sequence
uses internal stage IDs but emits a single public RPC record. The loader now
accepts a gap only when the response's stage records account for every ID, and
still rejects missing history or a cutoff inside a multi-stage RPC. The failed
preparation, original loader and error record are preserved separately.

Fifteen focused tests passed, including stage-ID accounting, causal cutoffs,
public-input isolation, explicit shared-memory provenance, unchanged baseline
prompt prefixes and the balanced physical schedule. Lint passed. The correction
affects the diagnostic loader, not the numerical robot service. The complete
registration and source snapshot were created after that repair and before
any model responses.

After offline completion, the first four physical launch commands failed before
creating any episode directory, physical agent or simulator: the parent
`physical/` directory did not exist. Their logs, return codes and zero-attempt
records were preserved under `physical-startup-errors/`. Creating that directory
repaired launch setup without changing source, prompts or numerical behavior;
the same four predeclared cases were then launched. These are separately
reported infrastructure errors, not discarded physical failures or extra
physical retries.

After all four episodes finished and the original source/input hashes were
verified, the driver was permanently corrected to create the physical output
parent before launching a child. Two regression tests exercise the actual
nonrecursive child-directory operation and preserve the no-restart behavior for
partial episodes. The final suite passed **17 tests**, and lint passed. The
original source snapshot and protocol remain unchanged; the one-line post-run
directory fix has an exact patch and before/after hashes. Numerical sources did
not change.

[Tests](../results/verification-prompt-study-20261021/final-tests.log) ·
[Post-run setup maintenance](../results/verification-prompt-study-20261021/post-run-maintenance.json) ·
[Completion verification](../results/verification-prompt-study-20261021/completion-verification.json)

Approximately **2.10 GB** remains in the completed study. Cleanup removed about
0.91 GB of redundant public sensor data, 2.95 GB of temporary offline input
copies and 0.18 GB of verified duplicate failed-preparation packets. All raw
outcomes, source/configuration evidence, relevant observations, trajectories,
videos, actor notes and reviews remain. All temporary actor workspaces were
removed. Original handoff evidence and the evaluated memory were preserved.

## Separate implementation lead

A read-only review of older development traces found actual grasp candidates
rejected by the fixed manipulation-budget reserve. For example, the older state
12 control trace records 146 candidates rejected because an approach estimated
at 121 steps exceeded an allowance of 43; later state 32 searches record negative
approach allowances. The current selector reserves 200 manipulation steps plus
40 closure steps before admitting an approach, and uses conservative timing for
that estimate. This is a documented policy rejection, not necessarily collision
or kinematic infeasibility. Actual executable cost and full-task recovery from
those candidates have not been established. No budget-policy fix was applied,
and this is not offered as a cause of the present four successful episodes.
[Separate evidence](../results/verification-prompt-study-20261021/next-investigation-budget-private.json).

[Protocol](../results/verification-prompt-study-20261021/protocol.json) ·
[Status](../results/verification-prompt-study-20261021/status.json) ·
[Private cases](../results/verification-prompt-study-20261021/cases-private.json) ·
[Private rubric](../results/verification-prompt-study-20261021/rubric-private.json) ·
[Earlier checkpoint comparison](checkpoint_diagnostic_20261021.md)

Collection is complete and no driver is running. Do not rerun or overwrite this
registered directory. The current driver includes the post-run directory setup
fix, so it intentionally differs from the archived measured driver. For a
separately authorized repeat, prepare a new directory and source snapshot. The
runner resumes only jobs that have never started; partial requests/episodes
require diagnosis and are not automatically repeated.

```bash
source scripts/server_runtime_env.sh
python -u scripts/run_verification_prompt_study.py \
  --output results/verification-prompt-study-NEW --prepare --offline --physical
```

The recorded scores and completed audit can be regenerated without new trials:

```bash
python scripts/report_checkpoint_diagnostic.py \
  --output results/verification-prompt-study-20261021 --summarize --group-by condition
python scripts/analyze_verification_prompt_study.py \
  --output results/verification-prompt-study-20261021
```
