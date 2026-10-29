# Reflexion checkpoint pilot — 21 October 2026

Status: **complete**. Fourteen simulator episodes ran through Luna/high using
the ChatGPT subscription, with no fine-tuning or separately billed model APIs.
The working-runtime comparison found 1/2 successes for fresh retries, 0/2 for
episode records and 1/2 for records plus reflection. The separate reviewer-context
follow-up finished 0/2. This small, single-checkpoint pilot did not establish a
reliable benefit from reflection. No portable or default learner memory was promoted.

During collection, actual actor errors revealed that the documented NumPy/SciPy
environment was unavailable. The original block is preserved unchanged;
a separate corrected block completed after the validated runtime repair. See
[the implementation defect and repair](public_analysis_runtime_20261021.md).

## Original block result and environment limitation

| Condition | Native successes | Total control steps by trial |
|---|---:|---|
| No new memory | 1/2 | 514, 600 |
| Episode record | 0/2 | 600, 600 |
| Reflection | 0/2 | 600, 600 |

All six scheduled episodes completed. The five failures exhausted the actual
control budget; there were no voluntary early failures. The successful run
stopped on the native predicate. No new lesson was promoted. These observations
are retained with the discovered runtime limitation, rather than treated as a
clean evaluation of the intended learner capabilities.

After closure, the source freeze passed and the runtime repair was integrated.
The second block uses the same actor/review prompts, order, checkpoint, common
memory and numerical robot tools. It adds working pinned analysis dependencies
and a mandatory startup capability check. The two blocks are reported separately.

## Working-runtime block

| Condition | Native successes | Total control steps by trial |
|---|---:|---|
| No new memory | 1/2 | 600, 362 |
| Episode record | 0/2 | 600, 600 |
| Reflection | 1/2 | 430, 600 |

The reflection condition did not outperform fresh retries in this small block.
The successful reflection attempt restored an earlier observation-derived hinge
hypothesis, kept the arm near its existing contact, and completed through fresh
arcs. Its next review concentrated on the final rigid-to-point planning switch;
the following actor did not repeat the earlier model repair and failed. This is
an observed difference between traces, not an isolated causal test of each action.

All six actors passed the numerical-library preflight and recorded no NumPy or
SciPy import failures. The block's 501 registered source hashes passed before
the subsequent reviewer-context repair.

## Explicit reviewer starting state

The separate reviewer had a completed episode record whose `checkpoint.json`
described its **terminal** state. It was asked to suggest a change for another
attempt at the same checkpoint, but was not explicitly told that the actor
would restart after event 12 at 277 steps. The post-success review focused on
the last six commands at a comparable final state. Its applicability to the
earlier retry state was therefore underspecified.

The repair supplies `retry-start.json` only in the reviewer's input copy and
labels the completed record's terminal snapshot explicitly. It contains the
same public start observation, configured model and 323-step remaining budget
already supplied to the actor. The configured model is preserved exactly,
including its erroneous estimate; no corrected coordinate, native object pose
or evaluator diagnosis is added. The actor prompt and canonical initial
experience packet remain byte-identical to the working-runtime block.

After all six working-runtime episodes closed, this two-file repair was applied
and 60 targeted tests passed. `study-v3` registers **two reflection-only attempts**
from the same original failed seed. Earlier controls are nonconcurrent; this is
an exploratory follow-up, not a third matched three-condition comparison.
Both follow-up attempts exhausted 600 control steps without native success.
Their final private opening measurements were 18.73° and 20.44°, from an identical
22.85° starting opening. These measurements were never supplied to the learner.

The first retry-aware review explicitly compared grounding at event 11 with the
edit at event 12 and proposed a test from the 277-step state. The second also
scoped its proposal to that boundary. Thus the input contract and recommendation
scope improved; physical recovery did not become reliable in these two attempts.

## Validation and remaining limits

- All **60 targeted tests** passed after both repairs. The real Luna numerical
  smoke also passed through its actual tool interface.
- **253 execution/data-integrity checks** passed across the 14 episodes, with
  **28 distinct acting/review contexts**. The separate numerical smoke used one
  additional context and no robot operations.
- Every replay matched the source robot trajectory, native checkpoint angle,
  configured model, geometry and all 48 current sensor arrays exactly. No hidden
  object geometry was used to initialize or repair the learner's model.
- All issued references matched the retained controller traces. Execution-time
  veto counts and motion after first native success were zero. All 11 failed
  episodes reached the actual 600-step control limit.
- Supplied read-only inputs remained unchanged; source snapshots match their
  registrations. The source fixes were applied only after the preceding blocks
  closed. The final 502 registered source files and 2,181 dependency files verify.
- The first block contains five observed actor NumPy import failures. The eight
  actors and eight reviewers after the runtime repair recorded no NumPy/SciPy
  import failures. Integrity checks do not erase the first block's capability defect.
- All 14 reviews passed format and exact-public-citation checks. Those checks
  validate references, not causal explanations or the usefulness of a proposed action.

These are two sequential attempts per condition on one deliberately selected
failure state. The second attempt's experience depends on the first attempt's
outcome, so the attempts are not independent task samples. The final two attempts
have no concurrent control arm. Do not pool the blocks into a model ranking,
claim a causal improvement from a single success, or infer transfer to new scenes.

A separate numerical-tool limitation remains unchanged: the grasp selector
reserves 200 manipulation steps and 40 closure steps, while estimating approaches
with a 60-step endpoint hold. Its minimum approach estimate is 61 steps, so at
control step 300 or later every candidate that reaches this budget check is
rejected. Public diagnostics recorded 120 such rejections in original-block
record trial 2 and 47 in follow-up trial 1. This is not proof that those cases
had feasible successful regrasp trajectories. A cost-aware replacement and its
physical effect require a separate test; no budget or numerical rule was relaxed
to obtain these results. See `budget-arithmetic-private.json`.

The main remaining learning question is how to preserve the earlier repairs and
preconditions that enabled a successful action. The success/failure trace pair
above is a concrete diagnostic lead. It does not justify saving the last
successful planner setting as a general recovery rule.

The experiment tests whether Luna's own post-attempt reflection helps it recover
from a previously observed failure. The earlier verification-prompt experiment
started from clean resets and succeeded in all four physical episodes; it did
not exercise this saved failure state. The adaptation is motivated by
[Reflexion](https://arxiv.org/pdf/2303.11366), with the distinctions and limitations
recorded in [the design note](reflexion_design_notes_20261021.md).

## Fixed comparison

Each of the first two registered blocks contains three conditions, each with
two sequential physical attempts:

| Condition | New experience available to the actor |
|---|---|
| No new memory | Common frozen guidance and the current episode prefix |
| Episode record | The above plus the preceding completed public episode |
| Reflection | The same record plus one compact Luna-authored reflection |

Each actor and reviewer uses Luna with high reasoning through the existing
ChatGPT subscription. Every condition receives a separate review call capped
at 300 seconds; the two controls discard its output. Caps are matched, while
actual time and token usage are measured rather than assumed equal.

All actors start in fresh contexts at the same state-45 checkpoint, after event
12 and 277 control steps. The source is the archived failed Luna episode from
the earlier scene-motion memory study. Initialization replays the original
issued joint commands and reruns the recorded public observation, grounding
and model-edit requests. It does not restore native object coordinates or
repair the erroneous planning model.

The episode retains the original 600-step limit, leaving 323 steps for the new
actor. Reset/replay and acting share a 1,800-second wall limit. Native success
immediately ends motion. An unsuccessful plan does not end an episode while
budget remains. Numerical tools, physics, collision checks and task predicates
are fixed throughout the block.

Round 1 uses the same archived failed episode as prior experience for the two
memory conditions. Round 2 uses each condition's own first completed episode.
Only the latest record and one reflection are retained; controls do not receive
their discarded review or actor-authored hindsight notes. The order is no new
memory → episode record → reflection, then the reverse order. These are six
counterfactual continuations of one practiced failure, not six independent
tasks or a held-out transfer evaluation.

## Input and memory boundaries

The common guidance is the existing frozen
`data/agent_memory/luna-scene-motion-v1/guidelines.json`, SHA-256
`65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2`.
The later generic verification paragraph and stronger-model diagnoses are not
supplied. Actor and reviewer sandboxes deny private project results and source.

Reviewers receive public tool requests, responses and available artifacts, plus
the official success boolean and termination/budget outcome. Native object
poses, opening angles, contact logs and evaluator explanations remain private.
Past-attempt outcomes are explicitly distinguished from the current attempt.
Unavailable historical sensor arrays are listed in the input packet.

The review must cite one to six exact public JSON fields and provide a brief
observation, hypothesis, proposed change, falsifiable test, uncertainty and
scope. Validation checks format and exact cited values; it does not certify
causal interpretations. Invalid reviews are preserved and withheld without a
rewrite or expert correction. Assigned conditions remain in the denominator.
Any eventual portable memory requires separate validation on different cases;
this study does not change the default learner memory.

## Validation before learner calls

The successful replay preflight reproduced the saved checkpoint exactly: all
48 sensor arrays, all three geometry point arrays, public robot state, model,
geometry epoch and issued command references matched. Replay took 85.34 seconds.
A preceding preflight reached the checkpoint but its audit attempted to read an
unreferenced robot-state sidecar. The audit now compares the public robot fields
already present in the tool response. Both preflights are retained, and neither
involved a learner or contributes to the six-attempt outcome denominator.

The focused test suite passed 33 tests covering reflection evidence, unavailable
or misquoted fields, memory limits, recorded-prefix construction and persistent
episode budgets. The registered implementation is snapshotted and hash-checked
before and after physical runs.

## Artifacts

- Protocol, frozen prompts and order:
  `results/reflexion-checkpoint-study-20261021/study-v1/protocol.json`
- Current progress: `study-v1/status.json`
- Per-attempt public traces, simulator evidence and audits: `study-v1/episodes/`
- All reviews, including discarded or invalid candidates: `study-v1/reviews/`
- Public previous-episode packets and separate private provenance: `study-v1/records/`
- Aggregate machine-readable results: `study-v1/summary.json`,
  `study-v2/summary.json`, and the reflection-only `study-v3/summary.json`
- Reviewer-context repair: `review-scope-fix.patch` and
  `study-v2/post-collection-review-scope-fix.json`
- Final audit and local-candidate index:
  `final-audit-private.json` and `candidate-memory-index.json`

The primary outcome is native task success across all scheduled attempts.
Secondary analysis examines budgets, cost, citation accuracy and whether the
next physical behavior tests the proposed lesson. More inspection or more
convincing explanations alone do not establish improvement.

All experiment drivers and temporary actor/reviewer workspaces have been closed
and cleaned. Results, original failures, raw public observations, trajectories,
videos, actor findings, candidate reviews, input manifests and source snapshots
remain available. `final-current-integrity.json` and `final-process-check.json`
record the final checks.
