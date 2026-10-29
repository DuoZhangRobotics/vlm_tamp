# Luna recorded-evidence diagnostic — 26 October 2026

**Completed: the prediction/revision prompt did not improve the registered
decision-label screen: 9/12 responses passed in each condition.** This study
does not measure physical recovery or learning transfer. Exact model inputs,
outputs, images and timing appear in the
[local report](../results/luna-evidence-loop-20261026/watch.html).

## Results

| Measure | Baseline | Predict / inspect / revise |
| --- | ---: | ---: |
| Valid output schemas | 12/12 | 12/12 |
| All registered label checks | 9/12 | 9/12 |
| Recognized stationary-door cases | 1/4 | 1/4 |
| Incorrectly claimed motion in those cases | 1/4 | 1/4 |
| Remained uncertain in those cases | 2/4 | 2/4 |
| Recognized moving-door controls | 4/4 | 4/4 |
| Correct success stop | 2/2 | 2/2 |
| Correct budget-rejection classification | 2/2 | 2/2 |
| Static next-request checks passed | 9/12 | 12/12 |
| Mean two-turn actor time | 61.62 s | 58.64 s |

All twelve matched pairs agreed on the label-screen outcome: nine both passed,
three both unresolved/incorrect. An uncertain answer is not automatically a
bad decision: it is counted separately from a false motion claim. The strict
motion-discrimination endpoint asks whether the available evidence resolved
motion versus no visible motion.

The invalid baseline requests were `measure_tracking` without a saved
`start_tracking` reference (runs 001, 018 and 022). They passed argument-schema
checks, but would reject on the service's state precondition. The treatment
had no such requests. This is a secondary descriptive process difference,
not proof of improved physical recovery. The qualitative review was not fully
blinded; the structured checks and case criteria were frozen before model calls.

The complete study took 1,548.73 s. Mean before/after turn times were
29.64/31.99 s baseline and 25.70/32.95 s treatment. Observed command intervals
averaged 0.16/0.09 s per two-turn session; remaining actor time includes model
processing, image tools, scheduling and transport. Mean sandbox preparation
was 2.93/2.71 s. No planner, grasp search, IK or robot execution ran.

All 24 sessions completed without infrastructure errors or timeouts.
[Audit](../results/luna-evidence-loop-20261026/audit.json) verifies exact original
image delivery (four before, eight paired after), fresh contexts, same-thread
continuation, immutable packets/notes, private-read isolation, Luna/high model
settings and 547 unchanged runtime/guide files.

## Concrete implementation finding and follow-up

The existing hinge fitter removes tracked points with XY displacement at or
below 1.5 mm before fitting. That is useful for rejecting stationary background
in hinge estimation, but the discarded observations can matter when diagnosing
why a commanded manipulation produced little movement. Its public
`correspondences` field counts the points remaining after that filter.

An exploratory observer replay reconstructed the actor-selected geometry and
recomputed the existing tracker using only saved public RGB-D. All four fit
results reproduced exactly. The two stationary-door cases had 88/99 and 88/90
near-stationary matches, while both moving-door cases had zero. The respective
median XY displacements were 0, 0, 60.1 and 73.3 mm. These counts do not certify
object identity: incorrect region selection can include stationary background.

The [sensor-feedback follow-up](../results/luna-landmark-feedback-20261026/watch.html)
tests a compact summary of these measured point displacements in a separate
paired experiment. It does not replace or reinterpret the null result above.
No extra evidence was sent to any actor in the original 24-session study.

## Question and comparison

The previous [visual-inspection pilot](luna_visual_inspection_ab_20261025.md)
made a retrieval helper available, but only one of five treatment episodes used
temporal comparisons. This follow-up tests a missing part of the process
motivated by [VISTA](https://arxiv.org/abs/2610.02200): writing a brief expected
observable outcome, inspecting the returned evidence, and revising the current
working assumption. It is not a reproduction of the full VISTA harness.

Both conditions receive identical original images directly, with the same
public event prefixes, compact history, full responses, manuals, frozen guide,
numerical libraries and final decision schema. The baseline summarizes the
current situation before the recorded request and assesses the returned result.
The treatment predicts an observable effect before seeing the result and then
explicitly compares it with its prediction. This holds direct image delivery
constant; it cannot isolate the benefit of direct delivery versus the previous
live interface.

There are six selected checkpoints and two fresh sessions per condition per
checkpoint: 24 sessions, each with two turns. All use `gpt-6-luna` with high
reasoning and a 240-second limit per turn. Sessions run serially, with condition
order alternated and reversed across repetitions. No stronger model, teacher,
reflection reviewer, external model API or fine-tuning is involved.

These are previously observed development checkpoints on one microwave asset.
Several share source trajectories. Repetitions are model samples, not new
environments or evidence of transfer.

## Evidence and isolation

The six checkpoints cover:

- Two executions where the robot moved but the native door did not.
- Measured RGB-D pose motion with an unaccepted hinge fit.
- Visible door motion with a rejected RGB-D fit and hand-predicted pose update.
- An execution stopped early by native task success.
- A grasp request rejected by the existing control-reserve rule before search.

Native motion/contact values and these categories are private evaluator data.
Actors receive public RGB-D, tool history and task status only. The initial
workspace contains just the before prefix and pending request. The host adds
the after prefix only after the first turn ends and the note is archived. The
second turn resumes the same Luna thread and has read-only access to that note.

Four original before images are attached in turn one. Eight original images
are attached in turn two, alternating before/after for frontview, agentview,
wrist and arm-side cameras. No target crop, annotation or semantic answer is
inserted. A planning response without a new capture repeats the last snapshot;
the prompt explicitly identifies this possibility. All public assets referenced
by the six prefixes are retained; none are missing.

The operating-system sandbox denies the repository, private rubric, native
diagnoses, other sessions and future packet. The runner preserves exact prompt
text, commands, input hashes, outputs and event logs. A private observer audits
only these experiment UUIDs to verify the attached image bytes and model
settings; it does not export private reasoning or system/developer messages.

The guide remains SHA256
`65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2`.
No cross-case memory update or promotion occurs. No simulator, planner, grasp
search, IK, collision service or robot execution is launched.

## Evaluation

The registered structured-label screen checks visible-motion discrimination,
pose-estimate provenance, task status and compatible next-step category. The
two stationary-door cases must not assume retained contact. Success must lead
to stopping without another robot request. An uncertain visual conclusion is
allowed, but does not count as correct discrimination on the four ordinary
arc cases with visibly clear motion/no motion. The tiny success-terminated
movement does not need to be visually resolved.

The label screen is insufficient on its own. Evidence and action review checks
whether the cited observations support the claim and whether a proposed next
request satisfies the documented contract and available state. Recognizing a
failure and possessing an effective recovery action are separate outcomes.
No physical success rate is inferred from an offline diagnosis.

Timing separates before/after model-session wall time, observed shell-command
intervals, sandbox preparation and archival work. Time outside observed
commands includes inference, image tools, scheduling and transport; it is not
pure reasoning time. Raw CLI usage aggregates include repeated/cached context
and are not billing or peak-context measurements.

## Implementation and artifacts

- [Packet builder, prompts and schema](../src/vlm_tamp/recorded_evidence.py)
- [Isolated two-turn runner](../scripts/run_luna_evidence_diagnostic.py)
- [Report builder](../scripts/report_luna_evidence_diagnostic.py)
- [Protocol](../results/luna-evidence-loop-20261026/protocol.json)
- [Private rubric](../results/luna-evidence-loop-20261026/rubric-private.json)
- [Observer audit](../results/luna-evidence-loop-20261026/audit_replay.py)
- [Focused tests](../tests/test_recorded_evidence.py)

Before model requests, 30 focused tests passed, covering temporal cutoff,
packet immutability, original image order, reporting schema and scoring
counterexamples. The diagnostic changes no default live-agent behavior.
